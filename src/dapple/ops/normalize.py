"""Per-pixel intensity normalization.

Three flavors, each with its own failure mode:

- `median_normalize` — divide each pixel by its non-zero median intensity. **Default**:
  robust to single dominating peaks, doesn't assume constant ionization efficiency.
- `tic_normalize` — divide each pixel by its total ion current. Common but fragile
  when ionization efficiency varies (e.g. saturating dominants).
- `reference_ion_normalize` — divide by the sum of intensities at the reference ions
  attached by ``detect_reference_ions``. Useful when matrix peaks (or other
  endogenous standards) are stable across the image, since their summed intensity
  is a per-pixel readout of ionization efficiency that doesn't depend on analyte
  abundance the way TIC does.

All three operate on PeakList backends in-place-by-copy: a new PeakList is returned
with scaled intensity arrays; m/z and offsets are reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dapple.data.dataset import MSIDataset, PeakList
from dapple.data.metadata import ExperimentParams
from dapple.ops.base import (
    Diagnostic,
    OpParams,
    OpResult,
    Operator,
    merge_op_record,
    register,
)

NormFactorMode = Literal["median", "tic"]


@dataclass(frozen=True)
class NormalizeParams(OpParams):
    method: NormFactorMode = field(
        default="median",
        metadata={
            "label": "Method",
            "help": (
                "median: divide each pixel by its non-zero median intensity. Robust "
                "to one dominating peak; the recommended default for MSI data. "
                "tic: divide by total ion current. Common but fragile when a saturating "
                "analyte or matrix effects vary spatially. Stick with median unless "
                "you specifically want TIC."
            ),
        },
    )
    eps: float = field(
        default=1e-12,
        metadata={
            "label": "Numerical floor",
            "help": (
                "Avoids divide-by-zero on empty pixels. Default 1e-12; you should "
                "essentially never need to change this."
            ),
        },
    )


@register
class MedianNormalize(Operator):
    """Divide each pixel by its non-zero median intensity. Survey default."""

    name = "median_normalize"
    params_cls = NormalizeParams

    def default_params(self, ep: ExperimentParams) -> NormalizeParams:
        return NormalizeParams(method="median")

    def apply(
        self,
        ds: MSIDataset,
        params: NormalizeParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        return _normalize_by_factor(ds, params, factor_kind="median", op_name=self.name)


@register
class TicNormalize(Operator):
    """Divide each pixel by its total ion current. Fragile when ionization varies."""

    name = "tic_normalize"
    params_cls = NormalizeParams

    def default_params(self, ep: ExperimentParams) -> NormalizeParams:
        return NormalizeParams(method="tic")

    def validate(self, ep: ExperimentParams) -> list[str]:
        return [
            "TIC normalization assumes ionization efficiency is roughly constant "
            "across pixels — known to fail on samples where one analyte saturates "
            "the detector or where matrix effects vary spatially. Prefer median."
        ]

    def apply(
        self,
        ds: MSIDataset,
        params: NormalizeParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        return _normalize_by_factor(ds, params, factor_kind="tic", op_name=self.name)


def _normalize_by_factor(
    ds: MSIDataset,
    params: NormalizeParams,
    *,
    factor_kind: NormFactorMode,
    op_name: str,
) -> OpResult:
    if not isinstance(ds.backend, PeakList):
        raise NotImplementedError(
            f"{op_name} currently requires a PeakList backend; "
            "post-consensus normalization will land with PeakMatrix support."
        )
    pl = ds.backend
    offsets = np.asarray(pl.offsets[:])
    intensity = np.asarray(pl.intensity[:]).astype(np.float32, copy=True)

    factors = np.empty(pl.n_pixels, dtype=np.float64)
    for i in range(pl.n_pixels):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if a == b:
            factors[i] = 1.0
            continue
        seg = intensity[a:b]
        if factor_kind == "tic":
            factors[i] = max(float(seg.sum()), params.eps)
        else:  # median over non-zero entries
            nz = seg[seg > 0]
            factors[i] = max(float(np.median(nz)) if nz.size else params.eps, params.eps)

    # Apply factors per pixel.
    for i in range(pl.n_pixels):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if a == b:
            continue
        intensity[a:b] = intensity[a:b] / factors[i]

    new_backend = PeakList(
        mz=np.asarray(pl.mz[:]).copy(),
        intensity=intensity,
        offsets=offsets.copy(),
        n_pixels=pl.n_pixels,
    )
    new_ds = ds.with_backend(new_backend)

    diag = Diagnostic(
        name=op_name,
        summary={
            "factor_min": float(factors.min()),
            "factor_q25": float(np.quantile(factors, 0.25)),
            "factor_median": float(np.median(factors)),
            "factor_q75": float(np.quantile(factors, 0.75)),
            "factor_max": float(factors.max()),
            "factor_kind_tic": float(factor_kind == "tic"),
        },
        payload={"per_pixel_factor": factors},
        figure_hint="histogram:per_pixel_factor",
    )

    record = merge_op_record(
        op_name=op_name,
        params=params,
        input_ds=ds,
        output_ds=new_ds,
        diagnostics=[diag],
    )
    new_ds = new_ds.with_history(record)
    return OpResult(dataset=new_ds, diagnostics=[diag])


@dataclass(frozen=True)
class ReferenceIonNormalizeParams(OpParams):
    eps: float = field(
        default=1e-12,
        metadata={
            "label": "Numerical floor",
            "help": (
                "Avoids divide-by-zero when a pixel has zero intensity at every "
                "reference ion. Pixels at the floor are flagged in the diagnostic. "
                "Default 1e-12; you should not need to change it."
            ),
        },
    )


@register
class ReferenceIonNormalize(Operator):
    """Divide each pixel by the sum of its reference-ion intensities.

    Requires ``detect_reference_ions`` to have run upstream so that
    ``ds.extra["reference_set"]`` carries the per-pixel reference-ion intensity
    matrix this operator reads.
    """

    name = "reference_ion_normalize"
    params_cls = ReferenceIonNormalizeParams

    def default_params(self, ep: ExperimentParams) -> ReferenceIonNormalizeParams:
        return ReferenceIonNormalizeParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return [
            "Reference-ion normalization assumes the reference ions are stable "
            "across the image (matrix peaks, internal standards). On heterogeneous "
            "tissue with locally variable matrix coverage this can over-correct."
        ]

    def apply(
        self,
        ds: MSIDataset,
        params: ReferenceIonNormalizeParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "reference_ion_normalize currently requires a PeakList backend."
            )
        ref = ds.extra.get("reference_set")
        if ref is None:
            raise RuntimeError(
                "reference_ion_normalize requires detect_reference_ions to have "
                "run first."
            )
        if ref.mz.size == 0:
            raise RuntimeError(
                "reference_ion_normalize: no reference ions in the upstream "
                "ReferenceSet. Lower min_prevalence on detect_reference_ions or "
                "fall back to median normalization."
            )

        pl = ds.backend
        offsets = np.asarray(pl.offsets[:])
        intensity = np.asarray(pl.intensity[:]).astype(np.float32, copy=True)
        n_pixels = pl.n_pixels

        # Per-pixel scale = sum of reference intensities; clamp at eps to avoid
        # division by zero on pixels where no reference ion was detected.
        per_pixel_ref_sum = ref.per_pixel_intensity.sum(axis=1).astype(np.float64)
        n_at_floor = int((per_pixel_ref_sum <= 0).sum())
        factors = np.maximum(per_pixel_ref_sum, params.eps)

        for i in range(n_pixels):
            a, b = int(offsets[i]), int(offsets[i + 1])
            if a == b:
                continue
            intensity[a:b] = intensity[a:b] / factors[i]

        new_backend = PeakList(
            mz=np.asarray(pl.mz[:]).copy(),
            intensity=intensity,
            offsets=offsets.copy(),
            n_pixels=n_pixels,
        )
        new_ds = ds.with_backend(new_backend)
        diag = Diagnostic(
            name=self.name,
            summary={
                "factor_min": float(factors.min()),
                "factor_q25": float(np.quantile(factors, 0.25)),
                "factor_median": float(np.median(factors)),
                "factor_q75": float(np.quantile(factors, 0.75)),
                "factor_max": float(factors.max()),
                "n_pixels_at_floor": float(n_at_floor),
                "n_reference_ions": float(ref.mz.size),
            },
            payload={"per_pixel_factor": factors},
            figure_hint="histogram:per_pixel_factor",
        )
        record = merge_op_record(
            op_name=self.name,
            params=params,
            input_ds=ds,
            output_ds=new_ds,
            diagnostics=[diag],
        )
        new_ds = new_ds.with_history(record)
        return OpResult(dataset=new_ds, diagnostics=[diag])
