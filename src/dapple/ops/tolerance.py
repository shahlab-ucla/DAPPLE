"""Mass-tolerance estimation: empirical and instrument-aware-parametric.

The recommended approach is **empirical tolerance from reference-ion shifts**:
take the per-pixel ppm errors from `detect_reference_ions`, fit a monotone function
of m/z (isotonic regression), and bootstrap a confidence envelope. This makes
essentially three auditable assumptions: (a) reference ions are correctly
identified, (b) reference shifts are representative of analyte shifts, (c) the
chosen quantile is the right notion of coverage. None of them require Gaussianity
or commit to a particular instrument-physics scaling law.

Output is a `ToleranceCurve` attached to `ds.extra` so downstream operators (consensus
alignment, recalibration) can ask `tol(mz) -> ppm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dapple.data.dataset import MSIDataset
from dapple.data.metadata import ExperimentParams
from dapple.ops.base import (
    Diagnostic,
    OpParams,
    OpResult,
    Operator,
    merge_op_record,
    register,
)
from dapple.ops.reference_ions import ReferenceSet


@dataclass(frozen=True)
class EmpiricalToleranceParams(OpParams):
    """alpha = 0.01 → 99% coverage. B is the bootstrap repeat count."""

    alpha: float = field(
        default=0.01,
        metadata={
            "label": "Coverage 1−α",
            "help": (
                "Sets the coverage of the tolerance estimate: α = 0.01 → tolerance is "
                "the 99th-percentile of |ppm error|. Smaller α makes the tolerance "
                "wider (more conservative); larger α makes it tighter (cuts more "
                "real peaks). 0.01 is the recommended default for typical MSI data."
            ),
        },
    )
    bootstrap_B: int = field(
        default=1000,
        metadata={
            "label": "Bootstrap iterations",
            "help": (
                "Number of bootstrap resamples used to compute the tolerance "
                "confidence interval. Default 1000 gives stable CIs. Drop to 200–500 "
                "for fast iteration on large images; raise to 5000 for very tight CIs."
            ),
        },
    )
    block_bootstrap: bool = field(
        default=True,
        metadata={
            "label": "Block bootstrap (spatial)",
            "help": (
                "Resample pixels in spatial blocks rather than independently, so the "
                "CI respects spatial autocorrelation. Default ON; disable only if "
                "pixels are demonstrably independent (e.g. well-separated cell array)."
            ),
        },
    )
    block_size_pixels: int = field(
        default=32,
        metadata={
            "label": "Block size (pixels)",
            "help": (
                "Side length of the spatial blocks for the block bootstrap. Default 32. "
                "Tune to roughly the spatial scale of your tissue features — too small "
                "and you under-correct for autocorrelation, too large and effective "
                "sample size collapses."
            ),
        },
    )


@dataclass(frozen=True)
class ToleranceCurve:
    """Monotone-non-decreasing |ppm error| as a function of m/z, with a CI band.

    `mz_grid` is a sorted 1-D float64 array; `ppm_quantile` is the upper-quantile
    estimate at each grid point (the working tolerance); `ci_low`/`ci_high` give the
    bootstrap 95% confidence interval. `evaluate(mz)` returns interpolated tolerance.
    """

    mz_grid: np.ndarray  # (G,) float64, ascending
    ppm_quantile: np.ndarray  # (G,) float64
    ci_low: np.ndarray  # (G,) float64
    ci_high: np.ndarray  # (G,) float64
    alpha: float
    bootstrap_B: int

    def evaluate(self, mz: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(mz, dtype=np.float64), self.mz_grid, self.ppm_quantile)

    def as_callable(self) -> Callable[[float | np.ndarray], np.ndarray]:
        return self.evaluate


@register
class EmpiricalToleranceFromReferenceIons(Operator):
    """Fit an isotonic |ppm error| vs m/z curve from reference-ion deviations.

    Requires `detect_reference_ions` to have been applied first; reads
    `ds.extra['reference_set']`.
    """

    name = "empirical_tolerance_from_reference_ions"
    params_cls = EmpiricalToleranceParams

    def default_params(self, ep: ExperimentParams) -> EmpiricalToleranceParams:
        return EmpiricalToleranceParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: EmpiricalToleranceParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        ref: ReferenceSet | None = ds.extra.get("reference_set")
        if ref is None:
            raise RuntimeError(
                "empirical_tolerance_from_reference_ions requires detect_reference_ions "
                "to have been applied first."
            )
        if ref.mz.size < 3:
            # Not enough anchors to fit a meaningful curve. Return a flat tolerance.
            return _flat_tolerance(ds, params, ref, rng, op_name=self.name)

        # Stack all (m/z_centroid, |ppm_error|) pairs from per-pixel observations.
        # per_pixel_ppm_error has shape (n_pixels, R); NaN where missing.
        ppm = np.abs(ref.per_pixel_ppm_error)
        mz_grid_per_obs = np.broadcast_to(ref.mz[None, :], ppm.shape)
        valid = ~np.isnan(ppm)
        x = mz_grid_per_obs[valid].astype(np.float64)
        y = ppm[valid].astype(np.float64)

        if x.size < 5:
            return _flat_tolerance(ds, params, ref, rng, op_name=self.name)

        # Sort by m/z for isotonic regression input.
        order = np.argsort(x)
        x_s = x[order]
        y_s = y[order]

        # Output grid: 200 evenly-spaced points across the observed m/z range.
        mz_lo, mz_hi = float(x_s[0]), float(x_s[-1])
        mz_grid = np.linspace(mz_lo, mz_hi, num=200)

        # Point estimate on the full sample.
        ppm_q = _isotonic_quantile_curve(x_s, y_s, mz_grid, alpha=params.alpha)

        # Bootstrap CI.
        B = max(int(params.bootstrap_B), 1)
        boot_curves = np.empty((B, mz_grid.size), dtype=np.float64)
        coords = ds.coords
        if params.block_bootstrap:
            block_blocks = _make_pixel_blocks(coords, ds.grid_shape, params.block_size_pixels)
            for b in range(B):
                pixel_sample = _block_resample(block_blocks, ds.n_pixels, rng)
                xb, yb = _gather_pairs(ref, pixel_sample)
                boot_curves[b, :] = _isotonic_quantile_curve(
                    xb, yb, mz_grid, alpha=params.alpha
                )
        else:
            for b in range(B):
                pixel_sample = rng.integers(0, ds.n_pixels, size=ds.n_pixels)
                xb, yb = _gather_pairs(ref, pixel_sample)
                boot_curves[b, :] = _isotonic_quantile_curve(
                    xb, yb, mz_grid, alpha=params.alpha
                )

        ci_low = np.quantile(boot_curves, 0.025, axis=0)
        ci_high = np.quantile(boot_curves, 0.975, axis=0)

        curve = ToleranceCurve(
            mz_grid=mz_grid,
            ppm_quantile=ppm_q,
            ci_low=ci_low,
            ci_high=ci_high,
            alpha=params.alpha,
            bootstrap_B=B,
        )

        new_extra = {**ds.extra, "tolerance_curve": curve}
        new_ds = _ds_with_extra(ds, new_extra)

        diag = Diagnostic(
            name=self.name,
            summary={
                "ppm_at_mz_lo": float(curve.ppm_quantile[0]),
                "ppm_at_mz_hi": float(curve.ppm_quantile[-1]),
                "ppm_median": float(np.median(curve.ppm_quantile)),
                "ci_band_median_width": float(np.median(curve.ci_high - curve.ci_low)),
                "n_reference_observations": float(x.size),
                "alpha": float(params.alpha),
                "bootstrap_B": float(B),
            },
            payload={
                "mz_grid": curve.mz_grid,
                "ppm_quantile": curve.ppm_quantile,
                "ci_low": curve.ci_low,
                "ci_high": curve.ci_high,
            },
            figure_hint="line_with_band:mz_vs_tolerance",
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


def _isotonic_quantile_curve(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, *, alpha: float
) -> np.ndarray:
    """Fit a monotone-non-decreasing (1 - α/2)-quantile of `y` as function of `x`.

    Approach: bin `x` into ~30 bins, take the (1 - α/2) quantile of `y` within each
    bin, then run a pool-adjacent-violators (PAV) pass to enforce monotonicity, then
    interpolate onto `grid`. This is distribution-free.
    """
    if x.size == 0:
        return np.zeros_like(grid)
    n_bins = max(8, min(30, x.size // 5))
    bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # Ensure strictly increasing edges (deduplicate ties).
    bin_edges = np.unique(bin_edges)
    if bin_edges.size < 2:
        return np.full_like(grid, float(np.quantile(np.abs(y), 1 - alpha / 2)))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    qs = np.empty(centers.size, dtype=np.float64)
    for i in range(centers.size):
        if i == centers.size - 1:
            mask = (x >= bin_edges[i]) & (x <= bin_edges[i + 1])
        else:
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if not mask.any():
            qs[i] = qs[i - 1] if i > 0 else 0.0
        else:
            qs[i] = float(np.quantile(np.abs(y[mask]), 1 - alpha / 2))

    # Pool-Adjacent-Violators for monotone non-decreasing.
    qs_pav = _pav_increasing(qs)
    return np.interp(grid, centers, qs_pav, left=qs_pav[0], right=qs_pav[-1])


def _pav_increasing(y: np.ndarray) -> np.ndarray:
    """In-place pool-adjacent-violators algorithm enforcing y[i+1] >= y[i]."""
    out = y.astype(np.float64, copy=True)
    weights = np.ones_like(out)
    n = out.size
    if n <= 1:
        return out
    i = 0
    while i < n - 1:
        if out[i + 1] >= out[i]:
            i += 1
            continue
        # pool [i, i+1] and walk back to maintain monotonicity
        j = i
        total = out[j] * weights[j] + out[j + 1] * weights[j + 1]
        w = weights[j] + weights[j + 1]
        while j > 0 and out[j - 1] > total / w:
            j -= 1
            total += out[j] * weights[j]
            w += weights[j]
        new = total / w
        # Spread the pooled value across all pooled positions.
        # Reconstruct positions: from j to i+1.
        out[j : i + 2] = new
        weights[j : i + 2] = w / (i + 2 - j)
        i = max(j, i + 1)
    return out


def _make_pixel_blocks(
    coords: np.ndarray, grid_shape: tuple[int, int], block_size: int
) -> list[np.ndarray]:
    """Group pixels into roughly square spatial blocks for block bootstrap."""
    h, w = grid_shape
    block_w = max(1, block_size)
    bx = (coords[:, 0] - coords[:, 0].min()) // block_w
    by = (coords[:, 1] - coords[:, 1].min()) // block_w
    block_id = bx * (h // block_w + 1) + by
    blocks: dict[int, list[int]] = {}
    for i, b in enumerate(block_id):
        blocks.setdefault(int(b), []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in blocks.values()]


def _block_resample(
    blocks: list[np.ndarray], target_n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample blocks (with replacement) until we have at least `target_n` pixels."""
    if not blocks:
        return rng.integers(0, target_n, size=target_n)
    out: list[int] = []
    while len(out) < target_n:
        idx = rng.integers(0, len(blocks))
        out.extend(int(p) for p in blocks[idx])
    return np.asarray(out[:target_n], dtype=np.int64)


def _gather_pairs(ref: ReferenceSet, pixel_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take a bootstrap sample of pixels and flatten their (mz, |ppm|) observations."""
    sub_ppm = np.abs(ref.per_pixel_ppm_error[pixel_idx, :])
    mz_grid = np.broadcast_to(ref.mz[None, :], sub_ppm.shape)
    valid = ~np.isnan(sub_ppm)
    return mz_grid[valid].astype(np.float64), sub_ppm[valid].astype(np.float64)


def _flat_tolerance(
    ds: MSIDataset,
    params: EmpiricalToleranceParams,
    ref: ReferenceSet,
    rng: np.random.Generator,
    *,
    op_name: str,
) -> OpResult:
    """Fall-back when there are too few reference observations to fit a curve.

    Emit a constant tolerance estimated from whatever we have; the diagnostic flags
    the small-N condition so the wizard's review page can warn the user.
    """
    ppm = np.abs(ref.per_pixel_ppm_error[~np.isnan(ref.per_pixel_ppm_error)])
    if ppm.size == 0:
        const = 50.0  # a nominal default
    else:
        const = float(np.quantile(ppm, 1 - params.alpha / 2))

    mz_lo = float(ds.metadata.mz_min) if ds.metadata.mz_min > 0 else 100.0
    mz_hi = float(ds.metadata.mz_max) if ds.metadata.mz_max > mz_lo else mz_lo + 1.0
    mz_grid = np.linspace(mz_lo, mz_hi, num=200)
    flat = np.full_like(mz_grid, const)
    curve = ToleranceCurve(
        mz_grid=mz_grid,
        ppm_quantile=flat,
        ci_low=flat,
        ci_high=flat,
        alpha=params.alpha,
        bootstrap_B=0,
    )
    new_ds = _ds_with_extra(ds, {**ds.extra, "tolerance_curve": curve})
    diag = Diagnostic(
        name=op_name,
        summary={
            "ppm_constant": const,
            "n_reference_observations": float(ppm.size),
            "warning_flat_tolerance": 1.0,
        },
        figure_hint=None,
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


def _ds_with_extra(ds: MSIDataset, new_extra: dict) -> MSIDataset:
    return ds.__class__(
        coords=ds.coords,
        grid_shape=ds.grid_shape,
        metadata=ds.metadata,
        backend=ds.backend,
        identity=ds.identity,
        history=ds.history,
        rois=ds.rois,
        rng_seed=ds.rng_seed,
        extra=new_extra,
    )
