"""detect_reference_ions: find endogenous peaks that recur across many pixels.

Identify N ≥ 5 (ideally ≥ 20) reference ions present in at least ``min_prevalence``
(default 0.8 centroided / 0.7 profile) of pixels. These anchor (a) empirical-
tolerance fitting and (b) recalibration. We don't require any particular molecule
list; matrix peaks, common contaminants, internal standards — any peak that is
consistently present across the image qualifies.

Algorithm
---------
1. Define log-spaced m/z bins of width `coarse_tol_ppm` (default 50). Bin index of m/z is
   `floor(log(m/z) / log(1 + 1e-6 * ppm))`.
2. For each pixel, mark the set of bins it contributes a peak to. The most-intense peak
   in each bin per pixel "owns" that bin for that pixel (so duplicate peaks in the same
   bin don't double-count).
3. Bin prevalence = fraction of pixels marking that bin.
4. Keep bins with prevalence ≥ `min_prevalence` and at least `min_count` total pixels.
5. Merge adjacent kept bins (a single reference ion can straddle a boundary).
6. For each merged group, the centroid m/z is the intensity-weighted mean across all
   contributing peaks in all pixels.

Output
------
A `ReferenceSet` with: per-reference m/z centroids and prevalences; a per-pixel matrix
of observed m/z (NaN where missing), intensities (0 where missing), and ppm errors
relative to the centroid (NaN where missing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class ReferenceIonsParams(OpParams):
    coarse_tol_ppm: float = field(
        default=50.0,
        metadata={
            "label": "Reference m/z tolerance (ppm)",
            "help": (
                "Width of the bin used to group similar m/z values across pixels when "
                "looking for repeating peaks. Defaults: ~5 ppm (Orbitrap/FT-ICR), "
                "~50 ppm (reflectron TOF / Q-TOF), ~200 ppm (axial linear MALDI-TOF). "
                "Increase if your instrument is uncalibrated or drifty; decrease if "
                "real peaks fall closer together than this window."
            ),
        },
    )
    min_prevalence: float = field(
        default=0.8,
        metadata={
            "label": "Minimum pixel prevalence",
            "help": (
                "Fraction of pixels in which a peak must appear to be treated as a "
                "reference ion. Default 0.8 for centroided data, 0.7 for profile. "
                "Lower (e.g. 0.5) if you expect few stable peaks across the image; "
                "raise toward 0.95 to be very selective."
            ),
        },
    )
    min_count: int = field(
        default=5,
        metadata={
            "label": "Minimum supporting pixels",
            "help": (
                "Absolute floor on the pixel count for a peak to qualify, regardless "
                "of prevalence. Default 5 — a minimum needed for stable downstream "
                "fits. Raise to be conservative on small images; rarely worth lowering."
            ),
        },
    )
    merge_adjacent_bins: bool = field(
        default=True,
        metadata={
            "label": "Merge adjacent bins",
            "help": (
                "When two neighbouring bins both pass the prevalence threshold, treat "
                "them as one reference ion (handles peaks that straddle a bin edge). "
                "Default ON. Disable only when you have very dense peaks that you "
                "expect to remain distinguishable at the configured tolerance."
            ),
        },
    )


@dataclass(frozen=True)
class ReferenceSet:
    """Output of detect_reference_ions, attached to MSIDataset.extra['reference_set']."""

    mz: np.ndarray  # (R,) float64 — centroid m/z of each reference ion
    prevalence: np.ndarray  # (R,) float64 in [0, 1]
    n_observations: np.ndarray  # (R,) int64 — pixel count contributing to each
    per_pixel_mz: np.ndarray  # (n_pixels, R) float64, NaN where missing
    per_pixel_intensity: np.ndarray  # (n_pixels, R) float32, 0 where missing
    per_pixel_ppm_error: np.ndarray  # (n_pixels, R) float64, NaN where missing


@register
class DetectReferenceIons(Operator):
    name = "detect_reference_ions"
    params_cls = ReferenceIonsParams

    def default_params(self, ep: ExperimentParams) -> ReferenceIonsParams:
        # Profile data has more noise floor — be a touch more permissive.
        prevalence = 0.7 if ep.profile_or_centroided == "profile" else 0.8
        # Larger coarse bins for low-resolution / axial TOF.
        if ep.instrument_family == "tof_axial":
            tol = 200.0
        elif ep.instrument_family in {"tof_reflectron", "qtof"}:
            tol = 50.0
        elif ep.instrument_family in {"orbitrap", "fticr"}:
            tol = 5.0
        else:
            tol = 50.0
        return ReferenceIonsParams(
            coarse_tol_ppm=tol,
            min_prevalence=prevalence,
            min_count=5,
        )

    def validate(self, ep: ExperimentParams) -> list[str]:
        warnings: list[str] = []
        if ep.instrument_family == "unknown":
            warnings.append(
                "instrument_family is unknown — default coarse_tol_ppm may be too tight or loose."
            )
        return warnings

    def apply(
        self,
        ds: MSIDataset,
        params: ReferenceIonsParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "detect_reference_ions currently requires a PeakList backend; "
                "running it on already-aligned data is unintended."
            )

        pl = ds.backend
        offsets = np.asarray(pl.offsets[:])
        mz_all = np.asarray(pl.mz[:])
        int_all = np.asarray(pl.intensity[:])

        if mz_all.size == 0:
            return _empty_result(ds, params, self.name)

        # ---- Step 1: log-bin every peak ----
        mz_min = float(mz_all.min())
        # log step = ln(1 + tol_ppm * 1e-6); bin index = floor(log(m/z) / step) - bin0
        step = np.log1p(params.coarse_tol_ppm * 1e-6)
        bin0 = int(np.floor(np.log(mz_min) / step))
        bin_idx = (np.floor(np.log(mz_all) / step) - bin0).astype(np.int64)
        n_bins = int(bin_idx.max()) + 1

        # ---- Step 2: per-pixel "best peak per bin" ----
        # We need: for each (pixel, bin), did the pixel contribute a peak? If so, which
        # m/z and intensity? Use a sparse structure: list of (pixel_idx, bin, mz, int).
        n_pixels = ds.n_pixels
        # Pixel index per peak:
        pixel_of_peak = np.repeat(
            np.arange(n_pixels, dtype=np.int64), np.diff(offsets).astype(np.int64)
        )

        # Within each (pixel, bin), keep the peak with the largest intensity. Sort by
        # (pixel, bin, -intensity), then take the first occurrence of each (pixel, bin).
        order = np.lexsort((-int_all, bin_idx, pixel_of_peak))
        sorted_pix = pixel_of_peak[order]
        sorted_bin = bin_idx[order]
        sorted_mz = mz_all[order]
        sorted_int = int_all[order]
        # Mark first occurrence per (pixel, bin):
        if order.size == 0:
            return _empty_result(ds, params, self.name)
        is_first = np.empty(order.size, dtype=bool)
        is_first[0] = True
        is_first[1:] = (sorted_pix[1:] != sorted_pix[:-1]) | (sorted_bin[1:] != sorted_bin[:-1])
        u_pix = sorted_pix[is_first]
        u_bin = sorted_bin[is_first]
        u_mz = sorted_mz[is_first]
        u_int = sorted_int[is_first]

        # ---- Step 3: per-bin prevalence ----
        bin_count = np.zeros(n_bins, dtype=np.int64)
        np.add.at(bin_count, u_bin, 1)
        prevalence = bin_count.astype(np.float64) / n_pixels
        keep_bin = (prevalence >= params.min_prevalence) & (bin_count >= params.min_count)

        # ---- Step 4: merge adjacent kept bins ----
        if params.merge_adjacent_bins:
            group_id = _adjacent_groups(keep_bin)
        else:
            group_id = np.where(keep_bin, np.arange(n_bins, dtype=np.int64), -1)

        kept_groups = np.unique(group_id[group_id >= 0])
        n_refs = kept_groups.size
        if n_refs == 0:
            return _empty_result(ds, params, self.name)

        # Map old group id -> new index in [0, R)
        new_idx = -np.ones(n_bins, dtype=np.int64)
        # group_id is per-bin; remap kept_groups to [0, R):
        for ri, gid in enumerate(kept_groups):
            new_idx[group_id == gid] = ri

        # ---- Step 5: aggregate per-reference per-pixel observations ----
        # For each unique-(pixel, bin) entry whose bin is in a kept group, accumulate
        # into per_pixel_mz / per_pixel_intensity at column = new_idx[bin].
        keep_mask = new_idx[u_bin] >= 0
        kp_pix = u_pix[keep_mask]
        kp_ref = new_idx[u_bin[keep_mask]]
        kp_mz = u_mz[keep_mask]
        kp_int = u_int[keep_mask]

        per_pixel_mz = np.full((n_pixels, n_refs), np.nan, dtype=np.float64)
        per_pixel_int = np.zeros((n_pixels, n_refs), dtype=np.float32)

        # If a pixel had peaks in multiple now-merged bins, sum intensities and use the
        # intensity-weighted mean m/z.
        # We aggregate again; small-N so a Python loop is fine.
        for px, rf, mz_v, in_v in zip(kp_pix, kp_ref, kp_mz, kp_int, strict=False):
            old_int = per_pixel_int[px, rf]
            if old_int == 0:
                per_pixel_mz[px, rf] = mz_v
                per_pixel_int[px, rf] = in_v
            else:
                new_int = old_int + in_v
                # weighted mean of m/z
                per_pixel_mz[px, rf] = (
                    per_pixel_mz[px, rf] * old_int + mz_v * in_v
                ) / new_int
                per_pixel_int[px, rf] = new_int

        # ---- Step 6: centroid m/z per reference (intensity-weighted across pixels) ----
        ref_mz = np.empty(n_refs, dtype=np.float64)
        n_obs = np.zeros(n_refs, dtype=np.int64)
        for r in range(n_refs):
            mz_col = per_pixel_mz[:, r]
            int_col = per_pixel_int[:, r].astype(np.float64)
            present = ~np.isnan(mz_col) & (int_col > 0)
            n_obs[r] = int(present.sum())
            wsum = int_col[present].sum()
            if wsum > 0:
                ref_mz[r] = float((mz_col[present] * int_col[present]).sum() / wsum)
            else:  # pragma: no cover — defensive; wouldn't be in kept_groups
                ref_mz[r] = float(np.nanmean(mz_col[present])) if present.any() else np.nan

        # Sort references by m/z so downstream consumers get them in spectral order.
        sort = np.argsort(ref_mz)
        ref_mz = ref_mz[sort]
        n_obs = n_obs[sort]
        per_pixel_mz = per_pixel_mz[:, sort]
        per_pixel_int = per_pixel_int[:, sort]
        ref_prevalence = n_obs.astype(np.float64) / n_pixels

        # ---- Step 7: per-pixel ppm error ----
        with np.errstate(invalid="ignore"):
            per_pixel_ppm_error = (per_pixel_mz - ref_mz[None, :]) / ref_mz[None, :] * 1e6

        ref_set = ReferenceSet(
            mz=ref_mz,
            prevalence=ref_prevalence,
            n_observations=n_obs,
            per_pixel_mz=per_pixel_mz,
            per_pixel_intensity=per_pixel_int,
            per_pixel_ppm_error=per_pixel_ppm_error,
        )

        diag = Diagnostic(
            name="detect_reference_ions",
            summary={
                "n_reference_ions": float(n_refs),
                "prevalence_min": float(ref_prevalence.min()),
                "prevalence_median": float(np.median(ref_prevalence)),
                "prevalence_max": float(ref_prevalence.max()),
                "n_pixels": float(n_pixels),
                "coarse_tol_ppm": float(params.coarse_tol_ppm),
                "min_prevalence": float(params.min_prevalence),
            },
            payload={
                "reference_mz": ref_mz,
                "prevalence": ref_prevalence,
                "n_observations": n_obs,
            },
            figure_hint="scatter:reference_mz_vs_prevalence",
        )

        # Stash on the dataset for downstream operators (tolerance, recalibrate).
        new_extra = {**ds.extra, "reference_set": ref_set}
        new_ds = ds.with_backend(ds.backend)  # no backend change; we just attach extra
        new_ds = new_ds.__class__(  # rebuild with new extra
            coords=new_ds.coords,
            grid_shape=new_ds.grid_shape,
            metadata=new_ds.metadata,
            backend=new_ds.backend,
            identity=new_ds.identity,
            history=new_ds.history,
            rois=new_ds.rois,
            rng_seed=new_ds.rng_seed,
            extra=new_extra,
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


def _adjacent_groups(keep: np.ndarray) -> np.ndarray:
    """For a boolean keep mask, return per-bin group id; -1 where !keep.

    Adjacent kept bins share a group id; gaps break groups.
    """
    out = -np.ones(keep.shape[0], dtype=np.int64)
    if not keep.any():
        return out
    edges = np.diff(keep.astype(np.int8), prepend=0)
    starts = np.flatnonzero(edges == 1)
    # Walk starts and assign ids until the group ends.
    end_edges = np.diff(keep.astype(np.int8), append=0)
    ends_excl = np.flatnonzero(end_edges == -1) + 1  # exclusive end indices
    for gid, (s, e) in enumerate(zip(starts, ends_excl, strict=True)):
        out[s:e] = gid
    return out


def _empty_result(ds: MSIDataset, params: ReferenceIonsParams, op_name: str) -> OpResult:
    """No reference ions detected — return the dataset unchanged with an empty ReferenceSet."""
    n_pixels = ds.n_pixels
    empty = ReferenceSet(
        mz=np.empty(0, dtype=np.float64),
        prevalence=np.empty(0, dtype=np.float64),
        n_observations=np.empty(0, dtype=np.int64),
        per_pixel_mz=np.empty((n_pixels, 0), dtype=np.float64),
        per_pixel_intensity=np.empty((n_pixels, 0), dtype=np.float32),
        per_pixel_ppm_error=np.empty((n_pixels, 0), dtype=np.float64),
    )
    new_extra = {**ds.extra, "reference_set": empty}
    new_ds = ds.__class__(
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
    diag = Diagnostic(
        name=op_name,
        summary={
            "n_reference_ions": 0.0,
            "n_pixels": float(n_pixels),
            "min_prevalence": float(params.min_prevalence),
        },
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
