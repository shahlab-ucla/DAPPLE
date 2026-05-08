"""Per-pixel mass recalibration from reference-ion observations.

Per-pixel mass drift is the dominant residual in unrecalibrated TOF / Q-TOF data
and is non-trivial even on well-locked Orbitrap acquisitions. After
``detect_reference_ions`` has anchored a small set of m/z values that recur
across the image, this operator fits a per-pixel piecewise-linear warp from each
pixel's *observed* reference m/z values to their *consensus* centroids, with
RANSAC-based outlier rejection so a single misidentified anchor doesn't drag the
warp.

Algorithm (one pixel at a time)

1. Gather visible (m/z_observed, m/z_reference) pairs for this pixel from the
   ``ReferenceSet`` attached upstream.
2. **RANSAC outlier rejection**: try ``ransac_n_trials`` random size-2 subsets;
   for each, fit a linear m/z → m/z map and count inliers (other pairs whose
   |residual| < ``ransac_threshold_ppm``). Keep the trial with the most
   inliers.
3. **Piecewise-linear warp through inliers**: sort the surviving pairs by
   m/z_observed; build a linear interpolant ``f: m/z_obs → m/z_ref`` via
   ``numpy.interp``. Outside the anchor range we extrapolate by holding the
   end-point correction constant — never extrapolate the slope past the data.
4. **Apply** ``f`` to every peak m/z in the pixel.

Pixels with fewer than ``min_inliers`` surviving anchors are left untouched and
recorded in the diagnostic so the user can see how often the fit failed.

The operator emits two diagnostics: a per-pixel ppm-residual scatter (pre vs
post) and a per-pixel inlier-count distribution. Together they tell you both
whether recalibration helped *globally* and which pixels it gave up on.
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
from dapple.ops.reference_ions import ReferenceSet
from dapple.ops.tolerance import ToleranceCurve


@dataclass(frozen=True)
class MsiwarpRecalibrateParams(OpParams):
    ransac_threshold_ppm: float = field(
        default=0.0,
        metadata={
            "label": "RANSAC inlier threshold (ppm)",
            "help": (
                "Maximum |ppm error| an anchor can have and still count as a RANSAC "
                "inlier. Default 0 means 'use 2× the empirical tolerance from the "
                "upstream tolerance curve at each anchor's m/z'. Override with an "
                "explicit ppm value to harden or relax outlier rejection. Smaller = "
                "stricter; larger = more anchors retained but noisier fit."
            ),
        },
    )
    ransac_n_trials: int = field(
        default=64,
        metadata={
            "label": "RANSAC trials per pixel",
            "help": (
                "Number of random 2-anchor subsets to try when looking for the "
                "best-supported linear fit. Default 64. With ≥ 5 anchors per pixel "
                "this finds a good seed essentially every time; raise to 256 only "
                "when you have many anchors with heavy contamination."
            ),
        },
    )
    min_inliers: int = field(
        default=3,
        metadata={
            "label": "Minimum inliers per pixel",
            "help": (
                "Skip recalibration for any pixel where fewer than this many anchors "
                "survive RANSAC. The pixel's m/z values are left untouched. Default 3 "
                "— two-point linear fits are unreliable. Raise to be conservative."
            ),
        },
    )
    rng_seed: int = field(
        default=0,
        metadata={
            "label": "RNG seed (per-pixel offset)",
            "help": (
                "Seed for the RANSAC random subset selection. Mixed with the global "
                "pipeline seed so re-runs are reproducible. Don't change unless you "
                "want to deliberately re-roll a stuck pixel."
            ),
        },
    )


@register
class MsiwarpRecalibrate(Operator):
    name = "msiwarp_recalibrate"
    params_cls = MsiwarpRecalibrateParams

    def default_params(self, ep: ExperimentParams) -> MsiwarpRecalibrateParams:
        return MsiwarpRecalibrateParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        out: list[str] = []
        if ep.instrument_family in {"orbitrap", "fticr"}:
            out.append(
                "Recalibration is rarely needed for high-resolution analyzers "
                "(Orbitrap / FT-ICR). Consider skipping this step or using "
                "ransac_threshold_ppm < 1."
            )
        return out

    def apply(
        self,
        ds: MSIDataset,
        params: MsiwarpRecalibrateParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "msiwarp_recalibrate currently requires a PeakList backend. "
                "Run it before consensus alignment."
            )
        ref: ReferenceSet | None = ds.extra.get("reference_set")
        if ref is None:
            raise RuntimeError(
                "msiwarp_recalibrate requires detect_reference_ions to have run first."
            )
        if ref.mz.size < 2:
            # Nothing to fit — pass through with a clear diagnostic.
            return _passthrough(ds, params, self.name, "fewer than 2 reference ions")

        tol_curve: ToleranceCurve | None = ds.extra.get("tolerance_curve")
        per_pixel_obs = ref.per_pixel_mz  # (n_pixels, R), NaN where missing
        per_pixel_ref = ref.mz  # (R,)
        n_pixels = per_pixel_obs.shape[0]

        # Per-anchor RANSAC threshold: either user-supplied or 2× the empirical
        # tolerance from the upstream curve.
        if params.ransac_threshold_ppm > 0:
            anchor_thresh_ppm = np.full(per_pixel_ref.size, float(params.ransac_threshold_ppm))
        elif tol_curve is not None:
            anchor_thresh_ppm = 2.0 * tol_curve.evaluate(per_pixel_ref)
        else:
            # No tolerance curve — fall back to a generous 50 ppm.
            anchor_thresh_ppm = np.full(per_pixel_ref.size, 50.0)

        # Pre-recalibration residual stats for the diagnostic.
        pre_ppm = np.abs(ref.per_pixel_ppm_error).copy()
        pre_ppm_valid = pre_ppm[~np.isnan(pre_ppm)]
        pre_median_ppm = float(np.median(pre_ppm_valid)) if pre_ppm_valid.size else np.nan

        pl = ds.backend
        offsets = np.asarray(pl.offsets[:])
        mz_in = np.asarray(pl.mz[:]).astype(np.float64, copy=True)
        intensity = np.asarray(pl.intensity[:])

        per_pixel_inliers = np.zeros(n_pixels, dtype=np.int64)
        per_pixel_used = np.zeros(n_pixels, dtype=bool)
        post_ppm_residuals_per_pixel = np.full_like(per_pixel_obs, np.nan)
        # Per-pixel rng derived from master seed + pixel index for reproducibility.
        master_seed = int(rng.integers(0, 2**31 - 1))

        # Walk pixels.
        for i in range(n_pixels):
            obs_row = per_pixel_obs[i, :]
            valid = ~np.isnan(obs_row)
            if int(valid.sum()) < params.min_inliers:
                continue
            obs_pts = obs_row[valid]
            ref_pts = per_pixel_ref[valid]
            anchor_thr = anchor_thresh_ppm[valid]

            inlier_mask = _ransac_linear_inliers(
                obs_pts,
                ref_pts,
                anchor_thr_ppm=anchor_thr,
                n_trials=int(params.ransac_n_trials),
                rng=np.random.default_rng(master_seed + i),
            )
            n_in = int(inlier_mask.sum())
            if n_in < params.min_inliers:
                continue

            inlier_obs = obs_pts[inlier_mask]
            inlier_ref = ref_pts[inlier_mask]
            # Sort for the piecewise-linear warp.
            order = np.argsort(inlier_obs)
            anchors_obs = inlier_obs[order]
            anchors_ref = inlier_ref[order]

            a, b = int(offsets[i]), int(offsets[i + 1])
            per_pixel_inliers[i] = n_in
            per_pixel_used[i] = True
            if a == b:
                continue

            # Out-of-range queries shift by the nearest anchor's correction
            # (translation, no slope extrapolation past the data).
            offset_lo = float(anchors_ref[0] - anchors_obs[0])
            offset_hi = float(anchors_ref[-1] - anchors_obs[-1])

            # Apply the piecewise-linear warp to every peak m/z in this pixel,
            # working from the original m/z slice (not from mz_in, which we are
            # about to overwrite).
            seg = np.asarray(pl.mz[:])[a:b].astype(np.float64, copy=True)
            warped = np.interp(seg, anchors_obs, anchors_ref)
            below = seg < anchors_obs[0]
            above = seg > anchors_obs[-1]
            warped[below] = seg[below] + offset_lo
            warped[above] = seg[above] + offset_hi
            mz_in[a:b] = warped

            # Per-anchor post-warp residuals for the diagnostic.
            warped_anchors = np.interp(obs_pts, anchors_obs, anchors_ref)
            below_a = obs_pts < anchors_obs[0]
            above_a = obs_pts > anchors_obs[-1]
            warped_anchors[below_a] = obs_pts[below_a] + offset_lo
            warped_anchors[above_a] = obs_pts[above_a] + offset_hi
            ppm_post = (warped_anchors - ref_pts) / ref_pts * 1e6
            post_ppm_residuals_per_pixel[i, valid] = np.abs(ppm_post)

        new_pl = PeakList(
            mz=mz_in,
            intensity=intensity.copy(),
            offsets=offsets.copy(),
            n_pixels=pl.n_pixels,
        )
        new_ds = ds.with_backend(new_pl)

        post_ppm_valid = post_ppm_residuals_per_pixel[~np.isnan(post_ppm_residuals_per_pixel)]
        post_median_ppm = float(np.median(post_ppm_valid)) if post_ppm_valid.size else np.nan
        improvement = (
            float(pre_median_ppm - post_median_ppm)
            if not np.isnan(pre_median_ppm) and not np.isnan(post_median_ppm)
            else 0.0
        )
        diag = Diagnostic(
            name=self.name,
            summary={
                "n_pixels_recalibrated": float(int(per_pixel_used.sum())),
                "n_pixels_skipped": float(int((~per_pixel_used).sum())),
                "fraction_pixels_recalibrated": float(per_pixel_used.mean()),
                "inliers_median": float(np.median(per_pixel_inliers[per_pixel_used]))
                if per_pixel_used.any()
                else 0.0,
                "inliers_min": float(int(per_pixel_inliers[per_pixel_used].min()))
                if per_pixel_used.any()
                else 0.0,
                "pre_residual_median_ppm": pre_median_ppm,
                "post_residual_median_ppm": post_median_ppm,
                "improvement_ppm": improvement,
            },
            payload={
                "per_pixel_inliers": per_pixel_inliers,
                "per_pixel_used": per_pixel_used,
                "pre_ppm_residual": pre_ppm,
                "post_ppm_residual": post_ppm_residuals_per_pixel,
            },
            figure_hint="scatter:pre_post_ppm_residual",
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


def _ransac_linear_inliers(
    obs: np.ndarray,
    ref: np.ndarray,
    *,
    anchor_thr_ppm: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """RANSAC over 2-anchor subsets; return the inlier mask of the best trial.

    "Inlier" = anchor whose ppm residual under the trial's linear fit is below
    the per-anchor threshold. Ties are broken by minimizing total residual sum.
    """
    n = obs.size
    if n < 2:
        return np.zeros(n, dtype=bool)
    if n == 2:
        return np.ones(2, dtype=bool)

    best_mask = np.zeros(n, dtype=bool)
    best_count = -1
    best_sse = np.inf
    # Cap trials at the number of distinct 2-subsets — more is wasted work.
    n_trials = min(int(n_trials), n * (n - 1) // 2)
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while attempts < n_trials and len(seen) < n * (n - 1) // 2:
        attempts += 1
        i, j = rng.choice(n, size=2, replace=False)
        key = (int(min(i, j)), int(max(i, j)))
        if key in seen:
            continue
        seen.add(key)
        o1, o2 = float(obs[i]), float(obs[j])
        r1, r2 = float(ref[i]), float(ref[j])
        if o1 == o2:
            continue
        slope = (r2 - r1) / (o2 - o1)
        intercept = r1 - slope * o1
        pred = slope * obs + intercept
        residual_ppm = np.abs((pred - ref) / ref) * 1e6
        mask = residual_ppm <= anchor_thr_ppm
        count = int(mask.sum())
        if count == 0:
            continue
        sse = float((residual_ppm[mask] ** 2).sum())
        if count > best_count or (count == best_count and sse < best_sse):
            best_mask = mask
            best_count = count
            best_sse = sse
    return best_mask


@dataclass(frozen=True)
class LockMassRecalibrateParams(OpParams):
    """Parameters for single-anchor (a.k.a. lock-mass) per-pixel recalibration.

    Where ``msiwarp_recalibrate`` fits a linear or piecewise-linear warp from
    *several* reference anchors per pixel, this operator picks one anchor per
    pixel and shifts every peak in the pixel by the matching ppm offset. Useful
    when only one reference ion is reliable, or when the dataset doesn't have
    enough anchors per pixel for a robust linear fit.
    """

    anchor_strategy: str = field(
        default="highest_intensity",
        metadata={
            "label": "Anchor strategy",
            "help": (
                "Which reference ion to use as the lock-mass anchor in each pixel. "
                "'highest_intensity' picks the ion with the strongest observed "
                "signal in that pixel — the most defensible default since "
                "low-intensity anchors carry larger m/z uncertainty. "
                "'highest_prevalence' picks the globally most-prevalent reference "
                "(stable across pixels but may be dim in some). 'closest_to_mz' "
                "picks whichever reference centroid is nearest to "
                "``explicit_anchor_mz``."
            ),
        },
    )
    explicit_anchor_mz: float = field(
        default=0.0,
        metadata={
            "label": "Explicit anchor m/z",
            "help": (
                "Only used when ``anchor_strategy = 'closest_to_mz'``. Specify the "
                "m/z of a known stable lock-mass molecule (e.g. a matrix peak you "
                "trust). The operator picks the reference centroid nearest this "
                "value as the anchor for every pixel. Default 0 means 'unset'."
            ),
        },
    )
    max_shift_ppm: float = field(
        default=500.0,
        metadata={
            "label": "Maximum allowed shift (ppm)",
            "help": (
                "Refuse to apply a per-pixel shift larger than this magnitude in "
                "ppm — a sanity check against single misidentified anchors that "
                "would otherwise blow up the pixel's m/z. Default 500 ppm "
                "covers axial linear MALDI-TOF; tighten for high-resolution "
                "instruments."
            ),
        },
    )


@register
class LockMassRecalibrate(Operator):
    """Single-anchor per-pixel m/z shift, a.k.a. classical lock-mass correction.

    For each pixel, picks one reference ion as the anchor and shifts every peak
    in the pixel by the matching ppm offset. Compared to ``msiwarp_recalibrate``
    this is much simpler (no linear fit, no RANSAC) and uses only one anchor —
    so it's the right choice when reference ions are sparse, or when a single
    reliable matrix peak is all you have.
    """

    name = "lock_mass_recalibrate"
    params_cls = LockMassRecalibrateParams

    def default_params(self, ep: ExperimentParams) -> LockMassRecalibrateParams:
        return LockMassRecalibrateParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: LockMassRecalibrateParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "lock_mass_recalibrate currently requires a PeakList backend."
            )
        ref: ReferenceSet | None = ds.extra.get("reference_set")
        if ref is None:
            raise RuntimeError(
                "lock_mass_recalibrate requires detect_reference_ions to have run first."
            )
        if ref.mz.size == 0:
            return _lockmass_passthrough(ds, params, "no reference ions available")

        # Choose an anchor index per pixel.
        per_pixel_obs = ref.per_pixel_mz  # (n_pixels, R), NaN where missing
        per_pixel_int = ref.per_pixel_intensity  # (n_pixels, R)
        n_pixels, n_refs = per_pixel_obs.shape
        anchor_idx = -np.ones(n_pixels, dtype=np.int64)

        if params.anchor_strategy == "closest_to_mz":
            if params.explicit_anchor_mz <= 0:
                raise RuntimeError(
                    "anchor_strategy='closest_to_mz' requires explicit_anchor_mz > 0."
                )
            target_idx = int(np.argmin(np.abs(ref.mz - params.explicit_anchor_mz)))
            visible = ~np.isnan(per_pixel_obs[:, target_idx])
            anchor_idx[visible] = target_idx
        elif params.anchor_strategy == "highest_prevalence":
            prevalence_order = np.argsort(ref.prevalence)[::-1]
            for i in range(n_pixels):
                for cand in prevalence_order:
                    if not np.isnan(per_pixel_obs[i, cand]):
                        anchor_idx[i] = int(cand)
                        break
        else:  # highest_intensity (default)
            visible = ~np.isnan(per_pixel_obs)
            # Replace missing with -inf so argmax is meaningful.
            scores = np.where(visible, per_pixel_int, -np.inf)
            best = scores.argmax(axis=1)
            any_visible = visible.any(axis=1)
            anchor_idx = np.where(any_visible, best, -1).astype(np.int64)

        n_used = int((anchor_idx >= 0).sum())
        if n_used == 0:
            return _lockmass_passthrough(ds, params, "no pixels have a visible anchor")

        # Compute per-pixel multiplicative shift = m/z_obs / m/z_ref. Each peak
        # m/z is divided by this factor (so observed m/z lands on its reference).
        shift_factor = np.ones(n_pixels, dtype=np.float64)
        applied_mask = np.zeros(n_pixels, dtype=bool)
        ppm_shift = np.zeros(n_pixels, dtype=np.float64)
        for i in range(n_pixels):
            ai = int(anchor_idx[i])
            if ai < 0:
                continue
            obs = float(per_pixel_obs[i, ai])
            r = float(ref.mz[ai])
            if not (obs > 0 and r > 0):
                continue
            ppm = (obs - r) / r * 1e6
            if abs(ppm) > params.max_shift_ppm:
                # Likely a misidentified anchor — leave the pixel alone.
                continue
            shift_factor[i] = obs / r
            ppm_shift[i] = ppm
            applied_mask[i] = True

        pl = ds.backend
        offsets = np.asarray(pl.offsets[:])
        mz_in = np.asarray(pl.mz[:]).astype(np.float64, copy=True)
        for i in range(n_pixels):
            if not applied_mask[i]:
                continue
            a, b = int(offsets[i]), int(offsets[i + 1])
            if a == b:
                continue
            mz_in[a:b] = mz_in[a:b] / shift_factor[i]

        new_pl = PeakList(
            mz=mz_in,
            intensity=np.asarray(pl.intensity[:]).copy(),
            offsets=offsets.copy(),
            n_pixels=pl.n_pixels,
        )
        new_ds = ds.with_backend(new_pl)
        diag = Diagnostic(
            name=self.name,
            summary={
                "n_pixels_recalibrated": float(int(applied_mask.sum())),
                "n_pixels_skipped": float(int((~applied_mask).sum())),
                "fraction_pixels_recalibrated": float(applied_mask.mean()),
                "ppm_shift_median": float(np.median(ppm_shift[applied_mask]))
                if applied_mask.any()
                else 0.0,
                "ppm_shift_abs_median": float(np.median(np.abs(ppm_shift[applied_mask])))
                if applied_mask.any()
                else 0.0,
                "ppm_shift_abs_max": float(np.abs(ppm_shift[applied_mask]).max())
                if applied_mask.any()
                else 0.0,
                "anchor_strategy_index": float(
                    {"highest_intensity": 0, "highest_prevalence": 1, "closest_to_mz": 2}.get(
                        params.anchor_strategy, 0
                    )
                ),
            },
            payload={
                "ppm_shift": ppm_shift,
                "anchor_idx": anchor_idx,
                "applied_mask": applied_mask,
            },
            figure_hint="histogram:per_pixel_shift",
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


def _lockmass_passthrough(
    ds: MSIDataset,
    params: LockMassRecalibrateParams,
    note: str,
) -> OpResult:
    diag = Diagnostic(
        name="lock_mass_recalibrate",
        summary={
            "n_pixels_recalibrated": 0.0,
            "n_pixels_skipped": float(ds.n_pixels),
            "fraction_pixels_recalibrated": 0.0,
        },
        payload={"note": np.array([note], dtype=object)},
    )
    record = merge_op_record(
        op_name="lock_mass_recalibrate",
        params=params,
        input_ds=ds,
        output_ds=ds,
        diagnostics=[diag],
    )
    return OpResult(dataset=ds.with_history(record), diagnostics=[diag])


def _passthrough(
    ds: MSIDataset,
    params: MsiwarpRecalibrateParams,
    op_name: str,
    note: str,
) -> OpResult:
    diag = Diagnostic(
        name=op_name,
        summary={
            "n_pixels_recalibrated": 0.0,
            "n_pixels_skipped": float(ds.n_pixels),
            "fraction_pixels_recalibrated": 0.0,
        },
        payload={"note": np.array([note], dtype=object)},
    )
    record = merge_op_record(
        op_name=op_name, params=params, input_ds=ds, output_ds=ds, diagnostics=[diag]
    )
    return OpResult(dataset=ds.with_history(record), diagnostics=[diag])
