"""KDE-based consensus alignment of per-pixel peaks into a shared m/z axis.

Algorithm: pool all picked peaks across pixels, fit a 1-D Gaussian KDE on log(m/z)
with a ppm-scaled bandwidth, call local maxima of the KDE as consensus peaks, then
for each consensus m/z extract per-pixel intensity (the most intense peak in the
tolerance window). A simple prevalence floor drops peaks observed in fewer than
``min_prevalence`` of pixels.

Output is an MSIDataset with a PeakMatrix backend keyed by the consensus m/z axis,
which is what the channels panel and TIFF writer consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.data.metadata import ExperimentParams
from dapple.ops.base import (
    Diagnostic,
    OpParams,
    OpResult,
    Operator,
    merge_op_record,
    register,
)
from dapple.ops.tolerance import ToleranceCurve


@dataclass(frozen=True)
class KdeConsensusParams(OpParams):
    """Parameters for KDE-based consensus alignment.

    `n_grid_points` — minimum m/z grid resolution for KDE evaluation. The effective
        grid is refined automatically to sample the requested bandwidth.
    `bandwidth_ppm` — Gaussian KDE bandwidth in ppm. The natural bandwidth for MSI is
        the per-peak m/z scatter (which is also the recalibration tolerance), NOT the
        Silverman rule (which derives a bandwidth from the *full* m/z range and
        produces wildly over-smoothed multimodal estimates here).
    `bandwidth_scale` — multiplier applied to `bandwidth_ppm`. <1 sharpens peak
        detection at the cost of splitting close peaks.
    `min_prominence_quantile` — local maxima must exceed this quantile of KDE density
        to be considered a candidate consensus peak. Default 0.5 (median density).
    `default_tol_ppm` — fallback tolerance window when no `tolerance_curve` is in
        ds.extra. Used to assign per-pixel peaks to consensus m/z bins.
    `min_prevalence` — drop consensus peaks present in fewer than this fraction of
        pixels (a simple, conservative prevalence filter).
    """

    n_grid_points: int = field(
        default=32768,
        metadata={
            "label": "KDE grid resolution",
            "help": (
                "Minimum number of evaluation points for the m/z density. DAPPLE "
                "automatically refines this to at least four samples per KDE "
                "bandwidth (capped at two million points), which prevents a 5 ppm "
                "Orbitrap bandwidth from being evaluated on a much coarser grid."
            ),
        },
    )
    bandwidth_ppm: float = field(
        default=50.0,
        metadata={
            "label": "KDE bandwidth (ppm)",
            "help": (
                "Width of the Gaussian kernel used to find consensus peaks. Should "
                "roughly match the per-pixel m/z scatter (i.e. the recalibration "
                "tolerance). Default 50 ppm fits reflectron TOF / Q-TOF data; use "
                "5 ppm for Orbitrap/FT-ICR, 200 ppm for axial linear MALDI-TOF. "
                "Tighten when peaks are unusually well-resolved; widen if drift is "
                "large enough that consensus peaks would otherwise split."
            ),
        },
    )
    bandwidth_scale: float = field(
        default=1.0,
        metadata={
            "label": "Bandwidth scale",
            "help": (
                "Multiplier on bandwidth (ppm) for fine-tuning. Default 1.0. Drop to "
                "0.5 to sharpen peak separation; raise to 1.5–2 to merge close peaks "
                "into one consensus channel. Prefer this over editing bandwidth_ppm."
            ),
        },
    )
    min_prominence_quantile: float = field(
        default=0.5,
        metadata={
            "label": "KDE prominence threshold (quantile)",
            "help": (
                "Local maxima of the KDE density must exceed this quantile of the "
                "density to count as a candidate. Default 0.5 (median density). "
                "Raise toward 0.7–0.9 to be more selective and keep only the most "
                "prominent peaks; lower to surface more candidates."
            ),
        },
    )
    default_tol_ppm: float = field(
        default=50.0,
        metadata={
            "label": "Per-channel m/z window (ppm)",
            "help": (
                "Tolerance window used to assign per-pixel peaks to a consensus m/z. "
                "Used only when no empirical tolerance curve is attached upstream. "
                "Default matches the instrument family (5 / 50 / 200 ppm for Orbitrap "
                "/ Q-TOF / linear TOF). Wider windows fold more peaks into each "
                "channel; narrower windows leave more pixels empty."
            ),
        },
    )
    min_prevalence: float = field(
        default=0.05,
        metadata={
            "label": "Drop peaks present in < this fraction of pixels",
            "help": (
                "Conservative prevalence floor: discard consensus peaks observed in "
                "fewer than this fraction of pixels. Default 0.05 (5%). Raise toward "
                "0.2 to be strict and only keep widespread peaks; lower to keep "
                "rare-but-real signals. For developmental studies, prefer cohort "
                "dataset prevalence when stage-local features must remain visible."
            ),
        },
    )

    def __post_init__(self) -> None:
        if self.n_grid_points < 3:
            raise ValueError("n_grid_points must be at least 3")
        if self.bandwidth_ppm <= 0 or self.bandwidth_scale <= 0:
            raise ValueError("bandwidth_ppm and bandwidth_scale must be positive")
        if not 0.0 <= self.min_prominence_quantile <= 1.0:
            raise ValueError("min_prominence_quantile must be in [0, 1]")
        if self.default_tol_ppm <= 0:
            raise ValueError("default_tol_ppm must be positive")
        if not 0.0 <= self.min_prevalence <= 1.0:
            raise ValueError("min_prevalence must be in [0, 1]")


@register
class KdeConsensusAlignment(Operator):
    name = "kde_consensus_alignment"
    params_cls = KdeConsensusParams

    def default_params(self, ep: ExperimentParams) -> KdeConsensusParams:
        if ep.instrument_family in {"orbitrap", "fticr"}:
            tol = 5.0
        elif ep.instrument_family in {"tof_reflectron", "qtof"}:
            tol = 50.0
        elif ep.instrument_family == "tof_axial":
            tol = 200.0
        else:
            tol = 50.0
        return KdeConsensusParams(default_tol_ppm=tol, bandwidth_ppm=tol)

    def apply(
        self,
        ds: MSIDataset,
        params: KdeConsensusParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "kde_consensus_alignment requires a PeakList backend (it produces a "
                "PeakMatrix)."
            )
        pl = ds.backend
        offsets = np.asarray(pl.offsets[:])
        peak_mz = np.asarray(pl.mz[:])
        peak_int = np.asarray(pl.intensity[:])

        if peak_mz.size == 0:
            raise RuntimeError("dataset has no peaks; cannot compute consensus alignment.")

        peak_pixel = np.repeat(
            np.arange(ds.n_pixels, dtype=np.int64), np.diff(offsets).astype(np.int64)
        )

        # ---- Pool peaks and fit KDE on log m/z (so a constant ppm bandwidth is uniform) ----
        # In log space, log(1 + ppm*1e-6) ≈ ppm*1e-6 for small ppm, so the bandwidth
        # in log-m/z is just the ppm tolerance times 1e-6.
        log_mz = np.log(peak_mz)
        weights = peak_int.astype(np.float64)
        bw = max(params.bandwidth_ppm * params.bandwidth_scale * 1e-6, 1e-12)

        # Evaluate KDE on a fine grid in log-m/z space. Extend the grid past the data
        # range by 5 * bandwidth so boundary peaks aren't underestimated by the kernel
        # spilling over the edge.
        grid_log, grid_was_capped = _adaptive_kde_grid(
            log_mz,
            bw,
            minimum_points=int(params.n_grid_points),
        )
        density = _kde_eval_1d(log_mz, weights, grid_log, bw)

        # ---- Find ALL local maxima first (rejection-budget recording), then filter
        # ---- by the prominence-quantile threshold. Tracking the unfiltered set lets
        # ---- the wizard's threshold explorer answer "how many peaks would survive
        # ---- if I raised min_prominence_quantile to X?" without re-running KDE.
        all_max_idx = _local_maxima(density, threshold=-np.inf)
        all_max_density = density[all_max_idx] if all_max_idx.size else np.empty(0)
        positive_density = density[density > 0]
        prominence_thr = float(
            np.quantile(
                positive_density if positive_density.size else density,
                params.min_prominence_quantile,
            )
        )
        prominence_keep = all_max_density > prominence_thr
        candidate_idx = all_max_idx[prominence_keep] if all_max_idx.size else all_max_idx
        n_rejected_by_prominence = int((~prominence_keep).sum()) if all_max_idx.size else 0
        if candidate_idx.size == 0:
            # Fallback: take the global argmax so we never produce empty output silently.
            candidate_idx = np.array([int(density.argmax())])

        candidate_mz = np.exp(grid_log[candidate_idx])

        # ---- Define tolerance windows in ppm and assign peaks to consensus bins ----
        tol_curve: ToleranceCurve | None = ds.extra.get("tolerance_curve")
        if tol_curve is not None:
            consensus_tol_ppm = tol_curve.evaluate(candidate_mz)
        else:
            consensus_tol_ppm = np.full(candidate_mz.size, params.default_tol_ppm)

        # Intensity matrix: (n_pixels, n_consensus). Take MAX intensity per (pixel, c).
        n_consensus = candidate_mz.size
        n_pixels = ds.n_pixels
        matrix = np.zeros((n_pixels, n_consensus), dtype=np.float32)
        # Track the total number of (peak, pixel) assignments per channel — i.e.
        # how many pre-aggregation peaks landed in each consensus channel across
        # the image. Distinct from ``prevalence`` (which counts distinct pixels)
        # and from ``matrix > 0`` (which max-aggregates to one cell per pixel).
        # The optional experimental prevalence sensitivity filter uses this as
        # its occupancy-model ball count. That model is intentionally disabled
        # by default and reports an explicit calibration warning.
        n_peaks_per_channel = np.zeros(n_consensus, dtype=np.int64)

        # Assign each input peak to at most one channel.  The previous left+right
        # scatter duplicated a peak into both channels whenever tolerance windows
        # overlapped, inflating prevalence and creating correlated duplicates.
        peak_idx, consensus_idx = _nearest_window_assignments(
            peak_mz,
            candidate_mz,
            consensus_tol_ppm,
        )
        np.maximum.at(
            matrix,
            (peak_pixel[peak_idx], consensus_idx),
            peak_int[peak_idx],
        )
        np.add.at(n_peaks_per_channel, consensus_idx, 1)

        # ---- Prevalence filter ----
        prevalence = (matrix > 0).sum(axis=0) / n_pixels
        keep = prevalence >= params.min_prevalence
        n_rejected_by_prevalence = int((~keep).sum())
        # Record the per-candidate prevalence BEFORE the filter so the wizard's
        # threshold explorer can answer "how many peaks would survive if I raised
        # min_prevalence to X?" without re-running KDE.
        all_candidate_mz = candidate_mz.copy()
        all_candidate_prevalence = prevalence.copy()
        all_candidate_max_intensity = matrix.max(axis=0).astype(np.float64)
        if not keep.any():
            # Don't return an empty matrix silently — raise so the caller knows the
            # threshold is too strict for this data.
            raise RuntimeError(
                f"No consensus peaks survived min_prevalence={params.min_prevalence}; "
                f"max prevalence found was {prevalence.max():.4f}. "
                "Lower min_prevalence or check upstream peak picking."
            )
        kept_mz = candidate_mz[keep]
        kept_matrix = matrix[:, keep]
        kept_prev = prevalence[keep]
        kept_n_peaks = n_peaks_per_channel[keep]

        # Sort kept consensus by m/z so downstream code sees monotone order.
        order_kept = np.argsort(kept_mz)
        kept_mz = kept_mz[order_kept]
        kept_matrix = kept_matrix[:, order_kept]
        kept_prev = kept_prev[order_kept]
        kept_n_peaks = kept_n_peaks[order_kept]

        peakmatrix = PeakMatrix(matrix=kept_matrix.astype(np.float32, copy=False), mz_axis=kept_mz)
        new_ds = ds.with_backend(peakmatrix)
        # Attach prevalence vector for the HyperspectralBrowser to display.
        # `consensus_n_peaks_per_channel` is the total count of (peak, pixel)
        # assignments per channel, retained for diagnostics and sensitivity work.
        new_extra = {
            **ds.extra,
            "consensus_prevalence": kept_prev.astype(np.float64, copy=False),
            "consensus_n_peaks_per_channel": kept_n_peaks.astype(np.int64, copy=False),
        }
        new_ds = new_ds.__class__(
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

        diag = Diagnostic(
            name=self.name,
            summary={
                # Final survivors
                "n_consensus_peaks": float(kept_mz.size),
                # Rejection budget — each filter's contribution
                "n_local_maxima_total": float(all_max_idx.size),
                "n_rejected_by_prominence": float(n_rejected_by_prominence),
                "n_post_prominence": float(candidate_mz.size),
                "n_rejected_by_prevalence": float(n_rejected_by_prevalence),
                # Survivor stats
                "prevalence_min": float(kept_prev.min()),
                "prevalence_median": float(np.median(kept_prev)),
                "prevalence_max": float(kept_prev.max()),
                # Operating point
                "bandwidth_log_mz": float(bw),
                "kde_grid_points": float(grid_log.size),
                "kde_grid_step_ppm": float(
                    (grid_log[1] - grid_log[0]) * 1e6 if grid_log.size > 1 else 0.0
                ),
                "kde_grid_capped": float(grid_was_capped),
                "min_prominence_quantile": float(params.min_prominence_quantile),
                "min_prevalence_threshold": float(params.min_prevalence),
            },
            payload={
                # Survivor arrays.
                "consensus_mz": kept_mz,
                "consensus_prevalence": kept_prev,
                # Pre-filter arrays for the threshold explorer: every local maximum,
                # paired with its KDE density and (where assigned) prevalence + max
                # intensity. Combined with the operator's filter parameters, these
                # answer "how many peaks survive at threshold X" without re-running.
                "all_local_max_density": all_max_density.astype(np.float64, copy=False),
                "prominence_threshold_value": np.asarray([prominence_thr], dtype=np.float64),
                "all_candidate_mz": all_candidate_mz.astype(np.float64, copy=False),
                "all_candidate_prevalence": all_candidate_prevalence.astype(np.float64, copy=False),
                "all_candidate_max_intensity": all_candidate_max_intensity,
                # KDE curve (for the line plot).
                "kde_grid_log_mz": _downsample_for_diagnostics(grid_log),
                "kde_density": _downsample_for_diagnostics(density),
            },
            figure_hint="line:kde_density_with_consensus_marks",
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


def _kde_eval_1d(x: np.ndarray, w: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Approximate a weighted 1-D Gaussian KDE in ``O(n + g)``.

    Samples are linearly deposited into a uniform histogram and convolved with a
    Gaussian.  This replaces the former Python loop over every grid point; with an
    adaptive million-point high-resolution grid it is both faster and far more
    accurate than evaluating a 5 ppm bandwidth on 30–70 ppm grid spacing.
    """
    from scipy.ndimage import gaussian_filter1d

    if x.size == 0 or grid.size == 0:
        return np.zeros_like(grid)
    if grid.size == 1:
        return np.asarray([float(np.maximum(w, 0).sum())], dtype=np.float64)
    step = float(grid[1] - grid[0])
    if step <= 0 or not np.allclose(np.diff(grid), step, rtol=1e-7, atol=1e-15):
        raise ValueError("KDE grid must be uniformly increasing")
    weights = np.maximum(np.asarray(w, dtype=np.float64), 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return np.zeros_like(grid, dtype=np.float64)
    position = (np.asarray(x, dtype=np.float64) - float(grid[0])) / step
    left = np.floor(position).astype(np.int64)
    fraction = position - left
    hist = np.zeros(grid.size, dtype=np.float64)
    valid_left = (left >= 0) & (left < grid.size)
    np.add.at(hist, left[valid_left], weights[valid_left] * (1.0 - fraction[valid_left]))
    right = left + 1
    valid_right = (right >= 0) & (right < grid.size)
    np.add.at(hist, right[valid_right], weights[valid_right] * fraction[valid_right])
    smoothed = gaussian_filter1d(
        hist,
        sigma=max(float(bw / step), 0.5),
        mode="constant",
        truncate=5.0,
    )
    return smoothed / (total_weight * step)


def _adaptive_kde_grid(
    log_mz: np.ndarray,
    bw: float,
    *,
    minimum_points: int,
    points_per_bandwidth: float = 4.0,
    max_points: int = 2_000_000,
) -> tuple[np.ndarray, bool]:
    """Build a bandwidth-aware log-m/z grid and report whether it hit the cap."""
    pad = 5.0 * bw
    lo = float(np.min(log_mz)) - pad
    hi = float(np.max(log_mz)) + pad
    if hi <= lo:
        hi = lo + max(bw, 1e-6)
    target_step = max(bw / points_per_bandwidth, np.finfo(np.float64).eps)
    required = int(np.ceil((hi - lo) / target_step)) + 1
    requested = max(int(minimum_points), required, 3)
    n_points = min(requested, int(max_points))
    return np.linspace(lo, hi, num=n_points), requested > n_points


def _nearest_window_assignments(
    peak_mz: np.ndarray,
    candidate_mz: np.ndarray,
    tolerance_ppm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each peak to its nearest eligible candidate, never to two channels."""
    peaks = np.asarray(peak_mz, dtype=np.float64)
    candidates = np.asarray(candidate_mz, dtype=np.float64)
    tolerances = np.asarray(tolerance_ppm, dtype=np.float64)
    if candidates.size == 0 or peaks.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(candidates)
    sorted_mz = candidates[order]
    sorted_tol = tolerances[order]
    right = np.searchsorted(sorted_mz, peaks)
    left = right - 1

    def _distance(which: np.ndarray) -> np.ndarray:
        clipped = np.clip(which, 0, sorted_mz.size - 1)
        center = sorted_mz[clipped]
        tol = sorted_tol[clipped]
        eligible = (which >= 0) & (which < sorted_mz.size)
        eligible &= np.abs(peaks - center) <= center * tol * 1e-6
        distance = np.abs(np.log(peaks) - np.log(center))
        return np.where(eligible, distance, np.inf)

    left_distance = _distance(left)
    right_distance = _distance(right)
    choose_right = right_distance < left_distance
    chosen = np.where(choose_right, right, left)
    valid = np.isfinite(np.minimum(left_distance, right_distance))
    peak_idx = np.flatnonzero(valid).astype(np.int64, copy=False)
    return peak_idx, order[chosen[valid]].astype(np.int64, copy=False)


def _downsample_for_diagnostics(values: np.ndarray, max_points: int = 50_000) -> np.ndarray:
    """Bound GUI/spec diagnostic payload size without changing peak detection."""
    stride = max(1, int(np.ceil(values.size / max_points)))
    return values[::stride]


def _local_maxima(y: np.ndarray, *, threshold: float) -> np.ndarray:
    """Indices where y[i] > y[i-1] and y[i] > y[i+1] and y[i] > threshold."""
    if y.size < 3:
        return np.empty(0, dtype=np.int64)
    is_max = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    above = y[1:-1] > threshold
    idx = np.flatnonzero(is_max & above) + 1
    return idx
