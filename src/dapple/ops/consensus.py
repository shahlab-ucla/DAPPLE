"""KDE-based consensus alignment of per-pixel peaks into a shared m/z axis.

Algorithm: pool all picked peaks across pixels, fit a 1-D Gaussian KDE on log(m/z)
with a ppm-scaled bandwidth, call local maxima of the KDE as consensus peaks, then
for each consensus m/z extract per-pixel intensity (the most intense peak in the
tolerance window). A simple prevalence floor drops peaks observed in fewer than
``min_prevalence`` of pixels; a permutation-FDR alternative is planned.

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

    `n_grid_points` — m/z grid resolution for KDE evaluation. Default 32768 → ~21 ppm
        spacing across a 200–800 m/z range, fine enough not to bias peak detection.
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
        pixels (a simple, conservative prevalence filter; a permutation-FDR
        alternative is planned).
    """

    n_grid_points: int = field(
        default=32768,
        metadata={
            "label": "KDE grid resolution",
            "help": (
                "Number of evaluation points for the m/z density. Default 32768 gives "
                "~21 ppm spacing across a 200–800 m/z range — fine enough not to bias "
                "peak detection. More = sharper peak detection but slower; less = "
                "faster but risks merging close peaks."
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
                "rare-but-real signals. A permutation-FDR alternative is planned."
            ),
        },
    )


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
        pad = 5.0 * bw
        log_lo = float(log_mz.min()) - pad
        log_hi = float(log_mz.max()) + pad
        if log_hi <= log_lo:
            log_hi = log_lo + 1e-6
        grid_log = np.linspace(log_lo, log_hi, num=int(params.n_grid_points))
        density = _kde_eval_1d(log_mz, weights, grid_log, bw)

        # ---- Find ALL local maxima first (rejection-budget recording), then filter
        # ---- by the prominence-quantile threshold. Tracking the unfiltered set lets
        # ---- the wizard's threshold explorer answer "how many peaks would survive
        # ---- if I raised min_prominence_quantile to X?" without re-running KDE.
        all_max_idx = _local_maxima(density, threshold=-np.inf)
        all_max_density = density[all_max_idx] if all_max_idx.size else np.empty(0)
        prominence_thr = float(np.quantile(density, params.min_prominence_quantile))
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

        # Vectorized assignment: a peak with m/z m belongs to consensus i iff
        #   m in [c_i * (1 - tol_i*1e-6), c_i * (1 + tol_i*1e-6)]
        # We do this via searchsorted on sorted candidate_mz; ranges may overlap.
        order = np.argsort(candidate_mz)
        candidate_mz_sorted = candidate_mz[order]
        consensus_tol_sorted = consensus_tol_ppm[order]
        lo = candidate_mz_sorted * (1 - consensus_tol_sorted * 1e-6)
        hi = candidate_mz_sorted * (1 + consensus_tol_sorted * 1e-6)

        # For each peak, find candidate(s) whose window contains it. Use searchsorted.
        # left = smallest i with candidate_mz_sorted[i] - tol >= peak_mz means peak is
        # below i's range ⇒ i is candidate iff peak in [lo[i-1], hi[i-1]]. Easier:
        # find i = searchsorted(candidate_mz_sorted, peak_mz). The peak's nearest
        # consensus is candidate_mz_sorted[i-1] or [i]; check both.
        idx_right = np.searchsorted(candidate_mz_sorted, peak_mz)
        idx_left = idx_right - 1

        # Intensity matrix: (n_pixels, n_consensus). Take MAX intensity per (pixel, c).
        n_consensus = candidate_mz.size
        n_pixels = ds.n_pixels
        matrix = np.zeros((n_pixels, n_consensus), dtype=np.float32)
        # Track the total number of (peak, pixel) assignments per channel — i.e.
        # how many pre-aggregation peaks landed in each consensus channel across
        # the image. Distinct from ``prevalence`` (which counts distinct pixels)
        # and from ``matrix > 0`` (which max-aggregates to one cell per pixel).
        # The PrevalenceFdrFilter operator uses this as the null-model's ball
        # count: "if these k_c peaks were placed uniformly at random across
        # n_pixels bins, would we expect to see this prevalence?"
        n_peaks_per_channel = np.zeros(n_consensus, dtype=np.int64)

        # Build flat (pixel, consensus_idx, intensity) triples for both neighbor
        # candidates, then keep the max per (pixel, consensus).
        for which in (idx_left, idx_right):
            valid = (which >= 0) & (which < n_consensus)
            if not valid.any():
                continue
            sel = which[valid]
            sel_peak_mz = peak_mz[valid]
            sel_peak_pix = peak_pixel[valid]
            sel_peak_int = peak_int[valid]
            in_window = (sel_peak_mz >= lo[sel]) & (sel_peak_mz <= hi[sel])
            if not in_window.any():
                continue
            sel = sel[in_window]
            sel_peak_pix = sel_peak_pix[in_window]
            sel_peak_int = sel_peak_int[in_window]
            sorted_consensus = order[sel]  # back to original consensus indexing
            # Maximum-aggregate. Use scatter via np.maximum.at for correctness.
            np.maximum.at(matrix, (sel_peak_pix, sorted_consensus), sel_peak_int)
            np.add.at(n_peaks_per_channel, sorted_consensus, 1)

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
        # assignments per channel — needed by PrevalenceFdrFilter's null model.
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
                "kde_grid_log_mz": grid_log,
                "kde_density": density,
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
    """Weighted Gaussian KDE evaluated on a 1-D grid. O(n + g log n) via sort & sweep
    over the truncated kernel support (±5σ).

    We avoid scipy.stats.gaussian_kde because it doesn't support weighted samples in
    older scipy and is also O(n*g) which gets painful at n ~ 200k, g ~ 8192.
    """
    if x.size == 0:
        return np.zeros_like(grid)
    inv_bw = 1.0 / bw
    out = np.zeros_like(grid, dtype=np.float64)
    # Sort samples once.
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    # For each sample, Gaussian contribution is non-negligible within ±cutoff*bw.
    cutoff = 5.0
    # For each grid point, find sample range [lo, hi) within cutoff*bw of grid value.
    los = np.searchsorted(xs, grid - cutoff * bw, side="left")
    his = np.searchsorted(xs, grid + cutoff * bw, side="right")
    # Vectorized inner loop is hard with variable-length slices; use a Python loop.
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bw * w.sum())
    for gi in range(grid.size):
        lo = int(los[gi])
        hi = int(his[gi])
        if hi <= lo:
            continue
        diffs = (grid[gi] - xs[lo:hi]) * inv_bw
        out[gi] = (ws[lo:hi] * np.exp(-0.5 * diffs * diffs)).sum()
    return out * norm


def _local_maxima(y: np.ndarray, *, threshold: float) -> np.ndarray:
    """Indices where y[i] > y[i-1] and y[i] > y[i+1] and y[i] > threshold."""
    if y.size < 3:
        return np.empty(0, dtype=np.int64)
    is_max = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    above = y[1:-1] > threshold
    idx = np.flatnonzero(is_max & above) + 1
    return idx
