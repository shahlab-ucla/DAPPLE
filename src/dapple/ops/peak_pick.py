"""Peak-picking operators.

`snr_peak_pick` is the default for centroided data. It filters each pixel's existing
peak list down to those above a per-pixel noise floor estimated as MAD-scaled, with
an additional optional empirical-quantile threshold. The empirical-quantile floor is
preferred over a parametric ``k·σ`` cutoff because mass-error and intensity
distributions in real MSI data are heavy-tailed (TOF detector dead-time, Orbitrap
AGC overshoot, etc.).

`cwt_peak_pick` is the natural choice for profile-mode data: it runs a Ricker /
Mexican-hat continuous-wavelet-transform peak finder on each pixel's intensity
trace, scoring candidates by SNR computed across a noise quantile of CWT
coefficients. It does the *picking*, turning an unbinned (m/z, intensity) trace
into a centroided peak list — meaningfully different from the SNR filter's
"trim a peak list that's already been picked" job.
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
class SnrPeakPickParams(OpParams):
    """Parameters for SNR peak picking on centroided data."""

    snr_mad: float = field(
        default=3.0,
        metadata={
            "label": "SNR threshold (× MAD)",
            "help": (
                "Keep peaks whose intensity exceeds snr_mad × (1.4826 × per-pixel MAD). "
                "Default 3; the 1.4826 factor turns MAD into a robust σ estimate "
                "without assuming Gaussianity. Raise to 5–10 for a stricter cut; "
                "drop to 1–2 to keep more low-intensity candidates at the cost of "
                "admitting more noise."
            ),
        },
    )
    min_intensity_quantile: float = field(
        default=0.0,
        metadata={
            "label": "Per-pixel intensity floor (quantile)",
            "help": (
                "Optional secondary filter: also drop peaks below this quantile of "
                "the pixel's own intensity distribution. Default 0 (disabled). "
                "0.5 keeps the top half of every pixel's peaks; 0.95 keeps the top 5%. "
                "Useful when the upstream picker is loose and you want to bias toward "
                "the strongest peaks per pixel."
            ),
        },
    )
    min_intensity_abs: float = field(
        default=0.0,
        metadata={
            "label": "Absolute intensity floor",
            "help": (
                "Hard floor on the intensity value (in the dataset's current units, "
                "post any prior normalization). Default 0 (disabled). Set when you "
                "know the noise floor — e.g., median dark-pixel intensity × 3."
            ),
        },
    )


@register
class SnrPeakPick(Operator):
    name = "snr_peak_pick"
    params_cls = SnrPeakPickParams

    def default_params(self, ep: ExperimentParams) -> SnrPeakPickParams:
        # Centroided data tends to be more aggressively filtered upstream — be gentle.
        # Profile data hasn't been picked yet, so a higher SNR threshold is appropriate.
        if ep.profile_or_centroided == "profile":
            return SnrPeakPickParams(snr_mad=3.0, min_intensity_quantile=0.0)
        return SnrPeakPickParams(snr_mad=3.0, min_intensity_quantile=0.0)

    def validate(self, ep: ExperimentParams) -> list[str]:
        if ep.profile_or_centroided == "profile":
            return [
                "snr_peak_pick on profile data treats every sample point as a candidate "
                "peak, which is rarely what you want; use cwt_peak_pick instead."
            ]
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: SnrPeakPickParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "snr_peak_pick currently requires a PeakList backend."
            )
        pl = ds.backend
        offsets_in = np.asarray(pl.offsets[:])
        mz_in = np.asarray(pl.mz[:])
        int_in = np.asarray(pl.intensity[:])
        n_pixels = pl.n_pixels

        # First pass: build per-pixel keep mask.
        keep_mask = np.zeros_like(int_in, dtype=bool)
        thresholds = np.empty(n_pixels, dtype=np.float64)
        kept_counts = np.empty(n_pixels, dtype=np.int64)
        for i in range(n_pixels):
            a, b = int(offsets_in[i]), int(offsets_in[i + 1])
            if a == b:
                thresholds[i] = 0.0
                kept_counts[i] = 0
                continue
            seg = int_in[a:b].astype(np.float64)
            mad = float(np.median(np.abs(seg - np.median(seg))))
            noise = 1.4826 * mad if mad > 0 else 0.0
            thr = params.snr_mad * noise
            if params.min_intensity_quantile > 0:
                thr = max(thr, float(np.quantile(seg, params.min_intensity_quantile)))
            thr = max(thr, params.min_intensity_abs)
            thresholds[i] = thr
            mask = seg > thr
            keep_mask[a:b] = mask
            kept_counts[i] = int(mask.sum())

        # Second pass: build output CSR arrays from kept peaks.
        new_offsets = np.empty(n_pixels + 1, dtype=np.int64)
        new_offsets[0] = 0
        np.cumsum(kept_counts, out=new_offsets[1:])
        total_out = int(new_offsets[-1])

        new_mz = np.empty(total_out, dtype=np.float64)
        new_int = np.empty(total_out, dtype=np.float32)
        for i in range(n_pixels):
            a, b = int(offsets_in[i]), int(offsets_in[i + 1])
            if a == b:
                continue
            seg_mask = keep_mask[a:b]
            ao, bo = int(new_offsets[i]), int(new_offsets[i + 1])
            if ao == bo:
                continue
            new_mz[ao:bo] = mz_in[a:b][seg_mask]
            new_int[ao:bo] = int_in[a:b][seg_mask]

        new_backend = PeakList(
            mz=new_mz, intensity=new_int, offsets=new_offsets, n_pixels=n_pixels
        )
        new_ds = ds.with_backend(new_backend)

        # Diagnostics: kept-peak counts, threshold distribution, total peak fraction kept.
        n_in_total = int(int_in.size)
        n_out_total = int(total_out)
        diag = Diagnostic(
            name=self.name,
            summary={
                "n_peaks_in": float(n_in_total),
                "n_peaks_out": float(n_out_total),
                "fraction_kept": float(n_out_total / max(n_in_total, 1)),
                "kept_per_pixel_min": float(kept_counts.min()),
                "kept_per_pixel_median": float(np.median(kept_counts)),
                "kept_per_pixel_max": float(kept_counts.max()),
                "threshold_median": float(np.median(thresholds)),
                "snr_mad": float(params.snr_mad),
            },
            payload={
                "per_pixel_threshold": thresholds,
                "per_pixel_kept_count": kept_counts,
            },
            figure_hint="histogram:per_pixel_kept_count",
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


@dataclass(frozen=True)
class CwtPeakPickParams(OpParams):
    """Parameters for CWT peak picking on profile data.

    The picker resamples each pixel's profile spectrum onto a uniform log-m/z grid
    (so a constant ppm scale is uniform on the resampled axis), runs scipy's
    Ricker-wavelet CWT peak finder over a range of widths, and emits a centroid
    + height for each detected peak.
    """

    width_min_ppm: float = field(
        default=20.0,
        metadata={
            "label": "Minimum peak width (ppm)",
            "help": (
                "Lower bound on the wavelet width in ppm. Should be smaller than "
                "the narrowest peak you expect — typically 10–30 ppm on Q-TOF / "
                "reflectron TOF, 1–3 ppm on Orbitrap. CWT searches from this width "
                "upward; smaller adds detection cost but catches narrow peaks."
            ),
        },
    )
    width_max_ppm: float = field(
        default=120.0,
        metadata={
            "label": "Maximum peak width (ppm)",
            "help": (
                "Upper bound on the wavelet width in ppm. Should be at least as "
                "large as the broadest real peak you expect. Default 120 ppm fits "
                "Q-TOF / reflectron TOF; raise to 500 for axial linear MALDI-TOF."
            ),
        },
    )
    n_widths: int = field(
        default=8,
        metadata={
            "label": "Number of wavelet widths",
            "help": (
                "How many distinct wavelet widths to evaluate between the min and "
                "max bounds (log-spaced). Default 8 is enough for typical peak "
                "shapes; raise to 16 if you see narrow peaks being missed."
            ),
        },
    )
    grid_ppm_step: float = field(
        default=5.0,
        metadata={
            "label": "Resampling grid step (ppm)",
            "help": (
                "Resample each pixel's profile spectrum onto a regular log-m/z "
                "grid with this ppm spacing before running the CWT. Default 5 ppm. "
                "Tighter grid = sharper centroids but more compute. Should be "
                "smaller than width_min_ppm."
            ),
        },
    )
    max_grid_points: int = field(
        default=1_000_000,
        metadata={
            "label": "Maximum resampling grid points",
            "help": (
                "Safety cap for the per-pixel log-m/z grid. Very broad Orbitrap "
                "ranges at a 0.5 ppm step otherwise create several million samples "
                "per pixel. When capped, DAPPLE reports the effective coarser step."
            ),
        },
    )
    max_native_oversampling: float = field(
        default=8.0,
        metadata={
            "label": "Maximum native-grid oversampling",
            "help": (
                "Do not interpolate a profile spectrum more finely than this "
                "multiple of its observed m/z sampling density. Interpolation "
                "cannot add mass resolution; the default 8× preserves smooth "
                "wavelet localization while avoiding needlessly huge grids."
            ),
        },
    )
    min_snr: float = field(
        default=3.0,
        metadata={
            "label": "Minimum SNR",
            "help": (
                "scipy.signal.find_peaks_cwt min_snr. Peaks whose CWT coefficient "
                "is below noise_perc·noise are dropped. Default 3 — the typical "
                "scipy default. Raise to 5 for stricter picking on noisy traces."
            ),
        },
    )
    noise_perc: float = field(
        default=10.0,
        metadata={
            "label": "Noise percentile",
            "help": (
                "Percentile (0–100) of CWT coefficients used as the noise floor. "
                "Default 10 (i.e. the 10th percentile of CWT coefficients is "
                "treated as the noise level). Raise to 25 if real signals are "
                "themselves dense (so 10th percentile is contaminated)."
            ),
        },
    )

    def __post_init__(self) -> None:
        if self.width_min_ppm <= 0 or self.width_max_ppm < self.width_min_ppm:
            raise ValueError("CWT widths must be positive and max >= min")
        if self.n_widths < 1:
            raise ValueError("n_widths must be at least 1")
        if self.grid_ppm_step <= 0:
            raise ValueError("grid_ppm_step must be positive")
        if self.max_grid_points < 3:
            raise ValueError("max_grid_points must be at least 3")
        if self.max_native_oversampling < 1:
            raise ValueError("max_native_oversampling must be at least 1")
        if self.min_snr <= 0:
            raise ValueError("min_snr must be positive")
        if not 0 <= self.noise_perc <= 100:
            raise ValueError("noise_perc must be in [0, 100]")


@register
class CwtPeakPick(Operator):
    """Continuous-wavelet-transform peak picker for profile-mode spectra.

    Each pixel's (m/z, intensity) trace is resampled onto a uniform log-m/z grid
    (so a constant ppm width is uniform on the grid), then ``scipy.signal.find_peaks_cwt``
    runs over a range of Ricker-wavelet widths matching the user's expected peak
    range. Detected peak indices are mapped back to m/z via centroid-of-mass over
    a small neighborhood (so we get sub-grid m/z accuracy).
    """

    name = "cwt_peak_pick"
    params_cls = CwtPeakPickParams

    def default_params(self, ep: ExperimentParams) -> CwtPeakPickParams:
        if ep.instrument_family in {"orbitrap", "fticr"}:
            return CwtPeakPickParams(width_min_ppm=2.0, width_max_ppm=20.0, grid_ppm_step=0.5)
        if ep.instrument_family == "tof_axial":
            return CwtPeakPickParams(width_min_ppm=80.0, width_max_ppm=500.0, grid_ppm_step=20.0)
        return CwtPeakPickParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        warnings: list[str] = []
        if ep.profile_or_centroided == "centroided":
            warnings.append(
                "cwt_peak_pick is intended for profile (continuous) data; on "
                "already-centroided peak lists it adds compute without benefit. "
                "Use snr_peak_pick instead."
            )
        return warnings

    def apply(
        self,
        ds: MSIDataset,
        params: CwtPeakPickParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        from scipy.signal import find_peaks_cwt

        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "cwt_peak_pick currently requires a PeakList backend."
            )
        pl = ds.backend
        offsets_in = np.asarray(pl.offsets[:])
        mz_in = np.asarray(pl.mz[:])
        int_in = np.asarray(pl.intensity[:])
        n_pixels = pl.n_pixels

        # Build a uniform log-m/z grid spanning every pixel's m/z range.
        # log_step = ln(1 + grid_ppm_step * 1e-6); peaks near that scale resolve
        # cleanly on the grid.
        if mz_in.size == 0 or n_pixels == 0:
            raise RuntimeError("cwt_peak_pick: dataset has no peaks to pick.")
        mz_min = float(mz_in.min())
        mz_max = float(mz_in.max())
        if mz_min <= 0 or mz_max <= mz_min:
            raise RuntimeError(
                f"cwt_peak_pick: nonsensical m/z bounds ({mz_min}, {mz_max})."
            )
        requested_log_step = float(np.log1p(params.grid_ppm_step * 1e-6))
        log_span = float(np.log(mz_max) - np.log(mz_min))
        requested_points = int(np.ceil(log_span / requested_log_step)) + 1

        # Most profile-mode formats store every pixel on one calibrated native
        # grid. Running directly on that grid is both faster and more faithful:
        # interpolation cannot create resolution and may manufacture thousands of
        # highly correlated samples. Accept common linear- or log-spaced grids.
        first_a, first_b = int(offsets_in[0]), int(offsets_in[1])
        first_grid = mz_in[first_a:first_b]
        shared_native_grid = first_grid.size >= 3 and first_grid.size <= params.max_grid_points
        if shared_native_grid:
            for pixel in range(1, n_pixels):
                a, b = int(offsets_in[pixel]), int(offsets_in[pixel + 1])
                segment = mz_in[a:b]
                if segment.shape != first_grid.shape or not np.allclose(
                    segment, first_grid, rtol=1e-10, atol=1e-12
                ):
                    shared_native_grid = False
                    break
        linear_deltas = np.diff(first_grid) if first_grid.size >= 2 else np.empty(0)
        log_deltas = (
            np.diff(np.log(first_grid))
            if first_grid.size >= 2 and np.all(first_grid > 0)
            else np.empty(0)
        )

        def _is_regular(deltas: np.ndarray) -> bool:
            if deltas.size == 0 or np.any(deltas <= 0):
                return False
            center = float(np.median(deltas))
            return bool(np.max(np.abs(deltas - center)) <= max(abs(center) * 1e-5, 1e-15))

        native_spacing = (
            "linear"
            if shared_native_grid and _is_regular(linear_deltas)
            else "log"
            if shared_native_grid and _is_regular(log_deltas)
            else None
        )
        use_native_grid = native_spacing is not None

        # Interpolating far below the native profile spacing adds no information
        # and made broad-range datasets spend most of their time on synthetic
        # samples. Estimate a robust within-spectrum spacing from a small,
        # deterministic pixel sample and cap oversampling accordingly.
        native_steps: list[float] = []
        sample_pixels = np.linspace(
            0, max(n_pixels - 1, 0), num=min(n_pixels, 32), dtype=np.int64
        )
        for pixel in np.unique(sample_pixels):
            a, b = int(offsets_in[pixel]), int(offsets_in[pixel + 1])
            segment = mz_in[a:b]
            valid = segment[np.isfinite(segment) & (segment > 0)]
            if valid.size < 2:
                continue
            deltas = np.diff(np.log(valid))
            deltas = deltas[deltas > 0]
            if deltas.size:
                native_steps.append(float(np.median(deltas)))
        native_log_step = float(np.median(native_steps)) if native_steps else 0.0
        resolution_limited_step = (
            native_log_step / float(params.max_native_oversampling)
            if native_log_step > 0
            else requested_log_step
        )
        effective_requested_step = max(requested_log_step, resolution_limited_step)
        effective_requested_points = int(np.ceil(log_span / effective_requested_step)) + 1
        grid_capped = effective_requested_points > int(params.max_grid_points)
        if use_native_grid:
            mz_grid = first_grid
            log_step = native_log_step
            grid_capped = False
        elif grid_capped:
            log_grid = np.linspace(
                np.log(mz_min),
                np.log(mz_max),
                num=int(params.max_grid_points),
            )
            log_step = float(log_grid[1] - log_grid[0])
        else:
            log_step = effective_requested_step
            log_grid = np.arange(np.log(mz_min), np.log(mz_max) + log_step, log_step)
        if not use_native_grid:
            mz_grid = np.exp(log_grid)
        # Convert ppm widths to grid samples.
        log_min_w = np.log1p(params.width_min_ppm * 1e-6)
        log_max_w = np.log1p(params.width_max_ppm * 1e-6)
        if native_spacing == "linear":
            linear_step = float(np.median(linear_deltas))
            width_min_samples = params.width_min_ppm * 1e-6 * mz_min / linear_step
            width_max_samples = params.width_max_ppm * 1e-6 * mz_max / linear_step
            widths = np.linspace(
                width_min_samples, width_max_samples, num=int(params.n_widths)
            )
        else:
            widths = np.linspace(
                log_min_w / log_step, log_max_w / log_step, num=int(params.n_widths)
            )
        widths = np.clip(widths, 1.0, None)

        kept_per_pixel: list[int] = []
        out_mz: list[np.ndarray] = []
        out_int: list[np.ndarray] = []
        new_offsets = [0]
        n_failed_pixels = 0

        for i in range(n_pixels):
            a, b = int(offsets_in[i]), int(offsets_in[i + 1])
            if a == b:
                kept_per_pixel.append(0)
                new_offsets.append(new_offsets[-1])
                continue
            seg_mz = mz_in[a:b]
            seg_int = int_in[a:b]
            resampled = (
                seg_int
                if use_native_grid
                else np.interp(mz_grid, seg_mz, seg_int, left=0.0, right=0.0)
            )

            try:
                peak_idx = find_peaks_cwt(
                    resampled,
                    widths=widths,
                    min_snr=float(params.min_snr),
                    noise_perc=float(params.noise_perc),
                )
            except (ValueError, FloatingPointError):
                n_failed_pixels += 1
                peak_idx = np.empty(0, dtype=np.int64)

            peak_idx = np.asarray(peak_idx, dtype=np.int64)
            if peak_idx.size == 0:
                kept_per_pixel.append(0)
                new_offsets.append(new_offsets[-1])
                continue

            # Sub-grid centroid: weighted mean of m/z within ±width samples.
            half_window = int(max(1, np.round(widths.mean())))
            centroid_mz = np.empty(peak_idx.size, dtype=np.float64)
            centroid_int = np.empty(peak_idx.size, dtype=np.float32)
            for k, idx in enumerate(peak_idx):
                lo = max(0, int(idx) - half_window)
                hi = min(resampled.size, int(idx) + half_window + 1)
                window_mz = mz_grid[lo:hi]
                window_int = resampled[lo:hi]
                wsum = float(window_int.sum())
                if wsum > 0:
                    centroid_mz[k] = float((window_mz * window_int).sum() / wsum)
                else:
                    centroid_mz[k] = float(mz_grid[idx])
                centroid_int[k] = float(window_int.max())

            order = np.argsort(centroid_mz)
            centroid_mz = centroid_mz[order]
            centroid_int = centroid_int[order]
            kept_per_pixel.append(centroid_mz.size)
            out_mz.append(centroid_mz)
            out_int.append(centroid_int)
            new_offsets.append(new_offsets[-1] + centroid_mz.size)

        if not out_mz:
            raise RuntimeError(
                "cwt_peak_pick: no peaks detected in any pixel. Lower min_snr, "
                "raise noise_perc, or widen the [width_min_ppm, width_max_ppm] range."
            )
        new_pl = PeakList(
            mz=np.concatenate(out_mz).astype(np.float64, copy=False),
            intensity=np.concatenate(out_int).astype(np.float32, copy=False),
            offsets=np.asarray(new_offsets, dtype=np.int64),
            n_pixels=n_pixels,
        )
        new_ds = ds.with_backend(new_pl)
        kept = np.asarray(kept_per_pixel, dtype=np.int64)
        diag = Diagnostic(
            name=self.name,
            summary={
                "n_pixels": float(n_pixels),
                "n_peaks_total": float(int(kept.sum())),
                "kept_per_pixel_min": float(int(kept.min())),
                "kept_per_pixel_median": float(np.median(kept)),
                "kept_per_pixel_max": float(int(kept.max())),
                "grid_size": float(mz_grid.size),
                "grid_requested_size": float(requested_points),
                "grid_capped": float(grid_capped),
                "grid_limited_by_native_resolution": float(
                    effective_requested_step > requested_log_step * (1.0 + 1e-12)
                ),
                "native_grid_used": float(use_native_grid),
                "native_grid_ppm_step": float(np.expm1(native_log_step) * 1e6),
                "effective_grid_ppm_step": float(np.expm1(log_step) * 1e6),
                "n_failed_pixels": float(n_failed_pixels),
                "n_widths": float(widths.size),
                "min_snr": float(params.min_snr),
            },
            payload={"per_pixel_kept_count": kept},
            figure_hint="histogram:per_pixel_kept_count",
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
