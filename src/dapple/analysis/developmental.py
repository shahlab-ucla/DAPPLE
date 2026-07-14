"""ROI enrichment and directed anatomical-axis pattern analysis.

These routines are intentionally read-only and require a post-consensus
``PeakMatrix``.  ROI Welch tests use populated pixels as their units; this is useful
for fast within-image screening but is not biological-replicate inference.  The
result metadata and warnings make that distinction explicit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata, t as student_t

from dapple.analysis.models import AxisProfileResult, RoiEnrichmentResult
from dapple.analysis.spatial import (
    AxisProjection,
    DirectedAxis,
    RoiMasks,
    coordinate_fingerprint,
    project_to_axis,
)
from dapple.data.dataset import MSIDataset, PeakMatrix


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values, preserving NaN positions.

    Only finite p-values participate in the family. This lets callers mark an
    untestable zero-variance channel as NaN without changing valid channels into
    failures or silently treating the invalid test as significant.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got {p.shape}")
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("finite p-values must lie in [0, 1]")
    out = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    q_sorted = ranked * valid.size / np.arange(1, valid.size + 1, dtype=np.float64)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    restored = np.empty_like(q_sorted)
    restored[order] = q_sorted
    out[valid] = restored
    return out


def analyze_roi_enrichment(
    ds: MSIDataset,
    roi_masks: RoiMasks,
    numerator: str | Sequence[str],
    denominator: str | Sequence[str],
    *,
    contrast_name: str | None = None,
    min_units_per_group: int = 3,
    trim_fraction: float = 0.1,
    pseudocount_quantile: float = 0.05,
    chunk_channels: int = 256,
) -> RoiEnrichmentResult:
    """Compare two named ROI unions channel-by-channel.

    Descriptive summaries are computed on untransformed intensities. The robust
    fold change uses a trimmed mean and a channel-adaptive pseudocount equal to half
    the requested positive-intensity quantile. Welch's statistic is computed on
    log2-transformed values in channel chunks.

    The test's units are pixels, so p-values describe within-image separation and
    must not be interpreted as population-level evidence. Biological-replicate
    workflows should first aggregate each ROI within each independent sample.
    """
    pm = _require_peak_matrix(ds)
    _validate_common_options(
        chunk_channels=chunk_channels,
        trim_fraction=trim_fraction,
        pseudocount_quantile=pseudocount_quantile,
    )
    if min_units_per_group < 2:
        raise ValueError("min_units_per_group must be at least 2 for Welch inference")
    if roi_masks.n_pixels != ds.n_pixels:
        raise ValueError(
            f"ROI masks cover {roi_masks.n_pixels} pixels but dataset has {ds.n_pixels}"
        )
    if roi_masks.coordinate_fingerprint != coordinate_fingerprint(ds):
        raise ValueError(
            "ROI masks were created for different grid coordinates or pixel row order"
        )

    numerator_names = _as_names(numerator)
    denominator_names = _as_names(denominator)
    num_mask = roi_masks.union(numerator_names)
    den_mask = roi_masks.union(denominator_names)
    contrast_overlap = num_mask & den_mask
    if contrast_overlap.any():
        raise ValueError(
            f"ROI contrast arms overlap at {int(contrast_overlap.sum())} populated "
            "pixel(s); use an exclusive rasterization policy"
        )
    num_idx = np.flatnonzero(num_mask)
    den_idx = np.flatnonzero(den_mask)
    n_num, n_den = int(num_idx.size), int(den_idx.size)
    mz = np.asarray(pm.mz_axis[:], dtype=np.float64)
    k = mz.size

    arrays = {
        name: np.full(k, np.nan, dtype=np.float64)
        for name in (
            "mean_num",
            "mean_den",
            "median_num",
            "median_den",
            "trim_num",
            "trim_den",
            "prev_num",
            "prev_den",
            "pc",
            "lfc",
            "mean_log_diff",
            "welch_t",
            "welch_df",
            "p",
        )
    }
    test_valid = np.zeros(k, dtype=bool)
    enough_units = n_num >= min_units_per_group and n_den >= min_units_per_group

    for start in range(0, k, chunk_channels):
        stop = min(start + chunk_channels, k)
        a = _read_rows_chunk(pm.matrix, num_idx, start, stop)
        b = _read_rows_chunk(pm.matrix, den_idx, start, stop)
        _validate_intensity_chunk(a, b)
        width = stop - start

        arrays["mean_num"][start:stop] = _mean_or_nan(a, width)
        arrays["mean_den"][start:stop] = _mean_or_nan(b, width)
        arrays["median_num"][start:stop] = _median_or_nan(a, width)
        arrays["median_den"][start:stop] = _median_or_nan(b, width)
        arrays["trim_num"][start:stop] = _trimmed_mean_or_nan(a, width, trim_fraction)
        arrays["trim_den"][start:stop] = _trimmed_mean_or_nan(b, width, trim_fraction)
        arrays["prev_num"][start:stop] = _prevalence_or_nan(a, width)
        arrays["prev_den"][start:stop] = _prevalence_or_nan(b, width)

        combined = np.concatenate([a, b], axis=0)
        pc = _adaptive_pseudocount(combined, pseudocount_quantile)
        arrays["pc"][start:stop] = pc
        with np.errstate(invalid="ignore", divide="ignore"):
            arrays["lfc"][start:stop] = np.log2(
                (arrays["trim_num"][start:stop] + pc)
                / (arrays["trim_den"][start:stop] + pc)
            )

        if a.shape[0] and b.shape[0]:
            log_a = np.log2(a + pc[None, :])
            log_b = np.log2(b + pc[None, :])
            arrays["mean_log_diff"][start:stop] = log_a.mean(axis=0) - log_b.mean(axis=0)
            if enough_units:
                t_stat, df, p_value, valid = _welch_from_transformed(log_a, log_b)
                arrays["welch_t"][start:stop] = t_stat
                arrays["welch_df"][start:stop] = df
                arrays["p"][start:stop] = p_value
                test_valid[start:stop] = valid

    q = benjamini_hochberg(arrays["p"])
    warnings: list[str] = [
        "Welch inference uses pixels as units and does not represent biological "
        "replicate-level inference; spatial autocorrelation can make p-values optimistic."
    ]
    if not enough_units:
        status = "insufficient_units"
        warnings.append(
            f"Inference skipped: numerator has {n_num} and denominator has {n_den} "
            f"pixels; each requires at least {min_units_per_group}."
        )
    elif not test_valid.all():
        status = "partial"
        warnings.append(
            f"Welch inference was undefined for {int((~test_valid).sum())} "
            "zero-variance channel(s); their p/q values are NaN."
        )
    else:
        status = "ok"

    default_name = f"{'+'.join(numerator_names)} vs {'+'.join(denominator_names)}"
    return RoiEnrichmentResult(
        contrast_name=contrast_name or default_name,
        numerator_names=numerator_names,
        denominator_names=denominator_names,
        mz=mz,
        n_numerator=n_num,
        n_denominator=n_den,
        mean_numerator=arrays["mean_num"],
        mean_denominator=arrays["mean_den"],
        median_numerator=arrays["median_num"],
        median_denominator=arrays["median_den"],
        trimmed_mean_numerator=arrays["trim_num"],
        trimmed_mean_denominator=arrays["trim_den"],
        prevalence_numerator=arrays["prev_num"],
        prevalence_denominator=arrays["prev_den"],
        pseudocount=arrays["pc"],
        log2_fold_change=arrays["lfc"],
        mean_log2_difference=arrays["mean_log_diff"],
        welch_t=arrays["welch_t"],
        welch_df=arrays["welch_df"],
        p_value=arrays["p"],
        q_value=q,
        test_valid=test_valid,
        inference_status=status,  # type: ignore[arg-type]
        spatial_fingerprint=roi_masks.fingerprint,
        warnings=tuple(warnings),
        source_dataset_hash=ds.hash(include_rois=False),
        overlap_policy=roi_masks.overlap_policy,
        min_units_per_group=int(min_units_per_group),
        trim_fraction=float(trim_fraction),
        pseudocount_quantile=float(pseudocount_quantile),
    )


def analyze_axis_profiles(
    ds: MSIDataset,
    axis: DirectedAxis | AxisProjection,
    *,
    include_mask: np.ndarray | None = None,
    selection_names: Sequence[str] = (),
    n_bins: int = 20,
    min_pixels_per_bin: int = 2,
    min_bins_for_trend: int = 4,
    endpoint_fraction: float = 0.2,
    min_endpoint_pixels: int = 3,
    n_permutations: int = 999,
    rng_seed: int = 0,
    trim_fraction: float = 0.1,
    pseudocount_quantile: float = 0.05,
    q_threshold: float = 0.05,
    trend_threshold: float = 0.5,
    concentration_threshold: float = 0.25,
    endpoint_effect_threshold: float = 1.0,
    chunk_channels: int = 256,
) -> AxisProfileResult:
    """Build binned profiles and test monotonic enrichment along a directed axis.

    Spearman's rho is evaluated on valid ordered bins. The empirical null shuffles
    bin order and uses a reversal-antithetic partner for every random permutation;
    consequently reversing the axis flips rho but leaves finite-sample p/q values
    exactly invariant for the same seed.
    """
    pm = _require_peak_matrix(ds)
    selected_names = tuple(str(name) for name in selection_names)
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("selection_names must be unique")
    _validate_common_options(
        chunk_channels=chunk_channels,
        trim_fraction=trim_fraction,
        pseudocount_quantile=pseudocount_quantile,
    )
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if min_pixels_per_bin < 1:
        raise ValueError("min_pixels_per_bin must be at least 1")
    if min_bins_for_trend < 3:
        raise ValueError("min_bins_for_trend must be at least 3")
    if not 0 < endpoint_fraction <= 0.5:
        raise ValueError("endpoint_fraction must lie in (0, 0.5]")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if min_endpoint_pixels < 1:
        raise ValueError("min_endpoint_pixels must be at least 1")
    if not 0 < q_threshold <= 1:
        raise ValueError("q_threshold must lie in (0, 1]")
    if not 0 <= trend_threshold <= 1:
        raise ValueError("trend_threshold must lie in [0, 1]")
    if not 0 <= concentration_threshold <= 1:
        raise ValueError("concentration_threshold must lie in [0, 1]")
    if not np.isfinite(endpoint_effect_threshold) or endpoint_effect_threshold < 0:
        raise ValueError("endpoint_effect_threshold must be finite and non-negative")

    projection = (
        project_to_axis(ds, axis, include_mask=include_mask)
        if isinstance(axis, DirectedAxis)
        else axis
    )
    if projection.t.shape != (ds.n_pixels,):
        raise ValueError(
            f"axis projection has {projection.t.size} pixels; dataset has {ds.n_pixels}"
        )
    if projection.coordinate_fingerprint != coordinate_fingerprint(ds):
        raise ValueError(
            "axis projection was created for different grid coordinates or pixel row order"
        )
    if include_mask is not None and isinstance(axis, AxisProjection):
        include = np.asarray(include_mask, dtype=bool)
        if include.shape != (ds.n_pixels,):
            raise ValueError(f"include_mask shape {include.shape} != ({ds.n_pixels},)")
        included = np.asarray(projection.included_mask, dtype=bool) & include
        projection = AxisProjection(
            axis=projection.axis,
            t=np.where(included, projection.t, np.nan),
            signed_distance_px=np.where(
                included, projection.signed_distance_px, np.nan
            ),
            included_mask=included,
            coordinate_fingerprint=projection.coordinate_fingerprint,
        )
    included = np.asarray(projection.included_mask, dtype=bool)
    rows = np.flatnonzero(included)
    if rows.size == 0:
        raise ValueError("axis and optional width/mask include no populated pixels")

    t_values = np.asarray(projection.t[rows], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.searchsorted(edges, t_values, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1).astype(np.int64, copy=False)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.int64)
    nonempty_bins = counts > 0
    valid_bins = counts >= min_pixels_per_bin
    n_valid_bins = int(valid_bins.sum())
    start_local = t_values <= endpoint_fraction + 1e-12
    end_local = t_values >= 1.0 - endpoint_fraction - 1e-12
    n_start, n_end = int(start_local.sum()), int(end_local.sum())

    mz = np.asarray(pm.mz_axis[:], dtype=np.float64)
    k = mz.size
    mean_int = np.full((n_bins, k), np.nan, dtype=np.float64)
    median_int = np.full((n_bins, k), np.nan, dtype=np.float64)
    mean_log = np.full((n_bins, k), np.nan, dtype=np.float64)
    prevalence = np.full((n_bins, k), np.nan, dtype=np.float64)
    pseudocount = np.full(k, np.nan, dtype=np.float64)
    endpoint_lfc = np.full(k, np.nan, dtype=np.float64)

    for start in range(0, k, chunk_channels):
        stop = min(start + chunk_channels, k)
        x = _read_rows_chunk(pm.matrix, rows, start, stop)
        _validate_intensity_chunk(x)
        pc = _adaptive_pseudocount(x, pseudocount_quantile)
        pseudocount[start:stop] = pc
        log_x = np.log2(x + pc[None, :])
        for bi in range(n_bins):
            sel = bin_idx == bi
            if not sel.any():
                continue
            xb = x[sel, :]
            mean_int[bi, start:stop] = xb.mean(axis=0)
            median_int[bi, start:stop] = np.median(xb, axis=0)
            mean_log[bi, start:stop] = log_x[sel, :].mean(axis=0)
            prevalence[bi, start:stop] = (xb > 0).mean(axis=0)
        if n_start >= min_endpoint_pixels and n_end >= min_endpoint_pixels:
            start_center = _trimmed_mean_or_nan(
                x[start_local, :], stop - start, trim_fraction
            )
            end_center = _trimmed_mean_or_nan(
                x[end_local, :], stop - start, trim_fraction
            )
            endpoint_lfc[start:stop] = np.log2(
                (end_center + pc) / (start_center + pc)
            )

    warnings: list[str] = [
        "Axis permutation scores use ordered bins within one image. Spatial "
        "autocorrelation and unequal bin precision can make them optimistic; "
        "population claims require specimen-level replicate inference."
    ]
    if n_valid_bins < min_bins_for_trend:
        rho = np.full(k, np.nan, dtype=np.float64)
        p_value = np.full(k, np.nan, dtype=np.float64)
        q_value = np.full(k, np.nan, dtype=np.float64)
        status = "insufficient_bins"
        effective_permutations = 0
        warnings.append(
            f"Trend inference skipped: {n_valid_bins} bins have at least "
            f"{min_pixels_per_bin} pixels; {min_bins_for_trend} are required."
        )
    else:
        profile_for_trend = mean_log[valid_bins, :]
        rng = np.random.default_rng(int(rng_seed))
        rho, p_value, effective_permutations = _ordered_spearman_permutation(
            profile_for_trend, n_permutations=n_permutations, rng=rng
        )
        q_value = benjamini_hochberg(p_value)
        status = "ok"
        minimum_single_signal_q = min(1.0, k / (effective_permutations + 1))
        if minimum_single_signal_q > q_threshold:
            warnings.append(
                f"Permutation resolution is limited for {k} channels: an isolated "
                f"minimum-p signal can have BH q no smaller than approximately "
                f"{minimum_single_signal_q:.3g}. Increase permutations or treat "
                "trend labels as descriptive."
            )

    peak_position, concentration = _profile_peak_metrics(
        mean_intensity=mean_int,
        bin_centers=centers,
        valid_bins=nonempty_bins,
    )
    labels = _pattern_labels(
        rho=rho,
        q_value=q_value,
        endpoint_lfc=endpoint_lfc,
        peak_position=peak_position,
        concentration=concentration,
        mean_intensity=mean_int,
        descriptive_bins=nonempty_bins,
        allow_localization=n_valid_bins >= min_bins_for_trend,
        q_threshold=q_threshold,
        trend_threshold=trend_threshold,
        concentration_threshold=concentration_threshold,
        endpoint_fraction=endpoint_fraction,
        endpoint_effect_threshold=endpoint_effect_threshold,
    )
    if n_start < min_endpoint_pixels or n_end < min_endpoint_pixels:
        warnings.append(
            f"Endpoint enrichment is unavailable: start has {n_start} and end has "
            f"{n_end} included pixels; each requires {min_endpoint_pixels}."
        )

    return AxisProfileResult(
        axis_name=projection.axis.name,
        start_label=projection.axis.start_label,
        end_label=projection.axis.end_label,
        mz=mz,
        bin_centers=centers,
        bin_edges=edges,
        bin_counts=counts,
        mean_intensity=mean_int,
        median_intensity=median_int,
        mean_log2_intensity=mean_log,
        prevalence=prevalence,
        pseudocount=pseudocount,
        spearman_rho=rho,
        p_value=p_value,
        q_value=q_value,
        endpoint_log2_enrichment=endpoint_lfc,
        peak_position=peak_position,
        concentration=concentration,
        pattern_label=labels,
        n_included_pixels=int(rows.size),
        n_start_pixels=n_start,
        n_end_pixels=n_end,
        n_permutations_requested=int(n_permutations),
        n_permutations_effective=int(effective_permutations),
        inference_status=status,  # type: ignore[arg-type]
        spatial_fingerprint=projection.fingerprint,
        warnings=tuple(warnings),
        source_dataset_hash=ds.hash(include_rois=False),
        axis_start_yx=tuple(float(v) for v in projection.axis.start_yx),
        axis_end_yx=tuple(float(v) for v in projection.axis.end_yx),
        half_width_px=(
            None
            if projection.axis.half_width_px is None
            else float(projection.axis.half_width_px)
        ),
        min_pixels_per_bin=int(min_pixels_per_bin),
        min_bins_for_trend=int(min_bins_for_trend),
        endpoint_fraction=float(endpoint_fraction),
        min_endpoint_pixels=int(min_endpoint_pixels),
        trim_fraction=float(trim_fraction),
        pseudocount_quantile=float(pseudocount_quantile),
        q_threshold=float(q_threshold),
        trend_threshold=float(trend_threshold),
        concentration_threshold=float(concentration_threshold),
        endpoint_effect_threshold=float(endpoint_effect_threshold),
        rng_seed=int(rng_seed),
        selection_names=selected_names,
    )


def _require_peak_matrix(ds: MSIDataset) -> PeakMatrix:
    if not isinstance(ds.backend, PeakMatrix):
        raise RuntimeError(
            "developmental analysis requires a harmonized PeakMatrix backend; "
            "run consensus alignment first"
        )
    if ds.backend.n_peaks == 0:
        raise RuntimeError("developmental analysis requires at least one shared channel")
    return ds.backend


def _as_names(names: str | Sequence[str]) -> tuple[str, ...]:
    out = (names,) if isinstance(names, str) else tuple(names)
    if not out:
        raise ValueError("at least one ROI name is required per contrast arm")
    if len(set(out)) != len(out):
        raise ValueError(f"ROI names within a contrast arm must be unique, got {out}")
    return out


def _validate_common_options(
    *, chunk_channels: int, trim_fraction: float, pseudocount_quantile: float
) -> None:
    if chunk_channels < 1:
        raise ValueError("chunk_channels must be at least 1")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    if not 0 < pseudocount_quantile <= 0.5:
        raise ValueError("pseudocount_quantile must lie in (0, 0.5]")


def _read_rows_chunk(matrix, rows: np.ndarray, start: int, stop: int) -> np.ndarray:  # noqa: ANN001
    """Read only selected rows and a contiguous channel chunk from numpy/Zarr."""
    rows = np.asarray(rows, dtype=np.int64)
    width = stop - start
    if rows.size == 0:
        return np.empty((0, width), dtype=np.float64)
    if isinstance(matrix, np.ndarray):
        return np.asarray(matrix[rows, start:stop], dtype=np.float64)
    oindex = getattr(matrix, "oindex", None)
    if oindex is not None:
        return np.asarray(oindex[rows, slice(start, stop)], dtype=np.float64)
    return np.asarray(matrix[rows, start:stop], dtype=np.float64)


def _validate_intensity_chunk(*chunks: np.ndarray) -> None:
    for chunk in chunks:
        if not np.isfinite(chunk).all():
            raise ValueError("analysis requires finite intensities")
        if (chunk < 0).any():
            raise ValueError("analysis requires non-negative intensities")


def _adaptive_pseudocount(x: np.ndarray, quantile: float) -> np.ndarray:
    """Half a low positive quantile per channel; 1 for all-zero channels."""
    k = x.shape[1]
    out = np.ones(k, dtype=np.float64)
    for c in range(k):
        positive = x[:, c][x[:, c] > 0]
        if positive.size:
            out[c] = max(0.5 * float(np.quantile(positive, quantile)), 1e-12)
    return out


def _mean_or_nan(x: np.ndarray, width: int) -> np.ndarray:
    return x.mean(axis=0) if x.shape[0] else np.full(width, np.nan)


def _median_or_nan(x: np.ndarray, width: int) -> np.ndarray:
    return np.median(x, axis=0) if x.shape[0] else np.full(width, np.nan)


def _prevalence_or_nan(x: np.ndarray, width: int) -> np.ndarray:
    return (x > 0).mean(axis=0) if x.shape[0] else np.full(width, np.nan)


def _trimmed_mean_or_nan(
    x: np.ndarray, width: int, trim_fraction: float
) -> np.ndarray:
    if x.shape[0] == 0:
        return np.full(width, np.nan)
    cut = int(np.floor(trim_fraction * x.shape[0]))
    if cut == 0:
        return x.mean(axis=0)
    ordered = np.sort(x, axis=0)
    return ordered[cut : x.shape[0] - cut, :].mean(axis=0)


def _welch_from_transformed(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Welch t-test without scipy's zero-variance RuntimeWarnings."""
    n_a, n_b = a.shape[0], b.shape[0]
    mean_a = a.mean(axis=0)
    mean_b = b.mean(axis=0)
    var_a = a.var(axis=0, ddof=1)
    var_b = b.var(axis=0, ddof=1)
    term_a = var_a / n_a
    term_b = var_b / n_b
    se2 = term_a + term_b
    df_denom = (term_a**2) / (n_a - 1) + (term_b**2) / (n_b - 1)
    valid = (se2 > np.finfo(np.float64).eps) & (df_denom > 0)
    t_stat = np.full(mean_a.shape, np.nan, dtype=np.float64)
    df = np.full(mean_a.shape, np.nan, dtype=np.float64)
    p_value = np.full(mean_a.shape, np.nan, dtype=np.float64)
    t_stat[valid] = (mean_a[valid] - mean_b[valid]) / np.sqrt(se2[valid])
    df[valid] = se2[valid] ** 2 / df_denom[valid]
    p_value[valid] = 2.0 * student_t.sf(np.abs(t_stat[valid]), df[valid])

    # Identical constants carry no evidence and have the natural neutral result.
    same_constant = (~valid) & np.isclose(mean_a, mean_b, rtol=1e-12, atol=1e-12)
    t_stat[same_constant] = 0.0
    df[same_constant] = np.inf
    p_value[same_constant] = 1.0
    valid[same_constant] = True
    return t_stat, df, p_value, valid


def _ordered_spearman_permutation(
    profile: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Spearman trend with reversal-paired random order permutations."""
    n_bins, k = profile.shape
    x_rank = np.arange(n_bins, dtype=np.float64)
    x_centered = x_rank - x_rank.mean()
    x_ss = float(x_centered @ x_centered)
    y_rank = rankdata(profile, axis=0, method="average")
    y_centered = y_rank - y_rank.mean(axis=0, keepdims=True)
    y_ss = (y_centered**2).sum(axis=0)
    denom = np.sqrt(x_ss * y_ss)
    rho = np.divide(
        x_centered @ y_centered,
        denom,
        out=np.zeros(k, dtype=np.float64),
        where=denom > 0,
    )
    extreme = np.zeros(k, dtype=np.int64)
    target = np.abs(rho)
    reversal = np.arange(n_bins - 1, -1, -1, dtype=np.int64)
    for _ in range(n_permutations):
        perm = rng.permutation(n_bins)
        for order in (perm, reversal[perm]):
            rho_perm = np.divide(
                x_centered @ y_centered[order, :],
                denom,
                out=np.zeros(k, dtype=np.float64),
                where=denom > 0,
            )
            extreme += np.abs(rho_perm) >= target - 1e-15
    effective = 2 * n_permutations
    p_value = (extreme + 1).astype(np.float64) / (effective + 1)
    return rho, p_value, effective


def _profile_peak_metrics(
    *, mean_intensity: np.ndarray, bin_centers: np.ndarray, valid_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    profile = np.nan_to_num(mean_intensity[valid_bins, :], nan=0.0)
    centers = bin_centers[valid_bins]
    k = mean_intensity.shape[1]
    peak_position = np.full(k, np.nan, dtype=np.float64)
    concentration = np.zeros(k, dtype=np.float64)
    if profile.shape[0] == 0:
        return peak_position, concentration
    total = profile.sum(axis=0)
    detected = total > 0
    # Average equally high peak locations. Besides representing a genuine plateau
    # better than first-occurrence argmax, this makes the position exactly covariant
    # under endpoint reversal for symmetric two-bin peaks.
    for c in np.flatnonzero(detected):
        maximum = float(profile[:, c].max())
        at_peak = np.isclose(profile[:, c], maximum, rtol=1e-7, atol=1e-12)
        peak_position[c] = float(centers[at_peak].mean())
    if profile.shape[0] == 1:
        concentration[detected] = 1.0
        return peak_position, concentration
    for c in np.flatnonzero(detected):
        probabilities = profile[:, c] / total[c]
        positive = probabilities > 0
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
        concentration[c] = np.clip(
            1.0 - entropy / np.log(profile.shape[0]), 0.0, 1.0
        )
    return peak_position, concentration


def _pattern_labels(
    *,
    rho: np.ndarray,
    q_value: np.ndarray,
    endpoint_lfc: np.ndarray,
    peak_position: np.ndarray,
    concentration: np.ndarray,
    mean_intensity: np.ndarray,
    descriptive_bins: np.ndarray,
    allow_localization: bool,
    q_threshold: float,
    trend_threshold: float,
    concentration_threshold: float,
    endpoint_fraction: float,
    endpoint_effect_threshold: float,
) -> np.ndarray:
    k = mean_intensity.shape[1]
    labels = np.empty(k, dtype=object)
    total = np.nansum(mean_intensity[descriptive_bins, :], axis=0)
    for c in range(k):
        if not np.isfinite(total[c]) or total[c] <= 0:
            labels[c] = "undetected"
        elif (
            np.isfinite(q_value[c])
            and q_value[c] <= q_threshold
            and rho[c] >= trend_threshold
        ):
            labels[c] = "increasing"
        elif (
            np.isfinite(q_value[c])
            and q_value[c] <= q_threshold
            and rho[c] <= -trend_threshold
        ):
            labels[c] = "decreasing"
        elif allow_localization and concentration[c] >= concentration_threshold:
            if peak_position[c] <= endpoint_fraction:
                labels[c] = "start_localized"
            elif peak_position[c] >= 1.0 - endpoint_fraction:
                labels[c] = "end_localized"
            else:
                labels[c] = "interior_localized"
        elif np.isfinite(endpoint_lfc[c]) and endpoint_lfc[c] >= endpoint_effect_threshold:
            labels[c] = "end_enriched"
        elif np.isfinite(endpoint_lfc[c]) and endpoint_lfc[c] <= -endpoint_effect_threshold:
            labels[c] = "start_enriched"
        elif not allow_localization:
            labels[c] = "detected_sparse"
        else:
            labels[c] = "diffuse_or_complex"
    return labels
