"""Pretty-print per-node diagnostics from a ``RunResult``.

The raw ``RunResult.diagnostics`` is a ``dict[node_id, list[Diagnostic]]`` and
each ``Diagnostic.summary`` is a flat ``dict[str, float]`` — readable by code,
ugly to look at. This module turns that structure into a tabular block of
text annotated with health-check glyphs (✓ healthy / ⚠ check / ✗ unhealthy)
that point the user at the keys most likely to indicate trouble.

The formatter is shared by the wizard's RunPage log and the headless
``dapple-apply-spec`` / ``dapple-cohort-align`` CLIs so the analyst sees the
same shape of output everywhere.

Public API
----------

- ``format_diagnostics(run_result) -> list[str]`` — full block (header,
  per-node tables, footer with watch list).
- ``format_node_diagnostics(node_id, diagnostics) -> list[str]`` — single
  node's block (used by tests and ad-hoc rendering).
- ``RUBRICS`` — per-key health rubric registry (extensible).

Health rubric design
--------------------

For each summary key we know how to interpret, ``RUBRICS`` carries:

- ``label``: human-friendly column name
- ``fmt``: format string for the value (e.g. ``"{:.3f}"``, ``"{:.0f}"``)
- ``healthy``, ``warning``, ``bad``: optional predicates. The first one
  that returns ``True`` (in priority order ``bad → warning → healthy``)
  wins; everything else is shown without a glyph.
- ``healthy_hint``: short string shown next to a ✓ to tell the user what
  the threshold means (``"≥ 5"``, ``"> 0"``, ...).
- ``why_watch``: short remediation hint when the check is *not* green —
  goes into the run's footer "what to watch" list.

Keys without a rubric are still displayed but unannotated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from dapple.ops.base import Diagnostic
from dapple.pipeline.runner import RunResult


# Visual marks. Plain unicode keeps this format-agnostic — works in QTextEdit,
# CLI stdout, GitHub markdown, etc.
GLYPH_OK = "✓"
GLYPH_WARN = "⚠"
GLYPH_BAD = "✗"


# ---------- Rubric ---------------------------------------------------------------


@dataclass(frozen=True)
class Rubric:
    """How to render and grade one summary key.

    The grading fields turn a numeric value into one of three states: ✓ healthy,
    ⚠ check, ✗ unhealthy. Different states warrant different remediation, so
    the rubric carries:

    - ``meaning``: 1-sentence plain-English explanation of *what the value
      represents* — surfaced in the footer when the key is flagged so the user
      knows why it matters.
    - ``warn_advice``: action to take when the metric grades ⚠ (borderline).
    - ``bad_advice``: action to take when the metric grades ✗ (unhealthy).
    - ``why_watch``: legacy fallback — used when ``warn_advice`` /
      ``bad_advice`` are unset. Kept for compatibility.

    The ``healthy_hint`` is a short string shown next to the ✓ glyph (e.g.
    ``"≥ 5"``). It's not advice, just the threshold the user passed.
    """

    label: str
    fmt: str = "{:.4g}"
    healthy: Callable[[float], bool] | None = None
    warning: Callable[[float], bool] | None = None
    bad: Callable[[float], bool] | None = None
    healthy_hint: str = ""
    why_watch: str = ""
    meaning: str = ""
    warn_advice: str = ""
    bad_advice: str = ""

    def advice_for(self, glyph: str) -> str:
        """Pick the right remediation string for the given grade glyph."""
        if glyph == GLYPH_BAD:
            return self.bad_advice or self.warn_advice or self.why_watch
        if glyph == GLYPH_WARN:
            return self.warn_advice or self.bad_advice or self.why_watch
        return ""


# ---------- Per-key rubrics ------------------------------------------------------

# Most rubrics are keyed by *summary key only* and apply to whichever operator
# emits that key (the keys are unique enough across operators that this works in
# practice). Where context matters (e.g. fraction_pixels_recalibrated means
# different things for msiwarp vs lock_mass), the same rubric still works.
RUBRICS: dict[str, Rubric] = {
    # detect_reference_ions
    "n_reference_ions": Rubric(
        label="reference ions",
        fmt="{:.0f}",
        healthy=lambda v: v >= 5,
        warning=lambda v: 3 <= v < 5,
        bad=lambda v: v < 3,
        healthy_hint="≥ 5",
        meaning="how many endogenous m/z anchors were detected across pixels. "
                "Tolerance fitting and recalibration both feed off these.",
        warn_advice="3-4 anchors works but yields wide CIs and unstable "
                    "recalibration. Lower 'Minimum pixel prevalence' (try "
                    "0.5-0.6) on the reference card to surface more, or "
                    "widen 'Coarse tolerance (ppm)' to merge near-duplicate bins.",
        bad_advice="fewer than 3 anchors makes empirical tolerance fall back "
                   "to a flat curve and disables msiwarp recalibration. "
                   "Lower 'Minimum pixel prevalence' to 0.3-0.5 and widen "
                   "'Coarse tolerance (ppm)' (e.g. doubled). If still empty, "
                   "the dataset has no reproducible peaks and downstream "
                   "consensus quality will be poor regardless.",
    ),
    "prevalence_min": Rubric(
        label="prevalence min",
        fmt="{:.3f}",
        healthy=lambda v: v >= 0.5,
        warning=lambda v: 0.3 <= v < 0.5,
        bad=lambda v: v < 0.3,
        healthy_hint="≥ 0.50",
        meaning="the *least* widely-shared reference: it sits in this fraction "
                "of pixels. A weak weakest-anchor means the tolerance fit's "
                "low-m/z support is shaky.",
        warn_advice="weakest anchor is between 30% and 50% of pixels. The "
                    "tolerance curve at that anchor's m/z will be noisier than "
                    "elsewhere. Either accept (the fit pools across all "
                    "anchors) or raise 'Minimum pixel prevalence' on the "
                    "reference card to drop this anchor.",
        bad_advice="weakest anchor is in fewer than 30% of pixels — likely a "
                   "false positive. Raise 'Minimum pixel prevalence' on the "
                   "reference card to 0.5-0.6 to drop unstable anchors.",
    ),
    "prevalence_median": Rubric(
        label="prevalence median",
        fmt="{:.3f}",
        healthy=lambda v: v >= 0.7,
        warning=lambda v: 0.5 <= v < 0.7,
        bad=lambda v: v < 0.5,
        healthy_hint="≥ 0.70",
        meaning="median fraction of pixels carrying each reference. High "
                "values (≥ 0.7) mean anchors are robust across the image.",
        warn_advice="anchors aren't very widespread (50-70% of pixels). The "
                    "tolerance fit is still usable but expect 1.5-2× wider CIs "
                    "than ideal. Common cause: heterogeneous matrix coverage. "
                    "Consider whether reference_ion_normalize is the right "
                    "normalization on this dataset.",
        bad_advice="reference ions don't recur across the image (median below "
                   "50%). The tolerance fit's average is being pulled by a "
                   "small number of well-represented anchors. Inspect the "
                   "reference list (the diagnostic plot shows m/z vs "
                   "prevalence) and tighten 'Minimum pixel prevalence' to drop "
                   "the rare ones.",
    ),
    "prevalence_max": Rubric(label="prevalence max", fmt="{:.3f}"),
    "n_reference_observations": Rubric(label="ref. observations", fmt="{:.0f}"),

    # empirical_tolerance_from_reference_ions
    "warning_flat_tolerance": Rubric(
        label="!! flat-tolerance fallback",
        fmt="{:.0f}",
        bad=lambda v: v >= 1.0,
        meaning="set to 1 when the empirical tolerance fit fell back to a "
                "single flat ppm value because too few (m/z, ppm) pairs were "
                "available to fit a curve.",
        bad_advice="upstream reference-ion detection found too few anchors. "
                   "Lower 'Minimum pixel prevalence' on the reference card "
                   "and/or widen 'Coarse tolerance (ppm)' to recover more "
                   "anchors. Without a curve, the consensus alignment uses "
                   "'Default tolerance (ppm)' for every m/z — usable but a "
                   "blunt instrument.",
    ),
    "ppm_at_mz_lo": Rubric(label="tolerance @ mz_lo (ppm)", fmt="{:.2f}"),
    "ppm_at_mz_hi": Rubric(label="tolerance @ mz_hi (ppm)", fmt="{:.2f}"),
    "ppm_median": Rubric(label="tolerance median (ppm)", fmt="{:.2f}"),
    "ci_band_median_width": Rubric(label="ci band width (ppm)", fmt="{:.2f}"),
    "alpha": Rubric(label="alpha (coverage)", fmt="{:.3f}"),
    "bootstrap_B": Rubric(label="bootstrap replicates", fmt="{:.0f}"),
    "coarse_tol_ppm": Rubric(label="coarse_tol_ppm", fmt="{:.2f}"),
    "n_pixels": Rubric(label="pixels in", fmt="{:.0f}"),
    "anchor_strategy_index": Rubric(
        label="anchor strategy",
        fmt="{:.0f}",
    ),
    "factor_kind_tic": Rubric(
        label="normalizer kind (0=median, 1=TIC)",
        fmt="{:.0f}",
    ),
    "factor_q25": Rubric(label="scale factor q25", fmt="{:.4g}"),
    "factor_q75": Rubric(label="scale factor q75", fmt="{:.4g}"),

    # msiwarp_recalibrate / lock_mass_recalibrate
    "fraction_pixels_recalibrated": Rubric(
        label="pixels recalibrated",
        fmt="{:.1%}",
        healthy=lambda v: v >= 0.9,
        warning=lambda v: 0.6 <= v < 0.9,
        bad=lambda v: v < 0.6,
        healthy_hint="≥ 90%",
        meaning="fraction of pixels where the recalibrator successfully fit a "
                "warp. Skipped pixels keep their original m/z values, which "
                "leaves a per-pixel-mass-axis mismatch that hurts consensus.",
        warn_advice="60-90% of pixels were corrected — the rest had too few "
                    "visible anchors. For msiwarp: drop 'Minimum inliers' from "
                    "3 to 2 to accept thinner anchor sets (less stable but "
                    "covers more pixels). For lock_mass: relax 'Max shift "
                    "(ppm)' if the diagnostic shows shifts at that boundary.",
        bad_advice="under 60% of pixels were corrected — recalibration is "
                   "doing little work. Either reference detection is yielding "
                   "few anchors per pixel (check 'reference ions' above), or "
                   "your tolerance is too tight for the actual drift. "
                   "Consider disabling recalibration and seeing if consensus "
                   "alignment still recovers your peaks; if it does, drop "
                   "msiwarp/lock_mass entirely.",
    ),
    "n_pixels_recalibrated": Rubric(label="pixels corrected", fmt="{:.0f}"),
    "n_pixels_skipped": Rubric(label="pixels skipped", fmt="{:.0f}"),
    "inliers_median": Rubric(
        label="inlier anchors / pixel (median)",
        fmt="{:.1f}",
        healthy=lambda v: v >= 5,
        warning=lambda v: 3 <= v < 5,
        bad=lambda v: v < 3,
        healthy_hint="≥ 5 anchors",
        meaning="median number of reference anchors that survived RANSAC for "
                "msiwarp's per-pixel linear fit. 5+ is comfortable; 2-3 risks "
                "overfitting; 1 is impossible to fit.",
        warn_advice="3-5 inliers per pixel — the linear fit is workable but "
                    "noisy. Either accept (most pixels still recalibrate "
                    "well) or relax 'RANSAC threshold (ppm)' to admit more "
                    "anchors as inliers.",
        bad_advice="fewer than 3 inliers per pixel — too thin for msiwarp's "
                   "linear fit. Switch to lock_mass_recalibrate (single-anchor "
                   "shift, more robust on sparse anchor sets), or accept that "
                   "msiwarp will skip most pixels.",
    ),
    "inliers_min": Rubric(label="inlier anchors / pixel (min)", fmt="{:.0f}"),
    "improvement_ppm": Rubric(
        label="residual improvement (ppm)",
        fmt="{:+.2f}",
        healthy=lambda v: v > 0,
        bad=lambda v: v <= 0,
        healthy_hint="above 0 (lower is better)",
        meaning="median |ppm residual| improvement, pre minus post. Positive "
                "means recalibration tightened the anchor residuals; zero or "
                "negative means it added noise.",
        bad_advice="recalibration did not improve residuals (the warp is "
                   "fitting noise more than drift). Common causes: (1) the "
                   "instrument is already locked-mass good — disable "
                   "recalibration entirely; (2) the reference set has "
                   "false-positive anchors that destabilize the fit — raise "
                   "'Minimum pixel prevalence' on the reference card; (3) the "
                   "RANSAC threshold is too permissive — lower 'RANSAC "
                   "threshold (ppm)' to reject bad anchor pairs.",
    ),
    "pre_residual_median_ppm": Rubric(label="pre-fit residual (ppm)", fmt="{:.2f}"),
    "post_residual_median_ppm": Rubric(label="post-fit residual (ppm)", fmt="{:.2f}"),
    "ppm_shift_median": Rubric(label="ppm shift median", fmt="{:+.2f}"),
    "ppm_shift_abs_median": Rubric(
        label="|ppm shift| median",
        fmt="{:.2f}",
        healthy=lambda v: v < 50,
        warning=lambda v: 50 <= v < 200,
        bad=lambda v: v >= 200,
        healthy_hint="under 50 ppm",
        meaning="median magnitude of the per-pixel shift the lock-mass "
                "recalibrator applied. Small shifts are good (small drift); "
                "large shifts mean either real drift or a bad anchor.",
        warn_advice="median shift between 50 and 200 ppm. This is plausible "
                    "for axial-linear MALDI-TOF but unusual on Q-TOF / "
                    "reflectron. Inspect the diagnostic to see if shifts are "
                    "uniform (real drift, fine) or bimodal (anchor "
                    "misidentification — switch 'Anchor strategy' from "
                    "highest_intensity to closest_to_mz with an explicit m/z).",
        bad_advice="median shift over 200 ppm is suspicious. Likely cause: "
                   "the anchor strategy is picking the wrong reference per "
                   "pixel. Switch to 'closest_to_mz' with an 'Explicit "
                   "anchor m/z' value you trust, and tighten 'Max shift "
                   "(ppm)' to refuse aggressive corrections.",
    ),
    "ppm_shift_abs_max": Rubric(label="|ppm shift| max", fmt="{:.2f}"),

    # normalize
    "factor_median": Rubric(label="scale factor median", fmt="{:.4g}"),
    "factor_min": Rubric(
        label="scale factor min",
        fmt="{:.4g}",
        bad=lambda v: v <= 0,
        healthy_hint="above 0",
        meaning="smallest per-pixel scale factor across the image. Zero or "
                "negative means at least one pixel had no usable signal — the "
                "normalizer fell back to its eps floor for those.",
        bad_advice="at least one pixel had a non-positive scale factor (no "
                   "signal). Those pixels were skipped (left at the "
                   "intensities they came in with). Usually safe to ignore, "
                   "but if it's many pixels, your hot-pixel filter or peak "
                   "picker may have over-trimmed upstream — check those "
                   "diagnostics first.",
    ),
    "factor_max": Rubric(label="scale factor max", fmt="{:.4g}"),
    "n_pixels_at_floor": Rubric(label="pixels at eps floor", fmt="{:.0f}"),

    # snr_peak_pick / cwt_peak_pick
    "n_peaks_in": Rubric(label="peaks in", fmt="{:.0f}"),
    "n_peaks_out": Rubric(label="peaks out", fmt="{:.0f}"),
    "n_peaks_total": Rubric(label="peaks total", fmt="{:.0f}"),
    "fraction_kept": Rubric(label="fraction kept", fmt="{:.1%}"),
    "kept_per_pixel_median": Rubric(label="peaks/pixel median", fmt="{:.0f}"),
    "kept_per_pixel_min": Rubric(label="peaks/pixel min", fmt="{:.0f}"),
    "kept_per_pixel_max": Rubric(label="peaks/pixel max", fmt="{:.0f}"),
    "threshold_median": Rubric(label="threshold median", fmt="{:.4g}"),
    "snr_mad": Rubric(label="snr_mad", fmt="{:.2f}"),
    "min_snr": Rubric(label="min_snr (cwt)", fmt="{:.1f}"),
    "n_widths": Rubric(label="cwt widths", fmt="{:.0f}"),
    "grid_size": Rubric(label="cwt grid size", fmt="{:.0f}"),

    # kde_consensus_alignment / dbscan_consensus
    "n_consensus_peaks": Rubric(
        label="consensus channels",
        fmt="{:.0f}",
        healthy=lambda v: v >= 10,
        warning=lambda v: 1 <= v < 10,
        bad=lambda v: v < 1,
        healthy_hint="≥ 10",
        meaning="number of m/z bins surviving the prominence + prevalence "
                "filters. This is the channel count you'll see in the "
                "Channels Panel and the multipage TIFF output.",
        warn_advice="fewer than 10 channels survived. The rejection-budget "
                    "summary on the consensus card tells you whether "
                    "prominence or prevalence is the dominant filter. To "
                    "loosen: lower 'Min prominence quantile' (try 0.3 instead "
                    "of 0.5) or 'Drop peaks present in < this fraction of "
                    "pixels' (try 0.02 instead of 0.05). On a synthetic "
                    "fixture with ≤ 5 planted peaks, 5 is correct.",
        bad_advice="zero consensus channels means every candidate was filtered "
                   "out. Check the consensus card's rejection-budget summary "
                   "to see which filter killed everything: lower the "
                   "corresponding parameter aggressively (try "
                   "'Min prominence quantile' = 0.1 or 'Min prevalence' = "
                   "0.01) and re-run. If still empty, peak picking upstream "
                   "is yielding too few peaks per pixel — check "
                   "'peaks/pixel median' on the picker card.",
    ),
    "n_local_maxima_total": Rubric(label="local maxima", fmt="{:.0f}"),
    "n_post_prominence": Rubric(label="post prominence filter", fmt="{:.0f}"),
    "n_rejected_by_prominence": Rubric(label="rejected by prominence", fmt="{:.0f}"),
    "n_rejected_by_prevalence": Rubric(label="rejected by prevalence", fmt="{:.0f}"),
    "prominence_threshold_value": Rubric(label="prominence threshold", fmt="{:.4g}"),
    "bandwidth_log_mz": Rubric(label="kde bandwidth (log m/z)", fmt="{:.2e}"),

    "n_clusters_pre_filter": Rubric(label="clusters pre-filter", fmt="{:.0f}"),
    "n_noise_peaks": Rubric(label="dbscan noise peaks", fmt="{:.0f}"),
    "fraction_noise": Rubric(
        label="fraction noise",
        fmt="{:.1%}",
        healthy=lambda v: v < 0.5,
        warning=lambda v: 0.5 <= v < 0.8,
        bad=lambda v: v >= 0.8,
        healthy_hint="under 50%",
        meaning="fraction of pooled peaks DBSCAN labeled as noise (label = -1). "
                "Real peaks should cluster; noise should not. Higher means "
                "more peaks were too isolated to belong to any cluster.",
        warn_advice="50-80% of peaks classified as noise — you're losing real "
                    "data to the noise label. Widen 'Eps (ppm)' (try 2× the "
                    "current value, or set to 0 to use the tolerance curve "
                    "fallback), or lower 'Min samples' from 5 to 3.",
        bad_advice="over 80% noise — DBSCAN can't see clusters. Either the "
                   "data is genuinely noisy, or your parameters are far too "
                   "strict. Set 'Eps (ppm)' to 0 (auto from tolerance curve) "
                   "and 'Min samples' to 2. If still bad, switch back to "
                   "kde_consensus_alignment, which is more robust on "
                   "low-density peak pools.",
    ),
    "eps_ppm_effective": Rubric(label="eps (ppm)", fmt="{:.2f}"),
    "eps_log_mz": Rubric(label="eps (log m/z)", fmt="{:.2e}"),
    "min_samples": Rubric(label="dbscan min_samples", fmt="{:.0f}"),
    "min_prevalence": Rubric(label="min_prevalence", fmt="{:.3f}"),
    "min_prominence_quantile": Rubric(label="min_prominence_quantile", fmt="{:.3f}"),
    "min_prevalence_threshold": Rubric(label="min_prevalence_threshold", fmt="{:.3f}"),

    # morans_i_permutation
    "n_channels_in": Rubric(label="channels in", fmt="{:.0f}"),
    "n_channels_out": Rubric(
        label="channels out",
        fmt="{:.0f}",
        healthy=lambda v: v >= 1,
        bad=lambda v: v < 1,
        healthy_hint="≥ 1",
        meaning="number of consensus channels surviving the spatial / "
                "background filter. Zero means every channel was dropped.",
        bad_advice="all channels were dropped. For Moran's I: raise "
                   "'q_threshold' (try 0.1 or 0.2) or check that "
                   "'sample_type' really is tissue — Moran's I on dispersed "
                   "(cell-culture, etc.) data legitimately fails. For "
                   "background_subtract: raise 'BG/FG ratio threshold' from "
                   "0.5 toward 0.8 to drop fewer channels.",
    ),
    "n_dropped_by_fdr": Rubric(label="dropped by fdr", fmt="{:.0f}"),
    "n_permutations": Rubric(label="permutations", fmt="{:.0f}"),
    "q_threshold": Rubric(label="q_threshold", fmt="{:.3f}"),

    # prevalence_fdr_filter
    "p_value_min": Rubric(label="occupancy p-score min", fmt="{:.4f}"),
    "p_value_median": Rubric(label="occupancy p-score median", fmt="{:.4f}"),
    "q_value_min": Rubric(label="adjusted sensitivity score min", fmt="{:.4f}"),
    "q_value_median": Rubric(label="adjusted sensitivity score median", fmt="{:.4f}"),
    "warning_uncalibrated_occupancy_null": Rubric(
        label="!! experimental occupancy model",
        fmt="{:.0f}",
        bad=lambda v: v >= 1.0,
        meaning="always set to 1 because consensus selection and per-pixel "
                "assignment violate the with-replacement occupancy null.",
        bad_advice="treat the reported values as sensitivity scores only. "
                   "Prefer a declared min_prevalence threshold or cohort "
                   "dataset prevalence for production decisions.",
    ),
    "warning_conservative_fallback": Rubric(
        label="!! conservative fallback",
        fmt="{:.0f}",
        bad=lambda v: v >= 1.0,
        meaning="set to 1 when the upstream consensus operator did not record "
                "per-channel peak counts. The sensitivity calculation falls "
                "back to the observed carrier count.",
        bad_advice="re-run the upstream consensus operator so it records "
                   "consensus_n_peaks_per_channel. Even with that count, the "
                   "occupancy model remains experimental and uncalibrated.",
    ),

    # hot_pixel_filter
    "n_hot_pixels": Rubric(label="hot pixels", fmt="{:.0f}"),
    "fraction_hot": Rubric(
        label="fraction hot",
        fmt="{:.1%}",
        healthy=lambda v: v <= 0.05,
        warning=lambda v: 0.05 < v <= 0.15,
        bad=lambda v: v > 0.15,
        healthy_hint="≤ 5%",
        meaning="fraction of pixels flagged as hot (TIC above median + "
                "k_mad·MAD). Real detector glitches are usually < 1%; numbers "
                "higher than that suggest legitimate bright tissue is being "
                "clipped.",
        warn_advice="5-15% of pixels flagged is high. Raise 'k_mad' from 5 to "
                    "7 or 8 to be more conservative. If your image has "
                    "genuinely hot tissue regions (a tumor margin, say), "
                    "consider switching 'Correction' to 'mark' so the filter "
                    "records the indices without mutating intensities — "
                    "downstream operators can decide whether to mask.",
        bad_advice="more than 15% of pixels flagged means the threshold is "
                   "way too aggressive for this image — you're clipping real "
                   "structure. Raise 'k_mad' to 8-10, or disable hot-pixel "
                   "filtering entirely on this dataset.",
    ),
    "tic_median": Rubric(label="tic median", fmt="{:.4g}"),
    "tic_mad": Rubric(label="tic MAD", fmt="{:.4g}"),
    "tic_threshold": Rubric(label="tic threshold", fmt="{:.4g}"),
    "k_mad": Rubric(label="k_mad", fmt="{:.2f}"),

    # background_subtract
    "n_fg_pixels": Rubric(label="foreground pixels", fmt="{:.0f}"),
    "n_bg_pixels": Rubric(label="background pixels", fmt="{:.0f}"),
    "ratio_median": Rubric(label="bg/fg ratio median", fmt="{:.3f}"),
    "ratio_max": Rubric(label="bg/fg ratio max", fmt="{:.3f}"),
    "bg_to_fg_ratio_threshold": Rubric(label="bg/fg threshold", fmt="{:.3f}"),
}


# ---------- Op headers (pretty per-node titles) -----------------------------------

OP_HEADERS: dict[str, str] = {
    "detect_reference_ions": "Reference-ion detection",
    "empirical_tolerance_from_reference_ions": "Empirical tolerance fit",
    "msiwarp_recalibrate": "Mass recalibration (MSIWarp)",
    "lock_mass_recalibrate": "Mass recalibration (lock-mass)",
    "median_normalize": "Per-pixel normalization (median)",
    "tic_normalize": "Per-pixel normalization (TIC)",
    "reference_ion_normalize": "Per-pixel normalization (reference-ion)",
    "snr_peak_pick": "Peak picking (SNR / centroided)",
    "cwt_peak_pick": "Peak picking (CWT / profile)",
    "kde_consensus_alignment": "Consensus alignment (KDE)",
    "dbscan_consensus": "Consensus alignment (DBSCAN)",
    "morans_i_permutation": "Spatial filter (Moran's I)",
    "prevalence_fdr_filter": "Experimental prevalence sensitivity filter",
    "hot_pixel_filter": "Hot-pixel filter",
    "background_subtract": "Background subtraction",
}


# ---------- Internal helpers -----------------------------------------------------


def _grade(rubric: Rubric, value: float) -> tuple[str, str]:
    """Apply the rubric's predicates and return (glyph, reason).

    Priority: bad → warning → healthy. Rubrics with no predicates return ('', '').
    """
    if rubric.bad is not None:
        try:
            if rubric.bad(value):
                return GLYPH_BAD, "unhealthy"
        except Exception:  # noqa: BLE001 — predicates may receive odd values
            pass
    if rubric.warning is not None:
        try:
            if rubric.warning(value):
                return GLYPH_WARN, "check"
        except Exception:  # noqa: BLE001
            pass
    if rubric.healthy is not None:
        try:
            if rubric.healthy(value):
                return GLYPH_OK, f"healthy ({rubric.healthy_hint})" if rubric.healthy_hint else "healthy"
        except Exception:  # noqa: BLE001
            pass
    return "", ""


def _format_value(rubric: Rubric | None, value: float) -> str:
    """Format `value` per the rubric (or fall through to %.4g)."""
    fmt = rubric.fmt if rubric is not None else "{:.4g}"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return repr(value)


def _row(label: str, value_str: str, glyph: str, reason: str, *, label_width: int, value_width: int) -> str:
    """Render a single 'key ......... value  glyph reason' row.

    Dots are used between the label and the value to give the eye a guide line.
    """
    pad_label = label.ljust(label_width, ".")
    pad_value = value_str.rjust(value_width)
    if glyph:
        return f"    {pad_label} {pad_value}   {glyph} {reason}"
    return f"    {pad_label} {pad_value}"


# ---------- Public API -----------------------------------------------------------


@dataclass(frozen=True)
class WatchItem:
    """A single non-healthy diagnostic flag with everything needed to act on it.

    Used by the formatter footer to surface actionable guidance per flagged
    metric (rather than a generic "needs attention" line).
    """

    node_id: str
    key: str
    label: str
    value_str: str
    glyph: str  # GLYPH_WARN or GLYPH_BAD
    meaning: str
    advice: str


def format_node_diagnostics(node_id: str, diagnostics: list[Diagnostic]) -> tuple[list[str], list[WatchItem]]:
    """Format a single node's diagnostics.

    Returns
    -------
    (lines, watches)
        ``lines`` is a list of pretty-printed text lines (header + key/value
        rows). ``watches`` is a list of ``WatchItem``s for any rubric that
        graded ⚠ or ✗ — the caller aggregates these into the run's footer
        with full per-key meaning + advice text.
    """
    if not diagnostics:
        return [], []

    op_name = diagnostics[0].name
    pretty_op = OP_HEADERS.get(op_name, op_name)
    header = f"  [{node_id}] {pretty_op}"

    # Aggregate every (key, value) pair across all diagnostics on this node.
    rows: list[tuple[str, str, str, str]] = []  # (label, value_str, glyph, reason)
    watches: list[WatchItem] = []
    for d in diagnostics:
        for key, value in d.summary.items():
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                continue
            rubric = RUBRICS.get(key)
            label = rubric.label if rubric is not None else key
            value_str = _format_value(rubric, fvalue)
            glyph = ""
            reason = ""
            if rubric is not None:
                glyph, reason = _grade(rubric, fvalue)
                if glyph in (GLYPH_WARN, GLYPH_BAD):
                    advice = rubric.advice_for(glyph)
                    if rubric.meaning or advice:
                        watches.append(WatchItem(
                            node_id=node_id,
                            key=key,
                            label=label,
                            value_str=value_str,
                            glyph=glyph,
                            meaning=rubric.meaning,
                            advice=advice,
                        ))
            rows.append((label, value_str, glyph, reason))

    if not rows:
        return [header, "    (no scalar summary keys)"], watches

    label_width = max(len(r[0]) for r in rows) + 1
    value_width = max(len(r[1]) for r in rows)
    lines = [header]
    for label, value_str, glyph, reason in rows:
        lines.append(_row(label, value_str, glyph, reason, label_width=label_width, value_width=value_width))
    return lines, watches


def format_diagnostics(run_result: RunResult, *, title: str = "Run diagnostics") -> list[str]:
    """Render every per-node diagnostic block for a RunResult.

    The output is a flat list of lines suitable for ``"\\n".join(...)``.
    Includes a top divider, per-node tables, and a footer "what to watch"
    block. The footer expands every flagged metric into:

    - what the metric measured (the rubric's ``meaning``)
    - what to do about it (state-specific ``warn_advice`` / ``bad_advice``)

    so the analyst can act without bouncing back to the algorithm reference.
    """
    lines: list[str] = []
    bar = "─" * 64
    lines.append(bar)
    lines.append(f"  {title}")
    lines.append(bar)

    all_watches: list[WatchItem] = []
    n_checks = 0
    n_healthy = 0
    for node_id, diags in run_result.diagnostics.items():
        block, watches = format_node_diagnostics(node_id, diags)
        if block:
            lines.append("")
            lines.extend(block)
        all_watches.extend(watches)
        # Count grades for the summary footer.
        for d in diags:
            for key, value in d.summary.items():
                if key not in RUBRICS:
                    continue
                rubric = RUBRICS[key]
                if rubric.healthy is None and rubric.warning is None and rubric.bad is None:
                    continue
                try:
                    fvalue = float(value)
                except (TypeError, ValueError):
                    continue
                glyph, _ = _grade(rubric, fvalue)
                if glyph == "":
                    continue
                n_checks += 1
                if glyph == GLYPH_OK:
                    n_healthy += 1

    # Footer.
    lines.append("")
    lines.append(bar)
    if n_checks > 0:
        if n_healthy == n_checks:
            lines.append(f"  {GLYPH_OK} {n_healthy}/{n_checks} health checks passed. Result is stable.")
        else:
            lines.append(
                f"  {n_healthy}/{n_checks} health checks passed. "
                f"{n_checks - n_healthy} need attention:"
            )
    if all_watches:
        # Deduplicate by (node, key) so a key reported by multiple diagnostics
        # only renders once.
        seen: set[tuple[str, str]] = set()
        for w in all_watches:
            sig = (w.node_id, w.key)
            if sig in seen:
                continue
            seen.add(sig)
            lines.extend(_format_watch_block(w))
    lines.append(bar)
    return lines


def _format_watch_block(w: WatchItem) -> list[str]:
    """Render one ``WatchItem`` as 3-4 lines: header, meaning, advice."""
    out: list[str] = []
    state_word = "unhealthy" if w.glyph == GLYPH_BAD else "borderline"
    out.append("")
    out.append(
        f"  {w.glyph} [{w.node_id}] {w.label} = {w.value_str} ({state_word})"
    )
    if w.meaning:
        out.extend(_wrap_indented(f"what: {w.meaning}", indent="      "))
    if w.advice:
        out.extend(_wrap_indented(f"fix:  {w.advice}", indent="      "))
    return out


def _wrap_indented(text: str, *, indent: str, width: int = 80) -> list[str]:
    """Wrap ``text`` to ``width`` chars and prepend ``indent`` to every line."""
    import textwrap

    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=indent + "      ",  # align continuation under the colon
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapper.wrap(text) or [indent + text]


def format_flat_summary(
    summary: Mapping[str, float],
    *,
    title: str = "Summary",
    extra_rubrics: dict[str, Rubric] | None = None,
) -> list[str]:
    """Pretty-print a flat ``{key: float}`` dict (e.g. cohort diagnostics).

    Uses the same RUBRICS table as ``format_node_diagnostics`` plus any
    ``extra_rubrics`` you supply. Output shape:

    ::

        ────────────────────────────────────────────────────────────────
          {title}
        ────────────────────────────────────────────────────────────────
            label1 ......... value1   ✓ healthy (≥ 5)
            label2 ......... value2
        ────────────────────────────────────────────────────────────────

    Returns a list of plain-text lines.
    """
    rubrics = dict(RUBRICS)
    if extra_rubrics:
        rubrics.update(extra_rubrics)

    rows: list[tuple[str, str, str, str]] = []
    watches: list[WatchItem] = []
    for key, value in summary.items():
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        rubric = rubrics.get(key)
        label = rubric.label if rubric is not None else key
        value_str = _format_value(rubric, fvalue)
        glyph, reason = ("", "")
        if rubric is not None:
            glyph, reason = _grade(rubric, fvalue)
            if glyph in (GLYPH_WARN, GLYPH_BAD):
                advice = rubric.advice_for(glyph)
                if rubric.meaning or advice:
                    watches.append(WatchItem(
                        node_id="cohort",
                        key=key,
                        label=label,
                        value_str=value_str,
                        glyph=glyph,
                        meaning=rubric.meaning,
                        advice=advice,
                    ))
        rows.append((label, value_str, glyph, reason))

    bar = "─" * 64
    if not rows:
        return [bar, f"  {title}", bar, "    (empty)", bar]

    label_width = max(len(r[0]) for r in rows) + 1
    value_width = max(len(r[1]) for r in rows)
    lines = [bar, f"  {title}", bar]
    for label, value_str, glyph, reason in rows:
        lines.append(_row(label, value_str, glyph, reason,
                           label_width=label_width, value_width=value_width))

    if watches:
        lines.append("")
        lines.append("  what to watch:")
        seen: set[str] = set()
        for w in watches:
            if w.key in seen:
                continue
            seen.add(w.key)
            lines.extend(_format_watch_block(w))
    lines.append(bar)
    return lines


# Cohort-specific rubrics (used by ``dapple-cohort-align``).
COHORT_RUBRICS: dict[str, Rubric] = {
    "n_datasets": Rubric(label="datasets", fmt="{:.0f}"),
    "n_total_pixels": Rubric(label="total pixels", fmt="{:.0f}"),
    "n_total_peaks_pooled": Rubric(label="pooled peaks", fmt="{:.0f}"),
    "n_consensus_candidates_pre_filter": Rubric(
        label="candidates pre-filter", fmt="{:.0f}"
    ),
    "n_consensus_shared": Rubric(
        label="shared consensus channels",
        fmt="{:.0f}",
        healthy=lambda v: v >= 10,
        warning=lambda v: 1 <= v < 10,
        bad=lambda v: v < 1,
        healthy_hint="≥ 10",
        meaning="number of m/z bins that survived the *cohort-wide* prevalence "
                "filter. Each cohort dataset gets a column for each of these.",
        warn_advice="fewer than 10 shared channels means the cohort members "
                    "don't agree on much. Either the datasets are too "
                    "different (different ionization, different sample types) "
                    "or 'min_prevalence' is too tight for cohort use. Try "
                    "lowering it to 0.2-0.3.",
        bad_advice="zero shared channels means the cohort has no common "
                   "ground. Check that all datasets use the same instrument "
                   "and ionization mode; if so, lower 'min_prevalence' "
                   "aggressively (e.g. 0.1) and inspect the per-dataset "
                   "prevalence in cohort_summary.json.",
    ),
    "cohort_prevalence_min": Rubric(label="cohort prevalence min", fmt="{:.3f}"),
    "cohort_prevalence_median": Rubric(
        label="cohort prevalence median",
        fmt="{:.3f}",
        healthy=lambda v: v >= 0.4,
        warning=lambda v: 0.2 <= v < 0.4,
        bad=lambda v: v < 0.2,
        healthy_hint="≥ 0.40",
        meaning="median fraction of *all cohort pixels* carrying signal at "
                "each shared channel. Higher means the channel set is broadly "
                "shared across cohort members.",
        warn_advice="median cohort prevalence between 20% and 40% means most "
                    "shared channels are present in only some cohort members. "
                    "Acceptable if you expect heterogeneous samples (e.g. "
                    "treatment vs control with one-sided induction); "
                    "concerning otherwise.",
        bad_advice="under 20% cohort prevalence — most channels appear in only "
                   "a fraction of pixels across the cohort. Either your "
                   "samples are very different from each other, or "
                   "min_prevalence is too low for cohort use. Tighten it to "
                   "0.3+ to focus on broadly-shared channels.",
    ),
    "cohort_prevalence_max": Rubric(label="cohort prevalence max", fmt="{:.3f}"),
}


__all__ = [
    "COHORT_RUBRICS",
    "GLYPH_OK",
    "GLYPH_WARN",
    "GLYPH_BAD",
    "RUBRICS",
    "OP_HEADERS",
    "Rubric",
    "WatchItem",
    "format_diagnostics",
    "format_flat_summary",
    "format_node_diagnostics",
]
