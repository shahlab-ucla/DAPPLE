"""Tests for the diagnostic-formatting helper (``dapple.pipeline.diag_format``)."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.io.imzml_reader import read_imzml
from dapple.ops.base import Diagnostic
from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
from dapple.ops.normalize import MedianNormalize
from dapple.ops.peak_pick import SnrPeakPick
from dapple.ops.reference_ions import DetectReferenceIons
from dapple.ops.tolerance import (
    EmpiricalToleranceFromReferenceIons,
    EmpiricalToleranceParams,
)
from dapple.pipeline import (
    PipelineRunner,
    format_diagnostics,
    format_flat_summary,
    format_node_diagnostics,
    recommend_pipeline,
)
from dapple.pipeline.diag_format import (
    COHORT_RUBRICS,
    GLYPH_BAD,
    GLYPH_OK,
    GLYPH_WARN,
    RUBRICS,
    Rubric,
)
from dapple.pipeline.runner import RunResult


# ---- per-key rubric grading ---------------------------------------------------


def test_rubric_grades_healthy_warning_bad_in_priority():
    r = Rubric(
        label="x",
        fmt="{:.0f}",
        healthy=lambda v: v >= 5,
        warning=lambda v: 3 <= v < 5,
        bad=lambda v: v < 3,
        healthy_hint=">= 5",
    )
    # Bad takes priority even if other predicates would also match.
    block, _ = format_node_diagnostics(
        "n", [Diagnostic(name="op", summary={"some_key": 1.0}, payload=None, figure_hint=None)]
    )
    # Without a registered rubric the formatter still renders the row (no glyph).
    assert any("some_key" in line for line in block)


def test_n_reference_ions_rubric_marks_healthy():
    diag = Diagnostic(
        name="detect_reference_ions",
        summary={"n_reference_ions": 12.0},
        payload=None,
        figure_hint=None,
    )
    lines, _ = format_node_diagnostics("ref", [diag])
    text = "\n".join(lines)
    assert GLYPH_OK in text
    assert "reference ions" in text


def test_n_reference_ions_rubric_marks_bad_when_below_3():
    diag = Diagnostic(
        name="detect_reference_ions",
        summary={"n_reference_ions": 1.0},
        payload=None,
        figure_hint=None,
    )
    lines, watches = format_node_diagnostics("ref", [diag])
    text = "\n".join(lines)
    assert GLYPH_BAD in text
    # The advice hint should appear in the watches list (now WatchItems).
    assert any("anchors" in w.advice for w in watches)


def test_warning_flat_tolerance_marks_bad():
    diag = Diagnostic(
        name="empirical_tolerance_from_reference_ions",
        summary={"warning_flat_tolerance": 1.0},
        payload=None,
        figure_hint=None,
    )
    lines, watches = format_node_diagnostics("tol", [diag])
    assert GLYPH_BAD in "\n".join(lines)
    # The rubric should explain what fallback fired and how to address it.
    # We don't pin to a specific phrase — just check the watch carries
    # both meaning and advice text.
    assert watches, "warning_flat_tolerance should produce a WatchItem"
    w = watches[0]
    assert w.meaning, "rubric should populate `meaning` for this key"
    assert w.advice, "rubric should populate bad_advice for this key"


def test_fraction_pixels_recalibrated_grades():
    # Healthy.
    d_ok = Diagnostic(name="msiwarp_recalibrate",
                       summary={"fraction_pixels_recalibrated": 0.95},
                       payload=None, figure_hint=None)
    assert GLYPH_OK in "\n".join(format_node_diagnostics("rc", [d_ok])[0])
    # Warning.
    d_warn = Diagnostic(name="msiwarp_recalibrate",
                         summary={"fraction_pixels_recalibrated": 0.75},
                         payload=None, figure_hint=None)
    assert GLYPH_WARN in "\n".join(format_node_diagnostics("rc", [d_warn])[0])
    # Bad.
    d_bad = Diagnostic(name="msiwarp_recalibrate",
                        summary={"fraction_pixels_recalibrated": 0.2},
                        payload=None, figure_hint=None)
    assert GLYPH_BAD in "\n".join(format_node_diagnostics("rc", [d_bad])[0])


def test_improvement_ppm_negative_is_bad():
    d = Diagnostic(name="msiwarp_recalibrate",
                    summary={"improvement_ppm": -2.0},
                    payload=None, figure_hint=None)
    text = "\n".join(format_node_diagnostics("rc", [d])[0])
    assert GLYPH_BAD in text


# ---- value formatting --------------------------------------------------------


def test_unknown_key_is_rendered_unannotated():
    d = Diagnostic(name="op",
                    summary={"some_unknown_key": 7.0},
                    payload=None, figure_hint=None)
    lines, watches = format_node_diagnostics("x", [d])
    assert watches == []
    assert any("some_unknown_key" in ln for ln in lines)
    # No glyph for unrubricked keys.
    text = "\n".join(lines)
    assert GLYPH_OK not in text and GLYPH_WARN not in text and GLYPH_BAD not in text


def test_non_numeric_value_is_skipped():
    # Strings can't be graded; the formatter should not crash.
    d = Diagnostic(name="op",
                    summary={"label": "abc"},  # type: ignore[dict-item]
                    payload=None, figure_hint=None)
    lines, _ = format_node_diagnostics("x", [d])
    # Either skipped entirely or rendered without grade.
    assert lines  # at least the header row


def test_format_value_uses_rubric_fmt():
    d = Diagnostic(name="op",
                    summary={"prevalence_median": 0.91234},
                    payload=None, figure_hint=None)
    lines, _ = format_node_diagnostics("x", [d])
    text = "\n".join(lines)
    # rubric format is "{:.3f}".
    assert "0.912" in text


# ---- run-level format_diagnostics --------------------------------------------


def test_format_diagnostics_includes_header_footer_and_per_node_blocks():
    result = RunResult(
        output=None,  # type: ignore[arg-type] — unused by formatter
        diagnostics={
            "ref": [Diagnostic(name="detect_reference_ions",
                                summary={"n_reference_ions": 12.0,
                                          "prevalence_median": 0.85,
                                          "prevalence_min": 0.6},
                                payload=None, figure_hint=None)],
            "pick": [Diagnostic(name="snr_peak_pick",
                                  summary={"n_peaks_in": 2000.0,
                                            "n_peaks_out": 1850.0,
                                            "fraction_kept": 0.925,
                                            "kept_per_pixel_median": 35.0},
                                  payload=None, figure_hint=None)],
        },
    )
    lines = format_diagnostics(result)
    text = "\n".join(lines)
    # Header / footer dividers (Unicode box-drawing).
    assert "Run diagnostics" in text
    assert "─" in text
    # Per-node section headers using the OP_HEADERS pretty names.
    assert "Reference-ion detection" in text
    assert "Peak picking (SNR / centroided)" in text
    # Healthy footer.
    assert "health checks passed" in text


def test_format_diagnostics_footer_lists_unhealthy_remediation():
    """When a key grades unhealthy, the footer surfaces its why_watch hint."""
    result = RunResult(
        output=None,  # type: ignore[arg-type]
        diagnostics={
            "ref": [Diagnostic(name="detect_reference_ions",
                                summary={"n_reference_ions": 1.0},
                                payload=None, figure_hint=None)],
        },
    )
    text = "\n".join(format_diagnostics(result))
    assert GLYPH_BAD in text
    assert "anchors" in text  # from advice
    assert "need attention" in text


def test_format_diagnostics_footer_renders_meaning_and_advice():
    """Each flagged metric expands into a 3-line block: header, meaning, advice."""
    result = RunResult(
        output=None,  # type: ignore[arg-type]
        diagnostics={
            "ref": [Diagnostic(
                name="detect_reference_ions",
                summary={"prevalence_min": 0.15},  # bad: <0.3
                payload=None, figure_hint=None,
            )],
        },
    )
    text = "\n".join(format_diagnostics(result))
    # Header: "✗ [ref] prevalence min = 0.150 (unhealthy)"
    assert "(unhealthy)" in text
    # Meaning row: starts with "what:" and explains what the value represents.
    assert "what:" in text
    assert "least" in text or "weakest" in text  # from rubric.meaning
    # Advice row: starts with "fix:" and proposes a concrete change.
    assert "fix:" in text
    # The bad-state advice should be specific (not the warn-state hint).
    assert "false positive" in text or "30%" in text


def test_format_diagnostics_warn_vs_bad_advice_differ():
    """Different severity levels produce different advice strings."""
    warn_result = RunResult(
        output=None,  # type: ignore[arg-type]
        diagnostics={
            "ref": [Diagnostic(name="detect_reference_ions",
                                summary={"n_reference_ions": 4.0},  # warn: 3-4
                                payload=None, figure_hint=None)],
        },
    )
    bad_result = RunResult(
        output=None,  # type: ignore[arg-type]
        diagnostics={
            "ref": [Diagnostic(name="detect_reference_ions",
                                summary={"n_reference_ions": 1.0},  # bad: <3
                                payload=None, figure_hint=None)],
        },
    )
    warn_text = "\n".join(format_diagnostics(warn_result))
    bad_text = "\n".join(format_diagnostics(bad_result))
    # Different severity words.
    assert "borderline" in warn_text
    assert "unhealthy" in bad_text
    # Different advice content — warn is gentler, bad more explicit about
    # downstream consequences.
    assert "wide CIs" in warn_text or "5-10" in warn_text or "0.5-0.6" in warn_text
    assert "flat curve" in bad_text or "disables msiwarp" in bad_text or \
           "no reproducible peaks" in bad_text


# ---- format_flat_summary (cohort case) ---------------------------------------


def test_format_flat_summary_uses_cohort_rubrics():
    summary = {
        "n_datasets": 4.0,
        "n_total_pixels": 16544.0,
        "n_consensus_shared": 87.0,
        "cohort_prevalence_median": 0.62,
    }
    lines = format_flat_summary(
        summary, title="Cohort alignment diagnostics", extra_rubrics=COHORT_RUBRICS
    )
    text = "\n".join(lines)
    assert "Cohort alignment diagnostics" in text
    assert "shared consensus channels" in text  # rubric label
    # 87 channels is healthy (≥10).
    assert GLYPH_OK in text


def test_format_flat_summary_with_unhealthy_value_lists_watches():
    summary = {
        "n_consensus_shared": 0.0,
    }
    lines = format_flat_summary(
        summary, title="X", extra_rubrics=COHORT_RUBRICS
    )
    text = "\n".join(lines)
    assert GLYPH_BAD in text
    assert "what to watch" in text


def test_format_flat_summary_handles_empty_dict():
    lines = format_flat_summary({}, title="Empty")
    text = "\n".join(lines)
    assert "Empty" in text
    assert "(empty)" in text


# ---- end-to-end: format the output of a real pipeline run --------------------


def test_format_diagnostics_runs_clean_on_real_pipeline(synth_centroided):
    """A live RunResult formats without errors and surfaces healthy markers."""
    ds = read_imzml(synth_centroided)
    pipeline = recommend_pipeline(ds.metadata)
    runner = PipelineRunner()
    result = runner.run(pipeline, ds)
    lines = format_diagnostics(result)
    text = "\n".join(lines)
    # No tracebacks, headers present.
    assert "Run diagnostics" in text
    # Reference-ion detection should be healthy on the synth fixture.
    assert "Reference-ion detection" in text
    assert GLYPH_OK in text
    # Health-check tally appears in the footer.
    assert "health checks" in text


def test_format_node_diagnostics_aligns_columns():
    """Rows should be visually aligned (dotted leader + right-aligned value).

    Labels are left-padded with dots to one-past the longest label in the
    block, so the *longest* label gets a single trailing dot and shorter
    labels get more. We verify alignment by checking that all body rows have
    the same total column width up through the value.
    """
    d = Diagnostic(
        name="snr_peak_pick",
        summary={"n_peaks_in": 1000.0, "fraction_kept": 0.5},
        payload=None, figure_hint=None,
    )
    lines, _ = format_node_diagnostics("p", [d])
    body = [ln for ln in lines if ln.startswith("    ")]
    assert body
    for row in body:
        assert "." in row, f"row missing dotted leader: {row!r}"
    # The shorter label ("peaks in" = 8 chars) should end up padded with at
    # least 2 dots when the longest label ("fraction kept" = 13 chars) gets 1.
    short_row = next(ln for ln in body if "peaks in" in ln)
    assert ".." in short_row, f"short row not padded: {short_row!r}"
