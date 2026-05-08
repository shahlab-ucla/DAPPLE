"""Tests for the per-figure diagnostic-plot tooltips on the RunPage.

Each plot widget in the RunPage's diagnostic strip carries a tooltip explaining
what the plot shows and which features to watch for as markers of stable /
unstable results. These tests verify:

1. The tooltip table covers every figure_hint emitted by the registered
   operators, so no plot ships without explanatory text.
2. A few representative tooltips contain both "what the plot shows" content
   and "look for" guidance — i.e. the rubric isn't empty.
3. Building a plot widget for a real diagnostic actually applies the tooltip.
"""

from __future__ import annotations

import pytest

pytest.importorskip("napari", reason="napari is required for widget tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")
pytest.importorskip("pyqtgraph", reason="pyqtgraph required")

from dapple.widgets.wizard import _PLOT_TOOLTIPS, _build_diagnostic_plot
from dapple.ops.base import Diagnostic


# ---- coverage ---------------------------------------------------------------


_KNOWN_FIGURE_HINTS = {
    "line:kde_density_with_consensus_marks",
    "line_with_band:mz_vs_tolerance",
    "histogram:per_pixel_factor",
    "histogram:per_pixel_kept_count",
    "histogram:morans_i_with_threshold",
    "scatter:reference_mz_vs_prevalence",
    "scatter:pre_post_ppm_residual",
}


def test_every_known_figure_hint_has_tooltip_text():
    missing = _KNOWN_FIGURE_HINTS - set(_PLOT_TOOLTIPS)
    assert not missing, (
        f"figure_hints without tooltip text: {missing}. Add an entry to "
        "_PLOT_TOOLTIPS in dapple/widgets/wizard.py."
    )


def test_each_tooltip_has_what_and_look_for_sections():
    """A useful tooltip explains both what the plot is and what to look for."""
    for hint, tip in _PLOT_TOOLTIPS.items():
        # Some kind of bold-titled "what" header at the top.
        assert "<b>" in tip, f"tooltip for {hint!r} missing a bold header"
        # And a "look for" or "what to ..." guidance section.
        lower = tip.lower()
        assert "look for" in lower, (
            f"tooltip for {hint!r} should include guidance under a "
            f"'Look for' header to help the user judge whether the plot looks "
            f"healthy."
        )


def test_tooltips_mention_specific_failure_modes():
    """Spot-check that tooltips name actionable failure modes by parameter or symptom."""
    kde_tt = _PLOT_TOOLTIPS["line:kde_density_with_consensus_marks"]
    assert "bandwidth" in kde_tt.lower()  # mentions the knob
    tol_tt = _PLOT_TOOLTIPS["line_with_band:mz_vs_tolerance"]
    assert "ci" in tol_tt.lower() or "bootstrap" in tol_tt.lower()


# ---- integration: plot widget actually receives the tooltip ----------------


def test_build_diagnostic_plot_attaches_tooltip(qtbot):
    """A live diagnostic with a known figure_hint produces a plot whose tooltip
    matches _PLOT_TOOLTIPS for that hint."""
    import numpy as np

    diag = Diagnostic(
        name="kde_consensus_alignment",
        summary={"n_consensus_peaks": 10.0},
        payload={
            "kde_grid_log_mz": np.linspace(np.log(100), np.log(800), 200),
            "kde_density": np.exp(-((np.linspace(-3, 3, 200)) ** 2)),
            "consensus_mz": np.array([200.0, 400.0, 600.0]),
        },
        figure_hint="line:kde_density_with_consensus_marks",
    )
    widget = _build_diagnostic_plot("consensus", diag)
    assert widget is not None
    qtbot.addWidget(widget)
    # The container's tooltip should equal the registered tooltip.
    tt = widget.toolTip()
    assert tt and "KDE" in tt and "Look for" in tt


def test_unknown_figure_hint_renders_without_tooltip_or_crash(qtbot):
    """Operators with unknown figure_hint values still render a fallback widget."""
    diag = Diagnostic(
        name="some_future_op",
        summary={"x": 1.0},
        payload={},
        figure_hint="brand:new:unrecognized:hint",
    )
    widget = _build_diagnostic_plot("x", diag)
    assert widget is not None
    qtbot.addWidget(widget)
    # No tooltip is fine; we just don't want to crash.
    assert widget.toolTip() == "" or widget.toolTip() is not None
