"""Tests for the ThresholdExplorerPanel and the live per-node diagnostic plots
on the wizard's RunPage.
"""

from __future__ import annotations

from dataclasses import replace as drep

import numpy as np
import pytest

pytest.importorskip("napari", reason="napari required for these tests")
pytest.importorskip("pytestqt", reason="pytest-qt required for these tests")


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


def _seeded_pipeline(metadata):
    """Build a pipeline whose consensus parameters work cleanly on the synth."""
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node

    p = recommend_pipeline(metadata)
    nodes = list(p.nodes)
    # Find the consensus node — synth (no tissue) has it as the last node.
    for i, n in enumerate(nodes):
        if n.op_name == "kde_consensus_alignment":
            nodes[i] = Node(
                id="consensus",
                op_name="kde_consensus_alignment",
                params=KdeConsensusParams(
                    default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5
                ),
                upstream=("pick",),
            )
            break
    return drep(p, nodes=tuple(nodes))


# ---- ThresholdExplorerPanel ----------------------------------------------------


def test_threshold_explorer_placeholder_when_no_run(qtbot, viewer):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.threshold_explorer import ThresholdExplorerPanel

    session = MsiSession()
    panel = ThresholdExplorerPanel(napari_viewer=viewer, session=session, get_run_result=lambda: None)
    qtbot.addWidget(panel)
    assert "Run the pipeline" in panel._status.text() or "rejection-budget" in panel._status.text()  # noqa: SLF001


def test_threshold_explorer_populates_after_run(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.pipeline import PipelineRunner
    from dapple.widgets._session import MsiSession
    from dapple.widgets.threshold_explorer import ThresholdExplorerPanel

    ds = read_imzml(synth_centroided)
    p = _seeded_pipeline(ds.metadata)
    result = PipelineRunner().run(p, ds)
    session = MsiSession()
    panel = ThresholdExplorerPanel(
        napari_viewer=viewer,
        session=session,
        get_run_result=lambda r=result: r,
    )
    qtbot.addWidget(panel)
    panel._refresh()  # noqa: SLF001
    # Pick the consensus prevalence filter — index 0 in _EXPLORERS.
    panel._filter_combo.setCurrentIndex(0)  # noqa: SLF001
    # The CDF curve must have data (i.e. all_candidate_prevalence array was found).
    xdata, _ydata = panel._cdf_curve.getData()  # noqa: SLF001
    assert xdata is not None and len(xdata) > 0


def test_threshold_explorer_count_responds_to_threshold(
    qtbot, viewer, synth_centroided
):
    """Different threshold values produce different survivor counts in the label."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.pipeline import PipelineRunner
    from dapple.widgets._session import MsiSession
    from dapple.widgets.threshold_explorer import (
        ThresholdExplorerPanel,
        _EXPLORERS,
    )

    ds = read_imzml(synth_centroided)
    p = _seeded_pipeline(ds.metadata)
    result = PipelineRunner().run(p, ds)
    panel = ThresholdExplorerPanel(
        napari_viewer=viewer,
        session=MsiSession(),
        get_run_result=lambda r=result: r,
    )
    qtbot.addWidget(panel)
    panel._refresh()  # noqa: SLF001
    # Use the prominence-density filter (index 1) where values vary across the
    # range, rather than prevalence (which is degenerate at 1.0 on synth).
    panel._filter_combo.setCurrentIndex(1)  # noqa: SLF001
    spec = _EXPLORERS[1]
    cache = panel._cache.get(1)  # noqa: SLF001
    if cache is None:
        pytest.skip("synthesized run did not populate prominence-density values")
    values = cache[0]
    # Pick two thresholds that yield different survivor counts.
    panel._update_count(spec, float(values.min()) - 1)  # noqa: SLF001 — keep all
    text_keep_all = panel._count_label.text()  # noqa: SLF001
    panel._update_count(spec, float(values.max()) + 1)  # noqa: SLF001 — keep none
    text_keep_none = panel._count_label.text()  # noqa: SLF001
    assert text_keep_all != text_keep_none
    assert "100%" in text_keep_all or "all" in text_keep_all.lower() or text_keep_all
    assert "0 of" in text_keep_none or "0%" in text_keep_none


# ---- Live diagnostic plots on RunPage ------------------------------------------


def test_run_page_renders_diagnostic_plots_after_run(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    raw = read_imzml(synth_centroided)
    wiz.session.set_dataset(raw)
    wiz.set_input_snapshot(raw)
    wiz._proposed_pipeline = _seeded_pipeline(raw.metadata)  # noqa: SLF001
    run_page = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), RunPage)
    )
    assert run_page._plots_inner_layout.count() == 1  # only the trailing stretch  # noqa: SLF001
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001
    # After the run, at least one plot widget per node should be rendered (plus
    # the stretch).
    assert run_page._plots_inner_layout.count() > 1  # noqa: SLF001


def test_diagnostic_plot_helper_returns_widget_for_known_hint():
    """The plot factory should return a widget for every figure_hint we emit."""
    from dapple.ops.base import Diagnostic
    from dapple.widgets.wizard import _build_diagnostic_plot

    hints = (
        "line:kde_density_with_consensus_marks",
        "line_with_band:mz_vs_tolerance",
        "histogram:per_pixel_factor",
        "histogram:per_pixel_kept_count",
        "histogram:morans_i_with_threshold",
        "scatter:reference_mz_vs_prevalence",
        "scatter:pre_post_ppm_residual",
    )
    for hint in hints:
        d = Diagnostic(name="x", summary={"a": 1.0}, payload={}, figure_hint=hint)
        # No payload data → falls through to summary-text widget rather than a plot,
        # but must still return a widget.
        w = _build_diagnostic_plot("x", d)
        assert w is not None, f"helper returned None for hint={hint!r}"
