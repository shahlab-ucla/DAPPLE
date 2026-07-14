"""Tests for user-feedback features: median fix on Boone-style data, log-y stems,
re-run support, rejection-budget diagnostics, and the unified ChannelsPanel.
"""

from __future__ import annotations

import os
from dataclasses import replace as drep
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("napari", reason="napari required for these tests")
pytest.importorskip("pytestqt", reason="pytest-qt required for these tests")
Qt = pytest.importorskip("qtpy.QtCore", reason="Qt binding required for these tests").Qt


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


# ---- Bug 1: median over zero-padded centroided data --------------------------


def _make_zero_padded_synth(synth_centroided: Path):
    """Inject zero-intensity sentinel entries into every pixel's peak list, mimicking
    the Xcalibur ANDI-MS centroided export pattern that produced TIC=14510 / median=0
    on the Boone DESI dataset."""
    from dapple.data.dataset import PeakList
    from dapple.io.imzml_reader import read_imzml

    ds = read_imzml(synth_centroided)
    pl: PeakList = ds.backend
    mz = np.asarray(pl.mz[:])
    intensity = np.asarray(pl.intensity[:])
    offsets = np.asarray(pl.offsets[:])
    new_mz: list[np.ndarray] = []
    new_int: list[np.ndarray] = []
    new_offsets = [0]
    for i in range(pl.n_pixels):
        a, b = int(offsets[i]), int(offsets[i + 1])
        # For each real peak, prepend two zero-intensity sentinels at slightly lower
        # m/z (the same "edge" pattern Xcalibur emits).
        seg_mz = mz[a:b]
        seg_int = intensity[a:b]
        padded_mz = np.concatenate([seg_mz - 0.001, seg_mz - 0.0005, seg_mz])
        padded_int = np.concatenate(
            [np.zeros_like(seg_int), np.zeros_like(seg_int), seg_int]
        )
        new_mz.append(padded_mz)
        new_int.append(padded_int)
        new_offsets.append(new_offsets[-1] + padded_mz.size)
    new_pl = PeakList(
        mz=np.concatenate(new_mz).astype(np.float64, copy=False),
        intensity=np.concatenate(new_int).astype(np.float32, copy=False),
        offsets=np.asarray(new_offsets, dtype=np.int64),
        n_pixels=pl.n_pixels,
    )
    return ds.with_backend(new_pl)


def test_median_projection_skips_zero_padded_entries(synth_centroided):
    """With 2/3 of every pixel's intensities being vendor-zero sentinels, the median
    projection must still compute over real peaks (non-zero), not collapse to 0."""
    ds = _make_zero_padded_synth(synth_centroided)
    img = ds.project("median")
    assert (img > 0).all(), "median projection should not be all zeros"


def test_rms_projection_skips_zero_padded_entries(synth_centroided):
    ds = _make_zero_padded_synth(synth_centroided)
    img = ds.project("rms")
    assert (img > 0).all(), "RMS projection should not be all zeros either"


def test_mean_projection_skips_zero_padded_entries(synth_centroided):
    ds = _make_zero_padded_synth(synth_centroided)
    # Mean is reachable through .project("mean_mz") only indirectly; test the
    # underlying reducer directly.
    means = ds.backend.per_pixel_reduce("mean")
    assert (means > 0).all()


@pytest.mark.real_data
@pytest.mark.skipif(
    os.environ.get("MSI_REAL_DATA", "0") != "1", reason="real-data only"
)
def test_real_boone_median_no_longer_zero(real_dataset_dir: Path):
    """Regression: on the Boone DESI dataset the median projection used to return 0
    everywhere. After the fix it should be populated for every pixel."""
    from dapple.io.cdf_image_reader import read_cdf_image

    boone = real_dataset_dir / "Boone cdf"
    if not boone.exists():
        pytest.skip(str(boone))
    ds = read_cdf_image(boone)
    img = ds.project("median")
    nz = img[img > 0]
    assert nz.size == 8680, f"expected 8680 populated medians, got {nz.size}"


# ---- Bug 2: log-y SpectrumPanel stems disappear ------------------------------


def test_spectrum_panel_log_y_does_not_lose_stems(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.spectrum import SpectrumPanel

    ds = read_imzml(synth_centroided)
    session = MsiSession()
    w = SpectrumPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    session.set_dataset(ds)
    session.set_selected_pixel((1, 1))
    items_linear = list(w._plot.listDataItems())  # noqa: SLF001
    assert items_linear, "linear-y plot should produce at least one stem item"
    # Toggle log-y on; the panel must re-render.
    w._log_check.setChecked(True)  # noqa: SLF001
    items_log = list(w._plot.listDataItems())  # noqa: SLF001
    assert items_log, (
        "log-y stems must remain visible: previously the baseline of 0.0 collapsed to "
        "log(0)=-inf and the whole curve disappeared"
    )


# ---- Re-run support ---------------------------------------------------------


def test_wizard_remembers_pipeline_when_revisiting_workflow(
    qtbot, viewer, synth_centroided
):
    """Going back to WorkflowPage after editing a parameter must NOT wipe the user's
    changes by re-running recommend_pipeline."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import WizardWidget, WorkflowPage

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    wiz.session.set_dataset(read_imzml(synth_centroided))
    page: WorkflowPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), WorkflowPage)
    )
    page.initializePage()
    consensus_card = next(c for c in page._cards if c._node.id == "consensus")  # noqa: SLF001
    bw_widget = consensus_card._inputs["bandwidth_ppm"]  # noqa: SLF001
    bw_widget.setValue(123.456)
    page.validatePage()  # commit edits into _proposed_pipeline

    # Simulate the user clicking Back: initializePage runs again.
    page.initializePage()
    consensus_card = next(c for c in page._cards if c._node.id == "consensus")  # noqa: SLF001
    bw_widget = consensus_card._inputs["bandwidth_ppm"]  # noqa: SLF001
    assert bw_widget.value() == pytest.approx(123.456), (
        "WorkflowPage.initializePage on Back navigation must preserve user edits — "
        "previously it called recommend_pipeline which clobbered them"
    )


def test_run_page_disables_save_buttons_when_pipeline_changes(
    qtbot, viewer, synth_centroided
):
    """After Run completes, going back, tweaking a param, and revisiting RunPage must
    flip the Save buttons off until the user re-runs."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget, WorkflowPage

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    ds = read_imzml(synth_centroided)
    wiz.session.set_dataset(ds)
    wiz.set_input_snapshot(ds)
    p = recommend_pipeline(ds.metadata)
    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    wiz._proposed_pipeline = drep(p, nodes=tuple(nodes))  # noqa: SLF001

    run_page: RunPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), RunPage)
    )
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001
    # Save buttons are enabled.
    assert run_page._save_spec_btn.isEnabled()  # noqa: SLF001

    # Now mutate the pipeline (as if the user went Back and edited).
    new_consensus = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=999.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    wiz._proposed_pipeline = drep(  # noqa: SLF001
        wiz._proposed_pipeline,  # noqa: SLF001
        nodes=tuple([*wiz._proposed_pipeline.nodes[:-1], new_consensus]),  # noqa: SLF001
    )
    run_page.initializePage()
    assert not run_page._save_spec_btn.isEnabled(), (
        "save buttons must be disabled after the user changes parameters — "
        "saving stale results would silently produce wrong outputs"
    )


def test_pipeline_runner_caching_reuses_unchanged_nodes(synth_centroided):
    """When the user changes only a downstream parameter, upstream nodes hit the cache
    rather than re-running."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import PipelineRunner, recommend_pipeline
    from dapple.pipeline.pipeline import Node

    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p1 = drep(p, nodes=tuple(nodes))

    runner = PipelineRunner()
    seen: list[str] = []
    runner.run(p1, ds, on_node_done=lambda nid, _r: seen.append(nid))
    # Synth fixture is tof_reflectron + maldi (no tissue), so the recommended
    # chain has the five core nodes plus a recalibration step.
    assert set(seen) == {n.id for n in p1.nodes}
    assert {"ref", "tol", "norm", "pick", "consensus"}.issubset(seen)

    # Tweak only the consensus parameter; the first 4 nodes must be served from cache.
    new_consensus = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=10.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p2 = drep(p1, nodes=tuple([*p1.nodes[:-1], new_consensus]))
    seen.clear()
    runner.run(p2, ds, on_node_done=lambda nid, _r: seen.append(nid))
    assert seen == ["consensus"], (
        f"only the modified node should re-execute; got {seen}"
    )


# ---- Rejection-budget diagnostics on KdeConsensusAlignment -------------------


def test_kde_consensus_diagnostic_carries_pre_filter_arrays(synth_centroided):
    """The KDE consensus operator must record per-candidate arrays (pre-filter) so a
    threshold explorer can answer 'how many peaks would survive at threshold X' without
    re-running."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
    from dapple.ops.normalize import MedianNormalize
    from dapple.ops.peak_pick import SnrPeakPick
    from dapple.ops.reference_ions import DetectReferenceIons
    from dapple.ops.tolerance import (
        EmpiricalToleranceFromReferenceIons,
        EmpiricalToleranceParams,
    )

    ds = read_imzml(synth_centroided)
    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(
        ds, DetectReferenceIons().default_params(ds.metadata), rng=rng
    ).dataset
    ds = EmpiricalToleranceFromReferenceIons().apply(
        ds,
        EmpiricalToleranceParams(alpha=0.01, bootstrap_B=20, block_bootstrap=False),
        rng=rng,
    ).dataset
    ds = MedianNormalize().apply(
        ds, MedianNormalize().default_params(ds.metadata), rng=rng
    ).dataset
    ds = SnrPeakPick().apply(ds, SnrPeakPick().default_params(ds.metadata), rng=rng).dataset
    op = KdeConsensusAlignment()
    result = op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        rng=rng,
    )
    diag = result.diagnostics[0]

    # Summary records the rejection budget by stage.
    for key in (
        "n_consensus_peaks",
        "n_local_maxima_total",
        "n_rejected_by_prominence",
        "n_post_prominence",
        "n_rejected_by_prevalence",
    ):
        assert key in diag.summary, f"missing rejection-budget key: {key!r}"

    # Payload exposes per-candidate arrays for re-thresholding.
    payload = diag.payload or {}
    for key in (
        "all_local_max_density",
        "prominence_threshold_value",
        "all_candidate_mz",
        "all_candidate_prevalence",
        "all_candidate_max_intensity",
    ):
        assert key in payload, f"missing rejection-budget array: {key!r}"

    # Sanity: the per-candidate prevalence should be the same length as
    # all_candidate_mz, and re-thresholding by min_prevalence should give the same
    # surviving count the operator reported.
    p = payload["all_candidate_prevalence"]
    threshold = float(diag.summary["min_prevalence_threshold"])
    surviving = int((p >= threshold).sum())
    assert surviving == int(diag.summary["n_consensus_peaks"]), (
        f"recomputed surviving count ({surviving}) != reported "
        f"({int(diag.summary['n_consensus_peaks'])})"
    )


# ---- ChannelsPanel — unified summaries + hyperspectral rendering -------------


def _aligned_synth(synth_centroided):
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import PipelineRunner, recommend_pipeline
    from dapple.pipeline.pipeline import Node

    from dapple.io.imzml_reader import read_imzml

    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))
    return PipelineRunner().run(p, ds).output


def test_channels_panel_lists_summaries_before_alignment(
    qtbot, viewer, synth_centroided
):
    from dapple.io.imzml_reader import read_imzml
    from dapple.viz.projections import PROJECTIONS
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(read_imzml(synth_centroided))
    # Pre-alignment: only summary rows, no consensus channels.
    assert panel._table.rowCount() == len(PROJECTIONS)  # noqa: SLF001


def test_channels_panel_lists_summaries_and_channels_after_alignment(
    qtbot, viewer, synth_centroided
):
    from dapple.viz.projections import PROJECTIONS
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_synth(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    expected = len(PROJECTIONS) + aligned.backend.n_peaks
    assert panel._table.rowCount() == expected  # noqa: SLF001


def test_channels_panel_per_row_lut_changes_layer_colormap(
    qtbot, viewer, synth_centroided
):
    """Changing the LUT dropdown for a row updates that layer's colormap directly."""
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_synth(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    # Tick row 0 (first summary).
    panel._table.item(0, 0).setCheckState(Qt.CheckState.Checked)  # noqa: SLF001
    layer_name = panel._layer_name_for_row(0)  # noqa: SLF001
    assert layer_name in viewer.layers
    # Change LUT to "magma" and confirm layer.colormap follows.
    lut_widget = panel._row_widgets[0]["lut"]  # noqa: SLF001
    lut_widget.setCurrentText("magma")
    assert viewer.layers[layer_name].colormap.name in {"magma", "Magma"} or (
        getattr(viewer.layers[layer_name].colormap, "colormap", None) is not None
    )


def test_channels_panel_blending_dropdown_propagates_to_visible_layers(
    qtbot, viewer, synth_centroided
):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_synth(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    # Show first two rows.
    panel._table.item(0, 0).setCheckState(Qt.CheckState.Checked)  # noqa: SLF001
    panel._table.item(1, 0).setCheckState(Qt.CheckState.Checked)  # noqa: SLF001
    # Switch blending mode.
    panel._blending_combo.setCurrentText("translucent")  # noqa: SLF001
    for idx in range(2):
        name = panel._layer_name_for_row(idx)  # noqa: SLF001
        blend = viewer.layers[name].blending
        # napari versions vary: blending may be a string or an enum.
        blend_str = blend.value if hasattr(blend, "value") else str(blend)
        assert blend_str == "translucent", f"row {idx} blending = {blend_str!r}"


def test_channels_panel_show_top_n_only_picks_channels_not_summaries(
    qtbot, viewer, synth_centroided
):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_synth(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    panel._top_n_spin.setValue(3)  # noqa: SLF001
    panel._show_top_btn.click()  # noqa: SLF001
    visible = panel.visible_layer_names()
    assert all("m/z" in n for n in visible), visible
    assert len(visible) == 3


def test_channels_panel_hide_all_clears_layers(qtbot, viewer, synth_centroided):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_synth(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    panel._top_n_spin.setValue(3)  # noqa: SLF001
    panel._show_top_btn.click()  # noqa: SLF001
    assert any("m/z" in layer.name for layer in viewer.layers)
    panel._hide_all_btn.click()  # noqa: SLF001
    assert not any("m/z" in layer.name for layer in viewer.layers)


def test_channels_panel_managed_layers_swept_on_dataset_change(
    qtbot, viewer, synth_centroided
):
    """When the dataset changes, layers we created get cleaned up automatically."""
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_synth(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    panel._table.item(0, 0).setCheckState(Qt.CheckState.Checked)  # noqa: SLF001
    n_before = sum(
        1
        for layer in viewer.layers
        if (layer.metadata or {}).get("dapple_managed_by") == "channels_panel"
    )
    assert n_before >= 1
    # Setting dataset to None must remove all our managed layers.
    session.set_dataset(None)
    n_after = sum(
        1
        for layer in viewer.layers
        if (layer.metadata or {}).get("dapple_managed_by") == "channels_panel"
    )
    assert n_after == 0
