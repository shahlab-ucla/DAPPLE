"""Smoke tests for the napari widgets via pytest-qt + napari's make_napari_viewer.

These tests instantiate each widget in a real napari viewer, drive a couple of
interactions, and verify side effects. They aren't full UI integration tests but
catch import/wiring breakage and the most common Qt-binding pitfalls.
"""

from __future__ import annotations

import numpy as np
import pytest

# Tests in this file all need a Qt application, hence the napari-test fixture.
pytest.importorskip("napari", reason="napari is required for widget tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")


@pytest.fixture
def viewer(make_napari_viewer):
    """Provide a napari Viewer for the duration of a test."""
    v = make_napari_viewer()
    yield v


def _aligned_dataset(synth_centroided):
    """Run the recommended pipeline; return the post-consensus MSIDataset."""
    from dataclasses import replace as drep

    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import PipelineRunner, Pipeline, recommend_pipeline
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
    p = drep(p, nodes=tuple(nodes))
    return PipelineRunner().run(p, ds).output, ds


def test_session_emits_dataset_changed(qtbot):
    from dapple.widgets._session import MsiSession

    s = MsiSession()
    received: list[object] = []
    s.dataset_changed.connect(lambda v: received.append(v))
    s.set_dataset(None)  # set to None
    assert received == [None]


def test_preview_widget_buttons_enable_with_dataset(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.preview import PreviewWidget

    session = MsiSession()
    w = PreviewWidget(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    # Before loading: buttons disabled.
    for btn in w._buttons.values():  # noqa: SLF001 — test inspection
        assert not btn.isEnabled()

    ds = read_imzml(synth_centroided)
    session.set_dataset(ds)
    for btn in w._buttons.values():  # noqa: SLF001
        assert btn.isEnabled()


def test_preview_widget_renders_projection_to_layer(
    qtbot, viewer, synth_centroided
):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.preview import PreviewWidget

    session = MsiSession()
    w = PreviewWidget(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    session.set_dataset(read_imzml(synth_centroided))
    # Click the TIC button.
    w._buttons["tic"].click()  # noqa: SLF001
    # Layer names now include the dataset stem: "synth_centroided · TIC".
    matching = [layer for layer in viewer.layers if "TIC" in layer.name]
    assert matching, f"no TIC layer; got {[l.name for l in viewer.layers]}"
    assert matching[0].data.shape == (5, 5)
    # Metadata records the projection kind so other widgets can introspect.
    assert matching[0].metadata.get("projection") == "tic"


def test_roi_widget_creates_shapes_layer(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.roi import SHAPES_LAYER_NAME, RoiWidget

    session = MsiSession()
    w = RoiWidget(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    session.set_dataset(read_imzml(synth_centroided))
    w._add_btn.click()  # noqa: SLF001
    assert SHAPES_LAYER_NAME in viewer.layers


def test_browser_populates_table_for_aligned_dataset(
    qtbot, viewer, synth_centroided
):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel as HyperspectralBrowser

    aligned, _ = _aligned_dataset(synth_centroided)
    session = MsiSession()
    w = HyperspectralBrowser(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    session.set_dataset(aligned)
    # The unified ChannelsPanel lists summary projections AND each consensus m/z
    # channel, so its row count is summaries + n_peaks.
    from dapple.viz.projections import PROJECTIONS as _P

    assert w._table.rowCount() == aligned.backend.n_peaks + len(_P)  # noqa: SLF001


def test_browser_show_top_n_creates_layers(qtbot, viewer, synth_centroided):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel as HyperspectralBrowser

    aligned, _ = _aligned_dataset(synth_centroided)
    session = MsiSession()
    w = HyperspectralBrowser(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    session.set_dataset(aligned)
    w._top_n_spin.setValue(3)  # noqa: SLF001
    w._show_top_btn.click()  # noqa: SLF001
    # New informative naming: "<stem> · m/z <value> · prev <pct>".
    n_channel_layers = sum(1 for layer in viewer.layers if "m/z" in layer.name)
    assert n_channel_layers >= 3


def test_spectrum_panel_renders_after_pixel_select(
    qtbot, viewer, synth_centroided
):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.spectrum import SpectrumPanel

    ds = read_imzml(synth_centroided)
    session = MsiSession()
    w = SpectrumPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(w)
    session.set_dataset(ds)
    # Simulate a pixel selection.
    session.set_selected_pixel((1, 1))
    # The plot should have at least one item now.
    items = [item for item in w._plot.listDataItems()]  # noqa: SLF001
    assert len(items) >= 1


def test_wizard_loads_dataset_through_load_page(
    qtbot, viewer, synth_centroided
):
    from dapple.widgets.wizard import LoadPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer)
    qtbot.addWidget(wiz)
    page: LoadPage = wiz.page(0)  # type: ignore[assignment]
    page._path_edit.setText(str(synth_centroided))  # noqa: SLF001
    page._on_load()  # noqa: SLF001
    assert wiz.session.dataset is not None
    assert wiz.session.dataset.n_pixels == 25


def test_wizard_recommend_pipeline_attaches_to_workflow_page(
    qtbot, viewer, synth_centroided
):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets.wizard import WizardWidget, WorkflowPage

    wiz = WizardWidget(napari_viewer=viewer)
    qtbot.addWidget(wiz)
    wiz.session.set_dataset(read_imzml(synth_centroided))
    page: WorkflowPage = next(
        wiz.page(i) for i in range(wiz.pageIds().__len__()) if isinstance(wiz.page(i), WorkflowPage)
    )  # type: ignore[assignment]
    page.initializePage()
    assert wiz._proposed_pipeline is not None  # noqa: SLF001
    # Synth fixture is tof_reflectron + maldi (no tissue tag), so the recommended
    # chain has the five core nodes plus a recalibration step. Verify the chain
    # contains everything we expect rather than locking in a specific count.
    ids = {n.id for n in wiz._proposed_pipeline.nodes}  # noqa: SLF001
    assert {"ref", "tol", "norm", "pick", "consensus"}.issubset(ids)


def test_wizard_run_page_executes_pipeline(qtbot, viewer, synth_centroided):
    from dataclasses import replace as drep

    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget

    # Use a private session so other tests don't see our changes via default_session().
    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    ds = read_imzml(synth_centroided)
    wiz.session.set_dataset(ds)
    # Override the pipeline so consensus actually finds peaks on the tiny synth.
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
    )  # type: ignore[assignment]
    run_page._on_run()  # noqa: SLF001
    # The worker runs in a background thread; wait for it to finish (≤ 30 s).
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001
    assert run_page._run_result is not None  # noqa: SLF001


def test_wizard_end_to_end_with_real_save(qtbot, viewer, synth_centroided, tmp_path):
    """Full wizard flow: load -> run -> save spec/imzml/tiff. Synthetic data only."""
    from dataclasses import replace as drep

    from dapple.io.imzml_reader import read_imzml
    from dapple.io.spec_xml import read_spec_xml
    from dapple.io.tiff_writer import read_hyperspectral_tiff_metadata
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import RunPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    ds = read_imzml(synth_centroided)
    wiz.session.set_dataset(ds)
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
    )  # type: ignore[assignment]
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001

    # Direct calls, bypassing the file dialog.
    spec_path = tmp_path / "wizard.spec.xml"
    from dapple.io.spec_xml import make_provenance, write_spec_xml

    write_spec_xml(
        spec_path,
        pipeline=wiz._proposed_pipeline,  # noqa: SLF001
        experiment_params=wiz.session.dataset.metadata,
        provenance=make_provenance(plugin_version="0.1.0.dev0", input_dataset_hash=ds.hash()),
        diagnostics=run_page._run_result.diagnostics,  # noqa: SLF001
    )
    assert spec_path.exists()
    loaded_pipe, loaded_ep, _ = read_spec_xml(spec_path)
    assert loaded_pipe.hash(input_hash=ds.hash()) == wiz._proposed_pipeline.hash(input_hash=ds.hash())  # noqa: SLF001

    from dapple.io.imzml_writer import write_imzml
    from dapple.io.tiff_writer import write_hyperspectral_tiff

    imzml_result = write_imzml(run_page._run_result.output, tmp_path / "wizard.imzML")  # noqa: SLF001
    assert imzml_result.imzml_path.exists() and imzml_result.ibd_path.exists()

    tiff_result = write_hyperspectral_tiff(run_page._run_result.output, tmp_path / "wizard.tif")  # noqa: SLF001
    assert tiff_result.tiff_path.exists() and tiff_result.csv_path.exists()
    pages = read_hyperspectral_tiff_metadata(tiff_result.tiff_path)
    assert len(pages) == tiff_result.n_pages
