"""Tests for the post-feedback UI polish: layer naming, contrast, tooltips, defaults,
threaded execution, shared session.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari", reason="napari required for these tests")
pytest.importorskip("pytestqt", reason="pytest-qt required for these tests")


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


# ---- viz helpers ----------------------------------------------------------------


def test_projection_layer_name_is_descriptive():
    from dapple.viz.projections import PROJECTIONS, projection_layer_name

    spec = next(s for s in PROJECTIONS if s.kind == "tic")
    assert projection_layer_name(spec, "jerboa-100825") == "jerboa-100825 · TIC"
    spec_pc = next(s for s in PROJECTIONS if s.kind == "peak_count")
    assert (
        projection_layer_name(spec_pc, "root1-total ion count")
        == "root1-total ion count · peak count"
    )


def test_channel_layer_name_includes_mz_and_prevalence():
    from dapple.viz.projections import channel_layer_name

    assert (
        channel_layer_name("jerboa-100825", 250.1234, 0.91)
        == "jerboa-100825 · m/z 250.1234 · prev 91%"
    )
    # Without prevalence — used when the dataset hasn't been aligned yet.
    assert channel_layer_name("ds", 100.0) == "ds · m/z 100.0000"


def test_percentile_contrast_handles_constant_image():
    from dapple.viz.projections import percentile_contrast

    img = np.full((10, 10), 14510.7, dtype=np.float32)
    lo, hi = percentile_contrast(img)
    # Constant filled image → both bounds are the constant (within FP tolerance);
    # we widen the upper bound by ε to keep napari happy.
    assert lo == pytest.approx(14510.7, rel=1e-5)
    assert hi > lo


def test_percentile_contrast_ignores_zero_background():
    from dapple.viz.projections import percentile_contrast

    img = np.zeros((20, 20), dtype=np.float32)
    img[5:10, 5] = 1.0  # 5 dim pixels
    img[5:10, 6] = 100.0
    img[5:10, 7] = 200.0  # 5 bright pixels
    lo, hi = percentile_contrast(img)
    # 1st-99th percentile on non-zero values [1×5, 100×5, 200×5]: lo near 1, hi near 200.
    assert 0 < lo <= 5.0, f"lo={lo}: expected ~1.0 from the 1st percentile of non-zero pixels"
    assert hi >= 100.0


# ---- OpParams field metadata ----------------------------------------------------


def test_opparams_fields_carry_labels_and_help():
    from dataclasses import fields

    from dapple.ops.base import field_help, field_label
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.ops.normalize import NormalizeParams
    from dapple.ops.peak_pick import SnrPeakPickParams
    from dapple.ops.reference_ions import ReferenceIonsParams
    from dapple.ops.tolerance import EmpiricalToleranceParams

    for params_cls in (
        ReferenceIonsParams,
        EmpiricalToleranceParams,
        NormalizeParams,
        SnrPeakPickParams,
        KdeConsensusParams,
    ):
        for f in fields(params_cls):
            label = field_label(f)
            help_text = field_help(f)
            assert label and label != f.name, (
                f"{params_cls.__name__}.{f.name}: label is just the field name; add "
                f"`field(metadata={{'label': ...}})`"
            )
            assert help_text, (
                f"{params_cls.__name__}.{f.name}: help text is empty; add "
                f"`field(metadata={{'help': ...}})`"
            )


# ---- LoadPage starts with peak_count, not TIC ----------------------------------


def test_loadpage_initial_layer_is_peak_count(qtbot, viewer, synth_centroided):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import LoadPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    page: LoadPage = wiz.page(0)  # type: ignore[assignment]
    page._path_edit.setText(str(synth_centroided))  # noqa: SLF001
    page._on_load()  # noqa: SLF001
    # Peak-count, not TIC.
    names = [layer.name for layer in viewer.layers]
    assert any("peak count" in n for n in names), names
    assert not any(n.endswith("· TIC") for n in names), names


# ---- WorkflowPage tooltips and defaults ----------------------------------------


def test_workflow_card_uses_field_metadata_for_labels_and_tooltips(
    qtbot, viewer, synth_centroided
):
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

    # The "consensus" card should have an input named `bandwidth_ppm` whose tooltip
    # contains the help text from the dataclass metadata, not the bare field name.
    consensus_card = next(c for c in page._cards if c._node.id == "consensus")  # noqa: SLF001
    bandwidth_widget = consensus_card._inputs["bandwidth_ppm"]  # noqa: SLF001
    tooltip = bandwidth_widget.toolTip()
    assert "kernel" in tooltip.lower() or "bandwidth" in tooltip.lower()
    assert len(tooltip) > 40, f"tooltip looks empty/short: {tooltip!r}"


def test_workflow_page_reset_to_defaults_clears_overrides(
    qtbot, viewer, synth_centroided
):
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

    # Mutate one node's parameters via its widget.
    consensus_card = next(c for c in page._cards if c._node.id == "consensus")  # noqa: SLF001
    bw_widget = consensus_card._inputs["bandwidth_ppm"]  # noqa: SLF001
    original = bw_widget.value()
    bw_widget.setValue(original * 2)
    assert bw_widget.value() != original

    # Click "Reset all to defaults". The bandwidth widget on a freshly built card
    # should now read the default value again.
    page._reset_btn.click()  # noqa: SLF001
    consensus_card = next(c for c in page._cards if c._node.id == "consensus")  # noqa: SLF001
    bw_widget = consensus_card._inputs["bandwidth_ppm"]  # noqa: SLF001
    assert bw_widget.value() == pytest.approx(original)


def test_node_card_per_card_reset(qtbot, viewer, synth_centroided):
    """Each card's own 'Reset' button restores only its node's params."""
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
    card = next(c for c in page._cards if c._node.id == "ref")  # noqa: SLF001
    coarse_widget = card._inputs["coarse_tol_ppm"]  # noqa: SLF001
    original = coarse_widget.value()
    coarse_widget.setValue(original * 3)
    assert coarse_widget.value() != original
    card._reset()  # noqa: SLF001
    assert coarse_widget.value() == pytest.approx(original)


# ---- Wizard size policies + sample_type drop -----------------------------------


def test_wizard_can_shrink_horizontally(qtbot, viewer):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    # The minimum width must be small enough that napari's dock can be made narrow.
    assert wiz.minimumWidth() <= 400, (
        f"WizardWidget minimum width {wiz.minimumWidth()} prevents resizing the dock "
        "narrower than 400 px"
    )


def test_paramspage_does_not_expose_sample_type(qtbot, viewer, synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import ParamsPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    wiz.session.set_dataset(read_imzml(synth_centroided))
    page: ParamsPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), ParamsPage)
    )
    page.initializePage()
    # sample_type intentionally absent: no current operator consumes it.
    assert "sample_type" not in page._inputs  # noqa: SLF001


def test_paramspage_validate_preserves_sample_type_from_dataset(
    qtbot, viewer, synth_centroided
):
    """Sidecars / spec.xml may set sample_type; validatePage must not erase it."""
    from dataclasses import replace as drep

    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.wizard import ParamsPage, WizardWidget

    wiz = WizardWidget(napari_viewer=viewer, session=MsiSession())
    qtbot.addWidget(wiz)
    ds = read_imzml(synth_centroided)
    ds = drep(ds, metadata=drep(ds.metadata, sample_type="tissue"))
    wiz.session.set_dataset(ds)

    page: ParamsPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), ParamsPage)
    )
    page.initializePage()
    assert page.validatePage()
    assert wiz.session.dataset.metadata.sample_type == "tissue"


# ---- Wizard shares default_session with other widgets --------------------------


def test_wizard_shares_default_session_by_default(qtbot, viewer):
    from dapple.widgets._session import default_session
    from dapple.widgets.wizard import WizardWidget

    wiz = WizardWidget(napari_viewer=viewer)
    qtbot.addWidget(wiz)
    assert wiz.session is default_session(), (
        "WizardWidget must use default_session() so the Hyperspectral Browser and "
        "Spectrum Panel reflect the same dataset."
    )


def test_wizard_run_pushes_post_consensus_to_browser(
    qtbot, viewer, synth_centroided, tmp_path
):
    """Critical end-to-end behaviour: when the wizard finishes the pipeline, the
    HyperspectralBrowser should populate without re-loading anything because they
    share the default session."""
    from dataclasses import replace as drep

    from dapple.io.imzml_reader import read_imzml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node
    from dapple.widgets._session import default_session
    from dapple.widgets.channels import ChannelsPanel as HyperspectralBrowser
    from dapple.widgets.wizard import RunPage, WizardWidget

    # Reset the default session so the test is independent.
    default_session().set_dataset(None)

    wiz = WizardWidget(napari_viewer=viewer)
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

    # Open the browser BEFORE the run completes. The dataset already on the session
    # is pre-consensus, so the panel shows summary projections only — no m/z channel
    # rows yet.
    from dapple.viz.projections import PROJECTIONS as _P

    browser = HyperspectralBrowser(napari_viewer=viewer)
    qtbot.addWidget(browser)
    initial_rows = browser._table.rowCount()  # noqa: SLF001
    assert initial_rows == len(_P), (
        f"pre-run panel should show {len(_P)} summary rows, got {initial_rows}"
    )

    # Now run.
    run_page: RunPage = next(
        wiz.page(i) for i in range(7) if isinstance(wiz.page(i), RunPage)
    )
    run_page._on_run()  # noqa: SLF001
    qtbot.waitUntil(lambda: run_page._completed, timeout=30000)  # noqa: SLF001

    # After the run, the browser's table should grow to summaries + consensus channels.
    assert browser._table.rowCount() > initial_rows  # noqa: SLF001
    assert any(
        "m/z" in (browser._layer_name_for_row(idx))  # noqa: SLF001
        for idx in range(browser._table.rowCount())  # noqa: SLF001
    )
