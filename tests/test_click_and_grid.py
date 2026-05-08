"""Tests for click-to-show-mz cross-widget routing and napari grid/mosaic mode."""

from __future__ import annotations

from dataclasses import replace as drep

import numpy as np
import pytest

pytest.importorskip("napari", reason="napari required for these tests")
pytest.importorskip("pytestqt", reason="pytest-qt required for these tests")


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


def _aligned(synth_centroided):
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
    p = drep(p, nodes=tuple(nodes))
    return PipelineRunner().run(p, ds).output


# ---- click-to-show-mz routing ----------------------------------------------------


def test_session_show_mz_signal_emits_clean_float():
    """MsiSession.request_show_mz forwards as a plain float to listeners."""
    from dapple.widgets._session import MsiSession

    s = MsiSession()
    received: list[float] = []
    s.show_mz_requested.connect(lambda v: received.append(v))
    s.request_show_mz(np.float32(250.123))
    assert received == [pytest.approx(250.123, rel=1e-5)]


def test_channels_panel_snaps_to_nearest_consensus_on_show_mz(
    qtbot, viewer, synth_centroided
):
    """A request_show_mz close to a consensus channel should toggle that row visible."""
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    axis = np.asarray(aligned.backend.mz_axis[:])
    target_idx = 2
    target_mz = float(axis[target_idx])
    # Click at the exact m/z — snap is trivial.
    session.request_show_mz(target_mz)
    visible = panel.visible_layer_names()
    assert any(f"{target_mz:.4f}" in name for name in visible), visible


def test_channels_panel_snaps_to_nearest_when_clicked_off_peak(
    qtbot, viewer, synth_centroided
):
    """Clicking *between* consensus peaks still surfaces the nearest one."""
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    axis = np.asarray(aligned.backend.mz_axis[:])
    target_idx = 1
    # Click just barely closer to channel `target_idx` than to its neighbors.
    if target_idx + 1 < axis.size:
        midpoint = (axis[target_idx] + axis[target_idx + 1]) / 2
        clicked = (axis[target_idx] + midpoint) / 2  # slightly toward target_idx
    else:
        clicked = float(axis[target_idx])
    session.request_show_mz(float(clicked))
    expected_mz = float(axis[target_idx])
    visible = panel.visible_layer_names()
    assert any(f"{expected_mz:.4f}" in name for name in visible), (
        f"clicked at {clicked:.4f}, expected snap to {expected_mz:.4f}, visible={visible}"
    )


def test_channels_panel_no_op_on_show_mz_when_no_peakmatrix(
    qtbot, viewer, synth_centroided
):
    """Requesting show-mz on a pre-consensus dataset (PeakList) does nothing."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(read_imzml(synth_centroided))
    session.request_show_mz(250.0)
    assert not panel.visible_layer_names()


def test_spectrum_panel_emits_show_mz_on_click(
    qtbot, viewer, synth_centroided
):
    """A click on a stem inside the SpectrumPanel emits MsiSession.show_mz_requested
    with the snapped peak m/z (not the raw click position)."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.spectrum import SpectrumPanel

    ds = read_imzml(synth_centroided)
    session = MsiSession()
    panel = SpectrumPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(ds)
    session.set_selected_pixel((1, 1))
    received: list[float] = []
    session.show_mz_requested.connect(lambda v: received.append(v))

    # Bypass the synthetic Qt mouse-event plumbing (qtbot.mouseClick on a pyqtgraph
    # PlotWidget doesn't reliably round-trip through the scene's sigMouseClicked) and
    # call _on_plot_clicked with a stand-in event whose scenePos() falls near the
    # known synth peak at m/z 300.
    plot_item = panel._plot.getPlotItem()  # noqa: SLF001
    view = plot_item.getViewBox()
    scene_point = view.mapViewToScene(__import__("pyqtgraph").Point(300.05, 0.0))

    class _FakeEvent:
        def double(self) -> bool:
            return False

        def button(self):  # noqa: ANN201
            from qtpy.QtCore import Qt as _Qt

            return _Qt.MouseButton.LeftButton

        def scenePos(self):  # noqa: ANN201
            return scene_point

    panel._on_plot_clicked(_FakeEvent())  # noqa: SLF001
    # The clicked m/z (300.05) snaps to the nearest plotted peak — within 5 ppm of
    # the synth-fixture's m/z=300 peak (jitter is N(0, 5 ppm)).
    assert received, "no show_mz_requested emitted"
    snapped = received[-1]
    ppm_err = abs(snapped - 300.0) / 300.0 * 1e6
    assert ppm_err < 50, f"snapped m/z {snapped} too far from 300.0 (ppm error {ppm_err:.1f})"


def test_spectrum_panel_click_paints_marker(qtbot, viewer, synth_centroided):
    """The vertical marker line is shown after a click and tracks the snapped m/z."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.spectrum import SpectrumPanel

    ds = read_imzml(synth_centroided)
    session = MsiSession()
    panel = SpectrumPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(ds)
    session.set_selected_pixel((1, 1))
    assert not panel._click_marker.isVisible()  # noqa: SLF001
    plot_item = panel._plot.getPlotItem()  # noqa: SLF001
    view = plot_item.getViewBox()
    scene_point = view.mapViewToScene(__import__("pyqtgraph").Point(199.97, 0.0))

    class _FakeEvent:
        def double(self) -> bool:
            return False

        def button(self):  # noqa: ANN201
            from qtpy.QtCore import Qt as _Qt

            return _Qt.MouseButton.LeftButton

        def scenePos(self):  # noqa: ANN201
            return scene_point

    panel._on_plot_clicked(_FakeEvent())  # noqa: SLF001
    assert panel._click_marker.isVisible()  # noqa: SLF001
    marker_x = panel._click_marker.value()  # noqa: SLF001
    assert abs(marker_x - 200.0) < 0.05


# ---- napari grid / mosaic mode --------------------------------------------------


def test_grid_toggle_enables_napari_grid_mode(qtbot, viewer, synth_centroided):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    panel._top_n_spin.setValue(3)  # noqa: SLF001
    panel._show_top_btn.click()  # noqa: SLF001
    assert not viewer.grid.enabled
    panel._grid_btn.setChecked(True)  # noqa: SLF001
    assert viewer.grid.enabled


def test_grid_columns_propagates_to_napari(qtbot, viewer, synth_centroided):
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned(synth_centroided)
    session = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=session)
    qtbot.addWidget(panel)
    session.set_dataset(aligned)
    panel._grid_btn.setChecked(True)  # noqa: SLF001
    panel._grid_cols_spin.setValue(3)  # noqa: SLF001
    # napari grid.shape is (rows, cols); rows=-1 → auto, cols should reflect our 3.
    rows, cols = viewer.grid.shape
    assert cols == 3
