"""Performance + correctness tests for ChannelsPanel after the signal-connection fix.

The original bug: ``ChannelsPanel._add_row`` connected ``QTableWidget.itemChanged``
to ``self._on_item_changed`` once *per row*. With N consensus channels the slot
fired N times for every checkbox click, which made the panel feel sluggish on
post-consensus datasets with many channels.

These tests verify:

1. The signal is connected exactly once, regardless of row count.
2. Toggling a checkbox triggers exactly one image projection (not N).
3. The image cache prevents re-projection on repeated toggle.
"""

from __future__ import annotations

import pytest

pytest.importorskip("napari", reason="napari is required for widget tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")


@pytest.fixture
def viewer(make_napari_viewer):
    yield make_napari_viewer()


def _aligned_dataset(synth_centroided):
    """Run the recommended pipeline; return a PeakMatrix-backed MSIDataset."""
    from dataclasses import replace as drep

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


def test_item_changed_signal_connected_exactly_once(qtbot, viewer, synth_centroided):
    """The ``itemChanged`` signal must be connected to ``_on_item_changed`` once,
    regardless of how many channel rows exist.

    Implementation detail: ``QObject.receivers`` returns the count of connected
    slots for a given signal. With the bug, this would be N+1 (once per row). With
    the fix, it's always 1.
    """
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_dataset(synth_centroided)
    s = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=s)
    qtbot.addWidget(panel)
    s.set_dataset(aligned)
    # 6 summary projections + 5 consensus channels (synth has 5 planted peaks).
    assert panel._table.rowCount() >= 6  # noqa: SLF001

    n_connections = panel._table.receivers(panel._table.itemChanged)
    assert n_connections == 1, (
        f"itemChanged should be connected exactly once; got {n_connections}. "
        "If this is N+1 where N=channel count, the per-row connection bug is back."
    )


def test_toggle_visible_does_one_projection_not_n(qtbot, viewer, synth_centroided):
    """Toggling a single checkbox must trigger ``_compute_row_image`` exactly once.

    With the bug, the slot fired N times and projected the image N times — every
    checkbox click multiplied work by the channel count.
    """
    from unittest.mock import patch

    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_dataset(synth_centroided)
    s = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=s)
    qtbot.addWidget(panel)
    s.set_dataset(aligned)

    # Find a channel row (not summary).
    channel_rows = [
        i for i, r in enumerate(panel._rows) if r.kind == "channel"  # noqa: SLF001
    ]
    assert channel_rows, "expected at least one consensus channel row"
    row_idx = channel_rows[0]

    real_compute = panel._compute_row_image  # noqa: SLF001

    def _wrapped(row, ds):
        return real_compute(row, ds)

    with patch.object(panel, "_compute_row_image", side_effect=_wrapped) as spy:  # noqa: SLF001
        from qtpy.QtCore import Qt as _Qt
        item = panel._table.item(row_idx, 0)  # noqa: SLF001
        item.setCheckState(_Qt.CheckState.Checked)

    # Exactly one projection — first time the row was made visible.
    assert spy.call_count == 1, (
        f"toggling one checkbox should compute the image once; got {spy.call_count}"
    )


def test_re_toggle_uses_cached_image(qtbot, viewer, synth_centroided):
    """Hiding then re-showing a channel must reuse the cached image, not re-project.

    The image cache is keyed by row index and cleared only on dataset change.
    """
    from unittest.mock import patch

    from qtpy.QtCore import Qt
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_dataset(synth_centroided)
    s = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=s)
    qtbot.addWidget(panel)
    s.set_dataset(aligned)

    channel_rows = [
        i for i, r in enumerate(panel._rows) if r.kind == "channel"  # noqa: SLF001
    ]
    row_idx = channel_rows[0]

    # First toggle: image gets computed and cached.
    item = panel._table.item(row_idx, 0)  # noqa: SLF001
    item.setCheckState(Qt.CheckState.Checked)
    assert row_idx in panel._row_image_cache  # noqa: SLF001

    # Hide.
    item.setCheckState(Qt.CheckState.Unchecked)
    # Cache survives hide.
    assert row_idx in panel._row_image_cache  # noqa: SLF001

    # Show again: should NOT re-compute the image.
    real_compute = panel._compute_row_image  # noqa: SLF001
    with patch.object(panel, "_compute_row_image", side_effect=real_compute) as spy:  # noqa: SLF001
        item.setCheckState(Qt.CheckState.Checked)
    assert spy.call_count == 0, (
        f"re-showing a previously-shown channel should reuse the cached image; "
        f"compute was called {spy.call_count} times"
    )


def test_dataset_change_clears_image_cache(qtbot, viewer, synth_centroided):
    """Loading a new dataset must invalidate the image cache."""
    from qtpy.QtCore import Qt
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _aligned_dataset(synth_centroided)
    s = MsiSession()
    panel = ChannelsPanel(napari_viewer=viewer, session=s)
    qtbot.addWidget(panel)
    s.set_dataset(aligned)

    channel_rows = [
        i for i, r in enumerate(panel._rows) if r.kind == "channel"  # noqa: SLF001
    ]
    item = panel._table.item(channel_rows[0], 0)  # noqa: SLF001
    item.setCheckState(Qt.CheckState.Checked)
    assert panel._row_image_cache  # noqa: SLF001 — populated

    # Re-set the same dataset (simulates a reload). Cache should be cleared
    # and rebuilt fresh.
    s.set_dataset(aligned)
    assert not panel._row_image_cache, (  # noqa: SLF001
        "image cache must be invalidated when the dataset changes"
    )
