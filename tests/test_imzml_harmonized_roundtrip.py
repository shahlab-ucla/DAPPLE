"""Tests for imzML round-trip of PeakMatrix-backed (harmonized) datasets.

Before the fix, ``write_imzml`` flattened a PeakMatrix into per-pixel sparse
peaks and ``read_imzml`` always rebuilt a PeakList — the structural fact that
"all pixels share one m/z axis" was lost on save+load. The Channels Panel
showed no consensus rows on a reloaded file because it gates on
``isinstance(ds.backend, PeakMatrix)``.

The fix:

- ``write_imzml`` emits a marker ``<userParam name="dapple-harmonized" value="true"/>``
  on the spectrum reference group plus a sibling ``.dapple-axis.json`` sidecar.
- ``read_imzml`` detects the marker and reconstructs the PeakMatrix from the
  per-pixel sparse data + the sidecar's shared axis.

These tests verify the round-trip preserves both the backend type and the
matrix contents within tolerance, and that the Channels Panel renders consensus
rows after a reload.
"""

from __future__ import annotations

from dataclasses import replace as drep
from pathlib import Path

import numpy as np
import pytest


def _build_aligned_dataset(synth_centroided):
    """Run the recommended pipeline; return the post-consensus MSIDataset."""
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


def test_write_imzml_emits_harmonized_marker_for_peakmatrix(synth_centroided, tmp_path):
    """When writing a PeakMatrix dataset, the imzML XML must carry the marker
    user-param and a sibling sidecar file."""
    from dapple.io.imzml_writer import write_imzml

    aligned = _build_aligned_dataset(synth_centroided)
    out_base = tmp_path / "harmonized"
    result = write_imzml(aligned, out_base.with_suffix(".imzML"))

    xml = result.imzml_path.read_text(encoding="utf-8")
    assert 'name="dapple-harmonized"' in xml
    assert 'value="true"' in xml

    sidecar = result.imzml_path.with_suffix(".dapple-axis.json")
    assert sidecar.exists(), "missing .dapple-axis.json sidecar"


def test_write_imzml_no_marker_for_peaklist(synth_centroided, tmp_path):
    """A non-harmonized dataset (raw PeakList) must NOT carry the marker."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.io.imzml_writer import write_imzml

    ds = read_imzml(synth_centroided)
    out = tmp_path / "raw_copy.imzML"
    result = write_imzml(ds, out)
    xml = result.imzml_path.read_text(encoding="utf-8")
    assert "dapple-harmonized" not in xml
    assert not result.imzml_path.with_suffix(".dapple-axis.json").exists()


def test_round_trip_restores_peakmatrix_backend(synth_centroided, tmp_path):
    """Writing a PeakMatrix and reading it back must yield a PeakMatrix-backed
    dataset (not the raw PeakList you'd get without the marker)."""
    from dapple.data.dataset import PeakMatrix
    from dapple.io.imzml_reader import read_imzml
    from dapple.io.imzml_writer import write_imzml

    aligned = _build_aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.imzML"
    write_imzml(aligned, out)
    reloaded = read_imzml(out)
    assert isinstance(reloaded.backend, PeakMatrix), (
        f"expected PeakMatrix backend after round-trip, got {type(reloaded.backend)}"
    )
    # Shape preserved.
    assert reloaded.backend.n_pixels == aligned.backend.n_pixels
    assert reloaded.backend.n_peaks == aligned.backend.n_peaks


def test_round_trip_preserves_matrix_intensities(synth_centroided, tmp_path):
    """The matrix values themselves should round-trip within float32 tolerance."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.io.imzml_writer import write_imzml

    aligned = _build_aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.imzML"
    write_imzml(aligned, out)
    reloaded = read_imzml(out)
    np.testing.assert_allclose(
        np.asarray(reloaded.backend.matrix[:]),
        np.asarray(aligned.backend.matrix[:]),
        rtol=1e-5, atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(reloaded.backend.mz_axis[:]),
        np.asarray(aligned.backend.mz_axis[:]),
    )


def test_round_trip_preserves_consensus_prevalence(synth_centroided, tmp_path):
    """The ``consensus_prevalence`` array stored in extras must come back."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.io.imzml_writer import write_imzml

    aligned = _build_aligned_dataset(synth_centroided)
    assert "consensus_prevalence" in aligned.extra, (
        "fixture should already have consensus_prevalence (the consensus operator attaches it)"
    )
    original_prev = np.asarray(aligned.extra["consensus_prevalence"])

    out = tmp_path / "harmonized.imzML"
    write_imzml(aligned, out)
    reloaded = read_imzml(out)
    assert "consensus_prevalence" in reloaded.extra, (
        "reload must restore the per-channel consensus prevalence"
    )
    np.testing.assert_allclose(
        np.asarray(reloaded.extra["consensus_prevalence"]),
        original_prev,
    )


def test_missing_sidecar_falls_back_to_peaklist(synth_centroided, tmp_path):
    """If the marker is present but the sidecar is missing, the reader should
    warn and fall through to a PeakList backend (not crash)."""
    from dapple.data.dataset import PeakList
    from dapple.io.imzml_reader import read_imzml
    from dapple.io.imzml_writer import write_imzml

    aligned = _build_aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.imzML"
    write_imzml(aligned, out)
    # Delete the sidecar.
    sidecar = out.with_suffix(".dapple-axis.json")
    sidecar.unlink()
    with pytest.warns(UserWarning, match="restoration failed"):
        reloaded = read_imzml(out)
    # Falls back to PeakList rather than crashing.
    assert isinstance(reloaded.backend, PeakList)


def test_channels_panel_renders_consensus_rows_after_round_trip(
    qtbot, make_napari_viewer, synth_centroided, tmp_path
):
    """The user-visible regression: Channels Panel must show consensus channel
    rows on a reloaded harmonized file, not just summary projections."""
    pytest.importorskip("napari", reason="napari is required for widget tests")
    pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")

    from dapple.io.imzml_reader import read_imzml
    from dapple.io.imzml_writer import write_imzml
    from dapple.widgets._session import MsiSession
    from dapple.widgets.channels import ChannelsPanel

    aligned = _build_aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.imzML"
    write_imzml(aligned, out)
    reloaded = read_imzml(out)

    s = MsiSession()
    viewer = make_napari_viewer()
    panel = ChannelsPanel(napari_viewer=viewer, session=s)
    qtbot.addWidget(panel)
    s.set_dataset(reloaded)

    # Should have 6 summaries plus N consensus channels (N >= 5 for synth).
    n_channels = sum(1 for r in panel._rows if r.kind == "channel")  # noqa: SLF001
    assert n_channels >= 5, (
        f"reloaded harmonized dataset should expose consensus channels in the "
        f"Channels Panel; got {n_channels}"
    )
