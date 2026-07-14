"""Tests for spec_xml read/write, multipage TIFF, and imzML writer.

All tests use synthetic fixtures so the suite stays fast and self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dapple.data.dataset import PeakMatrix
from dapple.io.imzml_reader import read_imzml
from dapple.io.imzml_writer import write_imzml
from dapple.io.spec_xml import (
    make_provenance,
    read_spec_xml,
    write_spec_xml,
)
from dapple.io.tiff_writer import (
    read_hyperspectral_tiff_metadata,
    write_hyperspectral_tiff,
)
from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
from dapple.pipeline import PipelineRunner, recommend_pipeline


def _aligned_dataset(synth_centroided: Path):
    """Run the recommended pipeline on the synth fixture and return the post-consensus
    dataset (PeakMatrix backend)."""
    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    # Tighten consensus for the small synth case.
    from dataclasses import replace as drep

    from dapple.pipeline.pipeline import Node

    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))
    runner = PipelineRunner()
    return runner.run(p, ds), p, ds


# ---- spec_xml ---------------------------------------------------------------------


def test_spec_xml_round_trips_pipeline_and_params(synth_centroided, tmp_path):
    _result, pipeline, ds = _aligned_dataset(synth_centroided)
    prov = make_provenance(
        plugin_version="0.1.0.dev0",
        input_dataset_hash=ds.hash(),
        declared_md5=ds.identity.declared_md5,
        notes="round-trip test",
    )
    out = tmp_path / "run.spec.xml"
    write_spec_xml(out, pipeline=pipeline, experiment_params=ds.metadata, provenance=prov)

    loaded_pipe, loaded_ep, loaded_prov = read_spec_xml(out)
    assert loaded_ep == ds.metadata
    assert loaded_prov.plugin_version == prov.plugin_version
    assert loaded_prov.input_dataset_hash == ds.hash()
    assert loaded_pipe.rng_seed == pipeline.rng_seed
    assert tuple(n.op_name for n in loaded_pipe.nodes) == tuple(
        n.op_name for n in pipeline.nodes
    )
    # Per-node parameters must round-trip exactly.
    for n_orig, n_loaded in zip(pipeline.nodes, loaded_pipe.nodes, strict=True):
        assert n_loaded.id == n_orig.id
        assert n_loaded.upstream == n_orig.upstream
        assert n_loaded.params == n_orig.params


def test_spec_xml_contains_diagnostics_summary(synth_centroided, tmp_path):
    result, pipeline, ds = _aligned_dataset(synth_centroided)
    prov = make_provenance(plugin_version="0.1.0.dev0", input_dataset_hash=ds.hash())
    out = tmp_path / "run.spec.xml"
    write_spec_xml(
        out,
        pipeline=pipeline,
        experiment_params=ds.metadata,
        provenance=prov,
        diagnostics=result.diagnostics,
    )
    text = out.read_text()
    assert "<diagnosticsSummary" in text
    # Reference-ion count summary should appear.
    assert "n_reference_ions" in text


def test_spec_xml_rejects_wrong_namespace(tmp_path):
    bad = tmp_path / "wrong.spec.xml"
    bad.write_text("<spec xmlns='urn:other:spec' version='1.0'/>")
    with pytest.raises(ValueError, match="namespace"):
        read_spec_xml(bad)


def test_spec_xml_pipeline_hash_invariant_under_roundtrip(synth_centroided, tmp_path):
    _result, pipeline, ds = _aligned_dataset(synth_centroided)
    prov = make_provenance(plugin_version="0.1.0.dev0", input_dataset_hash=ds.hash())
    out = tmp_path / "run.spec.xml"
    write_spec_xml(out, pipeline=pipeline, experiment_params=ds.metadata, provenance=prov)
    loaded, _, _ = read_spec_xml(out)
    h_orig = pipeline.hash(input_hash=ds.hash())
    h_loaded = loaded.hash(input_hash=ds.hash())
    assert h_orig == h_loaded


def test_spec_xml_round_trips_roi_geometry(synth_centroided, tmp_path):
    from dapple.data.metadata import RoiDef

    _result, pipeline, ds = _aligned_dataset(synth_centroided)
    rois = (
        RoiDef(
            name="early bud",
            vertices=((0.0, 0.0), (0.0, 2.5), (2.0, 0.0)),
            color="#123456",
        ),
        RoiDef(
            name="matrix",
            vertices=((3.0, 3.0), (3.0, 4.0), (4.0, 3.0)),
            is_background=True,
        ),
    )
    out = tmp_path / "rois.spec.xml"
    write_spec_xml(
        out,
        pipeline=pipeline,
        experiment_params=ds.metadata,
        provenance=make_provenance(
            plugin_version="0.1.0.dev0",
            input_dataset_hash=ds.hash(),
            roi_definitions=rois,
        ),
    )
    _, _, provenance = read_spec_xml(out)
    assert provenance.roi_definitions == rois
    assert "napari-data-yx-zero-based" in out.read_text(encoding="utf-8")


# ---- multipage TIFF ---------------------------------------------------------------


def test_tiff_writes_one_page_per_consensus_peak(synth_centroided, tmp_path):
    result, _, ds = _aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.tif"
    written = write_hyperspectral_tiff(result.output, out)
    assert written.tiff_path.exists()
    assert written.csv_path.exists()
    assert written.n_pages == result.output.backend.n_peaks
    assert written.image_shape == ds.grid_shape


def test_tiff_per_page_metadata_carries_mz(synth_centroided, tmp_path):
    result, _, _ds = _aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.tif"
    write_hyperspectral_tiff(result.output, out)
    pages = read_hyperspectral_tiff_metadata(out)
    assert len(pages) == result.output.backend.n_peaks
    # Every page records a numeric m/z and prevalence in [0, 1].
    for page_meta in pages:
        assert "mz" in page_meta and isinstance(page_meta["mz"], (int, float))
        assert 0.0 <= page_meta["prevalence"] <= 1.0
    # m/z values should match the PeakMatrix axis (allow tiny FP drift).
    mzs_meta = sorted(p["mz"] for p in pages)
    mzs_axis = sorted(np.asarray(result.output.backend.mz_axis[:]).tolist())
    np.testing.assert_allclose(mzs_meta, mzs_axis, rtol=1e-9)


def test_tiff_csv_has_one_row_per_page(synth_centroided, tmp_path):
    result, _, _ds = _aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.tif"
    written = write_hyperspectral_tiff(result.output, out)
    rows = written.csv_path.read_text(encoding="utf-8").strip().splitlines()
    # Hash column lives only in the TIFF page metadata; the CSV is human-friendly.
    assert rows[0].split(",")[:3] == ["page", "mz", "prevalence"]
    assert "op_history_hash" not in rows[0], (
        "op_history_hash belongs in the TIFF page metadata, not the user-visible CSV"
    )
    assert len(rows) == 1 + result.output.backend.n_peaks


def test_tiff_page_metadata_still_carries_hash(synth_centroided, tmp_path):
    """The hash is still present in each TIFF page's JSON description for tools that
    want to verify which pipeline produced the file."""
    result, _, _ds = _aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.tif"
    write_hyperspectral_tiff(result.output, out)
    pages = read_hyperspectral_tiff_metadata(out)
    for page_meta in pages:
        assert page_meta.get("op_history_hash"), (
            "TIFF pages must keep op_history_hash for provenance even when the CSV omits it"
        )


def test_tiff_rejects_pre_consensus_dataset(synth_centroided, tmp_path):
    ds = read_imzml(synth_centroided)
    out = tmp_path / "wont_write.tif"
    with pytest.raises(ValueError, match="PeakMatrix"):
        write_hyperspectral_tiff(ds, out)


def test_tiff_extra_per_channel_columns(synth_centroided, tmp_path):
    result, _, _ds = _aligned_dataset(synth_centroided)
    n = result.output.backend.n_peaks
    extra = {"morans_i": np.arange(n, dtype=np.float64)}
    out = tmp_path / "harmonized.tif"
    written = write_hyperspectral_tiff(result.output, out, extra_per_channel=extra)
    rows = written.csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert "morans_i" in rows[0]
    pages = read_hyperspectral_tiff_metadata(out)
    assert pages[3]["morans_i"] == 3


# ---- imzML writer -----------------------------------------------------------------


def test_imzml_round_trip_peaklist(synth_centroided, tmp_path):
    """Write a PeakList-backed dataset, read it back, peak counts match."""
    ds = read_imzml(synth_centroided)
    out = tmp_path / "rt.imzML"
    written = write_imzml(ds, out)
    assert written.imzml_path.exists()
    assert written.ibd_path.exists()
    assert written.n_spectra == ds.n_pixels
    # Read back through pyimzml-backed reader.
    ds2 = read_imzml(written.imzml_path)
    assert ds2.n_pixels == ds.n_pixels
    np.testing.assert_array_equal(ds.backend.per_pixel_count(), ds2.backend.per_pixel_count())
    # First pixel's m/z values should match within FP precision.
    mz_a, _ = ds.backend.pixel(0)
    mz_b, _ = ds2.backend.pixel(0)
    np.testing.assert_allclose(np.sort(mz_a), np.sort(mz_b), rtol=1e-12)


def test_imzml_round_trip_preserves_profile_mode_and_observed_mz_range(
    synth_centroided, tmp_path
):
    """Reader accepts the writer's fileContent-level observed-range terms."""
    from dataclasses import replace

    ds = read_imzml(synth_centroided)
    ds = replace(
        ds,
        metadata=replace(ds.metadata, profile_or_centroided="profile"),
    )
    all_mz = np.asarray(ds.backend.mz[:], dtype=np.float64)
    expected_min = float(all_mz.min())
    expected_max = float(all_mz.max())

    written = write_imzml(ds, tmp_path / "profile_roundtrip.imzML")
    reloaded = read_imzml(written.imzml_path)

    assert reloaded.metadata.profile_or_centroided == "profile"
    assert reloaded.extra["metadata_source"]["profile_or_centroided"] == "imzml"
    assert reloaded.metadata.mz_min == pytest.approx(expected_min)
    assert reloaded.metadata.mz_max == pytest.approx(expected_max)
    assert reloaded.extra["metadata_source"]["mz_min"] == "imzml"
    assert reloaded.extra["metadata_source"]["mz_max"] == "imzml"


def test_imzml_round_trip_peakmatrix(synth_centroided, tmp_path):
    """Write a post-consensus PeakMatrix-backed dataset; verify the round-trip
    preserves the matrix structure.

    DAPPLE marks PeakMatrix-backed exports with a ``dapple-harmonized`` user-param
    plus a sibling ``.dapple-axis.json`` so the reader can rebuild the dense
    matrix on load (the Channels Panel and Spectrum Panel both depend on this).
    See ``test_imzml_harmonized_roundtrip.py`` for the full restoration suite.
    """
    from dapple.data.dataset import PeakMatrix

    result, _, _ds = _aligned_dataset(synth_centroided)
    out = tmp_path / "harmonized.imzML"
    write_imzml(result.output, out)
    ds2 = read_imzml(out)
    assert ds2.n_pixels == result.output.n_pixels
    # The reloaded backend is a PeakMatrix with the same dimensions and values.
    assert isinstance(ds2.backend, PeakMatrix)
    assert ds2.backend.n_pixels == result.output.backend.n_pixels
    assert ds2.backend.n_peaks == result.output.backend.n_peaks
    np.testing.assert_allclose(
        np.asarray(ds2.backend.matrix[:]),
        np.asarray(result.output.backend.matrix[:]),
        rtol=1e-5, atol=1e-7,
    )


def test_imzml_md5_matches_declared(synth_centroided, tmp_path):
    """The MD5 declared in the written .imzML must match the .ibd contents."""
    ds = read_imzml(synth_centroided)
    out = tmp_path / "rt.imzML"
    written = write_imzml(ds, out)
    text = written.imzml_path.read_text()
    assert written.ibd_md5 in text
    # Reading back without warning means the MD5 matches.
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        read_imzml(written.imzml_path)
    md5_warnings = [w for w in caught if "MD5" in str(w.message)]
    assert not md5_warnings


def test_imzml_writer_default_extension_handling(synth_centroided, tmp_path):
    """Calling write_imzml with a path lacking .imzML still produces both files."""
    ds = read_imzml(synth_centroided)
    out_base = tmp_path / "stem_only"
    written = write_imzml(ds, out_base)
    assert written.imzml_path.exists() and written.imzml_path.suffix == ".imzML"
    assert written.ibd_path.exists() and written.ibd_path.suffix == ".ibd"
