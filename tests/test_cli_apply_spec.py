"""End-to-end test for the dapple-apply-spec CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dapple.cli.apply_spec import main as apply_spec_main


def _write_spec_for(synth_centroided: Path, spec_path: Path) -> Path:
    """Run the recommended pipeline once and save its .spec.xml so the CLI
    has something to reapply."""
    from dapple.io.imzml_reader import read_imzml
    from dapple.io.spec_xml import make_provenance, write_spec_xml
    from dapple.ops.consensus import KdeConsensusParams
    from dapple.pipeline import recommend_pipeline
    from dapple.pipeline.pipeline import Node

    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    nodes = list(p.nodes)
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
    p = p.__class__(nodes=tuple(nodes), rng_seed=p.rng_seed, library_versions=p.library_versions)
    prov = make_provenance(plugin_version="0.1.0.dev0", input_dataset_hash=ds.hash())
    write_spec_xml(spec_path, pipeline=p, experiment_params=ds.metadata, provenance=prov)
    return spec_path


def test_apply_spec_end_to_end(synth_centroided, tmp_path):
    spec_path = tmp_path / "saved.spec.xml"
    _write_spec_for(synth_centroided, spec_path)

    out_base = tmp_path / "out"
    rc = apply_spec_main(
        [str(synth_centroided), str(spec_path), "-o", str(out_base)]
    )
    assert rc == 0
    assert (out_base.with_suffix(".imzML")).exists()
    assert (out_base.with_suffix(".ibd")).exists()
    assert (out_base.with_suffix(".tif")).exists()
    assert (out_base.parent / (out_base.stem + "_channels.csv")).exists()
    assert (out_base.with_suffix(".spec.xml")).exists()


def test_apply_spec_no_imzml_flag(synth_centroided, tmp_path):
    spec_path = tmp_path / "saved.spec.xml"
    _write_spec_for(synth_centroided, spec_path)
    out_base = tmp_path / "out_tiff_only"
    rc = apply_spec_main(
        [str(synth_centroided), str(spec_path), "-o", str(out_base), "--no-imzml"]
    )
    assert rc == 0
    assert (out_base.with_suffix(".tif")).exists()
    assert not (out_base.with_suffix(".imzML")).exists()


def test_apply_spec_no_tiff_flag(synth_centroided, tmp_path):
    spec_path = tmp_path / "saved.spec.xml"
    _write_spec_for(synth_centroided, spec_path)
    out_base = tmp_path / "out_imzml_only"
    rc = apply_spec_main(
        [str(synth_centroided), str(spec_path), "-o", str(out_base), "--no-tiff"]
    )
    assert rc == 0
    assert (out_base.with_suffix(".imzML")).exists()
    assert not (out_base.with_suffix(".tif")).exists()


def test_apply_spec_rng_seed_override(synth_centroided, tmp_path):
    """Overriding the spec's RNG seed produces a fresh .spec.xml whose pipeline
    hash differs from the original."""
    from dapple.io.spec_xml import read_spec_xml

    spec_path = tmp_path / "saved.spec.xml"
    _write_spec_for(synth_centroided, spec_path)
    out_base = tmp_path / "out_seeded"
    rc = apply_spec_main(
        [
            str(synth_centroided), str(spec_path),
            "-o", str(out_base),
            "--rng-seed", "12345",
            "--no-tiff",
        ]
    )
    assert rc == 0
    new_pipeline, _, _ = read_spec_xml(out_base.with_suffix(".spec.xml"))
    assert new_pipeline.rng_seed == 12345


def test_apply_spec_missing_spec_returns_1(tmp_path, synth_centroided, capsys):
    rc = apply_spec_main([str(synth_centroided), str(tmp_path / "nonexistent.spec.xml")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()


def test_apply_spec_unrecognized_input_returns_1(tmp_path, capsys):
    bad = tmp_path / "junk.txt"
    bad.write_text("nope")
    spec = tmp_path / "any.spec.xml"
    # Make a tiny placeholder spec; the CLI will fail at input-load time first.
    spec.write_text(
        "<spec xmlns='urn:napari-msi:spec:v1' version='1.0'>"
        "<provenance><createdAt>2026-01-01T00:00:00Z</createdAt>"
        "<pluginVersion>0</pluginVersion></provenance>"
        "<experimentParams>"
        "<instrument_family>tof_reflectron</instrument_family>"
        "<ionization>maldi</ionization>"
        "<profile_or_centroided>centroided</profile_or_centroided>"
        "<polarity>negative</polarity>"
        "<mz_min>100</mz_min><mz_max>1000</mz_max>"
        "</experimentParams>"
        "<pipeline rngSeed='0'/></spec>",
        encoding="utf-8",
    )
    rc = apply_spec_main([str(bad), str(spec)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unrecognized" in err.lower() or "expected" in err.lower()
