"""Tests for Pipeline DAG, recommend_pipeline, and PipelineRunner."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.dataset import PeakMatrix
from dapple.data.metadata import ExperimentParams
from dapple.io.imzml_reader import read_imzml
from dapple.ops.normalize import NormalizeParams
from dapple.pipeline import (
    Node,
    Pipeline,
    PipelineRunner,
    detect_library_versions,
    recommend_pipeline,
)


def _ep() -> ExperimentParams:
    return ExperimentParams(
        instrument_family="tof_reflectron",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=1000.0,
        pixel_size_um=50.0,
    )


# ---- Pipeline structural validation ----


def test_pipeline_rejects_duplicate_node_ids():
    a = Node(id="x", op_name="median_normalize", params=NormalizeParams(method="median"))
    b = Node(id="x", op_name="tic_normalize", params=NormalizeParams(method="tic"))
    with pytest.raises(ValueError, match="duplicate node id"):
        Pipeline(nodes=(a, b))


def test_pipeline_rejects_unknown_upstream():
    a = Node(
        id="x",
        op_name="median_normalize",
        params=NormalizeParams(method="median"),
        upstream=("ghost",),
    )
    with pytest.raises(ValueError, match="unknown upstream"):
        Pipeline(nodes=(a,))


def test_pipeline_rejects_topological_violation():
    a = Node(
        id="a",
        op_name="median_normalize",
        params=NormalizeParams(method="median"),
        upstream=("b",),
    )
    b = Node(id="b", op_name="tic_normalize", params=NormalizeParams(method="tic"))
    with pytest.raises(ValueError, match="topologically sorted"):
        Pipeline(nodes=(a, b))


def test_pipeline_hash_is_input_dependent():
    p = recommend_pipeline(_ep())
    h1 = p.hash(input_hash="aa" * 32)
    h2 = p.hash(input_hash="bb" * 32)
    assert h1 != h2


def test_pipeline_hash_is_seed_dependent():
    p1 = recommend_pipeline(_ep(), rng_seed=1)
    p2 = recommend_pipeline(_ep(), rng_seed=2)
    h1 = p1.hash(input_hash="00" * 32)
    h2 = p2.hash(input_hash="00" * 32)
    assert h1 != h2


def test_detect_library_versions_returns_only_installed():
    versions = detect_library_versions()
    assert "numpy" in versions
    # napari isn't installed in the minimal env; key should be absent.
    assert "napari" not in versions or isinstance(versions["napari"], str)


# ---- recommend_pipeline ----


def test_recommend_pipeline_default_chain():
    p = recommend_pipeline(_ep())
    ids = tuple(n.id for n in p.nodes)
    # Core nodes always present, in order; recommend_pipeline may insert extra
    # instrument-specific or sample-specific nodes between or after these.
    for required in ("ref", "tol", "norm", "pick", "consensus"):
        assert required in ids, f"recommended pipeline missing {required!r}"
    # Topological invariants for the core chain.
    assert ids.index("tol") > ids.index("ref")
    assert ids.index("norm") > ids.index("tol")
    assert ids.index("pick") > ids.index("norm")
    assert ids.index("consensus") > ids.index("pick")
    assert p.nodes[0].upstream == ()
    # tof_reflectron family triggers a recalibration step between tol and norm.
    assert "recal" in ids
    assert ids.index("recal") > ids.index("tol")
    assert ids.index("recal") < ids.index("norm")


def test_recommend_pipeline_orbitrap_uses_5ppm_tolerance():
    ep = ExperimentParams(
        instrument_family="orbitrap",
        ionization="esi",
        profile_or_centroided="centroided",
        polarity="positive",
        mz_min=100.0,
        mz_max=1000.0,
    )
    p = recommend_pipeline(ep)
    ref_node = next(n for n in p.nodes if n.id == "ref")
    assert ref_node.params.coarse_tol_ppm == 5.0


# ---- PipelineRunner ----


def test_runner_executes_recommended_pipeline_on_synth(synth_centroided):
    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    # Override consensus parameters for the small synthetic case so it actually finds
    # the planted peaks (default min_prevalence is 0.05; bandwidth defaults are fine).
    nodes = list(p.nodes)
    from dataclasses import replace as drep

    from dapple.ops.consensus import KdeConsensusParams

    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))

    runner = PipelineRunner()
    progress_calls: list[tuple[int, int, str]] = []
    result = runner.run(p, ds, progress=lambda i, n, msg: progress_calls.append((i, n, msg)))

    assert isinstance(result.output.backend, PeakMatrix)
    assert result.output.backend.n_peaks >= 5
    # Every pipeline node must have emitted at least one diagnostic.
    assert set(result.diagnostics.keys()) == set(n.id for n in p.nodes)
    # The five core operators must always be present.
    assert {"ref", "tol", "norm", "pick", "consensus"}.issubset(result.diagnostics.keys())
    # Progress: at least one call per node + final "done".
    assert len(progress_calls) >= len(p.nodes) + 1
    assert progress_calls[-1][2] == "done"


def test_runner_caches_repeated_runs(synth_centroided):
    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    nodes = list(p.nodes)
    from dataclasses import replace as drep

    from dapple.ops.consensus import KdeConsensusParams

    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))

    runner = PipelineRunner()
    nodes_executed: list[str] = []
    runner.run(p, ds, on_node_done=lambda nid, _r: nodes_executed.append(nid))
    # Second run: every node should be served from cache (no on_node_done callbacks).
    nodes_executed.clear()
    runner.run(p, ds, on_node_done=lambda nid, _r: nodes_executed.append(nid))
    assert nodes_executed == []


def test_runner_per_node_seed_is_deterministic(synth_centroided):
    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata, rng_seed=12345)
    nodes = list(p.nodes)
    from dataclasses import replace as drep

    from dapple.ops.consensus import KdeConsensusParams

    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))

    a = PipelineRunner().run(p, ds).output
    b = PipelineRunner().run(p, ds).output
    np.testing.assert_array_equal(np.asarray(a.backend.matrix[:]), np.asarray(b.backend.matrix[:]))
