"""Tests for cohort harmonization (``align_cohort`` + ``dapple-cohort-align``)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dapple.cli.cohort_align import main as cohort_align_main
from dapple.cohort.align import (
    CohortAlignParams,
    CohortAlignResult,
    align_cohort,
    load_cohort_directory,
)
from dapple.data.dataset import PeakMatrix
from dapple.io.imzml_reader import read_imzml


def _make_two_member_cohort(synth_centroided: Path, tmp_path: Path) -> tuple[Path, Path]:
    """Copy the synth fixture twice so we have two distinct datasets to harmonize."""
    a_dir = tmp_path / "cohort_a"
    a_dir.mkdir()
    a_imzml = a_dir / "a.imzML"
    a_ibd = a_dir / "a.ibd"
    shutil.copy(synth_centroided, a_imzml)
    shutil.copy(synth_centroided.with_suffix(".ibd"), a_ibd)
    b_dir = tmp_path / "cohort_b"
    b_dir.mkdir()
    b_imzml = b_dir / "b.imzML"
    b_ibd = b_dir / "b.ibd"
    shutil.copy(synth_centroided, b_imzml)
    shutil.copy(synth_centroided.with_suffix(".ibd"), b_ibd)
    return a_imzml, b_imzml


# ---- align_cohort core ---------------------------------------------------------


def test_align_cohort_returns_shared_axis_for_each_dataset(synth_centroided, tmp_path):
    a, b = _make_two_member_cohort(synth_centroided, tmp_path)
    ds_a = read_imzml(a)
    ds_b = read_imzml(b)
    result = align_cohort(
        [ds_a, ds_b],
        params=CohortAlignParams(bandwidth_ppm=20.0, min_prevalence=0.5, recalibrate=False),
    )
    assert isinstance(result, CohortAlignResult)
    assert len(result.aligned_datasets) == 2
    axis_a = np.asarray(result.aligned_datasets[0].backend.mz_axis[:])
    axis_b = np.asarray(result.aligned_datasets[1].backend.mz_axis[:])
    np.testing.assert_array_equal(axis_a, axis_b)
    np.testing.assert_array_equal(axis_a, result.shared_consensus_mz)


def test_align_cohort_each_dataset_is_peakmatrix_backed(synth_centroided, tmp_path):
    a, b = _make_two_member_cohort(synth_centroided, tmp_path)
    result = align_cohort(
        [read_imzml(a), read_imzml(b)],
        params=CohortAlignParams(bandwidth_ppm=20.0, min_prevalence=0.5, recalibrate=False),
    )
    for d in result.aligned_datasets:
        assert isinstance(d.backend, PeakMatrix)
        # Each row of the matrix corresponds to one pixel of THIS dataset.
        assert d.backend.matrix.shape[0] == d.n_pixels


def test_align_cohort_diagnostics_summarize_run(synth_centroided, tmp_path):
    a, b = _make_two_member_cohort(synth_centroided, tmp_path)
    result = align_cohort(
        [read_imzml(a), read_imzml(b)],
        params=CohortAlignParams(bandwidth_ppm=20.0, min_prevalence=0.5, recalibrate=False),
    )
    diag = result.diagnostics
    for key in (
        "n_datasets", "n_total_pixels", "n_total_peaks_pooled",
        "n_consensus_shared", "cohort_prevalence_median",
    ):
        assert key in diag
    assert diag["n_datasets"] == 2.0
    assert diag["n_consensus_shared"] >= 5  # synth has 5 peaks


def test_align_cohort_per_dataset_prevalence_per_axis(synth_centroided, tmp_path):
    a, b = _make_two_member_cohort(synth_centroided, tmp_path)
    result = align_cohort(
        [read_imzml(a), read_imzml(b)],
        params=CohortAlignParams(bandwidth_ppm=20.0, min_prevalence=0.5, recalibrate=False),
    )
    n_axis = result.shared_consensus_mz.size
    for di, prev in result.per_dataset_prevalence.items():
        assert prev.shape == (n_axis,)
        # Synth peaks are mostly preserved per-dataset; the upstream picker can
        # drop a few low-intensity peaks per pixel, so we check that the
        # majority of each dataset's pixels carry each consensus channel.
        assert prev.mean() >= 0.5
        assert (prev > 0).all()


def test_align_cohort_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        align_cohort([])


def test_align_cohort_requires_peaklist_backend(synth_centroided, tmp_path):
    """Passing a PeakMatrix (post-consensus) dataset is rejected."""
    a, _ = _make_two_member_cohort(synth_centroided, tmp_path)
    ds = read_imzml(a)
    # Run consensus to convert ds to PeakMatrix backend, then try to feed back.
    from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
    from dapple.ops.normalize import MedianNormalize
    from dapple.ops.peak_pick import SnrPeakPick
    from dapple.ops.reference_ions import DetectReferenceIons
    from dapple.ops.tolerance import (
        EmpiricalToleranceFromReferenceIons,
        EmpiricalToleranceParams,
    )

    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(ds, DetectReferenceIons().default_params(ds.metadata), rng=rng).dataset
    ds = EmpiricalToleranceFromReferenceIons().apply(
        ds, EmpiricalToleranceParams(alpha=0.01, bootstrap_B=20, block_bootstrap=False), rng=rng
    ).dataset
    ds = MedianNormalize().apply(ds, MedianNormalize().default_params(ds.metadata), rng=rng).dataset
    ds = SnrPeakPick().apply(ds, SnrPeakPick().default_params(ds.metadata), rng=rng).dataset
    aligned_single = KdeConsensusAlignment().apply(
        ds, KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        rng=rng,
    ).dataset
    with pytest.raises(ValueError, match="PeakList"):
        align_cohort([aligned_single])


def test_align_cohort_consensus_axis_includes_planted_peaks(synth_centroided, tmp_path):
    """Planted peaks (200, 250, 300, 350, 400) should appear on the shared axis."""
    a, b = _make_two_member_cohort(synth_centroided, tmp_path)
    result = align_cohort(
        [read_imzml(a), read_imzml(b)],
        params=CohortAlignParams(bandwidth_ppm=20.0, min_prevalence=0.5, recalibrate=False),
    )
    expected = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
    for ex in expected:
        nearest = result.shared_consensus_mz[np.abs(result.shared_consensus_mz - ex).argmin()]
        ppm_err = abs(nearest - ex) / ex * 1e6
        assert ppm_err < 200, f"shared axis missed {ex}: closest was {nearest:.4f}"


# ---- load_cohort_directory ----------------------------------------------------


def test_load_cohort_directory_picks_up_imzml_files(synth_centroided, tmp_path):
    # Use a dedicated subdirectory so we don't accidentally pick up the
    # synth_centroided fixture that lives at tmp_path/synth_centroided.imzML.
    parent = tmp_path / "cohort_root"
    parent.mkdir()
    import shutil

    for name in ("a.imzML", "b.imzML"):
        shutil.copy(synth_centroided, parent / name)
        shutil.copy(synth_centroided.with_suffix(".ibd"), parent / name.replace(".imzML", ".ibd"))
    datasets = load_cohort_directory(parent, pattern="*.imzML", recursive=False)
    assert len(datasets) == 2


def test_load_cohort_directory_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files"):
        load_cohort_directory(tmp_path, pattern="*.imzML")


def test_load_cohort_directory_nondir_raises(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("nope")
    with pytest.raises(NotADirectoryError):
        load_cohort_directory(f)


# ---- dapple-cohort-align CLI --------------------------------------------------


def test_cohort_align_cli_end_to_end(synth_centroided, tmp_path):
    # Build a cohort directory containing two copies of the synth.
    root = tmp_path / "cohort_root"
    root.mkdir()
    shutil.copy(synth_centroided, root / "a.imzML")
    shutil.copy(synth_centroided.with_suffix(".ibd"), root / "a.ibd")
    shutil.copy(synth_centroided, root / "b.imzML")
    shutil.copy(synth_centroided.with_suffix(".ibd"), root / "b.ibd")

    out_dir = tmp_path / "cohort_out"
    rc = cohort_align_main(
        [
            str(root),
            "-o", str(out_dir),
            "--bandwidth-ppm", "20",
            "--min-prevalence", "0.5",
            "--no-recalibrate",
        ]
    )
    assert rc == 0
    assert (out_dir / "cohort_summary.json").exists()
    summary = json.loads((out_dir / "cohort_summary.json").read_text(encoding="utf-8"))
    assert summary["n_datasets"] == 2
    assert len(summary["shared_consensus_mz"]) >= 5
    assert "diagnostics" in summary
    # Per-dataset outputs are named <stem>_cohort.tif / .imzML.
    assert (out_dir / "a_cohort.tif").exists()
    assert (out_dir / "b_cohort.tif").exists()
    assert (out_dir / "a_cohort.imzML").exists()
    assert (out_dir / "b_cohort.imzML").exists()


def test_cohort_align_cli_missing_root_returns_1(tmp_path, capsys):
    rc = cohort_align_main([str(tmp_path / "nonexistent_dir")])
    assert rc == 1
    assert "dapple-cohort-align" in capsys.readouterr().err.lower()


def test_cohort_align_cli_no_files_returns_1(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = cohort_align_main([str(empty)])
    assert rc == 1
    assert "no files" in capsys.readouterr().err.lower()
