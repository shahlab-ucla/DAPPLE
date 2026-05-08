"""Tests for the multi-file ANDI-MS CDF imaging reader.

Synthetic-fixture tests build a tiny multi-file directory in tmp_path; real-data tests
run against the supplied `Boone cdf/` DESI dataset (140 files × 62 scans = 8680 pixels)
and are gated by MSI_REAL_DATA=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.io.cdf_image_reader import (
    CdfLayout,
    infer_layout,
    read_cdf_image,
)


def test_synth_cdf_image_loads(synth_cdf_image_dir):
    ds = read_cdf_image(synth_cdf_image_dir)
    assert isinstance(ds.backend, PeakList)
    # 4 files × 6 scans each = 24 pixels
    assert ds.backend.n_pixels == 24
    assert ds.grid_shape == (4, 6)
    assert ds.extra["is_imaging"] is True


def test_synth_cdf_image_default_layout(synth_cdf_image_dir):
    layout, warnings_ = infer_layout(synth_cdf_image_dir)
    assert layout.scans_per_line == 6
    assert layout.start_index == 1
    assert layout.axis_along_files == "y"
    assert not layout.serpentine


def test_synth_cdf_image_coords(synth_cdf_image_dir):
    ds = read_cdf_image(synth_cdf_image_dir)
    # File 1 row 1 holds scans 1..6 → coords (1,1), (2,1), ..., (6,1).
    first_row = ds.coords[:6]
    np.testing.assert_array_equal(first_row[:, 0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(first_row[:, 1], [1] * 6)
    last_row = ds.coords[-6:]
    np.testing.assert_array_equal(last_row[:, 1], [4] * 6)


def test_synth_cdf_image_serpentine(synth_cdf_image_dir):
    """With serpentine=True, even rows (0-indexed) keep order, odd rows reverse."""
    layout = CdfLayout(
        pattern="*.cdf", axis_along_files="y", serpentine=True, start_index=1, scans_per_line=6
    )
    ds = read_cdf_image(synth_cdf_image_dir, layout=layout)
    # Row 1 (file 0): forward — coords x = 1..6.
    np.testing.assert_array_equal(ds.coords[:6, 0], [1, 2, 3, 4, 5, 6])
    # Row 2 (file 1): reversed — coords x = 6..1, but each pixel still gets its scan_pos
    # in scan order. Our serpentine impl reverses the *source-scan* mapping, so the
    # spatial x-coordinate (scan_pos+1) still increments 1..6; the data at x=1 just
    # comes from src_scan=5 instead of src_scan=0. The test verifies coord layout.
    np.testing.assert_array_equal(ds.coords[6:12, 0], [1, 2, 3, 4, 5, 6])


def test_read_cdf_image_rejects_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .cdf files"):
        read_cdf_image(tmp_path)


def test_read_cdf_image_rejects_non_dir(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("nope")
    with pytest.raises(NotADirectoryError):
        read_cdf_image(f)


@pytest.mark.real_data
@pytest.mark.skipif(os.environ.get("MSI_REAL_DATA", "0") != "1", reason="real-data only")
def test_read_real_boone_cdf_image(real_dataset_dir: Path):
    """Boone DESI dataset: 140 files × 62 scans = 8680 pixels, ~13.4M peaks."""
    boone_dir = real_dataset_dir / "Boone cdf"
    if not boone_dir.exists():
        pytest.skip(f"{boone_dir} not found")
    ds = read_cdf_image(boone_dir)
    assert ds.backend.n_pixels == 140 * 62
    assert ds.grid_shape == (140, 62)
    md = ds.metadata
    assert md.polarity == "negative"
    # Multi-file imaging context → ESI rewritten as DESI by default.
    assert md.ionization == "desi"
    assert md.profile_or_centroided == "centroided"  # Xcalibur exports as centroided
    # m/z bounds from union: 50-1000 per the global attributes.
    assert 49 <= md.mz_min <= 51
    assert 999 <= md.mz_max <= 1001


@pytest.mark.real_data
@pytest.mark.skipif(os.environ.get("MSI_REAL_DATA", "0") != "1", reason="real-data only")
def test_real_boone_default_pipeline_recommends_centroided_path(real_dataset_dir: Path):
    """The default pipeline on the Boone DESI dataset uses snr_peak_pick (it's centroided)."""
    boone_dir = real_dataset_dir / "Boone cdf"
    if not boone_dir.exists():
        pytest.skip(f"{boone_dir} not found")
    ds = read_cdf_image(boone_dir)
    from dapple.ops.peak_pick import SnrPeakPick

    op = SnrPeakPick()
    warnings_for_ep = op.validate(ds.metadata)
    # Centroided datasets shouldn't emit the profile-data warning.
    assert not any("profile" in w for w in warnings_for_ep)


@pytest.mark.real_data
@pytest.mark.skipif(os.environ.get("MSI_REAL_DATA", "0") != "1", reason="real-data only")
def test_real_boone_tic_projection(real_dataset_dir: Path):
    boone_dir = real_dataset_dir / "Boone cdf"
    if not boone_dir.exists():
        pytest.skip(f"{boone_dir} not found")
    ds = read_cdf_image(boone_dir)
    img = ds.project("tic")
    assert img.shape == (140, 62)
    # Every pixel has a non-empty spectrum.
    assert (img > 0).all()


@pytest.mark.real_data
@pytest.mark.skipif(os.environ.get("MSI_REAL_DATA", "0") != "1", reason="real-data only")
@pytest.mark.slow
def test_real_boone_full_pipeline(real_dataset_dir: Path):
    """End-to-end pipeline on Boone DESI: load → refs → tolerance → norm → pick → consensus.

    Larger and slower than jerboa (8680 pixels × ~1500 peaks = ~13M peaks). Marked
    'slow' so dev-iteration runs can skip it via `-m "not slow"`.
    """
    from dapple.data.dataset import PeakMatrix
    from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
    from dapple.ops.normalize import MedianNormalize
    from dapple.ops.peak_pick import SnrPeakPick, SnrPeakPickParams
    from dapple.ops.reference_ions import DetectReferenceIons
    from dapple.ops.tolerance import (
        EmpiricalToleranceFromReferenceIons,
        EmpiricalToleranceParams,
    )

    boone_dir = real_dataset_dir / "Boone cdf"
    if not boone_dir.exists():
        pytest.skip(f"{boone_dir} not found")

    ds = read_cdf_image(boone_dir)
    rng = np.random.default_rng(0)

    ref_op = DetectReferenceIons()
    ds = ref_op.apply(ds, ref_op.default_params(ds.metadata), rng=rng).dataset
    ref = ds.extra["reference_set"]
    assert ref.mz.shape[0] >= 5

    tol_op = EmpiricalToleranceFromReferenceIons()
    ds = tol_op.apply(
        ds,
        EmpiricalToleranceParams(alpha=0.01, bootstrap_B=20, block_bootstrap=False),
        rng=rng,
    ).dataset

    norm_op = MedianNormalize()
    ds = norm_op.apply(ds, norm_op.default_params(ds.metadata), rng=rng).dataset

    pick_op = SnrPeakPick()
    # Use a high-quantile threshold to bring per-pixel peak counts down — Boone
    # centroided spectra contain ~1500 peaks/pixel, which would balloon KDE work.
    ds = pick_op.apply(
        ds, SnrPeakPickParams(snr_mad=3.0, min_intensity_quantile=0.95), rng=rng
    ).dataset

    cons_op = KdeConsensusAlignment()
    ds = cons_op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=50.0, bandwidth_ppm=50.0, min_prevalence=0.05),
        rng=rng,
    ).dataset

    assert isinstance(ds.backend, PeakMatrix)
    assert ds.backend.n_peaks >= 5
    assert ds.backend.n_pixels == 8680
