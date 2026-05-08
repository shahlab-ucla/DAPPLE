"""Smoke tests against the real jerboa MALDI-TOF dataset.

Gated by the MSI_REAL_DATA=1 environment variable. The dataset must live at
C:\\Users\\pavak\\MSI\\datasets\\jerboa-100825.imzML / .ibd (override with
MSI_DATA_DIR).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.io.imzml_reader import read_imzml
from dapple.ops.reference_ions import DetectReferenceIons


pytestmark = pytest.mark.skipif(
    os.environ.get("MSI_REAL_DATA", "0") != "1",
    reason="Set MSI_REAL_DATA=1 to run real-data integration tests.",
)


@pytest.fixture
def jerboa_path(real_dataset_dir: Path) -> Path:
    p = real_dataset_dir / "jerboa-100825.imzML"
    if not p.exists():
        pytest.skip(f"jerboa-100825.imzML not found at {p}")
    return p


@pytest.mark.real_data
def test_load_jerboa_dataset(jerboa_path: Path):
    ds = read_imzml(jerboa_path)
    assert ds.n_pixels == 4136
    assert ds.grid_shape == (82, 59)
    assert isinstance(ds.backend, PeakList)
    md = ds.metadata
    assert md.polarity == "negative"
    assert md.ionization == "maldi"
    assert md.profile_or_centroided == "centroided"
    assert md.instrument_family == "tof_reflectron"
    assert md.pixel_size_um == 50.0
    # m/z range: aggregated across all 4136 spectra. Spectrum 0 alone is 78.96–529.28
    # but the global min and max widen beyond that.
    assert 40 <= md.mz_min <= 100
    assert 500 <= md.mz_max <= 800


@pytest.mark.real_data
def test_jerboa_md5_matches(jerboa_path: Path):
    """Declared IBD MD5 must match the actual file MD5 (no warnings)."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        read_imzml(jerboa_path)
    md5_msgs = [w for w in caught if "MD5" in str(w.message)]
    assert not md5_msgs, f"unexpected MD5 warnings: {[str(w.message) for w in md5_msgs]}"


@pytest.mark.real_data
def test_jerboa_tic_projection(jerboa_path: Path):
    ds = read_imzml(jerboa_path)
    img = ds.project("tic")
    assert img.shape == (82, 59)
    assert img.dtype == np.float32
    # Some pixels are filled (data present), some are zero (no spectrum at that grid pos).
    assert (img > 0).sum() == 4136


@pytest.mark.real_data
def test_jerboa_peak_count_projection(jerboa_path: Path):
    ds = read_imzml(jerboa_path)
    img = ds.project("peak_count")
    # Centroided MALDI-TOF spectra typically carry ~50–100 peaks per pixel.
    nz = img[img > 0]
    assert nz.size == 4136
    assert nz.mean() > 30


@pytest.mark.real_data
def test_jerboa_pixel_spectrum_first_pixel(jerboa_path: Path):
    """First spectrum in the imzML is at position (43, 1) (1-indexed) per the XML."""
    ds = read_imzml(jerboa_path)
    mz, intensity = ds.pixel_spectrum(43, 1)
    assert mz.size > 0
    assert intensity.size == mz.size
    assert (mz > 78).all() and (mz < 600).all()
    assert (intensity > 0).all()


@pytest.mark.real_data
def test_jerboa_detect_reference_ions(jerboa_path: Path):
    ds = read_imzml(jerboa_path)
    op = DetectReferenceIons()
    params = op.default_params(ds.metadata)
    rng = np.random.default_rng(0)
    result = op.apply(ds, params, rng=rng)
    ref = result.dataset.extra["reference_set"]
    # Common matrix peaks across the bone tissue should give us at least 5
    # reference ions for the downstream tolerance fit.
    assert ref.mz.shape[0] >= 5
    # Every kept ion has prevalence above the threshold.
    assert (ref.prevalence >= params.min_prevalence).all()


@pytest.mark.real_data
def test_jerboa_full_pipeline_e2e(jerboa_path: Path):
    """End-to-end pipeline: load → refs → tolerance → normalize → pick → consensus.

    Exercises the recommended pipeline for centroided MALDI-TOF. It must complete
    in a few seconds and produce a non-empty PeakMatrix on the real data.
    """
    from dapple.data.dataset import PeakMatrix
    from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
    from dapple.ops.normalize import MedianNormalize
    from dapple.ops.peak_pick import SnrPeakPick
    from dapple.ops.tolerance import (
        EmpiricalToleranceFromReferenceIons,
        EmpiricalToleranceParams,
    )

    ds = read_imzml(jerboa_path)
    rng = np.random.default_rng(42)

    # 1. Detect reference ions.
    ref_op = DetectReferenceIons()
    ds = ref_op.apply(ds, ref_op.default_params(ds.metadata), rng=rng).dataset

    # 2. Empirical tolerance from refs.
    tol_op = EmpiricalToleranceFromReferenceIons()
    ds = tol_op.apply(
        ds,
        EmpiricalToleranceParams(alpha=0.01, bootstrap_B=100, block_bootstrap=False),
        rng=rng,
    ).dataset

    # 3. Median normalize.
    norm_op = MedianNormalize()
    ds = norm_op.apply(ds, norm_op.default_params(ds.metadata), rng=rng).dataset

    # 4. SNR peak pick (centroided default — expect to keep most peaks since the
    #    upstream Bruker SCiLS centroiding has already filtered noise).
    pick_op = SnrPeakPick()
    ds = pick_op.apply(ds, pick_op.default_params(ds.metadata), rng=rng).dataset

    # 5. KDE consensus alignment.
    cons_op = KdeConsensusAlignment()
    ds = cons_op.apply(ds, cons_op.default_params(ds.metadata), rng=rng).dataset

    # Final dataset is post-consensus → PeakMatrix.
    assert isinstance(ds.backend, PeakMatrix)
    assert ds.backend.n_peaks >= 5
    assert ds.backend.n_pixels == 4136
    # History records all 5 ops in order.
    op_names = [r.op_name for r in ds.history]
    assert op_names == [
        "detect_reference_ions",
        "empirical_tolerance_from_reference_ions",
        "median_normalize",
        "snr_peak_pick",
        "kde_consensus_alignment",
    ]
    # Each consensus peak has prevalence >= the configured filter.
    prev = ds.extra["consensus_prevalence"]
    assert prev.shape == (ds.backend.n_peaks,)
    assert (prev >= 0.05).all()
