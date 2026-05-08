"""imzML reader tests using a synthetic dataset written by conftest.

These tests do NOT require any real datasets — they round-trip a tiny generated
.imzML/.ibd pair through `read_imzml` and verify dimensions, metadata extraction,
and sample spectra.
"""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.io.imzml_reader import read_imzml


def test_read_synth_imzml_loads_correct_shape(synth_centroided):
    ds = read_imzml(synth_centroided)
    assert ds.n_pixels == 25
    assert ds.grid_shape == (5, 5)
    assert isinstance(ds.backend, PeakList)
    # Each pixel was generated with 5 peaks.
    counts = ds.backend.per_pixel_count()
    assert counts.shape == (25,)
    assert (counts == 5).all()


def test_read_synth_imzml_extracts_metadata(synth_centroided):
    ds = read_imzml(synth_centroided)
    md = ds.metadata
    assert md.profile_or_centroided == "centroided"
    assert md.polarity == "negative"
    assert md.ionization == "maldi"
    # tof family should be detected from MS:1000084 even without reflectron declared
    assert md.instrument_family in {"tof_axial", "tof_reflectron", "qtof"}
    assert md.pixel_size_um == 50.0


def test_read_synth_imzml_pixel_spectrum_roundtrip(synth_centroided):
    ds = read_imzml(synth_centroided)
    # Pixel (1, 1) is index 0 in our generated raster.
    mz, intensity = ds.pixel_spectrum(1, 1)
    assert mz.shape == (5,)
    assert intensity.shape == (5,)
    # m/z values are in spec.peaks_mz with at most 5 ppm jitter.
    base = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
    ppm_err = (mz - base) / base * 1e6
    assert np.abs(ppm_err).max() < 50  # 5 ppm spec, 10x cushion


def test_read_synth_imzml_tic_projection_shape(synth_centroided):
    ds = read_imzml(synth_centroided)
    img = ds.project("tic")
    assert img.shape == (5, 5)
    assert img.dtype == np.float32
    assert (img > 0).all()


def test_read_synth_imzml_peak_count_projection(synth_centroided):
    ds = read_imzml(synth_centroided)
    img = ds.project("peak_count")
    assert img.shape == (5, 5)
    assert (img == 5).all()


def test_read_synth_imzml_md5_validation(synth_centroided):
    """The MD5 declared in conftest matches; no MD5 warning should fire."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as recorded:
        _warnings.simplefilter("always")
        read_imzml(synth_centroided)
    md5_warnings = [w for w in recorded if "MD5" in str(w.message)]
    assert not md5_warnings, f"unexpected MD5 warnings: {md5_warnings}"


def test_read_imzml_rejects_unknown_extension(tmp_path):
    fake = tmp_path / "junk.txt"
    fake.write_text("nope")
    with pytest.raises(ValueError, match="expected .imzML or .ibd"):
        read_imzml(fake)
