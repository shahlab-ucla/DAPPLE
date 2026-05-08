"""Tests for the single-file ANDI-MS NetCDF reader.

The synthetic case is generated in `tests/conftest.py` (see `synth_lcms_cdf`); the
real-file case (the supplied Finnigan LC-MS) is gated by MSI_REAL_DATA=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.io.cdf_reader import read_cdf


def test_read_cdf_synth_loads(synth_lcms_cdf):
    with pytest.warns(UserWarning, match="not an MSI image"):
        ds = read_cdf(synth_lcms_cdf)
    assert isinstance(ds.backend, PeakList)
    assert ds.backend.n_pixels == 5  # 5 synthetic scans
    assert ds.grid_shape == (1, 5)
    assert ds.extra["is_imaging"] is False
    # ExperimentParams: synth fixture uses ESI-equivalent attributes.
    assert ds.metadata.ionization in {"esi", "unknown"}


def test_read_cdf_synth_pixel_spectra(synth_lcms_cdf):
    with pytest.warns(UserWarning):
        ds = read_cdf(synth_lcms_cdf)
    mz0, int0 = ds.backend.pixel(0)
    assert mz0.size > 0
    assert int0.size == mz0.size
    assert (mz0 > 0).all()


def test_read_cdf_rejects_non_andims(tmp_path):
    """A file lacking ANDI-MS variables should error helpfully, not silently load."""
    import netCDF4 as nc

    fake = tmp_path / "fake.cdf"
    nc_ds = nc.Dataset(str(fake), "w", format="NETCDF3_CLASSIC")
    try:
        nc_ds.createDimension("x", 5)
        v = nc_ds.createVariable("not_mass", "f8", ("x",))
        v[:] = [1, 2, 3, 4, 5]
    finally:
        nc_ds.close()
    with pytest.raises(ValueError, match="ANDI-MS"):
        read_cdf(fake)


@pytest.mark.real_data
@pytest.mark.skipif(os.environ.get("MSI_REAL_DATA", "0") != "1", reason="real-data only")
def test_read_real_finnigan_lcms(real_dataset_dir: Path):
    """The supplied M004-OG2-P11-0-9.cdf is Finnigan LC-MS — 1-D, must warn."""
    p = real_dataset_dir / "M004-OG2-P11-0-9.cdf"
    if not p.exists():
        pytest.skip(f"{p} not found")
    with pytest.warns(UserWarning, match="not an MSI image"):
        ds = read_cdf(p)
    assert ds.backend.n_pixels == 142
    assert ds.metadata.polarity == "negative"
    assert ds.metadata.ionization == "esi"  # 1-D context → keep as ESI, not DESI
    assert ds.extra["is_imaging"] is False
