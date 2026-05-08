"""Tests for cwt_peak_pick (profile-mode picker) and reference_ion_normalize."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from dapple.data.dataset import MSIDataset, PeakList
from dapple.data.metadata import DatasetIdentity, ExperimentParams
from dapple.io.imzml_reader import read_imzml
from dapple.ops.normalize import ReferenceIonNormalize
from dapple.ops.peak_pick import CwtPeakPick, CwtPeakPickParams
from dapple.ops.reference_ions import DetectReferenceIons


# ---- profile-mode synthetic fixture --------------------------------------------


def _make_profile_dataset() -> MSIDataset:
    """Synthesize a tiny 5×5 profile-mode dataset with three Gaussian peaks per
    pixel at known m/z positions."""
    rng = np.random.default_rng(123)
    n_x = n_y = 5
    n_pixels = n_x * n_y
    base_peaks = np.array([200.0, 300.0, 400.0])
    n_grid = 1024
    mz_grid = np.linspace(150.0, 450.0, n_grid)
    fwhm_mz = 0.05  # ~100 ppm at m/z 500

    pixel_mz: list[np.ndarray] = []
    pixel_int: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for iy in range(n_y):
        for ix in range(n_x):
            spectrum = rng.normal(0.5, 0.05, n_grid).clip(min=0.0)
            for mz0 in base_peaks:
                amp = 30.0 + 5.0 * rng.normal()
                spectrum += amp * np.exp(-(mz_grid - mz0) ** 2 / (2 * (fwhm_mz / 2.355) ** 2))
            pixel_mz.append(mz_grid.copy())
            pixel_int.append(spectrum.astype(np.float32))
            coords.append((ix + 1, iy + 1))
    flat_mz = np.concatenate(pixel_mz)
    flat_int = np.concatenate(pixel_int)
    offsets = np.arange(n_pixels + 1, dtype=np.int64) * n_grid
    pl = PeakList(
        mz=flat_mz.astype(np.float64),
        intensity=flat_int,
        offsets=offsets,
        n_pixels=n_pixels,
    )
    ep = ExperimentParams(
        instrument_family="qtof",
        ionization="esi",
        profile_or_centroided="profile",
        polarity="positive",
        mz_min=150.0,
        mz_max=450.0,
    )
    return MSIDataset(
        coords=np.asarray(coords, dtype=np.int32),
        grid_shape=(n_y, n_x),
        metadata=ep,
        backend=pl,
        identity=DatasetIdentity(
            source_path="synth_profile",
            content_sha256=hashlib.sha256(b"synth").hexdigest(),
        ),
    )


# ---- cwt_peak_pick --------------------------------------------------------------


def test_cwt_recovers_planted_peaks_on_profile_synth():
    ds = _make_profile_dataset()
    op = CwtPeakPick()
    result = op.apply(
        ds,
        CwtPeakPickParams(
            width_min_ppm=20.0,
            width_max_ppm=300.0,
            n_widths=10,
            grid_ppm_step=10.0,
            min_snr=2.0,
        ),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    assert isinstance(new.backend, PeakList)
    # Each pixel should yield ~3 peaks (one per planted m/z).
    counts = new.backend.per_pixel_count()
    assert int(np.median(counts)) >= 3
    # Pooled m/z values cluster near {200, 300, 400}.
    pooled = np.asarray(new.backend.mz[:])
    for target in (200.0, 300.0, 400.0):
        nearest = pooled[np.abs(pooled - target).argmin()]
        ppm_err = abs(nearest - target) / target * 1e6
        assert ppm_err < 200, f"closest peak to {target} is {nearest} ({ppm_err:.0f} ppm)"


def test_cwt_validate_warns_on_centroided(synth_centroided):
    """The validate() method warns when called with centroided ExperimentParams."""
    ds = read_imzml(synth_centroided)
    msgs = CwtPeakPick().validate(ds.metadata)
    assert any("profile" in m.lower() for m in msgs)


def test_cwt_default_params_have_labels_and_help():
    from dataclasses import fields

    from dapple.ops.base import field_help, field_label

    for f in fields(CwtPeakPickParams):
        assert field_label(f)
        assert field_help(f)


# ---- reference_ion_normalize ----------------------------------------------------


def test_reference_ion_normalize_requires_refs(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = ReferenceIonNormalize()
    with pytest.raises(RuntimeError, match="detect_reference_ions"):
        op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))


def test_reference_ion_normalize_per_pixel_factor_equals_ref_sum(synth_centroided):
    """The per-pixel factor must equal the sum of reference-ion intensities."""
    ds = read_imzml(synth_centroided)
    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(
        ds, DetectReferenceIons().default_params(ds.metadata), rng=rng
    ).dataset
    expected = ds.extra["reference_set"].per_pixel_intensity.sum(axis=1)
    op = ReferenceIonNormalize()
    result = op.apply(ds, op.default_params(ds.metadata), rng=rng)
    factors = result.diagnostics[0].payload["per_pixel_factor"]
    np.testing.assert_allclose(factors, np.maximum(expected, 1e-12))


def test_reference_ion_normalize_preserves_peak_count(synth_centroided):
    ds = read_imzml(synth_centroided)
    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(
        ds, DetectReferenceIons().default_params(ds.metadata), rng=rng
    ).dataset
    op = ReferenceIonNormalize()
    result = op.apply(ds, op.default_params(ds.metadata), rng=rng)
    np.testing.assert_array_equal(
        ds.backend.per_pixel_count(), result.dataset.backend.per_pixel_count()
    )


def test_reference_ion_normalize_validate_emits_caveat():
    from dapple.data.metadata import ExperimentParams

    ep = ExperimentParams(
        instrument_family="tof_reflectron",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=1000.0,
    )
    msgs = ReferenceIonNormalize().validate(ep)
    assert any("reference ions" in m.lower() for m in msgs)
