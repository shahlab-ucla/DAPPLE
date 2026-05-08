"""Tests for the hot-pixel detection + correction operator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.data.metadata import DatasetIdentity, ExperimentParams
from dapple.io.imzml_reader import read_imzml
from dapple.ops.hot_pixel import HotPixelFilter, HotPixelParams


def _make_synthetic_with_hotspot(synth_centroided: Path) -> "object":
    """Load the synth fixture, then 100x boost intensities of one pixel to make it hot."""
    ds = read_imzml(synth_centroided)
    pl: PeakList = ds.backend
    intensity = np.asarray(pl.intensity[:]).copy()
    offsets = np.asarray(pl.offsets[:])
    # Spam pixel index 12 (middle of 5x5) with 100x intensity.
    a, b = int(offsets[12]), int(offsets[13])
    intensity[a:b] *= 100.0
    new_pl = PeakList(
        mz=np.asarray(pl.mz[:]).copy(),
        intensity=intensity,
        offsets=offsets.copy(),
        n_pixels=pl.n_pixels,
    )
    return ds.with_backend(new_pl)


def test_hot_pixel_detects_obvious_spike(synth_centroided):
    ds = _make_synthetic_with_hotspot(synth_centroided)
    op = HotPixelFilter()
    rng = np.random.default_rng(0)
    result = op.apply(ds, HotPixelParams(k_mad=5.0, correction="zero"), rng=rng)
    diag = result.diagnostics[0]
    # The boosted pixel must be flagged.
    assert diag.summary["n_hot_pixels"] >= 1
    assert 12 in result.dataset.extra["hot_pixel_indices"].tolist()


def test_hot_pixel_zero_correction_zeroes_the_pixel(synth_centroided):
    ds = _make_synthetic_with_hotspot(synth_centroided)
    op = HotPixelFilter()
    rng = np.random.default_rng(0)
    result = op.apply(ds, HotPixelParams(k_mad=5.0, correction="zero"), rng=rng)
    pl: PeakList = result.dataset.backend
    mz, intensity = pl.pixel(12)
    # Zero correction blanks the spectrum at the hot pixel.
    assert (intensity == 0).all()


def test_hot_pixel_neighbors_median_scales_to_neighbor_tic(synth_centroided):
    """neighbors_median should rescale the hot pixel's spectrum so its TIC matches the
    median TIC of its valid neighbors. Peak counts and m/z values are preserved."""
    ds = _make_synthetic_with_hotspot(synth_centroided)
    op = HotPixelFilter()
    rng = np.random.default_rng(0)
    result = op.apply(
        ds,
        HotPixelParams(k_mad=5.0, correction="neighbors_median", min_neighbors=3),
        rng=rng,
    )
    pl_in: PeakList = ds.backend
    pl_out: PeakList = result.dataset.backend
    mz_in, _ = pl_in.pixel(12)
    mz_out, int_out = pl_out.pixel(12)
    # Peak count + m/z values preserved.
    assert mz_out.size == mz_in.size
    np.testing.assert_array_equal(np.sort(mz_in), np.sort(mz_out))
    # New TIC should be near the median TIC of the (untouched) neighbors.
    # Neighbors of pixel 12 in 5x5 are pixels 6,7,8,11,13,16,17,18.
    neighbor_tics = [pl_in.per_pixel_reduce("sum")[i] for i in (6, 7, 8, 11, 13, 16, 17, 18)]
    expected = float(np.median(neighbor_tics))
    np.testing.assert_allclose(int_out.sum(), expected, rtol=1e-3)


def test_hot_pixel_passthrough_when_no_hot(synth_centroided):
    """No spike in the data → operator is a no-op but still records a diagnostic."""
    ds = read_imzml(synth_centroided)
    op = HotPixelFilter()
    rng = np.random.default_rng(0)
    result = op.apply(ds, HotPixelParams(k_mad=5.0), rng=rng)
    assert result.diagnostics[0].summary["n_hot_pixels"] == 0


def test_hot_pixel_mark_only(synth_centroided):
    """correction='mark' records hot-pixel indices but doesn't mutate intensities."""
    ds = _make_synthetic_with_hotspot(synth_centroided)
    op = HotPixelFilter()
    rng = np.random.default_rng(0)
    result = op.apply(
        ds, HotPixelParams(k_mad=5.0, correction="mark"), rng=rng
    )
    pl_in: PeakList = ds.backend
    pl_out: PeakList = result.dataset.backend
    np.testing.assert_array_equal(
        np.asarray(pl_in.intensity[:]), np.asarray(pl_out.intensity[:])
    )
    assert "hot_pixel_indices" in result.dataset.extra


def test_hot_pixel_default_params_have_labels_and_help():
    from dataclasses import fields
    from dapple.ops.base import field_help, field_label

    for f in fields(HotPixelParams):
        assert field_label(f) and field_label(f) != f.name
        assert field_help(f)
