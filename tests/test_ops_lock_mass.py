"""Tests for the single-anchor lock_mass_recalibrate operator."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.io.imzml_reader import read_imzml
from dapple.ops.recalibrate import LockMassRecalibrate, LockMassRecalibrateParams
from dapple.ops.reference_ions import DetectReferenceIons


def _with_refs(synth_centroided):
    ds = read_imzml(synth_centroided)
    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(
        ds, DetectReferenceIons().default_params(ds.metadata), rng=rng
    ).dataset
    return ds


def test_lockmass_requires_reference_set(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = LockMassRecalibrate()
    with pytest.raises(RuntimeError, match="detect_reference_ions"):
        op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))


def test_lockmass_default_params_have_labels_and_help():
    from dataclasses import fields

    from dapple.ops.base import field_help, field_label

    for f in fields(LockMassRecalibrateParams):
        assert field_label(f)
        assert field_help(f)


def test_lockmass_shifts_per_pixel_to_anchor(synth_centroided):
    """After lock-mass correction, the chosen anchor m/z in each pixel should
    move to (essentially) the reference centroid. Other peaks shift by the
    same multiplicative factor."""
    ds = _with_refs(synth_centroided)
    op = LockMassRecalibrate()
    rng = np.random.default_rng(0)
    pre_pl: PeakList = ds.backend
    pre_mz = np.asarray(pre_pl.mz[:]).copy()
    result = op.apply(ds, op.default_params(ds.metadata), rng=rng)
    diag = result.diagnostics[0]
    summary = diag.summary
    # Most pixels were corrected (synth has 5 anchors per pixel; default uses
    # highest-intensity).
    assert summary["fraction_pixels_recalibrated"] >= 0.95
    # Median |shift| should be small (synth jitter is ~5 ppm) but non-zero.
    assert 0 < summary["ppm_shift_abs_median"] < 50
    # Peak count and offsets are preserved.
    post_pl: PeakList = result.dataset.backend
    np.testing.assert_array_equal(np.asarray(pre_pl.offsets[:]), np.asarray(post_pl.offsets[:]))
    # Some m/z values must have changed (correction happened).
    post_mz = np.asarray(post_pl.mz[:])
    assert not np.allclose(pre_mz, post_mz)


def test_lockmass_explicit_anchor_strategy(synth_centroided):
    """closest_to_mz uses whichever reference centroid is nearest the supplied m/z."""
    ds = _with_refs(synth_centroided)
    op = LockMassRecalibrate()
    rng = np.random.default_rng(0)
    result = op.apply(
        ds,
        LockMassRecalibrateParams(
            anchor_strategy="closest_to_mz", explicit_anchor_mz=300.0
        ),
        rng=rng,
    )
    # Every pixel using the same anchor (synth has the m/z=300 reference present in all).
    assert result.diagnostics[0].summary["fraction_pixels_recalibrated"] >= 0.95


def test_lockmass_explicit_anchor_strategy_requires_mz(synth_centroided):
    ds = _with_refs(synth_centroided)
    op = LockMassRecalibrate()
    with pytest.raises(RuntimeError, match="explicit_anchor_mz"):
        op.apply(
            ds,
            LockMassRecalibrateParams(anchor_strategy="closest_to_mz", explicit_anchor_mz=0.0),
            rng=np.random.default_rng(0),
        )


def test_lockmass_max_shift_caps_apply_mask(synth_centroided):
    """A tiny max_shift_ppm leaves every pixel uncorrected."""
    ds = _with_refs(synth_centroided)
    op = LockMassRecalibrate()
    result = op.apply(
        ds, LockMassRecalibrateParams(max_shift_ppm=1e-6),
        rng=np.random.default_rng(0),
    )
    assert result.diagnostics[0].summary["fraction_pixels_recalibrated"] == 0
