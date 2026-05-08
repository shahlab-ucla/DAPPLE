"""Tests for the per-pixel msiwarp_recalibrate operator."""

from __future__ import annotations

from dataclasses import replace as drep

import numpy as np
import pytest

from dapple.data.dataset import PeakList
from dapple.io.imzml_reader import read_imzml
from dapple.ops.recalibrate import MsiwarpRecalibrate, MsiwarpRecalibrateParams
from dapple.ops.reference_ions import DetectReferenceIons
from dapple.ops.tolerance import (
    EmpiricalToleranceFromReferenceIons,
    EmpiricalToleranceParams,
)


def _with_refs_and_tol(synth_centroided):
    ds = read_imzml(synth_centroided)
    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(
        ds, DetectReferenceIons().default_params(ds.metadata), rng=rng
    ).dataset
    ds = EmpiricalToleranceFromReferenceIons().apply(
        ds,
        EmpiricalToleranceParams(alpha=0.01, bootstrap_B=20, block_bootstrap=False),
        rng=rng,
    ).dataset
    return ds


def test_msiwarp_requires_reference_set(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = MsiwarpRecalibrate()
    with pytest.raises(RuntimeError, match="reference_ions"):
        op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))


def test_msiwarp_reduces_per_anchor_residual_on_synth(synth_centroided):
    """After recalibration, every per-pixel residual at an anchor m/z should be
    near zero (linear fit through ≥ 3 anchors absorbs both shift and scale)."""
    ds = _with_refs_and_tol(synth_centroided)
    pre_residuals = np.abs(ds.extra["reference_set"].per_pixel_ppm_error)
    pre_median = float(np.nanmedian(pre_residuals))
    op = MsiwarpRecalibrate()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    summary = result.diagnostics[0].summary
    # Operator reports an improvement, and the post-residual median is below the
    # pre-residual median for the synth jitter scale.
    assert summary["improvement_ppm"] >= 0
    assert summary["post_residual_median_ppm"] <= pre_median + 1e-6
    # Most pixels were recalibrated (synth has 5 anchors per pixel, well above
    # min_inliers = 3).
    assert summary["fraction_pixels_recalibrated"] >= 0.95


def test_msiwarp_preserves_peak_count_and_dataset_shape(synth_centroided):
    ds = _with_refs_and_tol(synth_centroided)
    op = MsiwarpRecalibrate()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    new = result.dataset
    assert isinstance(new.backend, PeakList)
    np.testing.assert_array_equal(
        np.asarray(ds.backend.offsets[:]), np.asarray(new.backend.offsets[:])
    )
    assert new.backend.mz.shape == ds.backend.mz.shape


def test_msiwarp_skips_pixels_with_too_few_inliers(synth_centroided):
    """Even with min_inliers raised above the synth's 5 anchors, every pixel must
    be skipped (not raise) and the dataset must round-trip unchanged."""
    ds = _with_refs_and_tol(synth_centroided)
    op = MsiwarpRecalibrate()
    result = op.apply(
        ds,
        MsiwarpRecalibrateParams(min_inliers=99),
        rng=np.random.default_rng(0),
    )
    summary = result.diagnostics[0].summary
    assert summary["n_pixels_recalibrated"] == 0
    assert summary["n_pixels_skipped"] == ds.n_pixels
    np.testing.assert_array_equal(
        np.asarray(ds.backend.mz[:]), np.asarray(result.dataset.backend.mz[:])
    )


def test_msiwarp_passthrough_with_too_few_reference_ions(synth_centroided):
    """If only one reference ion is detected, recalibration should pass through
    cleanly with a clear diagnostic note."""
    ds = _with_refs_and_tol(synth_centroided)
    # Manually shrink the ReferenceSet to one ion to simulate a sparse case.
    ref = ds.extra["reference_set"]
    new_ref = drep(ref, mz=ref.mz[:1], per_pixel_mz=ref.per_pixel_mz[:, :1],
                   per_pixel_intensity=ref.per_pixel_intensity[:, :1],
                   per_pixel_ppm_error=ref.per_pixel_ppm_error[:, :1],
                   prevalence=ref.prevalence[:1], n_observations=ref.n_observations[:1])
    new_extra = {**ds.extra, "reference_set": new_ref}
    ds = ds.__class__(
        coords=ds.coords,
        grid_shape=ds.grid_shape,
        metadata=ds.metadata,
        backend=ds.backend,
        identity=ds.identity,
        history=ds.history,
        rois=ds.rois,
        rng_seed=ds.rng_seed,
        extra=new_extra,
    )
    op = MsiwarpRecalibrate()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    assert result.diagnostics[0].summary["n_pixels_recalibrated"] == 0


def test_msiwarp_orbitrap_validate_warns():
    from dapple.data.metadata import ExperimentParams

    ep = ExperimentParams(
        instrument_family="orbitrap",
        ionization="esi",
        profile_or_centroided="centroided",
        polarity="positive",
        mz_min=100.0,
        mz_max=1000.0,
    )
    msgs = MsiwarpRecalibrate().validate(ep)
    assert any("Orbitrap" in m or "high-resolution" in m.lower() for m in msgs)


def test_msiwarp_default_params_have_labels_and_help():
    from dataclasses import fields

    from dapple.ops.base import field_help, field_label

    for f in fields(MsiwarpRecalibrateParams):
        assert field_label(f)
        assert field_help(f)
