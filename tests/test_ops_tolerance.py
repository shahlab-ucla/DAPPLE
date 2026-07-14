"""Tests for empirical_tolerance_from_reference_ions."""

from __future__ import annotations

import numpy as np

from dapple.io.imzml_reader import read_imzml
from dapple.ops.reference_ions import DetectReferenceIons
from dapple.ops.tolerance import (
    EmpiricalToleranceFromReferenceIons,
    EmpiricalToleranceParams,
    ToleranceCurve,
)


def _run_with_refs(ds, *, B=200):
    """Run detect_reference_ions then empirical_tolerance, returning the tolerance result."""
    rng = np.random.default_rng(1)
    ref_op = DetectReferenceIons()
    ref_result = ref_op.apply(ds, ref_op.default_params(ds.metadata), rng=rng)
    tol_op = EmpiricalToleranceFromReferenceIons()
    tol_result = tol_op.apply(
        ref_result.dataset,
        EmpiricalToleranceParams(alpha=0.01, bootstrap_B=B, block_bootstrap=False),
        rng=rng,
    )
    return tol_result


def test_empirical_tolerance_curve_attached(synth_centroided):
    ds = read_imzml(synth_centroided)
    result = _run_with_refs(ds)
    curve: ToleranceCurve | None = result.dataset.extra.get("tolerance_curve")
    assert curve is not None
    assert curve.mz_grid.size == 200
    assert curve.ppm_quantile.shape == (200,)
    assert curve.ci_low.shape == (200,)
    assert curve.ci_high.shape == (200,)


def test_empirical_tolerance_within_jitter_range(synth_centroided):
    """Synth jitter is N(0, 5 ppm) so the 99% quantile of |ppm| should be ~12 ppm."""
    ds = read_imzml(synth_centroided)
    result = _run_with_refs(ds)
    curve: ToleranceCurve = result.dataset.extra["tolerance_curve"]
    median_ppm = float(np.median(curve.ppm_quantile))
    # 5 ppm Gaussian -> 99.5th quantile ~ 14 ppm; allow generous margin.
    assert 1.0 < median_ppm < 30.0


def test_empirical_tolerance_ci_band_is_non_negative(synth_centroided):
    ds = read_imzml(synth_centroided)
    result = _run_with_refs(ds)
    curve: ToleranceCurve = result.dataset.extra["tolerance_curve"]
    assert (curve.ci_high >= curve.ci_low).all()


def test_stochastic_output_state_changes_dataset_hash(synth_centroided):
    """Direct calls with different RNG streams cannot alias in provenance/cache state."""
    ds = read_imzml(synth_centroided)
    ref_op = DetectReferenceIons()
    with_refs = ref_op.apply(
        ds, ref_op.default_params(ds.metadata), rng=np.random.default_rng(0)
    ).dataset
    op = EmpiricalToleranceFromReferenceIons()
    params = EmpiricalToleranceParams(
        alpha=0.01, bootstrap_B=30, block_bootstrap=False
    )
    a = op.apply(with_refs, params, rng=np.random.default_rng(10)).dataset
    b = op.apply(with_refs, params, rng=np.random.default_rng(11)).dataset
    assert not np.array_equal(
        a.extra["tolerance_curve"].ci_low,
        b.extra["tolerance_curve"].ci_low,
    )
    assert a.hash() != b.hash()


def test_empirical_tolerance_evaluate_interpolates(synth_centroided):
    ds = read_imzml(synth_centroided)
    result = _run_with_refs(ds)
    curve: ToleranceCurve = result.dataset.extra["tolerance_curve"]
    # Querying inside the grid returns interpolated; outside clamps.
    inside = curve.evaluate(np.array([curve.mz_grid[50], curve.mz_grid[100]]))
    assert np.allclose(inside, [curve.ppm_quantile[50], curve.ppm_quantile[100]], atol=1e-9)


def test_empirical_tolerance_requires_refs_first(synth_centroided):
    import pytest

    ds = read_imzml(synth_centroided)
    op = EmpiricalToleranceFromReferenceIons()
    with pytest.raises(RuntimeError, match="detect_reference_ions"):
        op.apply(ds, EmpiricalToleranceParams(), rng=np.random.default_rng(0))


def test_pav_increasing_simple_case():
    from dapple.ops.tolerance import _pav_increasing

    y = np.array([5.0, 3.0, 4.0, 2.0, 6.0])
    out = _pav_increasing(y)
    # Must be non-decreasing.
    assert (np.diff(out) >= -1e-12).all()
    # Mean is preserved.
    assert abs(out.mean() - y.mean()) < 1e-9
