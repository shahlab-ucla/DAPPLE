"""Tests for normalize ops and snr_peak_pick on synthetic data."""

from __future__ import annotations

import numpy as np

from dapple.io.imzml_reader import read_imzml
from dapple.ops.normalize import MedianNormalize, NormalizeParams, TicNormalize
from dapple.ops.peak_pick import SnrPeakPick, SnrPeakPickParams


def test_median_normalize_changes_intensities(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = MedianNormalize()
    rng = np.random.default_rng(0)
    result = op.apply(ds, op.default_params(ds.metadata), rng=rng)
    new = result.dataset
    # After median normalization, per-pixel median should be ~1.
    medians = new.backend.per_pixel_reduce("median")
    np.testing.assert_allclose(medians[medians > 0], 1.0, rtol=1e-3)


def test_median_normalize_preserves_peak_count(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = MedianNormalize()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    np.testing.assert_array_equal(
        ds.backend.per_pixel_count(), result.dataset.backend.per_pixel_count()
    )


def test_tic_normalize_makes_each_pixel_sum_to_one(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = TicNormalize()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    sums = result.dataset.backend.per_pixel_reduce("sum")
    np.testing.assert_allclose(sums, 1.0, rtol=1e-3)


def test_tic_normalize_emits_validation_warning():
    op = TicNormalize()
    from dapple.data.metadata import ExperimentParams

    warnings = op.validate(
        ExperimentParams(
            instrument_family="tof_reflectron",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="negative",
            mz_min=100,
            mz_max=1000,
        )
    )
    assert any("median" in w.lower() for w in warnings)


def test_normalize_history_records_op(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = MedianNormalize()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    assert result.dataset.history[-1].op_name == "median_normalize"


def test_snr_peak_pick_drops_low_intensity_peaks(synth_centroided):
    ds = read_imzml(synth_centroided)
    # First, double the noise floor by injecting low-intensity garbage peaks would
    # require a different fixture. Instead, set a high SNR threshold and confirm we
    # filter at least some peaks; the synth fixture has tight Gaussian intensity, so
    # MAD * 3 ≈ 0.3 std worth of peaks → most are kept.
    op = SnrPeakPick()
    params = SnrPeakPickParams(snr_mad=0.5, min_intensity_quantile=0.9)
    result = op.apply(ds, params, rng=np.random.default_rng(0))
    n_in = int(ds.backend.offsets[-1])
    n_out = int(result.dataset.backend.offsets[-1])
    assert n_out < n_in  # some peaks dropped
    assert n_out > 0  # but not all


def test_snr_peak_pick_zero_threshold_keeps_everything(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = SnrPeakPick()
    params = SnrPeakPickParams(snr_mad=0.0, min_intensity_quantile=0.0, min_intensity_abs=0.0)
    result = op.apply(ds, params, rng=np.random.default_rng(0))
    n_in = int(ds.backend.offsets[-1])
    n_out = int(result.dataset.backend.offsets[-1])
    assert n_out == n_in


def test_snr_peak_pick_records_diagnostics(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = SnrPeakPick()
    result = op.apply(ds, op.default_params(ds.metadata), rng=np.random.default_rng(0))
    assert result.diagnostics
    summary = result.diagnostics[0].summary
    assert summary["n_peaks_in"] >= summary["n_peaks_out"]
    assert summary["fraction_kept"] <= 1.0
