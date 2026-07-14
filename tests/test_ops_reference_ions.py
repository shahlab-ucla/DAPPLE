"""Tests for detect_reference_ions on synthetic data.

The synth dataset puts 5 known peaks at fixed m/z in every pixel — they should all be
detected as reference ions with prevalence ≈ 1.0.
"""

from __future__ import annotations

import numpy as np

from dapple.io.imzml_reader import read_imzml
from dapple.ops.reference_ions import DetectReferenceIons, ReferenceSet


def test_detect_reference_ions_finds_all_peaks(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = DetectReferenceIons()
    params = op.default_params(ds.metadata)
    rng = np.random.default_rng(0)
    result = op.apply(ds, params, rng=rng)

    ref: ReferenceSet = result.dataset.extra["reference_set"]
    # Synthetic data has 5 peaks present in 100% of pixels.
    assert ref.mz.shape[0] == 5
    assert (ref.prevalence > 0.95).all()

    # Detected centroids should be within 10 ppm of the planted values (5 ppm jitter
    # average, 10 ppm cushion).
    expected = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
    ppm_err = (ref.mz - expected) / expected * 1e6
    assert np.abs(ppm_err).max() < 10.0


def test_detect_reference_ions_appends_history(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = DetectReferenceIons()
    params = op.default_params(ds.metadata)
    rng = np.random.default_rng(0)
    result = op.apply(ds, params, rng=rng)

    assert len(result.dataset.history) == 1
    record = result.dataset.history[0]
    assert record.op_name == "detect_reference_ions"
    # Reference detection does not consume ROI geometry, so its cache/provenance
    # fingerprint intentionally ignores annotations.
    assert record.input_hash == ds.hash(include_rois=False)
    assert record.output_hash != record.input_hash
    assert any(d.name == "detect_reference_ions" for d in result.diagnostics)


def test_detect_reference_ions_high_prevalence_threshold(synth_centroided):
    """Threshold of 1.01 (impossible) should yield zero reference ions."""
    from dapple.ops.reference_ions import ReferenceIonsParams

    ds = read_imzml(synth_centroided)
    op = DetectReferenceIons()
    rng = np.random.default_rng(0)
    params = ReferenceIonsParams(coarse_tol_ppm=50.0, min_prevalence=1.01, min_count=5)
    result = op.apply(ds, params, rng=rng)
    ref: ReferenceSet = result.dataset.extra["reference_set"]
    assert ref.mz.shape[0] == 0


def test_detect_reference_ions_default_params_by_instrument():
    from dapple.data.metadata import ExperimentParams

    op = DetectReferenceIons()

    orbitrap_params = op.default_params(
        ExperimentParams(
            instrument_family="orbitrap",
            ionization="esi",
            profile_or_centroided="profile",
            polarity="positive",
            mz_min=100,
            mz_max=1000,
        )
    )
    assert orbitrap_params.coarse_tol_ppm == 5.0
    assert orbitrap_params.min_prevalence == 0.7  # profile

    tof_axial_params = op.default_params(
        ExperimentParams(
            instrument_family="tof_axial",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="negative",
            mz_min=100,
            mz_max=1000,
        )
    )
    assert tof_axial_params.coarse_tol_ppm == 200.0
    assert tof_axial_params.min_prevalence == 0.8  # centroided
