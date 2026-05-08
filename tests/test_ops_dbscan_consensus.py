"""Tests for dbscan_consensus."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.dataset import PeakMatrix
from dapple.io.imzml_reader import read_imzml
from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
from dapple.ops.dbscan_consensus import (
    DbscanConsensusAlignment,
    DbscanConsensusParams,
)
from dapple.ops.normalize import MedianNormalize
from dapple.ops.peak_pick import SnrPeakPick
from dapple.ops.reference_ions import DetectReferenceIons
from dapple.ops.tolerance import (
    EmpiricalToleranceFromReferenceIons,
    EmpiricalToleranceParams,
)


def _prep(synth_centroided):
    ds = read_imzml(synth_centroided)
    rng = np.random.default_rng(0)
    ds = DetectReferenceIons().apply(
        ds, DetectReferenceIons().default_params(ds.metadata), rng=rng
    ).dataset
    ds = EmpiricalToleranceFromReferenceIons().apply(
        ds, EmpiricalToleranceParams(alpha=0.01, bootstrap_B=20, block_bootstrap=False),
        rng=rng,
    ).dataset
    ds = MedianNormalize().apply(ds, MedianNormalize().default_params(ds.metadata), rng=rng).dataset
    ds = SnrPeakPick().apply(ds, SnrPeakPick().default_params(ds.metadata), rng=rng).dataset
    return ds


def test_dbscan_recovers_synth_peaks(synth_centroided):
    ds = _prep(synth_centroided)
    op = DbscanConsensusAlignment()
    result = op.apply(
        ds,
        DbscanConsensusParams(eps_ppm=200.0, min_samples=5, min_prevalence=0.5),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    assert isinstance(new.backend, PeakMatrix)
    axis = np.asarray(new.backend.mz_axis[:])
    # Synth planted peaks at 200, 250, 300, 350, 400.
    expected = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
    for ex in expected:
        diffs_ppm = np.abs(axis - ex) / ex * 1e6
        assert diffs_ppm.min() < 200, (
            f"DBSCAN missed {ex}: closest cluster was {axis[diffs_ppm.argmin()]:.4f}"
        )


def test_dbscan_uses_tolerance_curve_when_eps_zero(synth_centroided):
    """eps_ppm=0 falls back to 2× the upstream tolerance curve median."""
    ds = _prep(synth_centroided)
    op = DbscanConsensusAlignment()
    result = op.apply(
        ds,
        DbscanConsensusParams(eps_ppm=0.0, min_samples=5, min_prevalence=0.5),
        rng=np.random.default_rng(0),
    )
    eff = result.diagnostics[0].summary["eps_ppm_effective"]
    assert eff > 0


def test_dbscan_produces_similar_count_to_kde(synth_centroided):
    """On the synth fixture, DBSCAN and KDE should both find ~5 consensus peaks."""
    ds = _prep(synth_centroided)
    rng = np.random.default_rng(0)
    kde_result = KdeConsensusAlignment().apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        rng=rng,
    )
    db_result = DbscanConsensusAlignment().apply(
        ds,
        DbscanConsensusParams(eps_ppm=200.0, min_samples=5, min_prevalence=0.5),
        rng=rng,
    )
    n_kde = kde_result.dataset.backend.n_peaks
    n_db = db_result.dataset.backend.n_peaks
    assert abs(n_kde - n_db) <= 2, f"KDE found {n_kde}, DBSCAN found {n_db}"


def test_dbscan_too_strict_raises(synth_centroided):
    ds = _prep(synth_centroided)
    op = DbscanConsensusAlignment()
    with pytest.raises(RuntimeError, match="No DBSCAN consensus peaks survived|no clusters"):
        op.apply(
            ds,
            # Either no clusters (huge min_samples) or no survivors (impossible
            # prevalence) is acceptable — both signal "tighten params".
            DbscanConsensusParams(eps_ppm=10.0, min_samples=10000, min_prevalence=0.5),
            rng=np.random.default_rng(0),
        )


def test_dbscan_default_params_have_labels_and_help():
    from dataclasses import fields

    from dapple.ops.base import field_help, field_label

    for f in fields(DbscanConsensusParams):
        assert field_label(f)
        assert field_help(f)
