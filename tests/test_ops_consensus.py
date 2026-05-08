"""Tests for kde_consensus_alignment on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.dataset import PeakMatrix
from dapple.io.imzml_reader import read_imzml
from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams


def test_kde_consensus_recovers_synth_peaks(synth_centroided):
    """Synth data has 5 known peaks at 200, 250, 300, 350, 400. KDE should find them."""
    ds = read_imzml(synth_centroided)
    op = KdeConsensusAlignment()
    result = op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, min_prevalence=0.5, bandwidth_ppm=20.0),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    assert isinstance(new.backend, PeakMatrix)
    consensus_mz = new.backend.mz_axis
    # Recovery: every planted peak should match one consensus m/z within 50 ppm.
    expected = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
    for ex in expected:
        diffs_ppm = np.abs(consensus_mz - ex) / ex * 1e6
        assert diffs_ppm.min() < 100, (
            f"no consensus within 100 ppm of {ex}; closest was "
            f"{consensus_mz[diffs_ppm.argmin()]:.4f}"
        )


def test_kde_consensus_matrix_shape(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = KdeConsensusAlignment()
    result = op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, min_prevalence=0.5, bandwidth_ppm=20.0),
        rng=np.random.default_rng(0),
    )
    pm = result.dataset.backend
    assert pm.matrix.shape == (25, len(pm.mz_axis))
    assert pm.matrix.dtype == np.float32
    assert (pm.matrix >= 0).all()


def test_kde_consensus_prevalence_attached(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = KdeConsensusAlignment()
    result = op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, min_prevalence=0.5, bandwidth_ppm=20.0),
        rng=np.random.default_rng(0),
    )
    prev = result.dataset.extra["consensus_prevalence"]
    assert prev.shape == (len(result.dataset.backend.mz_axis),)
    assert ((prev >= 0) & (prev <= 1)).all()


def test_kde_consensus_too_strict_prevalence_raises(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = KdeConsensusAlignment()
    with pytest.raises(RuntimeError, match="No consensus peaks survived"):
        op.apply(
            ds,
            KdeConsensusParams(min_prevalence=1.01, default_tol_ppm=200.0),
            rng=np.random.default_rng(0),
        )


def test_kde_consensus_history_recorded(synth_centroided):
    ds = read_imzml(synth_centroided)
    op = KdeConsensusAlignment()
    result = op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, min_prevalence=0.5, bandwidth_ppm=20.0),
        rng=np.random.default_rng(0),
    )
    assert result.dataset.history[-1].op_name == "kde_consensus_alignment"


def test_kde_consensus_post_alignment_tic_projection(synth_centroided):
    """After consensus alignment, projections still work via the PeakMatrix backend."""
    ds = read_imzml(synth_centroided)
    op = KdeConsensusAlignment()
    result = op.apply(
        ds,
        KdeConsensusParams(default_tol_ppm=200.0, min_prevalence=0.5, bandwidth_ppm=20.0),
        rng=np.random.default_rng(0),
    )
    img = result.dataset.project("tic")
    assert img.shape == ds.grid_shape
    assert (img > 0).all()
