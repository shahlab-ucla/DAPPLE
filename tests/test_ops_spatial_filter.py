"""Tests for the morans_i_permutation spatial-coherence filter."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.dataset import MSIDataset, PeakMatrix
from dapple.data.metadata import DatasetIdentity, ExperimentParams
from dapple.ops.spatial_filter import (
    MoransIParams,
    MoransIPermutation,
    _bh_fdr,
)


def _make_grid_dataset(matrix: np.ndarray, h: int, w: int) -> MSIDataset:
    """Build a dense PeakMatrix-backed MSIDataset on a regular h×w grid."""
    n_pixels = h * w
    assert matrix.shape[0] == n_pixels
    coords = np.array(
        [(x + 1, y + 1) for y in range(h) for x in range(w)], dtype=np.int32
    )
    pm = PeakMatrix(
        matrix=matrix.astype(np.float32, copy=False),
        mz_axis=np.arange(matrix.shape[1], dtype=np.float64) + 100.0,
    )
    ep = ExperimentParams(
        instrument_family="tof_reflectron",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=200.0,
    )
    return MSIDataset(
        coords=coords,
        grid_shape=(h, w),
        metadata=ep,
        backend=pm,
        identity=DatasetIdentity(source_path="synth", content_sha256="a" * 64),
    )


def test_morans_i_keeps_strongly_clustered_channel():
    """A channel where the left half is intense and the right half is dark must be
    detected as spatially coherent (Moran's I positive, p-value tiny)."""
    h, w = 16, 16
    rng = np.random.default_rng(0)
    n = h * w
    # Two channels: one clustered (gradient + noise), one pure noise.
    clustered = np.zeros(n, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            clustered[y * w + x] = (1.0 if x < w // 2 else 0.05) + 0.05 * rng.normal()
    noise = rng.normal(size=n).astype(np.float32)
    matrix = np.stack([clustered, noise], axis=1)
    ds = _make_grid_dataset(matrix, h, w)

    op = MoransIPermutation()
    result = op.apply(
        ds,
        MoransIParams(n_permutations=199, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    diag = result.diagnostics[0]
    I_obs = diag.payload["I_obs"]
    p = diag.payload["p_values"]
    q = diag.payload["q_values"]
    assert I_obs[0] > I_obs[1], "clustered channel should have higher Moran's I than noise"
    assert p[0] < p[1]
    # Clustered channel survives; noise should fail FDR.
    kept = diag.payload["kept_mask"]
    assert kept[0]
    # The output's mz_axis should reflect surviving channels.
    assert result.dataset.backend.n_peaks == int(kept.sum())
    n_out = result.dataset.backend.n_peaks
    assert len(result.dataset.extra["morans_i_per_channel"]) == n_out
    assert len(result.dataset.extra["morans_i_p_values"]) == n_out
    assert len(result.dataset.extra["morans_i_q_values"]) == n_out


def test_morans_i_drops_most_pure_noise_channels():
    """A batch of pure-noise channels should mostly fail the FDR cut at q=0.05.

    We can't assert *all* channels are rejected — a noise channel can produce a
    p-value below the BH cut by chance — but the median p-value should be far
    above the threshold and the kept fraction should be small.
    """
    h, w = 16, 16
    rng = np.random.default_rng(0)
    n = h * w
    noise = rng.normal(size=(n, 32)).astype(np.float32)
    ds = _make_grid_dataset(noise, h, w)
    op = MoransIPermutation()
    try:
        result = op.apply(
            ds,
            MoransIParams(n_permutations=199, q_threshold=0.05),
            rng=np.random.default_rng(0),
        )
    except RuntimeError:
        # All-rejected is also a valid outcome on this synthetic input.
        return
    diag = result.diagnostics[0]
    p_values = diag.payload["p_values"]
    # Most p-values are large under the null.
    assert float(np.median(p_values)) > 0.2
    # And only a small fraction of channels survive the FDR cut.
    assert float(diag.summary["n_channels_out"]) / 32 < 0.25


def test_morans_i_passthrough_when_too_few_pixels():
    h, w = 4, 4  # 16 pixels < default min 64
    ds = _make_grid_dataset(np.zeros((16, 2), dtype=np.float32), h, w)
    op = MoransIPermutation()
    result = op.apply(
        ds, MoransIParams(min_pixels_for_test=64), rng=np.random.default_rng(0)
    )
    assert result.dataset.backend.n_peaks == 2  # nothing dropped


def test_morans_i_rook_neighborhood_smaller_than_queen():
    h, w = 16, 16
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(h * w, 1)).astype(np.float32)
    matrix[:, 0] += np.arange(h * w) % w  # add x-gradient → strong rook neighbor coherence
    ds = _make_grid_dataset(matrix, h, w)

    op = MoransIPermutation()
    queen_result = op.apply(
        ds, MoransIParams(neighborhood="queen", n_permutations=99), rng=np.random.default_rng(0)
    )
    rook_result = op.apply(
        ds, MoransIParams(neighborhood="rook", n_permutations=99), rng=np.random.default_rng(0)
    )
    queen_S0 = queen_result.diagnostics[0].summary["S0"]
    rook_S0 = rook_result.diagnostics[0].summary["S0"]
    # Queen neighborhood has more edges than rook, so S0 is larger.
    assert queen_S0 > rook_S0


def test_default_positive_tail_rejects_checkerboard_dispersion():
    h = w = 16
    yy, xx = np.indices((h, w))
    checkerboard = ((xx + yy) % 2).reshape(-1).astype(np.float32)
    clustered = (xx < w // 2).reshape(-1).astype(np.float32)
    ds = _make_grid_dataset(np.column_stack([checkerboard, clustered]), h, w)
    result = MoransIPermutation().apply(
        ds,
        MoransIParams(n_permutations=199, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    diagnostic = result.diagnostics[0]
    assert diagnostic.payload["I_obs"][0] < 0
    assert not diagnostic.payload["kept_mask"][0]
    assert diagnostic.payload["kept_mask"][1]


def test_bh_fdr_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.04, 0.5, 0.6, 0.99])
    q = _bh_fdr(p)
    assert (q <= 1).all()
    assert (q >= p).all()  # BH only inflates p-values


def test_morans_i_default_params_have_labels_and_help():
    from dataclasses import fields

    from dapple.ops.base import field_help, field_label

    for f in fields(MoransIParams):
        assert field_label(f)
        assert field_help(f)
