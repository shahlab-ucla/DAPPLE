"""Tests for ``PrevalenceFdrFilter`` — permutation-null FDR test for the prevalence filter.

Pipeline contract:

- The consensus operators (KDE, DBSCAN) record ``consensus_n_peaks_per_channel``
  in ``ds.extra``. The filter reads it as the null's "ball count" per channel.
- For each channel: simulate B uniform placements of ``k_c`` peaks into
  ``n_pixels`` pixels, count distinct pixels (occupancy), convert to
  prevalence. Compare to observed.
- Right-tail p-value with +1/+1 stabilizer, BH-FDR across channels, drop
  channels above ``q_threshold``.
"""

from __future__ import annotations

from dataclasses import replace as drep

import numpy as np
import pytest

from dapple.data.dataset import MSIDataset, PeakMatrix
from dapple.data.metadata import DatasetIdentity, ExperimentParams
from dapple.io.imzml_reader import read_imzml
from dapple.ops.base import field_help, field_label
from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
from dapple.ops.prevalence_filter import (
    PrevalenceFdrFilter,
    PrevalenceFdrParams,
)


# ---- helpers ----------------------------------------------------------------


def _aligned_dataset(synth_centroided):
    """Run the recommended pipeline up to KDE consensus; return PeakMatrix dataset."""
    from dapple.pipeline import PipelineRunner, recommend_pipeline
    from dapple.pipeline.pipeline import Node

    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    # Bump consensus to capture all 5 planted synth peaks reliably.
    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(
            default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.3
        ),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))
    return PipelineRunner().run(p, ds).output


def _make_fake_peakmatrix_dataset(
    n_pixels: int,
    prevalences: list[float],
    peaks_per_channel: list[int],
) -> MSIDataset:
    """Build a synthetic post-consensus dataset with hand-picked prevalences.

    ``prevalences[c]`` is the observed prevalence of channel c (fraction of
    pixels with non-zero intensity). ``peaks_per_channel[c]`` is the total
    peak count attributed to channel c (i.e. the null's ball count).

    Used to drive the FDR test directly without going through consensus
    alignment.
    """
    assert len(prevalences) == len(peaks_per_channel)
    n_channels = len(prevalences)
    rng = np.random.default_rng(0)
    matrix = np.zeros((n_pixels, n_channels), dtype=np.float32)
    for c, p in enumerate(prevalences):
        k = int(round(n_pixels * p))
        which = rng.choice(n_pixels, size=k, replace=False)
        matrix[which, c] = 1.0
    coords = np.column_stack(
        (np.arange(n_pixels, dtype=np.int32) % 100,
         np.arange(n_pixels, dtype=np.int32) // 100)
    )
    grid = (max(int(coords[:, 1].max()) + 1, 1), max(int(coords[:, 0].max()) + 1, 1))
    return MSIDataset(
        coords=coords,
        grid_shape=grid,
        metadata=ExperimentParams(
            instrument_family="tof_reflectron", ionization="maldi",
            profile_or_centroided="centroided", polarity="positive",
            mz_min=100.0, mz_max=1000.0,
        ),
        backend=PeakMatrix(
            matrix=matrix,
            mz_axis=np.linspace(200.0, 800.0, n_channels, dtype=np.float64),
        ),
        identity=DatasetIdentity(source_path="<synth>", content_sha256="x" * 64),
        history=(),
        rois=(),
        rng_seed=0,
        extra={
            "consensus_prevalence": np.asarray(prevalences, dtype=np.float64),
            "consensus_n_peaks_per_channel": np.asarray(peaks_per_channel, dtype=np.int64),
        },
    )


# ---- params -----------------------------------------------------------------


def test_default_params_have_labels_and_help():
    from dataclasses import fields

    for f in fields(PrevalenceFdrParams):
        assert field_label(f), f"missing label on {f.name}"
        assert field_help(f), f"missing help on {f.name}"


# ---- consensus operator records n_peaks_per_channel -------------------------


def test_consensus_operator_records_n_peaks_per_channel(synth_centroided):
    """The KDE consensus operator must attach ``consensus_n_peaks_per_channel``
    so this filter has the null's ball count available."""
    aligned = _aligned_dataset(synth_centroided)
    assert "consensus_n_peaks_per_channel" in aligned.extra, (
        "the KDE consensus operator must record per-channel total peak counts "
        "for downstream FDR tests"
    )
    k = np.asarray(aligned.extra["consensus_n_peaks_per_channel"])
    assert k.shape == (aligned.backend.n_peaks,)
    assert (k > 0).all(), "every consensus channel must have at least one peak"


# ---- end-to-end: synth fixture (high-prevalence channels survive) -----------


def test_synth_aligned_data_passes_through_with_all_channels_kept(synth_centroided):
    """The synth fixture has 5 planted peaks each present in every pixel — far
    above the random-placement null. Every channel should survive."""
    aligned = _aligned_dataset(synth_centroided)
    op = PrevalenceFdrFilter()
    result = op.apply(
        aligned,
        PrevalenceFdrParams(n_permutations=99, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    assert isinstance(new.backend, PeakMatrix)
    assert new.backend.n_peaks == aligned.backend.n_peaks, (
        "all planted-peak channels should be retained (their prevalence ≈ 1 is "
        "far above the random-placement null)"
    )


# ---- synthetic dataset: high-prevalence channels survive --------------------


def test_filter_keeps_high_prevalence_channels():
    """A channel with prevalence 0.9 from 100 peaks should be deeply
    significant — its q-value should be far below the default 0.05 threshold."""
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=200,
        prevalences=[0.9, 0.85, 0.95],
        peaks_per_channel=[100, 95, 110],
    )
    op = PrevalenceFdrFilter()
    result = op.apply(
        ds,
        PrevalenceFdrParams(n_permutations=199, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    assert new.backend.n_peaks == 3, "all three high-prevalence channels should survive"
    diag = result.diagnostics[0]
    assert diag.summary["n_channels_in"] == 3
    assert diag.summary["n_channels_out"] == 3
    assert diag.summary["n_dropped_by_fdr"] == 0


def test_filter_drops_low_prevalence_random_channels():
    """A channel where many peaks scatter to few pixels (prevalence ≈ k/n_pixels)
    matches the random-placement null — it should fail to reject."""
    n_pixels = 200
    # A channel with k = 100 peaks placed truly randomly across 200 bins yields
    # ~63% prevalence (occupancy with k = n bins gives 1 - 1/e ≈ 63%). Setting
    # observed prevalence at the random-expected level produces large p-values.
    expected = 1 - (1 - 1 / n_pixels) ** 100  # ~0.633
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=n_pixels,
        prevalences=[expected, expected, expected],
        peaks_per_channel=[100, 100, 100],
    )
    op = PrevalenceFdrFilter()
    with pytest.raises(RuntimeError, match="every channel rejected"):
        op.apply(
            ds,
            PrevalenceFdrParams(n_permutations=499, q_threshold=0.05),
            rng=np.random.default_rng(0),
        )


def test_filter_separates_real_from_noise():
    """Mix two clearly-real (prevalence-far-above-null) channels with two
    noise channels (prevalence-far-below-null) and verify only the real ones
    survive.

    Sizing: with n_pixels=200 and k=200 peaks per channel placed uniformly at
    random, the null occupancy expectation is ``n * (1 - (1 - 1/n)^k) ≈ 0.634``
    with SD ≈ 0.035. So:

    - Observed prevalence 0.95 (real channels) is +9 SD above null → tiny p.
    - Observed prevalence 0.50 (noise channels) is -4 SD *below* null →
      right-tail p ≈ 1, q ≈ 1, dropped.
    """
    n_pixels = 200
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=n_pixels,
        prevalences=[0.95, 0.95, 0.50, 0.50],
        peaks_per_channel=[200, 200, 200, 200],
    )
    op = PrevalenceFdrFilter()
    result = op.apply(
        ds,
        PrevalenceFdrParams(n_permutations=499, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    # Real channels (indices 0, 1) survive; noise (2, 3) dropped.
    assert new.backend.n_peaks == 2
    np.testing.assert_array_equal(
        np.asarray(new.backend.mz_axis[:]),
        np.asarray(ds.backend.mz_axis[:])[:2],
    )


# ---- diagnostic payload + companion arrays ----------------------------------


def test_diagnostic_summary_keys_are_present():
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=200,
        prevalences=[0.95, 0.9],
        peaks_per_channel=[120, 110],
    )
    result = PrevalenceFdrFilter().apply(
        ds,
        PrevalenceFdrParams(n_permutations=99),
        rng=np.random.default_rng(0),
    )
    s = result.diagnostics[0].summary
    for k in (
        "n_channels_in", "n_channels_out", "n_dropped_by_fdr", "n_permutations",
        "q_threshold", "p_value_min", "p_value_median", "q_value_min", "q_value_median",
    ):
        assert k in s, f"missing summary key {k!r}"


def test_companion_arrays_subset_after_filter():
    """After filtering, ``consensus_prevalence`` and
    ``consensus_n_peaks_per_channel`` in the output's extras must be the
    subset matching the kept channels."""
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=200,
        prevalences=[0.95, 0.95, 0.20, 0.20],
        peaks_per_channel=[200, 200, 50, 50],
    )
    result = PrevalenceFdrFilter().apply(
        ds,
        PrevalenceFdrParams(n_permutations=499, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    new = result.dataset
    assert len(new.extra["consensus_prevalence"]) == new.backend.n_peaks
    assert len(new.extra["consensus_n_peaks_per_channel"]) == new.backend.n_peaks
    # Top two channels had prevalence 0.95.
    np.testing.assert_allclose(
        np.asarray(new.extra["consensus_prevalence"]),
        np.array([0.95, 0.95]),
        atol=0.01,
    )


# ---- fallback paths ---------------------------------------------------------


def test_falls_back_to_passthrough_on_tiny_dataset():
    """Below ``min_pixels_for_test`` the operator is a no-op (the null
    distribution is too coarse to discriminate)."""
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=10,  # below default min 64
        prevalences=[0.9, 0.9, 0.9],
        peaks_per_channel=[10, 10, 10],
    )
    result = PrevalenceFdrFilter().apply(
        ds,
        PrevalenceFdrParams(),
        rng=np.random.default_rng(0),
    )
    # Passthrough — same channel count.
    assert result.dataset.backend.n_peaks == 3


def test_fallback_conservative_when_n_peaks_missing():
    """If ``consensus_n_peaks_per_channel`` is missing from extras (legacy
    dataset), the operator falls back to the observed non-zero count and
    flags the conservative-fallback warning."""
    n_pixels = 200
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=n_pixels,
        prevalences=[0.95, 0.95],
        peaks_per_channel=[200, 200],
    )
    # Drop the key from extras to simulate a legacy dataset.
    bad_extra = {k: v for k, v in ds.extra.items() if k != "consensus_n_peaks_per_channel"}
    ds = ds.__class__(
        coords=ds.coords, grid_shape=ds.grid_shape, metadata=ds.metadata,
        backend=ds.backend, identity=ds.identity, history=ds.history,
        rois=ds.rois, rng_seed=ds.rng_seed, extra=bad_extra,
    )
    result = PrevalenceFdrFilter().apply(
        ds,
        PrevalenceFdrParams(n_permutations=99, q_threshold=0.05),
        rng=np.random.default_rng(0),
    )
    assert result.diagnostics[0].summary.get("warning_conservative_fallback") == 1.0


def test_requires_peakmatrix_backend(synth_centroided):
    """Running on a PeakList dataset (pre-consensus) is rejected."""
    ds = read_imzml(synth_centroided)
    op = PrevalenceFdrFilter()
    with pytest.raises(RuntimeError, match="PeakMatrix"):
        op.apply(ds, PrevalenceFdrParams(), rng=np.random.default_rng(0))


# ---- determinism ------------------------------------------------------------


def test_deterministic_under_same_seed():
    """Same params + same rng_seed in OpParams should produce same q-values."""
    ds = _make_fake_peakmatrix_dataset(
        n_pixels=200,
        prevalences=[0.95, 0.5, 0.9],
        peaks_per_channel=[200, 110, 180],
    )
    op = PrevalenceFdrFilter()
    p = PrevalenceFdrParams(n_permutations=99, rng_seed=42)
    r1 = op.apply(ds, p, rng=np.random.default_rng(0))
    r2 = op.apply(ds, p, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(
        r1.diagnostics[0].payload["q_values"],
        r2.diagnostics[0].payload["q_values"],
    )
