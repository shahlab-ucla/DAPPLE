"""Tests for the background-subtraction operator."""

from __future__ import annotations

from dataclasses import replace as drep

import numpy as np
import pytest

from dapple.data.dataset import PeakMatrix
from dapple.data.metadata import RoiDef
from dapple.io.imzml_reader import read_imzml
from dapple.ops.background import (
    BackgroundSubtract,
    BackgroundSubtractParams,
    _foreground_background_masks,
)


def _aligned_with_rois(synth_centroided):
    """Run consensus alignment then return a PeakMatrix-backed dataset with two ROIs:
    a foreground polygon over the upper-left 3x3 corner and a background polygon over
    the lower-right 2x2 corner of the 5x5 synthetic image."""
    from dapple.ops.consensus import KdeConsensusAlignment, KdeConsensusParams
    from dapple.pipeline import PipelineRunner, recommend_pipeline
    from dapple.pipeline.pipeline import Node

    ds = read_imzml(synth_centroided)
    p = recommend_pipeline(ds.metadata)
    nodes = list(p.nodes)
    nodes[-1] = Node(
        id="consensus",
        op_name="kde_consensus_alignment",
        params=KdeConsensusParams(default_tol_ppm=200.0, bandwidth_ppm=20.0, min_prevalence=0.5),
        upstream=("pick",),
    )
    p = drep(p, nodes=tuple(nodes))
    aligned = PipelineRunner().run(p, ds).output
    fg = RoiDef(
        name="foreground", vertices=((0.0, 0.0), (0.0, 2.5), (2.5, 2.5), (2.5, 0.0))
    )
    bg = RoiDef(
        name="background",
        vertices=((3.0, 3.0), (3.0, 4.5), (4.5, 4.5), (4.5, 3.0)),
        is_background=True,
    )
    return drep(aligned, rois=(fg, bg))


def test_background_subtract_rejects_high_ratio_channels(synth_centroided):
    """Inject one channel with intensity in both fg and bg, and one channel only in fg.
    The bg-only channel should be rejected; the fg-only one should be kept."""
    aligned = _aligned_with_rois(synth_centroided)
    pm: PeakMatrix = aligned.backend
    matrix = np.asarray(pm.matrix[:]).copy()
    # All synth peaks are present in every pixel by construction. Make channel 0 a
    # "bg-only" peak (high in bg pixels, near-zero in fg) and channel 1 a "fg-only"
    # (high in fg, near-zero in bg).
    fg_mask = np.zeros(pm.n_pixels, dtype=bool)
    bg_mask = np.zeros(pm.n_pixels, dtype=bool)
    coords = aligned.coords
    for i in range(pm.n_pixels):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        if x <= 3 and y <= 3:
            fg_mask[i] = True
        elif x >= 4 and y >= 4:
            bg_mask[i] = True
    matrix[bg_mask, 0] = matrix[bg_mask, 0] * 100.0
    matrix[fg_mask, 0] = matrix[fg_mask, 0] * 0.001  # almost gone in fg
    matrix[fg_mask, 1] = matrix[fg_mask, 1] * 100.0
    matrix[bg_mask, 1] = matrix[bg_mask, 1] * 0.001
    aligned = aligned.with_backend(
        PeakMatrix(matrix=matrix.astype(np.float32, copy=False), mz_axis=pm.mz_axis)
    )

    op = BackgroundSubtract()
    rng = np.random.default_rng(0)
    result = op.apply(
        aligned,
        BackgroundSubtractParams(mode="reject_channels", bg_to_fg_ratio_threshold=0.5),
        rng=rng,
    )
    out_axis = list(np.asarray(result.dataset.backend.mz_axis[:]))
    in_axis = list(np.asarray(pm.mz_axis[:]))
    # Channel 0 (bg-dominant) must be removed.
    assert in_axis[0] not in out_axis
    # Channel 1 (fg-dominant) must survive.
    assert in_axis[1] in out_axis


def test_background_subtract_subtract_mode_clips_at_zero(synth_centroided):
    aligned = _aligned_with_rois(synth_centroided)
    n_channels = aligned.backend.n_peaks
    aligned = drep(
        aligned,
        extra={
            **aligned.extra,
            "morans_i_per_channel": np.ones(n_channels),
            "cohort_prevalence": np.ones(n_channels),
        },
    )
    op = BackgroundSubtract()
    rng = np.random.default_rng(0)
    result = op.apply(
        aligned,
        BackgroundSubtractParams(mode="subtract"),
        rng=rng,
    )
    out_matrix = np.asarray(result.dataset.backend.matrix[:])
    assert (out_matrix >= 0).all()
    # Same channel count as input.
    assert out_matrix.shape == np.asarray(aligned.backend.matrix[:]).shape
    np.testing.assert_array_equal(
        result.dataset.extra["consensus_n_peaks_per_channel"],
        (out_matrix > 0).sum(axis=0),
    )
    assert "morans_i_per_channel" not in result.dataset.extra
    assert "cohort_prevalence" not in result.dataset.extra


def test_background_subtract_requires_rois(synth_centroided):
    aligned = _aligned_with_rois(synth_centroided)
    aligned = drep(aligned, rois=())
    op = BackgroundSubtract()
    with pytest.raises(RuntimeError, match="ROI"):
        op.apply(aligned, BackgroundSubtractParams(), rng=np.random.default_rng(0))


def test_background_subtract_requires_peakmatrix(synth_centroided):
    """Pre-consensus PeakList raises a clear error."""
    ds = read_imzml(synth_centroided)
    fg = RoiDef(name="fg", vertices=((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)))
    ds = drep(ds, rois=(fg,))
    op = BackgroundSubtract()
    with pytest.raises(RuntimeError, match="PeakMatrix"):
        op.apply(ds, BackgroundSubtractParams(), rng=np.random.default_rng(0))


def test_background_subtract_outside_as_bg_default(synth_centroided):
    """When only a foreground ROI is drawn, outside-ROI pixels act as background."""
    aligned = _aligned_with_rois(synth_centroided)
    fg_only = drep(aligned, rois=(aligned.rois[0],))
    op = BackgroundSubtract()
    # With use_outside_as_bg=True the operator should still run.
    result = op.apply(
        fg_only,
        BackgroundSubtractParams(
            mode="subtract", use_outside_as_bg=True
        ),
        rng=np.random.default_rng(0),
    )
    assert result.dataset.backend.n_peaks == aligned.backend.n_peaks


def test_background_subtract_outside_as_bg_disabled_with_no_bg_roi_errors(synth_centroided):
    aligned = _aligned_with_rois(synth_centroided)
    fg_only = drep(aligned, rois=(aligned.rois[0],))
    op = BackgroundSubtract()
    with pytest.raises(RuntimeError, match="background"):
        op.apply(
            fg_only,
            BackgroundSubtractParams(use_outside_as_bg=False),
            rng=np.random.default_rng(0),
        )


def test_overlapping_foreground_rois_are_combined_as_one_class(synth_centroided):
    aligned = _aligned_with_rois(synth_centroided)
    first = RoiDef(
        name="fg_a",
        vertices=((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)),
    )
    second = RoiDef(
        name="fg_b",
        vertices=((1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)),
    )
    ds = drep(aligned, rois=(first, second))

    fg_mask, bg_mask = _foreground_background_masks(
        ds, use_outside_as_bg=True
    )

    assert fg_mask.any()
    assert bg_mask.any()
    assert not np.any(fg_mask & bg_mask)


def test_overlapping_background_rois_are_combined_as_one_class(synth_centroided):
    aligned = _aligned_with_rois(synth_centroided)
    foreground = RoiDef(
        name="fg",
        vertices=((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
    )
    first_bg = RoiDef(
        name="bg_a",
        vertices=((2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0)),
        is_background=True,
    )
    second_bg = RoiDef(
        name="bg_b",
        vertices=((3.0, 2.0), (3.0, 4.0), (4.0, 4.0), (4.0, 2.0)),
        is_background=True,
    )
    ds = drep(aligned, rois=(foreground, first_bg, second_bg))

    fg_mask, bg_mask = _foreground_background_masks(
        ds, use_outside_as_bg=False
    )

    assert fg_mask.any()
    assert bg_mask.any()
    assert not np.any(fg_mask & bg_mask)


def test_cross_class_roi_overlap_is_rejected(synth_centroided):
    aligned = _aligned_with_rois(synth_centroided)
    foreground = RoiDef(
        name="fg",
        vertices=((0.0, 0.0), (0.0, 3.0), (3.0, 3.0), (3.0, 0.0)),
    )
    background = RoiDef(
        name="bg",
        vertices=((2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0)),
        is_background=True,
    )
    ds = drep(aligned, rois=(foreground, background))

    with pytest.raises(RuntimeError, match="both semantic classes"):
        _foreground_background_masks(ds, use_outside_as_bg=False)


def test_background_subtract_default_params_have_labels_and_help():
    from dataclasses import fields
    from dapple.ops.base import field_help, field_label

    for f in fields(BackgroundSubtractParams):
        assert field_label(f) and field_label(f) != f.name
        assert field_help(f)
