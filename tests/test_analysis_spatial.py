"""Geometry tests for canonical ROI masks and directed axes."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.analysis.spatial import AxisProjection, DirectedAxis, project_to_axis, rasterize_rois
from dapple.data.dataset import MSIDataset, PeakMatrix
from dapple.data.metadata import DatasetIdentity, ExperimentParams, RoiDef


def _dataset(grid_shape: tuple[int, int]) -> MSIDataset:
    h, w = grid_shape
    coords = np.asarray(
        [(x + 1, y + 1) for y in range(h) for x in range(w)], dtype=np.int32
    )
    matrix = np.ones((h * w, 2), dtype=np.float32)
    return MSIDataset(
        coords=coords,
        grid_shape=grid_shape,
        metadata=ExperimentParams(
            instrument_family="qtof",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="positive",
            mz_min=99.0,
            mz_max=201.0,
        ),
        backend=PeakMatrix(
            matrix=matrix,
            mz_axis=np.asarray([100.0, 200.0], dtype=np.float64),
        ),
        identity=DatasetIdentity(source_path="synthetic", content_sha256="spatial"),
    )


def _overlapping_rois() -> tuple[RoiDef, RoiDef]:
    left = RoiDef(
        name="left",
        vertices=((0.0, 0.0), (0.0, 2.0), (3.0, 2.0), (3.0, 0.0)),
    )
    right = RoiDef(
        name="right",
        vertices=((0.0, 1.0), (0.0, 3.0), (3.0, 3.0), (3.0, 1.0)),
    )
    return left, right


def test_roi_overlap_policy_is_explicit_and_deterministic():
    ds = _dataset((4, 4))
    rois = _overlapping_rois()

    with pytest.raises(ValueError, match="multiple ROIs"):
        rasterize_rois(ds, rois, overlap_policy="error")

    allowed = rasterize_rois(ds, rois, overlap_policy="allow")
    assert allowed.pixel_counts.tolist() == [12, 12]
    assert int(allowed.overlap_mask.sum()) == 8

    excluded = rasterize_rois(ds, rois, overlap_policy="exclude")
    assert excluded.pixel_counts.tolist() == [4, 4]
    assert not np.any(excluded.masks.sum(axis=0) > 1)
    # The original overlap remains available for diagnostics.
    assert int(excluded.overlap_mask.sum()) == 8
    assert excluded.fingerprint != allowed.fingerprint

    first = rasterize_rois(ds, rois, overlap_policy="first")
    assert first.pixel_counts.tolist() == [12, 4]
    assert not np.any(first.masks.sum(axis=0) > 1)


def test_roi_union_and_name_validation():
    ds = _dataset((4, 4))
    left, right = _overlapping_rois()
    masks = rasterize_rois(ds, (left, right), overlap_policy="allow")
    np.testing.assert_array_equal(masks.union("left"), masks.masks[0])
    np.testing.assert_array_equal(
        masks.union(("left", "right")), np.any(masks.masks, axis=0)
    )
    with pytest.raises(KeyError, match="unknown ROI"):
        masks.union("missing")
    duplicate = RoiDef(name="left", vertices=right.vertices)
    with pytest.raises(ValueError, match="unique"):
        rasterize_rois(ds, (left, duplicate), overlap_policy="allow")


def test_directed_axis_projection_and_reversal_are_exact():
    ds = _dataset((3, 5))
    axis = DirectedAxis(
        name="anterior-posterior",
        start_yx=(1.0, 0.0),
        end_yx=(1.0, 4.0),
        start_label="anterior",
        end_label="posterior",
        half_width_px=0.1,
    )
    forward = project_to_axis(ds, axis)
    middle_row = np.asarray([5, 6, 7, 8, 9])
    assert int(forward.included_mask.sum()) == 5
    np.testing.assert_allclose(forward.t[middle_row], np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(forward.signed_distance_px[middle_row], 0.0)
    assert np.isnan(forward.t[~forward.included_mask]).all()

    reverse = project_to_axis(ds, axis.reversed())
    np.testing.assert_array_equal(reverse.included_mask, forward.included_mask)
    np.testing.assert_allclose(reverse.t[middle_row], 1.0 - forward.t[middle_row])
    np.testing.assert_allclose(
        reverse.signed_distance_px[middle_row],
        -forward.signed_distance_px[middle_row],
    )
    assert reverse.axis.start_label == "posterior"
    assert reverse.axis.end_label == "anterior"


def test_axis_half_width_and_include_mask_compose():
    ds = _dataset((3, 5))
    axis = DirectedAxis(
        name="horizontal", start_yx=(1.0, 0.0), end_yx=(1.0, 4.0), half_width_px=1.0
    )
    include = np.ones(ds.n_pixels, dtype=bool)
    include[[5, 9]] = False
    projection = project_to_axis(ds, axis, include_mask=include)
    assert int(projection.included_mask.sum()) == ds.n_pixels - 2
    assert not projection.included_mask[5]
    assert not projection.included_mask[9]


def test_degenerate_axis_and_bad_mask_fail_clearly():
    with pytest.raises(ValueError, match="distinct"):
        DirectedAxis(name="bad", start_yx=(1.0, 1.0), end_yx=(1.0, 1.0))
    ds = _dataset((3, 5))
    axis = DirectedAxis(name="ok", start_yx=(1.0, 0.0), end_yx=(1.0, 4.0))
    with pytest.raises(ValueError, match="include_mask shape"):
        project_to_axis(ds, axis, include_mask=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="finite and non-negative"):
        DirectedAxis(
            name="nan-width",
            start_yx=(1.0, 0.0),
            end_yx=(1.0, 4.0),
            half_width_px=float("nan"),
        )


def test_axis_projection_rejects_included_nan_coordinate():
    ds = _dataset((3, 5))
    axis = DirectedAxis(name="ok", start_yx=(1.0, 0.0), end_yx=(1.0, 4.0))
    projection = project_to_axis(ds, axis)
    bad_t = projection.t.copy()
    bad_t[np.flatnonzero(projection.included_mask)[0]] = np.nan
    with pytest.raises(ValueError, match="finite t"):
        AxisProjection(
            axis=axis,
            t=bad_t,
            signed_distance_px=projection.signed_distance_px,
            included_mask=projection.included_mask,
            coordinate_fingerprint=projection.coordinate_fingerprint,
        )
