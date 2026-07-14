"""Synthetic planted-pattern tests for ROI and directed-axis analyses."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dapple.analysis.developmental import (
    analyze_axis_profiles,
    analyze_roi_enrichment,
    benjamini_hochberg,
)
from dapple.analysis.export import export_analysis_result
from dapple.analysis.spatial import DirectedAxis, project_to_axis, rasterize_rois
from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.data.metadata import DatasetIdentity, ExperimentParams, RoiDef


def _dataset(matrix: np.ndarray, grid_shape: tuple[int, int]) -> MSIDataset:
    h, w = grid_shape
    coords = np.asarray(
        [(x + 1, y + 1) for y in range(h) for x in range(w)], dtype=np.int32
    )
    k = matrix.shape[1]
    return MSIDataset(
        coords=coords,
        grid_shape=grid_shape,
        metadata=ExperimentParams(
            instrument_family="qtof",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="positive",
            mz_min=99.0,
            mz_max=100.0 + 100.0 * k,
            sample_type="whole_organism",
        ),
        backend=PeakMatrix(
            matrix=np.asarray(matrix, dtype=np.float32),
            mz_axis=np.arange(1, k + 1, dtype=np.float64) * 100.0,
        ),
        identity=DatasetIdentity(source_path="synthetic", content_sha256="analysis"),
    )


def _left_right_rois(height: int, width: int) -> tuple[RoiDef, RoiDef]:
    split = width / 2.0
    return (
        RoiDef(
            name="left",
            vertices=(
                (0.0, 0.0),
                (0.0, split - 0.51),
                (height - 1.0, split - 0.51),
                (height - 1.0, 0.0),
            ),
        ),
        RoiDef(
            name="right",
            vertices=(
                (0.0, split - 0.49),
                (0.0, width - 1.0),
                (height - 1.0, width - 1.0),
                (height - 1.0, split - 0.49),
            ),
        ),
    )


def test_roi_enrichment_recovers_planted_effects_and_frames():
    rng = np.random.default_rng(13)
    h, w = 8, 8
    x_coord = np.tile(np.arange(w), h)
    left = x_coord < w // 2
    noise = rng.normal(0.0, 0.3, size=(h * w, 4))
    matrix = np.empty((h * w, 4), dtype=np.float64)
    matrix[:, 0] = np.where(left, 30.0, 2.0) + noise[:, 0]
    matrix[:, 1] = np.where(left, 2.0, 30.0) + noise[:, 1]
    # An exact null channel with within-group variance but identical left/right
    # distributions, so the test is deterministic rather than occasionally lucky.
    matrix[:, 2] = 8.0 + np.repeat(rng.normal(0.0, 0.3, h), w)
    matrix[:, 3] = np.where(left, 4.0 + np.abs(noise[:, 3]), 0.0)
    matrix = np.clip(matrix, 0.0, None)
    ds = _dataset(matrix, (h, w))
    masks = rasterize_rois(ds, _left_right_rois(h, w), overlap_policy="error")

    result = analyze_roi_enrichment(
        ds,
        masks,
        "left",
        "right",
        chunk_channels=2,
    )

    assert result.inference_status == "ok"
    assert result.n_numerator == result.n_denominator == 32
    assert result.log2_fold_change[0] > 2.5
    assert result.log2_fold_change[1] < -2.5
    assert abs(result.log2_fold_change[2]) < 0.2
    assert result.prevalence_numerator[3] == 1.0
    assert result.prevalence_denominator[3] == 0.0
    assert result.q_value[0] < 0.01
    assert result.q_value[1] < 0.01
    assert result.q_value[2] > 0.05
    assert all(pc > 0 for pc in result.pseudocount)
    frame = result.to_frame()
    assert frame.shape[0] == 4
    assert {"mz", "log2_fold_change", "p_value", "q_value"} <= set(frame.columns)
    assert "pixels as units" in result.warnings[0]


def test_roi_enrichment_insufficient_units_is_descriptive_only():
    matrix = np.asarray([[10.0, 1.0], [1.0, 10.0], [5.0, 5.0]], dtype=np.float32)
    ds = _dataset(matrix, (1, 3))
    a = RoiDef(name="a", vertices=((0.0, 0.0), (0.0, 0.4), (0.4, 0.0)))
    b = RoiDef(name="b", vertices=((0.0, 1.6), (0.0, 2.0), (0.4, 2.0)))
    masks = rasterize_rois(ds, (a, b), overlap_policy="error")
    result = analyze_roi_enrichment(ds, masks, "a", "b", min_units_per_group=3)
    assert result.inference_status == "insufficient_units"
    assert np.isnan(result.p_value).all()
    assert np.isnan(result.q_value).all()
    assert np.isfinite(result.log2_fold_change).all()


def test_analysis_requires_harmonized_peakmatrix():
    ds = _axis_dataset()
    n = ds.n_pixels
    raw = replace(
        ds,
        backend=PeakList(
            mz=np.full(n, 100.0, dtype=np.float64),
            intensity=np.ones(n, dtype=np.float32),
            offsets=np.arange(n + 1, dtype=np.int64),
            n_pixels=n,
        ),
    )
    axis = DirectedAxis(name="AP", start_yx=(2.5, 0.0), end_yx=(2.5, 19.0))
    with pytest.raises(RuntimeError, match="PeakMatrix"):
        analyze_axis_profiles(raw, axis)


def _axis_dataset() -> MSIDataset:
    rng = np.random.default_rng(41)
    h, w = 6, 20
    t = np.tile(np.linspace(0.0, 1.0, w), h)
    matrix = np.empty((h * w, 4), dtype=np.float64)
    matrix[:, 0] = 1.0 + 10.0 * t + rng.normal(0.0, 0.08, h * w)
    matrix[:, 1] = 11.0 - 10.0 * t + rng.normal(0.0, 0.08, h * w)
    matrix[:, 2] = 1.0 + 40.0 * np.exp(-((t - 0.5) / 0.055) ** 2)
    matrix[:, 3] = 5.0 + rng.normal(0.0, 0.08, h * w)
    return _dataset(np.clip(matrix, 0.0, None), (h, w))


def test_axis_profiles_find_gradients_localization_and_reversal_invariants():
    ds = _axis_dataset()
    axis = DirectedAxis(
        name="AP",
        start_yx=(2.5, 0.0),
        end_yx=(2.5, 19.0),
        start_label="anterior",
        end_label="posterior",
    )
    forward = analyze_axis_profiles(
        ds,
        axis,
        n_bins=20,
        min_pixels_per_bin=3,
        n_permutations=149,
        rng_seed=17,
        concentration_threshold=0.15,
        chunk_channels=2,
    )
    reverse = analyze_axis_profiles(
        ds,
        axis.reversed(),
        n_bins=20,
        min_pixels_per_bin=3,
        n_permutations=149,
        rng_seed=17,
        concentration_threshold=0.15,
        chunk_channels=2,
    )

    assert forward.inference_status == "ok"
    assert forward.n_permutations_effective == 298
    assert forward.spearman_rho[0] > 0.95
    assert forward.spearman_rho[1] < -0.95
    assert forward.q_value[0] < 0.05
    assert forward.q_value[1] < 0.05
    assert forward.pattern_label[0] == "increasing"
    assert forward.pattern_label[1] == "decreasing"
    assert forward.pattern_label[2] == "interior_localized"
    assert forward.concentration[2] > forward.concentration[3]

    np.testing.assert_array_equal(reverse.bin_counts, forward.bin_counts[::-1])
    np.testing.assert_allclose(
        reverse.mean_intensity, forward.mean_intensity[::-1, :], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(reverse.spearman_rho, -forward.spearman_rho, atol=1e-12)
    np.testing.assert_allclose(reverse.p_value, forward.p_value, atol=0.0)
    np.testing.assert_allclose(reverse.q_value, forward.q_value, atol=0.0)
    np.testing.assert_allclose(
        reverse.endpoint_log2_enrichment,
        -forward.endpoint_log2_enrichment,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reverse.peak_position,
        1.0 - forward.peak_position,
        atol=1e-12,
    )
    np.testing.assert_allclose(reverse.concentration, forward.concentration, atol=1e-12)
    assert reverse.pattern_label[0] == "decreasing"
    assert reverse.pattern_label[1] == "increasing"
    assert reverse.pattern_label[2] == "interior_localized"
    assert reverse.start_label == "posterior"
    assert reverse.end_label == "anterior"

    assert forward.profile_frame().shape[0] == 20 * 4
    assert forward.statistics_frame().shape[0] == 4


def test_axis_inference_handles_too_few_populated_bins():
    ds = _axis_dataset()
    axis = DirectedAxis(name="AP", start_yx=(2.5, 0.0), end_yx=(2.5, 19.0))
    result = analyze_axis_profiles(
        ds,
        axis,
        n_bins=20,
        min_pixels_per_bin=7,
        min_bins_for_trend=4,
        n_permutations=19,
    )
    assert result.inference_status == "insufficient_bins"
    assert result.n_permutations_effective == 0
    assert np.isnan(result.p_value).all()
    assert np.isnan(result.q_value).all()
    assert np.isfinite(result.mean_intensity).any()
    assert all(label != "undetected" for label in result.pattern_label)


def test_fdr_preserves_nan_and_is_monotone_by_rank():
    p = np.asarray([0.01, np.nan, 0.04, 0.2, 1.0])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    finite = np.isfinite(p)
    order = np.argsort(p[finite])
    assert np.all(np.diff(q[finite][order]) >= 0)
    assert np.all(q[finite] >= p[finite])


def test_export_writes_csv_tables_and_json_manifest(tmp_path):
    ds = _axis_dataset()
    axis = DirectedAxis(name="AP", start_yx=(2.5, 0.0), end_yx=(2.5, 19.0))
    result = analyze_axis_profiles(
        ds, axis, n_bins=10, min_pixels_per_bin=3, n_permutations=19, rng_seed=3
    )
    exported = export_analysis_result(result, tmp_path, stem="development")
    assert exported.manifest_path.exists()
    assert all(path.exists() for path in exported.table_paths)
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "dapple-analysis"
    assert manifest["kind"] == "axis_profile"
    assert manifest["axis_name"] == "AP"
    assert manifest["source_dataset_hash"] == ds.hash(include_rois=False)
    assert manifest["spatial_fingerprint"] == result.spatial_fingerprint
    assert manifest["axis_geometry"]["start_yx"] == [2.5, 0.0]
    assert manifest["axis_geometry"]["end_yx"] == [2.5, 19.0]
    assert manifest["analysis_parameters"]["min_pixels_per_bin"] == 3
    assert "python" in manifest["software"]
    assert "numpy" in manifest["software"]
    assert manifest["tables"] == [path.name for path in exported.table_paths]
    profiles = pd.read_csv(exported.table_paths[0])
    statistics = pd.read_csv(exported.table_paths[1])
    assert profiles.shape[0] == 10 * 4
    assert statistics.shape[0] == 4


def test_spatial_objects_are_bound_to_coordinate_row_order():
    ds = _axis_dataset()
    rois = _left_right_rois(*ds.grid_shape)
    masks = rasterize_rois(ds, rois, overlap_policy="error")
    axis = DirectedAxis(name="AP", start_yx=(2.5, 0.0), end_yx=(2.5, 19.0))
    projection = project_to_axis(ds, axis)
    reordered = replace(
        ds,
        coords=ds.coords[::-1].copy(),
        backend=PeakMatrix(
            matrix=np.asarray(ds.backend.matrix[:])[::-1].copy(),
            mz_axis=np.asarray(ds.backend.mz_axis[:]).copy(),
        ),
    )
    with pytest.raises(ValueError, match="different grid coordinates"):
        analyze_roi_enrichment(reordered, masks, "left", "right")
    with pytest.raises(ValueError, match="different grid coordinates"):
        analyze_axis_profiles(reordered, projection, n_permutations=9)


def test_additional_axis_mask_changes_effective_spatial_fingerprint():
    ds = _axis_dataset()
    axis = DirectedAxis(name="AP", start_yx=(2.5, 0.0), end_yx=(2.5, 19.0))
    projection = project_to_axis(ds, axis)
    full = analyze_axis_profiles(ds, projection, n_bins=10, n_permutations=19)
    include = np.ones(ds.n_pixels, dtype=bool)
    include[::2] = False
    subset = analyze_axis_profiles(
        ds,
        projection,
        include_mask=include,
        n_bins=10,
        n_permutations=19,
    )
    assert subset.n_included_pixels < full.n_included_pixels
    assert subset.spatial_fingerprint != full.spatial_fingerprint


def test_axis_endpoint_effect_requires_multiple_pixels_per_end():
    ds = _axis_dataset()
    axis = DirectedAxis(name="AP", start_yx=(0.0, 0.0), end_yx=(0.0, 19.0))
    include = np.zeros(ds.n_pixels, dtype=bool)
    include[:20] = True
    result = analyze_axis_profiles(
        ds,
        axis,
        include_mask=include,
        n_bins=20,
        min_pixels_per_bin=1,
        min_endpoint_pixels=2,
        n_permutations=19,
    )
    assert result.n_start_pixels == result.n_end_pixels == 4
    assert np.isfinite(result.endpoint_log2_enrichment).all()

    narrow = analyze_axis_profiles(
        ds,
        axis,
        include_mask=include,
        n_bins=20,
        min_pixels_per_bin=1,
        endpoint_fraction=0.01,
        min_endpoint_pixels=2,
        n_permutations=19,
    )
    assert narrow.n_start_pixels == narrow.n_end_pixels == 1
    assert np.isnan(narrow.endpoint_log2_enrichment).all()
    assert any("Endpoint enrichment is unavailable" in warning for warning in narrow.warnings)


def test_axis_warns_about_exchangeability_and_permutation_resolution():
    ds = _axis_dataset()
    axis = DirectedAxis(name="AP", start_yx=(2.5, 0.0), end_yx=(2.5, 19.0))
    result = analyze_axis_profiles(
        ds,
        axis,
        n_bins=10,
        min_pixels_per_bin=3,
        n_permutations=9,
    )
    assert any("Spatial autocorrelation" in warning for warning in result.warnings)
    assert any("Permutation resolution" in warning for warning in result.warnings)
