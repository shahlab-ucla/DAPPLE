"""Unit tests for the core data model: ExperimentParams, PeakList, PeakMatrix, MSIDataset."""

from __future__ import annotations

import numpy as np
import pytest

from dapple.data.coords import (
    coords_to_grid_index,
    grid_index_to_coords,
    project_to_grid,
)
from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.data.hashing import combine_hashes, hash_array_content, hash_obj, sha256_file
from dapple.data.metadata import (
    DatasetIdentity,
    ExperimentParams,
    RoiDef,
)


# -------- ExperimentParams ----------------------------------------------------------


def test_experiment_params_basic():
    ep = ExperimentParams(
        instrument_family="tof_reflectron",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=1000.0,
        pixel_size_um=50.0,
    )
    assert ep.mz_min == 100.0
    assert ep.pixel_size_um == 50.0


def test_experiment_params_rejects_inverted_mz_range():
    with pytest.raises(ValueError, match="mz_max"):
        ExperimentParams(
            instrument_family="tof_reflectron",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="negative",
            mz_min=1000.0,
            mz_max=100.0,
        )


def test_experiment_params_rejects_nonpositive_pixel_size():
    with pytest.raises(ValueError, match="pixel_size_um"):
        ExperimentParams(
            instrument_family="tof_reflectron",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="negative",
            mz_min=100.0,
            mz_max=1000.0,
            pixel_size_um=-1.0,
        )


def test_roi_def_rejects_too_few_vertices():
    with pytest.raises(ValueError, match=">=3 vertices"):
        RoiDef(name="r", vertices=((0.0, 0.0), (1.0, 1.0)))


# -------- coords -------------------------------------------------------------------


def test_coords_to_grid_index_one_indexed():
    coords = np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)
    flat = coords_to_grid_index(coords, (3, 3))
    # imzML 1-indexed: (1,1) -> (0,0) -> 0; (2,1) -> (0,1) -> 1; (1,2) -> (1,0) -> 3
    assert list(flat) == [0, 1, 3]


def test_coords_to_grid_index_zero_indexed():
    coords = np.array([[0, 0], [2, 1]], dtype=np.int32)
    flat = coords_to_grid_index(coords, (3, 3))
    assert list(flat) == [0, 5]  # (0,0)->0, (2,1)->1*3+2=5


def test_coords_to_grid_index_sparse_one_indexed_without_origin_pixels():
    coords = np.array([[3, 4], [5, 4], [3, 6]], dtype=np.int32)
    flat = coords_to_grid_index(coords, (8, 8))
    assert list(flat) == [26, 28, 42]


def test_coords_to_grid_index_explicit_zero_origin_for_interior_crop():
    coords = np.array([[3, 4], [5, 4]], dtype=np.int32)
    flat = coords_to_grid_index(coords, (8, 8), origin="zero")
    assert list(flat) == [35, 37]


def test_coords_to_grid_index_out_of_bounds():
    coords = np.array([[5, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="out of grid"):
        coords_to_grid_index(coords, (3, 3))


def test_grid_index_roundtrip():
    coords = np.array([[1, 1], [3, 2], [2, 3]], dtype=np.int32)
    flat = coords_to_grid_index(coords, (3, 3))
    back = grid_index_to_coords(flat, (3, 3))
    # Note: round-trip yields 0-indexed coords
    assert list(back[:, 0]) == [0, 2, 1]
    assert list(back[:, 1]) == [0, 1, 2]


def test_project_to_grid():
    coords = np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)
    vals = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    img = project_to_grid(vals, coords, (3, 3), fill=-1.0)
    assert img[0, 0] == 10.0
    assert img[0, 1] == 20.0
    assert img[1, 0] == 30.0
    assert img[2, 2] == -1.0  # not represented


# -------- PeakList -----------------------------------------------------------------


def _toy_peaklist() -> PeakList:
    mz = np.array([100.0, 200.0, 100.5, 200.5, 100.2, 200.2, 300.0], dtype=np.float64)
    intensity = np.array([10, 100, 20, 200, 30, 300, 999], dtype=np.float32)
    offsets = np.array([0, 2, 4, 7], dtype=np.int64)  # 3 pixels
    return PeakList(mz=mz, intensity=intensity, offsets=offsets, n_pixels=3)


def test_peaklist_pixel_returns_correct_slice():
    pl = _toy_peaklist()
    mz0, i0 = pl.pixel(0)
    np.testing.assert_array_equal(mz0, [100.0, 200.0])
    np.testing.assert_array_equal(i0, [10, 100])
    mz2, i2 = pl.pixel(2)
    np.testing.assert_array_equal(mz2, [100.2, 200.2, 300.0])


def test_peaklist_per_pixel_count():
    pl = _toy_peaklist()
    np.testing.assert_array_equal(pl.per_pixel_count(), [2, 2, 3])


def test_peaklist_per_pixel_reduce_sum():
    pl = _toy_peaklist()
    sums = pl.per_pixel_reduce("sum")
    np.testing.assert_allclose(sums, [110, 220, 1329])


def test_peaklist_per_pixel_reduce_max():
    pl = _toy_peaklist()
    maxes = pl.per_pixel_reduce("max")
    np.testing.assert_array_equal(maxes, [100, 200, 999])


def test_peaklist_vector_reductions_handle_empty_pixels():
    pl = PeakList(
        mz=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([3.0, 4.0, -2.0], dtype=np.float32),
        offsets=np.array([0, 2, 2, 3, 3], dtype=np.int64),
        n_pixels=4,
    )
    np.testing.assert_allclose(pl.per_pixel_reduce("sum"), [7.0, 0.0, -2.0, 0.0])
    np.testing.assert_allclose(pl.per_pixel_reduce("max"), [4.0, 0.0, -2.0, 0.0])
    np.testing.assert_allclose(pl.per_pixel_reduce("mean"), [3.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(pl.per_pixel_reduce("rms"), [np.sqrt(12.5), 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(pl.per_pixel_reduce("count"), [2, 0, 1, 0])


def test_peaklist_offsets_validation():
    with pytest.raises(ValueError, match="offsets length"):
        PeakList(
            mz=np.zeros(5, dtype=np.float64),
            intensity=np.zeros(5, dtype=np.float32),
            offsets=np.array([0, 5], dtype=np.int64),
            n_pixels=3,
        )


def test_peaklist_rejects_nonmonotone_and_incomplete_offsets():
    with pytest.raises(ValueError, match="monotonically"):
        PeakList(
            mz=np.zeros(3),
            intensity=np.zeros(3, dtype=np.float32),
            offsets=np.array([0, 2, 1, 3]),
            n_pixels=3,
        )
    with pytest.raises(ValueError, match="final offset"):
        PeakList(
            mz=np.zeros(3),
            intensity=np.zeros(3, dtype=np.float32),
            offsets=np.array([0, 1, 2]),
            n_pixels=2,
        )


def test_peaklist_rejects_nonfinite_spectra():
    with pytest.raises(ValueError, match="finite"):
        PeakList(
            mz=np.array([100.0, np.nan]),
            intensity=np.ones(2, dtype=np.float32),
            offsets=np.array([0, 2]),
            n_pixels=1,
        )


# -------- PeakMatrix ---------------------------------------------------------------


def test_peakmatrix_aggregate_methods():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    mz_axis = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    pm = PeakMatrix(matrix=matrix, mz_axis=mz_axis)
    mask = np.array([True, False, True])
    mz, agg_mean = pm.aggregate(mask, "mean")
    np.testing.assert_array_equal(mz, mz_axis)
    np.testing.assert_array_equal(agg_mean, [4, 5, 6])
    _, agg_sum = pm.aggregate(mask, "sum")
    np.testing.assert_array_equal(agg_sum, [8, 10, 12])
    _, agg_max = pm.aggregate(mask, "max")
    np.testing.assert_array_equal(agg_max, [7, 8, 9])


def test_peakmatrix_aggregate_empty_mask():
    matrix = np.zeros((3, 4), dtype=np.float32)
    pm = PeakMatrix(matrix=matrix, mz_axis=np.arange(4, dtype=np.float64))
    mask = np.zeros(3, dtype=bool)
    _, agg = pm.aggregate(mask, "mean")
    np.testing.assert_array_equal(agg, [0, 0, 0, 0])


def test_peakmatrix_rejects_unsorted_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        PeakMatrix(
            matrix=np.zeros((2, 3), dtype=np.float32),
            mz_axis=np.array([100.0, 99.0, 101.0]),
        )


def test_peakmatrix_rejects_nonfinite_matrix():
    with pytest.raises(ValueError, match="matrix must contain only finite"):
        PeakMatrix(
            matrix=np.array([[1.0, np.nan]], dtype=np.float32),
            mz_axis=np.array([100.0, 101.0]),
        )


# -------- hashing ------------------------------------------------------------------


def test_sha256_file_stable(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world")
    h = sha256_file(p)
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_hash_obj_stable_for_dataclass():
    ep = ExperimentParams(
        instrument_family="tof_reflectron",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=1000.0,
    )
    h1 = hash_obj(ep)
    h2 = hash_obj(ep)
    assert h1 == h2 and len(h1) == 64


def test_hash_obj_changes_with_value():
    ep1 = ExperimentParams(
        instrument_family="tof_reflectron",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=1000.0,
    )
    ep2 = ExperimentParams(
        instrument_family="tof_axial",
        ionization="maldi",
        profile_or_centroided="centroided",
        polarity="negative",
        mz_min=100.0,
        mz_max=1000.0,
    )
    assert hash_obj(ep1) != hash_obj(ep2)


def test_array_hash_tracks_content_shape_dtype_and_not_layout():
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert hash_array_content(base) == hash_array_content(np.asfortranarray(base))
    assert hash_array_content(base) == hash_array_content(base.astype(">f8"))
    assert hash_array_content(base) != hash_array_content(base.astype(np.float32))
    changed = base.copy()
    changed[0, 0] = 99.0
    assert hash_array_content(base) != hash_array_content(changed)
    assert hash_obj({"values": base}) != hash_obj({"values": changed})
    assert len(hash_array_content(np.empty((3, 0), dtype=np.float64))) == 64


def test_combine_hashes_order_sensitive():
    a, b = "00" * 32, "ff" * 32
    assert combine_hashes(a, b) != combine_hashes(b, a)


# -------- MSIDataset projections --------------------------------------------------


def test_msidataset_project_tic_via_peaklist(synth_centroided):
    from dapple.io.imzml_reader import read_imzml

    ds = read_imzml(synth_centroided)
    img = ds.project("tic")
    assert img.shape == (5, 5)
    assert img.dtype == np.float32
    # All pixels populated, all positive.
    assert (img > 0).all()


def test_msidataset_dataset_identity_hash(synth_centroided):
    from dapple.io.imzml_reader import read_imzml

    ds = read_imzml(synth_centroided)
    h = ds.hash()
    assert isinstance(h, str) and len(h) == 64
    # Same dataset, same hash.
    ds2 = read_imzml(synth_centroided)
    assert ds.hash() == ds2.hash()


def test_msidataset_hash_tracks_metadata_and_optionally_rois(synth_centroided):
    from dataclasses import replace

    from dapple.data.metadata import RoiDef
    from dapple.io.imzml_reader import read_imzml

    ds = read_imzml(synth_centroided)
    changed_metadata = replace(ds, metadata=replace(ds.metadata, sample_type="tissue"))
    assert ds.hash() != changed_metadata.hash()

    roi = RoiDef(
        name="region",
        vertices=((0.0, 0.0), (0.0, 2.0), (2.0, 0.0)),
    )
    annotated = ds.with_rois((roi,))
    assert ds.hash() != annotated.hash()
    assert ds.hash(include_rois=False) == annotated.hash(include_rois=False)


def test_msidataset_hash_tracks_geometry_backend_and_scientific_extra(synth_centroided):
    from dataclasses import replace

    from dapple.io.imzml_reader import read_imzml

    ds = read_imzml(synth_centroided)

    reordered = replace(ds, coords=ds.coords[::-1].copy())
    assert ds.hash() != reordered.hash()

    pl = ds.backend
    assert isinstance(pl, PeakList)
    changed_i = np.asarray(pl.intensity[:]).copy()
    changed_i[0] += 1.0
    changed_backend = PeakList(
        mz=np.asarray(pl.mz[:]).copy(),
        intensity=changed_i,
        offsets=np.asarray(pl.offsets[:]).copy(),
        n_pixels=pl.n_pixels,
    )
    assert ds.hash() != replace(ds, backend=changed_backend).hash()

    scientific = replace(ds, extra={**ds.extra, "score": np.array([1.0, 2.0])})
    scientific_changed = replace(
        ds, extra={**ds.extra, "score": np.array([1.0, 3.0])}
    )
    assert scientific.hash() != scientific_changed.hash()

    # Pure file-location helpers do not alter scientific processing state.
    relocated = replace(ds, extra={**ds.extra, "sidecar_path": "elsewhere.json"})
    assert ds.hash() == relocated.hash()


def test_msidataset_rejects_misaligned_channel_extra():
    ep = ExperimentParams(
        instrument_family="unknown",
        ionization="unknown",
        profile_or_centroided="centroided",
        polarity="unknown",
        mz_min=100.0,
        mz_max=200.0,
    )
    with pytest.raises(ValueError, match="channel-aligned extra"):
        MSIDataset(
            coords=np.array([[0, 0]], dtype=np.int32),
            grid_shape=(1, 1),
            metadata=ep,
            backend=PeakMatrix(
                matrix=np.ones((1, 2), dtype=np.float32),
                mz_axis=np.array([100.0, 200.0]),
            ),
            identity=DatasetIdentity(source_path="x", content_sha256="a" * 64),
            extra={"consensus_prevalence": np.array([1.0])},
        )


def test_msidataset_rejects_fractional_coordinates():
    ep = ExperimentParams(
        instrument_family="unknown",
        ionization="unknown",
        profile_or_centroided="centroided",
        polarity="unknown",
        mz_min=100.0,
        mz_max=200.0,
    )
    with pytest.raises(ValueError, match="integer dtype"):
        MSIDataset(
            coords=np.array([[0.5, 0.0]]),
            grid_shape=(1, 1),
            metadata=ep,
            backend=PeakMatrix(
                matrix=np.ones((1, 1), dtype=np.float32),
                mz_axis=np.array([100.0]),
            ),
            identity=DatasetIdentity(source_path="x", content_sha256="a" * 64),
        )


def test_dataset_identity_dataclass_immutable():
    di = DatasetIdentity(source_path="x", content_sha256="y")
    with pytest.raises(Exception):  # frozen dataclass
        di.source_path = "z"  # type: ignore[misc]


def test_msidataset_with_history_appends(synth_centroided):
    from dapple.io.imzml_reader import read_imzml
    from dapple.data.dataset import OpRecord

    ds = read_imzml(synth_centroided)
    rec = OpRecord(
        op_name="dummy",
        params_hash="00" * 32,
        input_hash=ds.hash(),
        output_hash="ff" * 32,
        diagnostics_summary=(),
    )
    ds2 = ds.with_history(rec)
    assert len(ds.history) == 0
    assert len(ds2.history) == 1
    assert ds2.history[0].op_name == "dummy"
