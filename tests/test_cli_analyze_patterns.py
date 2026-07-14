"""End-to-end tests for the headless spatial-pattern analysis CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dapple.cli.analyze_patterns import main
from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.data.metadata import DatasetIdentity, ExperimentParams, RoiDef
from dapple.io.imzml_writer import write_imzml
from dapple.io.spec_xml import make_provenance, write_spec_xml
from dapple.pipeline import Pipeline


def _dataset(*, harmonized: bool = True) -> MSIDataset:
    height, width = 4, 6
    coords = np.asarray(
        [(x + 1, y + 1) for y in range(height) for x in range(width)],
        dtype=np.int32,
    )
    x = np.tile(np.arange(width, dtype=np.float64), height)
    matrix = np.column_stack(
        [
            2.0 + 5.0 * x,
            27.0 - 5.0 * x,
            8.0 + 0.2 * np.tile(np.arange(height), width).reshape(-1),
        ]
    ).astype(np.float32)
    if harmonized:
        backend: PeakMatrix | PeakList = PeakMatrix(
            matrix=matrix,
            mz_axis=np.asarray([100.0, 200.0, 300.0], dtype=np.float64),
        )
    else:
        backend = PeakList(
            mz=np.full(coords.shape[0], 100.0, dtype=np.float64),
            intensity=np.ones(coords.shape[0], dtype=np.float32),
            offsets=np.arange(coords.shape[0] + 1, dtype=np.int64),
            n_pixels=coords.shape[0],
        )
    return MSIDataset(
        coords=coords,
        grid_shape=(height, width),
        metadata=ExperimentParams(
            instrument_family="qtof",
            ionization="maldi",
            profile_or_centroided="centroided",
            polarity="positive",
            mz_min=99.0,
            mz_max=301.0,
            sample_type="whole_organism",
        ),
        backend=backend,
        identity=DatasetIdentity(source_path="synthetic", content_sha256="cli-test"),
    )


def _rois() -> tuple[RoiDef, RoiDef]:
    return (
        RoiDef(
            name="anterior",
            vertices=((0.0, 0.0), (0.0, 2.49), (3.0, 2.49), (3.0, 0.0)),
        ),
        RoiDef(
            name="posterior",
            vertices=((0.0, 2.51), (0.0, 5.0), (3.0, 5.0), (3.0, 2.51)),
        ),
    )


def _write_harmonized_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = _dataset()
    imzml_path = tmp_path / "harmonized.imzML"
    write_imzml(dataset, imzml_path)
    spec_path = tmp_path / "regions.spec.xml"
    write_spec_xml(
        spec_path,
        pipeline=Pipeline(nodes=()),
        experiment_params=dataset.metadata,
        provenance=make_provenance(
            plugin_version="test",
            input_dataset_hash=dataset.hash(),
            roi_definitions=_rois(),
        ),
    )
    return imzml_path, spec_path


def test_roi_cli_loads_sidecar_and_spec_then_exports(tmp_path, capsys):
    imzml_path, spec_path = _write_harmonized_inputs(tmp_path)
    output_dir = tmp_path / "roi_output"

    status = main(
        [
            "roi",
            str(imzml_path),
            "--roi-spec",
            str(spec_path),
            "--numerator",
            "posterior",
            "--denominator",
            "anterior",
            "--overlap-policy",
            "error",
            "--stem",
            "ap",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    manifest_path = output_dir / "ap_manifest.json"
    table_path = output_dir / "ap_roi_enrichment.csv"
    assert manifest_path.exists()
    assert table_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "roi_enrichment"
    assert manifest["numerator_names"] == ["posterior"]
    assert manifest["denominator_names"] == ["anterior"]
    table = pd.read_csv(table_path)
    assert table.shape[0] == 3
    assert table.loc[0, "log2_fold_change"] > 1.0
    assert table.loc[1, "log2_fold_change"] < -1.0
    captured = capsys.readouterr()
    assert "3 harmonized channels" in captured.out
    assert "analysis status: ok" in captured.out
    assert str(manifest_path.resolve()) in captured.out
    assert "pixels as units" in captured.err


def test_axis_cli_exports_profiles_with_requested_seed_and_permutations(
    tmp_path, capsys
):
    imzml_path, _ = _write_harmonized_inputs(tmp_path)
    output_dir = tmp_path / "axis_output"

    status = main(
        [
            "axis",
            str(imzml_path),
            "--start",
            "1.5",
            "0",
            "--end",
            "1.5",
            "5",
            "--axis-name",
            "AP",
            "--start-label",
            "anterior",
            "--end-label",
            "posterior",
            "--bins",
            "6",
            "--permutations",
            "19",
            "--seed",
            "17",
            "--overlap-policy",
            "error",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    manifest_path = output_dir / "harmonized_axis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "axis_profile"
    assert manifest["axis_name"] == "AP"
    assert manifest["start_label"] == "anterior"
    assert manifest["end_label"] == "posterior"
    assert manifest["n_permutations_requested"] == 19
    assert manifest["n_permutations_effective"] == 38
    profiles = pd.read_csv(output_dir / "harmonized_axis_axis_profiles.csv")
    statistics = pd.read_csv(output_dir / "harmonized_axis_axis_statistics.csv")
    assert profiles.shape[0] == 6 * 3
    assert statistics.shape[0] == 3
    captured = capsys.readouterr()
    assert "bins=6, seed=17, permutations=19" in captured.out
    assert "analysis status: ok" in captured.out


def test_unknown_roi_error_lists_available_names(tmp_path, capsys):
    imzml_path, spec_path = _write_harmonized_inputs(tmp_path)

    status = main(
        [
            "roi",
            str(imzml_path),
            "--roi-spec",
            str(spec_path),
            "--numerator",
            "missing",
            "--denominator",
            "anterior",
            "--overlap-policy",
            "error",
        ]
    )

    assert status == 1
    error = capsys.readouterr().err
    assert "unknown ROI name(s): missing" in error
    assert "Available ROI names: anterior, posterior" in error


def test_raw_imzml_error_explains_harmonized_file_set(tmp_path, capsys):
    raw_path = tmp_path / "raw.imzML"
    write_imzml(_dataset(harmonized=False), raw_path)

    status = main(
        [
            "axis",
            str(raw_path),
            "--start",
            "1.5",
            "0",
            "--end",
            "1.5",
            "5",
            "--overlap-policy",
            "error",
        ]
    )

    assert status == 1
    error = capsys.readouterr().err
    assert "did not restore a harmonized PeakMatrix" in error
    assert ".imzML, .ibd, and raw.dapple-axis.json" in error


def test_axis_roi_restriction_requires_spec_and_name(tmp_path, capsys):
    imzml_path, _ = _write_harmonized_inputs(tmp_path)

    status = main(
        [
            "axis",
            str(imzml_path),
            "--start",
            "1.5",
            "0",
            "--end",
            "1.5",
            "5",
            "--roi-name",
            "anterior",
            "--overlap-policy",
            "error",
        ]
    )

    assert status == 1
    assert "requires both --roi-spec" in capsys.readouterr().err


def test_missing_harmonized_sidecar_is_actionable(tmp_path, capsys):
    imzml_path, _ = _write_harmonized_inputs(tmp_path)
    imzml_path.with_suffix(".dapple-axis.json").unlink()

    with pytest.warns(UserWarning, match="restoration failed"):
        status = main(
            [
                "axis",
                str(imzml_path),
                "--start",
                "1.5",
                "0",
                "--end",
                "1.5",
                "5",
                "--overlap-policy",
                "error",
            ]
        )

    assert status == 1
    error = capsys.readouterr().err
    assert "harmonized.dapple-axis.json" in error
    assert "keep its .imzML, .ibd" in error
