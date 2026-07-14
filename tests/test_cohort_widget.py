"""Tests for the napari CohortWidget — multi-dataset harmonization GUI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("napari", reason="napari is required for widget tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for widget tests")


@pytest.fixture
def viewer(make_napari_viewer):
    v = make_napari_viewer()
    yield v


def _two_member_cohort(synth_centroided: Path, tmp_path: Path) -> Path:
    """Build a tmp_path/cohort_root with two copies of the synth fixture."""
    root = tmp_path / "cohort_root"
    root.mkdir()
    for name in ("a.imzML", "b.imzML"):
        shutil.copy(synth_centroided, root / name)
        shutil.copy(synth_centroided.with_suffix(".ibd"), root / name.replace(".imzML", ".ibd"))
    return root


def test_cohort_widget_constructs(qtbot, viewer):
    """Bare instantiation: verifies imports, signal wiring, no Qt complaints."""
    from dapple.widgets.cohort import CohortWidget

    w = CohortWidget(napari_viewer=viewer)
    qtbot.addWidget(w)
    assert w._root_edit.text() == ""  # noqa: SLF001
    assert w._pool_weighting_combo.currentData() == "sample"  # noqa: SLF001
    assert w._prevalence_basis_combo.currentData() == "pixel"  # noqa: SLF001
    assert not w._run_btn.isEnabled()  # noqa: SLF001 — locked until Discover succeeds


def test_cohort_widget_builds_weighting_and_prevalence_params(qtbot, viewer):
    from dapple.widgets.cohort import CohortWidget

    w = CohortWidget(napari_viewer=viewer)
    qtbot.addWidget(w)
    w._pool_weighting_combo.setCurrentIndex(  # noqa: SLF001
        w._pool_weighting_combo.findData("intensity")  # noqa: SLF001
    )
    w._prevalence_basis_combo.setCurrentIndex(  # noqa: SLF001
        w._prevalence_basis_combo.findData("dataset")  # noqa: SLF001
    )

    params = w._build_params()  # noqa: SLF001

    assert params.pool_weighting == "intensity"
    assert params.prevalence_basis == "dataset"


def test_widget_output_bases_are_collision_safe(tmp_path):
    from dapple.widgets.cohort import _unique_output_bases

    datasets = [
        SimpleNamespace(identity=SimpleNamespace(source_path="one/sample.imzML")),
        SimpleNamespace(identity=SimpleNamespace(source_path="two/sample.imzML")),
    ]

    bases = _unique_output_bases(datasets, tmp_path)

    assert [base.name for base in bases] == ["sample_cohort", "sample_2_cohort"]


def test_widget_discovery_excludes_nested_output_directory(tmp_path):
    from dapple.widgets.cohort import _discover_cohort_files

    root = tmp_path / "cohort"
    out = root / "dapple_cohort"
    out.mkdir(parents=True)
    source = root / "source.imzML"
    generated = out / "source_cohort.imzML"
    source.touch()
    generated.touch()

    files, n_excluded = _discover_cohort_files(
        root,
        pattern="*.imzML",
        recursive=True,
        exclude_dir=out,
    )

    assert files == [source]
    assert n_excluded == 1


def test_widget_discovery_excludes_harmonized_outputs_when_output_is_root(tmp_path):
    from dapple.widgets.cohort import _discover_cohort_files

    root = tmp_path / "cohort"
    root.mkdir()
    source = root / "source.imzML"
    generated = root / "source_cohort.imzML"
    source.touch()
    generated.touch()
    generated.with_suffix(".dapple-axis.json").write_text(
        "{}", encoding="utf-8"
    )

    files, n_excluded = _discover_cohort_files(
        root,
        pattern="*.imzML",
        recursive=False,
        exclude_dir=root,
    )

    assert files == [source]
    assert n_excluded == 1


def test_discover_lists_files(qtbot, viewer, synth_centroided, tmp_path):
    """Discover populates the log and enables the Run button."""
    from dapple.widgets.cohort import CohortWidget

    root = _two_member_cohort(synth_centroided, tmp_path)
    w = CohortWidget(napari_viewer=viewer)
    qtbot.addWidget(w)
    w._root_edit.setText(str(root))  # noqa: SLF001
    w._on_discover()  # noqa: SLF001
    log = w._log.toPlainText()  # noqa: SLF001
    assert "discovered 2 dataset(s)" in log
    assert "a.imzML" in log
    assert "b.imzML" in log
    assert w._run_btn.isEnabled()  # noqa: SLF001


def test_discover_handles_nonexistent_root(qtbot, viewer, tmp_path):
    """Pointing at a non-existent dir surfaces an error dialog (not a crash)."""
    from dapple.widgets.cohort import CohortWidget

    w = CohortWidget(napari_viewer=viewer)
    qtbot.addWidget(w)
    w._root_edit.setText(str(tmp_path / "nope"))  # noqa: SLF001
    # The widget handles errors internally with a QMessageBox; make sure that
    # path doesn't raise out of the slot.
    from unittest.mock import patch
    with patch("dapple.widgets.cohort.QMessageBox.critical") as mb:
        w._on_discover()  # noqa: SLF001
    mb.assert_called_once()


def test_run_writes_outputs_and_renders_diagnostics(qtbot, viewer, synth_centroided, tmp_path):
    """End-to-end: discover → run (sync fallback) → outputs on disk + diagnostics in log."""
    from dapple.widgets.cohort import CohortWidget

    root = _two_member_cohort(synth_centroided, tmp_path)
    out = tmp_path / "out"
    w = CohortWidget(napari_viewer=viewer)
    qtbot.addWidget(w)
    w._root_edit.setText(str(root))  # noqa: SLF001
    w._out_edit.setText(str(out))  # noqa: SLF001
    # Tweak knobs so the synth fixture's 5 planted peaks survive the prevalence
    # filter on a 2-dataset cohort of 5x5 pixels.
    w._bandwidth_ppm.setValue(20.0)  # noqa: SLF001
    w._min_prevalence.setValue(0.5)  # noqa: SLF001
    w._recalibrate_cb.setChecked(False)  # noqa: SLF001 — keep test fast
    w._pool_weighting_combo.setCurrentIndex(  # noqa: SLF001
        w._pool_weighting_combo.findData("intensity")  # noqa: SLF001
    )
    w._prevalence_basis_combo.setCurrentIndex(  # noqa: SLF001
        w._prevalence_basis_combo.findData("dataset")  # noqa: SLF001
    )
    w._on_discover()  # noqa: SLF001
    assert w._run_btn.isEnabled()  # noqa: SLF001

    # Force the synchronous fallback by hiding napari.qt.thread_worker so the
    # test doesn't need a Qt event loop spin.
    import builtins
    real_import = builtins.__import__

    def _block_thread_worker(name, *args, **kwargs):
        if name == "napari.qt":
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    from unittest.mock import patch
    with patch("builtins.__import__", side_effect=_block_thread_worker):
        w._on_run()  # noqa: SLF001

    # Outputs.
    assert (out / "cohort_summary.json").exists()
    summary = json.loads((out / "cohort_summary.json").read_text(encoding="utf-8"))
    assert summary["n_datasets"] == 2
    assert len(summary["shared_consensus_mz"]) >= 5
    assert len(summary["dataset_prevalence"]) == len(summary["shared_consensus_mz"])
    assert summary["params"]["pool_weighting"] == "intensity"
    assert summary["params"]["prevalence_basis"] == "dataset"
    assert summary["prevalence_filter"]["basis"] == "dataset"
    assert summary["summary_schema_version"] == 2
    assert len(summary["output_files"]) == 11
    assert "a_cohort.dapple-axis.json" in summary["output_files"]
    assert all("output_files" in item for item in summary["datasets"])
    assert (out / "a_cohort.tif").exists()
    assert (out / "b_cohort.tif").exists()
    assert "dataset_prevalence" in (
        out / "a_cohort_channels.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert (out / "a_cohort.imzML").exists()
    assert (out / "b_cohort.imzML").exists()
    assert (out / "a_cohort.dapple-axis.json").exists()
    assert (out / "b_cohort.dapple-axis.json").exists()

    # Log contains the rubric-graded diagnostic block.
    log = w._log.toPlainText()  # noqa: SLF001
    assert "Cohort alignment diagnostics" in log
    assert "shared consensus channels" in log
    assert "11 output file(s) written" in log
    # The widget exposes the result for downstream use.
    assert w.last_result is not None
    assert w.last_result.shared_consensus_mz.size >= 5


def test_run_without_discover_warns(qtbot, viewer, tmp_path):
    """Clicking Run before Discover surfaces a warning dialog."""
    from dapple.widgets.cohort import CohortWidget
    from unittest.mock import patch

    w = CohortWidget(napari_viewer=viewer)
    qtbot.addWidget(w)
    with patch("dapple.widgets.cohort.QMessageBox.warning") as mb:
        # Force it to be 'enabled' to prove the discover-first guard works
        # even if someone bypasses the button state.
        w._run_btn.setEnabled(True)  # noqa: SLF001
        w._discovered_files = []  # noqa: SLF001
        w._on_run()  # noqa: SLF001
    mb.assert_called_once()


def test_widget_is_listed_in_napari_yaml():
    """The plugin manifest must register CohortWidget so it shows up in the menu."""
    import yaml

    manifest_path = Path(__file__).parent.parent / "src" / "dapple" / "napari.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    cmd_ids = {c["id"] for c in manifest["contributions"]["commands"]}
    widget_cmds = {w["command"] for w in manifest["contributions"]["widgets"]}
    assert "dapple.open_cohort" in cmd_ids
    assert "dapple.open_cohort" in widget_cmds
