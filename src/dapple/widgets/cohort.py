"""CohortWidget: napari dock that runs cohort harmonization on a directory of datasets.

Where the wizard is single-dataset, this widget is multi-dataset. It exposes
``dapple.cohort.align.align_cohort`` plus the writer side of
``dapple-cohort-align`` as a Qt panel so analysts can harmonize a cohort and
save its outputs without dropping to the CLI.

Flow:
    1. Pick a cohort root directory (or browse for it).
    2. Optionally adjust the file pattern (default ``*.imzML``), recursive
       toggle, and the alignment knobs (bandwidth, min_prevalence, ...).
    3. Click **Discover** to pre-flight (lists matching files and their pixel
       counts without running).
    4. Click **Run cohort alignment** — runs on a worker thread, emits live
       progress, then writes one ``.imzML``/``.tif``/``_channels.csv`` per
       dataset plus a single ``cohort_summary.json``.
    5. Diagnostics with the same health-check rubric used by ``dapple-apply-spec``
       render in the bottom panel.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dapple.cohort.align import (
    CohortAlignParams,
    CohortAlignResult,
    align_cohort,
    load_cohort_directory,
)
from dapple.io.imzml_writer import write_imzml
from dapple.io.tiff_writer import write_hyperspectral_tiff
from dapple.pipeline import COHORT_RUBRICS, format_flat_summary

if TYPE_CHECKING:
    import napari


logger = logging.getLogger(__name__)


class _CohortRelay(QObject):
    """Signal proxy used to marshal worker-thread updates onto the GUI thread."""

    progress = Signal(str)
    finished = Signal(object, list)  # CohortAlignResult, list[Path]
    failed = Signal(str)


class CohortWidget(QWidget):
    """Multi-dataset harmonization panel.

    Mirrors the ``dapple-cohort-align`` CLI's surface, plus a discover step that
    lets the user inspect the cohort *before* running. Writes outputs into a
    chosen directory (defaults to ``ROOT_DIR/dapple_cohort``).
    """

    def __init__(
        self,
        napari_viewer: "napari.Viewer | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = napari_viewer
        self._discovered_files: list[Path] = []
        self._last_result: CohortAlignResult | None = None
        self._worker = None
        self._relay = _CohortRelay(self)
        self._relay.progress.connect(self._on_progress)
        self._relay.finished.connect(self._on_finished)
        self._relay.failed.connect(self._on_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel("<b>Cohort harmonization</b>")
        outer.addWidget(title)
        explainer = QLabel(
            "Pick a directory containing multiple imzML datasets. The "
            "operator runs single-dataset preprocessing (reference detection "
            "→ tolerance fit → recalibration → normalization → peak picking) "
            "on each one independently, then pools every peak into a single "
            "KDE consensus. Each output is the same dataset re-projected "
            "onto the shared m/z axis, plus a ``cohort_summary.json``."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #888;")
        outer.addWidget(explainer)

        # ---- Inputs ----------------------------------------------------------
        form = QFormLayout()
        outer.addLayout(form)

        # Cohort root directory.
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("path/to/cohort_root")
        browse_root = QPushButton("Browse…")
        browse_root.clicked.connect(self._on_browse_root)
        root_row = QHBoxLayout()
        root_row.addWidget(self._root_edit, stretch=1)
        root_row.addWidget(browse_root)
        root_row_w = QWidget()
        root_row_w.setLayout(root_row)
        form.addRow("Cohort root:", root_row_w)

        # File pattern.
        self._pattern_edit = QLineEdit("*.imzML")
        self._pattern_edit.setToolTip(
            "Glob pattern relative to cohort root. Use *.imzml for "
            "lower-cased filenames; *.cdf to load NetCDF datasets instead."
        )
        form.addRow("File pattern:", self._pattern_edit)

        self._recursive_cb = QCheckBox("Walk subdirectories")
        self._recursive_cb.setToolTip(
            "Recurse into subdirectories of the cohort root. Useful when "
            "each dataset is in its own subfolder."
        )
        form.addRow("", self._recursive_cb)

        # Output directory.
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("(defaults to <cohort_root>/dapple_cohort)")
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._on_browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_edit, stretch=1)
        out_row.addWidget(browse_out)
        out_row_w = QWidget()
        out_row_w.setLayout(out_row)
        form.addRow("Output directory:", out_row_w)

        # Alignment knobs.
        self._bandwidth_ppm = QDoubleSpinBox()
        self._bandwidth_ppm.setRange(0.1, 10000.0)
        self._bandwidth_ppm.setDecimals(2)
        self._bandwidth_ppm.setValue(50.0)
        self._bandwidth_ppm.setToolTip(
            "KDE bandwidth in ppm for the pooled-cohort consensus. Match "
            "to the per-pixel m/z scatter of your cohort. Lower = sharper "
            "peaks but more risk of splitting; higher = more merging."
        )
        form.addRow("Bandwidth (ppm):", self._bandwidth_ppm)

        self._min_prevalence = QDoubleSpinBox()
        self._min_prevalence.setRange(0.0, 1.0)
        self._min_prevalence.setDecimals(3)
        self._min_prevalence.setSingleStep(0.05)
        self._min_prevalence.setValue(0.05)
        self._min_prevalence.setToolTip(
            "Drop shared-axis channels carried by less than this fraction "
            "of cohort-wide pixels. The cohort default (0.05) is permissive; "
            "raise to 0.3-0.5 to focus on broadly-shared channels."
        )
        form.addRow("Min cohort prevalence:", self._min_prevalence)

        self._min_prominence_q = QDoubleSpinBox()
        self._min_prominence_q.setRange(0.0, 1.0)
        self._min_prominence_q.setDecimals(3)
        self._min_prominence_q.setSingleStep(0.05)
        self._min_prominence_q.setValue(0.5)
        self._min_prominence_q.setToolTip(
            "KDE prominence quantile cutoff for the pooled cloud. Lower "
            "(0.3) keeps weaker peaks, higher (0.7-0.9) is stricter."
        )
        form.addRow("Min prominence quantile:", self._min_prominence_q)

        self._recalibrate_cb = QCheckBox("Recalibrate per dataset (TOF / Q-TOF only)")
        self._recalibrate_cb.setChecked(True)
        self._recalibrate_cb.setToolTip(
            "Run msiwarp on each dataset against its own reference ions "
            "before pooling. Disable for already-recalibrated data or "
            "Orbitrap / FT-ICR cohorts."
        )
        form.addRow("", self._recalibrate_cb)

        self._pool_normalize_cb = QCheckBox("Per-dataset median normalize before pooling")
        self._pool_normalize_cb.setChecked(True)
        self._pool_normalize_cb.setToolTip(
            "Apply median normalization within each dataset before pooling "
            "peaks for consensus. Disable only if you've pre-normalized "
            "externally."
        )
        form.addRow("", self._pool_normalize_cb)

        self._rng_seed = QSpinBox()
        self._rng_seed.setRange(0, 2_000_000_000)
        self._rng_seed.setValue(0)
        self._rng_seed.setToolTip("Master RNG seed for deterministic re-runs.")
        form.addRow("RNG seed:", self._rng_seed)

        self._write_imzml_cb = QCheckBox("Write per-dataset .imzML / .ibd")
        self._write_imzml_cb.setChecked(True)
        form.addRow("", self._write_imzml_cb)
        self._write_tiff_cb = QCheckBox("Write per-dataset .tif + _channels.csv")
        self._write_tiff_cb.setChecked(True)
        form.addRow("", self._write_tiff_cb)

        # ---- Actions ---------------------------------------------------------
        actions = QHBoxLayout()
        self._discover_btn = QPushButton("Discover datasets")
        self._discover_btn.setToolTip(
            "List the imzML / CDF files that would be loaded. Doesn't run "
            "the alignment."
        )
        self._discover_btn.clicked.connect(self._on_discover)
        actions.addWidget(self._discover_btn)
        self._run_btn = QPushButton("Run cohort alignment")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        actions.addWidget(self._run_btn)
        outer.addLayout(actions)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate by default
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        # Discovery / progress / diagnostics log.
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(160)
        outer.addWidget(self._log, stretch=1)

    # --- File-picker handlers ---------------------------------------------------

    def _on_browse_root(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose cohort root directory")
        if p:
            self._root_edit.setText(p)

    def _on_browse_out(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if p:
            self._out_edit.setText(p)

    # --- Discover ---------------------------------------------------------------

    def _on_discover(self) -> None:
        root = self._root_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "Pick a path", "Enter or browse to a cohort root directory.")
            return
        try:
            datasets = load_cohort_directory(
                Path(root),
                pattern=self._pattern_edit.text().strip() or "*.imzML",
                recursive=self._recursive_cb.isChecked(),
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self, "Discovery failed", f"{e}\n\n{traceback.format_exc()}"
            )
            return
        self._discovered_files = [Path(d.identity.source_path) for d in datasets]
        self._log.clear()
        self._log.append(f"discovered {len(datasets)} dataset(s) under {root}:")
        total_pixels = 0
        for i, d in enumerate(datasets):
            name = Path(d.identity.source_path).name
            self._log.append(
                f"  [{i}] {name}  ({d.n_pixels} pixels, grid {d.grid_shape}, "
                f"{d.metadata.instrument_family}/{d.metadata.profile_or_centroided})"
            )
            total_pixels += d.n_pixels
        self._log.append(f"\n→ total cohort pixels: {total_pixels}")
        self._run_btn.setEnabled(len(datasets) >= 1)

    # --- Run --------------------------------------------------------------------

    def _build_params(self) -> CohortAlignParams:
        return CohortAlignParams(
            bandwidth_ppm=float(self._bandwidth_ppm.value()),
            min_prevalence=float(self._min_prevalence.value()),
            min_prominence_quantile=float(self._min_prominence_q.value()),
            pool_normalize=self._pool_normalize_cb.isChecked(),
            recalibrate=self._recalibrate_cb.isChecked(),
            rng_seed=int(self._rng_seed.value()),
        )

    def _resolve_output_dir(self) -> Path:
        out = self._out_edit.text().strip()
        if out:
            return Path(out).expanduser().resolve()
        root = Path(self._root_edit.text().strip()).expanduser().resolve()
        return root / "dapple_cohort"

    def _on_run(self) -> None:
        if not self._discovered_files:
            QMessageBox.warning(
                self, "Discover first",
                "Click 'Discover datasets' to verify the cohort before running.",
            )
            return
        params = self._build_params()
        out_dir = self._resolve_output_dir()
        write_imzml_flag = self._write_imzml_cb.isChecked()
        write_tiff_flag = self._write_tiff_cb.isChecked()

        # Lock the UI while the worker runs.
        self._run_btn.setEnabled(False)
        self._discover_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log.append("\nstarting cohort alignment…")

        # The cohort runner is synchronous — wrap it in napari's thread_worker
        # so the GUI stays responsive. If napari isn't available (test mode),
        # fall back to running synchronously.
        try:
            from napari.qt import thread_worker
        except ImportError:
            thread_worker = None  # type: ignore[assignment]

        relay = self._relay
        files_snapshot = list(self._discovered_files)
        pattern = self._pattern_edit.text().strip() or "*.imzML"
        recursive = self._recursive_cb.isChecked()
        root = Path(self._root_edit.text().strip())

        def _do_work() -> tuple[CohortAlignResult, list[Path]]:
            relay.progress.emit(f"loading {len(files_snapshot)} dataset(s)…")
            datasets = load_cohort_directory(
                root, pattern=pattern, recursive=recursive
            )
            relay.progress.emit(
                f"running per-dataset preprocessing + pooled KDE consensus "
                f"(this may take a few minutes for large cohorts)…"
            )
            result = align_cohort(datasets, params=params)
            relay.progress.emit(
                f"  → shared axis: {result.shared_consensus_mz.size} channels"
            )
            relay.progress.emit("writing outputs…")
            out_dir.mkdir(parents=True, exist_ok=True)
            written: list[Path] = []
            for di, ds in enumerate(result.aligned_datasets):
                stem = Path(ds.identity.source_path).stem or f"dataset_{di:02d}"
                base = out_dir / f"{stem}_cohort"
                if write_tiff_flag:
                    extra = {
                        "cohort_prevalence": result.cohort_prevalence,
                        "per_dataset_prevalence": result.per_dataset_prevalence[di],
                    }
                    tw = write_hyperspectral_tiff(
                        ds, base.with_suffix(".tif"), extra_per_channel=extra
                    )
                    written.append(tw.tiff_path)
                    written.append(tw.csv_path)
                    relay.progress.emit(f"  [{di}] wrote {tw.tiff_path.name}")
                if write_imzml_flag:
                    iw = write_imzml(ds, base.with_suffix(".imzML"))
                    written.append(iw.imzml_path)
                    written.append(iw.ibd_path)
                    relay.progress.emit(f"  [{di}] wrote {iw.imzml_path.name}")
            # Always write the cohort summary JSON.
            summary_path = out_dir / "cohort_summary.json"
            summary_payload = {
                "n_datasets": len(datasets),
                "shared_consensus_mz": result.shared_consensus_mz.tolist(),
                "cohort_prevalence": result.cohort_prevalence.tolist(),
                "per_dataset_prevalence": {
                    str(k): v.tolist() for k, v in result.per_dataset_prevalence.items()
                },
                "datasets": [
                    {
                        "index": i,
                        "source_path": str(d.identity.source_path),
                        "n_pixels": int(d.n_pixels),
                        "grid_shape": list(d.grid_shape),
                        "instrument_family": d.metadata.instrument_family,
                        "ionization": d.metadata.ionization,
                        "polarity": d.metadata.polarity,
                        "mode": d.metadata.profile_or_centroided,
                    }
                    for i, d in enumerate(datasets)
                ],
                "diagnostics": result.diagnostics,
                "params": asdict(params),
            }
            summary_path.write_text(
                json.dumps(summary_payload, indent=2), encoding="utf-8"
            )
            written.append(summary_path)
            relay.progress.emit(f"wrote {summary_path.name}")
            return result, written

        if thread_worker is not None:
            @thread_worker
            def _runner():
                return _do_work()

            worker = _runner()
            worker.returned.connect(lambda payload: self._relay.finished.emit(payload[0], payload[1]))
            worker.errored.connect(lambda exc: self._relay.failed.emit(repr(exc)))
            worker.finished.connect(self._on_worker_done)
            worker.start()
            self._worker = worker
        else:
            # Synchronous fallback (used in tests where napari.qt isn't available).
            try:
                result, written = _do_work()
                self._relay.finished.emit(result, written)
            except Exception as e:  # noqa: BLE001
                self._relay.failed.emit(repr(e))
            finally:
                self._on_worker_done()

    # --- Worker callbacks -------------------------------------------------------

    def _on_progress(self, msg: str) -> None:
        self._log.append(msg)

    def _on_finished(self, result: CohortAlignResult, written: list[Path]) -> None:
        self._last_result = result
        self._log.append("")
        for line in format_flat_summary(
            result.diagnostics,
            title="Cohort alignment diagnostics",
            extra_rubrics=COHORT_RUBRICS,
        ):
            self._log.append(line)
        self._log.append(f"\n{len(written)} output file(s) written.")

    def _on_failed(self, msg: str) -> None:
        self._log.append(f"\nFAILED: {msg}")
        QMessageBox.critical(self, "Cohort alignment failed", msg)

    def _on_worker_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._discover_btn.setEnabled(True)
        self._progress.setVisible(False)

    # --- Test / API helper ------------------------------------------------------

    @property
    def last_result(self) -> CohortAlignResult | None:
        """Most recent successful result (None if not run)."""
        return self._last_result
