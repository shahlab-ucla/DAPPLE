"""Interactive ROI-enrichment and directed-axis pattern analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dapple.analysis import (
    AxisProfileResult,
    DirectedAxis,
    RoiEnrichmentResult,
    analyze_axis_profiles,
    analyze_roi_enrichment,
    export_analysis_result,
    rasterize_rois,
)
from dapple.data.dataset import PeakMatrix
from dapple.widgets._session import MsiSession, adopt_dataset_from_viewer, default_session

if TYPE_CHECKING:
    import napari


class DevelopmentalAnalysisWidget(QWidget):
    """Explore enrichment between ROIs or along a directed anatomical axis.

    The panel is intentionally downstream of harmonization: every result compares
    the same shared m/z channels and can send a selected ion back to Channels.
    """

    def __init__(
        self,
        napari_viewer: "napari.Viewer | None" = None,
        session: MsiSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = napari_viewer
        self._session = session or default_session()
        adopt_dataset_from_viewer(self._session, napari_viewer)
        self._result: RoiEnrichmentResult | AxisProfileResult | None = None
        self._row_channels: list[int] = []
        self._worker = None
        self._busy = False
        self._analysis_generation = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(QLabel("<b>Spatial and developmental patterns</b>"))
        self._dataset_info = QLabel()
        self._dataset_info.setWordWrap(True)
        outer.addWidget(self._dataset_info)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_roi_tab(), "ROI enrichment")
        self._tabs.addTab(self._build_axis_tab(), "Developmental axis")
        outer.addWidget(self._tabs)

        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run analysis")
        self._run_btn.clicked.connect(self._run)
        action_row.addWidget(self._run_btn)
        self._show_btn = QPushButton("Show selected ion")
        self._show_btn.setEnabled(False)
        self._show_btn.clicked.connect(self._show_selected_ion)
        action_row.addWidget(self._show_btn)
        self._export_btn = QPushButton("Export tables…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        action_row.addWidget(self._export_btn)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        self._status = QLabel("Results are descriptive until an analysis is run.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        outer.addWidget(self._status)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setMinimumHeight(180)
        outer.addWidget(self._plot)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._table.cellDoubleClicked.connect(lambda *_: self._show_selected_ion())
        outer.addWidget(self._table, stretch=1)

        self._session.dataset_changed.connect(self._session_changed)
        self._session.rois_changed.connect(self._session_changed)
        self._refresh_inputs()

    def _session_changed(self, *_args: object) -> None:
        """Invalidate results immediately when their dataset or ROIs change."""
        self._analysis_generation += 1
        message = (
            "Dataset/ROI changed; the running result will be discarded."
            if self._busy
            else "Dataset/ROI changed. Run analysis to refresh the results."
        )
        self._clear_result(message)
        self._refresh_inputs()

    def _clear_result(self, message: str) -> None:
        self._result = None
        self._row_channels.clear()
        self._show_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._plot.clear()
        self._table.clearContents()
        self._table.setRowCount(0)
        self._status.setText(message)

    def _build_roi_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._numerator = QComboBox()
        self._denominator = QComboBox()
        self._overlap = QComboBox()
        self._overlap.addItems(["error", "exclude", "first"])
        self._overlap.setToolTip(
            "How multiply-covered populated pixels are handled. 'error' is the "
            "safest default; exclude removes overlaps from both arms."
        )
        self._trim = QDoubleSpinBox()
        self._trim.setRange(0.0, 0.49)
        self._trim.setSingleStep(0.05)
        self._trim.setValue(0.1)
        self._min_units = QSpinBox()
        self._min_units.setRange(2, 1_000_000)
        self._min_units.setValue(3)
        form.addRow("Numerator ROI:", self._numerator)
        form.addRow("Denominator ROI:", self._denominator)
        form.addRow("ROI overlap policy:", self._overlap)
        form.addRow("Trim fraction:", self._trim)
        form.addRow("Minimum pixels/group:", self._min_units)
        caveat = QLabel(
            "Welch tests use pixels as units. Treat q-values as within-image "
            "screening, not biological-replicate evidence."
        )
        caveat.setWordWrap(True)
        form.addRow("", caveat)
        return tab

    def _build_axis_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._axis_name = QLineEdit("developmental_axis")
        self._start_y = self._coordinate_spin()
        self._start_x = self._coordinate_spin()
        self._end_y = self._coordinate_spin()
        self._end_x = self._coordinate_spin()
        self._end_x.setValue(10.0)
        self._start_label = QLineEdit("start")
        self._end_label = QLineEdit("end")
        self._limit_width = QCheckBox("Restrict pixels to a corridor around the axis")
        self._half_width = QDoubleSpinBox()
        self._half_width.setRange(0.0, 1_000_000.0)
        self._half_width.setValue(5.0)
        self._half_width.setSuffix(" px")
        self._half_width.setEnabled(False)
        self._limit_width.toggled.connect(self._half_width.setEnabled)
        self._axis_roi = QComboBox()
        self._axis_roi.addItem("All populated pixels")
        self._n_bins = QSpinBox()
        self._n_bins.setRange(2, 500)
        self._n_bins.setValue(20)
        self._permutations = QSpinBox()
        self._permutations.setRange(1, 100_000)
        self._permutations.setValue(999)
        self._min_endpoint_pixels = QSpinBox()
        self._min_endpoint_pixels.setRange(1, 1_000_000)
        self._min_endpoint_pixels.setValue(3)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(0)
        line_btn = QPushButton("Use endpoints of active Shapes line")
        line_btn.clicked.connect(self._read_active_line)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("y"))
        start_row.addWidget(self._start_y)
        start_row.addWidget(QLabel("x"))
        start_row.addWidget(self._start_x)
        start_widget = QWidget()
        start_widget.setLayout(start_row)
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("y"))
        end_row.addWidget(self._end_y)
        end_row.addWidget(QLabel("x"))
        end_row.addWidget(self._end_x)
        end_widget = QWidget()
        end_widget.setLayout(end_row)

        form.addRow("Axis name:", self._axis_name)
        form.addRow("Start (napari y, x):", start_widget)
        form.addRow("End (napari y, x):", end_widget)
        form.addRow("", line_btn)
        form.addRow("Start label:", self._start_label)
        form.addRow("End label:", self._end_label)
        form.addRow("", self._limit_width)
        form.addRow("Half-width:", self._half_width)
        form.addRow("Restrict to ROI:", self._axis_roi)
        form.addRow("Profile bins:", self._n_bins)
        form.addRow("Trend permutations:", self._permutations)
        form.addRow("Minimum pixels/endpoint:", self._min_endpoint_pixels)
        form.addRow("RNG seed:", self._seed)
        return tab

    @staticmethod
    def _coordinate_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setDecimals(2)
        return spin

    def _refresh_inputs(self) -> None:
        ds = self._session.dataset
        aligned = ds is not None and isinstance(ds.backend, PeakMatrix)
        if ds is None:
            self._dataset_info.setText("No MSI dataset is attached to this viewer.")
        elif not aligned:
            self._dataset_info.setText(
                "This dataset is not harmonized. Run consensus alignment before "
                "comparing shared m/z channels."
            )
        else:
            self._dataset_info.setText(
                f"{Path(ds.identity.source_path).name}: {ds.n_pixels} populated pixels, "
                f"{ds.backend.n_peaks} shared channels"
            )
        self._run_btn.setEnabled(aligned and not self._busy)

        current_num = self._numerator.currentText()
        current_den = self._denominator.currentText()
        current_axis_roi = self._axis_roi.currentText()
        names = [r.name for r in self._session.rois if not r.is_background]
        self._numerator.clear()
        self._denominator.clear()
        self._numerator.addItems(names)
        self._denominator.addItems(names)
        self._axis_roi.clear()
        self._axis_roi.addItem("All populated pixels")
        self._axis_roi.addItems(names)
        for combo, value in (
            (self._numerator, current_num),
            (self._denominator, current_den),
            (self._axis_roi, current_axis_roi),
        ):
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if len(names) > 1 and self._denominator.currentIndex() == self._numerator.currentIndex():
            self._denominator.setCurrentIndex(1)

    def _read_active_line(self) -> None:
        try:
            layer = self._viewer.layers.selection.active if self._viewer is not None else None
            if layer is None or type(layer).__name__ != "Shapes" or not layer.data:
                raise ValueError("select a Shapes layer containing a line or path")
            selected = sorted(getattr(layer, "selected_data", ()))
            index = selected[-1] if selected else len(layer.data) - 1
            vertices = np.asarray(layer.data[index], dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[0] < 2 or vertices.shape[1] < 2:
                raise ValueError("the selected shape needs at least two y/x vertices")
            start, end = vertices[0, -2:], vertices[-1, -2:]
            self._start_y.setValue(float(start[0]))
            self._start_x.setValue(float(start[1]))
            self._end_y.setValue(float(end[0]))
            self._end_x.setValue(float(end[1]))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not read axis", str(exc))

    def _prepare_computation(self):  # noqa: ANN201
        """Capture widget state on the GUI thread and return a pure computation."""
        ds = self._session.dataset
        if ds is None or not isinstance(ds.backend, PeakMatrix):
            raise RuntimeError("run consensus alignment before developmental analysis")
        if self._tabs.currentIndex() == 0:
            if not ds.rois:
                raise ValueError("draw and name at least two foreground ROIs first")
            numerator = self._numerator.currentText()
            denominator = self._denominator.currentText()
            if not numerator or not denominator or numerator == denominator:
                raise ValueError("choose two different numerator and denominator ROIs")
            selected_names = {numerator, denominator}
            foreground = tuple(
                r
                for r in ds.rois
                if not r.is_background and r.name in selected_names
            )
            masks = rasterize_rois(
                ds,
                foreground,
                overlap_policy=self._overlap.currentText(),  # type: ignore[arg-type]
            )
            min_units = int(self._min_units.value())
            trim_fraction = float(self._trim.value())

            def compute_roi() -> RoiEnrichmentResult:
                return analyze_roi_enrichment(
                    ds,
                    masks,
                    numerator,
                    denominator,
                    min_units_per_group=min_units,
                    trim_fraction=trim_fraction,
                )

            return compute_roi

        axis = DirectedAxis(
            name=self._axis_name.text().strip() or "developmental_axis",
            start_yx=(float(self._start_y.value()), float(self._start_x.value())),
            end_yx=(float(self._end_y.value()), float(self._end_x.value())),
            start_label=self._start_label.text().strip() or "start",
            end_label=self._end_label.text().strip() or "end",
            half_width_px=(
                float(self._half_width.value()) if self._limit_width.isChecked() else None
            ),
        )
        include_mask = None
        selection_names: tuple[str, ...] = ()
        if self._axis_roi.currentIndex() > 0:
            selected_name = self._axis_roi.currentText()
            selection_names = (selected_name,)
            foreground = tuple(
                r
                for r in ds.rois
                if not r.is_background and r.name == selected_name
            )
            masks = rasterize_rois(ds, foreground, overlap_policy="error")
            include_mask = masks.union(selected_name)
        n_bins = int(self._n_bins.value())
        n_permutations = int(self._permutations.value())
        min_endpoint_pixels = int(self._min_endpoint_pixels.value())
        rng_seed = int(self._seed.value())

        def compute_axis() -> AxisProfileResult:
            return analyze_axis_profiles(
                ds,
                axis,
                include_mask=include_mask,
                selection_names=selection_names,
                n_bins=n_bins,
                n_permutations=n_permutations,
                min_endpoint_pixels=min_endpoint_pixels,
                rng_seed=rng_seed,
            )

        return compute_axis

    def _compute(self) -> RoiEnrichmentResult | AxisProfileResult:
        """Synchronous analysis helper used by tests and headless callers."""
        return self._prepare_computation()()

    def _run(self) -> None:
        try:
            compute = self._prepare_computation()
        except Exception as exc:  # noqa: BLE001
            self._analysis_failed(exc)
            return
        self._analysis_generation += 1
        generation = self._analysis_generation
        self._busy = True
        self._clear_result("Running analysis…")
        self._run_btn.setEnabled(False)
        try:
            from napari.qt import thread_worker
        except ImportError:
            thread_worker = None  # type: ignore[assignment]

        if thread_worker is None:
            try:
                self._accept_result(generation, compute())
            except Exception as exc:  # noqa: BLE001
                self._analysis_failed(exc, generation=generation)
            finally:
                self._analysis_finished(generation)
            return

        @thread_worker
        def runner():
            return compute()

        worker = runner()
        worker.returned.connect(
            lambda result, token=generation: self._accept_result(token, result)
        )
        worker.errored.connect(
            lambda exc, token=generation: self._analysis_failed(exc, generation=token)
        )
        worker.finished.connect(
            lambda token=generation: self._analysis_finished(token)
        )
        self._worker = worker
        worker.start()

    def _analysis_finished(self, _generation: int | None = None) -> None:
        self._worker = None
        self._busy = False
        self._refresh_inputs()

    def _analysis_failed(
        self, exc: BaseException, *, generation: int | None = None
    ) -> None:
        if generation is not None and generation != self._analysis_generation:
            return
        self._status.setText(f"Analysis failed: {exc}")
        QMessageBox.critical(self, "Analysis failed", str(exc))

    def _accept_result(
        self,
        generation: int,
        result: RoiEnrichmentResult | AxisProfileResult,
    ) -> None:
        if generation != self._analysis_generation:
            return
        self._set_result(result)

    def _set_result(self, result: RoiEnrichmentResult | AxisProfileResult) -> None:
        self._result = result
        self._row_channels.clear()
        self._plot.clear()
        if isinstance(result, RoiEnrichmentResult):
            self._populate_roi_result(result)
        else:
            self._populate_axis_result(result)
        warning = f" {result.warnings[0]}" if result.warnings else ""
        self._status.setText(
            f"Status: {result.inference_status}; {result.mz.size} channels.{warning}"
        )
        self._export_btn.setEnabled(True)
        if self._table.rowCount():
            self._table.selectRow(0)

    @staticmethod
    def _rank(q: np.ndarray, effect: np.ndarray) -> np.ndarray:
        q_sort = np.where(np.isfinite(q), q, np.inf)
        effect_sort = np.where(np.isfinite(effect), np.abs(effect), -np.inf)
        return np.lexsort((-effect_sort, q_sort))

    def _populate_roi_result(self, result: RoiEnrichmentResult) -> None:
        order = self._rank(result.q_value, result.log2_fold_change)
        self._configure_table(["m/z", "log2 fold change", "q-value", "prevalence Δ"])
        self._table.setRowCount(order.size)
        for row, channel in enumerate(order):
            self._row_channels.append(int(channel))
            values = (
                result.mz[channel],
                result.log2_fold_change[channel],
                result.q_value[channel],
                result.prevalence_numerator[channel] - result.prevalence_denominator[channel],
            )
            self._set_row(row, values)
        finite = np.isfinite(result.log2_fold_change) & np.isfinite(result.q_value)
        if finite.any():
            y = -np.log10(np.maximum(result.q_value[finite], np.finfo(float).tiny))
            self._plot.plot(
                result.log2_fold_change[finite],
                y,
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=(80, 120, 210, 150),
            )
        self._plot.setLabel("bottom", "log2 fold change (numerator / denominator)")
        self._plot.setLabel("left", "-log10 q")

    def _populate_axis_result(self, result: AxisProfileResult) -> None:
        order = self._rank(result.q_value, result.spearman_rho)
        self._configure_table(
            ["m/z", "pattern", "Spearman ρ", "q-value", "end/start log2"]
        )
        self._table.setRowCount(order.size)
        for row, channel in enumerate(order):
            self._row_channels.append(int(channel))
            values = (
                result.mz[channel],
                str(result.pattern_label[channel]),
                result.spearman_rho[channel],
                result.q_value[channel],
                result.endpoint_log2_enrichment[channel],
            )
            self._set_row(row, values)
        self._plot.setLabel(
            "bottom", f"normalized position ({result.start_label} → {result.end_label})"
        )
        self._plot.setLabel("left", "mean intensity")

    def _configure_table(self, headers: list[str]) -> None:
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

    def _set_row(self, row: int, values: tuple[object, ...]) -> None:
        for column, value in enumerate(values):
            if isinstance(value, (float, np.floating)):
                text = "—" if not np.isfinite(value) else f"{float(value):.5g}"
            else:
                text = str(value)
            self._table.setItem(row, column, QTableWidgetItem(text))

    def _selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        valid = bool(rows) and self._result is not None
        self._show_btn.setEnabled(valid)
        if not valid or not isinstance(self._result, AxisProfileResult):
            return
        channel = self._row_channels[rows[0].row()]
        profile = self._result.mean_intensity[:, channel]
        finite = np.isfinite(profile)
        self._plot.clear()
        self._plot.plot(
            self._result.bin_centers[finite],
            profile[finite],
            pen=pg.mkPen("r", width=2),
            symbol="o",
        )
        self._plot.setLabel(
            "bottom",
            f"normalized position ({self._result.start_label} → {self._result.end_label})",
        )
        self._plot.setLabel("left", f"mean intensity at m/z {self._result.mz[channel]:.4f}")

    def _show_selected_ion(self) -> None:
        if self._result is None:
            return
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        channel = self._row_channels[rows[0].row()]
        self._session.request_show_mz(float(self._result.mz[channel]))

    def _export(self) -> None:
        if self._result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export DAPPLE analysis tables")
        if not directory:
            return
        raw_stem = (
            self._result.contrast_name
            if isinstance(self._result, RoiEnrichmentResult)
            else self._result.axis_name
        )
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_stem).strip("_") or "analysis"
        try:
            exported = export_analysis_result(self._result, Path(directory), stem=stem)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._status.setText(f"Exported {len(exported.table_paths)} table(s) and manifest.")


__all__ = ["DevelopmentalAnalysisWidget"]
