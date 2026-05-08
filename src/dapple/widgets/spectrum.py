"""SpectrumPanel: a pyqtgraph-backed mass spectrum view.

Shows the spectrum at the cursor pixel ("single pixel" mode) or aggregated across an
ROI ("polygon-aggregate" mode). Toggle between raw (PeakList input) and harmonized
(PeakMatrix backend after consensus alignment). The aggregation method is one of
mean / median / sum / max — for raw mode pre-consensus aggregation falls back to a
simple ±tol bin-and-sum because per-pixel peak m/z values don't align.

Uses pyqtgraph for fast updates on cursor moves; the plot widget itself is reset on
dataset change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from dapple.data.dataset import PeakList, PeakMatrix
from dapple.widgets._session import MsiSession, default_session

if TYPE_CHECKING:
    import napari


AggregationMethod = Literal["mean", "median", "sum", "max"]


class SpectrumPanel(QWidget):
    def __init__(
        self,
        napari_viewer: "napari.Viewer | None" = None,
        session: MsiSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = napari_viewer
        self._session = session or default_session()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Toolbar.
        bar = QHBoxLayout()
        self._raw_check = QCheckBox("Raw")
        self._raw_check.setChecked(True)
        self._raw_check.toggled.connect(self._refresh)
        bar.addWidget(self._raw_check)
        self._harm_check = QCheckBox("Harmonized")
        self._harm_check.setChecked(True)
        self._harm_check.toggled.connect(self._refresh)
        bar.addWidget(self._harm_check)

        bar.addWidget(QLabel("Aggregation:"))
        self._agg_combo = QComboBox()
        self._agg_combo.addItems(["mean", "median", "sum", "max"])
        self._agg_combo.currentTextChanged.connect(lambda *_: self._refresh())
        bar.addWidget(self._agg_combo)

        bar.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["single pixel", "polygon aggregate"])
        self._mode_combo.currentTextChanged.connect(lambda *_: self._refresh())
        bar.addWidget(self._mode_combo)

        self._log_check = QCheckBox("Log y")
        self._log_check.toggled.connect(self._set_log_axis)
        bar.addWidget(self._log_check)

        bar.addStretch(1)
        layout.addLayout(bar)

        self._info = QLabel("(no dataset loaded)")
        self._info.setStyleSheet("color: #888;")
        layout.addWidget(self._info)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.setLabel("bottom", "m/z")
        self._plot.setLabel("left", "intensity")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot, stretch=1)

        # Hint label below the plot — click semantics.
        self._click_hint = QLabel(
            "Tip: click a peak to surface the matching channel image."
        )
        self._click_hint.setStyleSheet("color: #999; font-size: 9pt;")
        layout.addWidget(self._click_hint)

        # Cache the (mz, intensity) of the most recently rendered series so click-handling
        # can find the nearest peak without re-walking the dataset.
        self._last_peaks_mz: np.ndarray | None = None
        self._click_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen("c", width=1.5))
        self._click_marker.hide()
        self._plot.addItem(self._click_marker)

        # Click → request_show_mz + draw marker.
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        # Connect to session events.
        self._session.dataset_changed.connect(lambda *_: self._refresh())
        self._session.selected_pixel_changed.connect(lambda *_: self._refresh())
        self._session.rois_changed.connect(lambda *_: self._refresh())

        # Wire napari viewer mouse callback to drive selected_pixel, and listen for
        # Shapes-layer events so polygon-aggregate mode picks up new polygons even
        # when the wizard's RoiWidget isn't alive (e.g. user opened this panel
        # straight from the Plugins menu).
        if napari_viewer is not None:
            try:
                napari_viewer.mouse_move_callbacks.append(self._on_mouse_move)
            except Exception:  # noqa: BLE001 — napari API drift
                pass
            self._wire_shapes_event_listeners(napari_viewer)

        self._refresh()

    def _wire_shapes_event_listeners(self, viewer) -> None:  # noqa: ANN001
        """Connect to viewer.layers events so this panel reacts to polygons being
        drawn / edited regardless of whether the wizard's RoiWidget is alive."""
        try:
            viewer.layers.events.inserted.connect(self._on_layer_inserted)
        except Exception:  # noqa: BLE001
            pass
        try:
            for layer in viewer.layers:
                self._connect_shapes_layer(layer)
        except Exception:  # noqa: BLE001
            pass

    def _on_layer_inserted(self, event) -> None:  # noqa: ANN001 — napari Event
        layer = getattr(event, "value", None) or getattr(event, "source", None)
        if layer is not None:
            self._connect_shapes_layer(layer)

    def _connect_shapes_layer(self, layer) -> None:  # noqa: ANN001
        if type(layer).__name__ != "Shapes":
            return
        # Multiple event names fire on different napari versions; connect to all of
        # them — duplicate refreshes are harmless because _refresh is idempotent.
        for event_name in ("data", "set_data", "refresh", "current_properties"):
            try:
                getattr(layer.events, event_name).connect(self._on_shapes_changed)
            except Exception:  # noqa: BLE001
                continue

    def _on_shapes_changed(self, event=None) -> None:  # noqa: ANN001
        # Only relevant in polygon-aggregate mode; in single-pixel mode polygons
        # don't matter.
        if self._mode_combo.currentText() == "polygon aggregate":
            self._refresh()

    def _on_plot_clicked(self, event) -> None:  # noqa: ANN001 — pyqtgraph MouseClickEvent
        """Single-click on a peak → snap the click to the nearest plotted m/z and ask
        the session to surface the matching image."""
        try:
            if event.double() or event.button() != Qt.MouseButton.LeftButton:
                return
            pos = event.scenePos()
        except Exception:  # noqa: BLE001 — defensive on minor pyqtgraph API drift
            return
        view = self._plot.getPlotItem().getViewBox()
        if view is None:
            return
        data_pos = view.mapSceneToView(pos)
        clicked_mz = float(data_pos.x())
        if not np.isfinite(clicked_mz) or self._last_peaks_mz is None:
            return
        peaks = self._last_peaks_mz
        if peaks.size == 0:
            return
        nearest = peaks[int(np.argmin(np.abs(peaks - clicked_mz)))]
        self._click_marker.setPos(float(nearest))
        self._click_marker.show()
        # Tell other widgets which m/z the user picked. The ChannelsPanel listens.
        self._session.request_show_mz(float(nearest))

    def _set_log_axis(self, checked: bool) -> None:
        self._plot.setLogMode(x=False, y=checked)
        # Re-draw stems so their baseline is positive when log-y is on (log(0) = -inf
        # would otherwise make the whole curve disappear).
        self._refresh()

    def _on_mouse_move(self, viewer, event) -> None:  # noqa: ANN001 — napari callback signature
        ds = self._session.dataset
        if ds is None:
            return
        try:
            pos = viewer.cursor.position
        except Exception:  # noqa: BLE001
            return
        # Cursor position is in world coords (data coords for an image layer).
        if len(pos) < 2:
            return
        y, x = float(pos[-2]), float(pos[-1])
        h, w = ds.grid_shape
        ix = int(round(x))
        iy = int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            self._session.set_selected_pixel(None)
            return
        # imzML coords are 1-indexed.
        self._session.set_selected_pixel((ix + 1, iy + 1))

    # --- rendering --------------------------------------------------------------

    def _refresh(self) -> None:
        ds = self._session.dataset
        self._plot.clear()
        # Re-add the click marker since clear() removed it. Keep it hidden until the
        # user clicks — preserves selection across refresh, but only after a click.
        self._plot.addItem(self._click_marker)
        # Reset the cached peak m/z list. _plot_stem appends to it as series are drawn.
        self._last_peaks_mz = None
        if ds is None:
            self._info.setText("(no dataset loaded)")
            return
        mode = self._mode_combo.currentText()
        agg = self._agg_combo.currentText()

        if mode == "single pixel":
            self._render_single_pixel(ds)
        else:
            self._render_polygon_aggregate(ds, agg)

    def _render_single_pixel(self, ds) -> None:  # noqa: ANN001
        sel = self._session.selected_pixel
        if sel is None:
            self._info.setText("Hover over a pixel to view its spectrum")
            return
        x, y = sel
        self._info.setText(f"single pixel: x={x}, y={y}")
        if self._raw_check.isChecked():
            mz, intensity = ds.pixel_spectrum(x, y)
            if mz.size > 0:
                self._plot_stem(mz, intensity, name="raw", color="b")
        if self._harm_check.isChecked() and isinstance(ds.backend, PeakMatrix):
            # Harmonized = the same pixel through the dense matrix.
            sel_mask = (ds.coords[:, 0] == x) & (ds.coords[:, 1] == y)
            idx = np.flatnonzero(sel_mask)
            if idx.size:
                row = np.asarray(ds.backend.matrix[int(idx[0]), :])
                mz_axis = np.asarray(ds.backend.mz_axis[:])
                nz = np.flatnonzero(row)
                if nz.size:
                    self._plot_stem(mz_axis[nz], row[nz], name="harmonized", color="r")

    def _render_polygon_aggregate(self, ds, method: AggregationMethod) -> None:  # noqa: ANN001
        # Build a pixel mask from the session's ROIs (foreground only). If the session
        # has no ROIs but a napari Shapes layer is sitting on the canvas with polygons
        # drawn, fall back to those — this keeps polygon-aggregate mode working when
        # the user opens this panel from the napari Plugins menu without going through
        # the wizard's RoiPage (which would otherwise be the only thing populating
        # session.rois).
        rois = self._session.rois
        if not rois:
            rois = self._fallback_rois_from_layer()
        foreground = [r for r in rois if not r.is_background]
        if not foreground:
            self._info.setText(
                "Draw at least one foreground ROI on the napari canvas — "
                "use the wizard's ROI page or add a Shapes layer + polygon tool."
            )
            return
        mask = self._mask_from_rois(ds, foreground)
        n_in = int(mask.sum())
        if n_in == 0:
            self._info.setText("ROI(s) cover no pixels")
            return
        self._info.setText(
            f"polygon aggregate: {n_in} pixels in {len(foreground)} ROI(s); "
            f"method={method}"
        )
        if self._harm_check.isChecked() and isinstance(ds.backend, PeakMatrix):
            mz_axis, agg = ds.backend.aggregate(mask, method)  # type: ignore[arg-type]
            nz = np.flatnonzero(agg > 0)
            if nz.size:
                self._plot_stem(mz_axis[nz], agg[nz], name="harmonized", color="r")
        if self._raw_check.isChecked() and isinstance(ds.backend, PeakList):
            mz_all, int_all = self._raw_aggregate_peaklist(ds, mask, method)
            if mz_all.size:
                self._plot_stem(mz_all, int_all, name="raw (binned)", color="b")

    def _fallback_rois_from_layer(self) -> tuple:
        """Build transient RoiDefs from polygons on the napari canvas.

        The ``RoiWidget`` is normally what populates ``session.rois``, but the user
        can also open the SpectrumPanel directly (Plugins menu) without ever going
        through the wizard's ROI page. In that case we introspect the napari viewer's
        Shapes layers ourselves so 'polygon aggregate' mode still works. All
        polygons are treated as foreground; the user must use the RoiWidget if they
        need to mark some as background.
        """
        from dapple.data.metadata import RoiDef
        from dapple.widgets.roi import (
            find_msi_shapes_layer,
            polygons_from_napari_layer,
        )

        if self._viewer is None:
            return ()
        layer = find_msi_shapes_layer(self._viewer)
        polygons = polygons_from_napari_layer(layer) if layer is not None else ()
        out = []
        for i, verts in enumerate(polygons):
            try:
                out.append(RoiDef(name=f"Polygon {i + 1}", vertices=verts))
            except ValueError:
                continue
        return tuple(out)

    def _raw_aggregate_peaklist(
        self, ds, mask: np.ndarray, method: AggregationMethod  # noqa: ANN001
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pre-consensus aggregation: pool peaks from masked pixels into ±50 ppm bins
        and aggregate intensities. This is a coarse view (the harmonized panel is
        the precise one); we surface it so the user can see what raw peaks look like
        before alignment.
        """
        pl: PeakList = ds.backend
        offsets = np.asarray(pl.offsets[:])
        peak_mz = np.asarray(pl.mz[:])
        peak_int = np.asarray(pl.intensity[:])
        # Per-peak pixel index, then keep only those in mask.
        peak_pixel = np.repeat(
            np.arange(pl.n_pixels, dtype=np.int64), np.diff(offsets).astype(np.int64)
        )
        keep = mask[peak_pixel]
        sub_mz = peak_mz[keep]
        sub_int = peak_int[keep]
        if sub_mz.size == 0:
            return np.empty(0), np.empty(0)
        # Log-space bin at 50 ppm.
        bw = 50e-6
        log_mz = np.log(sub_mz)
        bin_idx = np.floor((log_mz - log_mz.min()) / bw).astype(np.int64)
        # Aggregate per bin.
        n_bins = int(bin_idx.max()) + 1
        if method == "sum":
            out = np.bincount(bin_idx, weights=sub_int.astype(np.float64), minlength=n_bins)
        elif method == "max":
            out = np.zeros(n_bins, dtype=np.float64)
            np.maximum.at(out, bin_idx, sub_int.astype(np.float64))
        elif method == "mean":
            counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
            sums = np.bincount(bin_idx, weights=sub_int.astype(np.float64), minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(counts > 0, sums / counts, 0.0)
        else:  # median is expensive — approximate with mean for the raw view
            counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
            sums = np.bincount(bin_idx, weights=sub_int.astype(np.float64), minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(counts > 0, sums / counts, 0.0)
        # bin centers, dropping empty bins
        bin_centers_log = log_mz.min() + (np.arange(n_bins) + 0.5) * bw
        nz = np.flatnonzero(out > 0)
        return np.exp(bin_centers_log[nz]), out[nz].astype(np.float32)

    def _mask_from_rois(self, ds, rois) -> np.ndarray:  # noqa: ANN001
        from skimage.draw import polygon as sk_polygon

        h, w = ds.grid_shape
        grid_mask = np.zeros((h, w), dtype=bool)
        for r in rois:
            ys = np.asarray([v[0] for v in r.vertices])
            xs = np.asarray([v[1] for v in r.vertices])
            rr, cc = sk_polygon(ys, xs, shape=(h, w))
            grid_mask[rr, cc] = True
        # Map grid_mask back onto coords (n_pixels,).
        coords = ds.coords
        from dapple.data.coords import coords_to_grid_index

        flat_idx = coords_to_grid_index(coords, ds.grid_shape)
        return grid_mask.reshape(-1)[flat_idx]

    def _plot_stem(self, x: np.ndarray, y: np.ndarray, *, name: str, color: str) -> None:
        # Stem plot: vertical line per peak. pyqtgraph doesn't have one natively;
        # emulate with PlotCurveItem from interleaved (x, y) → (x, baseline) pairs.
        # Baseline must be a positive number when log-y is on, otherwise log(0) = -inf
        # and the whole curve disappears. Use a finite floor a few decades below the
        # smallest visible value.
        n = x.size
        if n == 0:
            return
        # Restrict to positive intensities — anything ≤ 0 is uninformative on either
        # linear or log axes, and log-y blows up on zeros.
        mask = np.isfinite(y) & (y > 0)
        if not mask.any():
            return
        x = x[mask]
        y = y[mask]
        n = x.size
        log_y = bool(self._log_check.isChecked())
        if log_y:
            baseline = float(y.min()) / 10.0
            baseline = max(baseline, 1e-30)
        else:
            baseline = 0.0

        xs = np.empty(n * 3, dtype=np.float64)
        ys = np.empty(n * 3, dtype=np.float64)
        xs[0::3] = x
        xs[1::3] = x
        xs[2::3] = np.nan
        ys[0::3] = baseline
        ys[1::3] = y
        ys[2::3] = np.nan
        self._plot.plot(
            xs,
            ys,
            pen=pg.mkPen(color, width=1.5),
            name=name,
            connect="finite",
        )
        # Track plotted m/z values so click-to-show-mz can snap to a real peak rather
        # than the user's click position. Across multiple series (raw + harmonized)
        # we accumulate the union.
        if self._last_peaks_mz is None:
            self._last_peaks_mz = x.copy()
        else:
            self._last_peaks_mz = np.unique(np.concatenate([self._last_peaks_mz, x]))
