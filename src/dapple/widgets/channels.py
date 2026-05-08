"""Unified Channels panel: per-row LUT + visibility, page-wide blending mode.

Replaces the older ``HyperspectralBrowser`` widget. The new panel lists every
viewable layer of the dataset in a single table:

- "Summary" rows for per-pixel projections (TIC, RMS, median, ...)
- "Channel" rows for consensus m/z peaks (only when a PeakMatrix is loaded)

Each row exposes:
- Visibility checkbox (creates / removes a napari layer on toggle)
- Colormap picker (LUT) per row — independent from the page default
- Min and max contrast spinners

A page-wide control sets the blending mode for every visible row at once
(`additive` / `translucent` / `opaque`), so combining channels into a single
composite image is one click rather than per-layer fiddling.

The panel observes `MsiSession.dataset_changed` so it refreshes when the wizard
finishes a run, and it shares all napari layers with whatever the wizard, preview,
or spectrum panel are doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dapple.data.coords import project_to_grid
from dapple.data.dataset import PeakList, PeakMatrix
from dapple.viz.projections import (
    PROJECTIONS,
    ProjectionSpec,
    channel_layer_name,
    percentile_contrast,
    projection_layer_name,
)
from dapple.widgets._session import MsiSession, default_session

if TYPE_CHECKING:
    import napari


COLORMAPS = (
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "turbo",
    "gray",
    "gray_r",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "bop blue",
    "bop orange",
    "bop purple",
)
"""Default LUT options. The first matches each ProjectionSpec's preferred colormap;
the saturated single-hue maps (red/green/blue/yellow/cyan/magenta) are the natural
choices when stacking multiple channels in additive blending mode."""

BLENDING_MODES = ("additive", "translucent", "opaque", "minimum")
"""napari layer blending modes. Additive is the natural choice for hyperspectral
overlays (signal sums across channels); translucent for selectively highlighting
one channel over others."""


@dataclass(frozen=True)
class _Row:
    """Identifies one row in the channels panel."""

    kind: Literal["summary", "channel"]
    key: str  # projection.kind for summaries; "mz_<idx>" for channels
    label: str
    default_colormap: str
    payload: int | None = None  # consensus index for channels, None for summaries


class ChannelsPanel(QWidget):
    """Unified per-channel viewer controls.

    Sections (top to bottom):
      1. Page-wide blending dropdown + bulk-toggle buttons.
      2. Summary projection rows (always 6).
      3. Consensus m/z channel rows (added once a PeakMatrix-backed dataset arrives;
         removed when the dataset reverts to PeakList).
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
        self._rows: list[_Row] = []
        self._row_widgets: list[dict[str, QWidget]] = []
        # Per-row cached projection image (HxW float32). Once a row has been
        # rendered, we hold onto the image so subsequent toggles don't re-project
        # from the (potentially Zarr-backed) PeakMatrix. Cleared when the
        # dataset changes.
        self._row_image_cache: dict[int, np.ndarray] = {}
        # Cached consensus_prevalence array for the current PeakMatrix dataset.
        # Avoids the np.asarray(...).sum(axis=0) recomputation in `_add_row`'s
        # hot path on each rebuild.
        self._cached_prevalence: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        title = QLabel("<b>Channels</b>")
        layout.addWidget(title)
        self._status = QLabel("(no dataset loaded)")
        self._status.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status)

        # Page-wide controls.
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Blending:"))
        self._blending_combo = QComboBox()
        self._blending_combo.addItems(list(BLENDING_MODES))
        self._blending_combo.setToolTip(
            "How the visible channels are composited in the napari canvas. "
            "Additive sums signal across channels (best for multi-LUT overlays). "
            "Translucent alpha-blends. Opaque hides everything below the top layer. "
            "Minimum is occasionally useful for masking."
        )
        self._blending_combo.currentTextChanged.connect(self._apply_blending_to_layers)
        controls.addWidget(self._blending_combo)
        controls.addStretch(1)
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setMinimum(1)
        self._top_n_spin.setMaximum(10000)
        self._top_n_spin.setValue(8)
        self._top_n_spin.setToolTip(
            "Number of consensus channels to show when clicking 'Show top-N'."
        )
        controls.addWidget(QLabel("Show top"))
        controls.addWidget(self._top_n_spin)
        self._show_top_btn = QPushButton("by prevalence")
        self._show_top_btn.clicked.connect(self._show_top_n_channels)
        controls.addWidget(self._show_top_btn)
        self._hide_all_btn = QPushButton("Hide all")
        self._hide_all_btn.clicked.connect(self._hide_all_visible)
        controls.addWidget(self._hide_all_btn)
        layout.addLayout(controls)

        # Mosaic / grid controls — napari has built-in support for tiling layers in
        # a grid; we expose the toggle and column count here.
        grid_row = QHBoxLayout()
        self._grid_btn = QPushButton("Mosaic mode")
        self._grid_btn.setCheckable(True)
        self._grid_btn.setToolTip(
            "Toggle napari's built-in grid mode: every visible layer is rendered in "
            "its own tile of a single canvas. Use this with a few visible channels "
            "to compare them side-by-side without overlay confusion."
        )
        self._grid_btn.toggled.connect(self._on_grid_toggled)
        grid_row.addWidget(self._grid_btn)
        grid_row.addWidget(QLabel("Columns:"))
        self._grid_cols_spin = QSpinBox()
        self._grid_cols_spin.setRange(0, 16)
        self._grid_cols_spin.setValue(0)
        self._grid_cols_spin.setSpecialValueText("auto")
        self._grid_cols_spin.setToolTip(
            "Number of columns in the mosaic. 'auto' lets napari pick a roughly "
            "square layout from the visible-layer count."
        )
        self._grid_cols_spin.valueChanged.connect(self._apply_grid_shape)
        grid_row.addWidget(self._grid_cols_spin)
        grid_row.addStretch(1)
        layout.addLayout(grid_row)

        # Channels table.
        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Show", "Layer", "LUT", "Contrast min", "Contrast max"]
        )
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self._table, stretch=1)

        # CRITICAL: connect itemChanged ONCE here, not in `_add_row`. Connecting
        # per row makes a single checkbox click fire the slot N times (once per
        # connection), which used to compute the projection N times and cause
        # the panel to feel sluggish on cohorts with many consensus channels.
        self._table.itemChanged.connect(self._on_item_changed)

        self._session.dataset_changed.connect(self._on_dataset_changed)
        # SpectrumPanel emits show_mz_requested when the user clicks a peak; we snap
        # to the nearest consensus row and toggle it visible.
        self._session.show_mz_requested.connect(self._on_show_mz_requested)
        self._on_dataset_changed(self._session.dataset)

    # --- session callbacks ------------------------------------------------------

    def _on_dataset_changed(self, ds) -> None:  # noqa: ANN001
        # Drop any leftover channel layers from the previous dataset.
        self._remove_managed_layers()
        self._row_image_cache.clear()
        self._cached_prevalence = None
        # Block table-level signals while we tear down + rebuild rows so the
        # per-row item-creations don't trigger a flood of itemChanged events
        # against partial state.
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._rows.clear()
        self._row_widgets.clear()
        self._table.blockSignals(False)

        if ds is None:
            self._status.setText("(no dataset loaded)")
            return

        # Block signals for the *whole* rebuild so creating N rows fires zero
        # itemChanged events. We unblock once at the end. Previously this was
        # implicit because each _add_row connected its own itemChanged slot, so
        # newly-created rows couldn't fire pre-creation events; with the bulk
        # connection we have to be explicit.
        self._table.blockSignals(True)
        try:
            # Section 1: per-pixel summary projections — always available.
            for spec in PROJECTIONS:
                self._add_row(
                    _Row(
                        kind="summary",
                        key=spec.kind,
                        label=spec.label,
                        default_colormap=spec.colormap,
                        payload=None,
                    )
                )

            # Section 2: consensus m/z channels — only when post-aligned.
            if isinstance(ds.backend, PeakMatrix):
                pm = ds.backend
                prev = ds.extra.get("consensus_prevalence")
                # Compute prevalence once per dataset; cached so toggle handlers
                # don't recompute it each time they read a row's metadata.
                if prev is None or len(prev) != pm.n_peaks:
                    # `(matrix > 0).sum(axis=0)` reads every chunk of the (n_pixels,
                    # n_peaks) array. On already-loaded numpy it's instant; on a
                    # Zarr-backed lazy backend it's the only place we want this
                    # full pass to happen.
                    prev = (np.asarray(pm.matrix[:]) > 0).sum(axis=0) / max(pm.n_pixels, 1)
                self._cached_prevalence = np.asarray(prev, dtype=np.float64)
                mz = np.asarray(pm.mz_axis[:])
                # Cycle of saturated single-hue LUTs so an additive overlay reads cleanly.
                hue_cycle = ("red", "green", "blue", "yellow", "cyan", "magenta")
                for c in range(pm.n_peaks):
                    self._add_row(
                        _Row(
                            kind="channel",
                            key=f"mz_{c}",
                            label=f"m/z {mz[c]:.4f}  ·  prev {self._cached_prevalence[c]:.0%}",
                            default_colormap=hue_cycle[c % len(hue_cycle)],
                            payload=c,
                        )
                    )
        finally:
            self._table.blockSignals(False)

        if isinstance(ds.backend, PeakMatrix):
            pm = ds.backend
            self._status.setText(
                f"{len(PROJECTIONS)} summary projections · {pm.n_peaks} consensus channels"
            )
            self._show_top_btn.setEnabled(True)
            self._top_n_spin.setEnabled(True)
            self._top_n_spin.setMaximum(int(pm.n_peaks))
        else:
            self._status.setText(
                f"{len(PROJECTIONS)} summary projections (run consensus alignment to "
                "see per-channel m/z layers)"
            )
            self._show_top_btn.setEnabled(False)
            self._top_n_spin.setEnabled(False)
        self._hide_all_btn.setEnabled(True)

    # --- table construction -----------------------------------------------------

    def _add_row(self, row: _Row) -> None:
        idx = self._table.rowCount()
        self._table.insertRow(idx)
        self._rows.append(row)
        widgets: dict[str, QWidget] = {}

        # Show checkbox
        show = QTableWidgetItem()
        show.setFlags(
            show.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
        )
        show.setCheckState(Qt.CheckState.Unchecked)
        self._table.setItem(idx, 0, show)
        widgets["show"] = show  # type: ignore[assignment]

        # Layer name
        kind_prefix = "summary" if row.kind == "summary" else "channel"
        name_item = QTableWidgetItem(f"{kind_prefix} · {row.label}")
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(idx, 1, name_item)
        widgets["name"] = name_item  # type: ignore[assignment]

        # LUT picker
        lut = QComboBox()
        lut.addItems(list(COLORMAPS))
        lut.setCurrentText(row.default_colormap)
        lut.currentTextChanged.connect(lambda v, r=idx: self._on_lut_changed(r, v))
        self._table.setCellWidget(idx, 2, lut)
        widgets["lut"] = lut

        # Contrast min / max
        cmin = QDoubleSpinBox()
        cmin.setRange(-1e15, 1e15)
        cmin.setDecimals(4)
        cmin.valueChanged.connect(lambda v, r=idx: self._on_contrast_changed(r))
        self._table.setCellWidget(idx, 3, cmin)
        widgets["cmin"] = cmin

        cmax = QDoubleSpinBox()
        cmax.setRange(-1e15, 1e15)
        cmax.setDecimals(4)
        cmax.valueChanged.connect(lambda v, r=idx: self._on_contrast_changed(r))
        self._table.setCellWidget(idx, 4, cmax)
        widgets["cmax"] = cmax

        self._row_widgets.append(widgets)

    # --- row event handlers -----------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        # Only react to the "show" column.
        if item.column() != 0:
            return
        row_idx = item.row()
        visible = item.checkState() == Qt.CheckState.Checked
        self._set_row_visible(row_idx, visible)

    def _on_lut_changed(self, row_idx: int, new_lut: str) -> None:
        if self._viewer is None:
            return
        layer_name = self._layer_name_for_row(row_idx)
        if layer_name in self._viewer.layers:
            self._viewer.layers[layer_name].colormap = new_lut

    def _on_contrast_changed(self, row_idx: int) -> None:
        if self._viewer is None:
            return
        layer_name = self._layer_name_for_row(row_idx)
        if layer_name not in self._viewer.layers:
            return
        widgets = self._row_widgets[row_idx]
        cmin = float(widgets["cmin"].value())  # type: ignore[arg-type]
        cmax = float(widgets["cmax"].value())  # type: ignore[arg-type]
        if cmax <= cmin:
            cmax = cmin + 1e-12
        self._viewer.layers[layer_name].contrast_limits = (cmin, cmax)

    def _set_row_visible(self, row_idx: int, visible: bool) -> None:
        ds = self._session.dataset
        if ds is None or self._viewer is None:
            return
        row = self._rows[row_idx]
        layer_name = self._layer_name_for_row(row_idx)
        if visible:
            # Hot path: prefer the cached image so toggling visible doesn't
            # re-project from the (potentially Zarr-backed) PeakMatrix.
            cached = self._row_image_cache.get(row_idx)
            if cached is not None:
                img = cached
            else:
                img = self._compute_row_image(row, ds)
                if img is None:
                    return
                self._row_image_cache[row_idx] = img
            cmin, cmax = percentile_contrast(img, lo_pct=1.0, hi_pct=99.0)
            widgets = self._row_widgets[row_idx]
            lut = widgets["lut"].currentText()  # type: ignore[union-attr]
            if layer_name in self._viewer.layers:
                # Layer already exists — only flip visibility. Avoid the
                # five `setattr`s of the previous code path which each fired
                # napari layer-event callbacks redundantly.
                self._viewer.layers[layer_name].visible = True
            else:
                self._viewer.add_image(
                    img,
                    name=layer_name,
                    colormap=lut,
                    contrast_limits=(cmin, cmax),
                    blending=self._blending_combo.currentText(),
                    metadata=self._layer_metadata(row, ds),
                )
            # Reflect picked contrast in the spinners (without firing change events).
            for w_key, val in (("cmin", cmin), ("cmax", cmax)):
                w = widgets[w_key]
                w.blockSignals(True)
                try:
                    w.setValue(float(val))  # type: ignore[arg-type]
                finally:
                    w.blockSignals(False)
        else:
            if layer_name in self._viewer.layers:
                self._viewer.layers.remove(layer_name)

    # --- show-m/z request from SpectrumPanel ------------------------------------

    def _on_show_mz_requested(self, mz: float) -> None:
        """A SpectrumPanel click asks us to surface the channel nearest this m/z.

        The match is by absolute distance to the consensus m/z axis. If no PeakMatrix
        backend is loaded the request is a no-op (there's nothing to surface — the
        SpectrumPanel handles pre-consensus rendering itself). Found rows are
        scrolled into view, ticked visible, and selected so the user sees them.
        """
        ds = self._session.dataset
        if ds is None or not isinstance(ds.backend, PeakMatrix):
            return
        axis = np.asarray(ds.backend.mz_axis[:])
        if axis.size == 0:
            return
        nearest_channel = int(np.argmin(np.abs(axis - float(mz))))
        # Find the table row corresponding to that channel index.
        target_row: int | None = None
        for idx, row in enumerate(self._rows):
            if row.kind == "channel" and row.payload == nearest_channel:
                target_row = idx
                break
        if target_row is None:
            return
        # Tick & show — handler below creates the layer if needed.
        item = self._table.item(target_row, 0)
        if item is None:
            return
        if item.checkState() != Qt.CheckState.Checked:
            self._table.blockSignals(True)
            item.setCheckState(Qt.CheckState.Checked)
            self._table.blockSignals(False)
            self._set_row_visible(target_row, True)
        self._table.scrollToItem(item, hint=QAbstractItemView.ScrollHint.PositionAtCenter)
        self._table.selectRow(target_row)

    # --- napari grid / mosaic mode ---------------------------------------------

    def _on_grid_toggled(self, checked: bool) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.grid.enabled = bool(checked)
        except Exception:  # noqa: BLE001 — older napari versions used a different API
            pass
        self._apply_grid_shape()

    def _apply_grid_shape(self) -> None:
        if self._viewer is None or not getattr(self._viewer.grid, "enabled", False):
            return
        cols = int(self._grid_cols_spin.value())
        try:
            # napari API: shape is (rows, cols); -1 = auto.
            rows = -1
            cols = cols if cols > 0 else -1
            self._viewer.grid.shape = (rows, cols)
        except Exception:  # noqa: BLE001
            pass

    # --- bulk actions -----------------------------------------------------------

    def _show_top_n_channels(self) -> None:
        """Tick the top-N channel rows by prevalence."""
        ds = self._session.dataset
        if ds is None or not isinstance(ds.backend, PeakMatrix):
            return
        prev = ds.extra.get("consensus_prevalence")
        if prev is None:
            return
        prev = np.asarray(prev)
        n = int(self._top_n_spin.value())
        order = np.argsort(prev)[::-1][:n]
        target_payloads = set(int(c) for c in order)
        # Tick the appropriate rows; let _on_item_changed handle layer creation.
        self._table.blockSignals(True)
        for idx, row in enumerate(self._rows):
            if row.kind != "channel":
                continue
            target = row.payload in target_payloads
            self._table.item(idx, 0).setCheckState(
                Qt.CheckState.Checked if target else Qt.CheckState.Unchecked
            )
        self._table.blockSignals(False)
        for idx, row in enumerate(self._rows):
            if row.kind != "channel":
                continue
            should_show = row.payload in target_payloads
            self._set_row_visible(idx, should_show)

    def _hide_all_visible(self) -> None:
        self._table.blockSignals(True)
        for idx in range(self._table.rowCount()):
            self._table.item(idx, 0).setCheckState(Qt.CheckState.Unchecked)
        self._table.blockSignals(False)
        for idx in range(self._table.rowCount()):
            self._set_row_visible(idx, False)

    def _apply_blending_to_layers(self, mode: str) -> None:
        if self._viewer is None:
            return
        for idx in range(self._table.rowCount()):
            name = self._layer_name_for_row(idx)
            if name in self._viewer.layers:
                self._viewer.layers[name].blending = mode

    # --- helpers ----------------------------------------------------------------

    def _layer_name_for_row(self, row_idx: int) -> str:
        ds = self._session.dataset
        if ds is None:
            return ""
        stem = Path(ds.identity.source_path).stem or "dataset"
        row = self._rows[row_idx]
        if row.kind == "summary":
            spec = next(s for s in PROJECTIONS if s.kind == row.key)
            return projection_layer_name(spec, stem)
        # channel
        c = int(row.payload or 0)  # type: ignore[arg-type]
        pm: PeakMatrix = ds.backend  # type: ignore[assignment]
        prev = ds.extra.get("consensus_prevalence")
        prev_v = float(prev[c]) if prev is not None and len(prev) > c else None
        return channel_layer_name(stem, float(pm.mz_axis[c]), prev_v)

    def _compute_row_image(self, row: _Row, ds) -> np.ndarray | None:  # noqa: ANN001
        if row.kind == "summary":
            return np.asarray(ds.project(row.key))  # type: ignore[arg-type]
        if row.kind == "channel" and isinstance(ds.backend, PeakMatrix):
            c = int(row.payload or 0)
            # Read just the column we need. ``ds.backend.matrix`` is typically a
            # numpy array (after consensus alignment) — column slicing is O(N)
            # and stride-friendly. If the backend is later switched to Zarr,
            # this is still the right call: zarr column slicing is O(chunks_y),
            # which is what we want.
            data = np.asarray(ds.backend.matrix[:, c])
            return project_to_grid(
                data.astype(np.float32, copy=False), ds.coords, ds.grid_shape
            )
        return None

    def _layer_metadata(self, row: _Row, ds) -> dict:  # noqa: ANN001
        if row.kind == "summary":
            return {
                "projection": row.key,
                "projection_label": row.label,
                "msi_dataset_hash": ds.hash(),
                "dapple_managed_by": "channels_panel",
            }
        c = int(row.payload or 0)
        pm: PeakMatrix = ds.backend  # type: ignore[assignment]
        return {
            "mz": float(pm.mz_axis[c]),
            "channel_idx": c,
            "msi_dataset_hash": ds.hash(),
            "dapple_managed_by": "channels_panel",
        }

    def _remove_managed_layers(self) -> None:
        if self._viewer is None:
            return
        # Iterate over a snapshot since removal mutates the layer list.
        for layer in list(self._viewer.layers):
            md = getattr(layer, "metadata", None) or {}
            if md.get("dapple_managed_by") == "channels_panel":
                self._viewer.layers.remove(layer)

    # --- public for tests -------------------------------------------------------

    def visible_layer_names(self) -> list[str]:
        out: list[str] = []
        for idx in range(self._table.rowCount()):
            item = self._table.item(idx, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(self._layer_name_for_row(idx))
        return out
