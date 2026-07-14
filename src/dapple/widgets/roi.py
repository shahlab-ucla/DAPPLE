"""RoiWidget: synchronizes a napari `Shapes` layer with the session's RoiDef list.

A polygon drawn on the dedicated "MSI ROIs" Shapes layer is captured into a `RoiDef`,
named, and optionally tagged "background" (so the operator chain can treat
non-foreground pixels as the implicit reference). The widget shows the current ROI
list, lets the user rename or delete entries, and toggle the background flag.

The shape ↔ RoiDef mapping uses the shape's index in the layer plus a sidecar list
on the widget. Polygon vertices use napari's zero-based ``(y, x)`` data coordinates;
dataset spectrum coordinates are converted separately when masks are sampled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dapple.data.metadata import RoiDef
from dapple.widgets._session import MsiSession, adopt_dataset_from_viewer, default_session

if TYPE_CHECKING:
    import napari


SHAPES_LAYER_NAME = "MSI ROIs"


def polygons_from_napari_layer(layer) -> tuple[tuple[tuple[float, float], ...], ...]:  # noqa: ANN001
    """Pull polygon vertex tuples out of a napari Shapes layer.

    Returned tuples are ``((y, x), ...)`` per polygon, in pixel coordinates. Shapes
    that are not polygons (lines, ellipses, paths) and degenerate ones (< 3 vertices)
    are dropped. Used by both ``RoiWidget`` and ``SpectrumPanel`` so ROI discovery
    doesn't depend on which widget is alive at any given moment.
    """
    out: list[tuple[tuple[float, float], ...]] = []
    if layer is None:
        return tuple(out)
    try:
        data = list(layer.data)
        raw_types = getattr(layer, "shape_type", None)
        shape_types = list(raw_types) if raw_types is not None else ["polygon"] * len(data)
    except Exception:  # noqa: BLE001 — defensive: napari API drift
        return tuple(out)
    for i, shape in enumerate(data):
        try:
            verts = np.asarray(shape, dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if verts.ndim != 2 or verts.shape[1] < 2 or verts.shape[0] < 3:
            continue
        st = shape_types[i] if i < len(shape_types) else "polygon"
        if st != "polygon":
            continue
        out.append(tuple((float(p[0]), float(p[1])) for p in verts))
    return tuple(out)


def find_msi_shapes_layer(viewer):  # noqa: ANN001 — viewer: napari.Viewer | None
    """Locate the dedicated ``MSI ROIs`` Shapes layer if present, else fall back to
    any Shapes layer in the viewer so users who renamed the layer still get picked up.
    """
    if viewer is None:
        return None
    try:
        if SHAPES_LAYER_NAME in viewer.layers:
            return viewer.layers[SHAPES_LAYER_NAME]
    except Exception:  # noqa: BLE001
        pass
    try:
        for layer in viewer.layers:
            if type(layer).__name__ == "Shapes":
                return layer
    except Exception:  # noqa: BLE001
        pass
    return None


class RoiWidget(QWidget):
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
        self._roi_names: list[str] = []
        self._is_background: list[bool] = []
        self._suppress_layer_event = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Polygon ROIs"))

        controls = QHBoxLayout()
        self._add_btn = QPushButton("Add ROI layer")
        self._add_btn.clicked.connect(self._ensure_shapes_layer)
        controls.addWidget(self._add_btn)
        self._sync_btn = QPushButton("Sync from layer")
        self._sync_btn.clicked.connect(self._sync_from_layer)
        controls.addWidget(self._sync_btn)
        layout.addLayout(controls)

        self._bg_outside = QCheckBox("Treat outside-ROI pixels as background")
        self._bg_outside.setChecked(False)
        self._bg_outside.setToolTip(
            "Sets the default on the optional background-subtraction card. Explicit "
            "background polygons take priority."
        )
        layout.addWidget(self._bg_outside)

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Name", "Background", "Vertices"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table)

        self._session.dataset_changed.connect(self._on_dataset_changed)
        self._on_dataset_changed(self._session.dataset)

    # --- napari layer ------------------------------------------------------------

    def _ensure_shapes_layer(self):
        if self._viewer is None:
            return None
        if SHAPES_LAYER_NAME in self._viewer.layers:
            layer = self._viewer.layers[SHAPES_LAYER_NAME]
        else:
            layer = self._viewer.add_shapes(
                name=SHAPES_LAYER_NAME,
                shape_type="polygon",
                edge_color="orange",
                face_color=[0, 0, 0, 0],
                edge_width=1.0,
            )
        # Connect to every event-source we know about. napari's Shapes-layer event
        # surface has shifted shape across versions: ``events.data`` is the canonical
        # signal but is sometimes only emitted on full reassignment, while interactive
        # polygon completion has been fired through ``set_data`` and ``refresh`` in
        # different builds. Connecting to all of them is the cheapest way to be
        # robust without version-sniffing.
        for event_name in ("data", "set_data", "refresh", "current_properties"):
            try:
                getattr(layer.events, event_name).connect(self._on_layer_data_changed)
            except Exception:  # noqa: BLE001 — event missing in this napari version
                continue
        layer.mode = "add_polygon"
        return layer

    def _on_layer_data_changed(self, event=None) -> None:  # noqa: ANN001 — napari Event
        if self._suppress_layer_event:
            return
        self._sync_from_layer()

    def _sync_from_layer(self) -> None:
        if self._viewer is None:
            return
        layer = find_msi_shapes_layer(self._viewer)
        if layer is None:
            return
        polygons = polygons_from_napari_layer(layer)
        # Build RoiDef list from each polygon, preserving any existing per-shape name
        # and background flag from this widget's table (so editing the name once
        # sticks even after another draw).
        new_rois: list[RoiDef] = []
        for i, verts in enumerate(polygons):
            name = self._roi_names[i] if i < len(self._roi_names) else f"ROI {i + 1}"
            is_bg = self._is_background[i] if i < len(self._is_background) else False
            new_rois.append(RoiDef(name=name, vertices=verts, is_background=is_bg))
        # Pad book-keeping arrays.
        while len(self._roi_names) < len(new_rois):
            self._roi_names.append(f"ROI {len(self._roi_names) + 1}")
        while len(self._is_background) < len(new_rois):
            self._is_background.append(False)
        self._roi_names = self._roi_names[: len(new_rois)]
        self._is_background = self._is_background[: len(new_rois)]
        self._refresh_table(new_rois)
        self._session.set_rois(tuple(new_rois))

    def _refresh_table(self, rois: list[RoiDef]) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(len(rois))
        for r, roi in enumerate(rois):
            name_item = QTableWidgetItem(roi.name)
            name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 0, name_item)

            bg_item = QTableWidgetItem()
            bg_item.setFlags(
                bg_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            bg_item.setCheckState(
                Qt.CheckState.Checked if roi.is_background else Qt.CheckState.Unchecked
            )
            bg_item.setText("background" if roi.is_background else "foreground")
            self._table.setItem(r, 1, bg_item)

            verts_item = QTableWidgetItem(f"{len(roi.vertices)} vertices")
            verts_item.setFlags(verts_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(r, 2, verts_item)
        self._table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        col = item.column()
        if col == 0:
            self._roi_names[row] = item.text() or f"ROI {row + 1}"
        elif col == 1:
            self._is_background[row] = item.checkState() == Qt.CheckState.Checked
            item.setText("background" if self._is_background[row] else "foreground")
        # Re-emit ROIs through the session.
        self._sync_from_layer()

    # --- session ----------------------------------------------------------------

    def _on_dataset_changed(self, ds) -> None:  # noqa: ANN001
        self._add_btn.setEnabled(self._viewer is not None and ds is not None)
        self._sync_btn.setEnabled(self._viewer is not None and ds is not None)

    @property
    def background_outside(self) -> bool:
        return self._bg_outside.isChecked()
