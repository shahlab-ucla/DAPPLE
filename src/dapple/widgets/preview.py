"""PreviewWidget: per-pixel projections (TIC, RMS, median, ...) as a panel of buttons.

Each button computes the named projection on the session's MSIDataset and adds it
as a new napari `Image` layer (or replaces an existing layer with the same name).
The widget gracefully handles "no dataset loaded yet" by showing a placeholder.

The projection list comes from `dapple.viz.projections.PROJECTIONS` so it stays
in sync with `MSIDataset.project()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dapple.viz.projections import (
    PROJECTIONS,
    ProjectionSpec,
    percentile_contrast,
    projection_layer_name,
)
from dapple.widgets._session import MsiSession, default_session

if TYPE_CHECKING:
    import napari


class PreviewWidget(QWidget):
    """A column of buttons, one per projection. Clicking promotes it to a napari layer."""

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
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("Per-pixel projections")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        self._status = QLabel("(no dataset loaded)")
        self._status.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status)

        self._buttons: dict[str, QPushButton] = {}
        for spec in PROJECTIONS:
            row = QHBoxLayout()
            btn = QPushButton(spec.label)
            btn.setToolTip(spec.description)
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked=False, s=spec: self._render(s))
            self._buttons[spec.kind] = btn
            row.addWidget(btn)
            layout.addLayout(row)

        layout.addStretch(1)

        self._session.dataset_changed.connect(self._on_dataset_changed)
        # Initialize from current state.
        self._on_dataset_changed(self._session.dataset)

    # --- handlers ---------------------------------------------------------------

    def _on_dataset_changed(self, ds) -> None:  # noqa: ANN001 — Any to match Signal type
        enabled = ds is not None
        for btn in self._buttons.values():
            btn.setEnabled(enabled)
        if ds is None:
            self._status.setText("(no dataset loaded)")
        else:
            self._status.setText(
                f"{ds.n_pixels} pixels · grid {ds.grid_shape[0]}×{ds.grid_shape[1]} · "
                f"{ds.metadata.instrument_family} / {ds.metadata.ionization}"
            )

    def _render(self, spec: ProjectionSpec) -> None:
        ds = self._session.dataset
        if ds is None or self._viewer is None:
            return
        img = np.asarray(ds.project(spec.kind))
        stem = Path(ds.identity.source_path).stem or "dataset"
        layer_name = projection_layer_name(spec, stem)
        # Robust contrast: 1st-99th percentile of non-zero pixels avoids the common
        # "everything looks bimodal" problem when a tissue mask gives a lot of zeros
        # outside the sample, or when the dataset is pre-normalized (constant TIC).
        clim = percentile_contrast(img, lo_pct=1.0, hi_pct=99.0)
        if layer_name in self._viewer.layers:
            self._viewer.layers[layer_name].data = img
            self._viewer.layers[layer_name].contrast_limits = clim
        else:
            self._viewer.add_image(
                img,
                name=layer_name,
                colormap=spec.colormap,
                contrast_limits=clim,
                metadata={
                    "projection": spec.kind,
                    "projection_label": spec.label,
                    "msi_dataset_hash": ds.hash(),
                },
            )
