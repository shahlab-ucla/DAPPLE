"""Threshold explorer: re-threshold the most recent run without re-running operators.

The pipeline runner records per-candidate filter values for every operator that
drops peaks (consensus alignment's prominence + prevalence, peak picker's SNR,
reference-ion detection's prevalence). This panel reads those arrays from the
last run's diagnostics and shows, for each filter:

- the empirical CDF of candidate values (so you can see whether the rejection
  cutoff sits in a dense or sparse region of the distribution)
- a slider that lets you preview a different threshold
- a live count of "X candidates would survive at threshold Y"

Nothing actually re-runs — the panel is purely a what-if tool. Once you've
identified a threshold you like, switch to the wizard's Workflow page, edit the
matching operator's parameter, and click Run pipeline. The runner will
short-circuit every node that didn't change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dapple.widgets._session import MsiSession, default_session

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _ExplorerSpec:
    """One re-thresholdable filter, surfaced from a node's diagnostic payload."""

    label: str
    node_id: str
    op_name: str
    param_name: str
    payload_key: str
    threshold_summary_key: str | None
    direction: str  # "above" → keep candidates with value > threshold; "below" inverted
    tooltip: str


_EXPLORERS: tuple[_ExplorerSpec, ...] = (
    _ExplorerSpec(
        label="Consensus prevalence",
        node_id="consensus",
        op_name="kde_consensus_alignment",
        param_name="min_prevalence",
        payload_key="all_candidate_prevalence",
        threshold_summary_key="min_prevalence_threshold",
        direction="above",
        tooltip=(
            "Per-candidate prevalence (fraction of pixels carrying the candidate "
            "consensus peak). The current min_prevalence drops peaks below the "
            "threshold; this slider lets you see how many would survive at any "
            "other threshold without re-running the operator."
        ),
    ),
    _ExplorerSpec(
        label="Consensus KDE prominence",
        node_id="consensus",
        op_name="kde_consensus_alignment",
        param_name="min_prominence_quantile",
        payload_key="all_local_max_density",
        threshold_summary_key=None,  # use prominence_threshold_value below
        direction="above",
        tooltip=(
            "KDE density at every local maximum (pre prominence-quantile filter). "
            "Slide the threshold to see how many maxima would survive — useful "
            "when too many noise peaks are getting through or when sharp peaks "
            "are being filtered out."
        ),
    ),
)


class ThresholdExplorerPanel(QWidget):
    """Re-threshold the most recent pipeline run without re-running anything.

    Reads diagnostic payloads off ``MsiSession`` (specifically, off the
    ``WizardWidget``'s ``_last_run_result`` if a wizard is active). When no run
    has been recorded yet, the panel shows a placeholder.
    """

    def __init__(
        self,
        napari_viewer=None,  # noqa: ANN001 — napari.Viewer | None, optional
        session: MsiSession | None = None,
        get_run_result: Callable[[], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = session or default_session()
        # ``get_run_result`` is the bridge between this panel and whoever owns the
        # run results (typically a WizardWidget). When called from the napari
        # Plugins menu without a wizard, we attempt to find one via the active
        # napari viewer's dock widgets; failing that, the panel just shows the
        # placeholder.
        self._get_run_result = get_run_result or _make_default_run_result_getter(napari_viewer)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Threshold explorer</b>"))
        self._status = QLabel(
            "Run the pipeline once to populate the rejection-budget arrays."
        )
        self._status.setStyleSheet("color: #888; font-style: italic;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Filter selector.
        selector = QHBoxLayout()
        selector.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        for spec in _EXPLORERS:
            self._filter_combo.addItem(spec.label)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        selector.addWidget(self._filter_combo, stretch=1)
        layout.addLayout(selector)

        self._description = QLabel("")
        self._description.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._description.setWordWrap(True)
        layout.addWidget(self._description)

        # CDF plot.
        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.setLabel("bottom", "filter value")
        self._plot.setLabel("left", "fraction kept (cumulative)")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._cdf_curve = self._plot.plot(pen=pg.mkPen("k", width=1.5))
        self._threshold_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("r", width=1.5)
        )
        self._threshold_line.sigPositionChanged.connect(self._on_slider_moved)
        self._plot.addItem(self._threshold_line)
        layout.addWidget(self._plot, stretch=1)

        # Live count + suggestion.
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("font-size: 10pt; color: #d4d4d4;")
        layout.addWidget(self._count_label)

        # Refresh button.
        controls = QHBoxLayout()
        controls.addStretch(1)
        refresh = QPushButton("Refresh from last run")
        refresh.setToolTip(
            "Re-read the most recent pipeline run's diagnostics. Click this "
            "after a successful Run to repopulate the CDF."
        )
        refresh.clicked.connect(self._refresh)
        controls.addWidget(refresh)
        layout.addLayout(controls)

        # Cache: the most-recent (sorted-values, sort_idx) per filter spec.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self._refresh()

    # --- data ingestion ---------------------------------------------------------

    def _refresh(self) -> None:
        run_result = self._get_run_result() if self._get_run_result else None
        if run_result is None:
            self._status.setText(
                "Run the pipeline once to populate the rejection-budget arrays."
            )
            self._cdf_curve.setData([], [])
            self._threshold_line.setMovable(False)
            self._count_label.setText("")
            return
        # Walk the registered explorers and cache the data we have.
        diag_by_node = dict(getattr(run_result, "diagnostics", {}) or {})
        self._cache.clear()
        any_present = False
        for idx, spec in enumerate(_EXPLORERS):
            diags = diag_by_node.get(spec.node_id, [])
            for d in diags:
                payload = getattr(d, "payload", None) or {}
                if spec.payload_key not in payload:
                    continue
                values = np.asarray(payload[spec.payload_key], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                values_sorted = np.sort(values)
                # Pull the operator's actual current threshold for the marker.
                summary = getattr(d, "summary", {}) or {}
                if spec.threshold_summary_key is not None:
                    current = float(summary.get(spec.threshold_summary_key, np.nan))
                else:
                    # KDE prominence: stored explicitly in the payload as a 1-D array.
                    pt = payload.get("prominence_threshold_value")
                    current = float(pt[0]) if pt is not None and len(pt) else np.nan
                self._cache[idx] = (values_sorted, np.arange(values_sorted.size), current)
                any_present = True
                break
        if not any_present:
            self._status.setText(
                "The most recent run didn't emit any rejection-budget arrays. "
                "Run a pipeline that includes consensus alignment to populate them."
            )
        else:
            self._status.setText("Drag the red threshold line to preview survival counts.")
        self._on_filter_changed(self._filter_combo.currentIndex())

    # --- UI handlers ------------------------------------------------------------

    def _on_filter_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(_EXPLORERS):
            return
        spec = _EXPLORERS[idx]
        self._description.setText(spec.tooltip)
        cache = self._cache.get(idx)
        if cache is None:
            self._cdf_curve.setData([], [])
            self._count_label.setText(
                f"No data yet for {spec.label}. Run the pipeline first."
            )
            self._threshold_line.setMovable(False)
            return
        values, _, current = cache
        # CDF: x = sorted value, y = fraction at-or-below.
        n = values.size
        y = np.arange(1, n + 1) / n
        self._cdf_curve.setData(values, y)
        # Position the slider at the current operator threshold (if known) or the
        # 50th percentile otherwise.
        if np.isfinite(current):
            self._threshold_line.setPos(float(current))
        else:
            self._threshold_line.setPos(float(values[n // 2]))
        self._threshold_line.setMovable(True)
        # Constrain the slider's data range to the CDF span.
        try:
            self._threshold_line.setBounds([float(values.min()), float(values.max())])
        except Exception:  # noqa: BLE001 — older pyqtgraph
            pass
        self._update_count(spec, current)

    def _on_slider_moved(self) -> None:
        idx = self._filter_combo.currentIndex()
        if idx < 0 or idx >= len(_EXPLORERS):
            return
        cache = self._cache.get(idx)
        if cache is None:
            return
        spec = _EXPLORERS[idx]
        threshold = float(self._threshold_line.value())
        self._update_count(spec, threshold)

    def _update_count(self, spec: _ExplorerSpec, threshold: float) -> None:
        cache = self._cache.get(self._filter_combo.currentIndex())
        if cache is None:
            return
        values, _, current = cache
        if spec.direction == "above":
            survives = int((values >= threshold).sum())
        else:
            survives = int((values <= threshold).sum())
        n = values.size
        delta = ""
        if np.isfinite(current):
            current_survives = int(
                (values >= current).sum() if spec.direction == "above" else (values <= current).sum()
            )
            d = survives - current_survives
            sign = "+" if d > 0 else ""
            delta = (
                f"  (current threshold {current:.4g} keeps {current_survives}; "
                f"change: {sign}{d})"
            )
        self._count_label.setText(
            f"At threshold {threshold:.4g}, "
            f"<b>{survives}</b> of {n} candidates would survive ({survives / n:.0%}).{delta}"
        )


def _make_default_run_result_getter(napari_viewer):  # noqa: ANN001
    """Return a callable that retrieves the most-recent run result, hunting first
    for a ``WizardWidget`` in the napari viewer's dock widgets."""

    def _getter():  # noqa: ANN202
        if napari_viewer is None:
            return None
        # napari exposes added dock widgets via window._dock_widgets in recent
        # versions; defensive coding for API drift.
        try:
            from dapple.widgets.wizard import WizardWidget

            for w in getattr(napari_viewer.window, "_dock_widgets", {}).values():
                widget = getattr(w, "widget", None) or w
                if isinstance(widget, WizardWidget):
                    return getattr(widget, "_last_run_result", None)
        except Exception:  # noqa: BLE001
            pass
        return None

    return _getter
