"""WizardWidget: multi-page guided processing flow.

Pages, in order:
    1. LoadPage      — pick imzML/.cdf/directory, validate; populate session.
    2. ParamsPage    — confirm/override ExperimentParams.
    3. PreviewPage   — pick a projection to render as the main napari layer.
    4. RoiPage       — embed RoiWidget; hint: draw polygons on "MSI ROIs" layer.
    5. WorkflowPage  — render `recommend_pipeline(ep)` as editable param cards.
    6. ReviewPage    — final summary before running.
    7. RunPage       — execute pipeline, display per-node progress + diagnostics.

The wizard owns a single `MsiSession` instance which it shares with the embedded
preview, ROI, browser, and spectrum widgets (so they all observe the same dataset).
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import fields, is_dataclass, replace as dataclass_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from dapple.data.metadata import ExperimentParams
from dapple.io.cdf_image_reader import read_cdf_image
from dapple.io.cdf_reader import read_cdf
from dapple.io.imzml_reader import read_imzml
from dapple.io.imzml_writer import write_imzml
from dapple.io.spec_xml import make_provenance, write_spec_xml
from dapple.io.tiff_writer import write_hyperspectral_tiff
from dapple.ops.base import REGISTRY, OpParams, field_help, field_label
from dapple.pipeline import (
    Node,
    Pipeline,
    PipelineRunner,
    detect_library_versions,
    format_diagnostics,
    recommend_pipeline,
)
from dapple.viz.projections import (
    PROJECTIONS,
    percentile_contrast,
    projection_layer_name,
)
from dapple.widgets._session import MsiSession, default_session
from dapple.widgets.preview import PreviewWidget
from dapple.widgets.roi import RoiWidget

if TYPE_CHECKING:
    import napari


logger = logging.getLogger(__name__)


# ---------- Pages ------------------------------------------------------------------


class _BasePage(QWizardPage):
    def __init__(self, wizard: "WizardWidget", title: str) -> None:
        super().__init__()
        self._wizard_ref = wizard
        self.setTitle(title)


class LoadPage(_BasePage):
    """Pick a dataset (file or directory)."""

    file_loaded = Signal(object)  # MSIDataset

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "1. Load dataset")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select an imzML file, single .cdf, or a directory of .cdf files."))

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("path/to/dataset.imzML")
        path_row.addWidget(self._path_edit)
        self._browse_file = QPushButton("Browse file…")
        self._browse_file.clicked.connect(self._on_browse_file)
        path_row.addWidget(self._browse_file)
        self._browse_dir = QPushButton("Browse directory…")
        self._browse_dir.clicked.connect(self._on_browse_dir)
        path_row.addWidget(self._browse_dir)
        layout.addLayout(path_row)

        self._load_btn = QPushButton("Load")
        self._load_btn.clicked.connect(self._on_load)
        layout.addWidget(self._load_btn)

        self._info = QLabel("(not loaded)")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        layout.addStretch(1)

        self._loaded = False

    def _on_browse_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an MSI dataset",
            "",
            "MSI files (*.imzML *.imzml *.ibd *.cdf *.nc);;All files (*.*)",
        )
        if p:
            self._path_edit.setText(p)

    def _on_browse_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose a directory of .cdf files")
        if p:
            self._path_edit.setText(p)

    def _on_load(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Pick a path", "Please enter or browse to a dataset.")
            return
        path = Path(path_str)
        try:
            if path.is_dir():
                ds = read_cdf_image(path)
            elif path.suffix.lower() in {".imzml", ".ibd"}:
                ds = read_imzml(path)
            elif path.suffix.lower() in {".cdf", ".nc"}:
                ds = read_cdf(path)
            else:
                raise ValueError(f"unrecognized path: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", f"{e}\n\n{traceback.format_exc()}")
            return
        self._info.setText(
            f"loaded {path.name}: {ds.n_pixels} pixels, grid {ds.grid_shape}, "
            f"{ds.metadata.instrument_family}/{ds.metadata.ionization}/"
            f"{ds.metadata.profile_or_centroided}/{ds.metadata.polarity}"
        )
        self._wizard_ref.session.set_dataset(ds)
        # Anchor the pipeline-input snapshot here so any re-runs (after the user
        # tweaks operator parameters) feed this same input through the operators
        # rather than the post-consensus PeakMatrix from the previous run.
        self._wizard_ref.set_input_snapshot(ds)
        # Add an initial preview layer. We default to peak_count rather than TIC
        # because TIC is often a constant (vendor pre-normalized data — e.g. SCiLS Lab
        # exports every pixel scaled to the same TIC). Peak count varies meaningfully
        # in either case, making it the more informative default first view.
        viewer = self._wizard_ref.viewer
        if viewer is not None:
            stem = Path(ds.identity.source_path).stem or "dataset"
            spec = next(s for s in PROJECTIONS if s.kind == "peak_count")
            img = np.asarray(ds.project("peak_count"))
            clim = percentile_contrast(img, lo_pct=1.0, hi_pct=99.0)
            name = projection_layer_name(spec, stem)
            if name in viewer.layers:
                viewer.layers[name].data = img
                viewer.layers[name].contrast_limits = clim
            else:
                viewer.add_image(
                    img,
                    name=name,
                    colormap=spec.colormap,
                    contrast_limits=clim,
                    metadata={
                        "projection": spec.kind,
                        "projection_label": spec.label,
                        "msi_dataset_hash": ds.hash(),
                    },
                )
        self._loaded = True
        self.completeChanged.emit()
        self.file_loaded.emit(ds)

    def isComplete(self) -> bool:
        return self._loaded


class ParamsPage(_BasePage):
    """Confirm/override ExperimentParams.

    On enter, fields are auto-populated from the dataset metadata extracted by the
    reader (imzML CV terms, ANDI-MS attributes) plus any sidecar `metadata.json` /
    `.spec.xml` discovered next to the data. Each field gets a badge showing the
    source so the user can see at a glance which values were detected vs. defaulted.
    """

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "2. Experiment parameters")
        from dapple.data.metadata_source import labels_for_source

        self._labels_for_source = labels_for_source
        layout = QVBoxLayout(self)

        self._sidecar_label = QLabel("")
        self._sidecar_label.setWordWrap(True)
        layout.addWidget(self._sidecar_label)

        form = QFormLayout()
        layout.addLayout(form)

        self._inputs: dict[str, QWidget] = {}
        self._badges: dict[str, QLabel] = {}
        # Snapshot of the auto-detected values so "Reset to detected" can restore them.
        self._detected: dict[str, object] = {}
        self._detected_source: dict[str, str] = {}

        def _add_row(field_name: str, label: str, widget: QWidget) -> None:
            badge = QLabel("")
            badge.setStyleSheet("font-size: 10pt;")
            row_widget = QWidget(self)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(widget, stretch=1)
            row_layout.addWidget(badge)
            form.addRow(label, row_widget)
            self._inputs[field_name] = widget
            self._badges[field_name] = badge

        self._instrument = QComboBox()
        self._instrument.addItems(
            ["unknown", "tof_axial", "tof_reflectron", "qtof", "orbitrap", "fticr"]
        )
        _add_row("instrument_family", "Instrument family", self._instrument)

        self._ionization = QComboBox()
        self._ionization.addItems(["unknown", "maldi", "desi", "sims", "esi"])
        _add_row("ionization", "Ionization", self._ionization)

        self._mode = QComboBox()
        self._mode.addItems(["unknown", "centroided", "profile"])
        _add_row("profile_or_centroided", "Mode", self._mode)

        self._polarity = QComboBox()
        self._polarity.addItems(["positive", "negative"])
        _add_row("polarity", "Polarity", self._polarity)

        self._mz_min = QDoubleSpinBox()
        self._mz_min.setRange(0, 100000)
        self._mz_min.setDecimals(3)
        _add_row("mz_min", "m/z min", self._mz_min)

        self._mz_max = QDoubleSpinBox()
        self._mz_max.setRange(0, 100000)
        self._mz_max.setDecimals(3)
        _add_row("mz_max", "m/z max", self._mz_max)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0, 100000)
        self._pixel_size.setDecimals(3)
        self._pixel_size.setSpecialValueText("(unset)")
        _add_row("pixel_size_um", "Pixel size (µm)", self._pixel_size)

        # `sample_type` is intentionally not exposed in the wizard UI today: no
        # current operator consumes it. Future spatial-filter operators (e.g. Moran's
        # I with permutation null) are expected to switch their ON/OFF default based
        # on it; until then we keep the field on the data model (so JSON sidecars and
        # .spec.xml files can still set it) but don't ask the user to fill in
        # something that doesn't yet do anything.

        controls = QHBoxLayout()
        controls.addStretch(1)
        self._reset_btn = QPushButton("Reset to auto-detected")
        self._reset_btn.clicked.connect(self._reset_to_detected)
        controls.addWidget(self._reset_btn)
        layout.addLayout(controls)

        # Wire change signals so badges flip to "user override" when the user edits.
        for fname, widget in self._inputs.items():
            self._wire_change_signal(fname, widget)

    def _wire_change_signal(self, fname: str, widget: QWidget) -> None:
        if isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(lambda _v, f=fname: self._on_user_edit(f))
        elif isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(lambda _v, f=fname: self._on_user_edit(f))

    def _on_user_edit(self, fname: str) -> None:
        # Only flip to "user override" if the new value differs from the detected one.
        cur = self._read_value(fname)
        det = self._detected.get(fname)
        if cur != det:
            self._set_badge(fname, "user")
        else:
            self._set_badge(fname, self._detected_source.get(fname, "default"))

    def _set_badge(self, fname: str, src: str) -> None:
        text, color = self._labels_for_source(src)
        badge = self._badges.get(fname)
        if badge is None:
            return
        badge.setText(text)
        badge.setStyleSheet(f"color: {color}; font-size: 10pt;")

    def initializePage(self) -> None:
        ds = self._wizard_ref.session.dataset
        if ds is None:
            return
        ep = ds.metadata
        source: dict[str, str] = dict(ds.extra.get("metadata_source", {}))
        sidecar_path = ds.extra.get("sidecar_path")
        if sidecar_path:
            self._sidecar_label.setText(
                f"Loaded sidecar metadata from <code>{sidecar_path}</code>"
            )
            self._sidecar_label.setStyleSheet("color: #1565c0;")
        else:
            self._sidecar_label.setText(
                "Auto-populated from dataset CV terms / attributes. No sidecar file found."
            )
            self._sidecar_label.setStyleSheet("color: #666;")
        self._sidecar_label.setTextFormat(Qt.TextFormat.RichText)

        # Populate input widgets without firing change signals.
        for w in self._inputs.values():
            w.blockSignals(True)
        try:
            self._write_value("instrument_family", ep.instrument_family)
            self._write_value("ionization", ep.ionization)
            self._write_value("profile_or_centroided", ep.profile_or_centroided)
            self._write_value("polarity", ep.polarity)
            self._write_value("mz_min", float(ep.mz_min))
            self._write_value("mz_max", float(ep.mz_max))
            self._write_value("pixel_size_um", float(ep.pixel_size_um or 0))
        finally:
            for w in self._inputs.values():
                w.blockSignals(False)

        # Snapshot for reset.
        self._detected = {fname: self._read_value(fname) for fname in self._inputs}
        self._detected_source = {
            fname: source.get(fname, "default") for fname in self._inputs
        }
        for fname in self._inputs:
            self._set_badge(fname, self._detected_source[fname])

    def validatePage(self) -> bool:
        ds = self._wizard_ref.session.dataset
        if ds is None:
            return False
        try:
            ep = ExperimentParams(
                instrument_family=self._instrument.currentText(),  # type: ignore[arg-type]
                ionization=self._ionization.currentText(),  # type: ignore[arg-type]
                profile_or_centroided=self._mode.currentText(),  # type: ignore[arg-type]
                polarity=self._polarity.currentText(),  # type: ignore[arg-type]
                mz_min=float(self._mz_min.value()),
                mz_max=float(self._mz_max.value()),
                pixel_size_um=(float(self._pixel_size.value()) if self._pixel_size.value() > 0 else None),
                sample_type=ds.metadata.sample_type,  # carried through unchanged
            )
        except ValueError as e:
            QMessageBox.warning(self, "Invalid parameters", str(e))
            return False
        # Update the dataset's metadata + record per-field source for any user edits.
        from dataclasses import replace as drep

        existing_source = dict(ds.extra.get("metadata_source", {}))
        for fname in self._inputs:
            cur = self._read_value(fname)
            if cur != self._detected.get(fname):
                existing_source[fname] = "user"
            else:
                existing_source[fname] = self._detected_source.get(fname, "default")
        new_extra = {**ds.extra, "metadata_source": existing_source}
        new_ds = drep(ds, metadata=ep, extra=new_extra)
        self._wizard_ref.session.set_dataset(new_ds)
        # Apply the metadata edit to the pipeline-input snapshot too. Use
        # update_input_metadata (not set_input_snapshot) so we don't clobber the
        # snapshot's backend with a post-consensus PeakMatrix when revisiting this
        # page after a successful run.
        self._wizard_ref.update_input_metadata(ep)
        return True

    def _reset_to_detected(self) -> None:
        for fname, value in self._detected.items():
            self._inputs[fname].blockSignals(True)
            try:
                self._write_value(fname, value)
            finally:
                self._inputs[fname].blockSignals(False)
            self._set_badge(fname, self._detected_source.get(fname, "default"))

    def _read_value(self, fname: str) -> object:
        widget = self._inputs[fname]
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        return None

    def _write_value(self, fname: str, value: object) -> None:
        widget = self._inputs[fname]
        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QDoubleSpinBox):
            try:
                widget.setValue(float(value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass


class PreviewPage(_BasePage):
    """Pick a projection. Embeds the PreviewWidget."""

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "3. Preview projections")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Click a projection button to display it on the canvas."))
        self._preview = PreviewWidget(
            napari_viewer=wizard.viewer, session=wizard.session, parent=self
        )
        layout.addWidget(self._preview)


class RoiPage(_BasePage):
    """Draw polygon ROIs. Embeds the RoiWidget."""

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "4. Region(s) of interest")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Click 'Add ROI layer' to start drawing polygons on the napari canvas. "
                "Each polygon becomes one ROI; tick the 'Background' column to mark the "
                "polygon as a background reference rather than a structure of interest."
            )
        )
        self._roi = RoiWidget(napari_viewer=wizard.viewer, session=wizard.session, parent=self)
        layout.addWidget(self._roi)


class WorkflowPage(_BasePage):
    """Render the recommended pipeline as editable parameter cards.

    On top of the pipeline returned by ``recommend_pipeline`` we also surface
    two optional cleanup operators that the user can toggle on:

    - ``hot_pixel_filter`` injected at the *front* of the chain (before
      reference detection), since it cleans the input PeakList.
    - ``background_subtract`` injected at the *end* (after consensus), since
      it operates on the post-consensus PeakMatrix and needs ROIs.

    Each optional card carries an "Enable" checkbox. Disabled optional cards
    stay visible (so users can preview parameters) but are skipped when the
    Pipeline is built.
    """

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "5. Recommended workflow")
        self._layout = QVBoxLayout(self)
        self._cards: list[_NodeCard] = []
        # Persist optional-card enable state across rebuilds (Back/Forward).
        # Key: op_name. The user's choice survives every recommend_pipeline
        # rebuild on this page.
        self._optional_enabled: dict[str, bool] = {
            "hot_pixel_filter": False,
            "background_subtract": False,
        }
        self._summary = QLabel("(loading…)")
        self._summary.setWordWrap(True)
        self._layout.addWidget(self._summary)

        # Cards live in a scroll area so the page stays useful at small sizes.
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_inner = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_inner)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.addStretch(1)
        scroll.setWidget(scroll_inner)
        self._layout.addWidget(scroll, stretch=1)

        # Reset row.
        controls = QHBoxLayout()
        controls.addStretch(1)
        self._reset_btn = QPushButton("Reset all to defaults")
        self._reset_btn.setToolTip(
            "Re-run recommend_pipeline on the current ExperimentParams and clear "
            "every per-node parameter override."
        )
        self._reset_btn.clicked.connect(self._on_reset_defaults)
        controls.addWidget(self._reset_btn)
        self._layout.addLayout(controls)

    def initializePage(self) -> None:
        # Preserve any user edits when re-entering the page (Back navigation, or after
        # a Run when the user wants to tweak a parameter and re-run). The "Reset all to
        # defaults" button explicitly forces a fresh recommend_pipeline if the user
        # really wants to throw their edits away.
        self._rebuild_cards(use_proposed=True)

    def _rebuild_cards(self, *, use_proposed: bool) -> None:
        # Snapshot the user's optional-card enable state from the *currently*
        # rendered cards before tearing them down, so toggling a checkbox is
        # preserved across the rebuild.
        for card in self._cards:
            if card.is_optional():
                self._optional_enabled[card._node.op_name] = card.is_enabled()
            card.setParent(None)
        self._cards.clear()
        ds = self._wizard_ref.session.dataset
        if ds is None:
            self._summary.setText("(no dataset loaded)")
            return
        if use_proposed and self._wizard_ref._proposed_pipeline is not None:  # noqa: SLF001
            pipeline = self._wizard_ref._proposed_pipeline  # noqa: SLF001
            # Strip any previously-injected optional nodes so the rebuild is
            # idempotent — the cards re-add them based on _optional_enabled.
            stripped = tuple(
                n for n in pipeline.nodes
                if n.op_name not in {"hot_pixel_filter", "background_subtract"}
            )
            if len(stripped) != len(pipeline.nodes):
                pipeline = pipeline.__class__(
                    nodes=stripped, rng_seed=pipeline.rng_seed,
                    library_versions=pipeline.library_versions,
                )
        else:
            pipeline = recommend_pipeline(ds.metadata)
        self._wizard_ref._proposed_pipeline = pipeline  # noqa: SLF001
        self._summary.setText(
            f"{len(pipeline.nodes)} operators in the recommended chain. "
            f"Hover any field for guidance on when and how to adjust it. "
            f"Optional cleanup operators (hot-pixel filter, background subtract) "
            f"are listed at the top and bottom — enable them with the checkbox "
            f"on each card."
        )
        # Insert cards before the trailing stretch. If we have a previous run's
        # result on hand, look up the matching node's diagnostics so the card can
        # render rejection-budget guidance.
        last = getattr(self._wizard_ref, "_last_run_result", None)
        diags_by_node = (
            dict(getattr(last, "diagnostics", {}) or {}) if last is not None else {}
        )

        # 1. Optional pre-pipeline cleanup: hot_pixel_filter at the FRONT.
        self._add_optional_card(
            ds.metadata, op_name="hot_pixel_filter", node_id="hotpx",
            optional_hint=(
                "Replaces detector-glitch / matrix-crystal hot pixels with their "
                "neighbors' median TIC. Enable when the Preview shows isolated "
                "extreme-bright pixels that dominate auto-contrast."
            ),
        )

        # 2. Recommended-pipeline cards (always-on).
        for node in pipeline.nodes:
            warns = self._operator_warnings(node, ds.metadata)
            card = _NodeCard(
                node,
                warnings=warns,
                last_diagnostics=diags_by_node.get(node.id),
                parent=self,
                optional=False,
            )
            self._cards.append(card)
            self._scroll_layout.insertWidget(self._scroll_layout.count() - 1, card)

        # 3. Optional post-pipeline cleanup: background_subtract at the END.
        self._add_optional_card(
            ds.metadata, op_name="background_subtract", node_id="bgsub",
            optional_hint=(
                "Drops consensus channels dominated by background pixels. Requires "
                "consensus alignment AND at least one ROI on the dataset (draw a "
                "foreground polygon on the ROI page first)."
            ),
        )

    def _add_optional_card(
        self,
        ep: ExperimentParams,
        *,
        op_name: str,
        node_id: str,
        optional_hint: str,
    ) -> None:
        """Construct an optional card for ``op_name`` and append it to the page."""
        try:
            op = REGISTRY.get(op_name)()
        except KeyError:
            return  # Operator not registered — skip silently.
        node = Node(
            id=node_id,
            op_name=op_name,
            params=op.default_params(ep),
            upstream=(),
        )
        warns = self._operator_warnings(node, ep)
        last = getattr(self._wizard_ref, "_last_run_result", None)
        diags_by_node = (
            dict(getattr(last, "diagnostics", {}) or {}) if last is not None else {}
        )
        card = _NodeCard(
            node,
            warnings=warns,
            last_diagnostics=diags_by_node.get(node_id),
            parent=self,
            optional=True,
            enabled=self._optional_enabled.get(op_name, False),
            optional_hint=optional_hint,
        )
        self._cards.append(card)
        self._scroll_layout.insertWidget(self._scroll_layout.count() - 1, card)

    def _on_reset_defaults(self) -> None:
        self._wizard_ref._proposed_pipeline = None  # noqa: SLF001 — force rebuild
        self._rebuild_cards(use_proposed=False)

    def validatePage(self) -> bool:
        # Drop optional cards the user disabled, then re-thread upstream so each
        # node points to the previous *enabled* node. This keeps the chain
        # intact when an optional cleanup card is toggled on or off.
        active_cards = [c for c in self._cards if c.is_enabled()]
        nodes: list[Node] = []
        prev_id: str | None = None
        for card in active_cards:
            n = card.to_node()
            new_upstream: tuple[str, ...] = (prev_id,) if prev_id is not None else ()
            if n.upstream != new_upstream:
                n = Node(
                    id=n.id,
                    op_name=n.op_name,
                    params=n.params,
                    upstream=new_upstream,
                )
            nodes.append(n)
            prev_id = n.id
        pipeline = Pipeline(
            nodes=tuple(nodes),
            rng_seed=self._wizard_ref._proposed_pipeline.rng_seed,
            library_versions=detect_library_versions(),
        )
        self._wizard_ref._proposed_pipeline = pipeline
        return True

    @staticmethod
    def _operator_warnings(node: Node, ep: ExperimentParams) -> list[str]:
        try:
            op_cls = REGISTRY.get(node.op_name)
            return op_cls().validate(ep)
        except Exception as e:  # noqa: BLE001
            return [f"validate() raised: {e}"]


class ReviewPage(_BasePage):
    """Summary before running."""

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "6. Review")
        layout = QVBoxLayout(self)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        layout.addWidget(self._summary)

    def initializePage(self) -> None:
        ds = self._wizard_ref.session.dataset
        pipeline = self._wizard_ref._proposed_pipeline
        if ds is None or pipeline is None:
            self._summary.setText("(missing dataset or pipeline)")
            return
        lines = [
            f"Dataset: {ds.identity.source_path}",
            f"  pixels:        {ds.n_pixels}",
            f"  grid:          {ds.grid_shape[0]}×{ds.grid_shape[1]}",
            f"  hash:          {ds.hash()[:16]}…",
            f"  experiment:    {ds.metadata.instrument_family} / "
            f"{ds.metadata.ionization} / {ds.metadata.profile_or_centroided} / "
            f"{ds.metadata.polarity}",
            f"  m/z range:     {ds.metadata.mz_min:.3f}–{ds.metadata.mz_max:.3f}",
            f"  ROIs:          {len(self._wizard_ref.session.rois)} polygon(s)",
            "",
            f"Pipeline ({len(pipeline.nodes)} nodes):",
        ]
        for n in pipeline.nodes:
            lines.append(f"  - {n.id}: {n.op_name}")
            for f in fields(n.params):
                lines.append(f"      {f.name} = {getattr(n.params, f.name)!r}")
        lines.append(f"\nrng_seed: {pipeline.rng_seed}")
        self._summary.setPlainText("\n".join(lines))


class _ProgressRelay(QObject):
    """Marshals worker-thread progress callbacks onto the GUI thread via a Qt signal."""

    progress = Signal(int, int, str)


class RunPage(_BasePage):
    """Execute pipeline (in a worker thread so the GUI stays responsive), write outputs."""

    def __init__(self, wizard: "WizardWidget") -> None:
        super().__init__(wizard, "7. Run")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Run the pipeline. On completion, save to disk."))

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        # Keep the log a sensible minimum size but allow it to grow / shrink with the
        # dock widget — no fixed sizes that would resist resizing.
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._log.setMinimumHeight(80)
        self._log.setMaximumHeight(200)
        layout.addWidget(self._log)

        # Live diagnostic plots — one small pyqtgraph plot per completed node.
        # The container is a scrollable column so a long pipeline doesn't overflow.
        from qtpy.QtWidgets import QScrollArea as _QScrollArea

        self._plots_scroll = _QScrollArea()
        self._plots_scroll.setWidgetResizable(True)
        self._plots_scroll.setMinimumHeight(220)
        self._plots_inner = QWidget()
        self._plots_inner_layout = QVBoxLayout(self._plots_inner)
        self._plots_inner_layout.setContentsMargins(0, 0, 0, 0)
        self._plots_inner_layout.setSpacing(8)
        self._plots_inner_layout.addStretch(1)
        self._plots_scroll.setWidget(self._plots_inner)
        layout.addWidget(self._plots_scroll, stretch=1)

        actions = QHBoxLayout()
        self._run_btn = QPushButton("Run pipeline")
        self._run_btn.clicked.connect(self._on_run)
        actions.addWidget(self._run_btn)
        # "Tune & rerun" sends the user back to the workflow page so they can
        # adjust parameters informed by the rejection-budget guidance the cards
        # render after a run. After they reach Run again they click Run pipeline.
        self._tune_btn = QPushButton("Tune && rerun")
        self._tune_btn.setToolTip(
            "Go back to the workflow page to adjust operator parameters. The "
            "cards will show how many candidates each filter dropped on the last "
            "run so you know which knob to tune."
        )
        self._tune_btn.clicked.connect(self._on_tune_and_rerun)
        self._tune_btn.setEnabled(False)
        actions.addWidget(self._tune_btn)
        self._save_spec_btn = QPushButton("Save .spec.xml")
        self._save_spec_btn.clicked.connect(self._on_save_spec)
        self._save_spec_btn.setEnabled(False)
        actions.addWidget(self._save_spec_btn)
        self._save_imzml_btn = QPushButton("Save imzML")
        self._save_imzml_btn.clicked.connect(self._on_save_imzml)
        self._save_imzml_btn.setEnabled(False)
        actions.addWidget(self._save_imzml_btn)
        self._save_tiff_btn = QPushButton("Save TIFF + CSV")
        self._save_tiff_btn.clicked.connect(self._on_save_tiff)
        self._save_tiff_btn.setEnabled(False)
        actions.addWidget(self._save_tiff_btn)
        layout.addLayout(actions)

        self._run_result = None
        self._completed = False
        self._worker = None
        # Cache hash of the (pipeline, input dataset) that produced `_run_result`. When
        # the user goes back to the WorkflowPage, tweaks something, and comes back, we
        # detect the mismatch and disable Save until they re-Run.
        self._last_run_signature: str | None = None
        self._relay = _ProgressRelay(self)
        self._relay.progress.connect(self._on_progress)
        # Reuse a single PipelineRunner across runs so per-node caching short-circuits
        # unchanged operators on subsequent re-runs.
        self._runner = PipelineRunner()

    def initializePage(self) -> None:
        """Each time the user lands on this page, check whether the pipeline they're
        about to run matches the last completed result. If not, gate Save behind a
        fresh Run."""
        sig = self._current_signature()
        if self._completed and sig is not None and sig != self._last_run_signature:
            self._log.append(
                "\n[!] Parameters changed since the last run — click Run pipeline "
                "to refresh results before saving."
            )
            self._save_spec_btn.setEnabled(False)
            self._save_imzml_btn.setEnabled(False)
            self._save_tiff_btn.setEnabled(False)
            self._completed = False
            self.completeChanged.emit()

    def _current_signature(self) -> str | None:
        """Hash the (pipeline, input-dataset) pair that would run if the user clicks Run.

        Returns None when either piece is missing.
        """
        pipeline = self._wizard_ref._proposed_pipeline  # noqa: SLF001
        ds = self._wizard_ref.session.dataset
        if pipeline is None or ds is None:
            return None
        # Use the *original* input hash, not the post-run dataset hash, so re-running
        # with identical params hits the cache.
        input_hash = self._wizard_ref._input_hash_at_workflow  # noqa: SLF001
        if input_hash is None:
            input_hash = ds.hash()
        return pipeline.hash(input_hash=input_hash)

    def _on_progress(self, i: int, n: int, msg: str) -> None:
        pct = int(100 * i / max(n, 1))
        self._progress.setValue(pct)
        self._log.append(f"[{i}/{n}] {msg}")

    def _on_run(self) -> None:
        ds = self._wizard_ref.session.dataset
        pipeline = self._wizard_ref._proposed_pipeline
        if ds is None or pipeline is None:
            QMessageBox.warning(self, "Nothing to run", "Load a dataset first.")
            return
        # Use the input snapshot captured before WorkflowPage so re-runs hit the cache.
        input_ds = self._wizard_ref._input_dataset_snapshot or ds  # noqa: SLF001
        self._log.clear()
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")

        relay = self._relay
        runner = self._runner

        # Inline import — napari only available when running inside napari.
        from napari.qt import thread_worker

        @thread_worker
        def _do_run():  # runs in a worker thread; emits via the relay signal
            return runner.run(
                pipeline,
                input_ds,
                progress=lambda i, n, msg: relay.progress.emit(i, n, msg),
            )

        worker = _do_run()
        worker.returned.connect(self._on_run_finished)
        worker.errored.connect(self._on_run_error)
        worker.finished.connect(self._on_worker_finished)
        worker.start()
        self._worker = worker

    def _on_run_finished(self, run_result) -> None:  # noqa: ANN001 — RunResult
        self._run_result = run_result
        # Hand the result up to the wizard so the WorkflowPage cards can read its
        # diagnostics and render rejection-budget guidance the next time the user
        # visits the workflow page.
        self._wizard_ref._last_run_result = run_result  # noqa: SLF001
        self._progress.setValue(100)
        # Render the per-node diagnostic block in a tabular, health-annotated
        # format (✓/⚠/✗ glyphs flag values that look stable, borderline, or
        # unhealthy — see ``dapple.pipeline.diag_format`` for the rubric).
        self._log.append("")
        for line in format_diagnostics(run_result):
            self._log.append(line)
        # Replace the live diagnostic plots with the latest run's payload.
        self._render_diagnostic_plots(run_result)
        self._wizard_ref.session.set_dataset(run_result.output)
        self._last_run_signature = self._current_signature()
        self._save_spec_btn.setEnabled(True)
        self._save_imzml_btn.setEnabled(True)
        self._save_tiff_btn.setEnabled(True)
        self._tune_btn.setEnabled(True)
        self._completed = True
        self.completeChanged.emit()

    def _render_diagnostic_plots(self, run_result) -> None:  # noqa: ANN001
        """Replace the per-node diagnostic plot strip with up-to-date plots from the
        most recent run. Each node's ``Diagnostic.figure_hint`` chooses the plot."""
        # Clear any previous plot widgets, leaving the trailing stretch in place.
        while self._plots_inner_layout.count() > 1:
            item = self._plots_inner_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)
        for node_id, diags in run_result.diagnostics.items():
            for d in diags:
                widget = _build_diagnostic_plot(node_id, d)
                if widget is not None:
                    self._plots_inner_layout.insertWidget(
                        self._plots_inner_layout.count() - 1, widget
                    )

    def _on_run_error(self, exc: BaseException) -> None:
        self._log.append(f"\nFAILED: {exc!r}")
        QMessageBox.critical(self, "Pipeline failed", str(exc))

    def _on_tune_and_rerun(self) -> None:
        """Navigate back to the WorkflowPage so the user can adjust parameters
        informed by the rejection-budget guidance now visible on each card.

        Implementation: ``QWizard`` has no direct "jump to page" API, so we walk
        the navigation stack backwards via ``back()`` until we land on the
        WorkflowPage. The page's ``initializePage`` is called as a side-effect
        and re-renders the cards with up-to-date diagnostics.
        """
        wiz = self._wizard_ref
        for _ in range(len(wiz.pageIds())):
            current = wiz.currentPage()
            if isinstance(current, WorkflowPage):
                return
            wiz.back()

    def _on_worker_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run pipeline")

    def _on_save_spec(self) -> None:
        if self._run_result is None or self._wizard_ref._proposed_pipeline is None:
            return
        ds = self._wizard_ref.session.dataset
        if ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save .spec.xml", "run.spec.xml", "Spec XML (*.spec.xml)"
        )
        if not path:
            return
        prov = make_provenance(
            plugin_version=_plugin_version(),
            input_dataset_hash=ds.hash(),
            declared_md5=ds.identity.declared_md5,
        )
        write_spec_xml(
            path,
            pipeline=self._wizard_ref._proposed_pipeline,
            experiment_params=ds.metadata,
            provenance=prov,
            diagnostics=self._run_result.diagnostics,
        )
        self._log.append(f"wrote {path}")

    def _on_save_imzml(self) -> None:
        if self._run_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save imzML", "harmonized.imzML", "imzML (*.imzML)"
        )
        if not path:
            return
        result = write_imzml(self._run_result.output, path)
        self._log.append(
            f"wrote {result.imzml_path} (+ {result.ibd_path}); "
            f"{result.n_spectra} spectra / {result.total_peaks} peaks"
        )

    def _on_save_tiff(self) -> None:
        if self._run_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save TIFF", "harmonized.tif", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            result = write_hyperspectral_tiff(self._run_result.output, path)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot write TIFF", str(e))
            return
        self._log.append(
            f"wrote {result.tiff_path} ({result.n_pages} pages); "
            f"sidecar {result.csv_path.name}"
        )

    def isComplete(self) -> bool:
        return self._completed


# ---------- Node card --------------------------------------------------------------


_OP_DISPLAY_NAMES: dict[str, str] = {
    "detect_reference_ions": "1. Reference-ion detection",
    "empirical_tolerance_from_reference_ions": "2. Empirical tolerance fit",
    "msiwarp_recalibrate": "3. Mass recalibration",
    "median_normalize": "4. Per-pixel normalization (median)",
    "tic_normalize": "4. Per-pixel normalization (TIC)",
    "reference_ion_normalize": "4. Per-pixel normalization (reference-ion)",
    "snr_peak_pick": "5. Peak picking (SNR / centroided)",
    "cwt_peak_pick": "5. Peak picking (CWT / profile)",
    "kde_consensus_alignment": "6. Consensus peak alignment",
    "morans_i_permutation": "7. Spatial filter (Moran's I)",
    "background_subtract": "(optional) Background subtraction",
    "hot_pixel_filter": "(optional) Hot-pixel correction",
}


_PLOT_TOOLTIPS: dict[str, str] = {
    "line:kde_density_with_consensus_marks": (
        "<b>KDE density of pooled peaks vs. m/z</b>"
        "<p>Blue curve: kernel-density estimate of every pixel's peaks pooled "
        "in log-m/z space. Red vertical lines mark consensus peaks the operator "
        "kept after the prominence + prevalence filters.</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Sharp, well-separated peaks</b> with one red mark per peak — "
        "indicates the bandwidth is well-tuned.</li>"
        "<li><b>Broad humps with multiple red marks inside</b> — bandwidth too "
        "narrow; raise <code>bandwidth_ppm</code> or <code>bandwidth_scale</code>.</li>"
        "<li><b>Distinct peaks merged into one red mark</b> — bandwidth too "
        "wide; lower it.</li>"
        "<li><b>Red marks on flat baseline regions</b> — prominence threshold "
        "too low; raise <code>min_prominence_quantile</code>.</li>"
        "</ul>"
    ),
    "line_with_band:mz_vs_tolerance": (
        "<b>Empirical tolerance vs. m/z</b>"
        "<p>Blue line: per-m/z (1−α/2)-quantile of |ppm error| from the "
        "reference ions, monotonized via pool-adjacent-violators. Shaded band: "
        "95% bootstrap CI envelope.</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Smoothly increasing curve with a tight band</b> — tolerance "
        "fit is stable; trust the curve as the per-m/z window.</li>"
        "<li><b>Wide CI band at the m/z extremes</b> — you have few reference "
        "ions in those regions; lower <code>coarse_tol_ppm</code> or "
        "<code>min_prevalence</code> on the reference card.</li>"
        "<li><b>Step-function shape</b> — PAV pooled many bins; not an error, "
        "just means the underlying data was non-monotone before pooling.</li>"
        "<li><b>Constant flat line</b> — fallback fired (too few pairs). "
        "Re-check reference detection.</li>"
        "</ul>"
    ),
    "histogram:per_pixel_factor": (
        "<b>Distribution of per-pixel normalization factors</b>"
        "<p>Each pixel's intensities were divided by this scale factor "
        "(median or TIC, depending on the chosen normalizer).</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Roughly unimodal, narrow distribution</b> — pixels are "
        "comparable; the normalizer is doing meaningful work.</li>"
        "<li><b>Bimodal / multi-modal</b> — likely two tissue regions with "
        "different ionization. The normalizer is partially correcting; you "
        "may want to draw ROIs and use <code>reference_ion_normalize</code> "
        "for a more sample-aware factor.</li>"
        "<li><b>A long tail toward zero</b> — many pixels with little signal. "
        "Hot-pixel filter or background subtract may help.</li>"
        "<li><b>One spike at the eps floor</b> — pixels with no signal at "
        "all. Check upstream peak picking thresholds.</li>"
        "</ul>"
    ),
    "histogram:per_pixel_kept_count": (
        "<b>Peaks kept per pixel</b>"
        "<p>How many peaks survived the picker per pixel. The picker's job is "
        "to get this number stable across pixels at a meaningful peak count.</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Tight, central distribution</b> — all pixels have a similar "
        "peak count; the threshold is well-tuned.</li>"
        "<li><b>Median below ~10</b> — too aggressive; lower "
        "<code>snr_mad</code> (SNR picker) or <code>min_snr</code> (CWT) to "
        "keep more peaks.</li>"
        "<li><b>Wide left tail (some pixels with very few peaks)</b> — those "
        "pixels have no signal; check the dataset for empty regions or use "
        "the hot-pixel filter on the bright outliers that may be skewing the "
        "noise floor.</li>"
        "<li><b>Median above 200-300</b> — likely admitting noise; raise the "
        "picker threshold.</li>"
        "</ul>"
    ),
    "histogram:morans_i_with_threshold": (
        "<b>Distribution of observed Moran's I across consensus channels</b>"
        "<p>Each consensus channel's spatial autocorrelation. Positive values "
        "mean clustered/structured intensity; near zero means noise.</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Clear right-skewed bulk in [0.1, 0.6]</b> — most channels are "
        "spatially structured (typical for tissue).</li>"
        "<li><b>Bimodal — one mode near 0, another in [0.2, 0.5]</b> — the "
        "Moran's I filter should cleanly separate noise from signal channels. "
        "If the FDR threshold isn't catching this, the permutation null may "
        "need more iterations.</li>"
        "<li><b>Distribution centered near 0</b> — no spatial structure (the "
        "sample isn't tissue, or the consensus channels are mostly noise). "
        "Disable the spatial filter on non-tissue samples.</li>"
        "</ul>"
    ),
    "scatter:reference_mz_vs_prevalence": (
        "<b>Reference ions: m/z vs. pixel prevalence</b>"
        "<p>Each point is one reference ion. X = its m/z; Y = the fraction of "
        "pixels in which it appears.</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Most points clustered near prevalence ≈ 1.0</b> — anchors "
        "are robust across the image; the tolerance fit will be tight.</li>"
        "<li><b>A few scattered low-prevalence points</b> — outlier anchors. "
        "Raise <code>min_prevalence</code> on the reference card to drop "
        "them.</li>"
        "<li><b>Wide spread in m/z</b> — anchors span the working range, "
        "good. Compressed to one m/z region — the tolerance fit at far m/z "
        "will be extrapolated.</li>"
        "<li><b>Few points (&lt; 5)</b> — too thin to fit a curve. Lower "
        "<code>min_prevalence</code> or widen <code>coarse_tol_ppm</code>.</li>"
        "</ul>"
    ),
    "scatter:pre_post_ppm_residual": (
        "<b>|ppm residual| pre vs. post recalibration</b>"
        "<p>Two histograms: gray = anchor residuals before the recalibration "
        "warp was applied; blue = after. The blue should sit to the left of "
        "the gray.</p>"
        "<p><b>Look for:</b></p>"
        "<ul>"
        "<li><b>Blue histogram centered at lower |ppm| than gray</b> — "
        "recalibration tightened anchor residuals (the desired behavior).</li>"
        "<li><b>Blue and gray overlap or blue is wider</b> — the warp added "
        "noise. Check the operator's <code>improvement_ppm</code> diagnostic; "
        "consider disabling recalibration on this dataset.</li>"
        "<li><b>Tail of large pre-residuals</b> with the corresponding tail "
        "tightened — RANSAC successfully rejected outlier anchors.</li>"
        "</ul>"
    ),
}


def _build_diagnostic_plot(node_id: str, diag) -> "QWidget | None":  # noqa: ANN001 — Diagnostic
    """Build a small pyqtgraph plot for one operator's diagnostic, or None if the
    operator's figure_hint isn't one we know how to render. Plots are intended to
    be ~120 px tall so several can stack in the RunPage's scroll area.

    Each rendered plot is also wrapped in a tooltip describing what the plot
    shows and which features to look for as markers of stable / unstable
    results — see ``_PLOT_TOOLTIPS``.
    """
    import pyqtgraph as pg
    from qtpy.QtWidgets import QLabel as _QLabel
    from qtpy.QtWidgets import QVBoxLayout as _QVBoxLayout
    from qtpy.QtWidgets import QWidget as _QWidget

    hint = getattr(diag, "figure_hint", None) or ""
    payload = getattr(diag, "payload", None) or {}
    summary = getattr(diag, "summary", None) or {}
    container = _QWidget()
    cl = _QVBoxLayout(container)
    cl.setContentsMargins(0, 0, 0, 0)
    cl.setSpacing(2)
    title_text = _OP_DISPLAY_NAMES.get(getattr(diag, "name", node_id), node_id)
    title_label = _QLabel(f"<b>{title_text}</b>  <span style='color:#888'>(node {node_id})</span>")
    # Apply per-figure-hint tooltip on the title and (below) the plot itself.
    tooltip = _PLOT_TOOLTIPS.get(hint, "")
    if tooltip:
        title_label.setToolTip(tooltip)
    cl.addWidget(title_label)

    plot = pg.PlotWidget()
    plot.setBackground("w")
    plot.setMinimumHeight(120)
    plot.setMaximumHeight(180)
    plot.showGrid(x=True, y=True, alpha=0.25)

    rendered = False
    if hint == "line:kde_density_with_consensus_marks":
        grid = payload.get("kde_grid_log_mz")
        density = payload.get("kde_density")
        consensus_mz = payload.get("consensus_mz")
        if grid is not None and density is not None:
            mz_grid = np.exp(np.asarray(grid))
            plot.plot(mz_grid, np.asarray(density), pen=pg.mkPen("#1f77b4", width=1.2))
            plot.setLabel("bottom", "m/z")
            plot.setLabel("left", "KDE density")
            if consensus_mz is not None:
                for c in np.asarray(consensus_mz):
                    plot.addItem(pg.InfiniteLine(pos=float(c), angle=90, pen=pg.mkPen("#d62728", width=0.8)))
            rendered = True
    elif hint == "line_with_band:mz_vs_tolerance":
        mz_grid = payload.get("mz_grid")
        ppm_q = payload.get("ppm_quantile")
        ci_lo = payload.get("ci_low")
        ci_hi = payload.get("ci_high")
        if mz_grid is not None and ppm_q is not None:
            mz_grid = np.asarray(mz_grid)
            ppm_q = np.asarray(ppm_q)
            if ci_lo is not None and ci_hi is not None:
                lo_curve = pg.PlotDataItem(mz_grid, np.asarray(ci_lo))
                hi_curve = pg.PlotDataItem(mz_grid, np.asarray(ci_hi))
                fill = pg.FillBetweenItem(lo_curve, hi_curve, brush=(31, 119, 180, 60))
                plot.addItem(fill)
            plot.plot(mz_grid, ppm_q, pen=pg.mkPen("#1f77b4", width=1.5))
            plot.setLabel("bottom", "m/z")
            plot.setLabel("left", "tolerance (ppm)")
            rendered = True
    elif hint == "histogram:per_pixel_factor":
        f = payload.get("per_pixel_factor")
        if f is not None and len(f):
            arr = np.asarray(f)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                bins = np.histogram_bin_edges(arr, bins=min(50, max(8, int(np.sqrt(arr.size)))))
                hist, _ = np.histogram(arr, bins=bins)
                plot.plot(
                    bins,
                    np.r_[hist, hist[-1]],
                    stepMode="right",
                    pen=pg.mkPen("#2ca02c", width=1.2),
                )
                plot.setLabel("bottom", "per-pixel factor")
                plot.setLabel("left", "pixel count")
                rendered = True
    elif hint == "histogram:per_pixel_kept_count":
        c = payload.get("per_pixel_kept_count")
        if c is not None and len(c):
            arr = np.asarray(c)
            bins = np.arange(int(arr.min()), int(arr.max()) + 2)
            hist, _ = np.histogram(arr, bins=bins)
            plot.plot(
                bins,
                np.r_[hist, hist[-1]],
                stepMode="right",
                pen=pg.mkPen("#9467bd", width=1.2),
            )
            plot.setLabel("bottom", "kept peaks per pixel")
            plot.setLabel("left", "pixel count")
            rendered = True
    elif hint == "histogram:morans_i_with_threshold":
        I = payload.get("I_obs")
        if I is not None and len(I):
            arr = np.asarray(I)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                bins = np.histogram_bin_edges(arr, bins=min(50, max(8, int(np.sqrt(arr.size)))))
                hist, _ = np.histogram(arr, bins=bins)
                plot.plot(
                    bins,
                    np.r_[hist, hist[-1]],
                    stepMode="right",
                    pen=pg.mkPen("#ff7f0e", width=1.2),
                )
                plot.setLabel("bottom", "Moran's I")
                plot.setLabel("left", "channel count")
                rendered = True
    elif hint == "scatter:reference_mz_vs_prevalence":
        mz = payload.get("reference_mz")
        prev = payload.get("prevalence")
        if mz is not None and prev is not None and len(mz):
            plot.plot(
                np.asarray(mz),
                np.asarray(prev),
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush="#17becf",
            )
            plot.setLabel("bottom", "reference m/z")
            plot.setLabel("left", "prevalence")
            rendered = True
    elif hint == "scatter:pre_post_ppm_residual":
        pre = payload.get("pre_ppm_residual")
        post = payload.get("post_ppm_residual")
        if pre is not None and post is not None:
            pre_a = np.asarray(pre)
            post_a = np.asarray(post)
            valid = np.isfinite(pre_a) & np.isfinite(post_a)
            pre_v = pre_a[valid].ravel()
            post_v = post_a[valid].ravel()
            if pre_v.size:
                bins = np.linspace(0, max(float(pre_v.max()), float(post_v.max() if post_v.size else 1.0)) + 1e-9, 40)
                hist_pre, _ = np.histogram(pre_v, bins=bins)
                hist_post, _ = np.histogram(post_v if post_v.size else np.array([0.0]), bins=bins)
                plot.plot(bins, np.r_[hist_pre, hist_pre[-1]], stepMode="right", pen=pg.mkPen("#888", width=1.0), name="pre")
                plot.plot(bins, np.r_[hist_post, hist_post[-1]], stepMode="right", pen=pg.mkPen("#1f77b4", width=1.5), name="post")
                plot.setLabel("bottom", "|ppm residual|")
                plot.setLabel("left", "anchor count")
                rendered = True

    if not rendered:
        # No plot for this operator — show a compact summary line instead.
        summary_text = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(summary.items())[:4])
        cl.addWidget(_QLabel(summary_text or "(no diagnostic data)"))
        plot.setParent(None)
        return container
    # Apply the per-figure-hint tooltip on the plot widget too — pyqtgraph
    # PlotWidget shows tooltips natively over its viewport.
    if tooltip:
        plot.setToolTip(tooltip)
        # Also set on the container so hovering the title row works the same.
        container.setToolTip(tooltip)
    cl.addWidget(plot)
    return container


def _summarize_rejection_budget(op_name: str, summary: dict) -> list[str]:
    """Pull a short, plain-English rejection-budget paragraph out of an operator's
    diagnostic summary. Returns a list of bullet-point lines.

    The point isn't a faithful re-render of every key — it's giving the user a one-
    glance answer to "which of this card's parameters is dropping the most peaks
    right now?" so they know which knob to nudge before re-running.
    """
    s = summary or {}
    out: list[str] = []
    if op_name == "detect_reference_ions":
        n = int(s.get("n_reference_ions", 0))
        out.append(f"Found {n} reference ions.")
        prev_min = s.get("prevalence_min")
        if prev_min is not None and n > 0:
            out.append(f"Min prevalence kept: {prev_min:.0%}.")
        if n < 5:
            out.append("⚠ Lower min_prevalence or coarse_tol_ppm to surface more.")
    elif op_name == "empirical_tolerance_from_reference_ions":
        n_obs = int(s.get("n_reference_observations", 0))
        ppm_med = s.get("ppm_median")
        if ppm_med is not None:
            out.append(f"Median tolerance: {ppm_med:.1f} ppm (from {n_obs} observations).")
        ci = s.get("ci_band_median_width")
        if ci is not None:
            out.append(f"95% CI band: ±{ci/2:.1f} ppm.")
    elif op_name in {"median_normalize", "tic_normalize"}:
        med = s.get("factor_median")
        if med is not None:
            out.append(f"Per-pixel scale factor: median={med:.3g}.")
    elif op_name == "snr_peak_pick":
        n_in = int(s.get("n_peaks_in", 0))
        n_out = int(s.get("n_peaks_out", 0))
        kept = s.get("fraction_kept")
        if n_in > 0 and kept is not None:
            out.append(
                f"Kept {n_out:,} of {n_in:,} peaks ({kept:.0%}); rejected "
                f"{n_in - n_out:,}."
            )
        if (kept or 0) < 0.1:
            out.append("⚠ Lower snr_mad to keep more peaks.")
    elif op_name == "kde_consensus_alignment":
        n_total = int(s.get("n_local_maxima_total", 0))
        n_post_prom = int(s.get("n_post_prominence", 0))
        n_rej_prom = int(s.get("n_rejected_by_prominence", 0))
        n_rej_prev = int(s.get("n_rejected_by_prevalence", 0))
        n_kept = int(s.get("n_consensus_peaks", 0))
        if n_total > 0:
            out.append(
                f"Found {n_kept} consensus peaks from {n_total} local maxima."
            )
            # Identify the dominant rejection cause so the user knows which knob to
            # tune. Pick the larger of the two rejection contributors.
            if n_rej_prom >= n_rej_prev and n_rej_prom > 0:
                out.append(
                    f"Largest contributor: prominence threshold rejected "
                    f"{n_rej_prom:,} candidates."
                )
                out.append(
                    "→ Lower min_prominence_quantile to keep more candidates."
                )
            elif n_rej_prev > 0:
                out.append(
                    f"Largest contributor: prevalence filter rejected "
                    f"{n_rej_prev:,} of {n_post_prom:,} candidates."
                )
                out.append(
                    "→ Lower min_prevalence to keep less-widespread peaks."
                )
    elif op_name == "hot_pixel_filter":
        n = int(s.get("n_hot_pixels", 0))
        if n > 0:
            out.append(f"Flagged {n} hot pixel(s) (TIC > median + k_mad·MAD).")
    elif op_name == "background_subtract":
        n_in = int(s.get("n_channels_in", 0))
        n_out = int(s.get("n_channels_out", 0))
        if n_in > 0:
            out.append(
                f"Kept {n_out} of {n_in} channels; rejected {n_in - n_out} "
                f"as background-dominant."
            )
    return out


class _NodeCard(QWidget):
    """Editable card for one Node — exposes each OpParams field as a Qt input.

    Labels and tooltips come from ``field(metadata={'label': ..., 'help': ...})`` on
    each OpParams subclass (see ``ops/base.py``). A "Reset" button per card restores
    just that node's parameters.

    Optional cards (``optional=True``) carry an "Enable" checkbox in the title row.
    When unchecked the card's param widgets are visually greyed and the WorkflowPage
    skips the node when building the Pipeline. State is queried via
    ``is_enabled()``; ``optional=False`` cards (the recommended-pipeline nodes)
    always report enabled.
    """

    def __init__(
        self,
        node: Node,
        *,
        warnings: list[str],
        last_diagnostics=None,  # noqa: ANN001 — list[Diagnostic] | None
        parent: QWidget | None = None,
        optional: bool = False,
        enabled: bool = True,
        optional_hint: str = "",
    ) -> None:
        super().__init__(parent)
        self._node = node
        self._defaults = node.params  # snapshot of the recommended defaults
        self._inputs: dict[str, QWidget] = {}
        self._optional = optional
        self._enable_checkbox: QCheckBox | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Title bar with a per-card reset button.
        title_row = QHBoxLayout()
        display_name = _OP_DISPLAY_NAMES.get(node.op_name, node.op_name.replace("_", " "))
        if optional:
            self._enable_checkbox = QCheckBox(f"Enable")
            self._enable_checkbox.setChecked(enabled)
            self._enable_checkbox.setToolTip(
                "Add this optional operator to the pipeline. " + (optional_hint or "")
            )
            self._enable_checkbox.toggled.connect(self._on_enable_toggled)
            title_row.addWidget(self._enable_checkbox)
        title = QLabel(f"<b>{display_name}</b>")
        title.setToolTip(f"node id: <code>{node.id}</code> · op: <code>{node.op_name}</code>")
        title_row.addWidget(title, stretch=1)
        reset = QPushButton("Reset")
        reset.setToolTip("Restore this node's parameters to the recommended defaults.")
        reset.clicked.connect(self._reset)
        title_row.addWidget(reset)
        outer.addLayout(title_row)

        # Optional cards get a one-liner explainer below the title so the user
        # knows when to enable them.
        if optional and optional_hint:
            hint_lbl = QLabel(optional_hint)
            hint_lbl.setWordWrap(True)
            hint_lbl.setStyleSheet("color: #888; font-style: italic;")
            outer.addWidget(hint_lbl)

        if warnings:
            for w in warnings:
                lbl = QLabel(f"⚠ {w}")
                lbl.setStyleSheet("color: #b58a00;")
                lbl.setWordWrap(True)
                outer.addWidget(lbl)

        # Rejection-budget guidance from the previous run, if any. This is the
        # in-place "what's this knob actually doing right now" answer that turns a
        # parameter-tweak loop from guesswork into informed adjustment.
        guidance_lines: list[str] = []
        if last_diagnostics:
            for d in last_diagnostics:
                guidance_lines.extend(
                    _summarize_rejection_budget(node.op_name, getattr(d, "summary", {}) or {})
                )
        if guidance_lines:
            box = QLabel("<br>".join(f"• {line}" for line in guidance_lines))
            box.setWordWrap(True)
            box.setStyleSheet(
                "background-color: #1e1e1e; "
                "color: #d4d4d4; "
                "padding: 6px; "
                "border-left: 3px solid #4ec9b0; "
                "font-size: 9pt;"
            )
            box.setToolTip(
                "Rejection-budget summary from the most recent run. Use this to "
                "decide which parameter to tune before clicking Run again."
            )
            outer.addWidget(box)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._form_widget = QWidget()
        self._form_widget.setLayout(form)
        outer.addWidget(self._form_widget)
        if not is_dataclass(node.params):
            return
        for f in fields(node.params):
            val = getattr(node.params, f.name)
            widget = self._make_input(val)
            help_text = field_help(f) or f.name
            label_text = field_label(f)
            label_widget = QLabel(label_text)
            label_widget.setToolTip(help_text)
            widget.setToolTip(help_text)
            self._inputs[f.name] = widget
            form.addRow(label_widget, widget)
        # Apply initial enable state for optional cards.
        if self._optional and self._enable_checkbox is not None:
            self._on_enable_toggled(self._enable_checkbox.isChecked())

    # ---- Optional-card enable/disable plumbing ----------------------------------

    def is_enabled(self) -> bool:
        """Return True for required cards, or the user's toggle for optional ones."""
        if self._enable_checkbox is None:
            return True
        return bool(self._enable_checkbox.isChecked())

    def is_optional(self) -> bool:
        return self._optional

    def _on_enable_toggled(self, enabled: bool) -> None:
        """Grey out the form when the user disables an optional card."""
        if hasattr(self, "_form_widget") and self._form_widget is not None:
            self._form_widget.setEnabled(bool(enabled))

    def _make_input(self, val: Any) -> QWidget:
        if isinstance(val, bool):
            cb = QCheckBox()
            cb.setChecked(val)
            return cb
        if isinstance(val, int) and not isinstance(val, bool):
            sb = QSpinBox()
            sb.setRange(-(1 << 30), 1 << 30)
            sb.setValue(int(val))
            return sb
        if isinstance(val, float):
            db = QDoubleSpinBox()
            db.setDecimals(6)
            db.setRange(-1e12, 1e12)
            db.setValue(float(val))
            return db
        le = QLineEdit(str(val))
        return le

    def _reset(self) -> None:
        if not is_dataclass(self._defaults):
            return
        for f in fields(self._defaults):
            val = getattr(self._defaults, f.name)
            widget = self._inputs.get(f.name)
            if widget is None:
                continue
            widget.blockSignals(True)
            try:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(val))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(val))
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(val))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(val))
            finally:
                widget.blockSignals(False)

    def to_node(self) -> Node:
        if not is_dataclass(self._node.params):
            return self._node
        kwargs: dict[str, Any] = {}
        for f in fields(self._node.params):
            widget = self._inputs.get(f.name)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                kwargs[f.name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                kwargs[f.name] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                kwargs[f.name] = float(widget.value())
            elif isinstance(widget, QLineEdit):
                kwargs[f.name] = widget.text()
        new_params = dataclass_replace(self._node.params, **kwargs)
        return Node(
            id=self._node.id,
            op_name=self._node.op_name,
            params=new_params,
            upstream=self._node.upstream,
        )


# ---------- The wizard itself ------------------------------------------------------


class WizardWidget(QWizard):
    """The 7-page MSI processing wizard.

    Uses the module-level `default_session()` so widgets opened outside the wizard
    (Hyperspectral Browser, Spectrum Panel, Preview) automatically reflect the dataset
    the wizard is working with.
    """

    def __init__(
        self,
        napari_viewer: "napari.Viewer | None" = None,
        parent: QWidget | None = None,
        session: MsiSession | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = napari_viewer
        # Share state with the browser + spectrum panel by default. Tests can pass an
        # isolated session.
        self._session = session if session is not None else default_session()
        self._proposed_pipeline: Pipeline | None = None
        # Snapshot of the dataset that should feed the pipeline's first node. Set when
        # LoadPage finishes, and updated when ExperimentParams or ROIs change. The
        # pipeline runs on this snapshot, so re-running with tweaked params doesn't
        # try to feed a post-consensus PeakMatrix back through reference detection.
        self._input_dataset_snapshot = None  # type: ignore[assignment]
        self._input_hash_at_workflow: str | None = None
        # Most recent RunResult, populated by RunPage on completion. The
        # WorkflowPage uses this to show rejection-budget guidance per card so the
        # user can see which parameters dropped the most candidates last time.
        self._last_run_result = None  # type: ignore[assignment]

        # When the session's ROIs change (RoiWidget edits), the dataset gets a new
        # `rois` tuple but its identity is unchanged. Capture the rois-updated dataset
        # as the new pipeline input.
        self._session.rois_changed.connect(self._on_rois_changed)

        self.setWindowTitle("DAPPLE Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        # Allow the dock widget to be shrunk horizontally and vertically. Without this,
        # QWizard's internal layout pins the wizard at its preferred size and napari
        # can't resize the dock smaller than that.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(360)
        self.setMinimumHeight(420)

        self.addPage(LoadPage(self))
        self.addPage(ParamsPage(self))
        self.addPage(PreviewPage(self))
        self.addPage(RoiPage(self))
        self.addPage(WorkflowPage(self))
        self.addPage(ReviewPage(self))
        self.addPage(RunPage(self))

    @property
    def viewer(self) -> "napari.Viewer | None":
        return self._viewer

    @property
    def session(self) -> MsiSession:
        return self._session

    def set_input_snapshot(self, ds) -> None:  # noqa: ANN001 — MSIDataset
        """Replace the pipeline-input snapshot with a freshly loaded dataset.

        Refuses datasets whose backend is anything other than ``PeakList`` so that
        a post-consensus ``PeakMatrix`` (returned by an earlier run) can never
        accidentally become the input for the next re-run — that would feed a
        dense aligned matrix into reference detection, which expects raw peak
        lists.
        """
        from dapple.data.dataset import PeakList

        if ds is not None and not isinstance(ds.backend, PeakList):
            return
        self._input_dataset_snapshot = ds
        self._input_hash_at_workflow = ds.hash() if ds is not None else None

    def update_input_metadata(self, metadata) -> None:  # noqa: ANN001
        """Apply edited ExperimentParams to the snapshot without changing its
        backend. Called from ParamsPage when the user edits a field.

        We deliberately do *not* re-snapshot ``session.dataset`` here, because that
        dataset might be a post-consensus PeakMatrix from an earlier run.
        """
        from dataclasses import replace as drep

        if self._input_dataset_snapshot is None:
            return
        self._input_dataset_snapshot = drep(
            self._input_dataset_snapshot, metadata=metadata
        )
        self._input_hash_at_workflow = self._input_dataset_snapshot.hash()

    def update_input_rois(self, rois) -> None:  # noqa: ANN001
        """Attach updated ROIs to the snapshot. Called from the ROI sync chain so
        background-aware operators see the latest polygons on re-run."""
        if self._input_dataset_snapshot is None:
            return
        self._input_dataset_snapshot = self._input_dataset_snapshot.with_rois(
            tuple(rois)
        )
        self._input_hash_at_workflow = self._input_dataset_snapshot.hash()

    def _on_rois_changed(self, rois) -> None:  # noqa: ANN001 — tuple[RoiDef, ...]
        """Forward the session's ROIs onto the input snapshot. Never re-snapshots
        ``session.dataset`` directly because after a run that's a PeakMatrix."""
        self.update_input_rois(rois)


def _plugin_version() -> str:
    try:
        from dapple import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"
