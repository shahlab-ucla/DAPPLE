"""Session mediator: a single source of truth tying widgets to the loaded MSIDataset.

Widgets subscribe to `MsiSession.dataset_changed` rather than holding their own dataset
references. The session also exposes "selected pixel" and "selected ROI" so the
SpectrumPanel can update on user interaction without each widget knowing about the
others.

We use `psygnal` (already a napari dep) for typed signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psygnal import Signal

from dapple.data.dataset import PeakList

if TYPE_CHECKING:
    from dapple.data.dataset import MSIDataset
    from dapple.data.metadata import RoiDef


class MsiSession:
    """Per-viewer container of MSI state; widgets subscribe to its signals."""

    dataset_changed = Signal(object)  # MSIDataset | None
    selected_pixel_changed = Signal(object)  # tuple[int, int] | None  (x, y) 1-indexed
    selected_roi_changed = Signal(object)  # str | None
    rois_changed = Signal(tuple)
    # Cross-widget request: a SpectrumPanel click asks the ChannelsPanel to show the
    # consensus channel nearest the clicked m/z. Other widgets may listen too
    # (e.g. a future "annotate this peak" dialog).
    show_mz_requested = Signal(float)

    def __init__(self) -> None:
        self._dataset: "MSIDataset | None" = None
        self._raw_dataset: "MSIDataset | None" = None
        self._selected_pixel: tuple[int, int] | None = None
        self._selected_roi: str | None = None
        self._rois: tuple["RoiDef", ...] = ()

    # --- dataset ----------------------------------------------------------------
    @property
    def dataset(self) -> "MSIDataset | None":
        return self._dataset

    @property
    def raw_dataset(self) -> "MSIDataset | None":
        """Original peak-list dataset retained for before/after inspection."""
        return self._raw_dataset

    def set_dataset(self, ds: "MSIDataset | None") -> None:
        previous_raw = self._raw_dataset
        if ds is None:
            self._raw_dataset = None
        elif isinstance(ds.backend, PeakList):
            self._raw_dataset = ds
        elif previous_raw is not None and previous_raw.identity != ds.identity:
            # Opening an unrelated processed file must not retain a stale raw trace.
            self._raw_dataset = None
        self._dataset = ds
        if ds is not None:
            self._rois = ds.rois
        self.dataset_changed.emit(ds)

    # --- pixel selection --------------------------------------------------------
    @property
    def selected_pixel(self) -> tuple[int, int] | None:
        return self._selected_pixel

    def set_selected_pixel(self, xy: tuple[int, int] | None) -> None:
        if xy != self._selected_pixel:
            self._selected_pixel = xy
            self.selected_pixel_changed.emit(xy)

    # --- ROI selection ----------------------------------------------------------
    @property
    def selected_roi(self) -> str | None:
        return self._selected_roi

    def set_selected_roi(self, name: str | None) -> None:
        if name != self._selected_roi:
            self._selected_roi = name
            self.selected_roi_changed.emit(name)

    # --- ROI list ---------------------------------------------------------------
    @property
    def rois(self) -> tuple["RoiDef", ...]:
        return self._rois

    def set_rois(self, rois: tuple["RoiDef", ...]) -> None:
        self._rois = rois
        if self._dataset is not None:
            self._dataset = self._dataset.with_rois(rois)
        if self._raw_dataset is not None:
            self._raw_dataset = self._raw_dataset.with_rois(rois)
        self.rois_changed.emit(rois)

    # --- show-m/z request -------------------------------------------------------
    def request_show_mz(self, mz: float) -> None:
        """Ask listeners to surface the channel nearest to this m/z. Typically called
        by SpectrumPanel when the user clicks on a peak."""
        self.show_mz_requested.emit(float(mz))


# Module-global default session for the convenience case where widgets are opened
# one-at-a-time from the napari menu without an explicit wizard. The wizard installs
# its own session.
_DEFAULT_SESSION: MsiSession | None = None


def default_session() -> MsiSession:
    global _DEFAULT_SESSION
    if _DEFAULT_SESSION is None:
        _DEFAULT_SESSION = MsiSession()
    return _DEFAULT_SESSION


def adopt_dataset_from_viewer(session: MsiSession, viewer: object | None) -> bool:
    """Adopt an MSI dataset already attached to a napari layer.

    Reader contributions put the originating :class:`MSIDataset` in layer metadata.
    Standalone widgets used to ignore that state, so opening an imzML file directly
    and then opening Channels/Spectrum still showed "no dataset loaded".  Prefer the
    active layer, then scan newest-to-oldest, and never replace an explicit session.
    """
    if session.dataset is not None or viewer is None:
        return session.dataset is not None
    layers = getattr(viewer, "layers", None)
    if layers is None:
        return False
    candidates: list[object] = []
    try:
        active = layers.selection.active
        if active is not None:
            candidates.append(active)
    except Exception:  # noqa: BLE001 — napari API varies across minor versions
        pass
    try:
        candidates.extend(layer for layer in reversed(list(layers)) if layer not in candidates)
    except Exception:  # noqa: BLE001
        return False
    for layer in candidates:
        metadata = getattr(layer, "metadata", None) or {}
        ds = metadata.get("msi_dataset") if isinstance(metadata, dict) else None
        if ds is not None:
            session.set_dataset(ds)
            return True
    return False
