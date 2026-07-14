"""Canonical ROI masks and directed straight-axis geometry.

Napari shapes live in zero-based image data coordinates ``(y, x)`` while
``MSIDataset.coords`` are stored as ``(x, y)`` and may be one-based.  This module
uses the dataset's established coordinate-index helper once, then performs all
spatial operations in the zero-based raster coordinate system.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from skimage.draw import polygon as sk_polygon

from dapple.data.coords import coords_to_grid_index
from dapple.data.dataset import MSIDataset
from dapple.data.metadata import RoiDef

OverlapPolicy = Literal["error", "exclude", "first", "allow"]


@dataclass(frozen=True, eq=False)
class RoiMasks:
    """ROI membership over populated MSI pixels.

    ``masks`` has shape ``(n_rois, ds.n_pixels)``. ``overlap_mask`` records pixels
    covered by multiple *original* polygons even when the chosen policy subsequently
    removes or resolves those memberships.
    """

    names: tuple[str, ...]
    masks: np.ndarray
    overlap_mask: np.ndarray
    overlap_policy: OverlapPolicy
    coordinate_fingerprint: str

    def __post_init__(self) -> None:
        masks = np.asarray(self.masks)
        overlap = np.asarray(self.overlap_mask)
        if masks.ndim != 2 or masks.shape[0] != len(self.names):
            raise ValueError(
                f"masks must be (n_rois, n_pixels), got {masks.shape} for "
                f"{len(self.names)} names"
            )
        if overlap.shape != (masks.shape[1],):
            raise ValueError(
                f"overlap_mask shape {overlap.shape} != ({masks.shape[1]},)"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError("ROI names must be unique")
        if not self.coordinate_fingerprint:
            raise ValueError("coordinate_fingerprint is required")

    @property
    def n_pixels(self) -> int:
        return int(self.masks.shape[1])

    @property
    def pixel_counts(self) -> np.ndarray:
        return np.asarray(self.masks, dtype=bool).sum(axis=1).astype(np.int64)

    @property
    def fingerprint(self) -> str:
        """Stable hash of analysis-visible names, policy, and populated-pixel masks."""
        digest = hashlib.sha256()
        digest.update(self.overlap_policy.encode("utf-8"))
        digest.update(self.coordinate_fingerprint.encode("ascii"))
        for name in self.names:
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        digest.update(np.asarray(self.masks.shape, dtype="<i8").tobytes())
        digest.update(np.packbits(np.asarray(self.masks, dtype=bool), axis=None).tobytes())
        return digest.hexdigest()

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown ROI {name!r}; available: {self.names}") from exc

    def union(self, names: str | Sequence[str]) -> np.ndarray:
        """Return a populated-pixel mask for one ROI or a union of named ROIs."""
        selected = (names,) if isinstance(names, str) else tuple(names)
        if not selected:
            raise ValueError("at least one ROI name is required")
        idx = [self.index(name) for name in selected]
        return np.any(np.asarray(self.masks, dtype=bool)[idx, :], axis=0)


@dataclass(frozen=True)
class DirectedAxis:
    """A finite, directed straight anatomical axis in napari ``(y, x)`` pixels."""

    name: str
    start_yx: tuple[float, float]
    end_yx: tuple[float, float]
    start_label: str = "start"
    end_label: str = "end"
    half_width_px: float | None = None

    def __post_init__(self) -> None:
        start = np.asarray(self.start_yx, dtype=np.float64)
        end = np.asarray(self.end_yx, dtype=np.float64)
        if start.shape != (2,) or end.shape != (2,):
            raise ValueError("axis start_yx and end_yx must each have two coordinates")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError("axis coordinates must be finite")
        if np.allclose(start, end, rtol=0.0, atol=1e-12):
            raise ValueError("axis start and end must be distinct")
        if self.half_width_px is not None:
            if not np.isfinite(self.half_width_px) or self.half_width_px < 0:
                raise ValueError("axis half_width_px must be finite and non-negative")

    @property
    def length_px(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.end_yx, dtype=np.float64)
                - np.asarray(self.start_yx, dtype=np.float64)
            )
        )

    def reversed(self, *, name: str | None = None) -> "DirectedAxis":
        """Return the same segment with direction and endpoint labels swapped."""
        return DirectedAxis(
            name=name or self.name,
            start_yx=self.end_yx,
            end_yx=self.start_yx,
            start_label=self.end_label,
            end_label=self.start_label,
            half_width_px=self.half_width_px,
        )


@dataclass(frozen=True, eq=False)
class AxisProjection:
    """Normalized axis coordinate and signed distance for populated MSI pixels."""

    axis: DirectedAxis
    t: np.ndarray
    signed_distance_px: np.ndarray
    included_mask: np.ndarray
    coordinate_fingerprint: str

    def __post_init__(self) -> None:
        n = int(np.asarray(self.t).size)
        for name in ("t", "signed_distance_px", "included_mask"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n,):
                raise ValueError(f"{name} shape {arr.shape} != ({n},)")
        if not self.coordinate_fingerprint:
            raise ValueError("coordinate_fingerprint is required")
        included = np.asarray(self.included_mask, dtype=bool)
        t = np.asarray(self.t, dtype=np.float64)
        signed = np.asarray(self.signed_distance_px, dtype=np.float64)
        if np.any(~np.isfinite(t[included])) or np.any(~np.isfinite(signed[included])):
            raise ValueError("included axis pixels require finite t and signed distance")
        if np.any((t[included] < 0.0) | (t[included] > 1.0)):
            raise ValueError("included axis t values must lie in [0, 1]")
        if np.any(np.isfinite(t[~included])) or np.any(np.isfinite(signed[~included])):
            raise ValueError("excluded axis pixels must have NaN coordinates")

    @property
    def fingerprint(self) -> str:
        """Stable hash of directed geometry and the selected populated pixels."""
        payload = {
            "name": self.axis.name,
            "start_yx": list(self.axis.start_yx),
            "end_yx": list(self.axis.end_yx),
            "start_label": self.axis.start_label,
            "end_label": self.axis.end_label,
            "half_width_px": self.axis.half_width_px,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        included = np.asarray(self.included_mask, dtype=bool)
        digest.update(self.coordinate_fingerprint.encode("ascii"))
        digest.update(np.packbits(included, axis=None).tobytes())
        digest.update(np.asarray(self.t[included], dtype="<f8").tobytes())
        return digest.hexdigest()


def coordinate_fingerprint(ds: MSIDataset) -> str:
    """Hash grid geometry plus populated-pixel row order for mask binding."""
    return ds.coordinate_fingerprint


def rasterize_rois(
    ds: MSIDataset,
    rois: Sequence[RoiDef] | None = None,
    *,
    overlap_policy: OverlapPolicy = "error",
) -> RoiMasks:
    """Rasterize polygons onto populated pixels with explicit overlap semantics.

    Parameters
    ----------
    ds:
        Dataset whose raster and populated coordinates define the target pixels.
    rois:
        Polygons in napari zero-based ``(y, x)`` coordinates. Defaults to ``ds.rois``.
    overlap_policy:
        ``"error"`` rejects any multiply-covered populated pixel; ``"exclude"``
        removes overlaps from every ROI; ``"first"`` assigns each pixel to the
        first polygon that covers it; and ``"allow"`` preserves all memberships.
    """
    if overlap_policy not in {"error", "exclude", "first", "allow"}:
        raise ValueError(f"unknown overlap_policy {overlap_policy!r}")
    selected = tuple(ds.rois if rois is None else rois)
    if not selected:
        raise ValueError("at least one ROI is required")
    names = tuple(r.name for r in selected)
    if len(set(names)) != len(names):
        raise ValueError(f"ROI names must be unique, got {names}")

    h, w = ds.grid_shape
    flat_idx = coords_to_grid_index(ds.coords, ds.grid_shape)
    masks = np.zeros((len(selected), ds.n_pixels), dtype=bool)
    for i, roi in enumerate(selected):
        vertices = np.asarray(roi.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
            raise ValueError(f"ROI {roi.name!r} has invalid vertices shape {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError(f"ROI {roi.name!r} contains non-finite vertices")
        rr, cc = sk_polygon(vertices[:, 0], vertices[:, 1], shape=(h, w))
        grid_mask = np.zeros(h * w, dtype=bool)
        grid_mask[rr.astype(np.int64) * w + cc.astype(np.int64)] = True
        masks[i, :] = grid_mask[flat_idx]

    original_overlap = masks.sum(axis=0) > 1
    n_overlap = int(original_overlap.sum())
    if n_overlap and overlap_policy == "error":
        raise ValueError(
            f"{n_overlap} populated pixel(s) belong to multiple ROIs; choose "
            "overlap_policy='exclude', 'first', or 'allow' explicitly"
        )
    if overlap_policy == "exclude" and n_overlap:
        masks[:, original_overlap] = False
    elif overlap_policy == "first" and n_overlap:
        claimed = np.zeros(ds.n_pixels, dtype=bool)
        for i in range(masks.shape[0]):
            masks[i, :] &= ~claimed
            claimed |= masks[i, :]

    return RoiMasks(
        names=names,
        masks=masks,
        overlap_mask=original_overlap,
        overlap_policy=overlap_policy,
        coordinate_fingerprint=coordinate_fingerprint(ds),
    )


def project_to_axis(
    ds: MSIDataset,
    axis: DirectedAxis,
    *,
    include_mask: np.ndarray | None = None,
) -> AxisProjection:
    """Project populated pixels onto a finite directed segment.

    Included pixels have a normalized coordinate ``t=0`` at ``start_yx`` and
    ``t=1`` at ``end_yx``. Pixels beyond either endpoint, outside an optional
    half-width, or excluded by ``include_mask`` receive NaN coordinates. Signed
    distance is positive on the segment's left-hand side in conventional ``(x,y)``
    geometry and flips sign when the axis is reversed.
    """
    flat_idx = coords_to_grid_index(ds.coords, ds.grid_shape)
    _, w = ds.grid_shape
    y = (flat_idx // w).astype(np.float64)
    x = (flat_idx % w).astype(np.float64)
    points = np.column_stack([y, x])

    start = np.asarray(axis.start_yx, dtype=np.float64)
    end = np.asarray(axis.end_yx, dtype=np.float64)
    vector = end - start
    length_sq = float(vector @ vector)
    length = float(np.sqrt(length_sq))
    relative = points - start[None, :]
    t_raw = (relative @ vector) / length_sq
    # In x/y coordinates, cross(v, p) = vx*py - vy*px. Arrays here are y/x.
    signed = (vector[1] * relative[:, 0] - vector[0] * relative[:, 1]) / length

    included = (t_raw >= -1e-12) & (t_raw <= 1.0 + 1e-12)
    if axis.half_width_px is not None:
        included &= np.abs(signed) <= float(axis.half_width_px) + 1e-12
    if include_mask is not None:
        include = np.asarray(include_mask, dtype=bool)
        if include.shape != (ds.n_pixels,):
            raise ValueError(
                f"include_mask shape {include.shape} != ({ds.n_pixels},)"
            )
        included &= include

    t = np.where(included, np.clip(t_raw, 0.0, 1.0), np.nan)
    signed_out = np.where(included, signed, np.nan)
    return AxisProjection(
        axis=axis,
        t=t.astype(np.float64, copy=False),
        signed_distance_px=signed_out.astype(np.float64, copy=False),
        included_mask=included,
        coordinate_fingerprint=coordinate_fingerprint(ds),
    )
