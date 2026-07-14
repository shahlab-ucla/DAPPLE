"""Pixel-coordinate helpers.

imzML stores spectra in arbitrary order with an explicit (x, y) per spectrum. We need
fast bidirectional mapping between flat spectrum index, (y, x) raster coordinates, and
the dense grid raster used by napari.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

CoordinateOrigin = Literal["auto", "zero", "one"]


def coords_to_grid_index(
    coords: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    origin: CoordinateOrigin = "auto",
) -> np.ndarray:
    """Map (N, 2) integer (x, y) coords to flat indices into a (H, W) grid.

    Coords are 1-indexed in imzML; we accept 0- or 1-indexed input and return
    0-indexed flat indices ``iy * W + ix``. ``auto`` treats an all-positive
    coordinate set as one-based, which correctly handles sparse/cropped imzML
    images whose first populated row or column is greater than one. Pass an
    explicit origin for cropped zero-based arrays that contain no zero coordinate.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {coords.shape}")
    height, width = grid_shape
    cmin = coords.min(axis=0)
    if (cmin < 0).any():
        raise ValueError(f"negative coordinate found: {cmin}")
    if origin not in {"auto", "zero", "one"}:
        raise ValueError(f"unknown coordinate origin {origin!r}")
    one_based = origin == "one" or (origin == "auto" and (cmin >= 1).all())
    if one_based:
        x = coords[:, 0] - 1
        y = coords[:, 1] - 1
    else:
        x = coords[:, 0]
        y = coords[:, 1]
    if (x >= width).any() or (y >= height).any():
        raise ValueError(
            f"coord out of grid bounds: max=(x={x.max()}, y={y.max()}) grid={grid_shape}"
        )
    return (y.astype(np.int64) * width + x.astype(np.int64)).astype(np.int64)


def grid_index_to_coords(idx: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Inverse of `coords_to_grid_index`. Returns 0-indexed (x, y) pairs."""
    height, width = grid_shape
    iy, ix = divmod(idx.astype(np.int64), width)
    if (iy >= height).any():
        raise ValueError(f"flat idx out of bounds for grid {grid_shape}")
    return np.stack([ix, iy], axis=1).astype(np.int32)


def project_to_grid(
    flat_values: np.ndarray, coords: np.ndarray, grid_shape: tuple[int, int], fill: float = 0.0
) -> np.ndarray:
    """Scatter a (N,) array of per-pixel values onto a (H, W) image with `fill` elsewhere.

    Pixels not represented in `coords` are left at `fill`. This is how every projection
    (TIC, RMS, ...) is rendered for napari.
    """
    if flat_values.ndim != 1 or flat_values.shape[0] != coords.shape[0]:
        raise ValueError(
            f"flat_values shape {flat_values.shape} doesn't match coords {coords.shape}"
        )
    height, width = grid_shape
    img = np.full((height, width), fill, dtype=np.float32)
    flat_idx = coords_to_grid_index(coords, grid_shape)
    img.reshape(-1)[flat_idx] = flat_values.astype(np.float32, copy=False)
    return img
