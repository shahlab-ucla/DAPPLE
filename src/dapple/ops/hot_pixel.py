"""Hot-pixel detection and correction.

A "hot pixel" in MSI is a single raster position with anomalously high intensity in
one or many channels — usually a detector glitch, charge-spike, or matrix crystal
artifact. Left in, hot pixels:
  - dominate the auto-contrast bounds for every channel they're in
  - skew per-pixel normalization factors
  - become "consensus peaks" that exist only at one location

Detection is always TIC-based and post per-pixel-MAD. Correction options:
  - ``neighbors_median``: replace every peak's intensity at the hot pixel with the
    median intensity of the same peak across the 8 spatial neighbors. Conservative
    — preserves peak counts but kills the spike.
  - ``zero``: zero every intensity in the hot pixel. Simple but destroys the pixel.
  - ``mark``: don't touch intensities, but record the pixel index so downstream
    operators (TIC norm, projections) can mask it. The mask is read by other
    operators if and when they opt to consult it.

The operator emits per-pixel hot-pixel scores in its diagnostic so the threshold
can be retuned without re-running peak picking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dapple.data.coords import coords_to_grid_index, project_to_grid
from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.data.metadata import ExperimentParams
from dapple.ops.base import (
    Diagnostic,
    OpParams,
    OpResult,
    Operator,
    merge_op_record,
    register,
)

CorrectionMode = Literal["neighbors_median", "zero", "mark"]


@dataclass(frozen=True)
class HotPixelParams(OpParams):
    """Per-pixel TIC must exceed (median + k_mad × 1.4826 × MAD) to qualify as hot."""

    k_mad: float = field(
        default=5.0,
        metadata={
            "label": "Hot-pixel threshold (× MAD)",
            "help": (
                "A pixel is flagged hot when its total ion current exceeds "
                "median(TIC) + k × 1.4826 × MAD(TIC). Default 5 — a Gaussian-tail "
                "equivalent of ~3.4·σ on a robust scale, conservative for real MSI "
                "data. Drop to 3 to catch more spikes (false-positive risk on "
                "hotspots that are genuine signal); raise to 8–10 to catch only "
                "extreme glitches."
            ),
        },
    )
    correction: CorrectionMode = field(
        default="neighbors_median",
        metadata={
            "label": "Correction strategy",
            "help": (
                "neighbors_median: replace intensity at the hot pixel with the "
                "median of its 8 spatial neighbors per channel. Conservative — "
                "preserves peak counts. zero: set all intensities at the hot "
                "pixel to 0 (simplest, destroys the pixel). mark: leave "
                "intensities alone but tag the pixel id so projections / "
                "normalization can opt to ignore it."
            ),
        },
    )
    min_neighbors: int = field(
        default=3,
        metadata={
            "label": "Minimum non-hot neighbors",
            "help": (
                "When using neighbors_median, require at least this many of the 8 "
                "neighbors to be non-hot. Pixels that fail (e.g. a hot pixel at the "
                "image edge with too few valid neighbors) fall back to the 'zero' "
                "strategy. Default 3."
            ),
        },
    )


@register
class HotPixelFilter(Operator):
    name = "hot_pixel_filter"
    params_cls = HotPixelParams

    def default_params(self, ep: ExperimentParams) -> HotPixelParams:
        return HotPixelParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: HotPixelParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        # Compute per-pixel TIC and the robust threshold.
        tic = ds.project("tic")
        flat_tic = np.asarray(tic).ravel()
        # Restrict the noise floor estimate to populated pixels (TIC > 0). Empty
        # raster cells would otherwise drag the median to 0 and make every cell hot.
        populated_idx = coords_to_grid_index(ds.coords, ds.grid_shape)
        per_pixel_tic = flat_tic[populated_idx]
        if per_pixel_tic.size == 0:
            return _passthrough(ds, params, self.name, "no populated pixels — nothing to do")

        med = float(np.median(per_pixel_tic))
        mad = float(np.median(np.abs(per_pixel_tic - med)))
        threshold = med + params.k_mad * 1.4826 * mad
        is_hot = per_pixel_tic > threshold
        n_hot = int(is_hot.sum())
        hot_pixel_indices = np.flatnonzero(is_hot)  # indices into ds.coords

        # Build the corrected dataset.
        if n_hot == 0:
            return _passthrough(ds, params, self.name, "no hot pixels detected")

        if params.correction == "mark":
            # No mutation — record the indices in extra so projections can mask.
            new_extra = {
                **ds.extra,
                "hot_pixel_indices": hot_pixel_indices.astype(np.int64, copy=False),
                "hot_pixel_threshold_tic": threshold,
            }
            new_ds = _ds_with_extra(ds, new_extra)
        else:
            corrected = _correct_hot_pixels(
                ds,
                hot_pixel_indices=hot_pixel_indices,
                correction=params.correction,
                min_neighbors=params.min_neighbors,
            )
            new_extra = {
                **ds.extra,
                "hot_pixel_indices": hot_pixel_indices.astype(np.int64, copy=False),
                "hot_pixel_threshold_tic": threshold,
            }
            new_ds = corrected.__class__(
                coords=corrected.coords,
                grid_shape=corrected.grid_shape,
                metadata=corrected.metadata,
                backend=corrected.backend,
                identity=corrected.identity,
                history=corrected.history,
                rois=corrected.rois,
                rng_seed=corrected.rng_seed,
                extra=new_extra,
            )

        diag = Diagnostic(
            name=self.name,
            summary={
                "n_hot_pixels": float(n_hot),
                "fraction_hot": float(n_hot / max(per_pixel_tic.size, 1)),
                "tic_median": med,
                "tic_mad": mad,
                "tic_threshold": threshold,
                "k_mad": float(params.k_mad),
            },
            payload={
                "per_pixel_tic": per_pixel_tic,
                "hot_pixel_indices": hot_pixel_indices,
            },
            figure_hint="histogram:per_pixel_tic_with_threshold_marker",
        )
        record = merge_op_record(
            op_name=self.name,
            params=params,
            input_ds=ds,
            output_ds=new_ds,
            diagnostics=[diag],
        )
        new_ds = new_ds.with_history(record)
        return OpResult(dataset=new_ds, diagnostics=[diag])


def _correct_hot_pixels(
    ds: MSIDataset,
    *,
    hot_pixel_indices: np.ndarray,
    correction: CorrectionMode,
    min_neighbors: int,
) -> MSIDataset:
    """Return a copy of ds with hot-pixel intensities replaced according to `correction`."""
    if isinstance(ds.backend, PeakMatrix):
        return _correct_peakmatrix(ds, hot_pixel_indices, correction, min_neighbors)
    if isinstance(ds.backend, PeakList):
        return _correct_peaklist(ds, hot_pixel_indices, correction, min_neighbors)
    raise TypeError(f"unknown backend {type(ds.backend)}")


def _correct_peakmatrix(
    ds: MSIDataset,
    hot_pixel_indices: np.ndarray,
    correction: CorrectionMode,
    min_neighbors: int,
) -> MSIDataset:
    pm: PeakMatrix = ds.backend  # type: ignore[assignment]
    matrix = np.asarray(pm.matrix[:]).copy()
    if correction == "zero":
        matrix[hot_pixel_indices, :] = 0.0
    else:  # neighbors_median
        neighbor_grid = _grid_neighbor_indices(ds)
        for px_idx in hot_pixel_indices:
            neighbors = neighbor_grid[px_idx]
            valid_neighbors = [n for n in neighbors if n >= 0 and n not in set(hot_pixel_indices)]
            if len(valid_neighbors) < min_neighbors:
                matrix[px_idx, :] = 0.0
                continue
            matrix[px_idx, :] = np.median(matrix[valid_neighbors, :], axis=0)
    new_pm = PeakMatrix(matrix=matrix.astype(np.float32, copy=False), mz_axis=pm.mz_axis)
    return ds.with_backend(new_pm)


def _correct_peaklist(
    ds: MSIDataset,
    hot_pixel_indices: np.ndarray,
    correction: CorrectionMode,
    min_neighbors: int,
) -> MSIDataset:
    """For PeakList we either zero out a hot pixel's peaks, or replace them with the
    median-intensity envelope of the neighbor pixels' peak lists.

    For ``neighbors_median`` we don't try to interpolate the (m/z, intensity) graph
    across neighbors — the m/z axes don't line up before consensus alignment. Instead,
    we scale the hot pixel's intensities so its TIC matches the median TIC of valid
    neighbors. That preserves the hot pixel's m/z structure but kills the spike,
    which is the property hot-pixel correction usually exists to provide.
    """
    pl: PeakList = ds.backend  # type: ignore[assignment]
    offsets = np.asarray(pl.offsets[:])
    intensity = np.asarray(pl.intensity[:]).astype(np.float32, copy=True)

    if correction == "zero":
        for px_idx in hot_pixel_indices:
            a, b = int(offsets[px_idx]), int(offsets[px_idx + 1])
            intensity[a:b] = 0.0
    else:  # neighbors_median
        tic = ds.project("tic").ravel()
        flat_idx = coords_to_grid_index(ds.coords, ds.grid_shape)
        per_pixel_tic = tic[flat_idx]
        neighbor_grid = _grid_neighbor_indices(ds)
        hot_set = set(int(i) for i in hot_pixel_indices)
        for px_idx in hot_pixel_indices:
            neighbors = neighbor_grid[int(px_idx)]
            valid = [n for n in neighbors if n >= 0 and n not in hot_set]
            if len(valid) < min_neighbors:
                a, b = int(offsets[px_idx]), int(offsets[px_idx + 1])
                intensity[a:b] = 0.0
                continue
            target_tic = float(np.median(per_pixel_tic[valid]))
            current_tic = float(per_pixel_tic[int(px_idx)])
            if current_tic <= 0:
                continue
            scale = target_tic / current_tic
            a, b = int(offsets[px_idx]), int(offsets[px_idx + 1])
            intensity[a:b] = (intensity[a:b] * scale).astype(np.float32, copy=False)

    new_pl = PeakList(
        mz=np.asarray(pl.mz[:]).copy(),
        intensity=intensity,
        offsets=offsets.copy(),
        n_pixels=pl.n_pixels,
    )
    return ds.with_backend(new_pl)


def _grid_neighbor_indices(ds: MSIDataset) -> np.ndarray:
    """For each populated pixel, return the indices of its 8 neighbors (or -1 outside).

    Returns shape ``(n_pixels, 8)`` int64.
    """
    h, w = ds.grid_shape
    flat_idx = coords_to_grid_index(ds.coords, ds.grid_shape)
    # Build a grid → pixel-index map so neighbor lookups are O(1).
    grid_to_pixel = -np.ones(h * w, dtype=np.int64)
    grid_to_pixel[flat_idx] = np.arange(ds.n_pixels, dtype=np.int64)

    iy, ix = divmod(flat_idx, w)
    out = -np.ones((ds.n_pixels, 8), dtype=np.int64)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k, (dy, dx) in enumerate(offsets):
        ny = iy + dy
        nx = ix + dx
        in_bounds = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        flat_n = np.where(in_bounds, ny * w + nx, 0)
        neighbor_pix = grid_to_pixel[flat_n]
        out[:, k] = np.where(in_bounds, neighbor_pix, -1)
    return out


def _passthrough(
    ds: MSIDataset, params: HotPixelParams, op_name: str, note: str
) -> OpResult:
    diag = Diagnostic(
        name=op_name,
        summary={"n_hot_pixels": 0.0, "k_mad": float(params.k_mad)},
        payload={"note": np.array([note], dtype=object)},
    )
    record = merge_op_record(
        op_name=op_name, params=params, input_ds=ds, output_ds=ds, diagnostics=[diag]
    )
    return OpResult(dataset=ds.with_history(record), diagnostics=[diag])


def _ds_with_extra(ds: MSIDataset, new_extra: dict) -> MSIDataset:
    return ds.__class__(
        coords=ds.coords,
        grid_shape=ds.grid_shape,
        metadata=ds.metadata,
        backend=ds.backend,
        identity=ds.identity,
        history=ds.history,
        rois=ds.rois,
        rng_seed=ds.rng_seed,
        extra=new_extra,
    )
