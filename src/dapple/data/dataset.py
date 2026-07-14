"""The central MSIDataset with two backends: PeakList (sparse) and PeakMatrix (dense).

PeakList is the natural representation for processed-mode imzML (one variable-length
peak list per pixel). It's stored CSR-style: three flat arrays plus an offsets vector.
Each backing array can be a numpy array (in-RAM) or a zarr.Array (on-disk lazy).

PeakMatrix is the post-consensus representation: a dense (n_pixels, n_peaks) float32
array plus a shared float64 m/z axis. This is what the HyperspectralBrowser renders
from and what writes to multipage TIFF.

MSIDataset wraps a backend plus its coordinates, ExperimentParams, history of applied
operators, and a stable identity hash. The class itself is *immutable* in spirit —
operators return a new MSIDataset rather than mutating in place — but the underlying
arrays may be lazy/zarr-backed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from dapple.data.coords import coords_to_grid_index, project_to_grid
from dapple.data.hashing import combine_hashes, hash_array_content, hash_obj
from dapple.data.metadata import DatasetIdentity, ExperimentParams, RoiDef

ProjectionKind = Literal["tic", "rms", "median", "base_peak", "peak_count", "mean_mz"]
AggregateMethod = Literal["mean", "median", "sum", "max"]

# Arrays under these keys describe the current PeakMatrix m/z axis one value
# per channel.  Any operator that removes channels must subset all of them,
# otherwise later displays/exports can silently associate a statistic with the
# wrong ion.
CHANNEL_ALIGNED_EXTRA_KEYS = frozenset(
    {
        "consensus_prevalence",
        "consensus_n_peaks_per_channel",
        "n_peaks_per_channel",  # legacy spelling
        "cohort_prevalence",  # legacy spelling
        "cohort_dataset_prevalence",
        "dataset_prevalence",  # legacy spelling
        "morans_i_per_channel",
        "morans_i_p_values",
        "morans_i_q_values",
        "prevalence_fdr_p_values",
        "prevalence_fdr_q_values",
        "bg_subtract_per_channel_bg_mean",
    }
)


class _ArrayLike(Protocol):
    """Both numpy arrays and zarr.Array implement this shape — duck-type accordingly."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    def __getitem__(self, key: Any) -> np.ndarray: ...
    def __len__(self) -> int: ...


@dataclass(frozen=True, eq=False)
class PeakList:
    """CSR-style sparse peak list backend.

    For pixel `i` (0 <= i < n_pixels):
        mz[offsets[i]:offsets[i+1]]  is the m/z array
        intensity[offsets[i]:offsets[i+1]]  is the matching intensity

    `mz` is float64 (precision matters for ppm-scale work); `intensity` is float32
    (counts/AU don't need the extra precision).
    """

    mz: _ArrayLike  # 1-D float64
    intensity: _ArrayLike  # 1-D float32
    offsets: _ArrayLike  # 1-D int64, length n_pixels + 1
    n_pixels: int
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_pixels < 0:
            raise ValueError("n_pixels must be non-negative")
        if len(self.mz.shape) != 1 or len(self.intensity.shape) != 1:
            raise ValueError("PeakList mz and intensity arrays must be one-dimensional")
        if self.mz.shape[0] != self.intensity.shape[0]:
            raise ValueError(
                f"mz length {self.mz.shape[0]} != intensity length {self.intensity.shape[0]}"
            )
        if len(self.offsets) != self.n_pixels + 1:
            raise ValueError(
                f"offsets length {len(self.offsets)} != n_pixels+1 ({self.n_pixels + 1})"
            )
        offsets = np.asarray(self.offsets[:])
        mz_values = np.asarray(self.mz[:])
        intensity_values = np.asarray(self.intensity[:])
        if offsets.ndim != 1:
            raise ValueError("PeakList offsets must be one-dimensional")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("PeakList offsets must have an integer dtype")
        if offsets.size and int(offsets[0]) != 0:
            raise ValueError("PeakList offsets must start at zero")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("PeakList offsets must be monotonically non-decreasing")
        if offsets.size and int(offsets[-1]) != int(self.mz.shape[0]):
            raise ValueError(
                f"final offset {int(offsets[-1])} != peak-array length {self.mz.shape[0]}"
            )
        if not np.isfinite(mz_values).all() or not np.isfinite(intensity_values).all():
            raise ValueError("PeakList m/z and intensity arrays must contain only finite values")
        if (mz_values < 0).any():
            raise ValueError("PeakList m/z values must be non-negative")
        object.__setattr__(
            self,
            "_fingerprint",
            combine_hashes(
                hash_array_content(mz_values),
                hash_array_content(intensity_values),
                hash_array_content(offsets),
                hash_obj(self.n_pixels),
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Cached content fingerprint for cache/provenance identity."""
        return self._fingerprint

    def pixel(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a = int(self.offsets[i])
        b = int(self.offsets[i + 1])
        return np.asarray(self.mz[a:b]), np.asarray(self.intensity[a:b])

    def per_pixel_count(self) -> np.ndarray:
        offs = np.asarray(self.offsets[:])
        return np.diff(offs).astype(np.int64)

    def per_pixel_reduce(
        self,
        fn: Literal["sum", "mean", "median", "max", "rms", "count", "argmax_mz"],
    ) -> np.ndarray:
        """Reduce each pixel's intensity (or m/z) array to a scalar; return shape (n_pixels,).

        ``mean``, ``median``, and ``rms`` reduce over *non-zero* intensities only.
        Vendor centroided exports — notably Xcalibur ANDI-MS .cdf — flank every detected
        peak with zero-intensity sentinels marking peak edges, so a plain median over
        the full array collapses to 0 and the resulting per-pixel projection is
        uninformative. ``sum``, ``max``, ``count``, and ``argmax_mz`` use the full array.
        """
        offs = np.asarray(self.offsets[:], dtype=np.int64)
        intensity = np.asarray(self.intensity[:])
        counts = np.diff(offs)
        out = np.zeros(self.n_pixels, dtype=np.float32)
        if fn == "count":
            return counts.astype(np.float32, copy=False)
        if intensity.size == 0:
            return out

        nonempty = counts > 0
        starts = offs[:-1][nonempty]
        if fn in {"sum", "mean", "max", "rms"}:
            if fn == "sum":
                reduced = np.add.reduceat(intensity.astype(np.float64), starts)
            elif fn == "max":
                reduced = np.maximum.reduceat(intensity, starts)
            else:
                positive = intensity > 0
                n_positive = np.add.reduceat(positive.astype(np.int64), starts)
                if fn == "mean":
                    numerator = np.add.reduceat(
                        np.where(positive, intensity, 0.0).astype(np.float64), starts
                    )
                else:
                    numerator = np.add.reduceat(
                        np.where(positive, intensity, 0.0).astype(np.float64) ** 2,
                        starts,
                    )
                with np.errstate(invalid="ignore", divide="ignore"):
                    reduced = np.divide(
                        numerator,
                        n_positive,
                        out=np.zeros_like(numerator),
                        where=n_positive > 0,
                    )
                if fn == "rms":
                    np.sqrt(reduced, out=reduced)
            out[nonempty] = reduced.astype(np.float32, copy=False)
            return out

        mz = self.mz  # only materialize for argmax_mz
        for i in range(self.n_pixels):
            a, b = int(offs[i]), int(offs[i + 1])
            if a == b:
                continue
            seg = intensity[a:b]
            if fn == "median":
                nz = seg[seg > 0]
                out[i] = float(np.median(nz)) if nz.size else 0.0
            elif fn == "argmax_mz":
                out[i] = float(np.asarray(mz[a:b])[seg.argmax()])
            else:  # pragma: no cover
                raise ValueError(f"unknown reduction {fn!r}")
        return out


@dataclass(frozen=True, eq=False)
class PeakMatrix:
    """Dense post-consensus backend: (n_pixels, n_peaks) float32 + 1-D float64 m/z axis."""

    matrix: _ArrayLike  # 2-D float32, shape (n_pixels, n_peaks)
    mz_axis: _ArrayLike  # 1-D float64, length n_peaks
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.matrix.shape) != 2 or len(self.mz_axis.shape) != 1:
            raise ValueError("PeakMatrix matrix must be 2-D and mz_axis must be 1-D")
        if self.matrix.shape[1] != len(self.mz_axis):
            raise ValueError(
                f"matrix peaks dim {self.matrix.shape[1]} != mz_axis length {len(self.mz_axis)}"
            )
        axis = np.asarray(self.mz_axis[:], dtype=np.float64)
        matrix = np.asarray(self.matrix[:])
        if not np.isfinite(axis).all():
            raise ValueError("PeakMatrix mz_axis must contain only finite values")
        if not np.isfinite(matrix).all():
            raise ValueError("PeakMatrix matrix must contain only finite values")
        if axis.size > 1 and np.any(np.diff(axis) <= 0):
            raise ValueError("PeakMatrix mz_axis must be strictly increasing")
        object.__setattr__(
            self,
            "_fingerprint",
            combine_hashes(
                hash_array_content(matrix),
                hash_array_content(axis),
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Cached content fingerprint for cache/provenance identity."""
        return self._fingerprint

    @property
    def n_pixels(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_peaks(self) -> int:
        return int(self.matrix.shape[1])

    def pixel(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return non-zero peaks for pixel i (mz, intensity) — sparse view of dense row."""
        row = np.asarray(self.matrix[i, :])
        nz = np.flatnonzero(row)
        return np.asarray(self.mz_axis[:])[nz], row[nz]

    def aggregate(self, mask: np.ndarray, method: AggregateMethod) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate a subset of pixels (mask: bool (n_pixels,)) along axis 0."""
        if mask.shape != (self.n_pixels,):
            raise ValueError(f"mask shape {mask.shape} != ({self.n_pixels},)")
        if not mask.any():
            return np.asarray(self.mz_axis[:]), np.zeros(self.n_peaks, dtype=np.float32)
        sub = np.asarray(self.matrix[mask, :])
        if method == "mean":
            agg = sub.mean(axis=0)
        elif method == "median":
            agg = np.median(sub, axis=0)
        elif method == "sum":
            agg = sub.sum(axis=0)
        elif method == "max":
            agg = sub.max(axis=0)
        else:  # pragma: no cover
            raise ValueError(f"unknown method {method!r}")
        return np.asarray(self.mz_axis[:]), agg.astype(np.float32, copy=False)


@dataclass(frozen=True)
class OpRecord:
    """Provenance record for one operator application."""

    op_name: str
    params_hash: str
    input_hash: str
    output_hash: str
    diagnostics_summary: tuple[tuple[str, dict[str, float]], ...]


@dataclass(frozen=True, eq=False)
class MSIDataset:
    """A loaded MSI dataset (raw or processed).

    `coords[i]` is the (x, y) pixel of the spectrum at backend index i.
    `grid_shape` is (height, width) for the napari raster.
    `metadata` is the experiment params; `history` is the chain of applied operators.
    `identity` is the stable on-disk identity (path + content hash).

    The dataclass uses `eq=False` because some fields (numpy arrays, dicts) are not
    hashable; use `MSIDataset.hash()` for content-based identity instead.
    """

    coords: np.ndarray  # (n_pixels, 2) int32
    grid_shape: tuple[int, int]
    metadata: ExperimentParams
    backend: PeakList | PeakMatrix
    identity: DatasetIdentity
    history: tuple[OpRecord, ...] = ()
    rois: tuple[RoiDef, ...] = ()
    rng_seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    _coordinate_fingerprint: str = field(init=False, repr=False, compare=False)
    _extra_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2), got {self.coords.shape}")
        if not np.issubdtype(self.coords.dtype, np.integer):
            raise ValueError("coords must have an integer dtype")
        if self.coords.shape[0] != self.backend.n_pixels:
            raise ValueError(
                f"coords pixel count {self.coords.shape[0]} != backend "
                f"n_pixels {self.backend.n_pixels}"
            )
        if len(self.grid_shape) != 2 or any(int(v) <= 0 for v in self.grid_shape):
            raise ValueError(f"grid_shape must contain two positive dimensions, got {self.grid_shape}")
        if self.coords.size:
            flat = coords_to_grid_index(self.coords, self.grid_shape)
            if np.unique(flat).size != flat.size:
                raise ValueError("coords contain duplicate populated-pixel positions")
        else:
            flat = np.empty(0, dtype=np.int64)
        object.__setattr__(
            self,
            "_coordinate_fingerprint",
            combine_hashes(hash_obj(self.grid_shape), hash_array_content(flat)),
        )
        # File locations are provenance already represented by ``identity`` and
        # must not make otherwise identical scientific state depend on where a
        # sidecar happened to be mounted.  All other extras are state: many
        # operators attach tolerance curves, reference intensities, or channel
        # statistics that directly affect downstream results.
        scientific_extra = {
            key: value
            for key, value in self.extra.items()
            if key not in {"imzml_path", "ibd_path", "sidecar_path"}
        }
        if isinstance(self.backend, PeakMatrix):
            for key in CHANNEL_ALIGNED_EXTRA_KEYS:
                if key not in self.extra:
                    continue
                shape = np.asarray(self.extra[key]).shape
                if shape != (self.backend.n_peaks,):
                    raise ValueError(
                        f"channel-aligned extra {key!r} has shape {shape}; "
                        f"expected ({self.backend.n_peaks},)"
                    )
        object.__setattr__(self, "_extra_fingerprint", hash_obj(scientific_extra))

    @property
    def n_pixels(self) -> int:
        return self.backend.n_pixels

    @property
    def is_aligned(self) -> bool:
        """True iff backend is dense post-consensus PeakMatrix."""
        return isinstance(self.backend, PeakMatrix)

    @property
    def coordinate_fingerprint(self) -> str:
        """Cached fingerprint of populated positions and raster geometry."""
        return self._coordinate_fingerprint

    @property
    def scientific_extra_fingerprint(self) -> str:
        """Cached fingerprint of downstream-relevant attached analysis state."""
        return self._extra_fingerprint

    def pixel_spectrum(self, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        """Spectrum at raster (x, y). Empty arrays if no spectrum at that pixel."""
        sel = (self.coords[:, 0] == x) & (self.coords[:, 1] == y)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)
        return self.backend.pixel(int(idx[0]))

    def project(self, kind: ProjectionKind) -> np.ndarray:
        """Compute a per-pixel scalar summary and scatter it onto the (H, W) raster.

        `tic`         total ion current per pixel (sum of intensities)
        `rms`         root-mean-square intensity per pixel
        `median`      median intensity per pixel
        `base_peak`   max intensity per pixel
        `peak_count`  number of peaks per pixel (PeakList only; constant for PeakMatrix nz count)
        `mean_mz`     intensity-weighted mean m/z per pixel (only on PeakList; needs revamp on
                      PeakMatrix — use the m/z axis directly for that backend)
        """
        if isinstance(self.backend, PeakList):
            pl = self.backend
            if kind == "tic":
                vals = pl.per_pixel_reduce("sum")
            elif kind == "rms":
                vals = pl.per_pixel_reduce("rms")
            elif kind == "median":
                vals = pl.per_pixel_reduce("median")
            elif kind == "base_peak":
                vals = pl.per_pixel_reduce("max")
            elif kind == "peak_count":
                vals = pl.per_pixel_reduce("count")
            elif kind == "mean_mz":
                vals = _mean_mz_per_pixel_peaklist(pl)
            else:  # pragma: no cover
                raise ValueError(f"unknown projection {kind!r}")
        else:  # PeakMatrix
            pm = self.backend
            mat = np.asarray(pm.matrix[:])  # densify; PeakMatrix is already dense
            if kind == "tic":
                vals = mat.sum(axis=1)
            elif kind == "rms":
                vals = np.sqrt(np.mean(mat.astype(np.float64) ** 2, axis=1))
            elif kind == "median":
                vals = np.median(mat, axis=1)
            elif kind == "base_peak":
                vals = mat.max(axis=1)
            elif kind == "peak_count":
                vals = (mat > 0).sum(axis=1)
            elif kind == "mean_mz":
                axis = np.asarray(pm.mz_axis[:])
                weights = mat.astype(np.float64)
                wsum = weights.sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals = np.where(wsum > 0, (weights * axis).sum(axis=1) / wsum, 0.0)
            else:  # pragma: no cover
                raise ValueError(f"unknown projection {kind!r}")
        return project_to_grid(vals.astype(np.float32, copy=False), self.coords, self.grid_shape)

    def with_history(self, record: OpRecord) -> MSIDataset:
        return replace(self, history=(*self.history, record))

    def with_backend(self, backend: PeakList | PeakMatrix) -> MSIDataset:
        return replace(self, backend=backend)

    def with_rois(self, rois: tuple[RoiDef, ...]) -> MSIDataset:
        return replace(self, rois=rois)

    def hash(self, *, include_rois: bool = True) -> str:
        """Return a stable fingerprint of the dataset's processing state.

        Metadata is part of the fingerprint because it can change operator defaults
        and fallback behaviour.  ROI geometry is included by default so saved-run
        signatures and provenance change when an analyst edits a polygon.  The
        runner may set ``include_rois=False`` for operators that provably do not
        inspect ROIs; this preserves upstream cache hits while keeping ROI-dependent
        operators correct.
        """
        parts = [
            hash_obj(self.identity),
            hash_obj(self.metadata),
            self.coordinate_fingerprint,
            self.backend.fingerprint,
            self.scientific_extra_fingerprint,
        ]
        if self.history:
            parts.append(combine_hashes(*(r.output_hash for r in self.history)))
        if include_rois:
            parts.append(hash_obj(self.rois))
        return combine_hashes(*parts)


def _mean_mz_per_pixel_peaklist(pl: PeakList) -> np.ndarray:
    offs = np.asarray(pl.offsets[:], dtype=np.int64)
    intensity = np.asarray(pl.intensity[:], dtype=np.float64)
    mz = np.asarray(pl.mz[:], dtype=np.float64)
    out = np.zeros(pl.n_pixels, dtype=np.float32)
    if intensity.size == 0:
        return out
    nonempty = np.diff(offs) > 0
    starts = offs[:-1][nonempty]
    weight_sum = np.add.reduceat(intensity, starts)
    weighted_mz = np.add.reduceat(intensity * mz, starts)
    values = np.divide(
        weighted_mz,
        weight_sum,
        out=np.zeros_like(weighted_mz),
        where=weight_sum > 0,
    )
    out[nonempty] = values.astype(np.float32, copy=False)
    return out


def subset_channel_aligned_extra(
    extra: dict[str, Any],
    keep: np.ndarray,
    n_channels: int,
    *,
    replacements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy extras while preserving the current PeakMatrix channel invariant.

    Known one-value-per-channel arrays are subset with ``keep``.  A known key
    with a non-matching shape is rejected rather than silently carried onto a
    different m/z axis.  ``replacements`` are applied last and are validated
    against the number of surviving channels.
    """
    keep_arr = np.asarray(keep, dtype=bool)
    if keep_arr.shape != (n_channels,):
        raise ValueError(f"keep shape {keep_arr.shape} != ({n_channels},)")
    out = dict(extra)
    for key in CHANNEL_ALIGNED_EXTRA_KEYS:
        if key not in out:
            continue
        values = np.asarray(out[key])
        if values.shape != (n_channels,):
            raise ValueError(
                f"channel-aligned extra {key!r} has shape {values.shape}; "
                f"expected ({n_channels},)"
            )
        out[key] = values[keep_arr]
    if replacements:
        out.update(replacements)
    n_out = int(keep_arr.sum())
    for key in CHANNEL_ALIGNED_EXTRA_KEYS:
        if key in out and np.asarray(out[key]).shape != (n_out,):
            raise ValueError(
                f"replacement channel-aligned extra {key!r} has shape "
                f"{np.asarray(out[key]).shape}; expected ({n_out},)"
            )
    return out


def open_imzml(path: Path | str, *, lazy: bool = True) -> MSIDataset:
    """Convenience: open an imzML file. Imports lazily to avoid a circular import."""
    from dapple.io.imzml_reader import read_imzml

    return read_imzml(Path(path), lazy=lazy)
