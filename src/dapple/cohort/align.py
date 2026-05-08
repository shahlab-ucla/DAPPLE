"""Cohort alignment: shared consensus across multiple datasets.

The single-dataset pipeline ends with a per-dataset consensus axis. When you
have several datasets that should be analyzed jointly (treatment / control,
replicates, before / after), comparing them post-hoc requires aligning their
consensus axes — otherwise the same molecular ion ends up in slightly different
columns of each dataset's PeakMatrix.

This module's ``align_cohort`` function implements *pooled consensus*:

1. Per-dataset preprocessing (identical to the single-dataset pipeline up
   through peak picking, optionally recalibrated against per-dataset reference
   ions).
2. **Pool** every post-pick peak from every pixel of every dataset into a
   single (m/z, intensity, dataset_id, pixel_id) cloud.
3. Run a **single KDE consensus** on the pooled cloud, producing one shared
   m/z axis.
4. For each dataset, build its own ``(n_pixels_d, n_consensus_shared)``
   ``PeakMatrix`` by assigning the dataset's peaks to the shared axis using
   the same window logic as ``KdeConsensusAlignment``.

The result is a ``CohortAlignResult``: each dataset gets its own MSIDataset
with a ``PeakMatrix`` backend whose m/z axis matches every other dataset in the
cohort, plus a single ``shared_consensus_mz`` array and per-dataset prevalence.

This is the simplest defensible cohort algorithm. More sophisticated
approaches (Procrustes alignment of per-dataset axes, batch-effect
correction via ComBat) belong in operator modules of their own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.io.cdf_image_reader import read_cdf_image
from dapple.io.imzml_reader import read_imzml
from dapple.ops.consensus import KdeConsensusParams
from dapple.ops.normalize import MedianNormalize
from dapple.ops.peak_pick import SnrPeakPick
from dapple.ops.recalibrate import MsiwarpRecalibrate
from dapple.ops.reference_ions import DetectReferenceIons
from dapple.ops.tolerance import (
    EmpiricalToleranceFromReferenceIons,
    EmpiricalToleranceParams,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CohortAlignParams:
    """Parameters that govern the per-dataset prep + the pooled consensus.

    Most knobs mirror their single-dataset counterparts. The ones unique to
    cohort processing are ``pool_normalize`` (whether to normalize within each
    dataset before pooling — usually yes; turns off only if you've pre-
    normalized externally) and ``recalibrate`` (whether to run msiwarp on each
    dataset before pooling — yes for TOF-class instruments, no for Orbitrap).
    """

    bandwidth_ppm: float = 50.0
    bandwidth_scale: float = 1.0
    n_grid_points: int = 32768
    min_prominence_quantile: float = 0.5
    min_prevalence: float = 0.05
    default_tol_ppm: float = 50.0
    pool_normalize: bool = True
    recalibrate: bool = True
    rng_seed: int = 0


@dataclass
class CohortAlignResult:
    """Output of ``align_cohort``.

    ``aligned_datasets`` parallels the input list: each entry is a new
    MSIDataset with a PeakMatrix backend whose ``mz_axis`` equals
    ``shared_consensus_mz``. Per-dataset prevalence (fraction of that dataset's
    pixels carrying each consensus peak) is stored in
    ``per_dataset_prevalence`` keyed by dataset index.

    ``cohort_prevalence`` is the *cross-dataset* prevalence — the fraction of
    cohort-wide pixels (summed across all datasets) carrying each consensus
    peak. It's how the pooled-KDE consensus filter selected the surviving
    channels.

    ``diagnostics`` is a flat dict of summary scalars suitable for logging /
    saving to a manifest.
    """

    aligned_datasets: list[MSIDataset]
    shared_consensus_mz: np.ndarray
    cohort_prevalence: np.ndarray
    per_dataset_prevalence: dict[int, np.ndarray]
    diagnostics: dict[str, float]


def align_cohort(
    datasets: Sequence[MSIDataset],
    params: CohortAlignParams | None = None,
) -> CohortAlignResult:
    """Compute a shared consensus axis across `datasets` and return per-dataset
    PeakMatrix-backed datasets aligned to it.

    Each input dataset must have a ``PeakList`` backend (raw or post-pick — the
    function doesn't re-pick if peaks are already centroided). Reference-ion
    detection and tolerance fitting are run *per dataset* so each gets its own
    drift estimate; if ``params.recalibrate`` is True, msiwarp is also run per
    dataset before pooling.
    """
    if not datasets:
        raise ValueError("align_cohort: empty dataset list")
    params = params or CohortAlignParams()

    rng = np.random.default_rng(int(params.rng_seed))

    # --- step 1: per-dataset preprocessing --------------------------------------
    prepped: list[MSIDataset] = []
    for di, ds in enumerate(datasets):
        if not isinstance(ds.backend, PeakList):
            raise ValueError(
                f"dataset {di}: align_cohort requires PeakList-backed inputs "
                f"(got {type(ds.backend).__name__})."
            )
        logger.info("cohort step 1/3: dataset %d/%d preprocessing", di + 1, len(datasets))
        d = ds
        d = DetectReferenceIons().apply(
            d, DetectReferenceIons().default_params(d.metadata), rng=rng
        ).dataset
        d = EmpiricalToleranceFromReferenceIons().apply(
            d,
            EmpiricalToleranceParams(alpha=0.01, bootstrap_B=200, block_bootstrap=False),
            rng=rng,
        ).dataset
        if params.recalibrate and d.metadata.instrument_family in {
            "tof_axial", "tof_reflectron", "qtof"
        }:
            d = MsiwarpRecalibrate().apply(
                d, MsiwarpRecalibrate().default_params(d.metadata), rng=rng
            ).dataset
        if params.pool_normalize:
            d = MedianNormalize().apply(
                d, MedianNormalize().default_params(d.metadata), rng=rng
            ).dataset
        # Run SNR peak pick only if the data is centroided (the same default as
        # the single-dataset pipeline). For profile data the user should
        # pre-process with cwt_peak_pick before calling align_cohort.
        if d.metadata.profile_or_centroided != "profile":
            d = SnrPeakPick().apply(
                d, SnrPeakPick().default_params(d.metadata), rng=rng
            ).dataset
        prepped.append(d)

    # --- step 2: pool peaks + run a single KDE consensus -----------------------
    logger.info("cohort step 2/3: pooling peaks and running shared KDE consensus")
    pooled_mz, pooled_int, pooled_dataset_idx, pooled_pixel_idx = _pool_peaks(prepped)
    if pooled_mz.size == 0:
        raise RuntimeError("align_cohort: no peaks survived per-dataset preprocessing.")
    shared_axis, candidate_axis = _shared_kde_consensus(
        pooled_mz, pooled_int, params=params
    )

    # --- step 3: build per-dataset matrices on the shared axis -----------------
    logger.info(
        "cohort step 3/3: building %d per-dataset PeakMatrices on shared axis (%d ions)",
        len(prepped), shared_axis.size,
    )
    aligned: list[MSIDataset] = []
    per_dataset_prev: dict[int, np.ndarray] = {}
    cohort_total_pixels = sum(d.n_pixels for d in prepped)
    cohort_carriers = np.zeros(shared_axis.size, dtype=np.int64)
    for di, d in enumerate(prepped):
        matrix = _assign_peaks_to_axis(
            d, shared_axis, default_tol_ppm=params.default_tol_ppm
        )
        prev = (matrix > 0).sum(axis=0) / max(d.n_pixels, 1)
        per_dataset_prev[di] = prev.astype(np.float64, copy=False)
        cohort_carriers += (matrix > 0).sum(axis=0).astype(np.int64)
        new_pm = PeakMatrix(
            matrix=matrix.astype(np.float32, copy=False),
            mz_axis=shared_axis.astype(np.float64, copy=False),
        )
        new_extra = {**d.extra, "consensus_prevalence": prev.astype(np.float64, copy=False)}
        new_extra["cohort_dataset_index"] = di
        new_extra["cohort_size"] = len(prepped)
        new_ds = d.__class__(
            coords=d.coords,
            grid_shape=d.grid_shape,
            metadata=d.metadata,
            backend=new_pm,
            identity=d.identity,
            history=d.history,
            rois=d.rois,
            rng_seed=d.rng_seed,
            extra=new_extra,
        )
        aligned.append(new_ds)

    cohort_prev = cohort_carriers.astype(np.float64) / max(cohort_total_pixels, 1)

    diagnostics = {
        "n_datasets": float(len(datasets)),
        "n_total_pixels": float(cohort_total_pixels),
        "n_total_peaks_pooled": float(int(pooled_mz.size)),
        "n_consensus_candidates_pre_filter": float(int(candidate_axis.size)),
        "n_consensus_shared": float(int(shared_axis.size)),
        "cohort_prevalence_min": float(cohort_prev.min()),
        "cohort_prevalence_median": float(np.median(cohort_prev)),
        "cohort_prevalence_max": float(cohort_prev.max()),
    }
    return CohortAlignResult(
        aligned_datasets=aligned,
        shared_consensus_mz=shared_axis,
        cohort_prevalence=cohort_prev,
        per_dataset_prevalence=per_dataset_prev,
        diagnostics=diagnostics,
    )


def _pool_peaks(
    datasets: Sequence[MSIDataset],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every (m/z, intensity, dataset_idx, pixel_idx) tuple into flat arrays.

    pixel_idx is the within-dataset pixel index. dataset_idx is its index in
    the input list.
    """
    mz_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []
    di_parts: list[np.ndarray] = []
    px_parts: list[np.ndarray] = []
    for di, d in enumerate(datasets):
        pl: PeakList = d.backend  # type: ignore[assignment]
        mz_parts.append(np.asarray(pl.mz[:], dtype=np.float64))
        int_parts.append(np.asarray(pl.intensity[:], dtype=np.float32))
        offsets = np.asarray(pl.offsets[:])
        per_peak_pixel = np.repeat(
            np.arange(pl.n_pixels, dtype=np.int64),
            np.diff(offsets).astype(np.int64),
        )
        px_parts.append(per_peak_pixel)
        di_parts.append(np.full(per_peak_pixel.size, di, dtype=np.int64))
    return (
        np.concatenate(mz_parts),
        np.concatenate(int_parts),
        np.concatenate(di_parts),
        np.concatenate(px_parts),
    )


def _shared_kde_consensus(
    pooled_mz: np.ndarray,
    pooled_int: np.ndarray,
    *,
    params: CohortAlignParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single KDE-based consensus on pooled peaks; return (kept axis, candidates).

    Same algorithm as ``KdeConsensusAlignment`` but without the per-pixel
    matrix construction (that's per-dataset, done separately). The shape of
    the candidate-axis selection is identical so cohort and single-dataset
    behaviour line up exactly when ``len(datasets) == 1``.
    """
    from dapple.ops.consensus import _kde_eval_1d, _local_maxima

    log_mz = np.log(pooled_mz)
    weights = pooled_int.astype(np.float64)
    bw = max(params.bandwidth_ppm * params.bandwidth_scale * 1e-6, 1e-12)

    pad = 5.0 * bw
    log_lo = float(log_mz.min()) - pad
    log_hi = float(log_mz.max()) + pad
    if log_hi <= log_lo:
        log_hi = log_lo + 1e-6
    grid_log = np.linspace(log_lo, log_hi, num=int(params.n_grid_points))
    density = _kde_eval_1d(log_mz, weights, grid_log, bw)

    all_max_idx = _local_maxima(density, threshold=-np.inf)
    if all_max_idx.size == 0:
        all_max_idx = np.array([int(density.argmax())])
    prominence_thr = float(np.quantile(density, params.min_prominence_quantile))
    keep = density[all_max_idx] > prominence_thr
    candidate_idx = all_max_idx[keep] if all_max_idx.size else all_max_idx
    if candidate_idx.size == 0:
        candidate_idx = np.array([int(density.argmax())])

    candidate_mz = np.exp(grid_log[candidate_idx])
    return candidate_mz, np.exp(grid_log[all_max_idx])


def _assign_peaks_to_axis(
    ds: MSIDataset,
    shared_axis: np.ndarray,
    *,
    default_tol_ppm: float,
) -> np.ndarray:
    """Bin one dataset's peaks onto a pre-computed shared m/z axis.

    For each peak, find the nearest shared-axis m/z; if the peak falls within
    a ``tol_ppm`` window of that axis value, assign it (max-aggregate).
    """
    pl: PeakList = ds.backend  # type: ignore[assignment]
    n_pixels = ds.n_pixels
    n_axis = shared_axis.size
    matrix = np.zeros((n_pixels, n_axis), dtype=np.float32)

    offsets = np.asarray(pl.offsets[:])
    peak_mz = np.asarray(pl.mz[:])
    peak_int = np.asarray(pl.intensity[:])
    if peak_mz.size == 0:
        return matrix
    peak_pixel = np.repeat(
        np.arange(n_pixels, dtype=np.int64), np.diff(offsets).astype(np.int64)
    )
    # Use the upstream tolerance curve if attached, else default_tol_ppm.
    tol_curve = ds.extra.get("tolerance_curve")
    if tol_curve is not None:
        axis_tol_ppm = tol_curve.evaluate(shared_axis)
    else:
        axis_tol_ppm = np.full(n_axis, default_tol_ppm)
    lo = shared_axis * (1 - axis_tol_ppm * 1e-6)
    hi = shared_axis * (1 + axis_tol_ppm * 1e-6)

    sorted_axis = np.argsort(shared_axis)
    sorted_mz = shared_axis[sorted_axis]
    idx_right = np.searchsorted(sorted_mz, peak_mz)
    idx_left = idx_right - 1
    for which in (idx_left, idx_right):
        valid = (which >= 0) & (which < n_axis)
        if not valid.any():
            continue
        sel = which[valid]
        in_window = (peak_mz[valid] >= lo[sorted_axis[sel]]) & (
            peak_mz[valid] <= hi[sorted_axis[sel]]
        )
        if not in_window.any():
            continue
        sel = sel[in_window]
        sel_orig_axis = sorted_axis[sel]
        np.maximum.at(
            matrix,
            (peak_pixel[valid][in_window], sel_orig_axis),
            peak_int[valid][in_window],
        )
    return matrix


# ---- directory loader ---------------------------------------------------------


def load_cohort_directory(
    root: Path | str,
    *,
    pattern: str = "*.imzML",
    recursive: bool = False,
) -> list[MSIDataset]:
    """Load every dataset matching ``pattern`` under ``root`` as a list of
    MSIDatasets, in sorted-filename order. ``recursive`` walks subdirectories.

    Useful for cohort runs where the inputs are several imzML files in a folder.
    For multi-file CDF imaging cohorts, pass each cohort member's directory
    explicitly to ``align_cohort`` rather than using this helper.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    out: list[MSIDataset] = []
    for f in files:
        if f.suffix.lower() in {".imzml", ".ibd"}:
            out.append(read_imzml(f))
        elif f.suffix.lower() in {".cdf", ".nc"}:
            from dapple.io.cdf_reader import read_cdf

            out.append(read_cdf(f))
        else:
            raise ValueError(f"don't know how to load {f}")
    return out
