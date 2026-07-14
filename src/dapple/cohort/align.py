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

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from dapple.data.dataset import (
    CHANNEL_ALIGNED_EXTRA_KEYS,
    MSIDataset,
    OpRecord,
    PeakList,
    PeakMatrix,
)
from dapple.data.hashing import combine_hashes, hash_obj
from dapple.io.cdf_image_reader import read_cdf_image
from dapple.io.imzml_reader import read_imzml
from dapple.ops.consensus import KdeConsensusParams
from dapple.ops.normalize import MedianNormalize
from dapple.ops.peak_pick import CwtPeakPick, SnrPeakPick
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
    pool_weighting: Literal["sample", "intensity"] = "sample"
    prevalence_basis: Literal["pixel", "dataset"] = "pixel"
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.bandwidth_ppm <= 0 or self.bandwidth_scale <= 0:
            raise ValueError("bandwidth_ppm and bandwidth_scale must be positive")
        if self.n_grid_points < 3:
            raise ValueError("n_grid_points must be at least 3")
        if not 0.0 <= self.min_prominence_quantile <= 1.0:
            raise ValueError("min_prominence_quantile must be in [0, 1]")
        if not 0.0 <= self.min_prevalence <= 1.0:
            raise ValueError("min_prevalence must be in [0, 1]")
        if self.default_tol_ppm <= 0:
            raise ValueError("default_tol_ppm must be positive")
        if self.pool_weighting not in {"sample", "intensity"}:
            raise ValueError("pool_weighting must be 'sample' or 'intensity'")
        if self.prevalence_basis not in {"pixel", "dataset"}:
            raise ValueError("prevalence_basis must be 'pixel' or 'dataset'")


@dataclass
class CohortAlignResult:
    """Output of ``align_cohort``.

    ``aligned_datasets`` parallels the input list: each entry is a new
    MSIDataset with a PeakMatrix backend whose ``mz_axis`` equals
    ``shared_consensus_mz``. Per-dataset prevalence (fraction of that dataset's
    pixels carrying each consensus peak) is stored in
    ``per_dataset_prevalence`` keyed by dataset index.

    ``cohort_prevalence`` is the pixel-level cohort prevalence — the fraction
    of cohort-wide pixels carrying each consensus peak. ``dataset_prevalence``
    is the fraction of datasets carrying it. The configured
    ``prevalence_basis`` determines which measure filters candidates.

    ``diagnostics`` is a flat dict of summary scalars suitable for logging /
    saving to a manifest.
    """

    aligned_datasets: list[MSIDataset]
    shared_consensus_mz: np.ndarray
    cohort_prevalence: np.ndarray
    dataset_prevalence: np.ndarray
    per_dataset_prevalence: dict[int, np.ndarray]
    diagnostics: dict[str, float]


def align_cohort(
    datasets: Sequence[MSIDataset],
    params: CohortAlignParams | None = None,
) -> CohortAlignResult:
    """Compute a shared consensus axis across `datasets` and return per-dataset
    PeakMatrix-backed datasets aligned to it.

    Each input dataset must have a ``PeakList`` backend. A prior DAPPLE peak-pick
    record is respected; otherwise profile inputs use CWT and centroided inputs
    use the SNR filter. Reference-ion
    detection and tolerance fitting are run *per dataset* so each gets its own
    drift estimate; if ``params.recalibrate`` is True, msiwarp is also run per
    dataset before pooling.
    """
    if not datasets:
        raise ValueError("align_cohort: empty dataset list")
    params = params or CohortAlignParams()

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
        history_ops = {record.op_name for record in d.history}
        already_peak_picked = bool(
            history_ops & {"cwt_peak_pick", "snr_peak_pick"}
        )
        if d.metadata.profile_or_centroided == "profile" and not already_peak_picked:
            # Dense profile points are not valid reference-ion candidates. Pick
            # centroids first or adjacent prevalent bins collapse into one feature.
            d = CwtPeakPick().apply(
                d,
                CwtPeakPick().default_params(d.metadata),
                rng=_dataset_rng(params.rng_seed, d, "cwt_peak_pick"),
            ).dataset
        d = DetectReferenceIons().apply(
            d,
            DetectReferenceIons().default_params(d.metadata),
            rng=_dataset_rng(params.rng_seed, d, "detect_reference_ions"),
        ).dataset
        d = EmpiricalToleranceFromReferenceIons().apply(
            d,
            EmpiricalToleranceParams(alpha=0.01, bootstrap_B=200, block_bootstrap=False),
            rng=_dataset_rng(params.rng_seed, d, "empirical_tolerance"),
        ).dataset
        if params.recalibrate and d.metadata.instrument_family in {
            "tof_axial", "tof_reflectron", "qtof"
        }:
            d = MsiwarpRecalibrate().apply(
                d,
                MsiwarpRecalibrate().default_params(d.metadata),
                rng=_dataset_rng(params.rng_seed, d, "msiwarp_recalibrate"),
            ).dataset
        if params.pool_normalize:
            d = MedianNormalize().apply(
                d,
                MedianNormalize().default_params(d.metadata),
                rng=_dataset_rng(params.rng_seed, d, "median_normalize"),
            ).dataset
        if d.metadata.profile_or_centroided != "profile" and not already_peak_picked:
            d = SnrPeakPick().apply(
                d,
                SnrPeakPick().default_params(d.metadata),
                rng=_dataset_rng(params.rng_seed, d, "snr_peak_pick"),
            ).dataset
        prepped.append(d)

    # --- step 2: pool peaks + run a single KDE consensus -----------------------
    logger.info("cohort step 2/3: pooling peaks and running shared KDE consensus")
    pooled_mz, pooled_int, pooled_dataset_idx = _pool_peaks(prepped)
    if pooled_mz.size == 0:
        raise RuntimeError("align_cohort: no peaks survived per-dataset preprocessing.")
    pooled_weights = _cohort_kde_weights(
        pooled_int,
        pooled_dataset_idx,
        n_datasets=len(prepped),
        mode=params.pool_weighting,
    )
    candidate_axis, all_local_maxima_axis = _shared_kde_consensus(
        pooled_mz, pooled_weights, params=params
    )

    # First assignment pass accumulates only carrier counts. Keeping every full
    # candidate matrix until filtering roughly doubled peak memory on large
    # cohorts; reassigning the smaller surviving axis below is deliberately
    # compute-for-memory and bounds this stage to one dataset matrix at a time.
    cohort_total_pixels = sum(d.n_pixels for d in prepped)
    candidate_carriers = np.zeros(candidate_axis.size, dtype=np.int64)
    candidate_dataset_carriers = np.zeros(candidate_axis.size, dtype=np.int64)
    for d in prepped:
        candidate_matrix = _assign_peaks_to_axis(
            d,
            candidate_axis,
            default_tol_ppm=params.default_tol_ppm,
        )
        occupied = candidate_matrix > 0
        candidate_carriers += occupied.sum(axis=0).astype(np.int64)
        candidate_dataset_carriers += occupied.any(axis=0).astype(np.int64)
    candidate_pixel_prev = candidate_carriers.astype(np.float64) / max(
        cohort_total_pixels, 1
    )
    candidate_dataset_prev = candidate_dataset_carriers.astype(np.float64) / len(prepped)
    filter_prev = (
        candidate_pixel_prev
        if params.prevalence_basis == "pixel"
        else candidate_dataset_prev
    )
    prevalence_keep = filter_prev >= params.min_prevalence
    if not prevalence_keep.any():
        raise RuntimeError(
            "align_cohort: no shared channels survived "
            f"{params.prevalence_basis} prevalence >= {params.min_prevalence}; "
            f"maximum was {float(filter_prev.max()):.4f}."
        )
    shared_axis = candidate_axis[prevalence_keep]

    # --- step 3: build per-dataset matrices on the shared axis -----------------
    logger.info(
        "cohort step 3/3: building %d per-dataset PeakMatrices on shared axis (%d ions)",
        len(prepped), shared_axis.size,
    )
    aligned: list[MSIDataset] = []
    per_dataset_prev: dict[int, np.ndarray] = {}
    cohort_carriers = np.zeros(shared_axis.size, dtype=np.int64)
    dataset_carriers = np.zeros(shared_axis.size, dtype=np.int64)
    cohort_record = _cohort_history_record(
        params=params,
        inputs=prepped,
        shared_axis=shared_axis,
    )
    for di, d in enumerate(prepped):
        matrix = _assign_peaks_to_axis(
            d,
            shared_axis,
            default_tol_ppm=params.default_tol_ppm,
        )
        occupied = matrix > 0
        prev = occupied.sum(axis=0) / max(d.n_pixels, 1)
        per_dataset_prev[di] = prev.astype(np.float64, copy=False)
        cohort_carriers += occupied.sum(axis=0).astype(np.int64)
        dataset_carriers += occupied.any(axis=0).astype(np.int64)
        new_pm = PeakMatrix(
            matrix=matrix.astype(np.float32, copy=False),
            mz_axis=shared_axis.astype(np.float64, copy=False),
        )
        # The shared cohort axis is unrelated to each input dataset's former
        # channel axis.  Drop every old channel-aligned companion array before
        # attaching statistics computed for the shared axis.
        new_extra = {
            key: value
            for key, value in d.extra.items()
            if key not in CHANNEL_ALIGNED_EXTRA_KEYS
        }
        new_extra.update(
            {
                "consensus_prevalence": prev.astype(np.float64, copy=False),
                "cohort_prevalence": candidate_pixel_prev[prevalence_keep],
                "cohort_dataset_prevalence": candidate_dataset_prev[prevalence_keep],
                "cohort_dataset_index": di,
                "cohort_size": len(prepped),
            }
        )
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
        new_ds = new_ds.with_history(cohort_record)
        aligned.append(new_ds)

    cohort_prev = cohort_carriers.astype(np.float64) / max(cohort_total_pixels, 1)
    dataset_prev = dataset_carriers.astype(np.float64) / max(len(prepped), 1)

    diagnostics = {
        "n_datasets": float(len(datasets)),
        "n_total_pixels": float(cohort_total_pixels),
        "n_total_peaks_pooled": float(int(pooled_mz.size)),
        "n_local_maxima_total": float(int(all_local_maxima_axis.size)),
        "n_consensus_candidates_pre_filter": float(int(candidate_axis.size)),
        "n_rejected_by_prevalence": float(int((~prevalence_keep).sum())),
        "n_consensus_shared": float(int(shared_axis.size)),
        "cohort_prevalence_min": float(cohort_prev.min()),
        "cohort_prevalence_median": float(np.median(cohort_prev)),
        "cohort_prevalence_max": float(cohort_prev.max()),
        "dataset_prevalence_min": float(dataset_prev.min()),
        "dataset_prevalence_median": float(np.median(dataset_prev)),
        "dataset_prevalence_max": float(dataset_prev.max()),
        "min_prevalence_threshold": float(params.min_prevalence),
        "prevalence_basis_dataset": float(params.prevalence_basis == "dataset"),
    }
    return CohortAlignResult(
        aligned_datasets=aligned,
        shared_consensus_mz=shared_axis,
        cohort_prevalence=cohort_prev,
        dataset_prevalence=dataset_prev,
        per_dataset_prevalence=per_dataset_prev,
        diagnostics=diagnostics,
    )


def _pool_peaks(
    datasets: Sequence[MSIDataset],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``(m/z, intensity, dataset_idx)`` arrays for pooled KDE."""
    mz_parts: list[np.ndarray] = []
    int_parts: list[np.ndarray] = []
    di_parts: list[np.ndarray] = []
    for di, d in enumerate(datasets):
        pl: PeakList = d.backend  # type: ignore[assignment]
        mz_parts.append(np.asarray(pl.mz[:], dtype=np.float64))
        int_parts.append(np.asarray(pl.intensity[:], dtype=np.float32))
        di_parts.append(np.full(len(pl.mz), di, dtype=np.int64))
    return (
        np.concatenate(mz_parts),
        np.concatenate(int_parts),
        np.concatenate(di_parts),
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
    from dapple.ops.consensus import _adaptive_kde_grid, _kde_eval_1d, _local_maxima

    log_mz = np.log(pooled_mz)
    weights = pooled_int.astype(np.float64)
    bw = max(params.bandwidth_ppm * params.bandwidth_scale * 1e-6, 1e-12)

    grid_log, _grid_was_capped = _adaptive_kde_grid(
        log_mz,
        bw,
        minimum_points=int(params.n_grid_points),
    )
    density = _kde_eval_1d(log_mz, weights, grid_log, bw)

    all_max_idx = _local_maxima(density, threshold=-np.inf)
    if all_max_idx.size == 0:
        all_max_idx = np.array([int(density.argmax())])
    positive_density = density[density > 0]
    prominence_thr = float(
        np.quantile(
            positive_density if positive_density.size else density,
            params.min_prominence_quantile,
        )
    )
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
    from dapple.ops.consensus import _nearest_window_assignments

    peak_idx, axis_idx = _nearest_window_assignments(
        peak_mz,
        shared_axis,
        np.asarray(axis_tol_ppm, dtype=np.float64),
    )
    np.maximum.at(
        matrix,
        (peak_pixel[peak_idx], axis_idx),
        peak_int[peak_idx],
    )
    return matrix


def _dataset_rng(seed: int, ds: MSIDataset, stage: str) -> np.random.Generator:
    """Derive order-independent per-dataset/per-stage randomness."""
    token = hash_obj(
        {
            "seed": int(seed),
            "dataset": ds.identity.content_sha256,
            "stage": stage,
        }
    )
    return np.random.default_rng(int(token[:16], 16) & 0xFFFFFFFF)


def _cohort_kde_weights(
    intensities: np.ndarray,
    dataset_index: np.ndarray,
    *,
    n_datasets: int,
    mode: Literal["sample", "intensity"],
) -> np.ndarray:
    """Return raw weights or equal-total-weight sample contributions."""
    weights = np.maximum(np.asarray(intensities, dtype=np.float64), 0.0)
    if mode == "intensity":
        return weights
    balanced = np.zeros_like(weights)
    for di in range(n_datasets):
        mask = dataset_index == di
        total = float(weights[mask].sum())
        if total > 0:
            balanced[mask] = weights[mask] / total
        elif mask.any():
            balanced[mask] = 1.0 / int(mask.sum())
    return balanced


def _cohort_history_record(
    *,
    params: CohortAlignParams,
    inputs: Sequence[MSIDataset],
    shared_axis: np.ndarray,
) -> OpRecord:
    """Fingerprint cohort composition, parameters, and resulting shared axis."""
    input_hash = combine_hashes(*(ds.hash(include_rois=False) for ds in inputs))
    axis_hash = hashlib.sha256(
        np.asarray(shared_axis, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    output_hash = combine_hashes(
        "cohort_alignment",
        input_hash,
        hash_obj(params),
        axis_hash,
    )
    return OpRecord(
        op_name="cohort_alignment",
        params_hash=hash_obj(params),
        input_hash=input_hash,
        output_hash=output_hash,
        diagnostics_summary=(),
    )


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
