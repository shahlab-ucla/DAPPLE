"""DBSCAN-based consensus alignment.

A density-based alternative to ``kde_consensus_alignment``. Where KDE pools
peaks into a continuous m/z density and finds local maxima, DBSCAN clusters the
peaks directly: any group of ≥ ``min_samples`` peaks sitting within ``eps_ppm``
of each other on the log-m/z axis becomes one consensus channel.

This is the right choice when:

- The peak distribution is sparse (few hundred peaks across the dataset) and
  KDE bandwidth selection becomes unstable.
- You want a hard, parameter-driven definition of a consensus peak rather than
  a quantile of a smooth density.
- The dataset has bimodal peak shapes (centroids + shoulders) that confuse the
  KDE's local-max finder.

DBSCAN doesn't require a prior on bandwidth, but it does require ``eps_ppm`` —
the radius (in ppm) within which two peaks count as neighbors. Pick this from
the empirical tolerance curve attached upstream by
``empirical_tolerance_from_reference_ions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
from dapple.ops.tolerance import ToleranceCurve


@dataclass(frozen=True)
class DbscanConsensusParams(OpParams):
    eps_ppm: float = field(
        default=0.0,
        metadata={
            "label": "Cluster radius (ppm)",
            "help": (
                "DBSCAN ``eps`` in ppm: two peaks are neighbors if their m/z "
                "differ by less than this fraction. Default 0 means 'use 2× the "
                "median empirical tolerance from the upstream tolerance curve' — "
                "tracks the data's actual scatter without committing to a "
                "constant-ppm assumption. Set explicitly (e.g. 50) to override."
            ),
        },
    )
    min_samples: int = field(
        default=5,
        metadata={
            "label": "Minimum peaks per cluster",
            "help": (
                "DBSCAN ``min_samples``. A peak is a 'core point' if it has at "
                "least this many neighbors within ``eps_ppm``. Lower → more "
                "small clusters survive (sensitive); higher → only widespread "
                "clusters become consensus peaks (specific). Default 5."
            ),
        },
    )
    min_prevalence: float = field(
        default=0.05,
        metadata={
            "label": "Drop peaks present in < this fraction of pixels",
            "help": (
                "Same prevalence floor as ``kde_consensus_alignment``: discard "
                "consensus channels observed in fewer than this fraction of "
                "pixels. Default 0.05."
            ),
        },
    )


@register
class DbscanConsensusAlignment(Operator):
    name = "dbscan_consensus"
    params_cls = DbscanConsensusParams

    def default_params(self, ep: ExperimentParams) -> DbscanConsensusParams:
        return DbscanConsensusParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: DbscanConsensusParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        from sklearn.cluster import DBSCAN  # imported lazily — sklearn is otherwise unused

        if not isinstance(ds.backend, PeakList):
            raise NotImplementedError(
                "dbscan_consensus requires a PeakList backend (it produces a "
                "PeakMatrix)."
            )
        pl = ds.backend
        offsets = np.asarray(pl.offsets[:])
        peak_mz = np.asarray(pl.mz[:])
        peak_int = np.asarray(pl.intensity[:])
        n_pixels = ds.n_pixels
        if peak_mz.size == 0:
            raise RuntimeError("dataset has no peaks; cannot compute consensus alignment.")

        # Resolve eps in log-m/z. log(1 + ppm·1e-6) ≈ ppm·1e-6 for small ppm.
        if params.eps_ppm > 0:
            eps_log = float(np.log1p(params.eps_ppm * 1e-6))
        else:
            tol_curve: ToleranceCurve | None = ds.extra.get("tolerance_curve")
            if tol_curve is not None:
                # 2× the median tolerance — covers most scatter without merging unrelated peaks.
                med_ppm = float(np.median(tol_curve.ppm_quantile))
                eps_log = float(np.log1p(2 * med_ppm * 1e-6))
            else:
                # No upstream curve — fall back to 50 ppm.
                eps_log = float(np.log1p(50.0 * 1e-6))

        # Cluster peaks on log m/z. DBSCAN expects 2D input → reshape.
        log_mz = np.log(peak_mz).reshape(-1, 1)
        clusterer = DBSCAN(
            eps=eps_log, min_samples=int(params.min_samples), metric="euclidean"
        )
        labels = clusterer.fit_predict(log_mz)
        # ``-1`` is DBSCAN's noise label; we drop those peaks.
        unique_labels = np.unique(labels[labels >= 0])
        if unique_labels.size == 0:
            raise RuntimeError(
                f"DBSCAN produced no clusters at eps_ppm={params.eps_ppm} / "
                f"min_samples={params.min_samples}. Loosen eps_ppm or lower "
                f"min_samples."
            )

        # Compute per-cluster intensity-weighted centroid m/z.
        cluster_centroid = np.zeros(unique_labels.size, dtype=np.float64)
        for ci, lbl in enumerate(unique_labels):
            mask = labels == lbl
            mzs = peak_mz[mask]
            ints = peak_int[mask].astype(np.float64)
            wsum = float(ints.sum())
            cluster_centroid[ci] = (
                float((mzs * ints).sum() / wsum) if wsum > 0 else float(mzs.mean())
            )

        # Build (n_pixels, n_clusters) matrix: max intensity per (pixel, cluster).
        peak_pixel = np.repeat(
            np.arange(n_pixels, dtype=np.int64), np.diff(offsets).astype(np.int64)
        )
        # Map original DBSCAN labels to consensus column indices.
        label_to_col = {int(lbl): i for i, lbl in enumerate(unique_labels)}
        matrix = np.zeros((n_pixels, unique_labels.size), dtype=np.float32)
        # Track total peak assignments per cluster for diagnostics and the optional
        # experimental prevalence sensitivity calculation. See KDE for the same
        # accounting.
        n_peaks_per_channel = np.zeros(unique_labels.size, dtype=np.int64)
        for i, lbl in enumerate(labels):
            if lbl < 0:
                continue
            col = label_to_col[int(lbl)]
            row = int(peak_pixel[i])
            v = peak_int[i]
            if v > matrix[row, col]:
                matrix[row, col] = v
            n_peaks_per_channel[col] += 1

        # Sort by centroid m/z so downstream consumers see a monotone axis.
        order = np.argsort(cluster_centroid)
        cluster_centroid = cluster_centroid[order]
        matrix = matrix[:, order]
        n_peaks_per_channel = n_peaks_per_channel[order]

        prevalence = (matrix > 0).sum(axis=0) / max(n_pixels, 1)
        keep = prevalence >= params.min_prevalence
        if not keep.any():
            raise RuntimeError(
                f"No DBSCAN consensus peaks survived min_prevalence="
                f"{params.min_prevalence}; max prevalence found was "
                f"{prevalence.max():.4f}."
            )
        kept_centroid = cluster_centroid[keep]
        kept_matrix = matrix[:, keep]
        kept_prev = prevalence[keep]
        kept_n_peaks = n_peaks_per_channel[keep]

        peakmatrix = PeakMatrix(
            matrix=kept_matrix.astype(np.float32, copy=False), mz_axis=kept_centroid
        )
        new_extra = {
            **ds.extra,
            "consensus_prevalence": kept_prev.astype(np.float64, copy=False),
            "consensus_n_peaks_per_channel": kept_n_peaks.astype(np.int64, copy=False),
        }
        new_ds = ds.with_backend(peakmatrix).__class__(
            coords=ds.coords,
            grid_shape=ds.grid_shape,
            metadata=ds.metadata,
            backend=peakmatrix,
            identity=ds.identity,
            history=ds.history,
            rois=ds.rois,
            rng_seed=ds.rng_seed,
            extra=new_extra,
        )

        diag = Diagnostic(
            name=self.name,
            summary={
                "n_consensus_peaks": float(int(kept_centroid.size)),
                "n_clusters_pre_filter": float(int(unique_labels.size)),
                "n_noise_peaks": float(int((labels < 0).sum())),
                "fraction_noise": float((labels < 0).mean()),
                "eps_log_mz": float(eps_log),
                "eps_ppm_effective": float((np.exp(eps_log) - 1) * 1e6),
                "min_samples": float(params.min_samples),
                "prevalence_min": float(kept_prev.min()),
                "prevalence_median": float(np.median(kept_prev)),
                "prevalence_max": float(kept_prev.max()),
            },
            payload={
                "cluster_centroid": kept_centroid,
                "cluster_prevalence": kept_prev,
                "labels": labels,
            },
            figure_hint="scatter:dbscan_clusters",
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
