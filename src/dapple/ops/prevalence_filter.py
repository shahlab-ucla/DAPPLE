"""Experimental occupancy sensitivity filter for consensus prevalence.

This operator simulates the occupancy distribution obtained by placing a
channel's ``k_c`` assigned peaks independently and uniformly across pixels,
then applies a right-tail Monte-Carlo test and Benjamini-Hochberg adjustment.
It is retained for exploratory sensitivity analysis and is disabled by
default.

Important statistical limitation
--------------------------------

This is *not* a calibrated inferential replacement for ``min_prevalence``.
Consensus construction and peak picking constrain the assignments used to
define a channel. In the common case where a channel has at most one assigned
peak per carrier pixel, ``k_c`` is equal (or close) to the number of occupied
pixels. The with-replacement occupancy null allows collisions, so the observed
collision-free assignments can appear spuriously significant even when the
set of carrier pixels is random. Selection of the channel on the same data
introduces an additional post-selection bias.

For each channel ``c`` the implementation:

1. measures ``p_obs(c) = (matrix[:, c] > 0).sum() / n_pixels``;
2. reads ``k_c`` from ``consensus_n_peaks_per_channel``;
3. simulates B with-replacement placements of those peaks across pixels;
4. computes a stabilized empirical right-tail p-value; and
5. applies Benjamini-Hochberg adjustment across channels.

Use a declared ``min_prevalence`` threshold for single-image filtering, and
dataset-level prevalence in cohort harmonization for cross-sample evidence.
If this operator is enabled, treat its q-values only as sensitivity scores,
report that choice, and verify conclusions across multiple thresholds. For
very small datasets the operator passes through without filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dapple.data.dataset import (
    MSIDataset,
    PeakMatrix,
    subset_channel_aligned_extra,
)
from dapple.data.metadata import ExperimentParams
from dapple.ops.base import (
    Diagnostic,
    OpParams,
    OpResult,
    Operator,
    merge_op_record,
    register,
)
from dapple.ops.spatial_filter import _bh_fdr


@dataclass(frozen=True)
class PrevalenceFdrParams(OpParams):
    n_permutations: int = field(
        default=499,
        metadata={
            "label": "Permutation count",
            "help": (
                "Number of Monte-Carlo permutations used to estimate the null "
                "occupancy sensitivity distribution per channel. Default 499 "
                "gives score resolution of ~0.002. These scores are not "
                "calibrated inferential p-values; see the operator warning."
            ),
        },
    )
    q_threshold: float = field(
        default=0.05,
        metadata={
            "label": "BH-FDR q threshold",
            "help": (
                "Channels with Benjamini-Hochberg-adjusted q-value above this "
                "sensitivity cutoff are dropped. The conventional 0.05 default "
                "does not make this occupancy model a calibrated hypothesis "
                "test. Prefer fixed min_prevalence for production workflows."
            ),
        },
    )
    min_pixels_for_test: int = field(
        default=64,
        metadata={
            "label": "Minimum pixels for the test",
            "help": (
                "Skip the test (keep every channel as-is) when fewer than this "
                "many pixels are populated. With very few pixels the occupancy "
                "null distribution is too coarse to discriminate. Default 64."
            ),
        },
    )
    rng_seed: int = field(
        default=0,
        metadata={
            "label": "RNG seed",
            "help": (
                "Mixed with the global pipeline seed for reproducible "
                "permutations. Don't change unless you want a fresh roll."
            ),
        },
    )

    def __post_init__(self) -> None:
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be at least 1")
        if not 0 < self.q_threshold <= 1:
            raise ValueError("q_threshold must be in (0, 1]")
        if self.min_pixels_for_test < 1:
            raise ValueError("min_pixels_for_test must be at least 1")


@register
class PrevalenceFdrFilter(Operator):
    """Experimental occupancy-based prevalence sensitivity filter."""

    name = "prevalence_fdr_filter"
    params_cls = PrevalenceFdrParams

    def default_params(self, ep: ExperimentParams) -> PrevalenceFdrParams:
        return PrevalenceFdrParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return [
            "Experimental only: the with-replacement occupancy null is not "
            "calibrated after peak picking and consensus assignment and can be "
            "anti-conservative. Prefer a declared min_prevalence threshold or "
            "cohort dataset prevalence; interpret q-values as sensitivity "
            "scores, not confirmatory FDR."
        ]

    def apply(
        self,
        ds: MSIDataset,
        params: PrevalenceFdrParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakMatrix):
            raise RuntimeError(
                "prevalence_fdr_filter requires a PeakMatrix backend (post "
                "consensus alignment)."
            )
        n_pixels = ds.n_pixels
        if n_pixels < int(params.min_pixels_for_test):
            return _passthrough(
                ds, params, self.name,
                f"only {n_pixels} pixels — below min_pixels_for_test "
                f"({params.min_pixels_for_test}); occupancy null is too coarse.",
            )

        pm: PeakMatrix = ds.backend
        matrix = np.asarray(pm.matrix[:])
        n_channels = matrix.shape[1]
        if n_channels == 0:
            return _passthrough(ds, params, self.name, "no channels to test")

        # Preserve integer carrier counts for the Monte-Carlo comparison.
        observed_carriers = (matrix > 0).sum(axis=0).astype(np.int64)
        p_obs = observed_carriers.astype(np.float64) / max(n_pixels, 1)

        # Per-channel total peak count (the null's ball count). Recorded by the
        # consensus operator upstream. If it is missing (for example in a
        # legacy dataset), fall back to the observed carrier count. This keeps
        # the exploratory operator runnable but does not restore calibration.
        k_per = ds.extra.get("consensus_n_peaks_per_channel")
        if k_per is None or len(k_per) != n_channels:
            k_per = np.maximum(observed_carriers, 1)
            note = (
                "consensus_n_peaks_per_channel missing — falling back to "
                "observed carrier count; occupancy scores remain uncalibrated."
            )
        else:
            k_per = np.asarray(k_per, dtype=np.int64)
            note = ""

        # Permutation null. Count distinct sorted draws rather than allocating
        # a dense (B, n_pixels) presence matrix. This keeps memory proportional
        # to sampled peaks instead of image size for sparse MSI images.
        B = int(params.n_permutations)
        seed = int(rng.integers(0, 2**31 - 1)) ^ int(params.rng_seed)
        prng = np.random.default_rng(seed)
        n_extreme = np.zeros(n_channels, dtype=np.int64)
        # For each channel, count how many samples have occupancy at least as
        # large as the observed carrier count.
        for c in range(n_channels):
            k = int(k_per[c])
            if k <= 0:
                # No peaks at all — the observed prevalence is 0, the null
                # prevalence is also 0. p-value = 1 (always extreme).
                n_extreme[c] = B
                continue
            # Sample uniform pixel indices in memory-bounded replicate chunks.
            if k == 1:
                n_extreme[c] = B if observed_carriers[c] <= 1 else 0
                continue

            # Sorting needs one integer draw array and one adjacent-difference
            # boolean array. Keep their combined working set near 32 MiB. A
            # single replicate may exceed that target when k itself is huge,
            # but memory no longer scales with the image's pixel count.
            draw_dtype = (
                np.int32 if n_pixels <= np.iinfo(np.int32).max else np.int64
            )
            bytes_per_draw = (
                np.dtype(draw_dtype).itemsize + np.dtype(bool).itemsize
            )
            target_bytes = 32 * 1024 * 1024
            chunk = max(
                1,
                min(B, target_bytes // max(k * bytes_per_draw, 1)),
            )
            done = 0
            extreme = 0
            while done < B:
                b = min(chunk, B - done)
                pixels = prng.integers(
                    0, n_pixels, size=(b, k), dtype=draw_dtype
                )
                pixels.sort(axis=1)
                distinct = 1 + np.count_nonzero(
                    pixels[:, 1:] != pixels[:, :-1], axis=1
                )
                extreme += int(
                    np.count_nonzero(distinct >= observed_carriers[c])
                )
                done += b
            n_extreme[c] = extreme

        # Empirical right-tail p-value with the +1/+1 stabilizer (so p_value
        # is never exactly 0, which would crash BH-FDR's downstream log).
        p_values = (n_extreme + 1).astype(np.float64) / (B + 1)
        q_values = _bh_fdr(p_values)
        keep = q_values < float(params.q_threshold)
        if not keep.any():
            raise RuntimeError(
                f"prevalence_fdr_filter: every channel rejected at q < "
                f"{params.q_threshold}. No channel passed this experimental "
                "sensitivity cutoff; use the declared min_prevalence result or "
                "raise the cutoff for sensitivity comparison."
            )

        # Trim the matrix to surviving channels.
        new_matrix = matrix[:, keep].astype(np.float32, copy=False)
        new_axis = np.asarray(pm.mz_axis[:])[keep].astype(np.float64, copy=False)
        new_pm = PeakMatrix(matrix=new_matrix, mz_axis=new_axis)

        # Subset every known channel-aligned companion array to match.
        new_extra = subset_channel_aligned_extra(
            ds.extra,
            keep,
            n_channels,
            replacements={
                "prevalence_fdr_p_values": p_values[keep],
                "prevalence_fdr_q_values": q_values[keep],
            },
        )
        new_extra["prevalence_fdr_dropped_n"] = int((~keep).sum())

        new_ds = ds.__class__(
            coords=ds.coords,
            grid_shape=ds.grid_shape,
            metadata=ds.metadata,
            backend=new_pm,
            identity=ds.identity,
            history=ds.history,
            rois=ds.rois,
            rng_seed=ds.rng_seed,
            extra=new_extra,
        )

        summary = {
            "n_channels_in": float(n_channels),
            "n_channels_out": float(int(keep.sum())),
            "n_dropped_by_fdr": float(int((~keep).sum())),
            "n_permutations": float(B),
            "q_threshold": float(params.q_threshold),
            "p_value_min": float(p_values.min()),
            "p_value_median": float(np.median(p_values)),
            "q_value_min": float(q_values.min()),
            "q_value_median": float(np.median(q_values)),
            "warning_uncalibrated_occupancy_null": 1.0,
        }
        if note:
            summary["warning_conservative_fallback"] = 1.0
        diag = Diagnostic(
            name=self.name,
            summary=summary,
            payload={
                "p_values": p_values,
                "q_values": q_values,
                "kept_mask": keep.astype(bool),
                "channel_mz_in": np.asarray(pm.mz_axis[:]),
                "k_per_channel": k_per.astype(np.int64, copy=False),
                "p_obs": p_obs,
                "model_note": np.array(
                    [
                        "Experimental sensitivity score: the with-replacement "
                        "occupancy null is not calibrated after channel selection."
                    ],
                    dtype=object,
                ),
                **({"note": np.array([note], dtype=object)} if note else {}),
            },
            figure_hint="histogram:prevalence_fdr_q_values",
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


def _passthrough(
    ds: MSIDataset,
    params: PrevalenceFdrParams,
    op_name: str,
    note: str,
) -> OpResult:
    """Skip the test entirely and forward the input dataset unmodified."""
    n_channels = int(getattr(ds.backend, "n_peaks", 0))
    diag = Diagnostic(
        name=op_name,
        summary={
            "n_channels_in": float(n_channels),
            "n_channels_out": float(n_channels),
            "n_dropped_by_fdr": 0.0,
            "n_permutations": float(params.n_permutations),
            "q_threshold": float(params.q_threshold),
            "warning_uncalibrated_occupancy_null": 1.0,
        },
        payload={
            "note": np.array([note], dtype=object),
            "model_note": np.array(
                [
                    "Experimental sensitivity score: the with-replacement "
                    "occupancy null is not calibrated after channel selection."
                ],
                dtype=object,
            ),
        },
    )
    record = merge_op_record(
        op_name=op_name, params=params, input_ds=ds, output_ds=ds, diagnostics=[diag]
    )
    return OpResult(dataset=ds.with_history(record), diagnostics=[diag])
