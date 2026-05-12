"""Permutation-null FDR test for the prevalence filter.

The single-dataset consensus operators (``kde_consensus_alignment``,
``dbscan_consensus``) ship with a conservative ``min_prevalence`` floor —
channels whose fraction of non-zero pixels is below the floor are dropped
without further test. This module replaces that thresholding with an
**empirical permutation-null test** for "is this channel's prevalence higher
than would be expected if peaks were randomly distributed across pixels?".

Why permutation null
--------------------

The natural one-sample null for a channel's prevalence is the **occupancy
problem**: given ``k_c`` peaks placed independently and uniformly across
``n_pixels`` pixels, what's the distribution of "number of distinct pixels
hit"? An exact closed form exists (a Stirling-numbers identity) but is
numerically unstable for the (k_c, n_pixels) regime we care about. A
Monte-Carlo permutation gives the same answer with arbitrary precision in O(B
* k_c) per channel — usually milliseconds for sane parameters.

For each channel ``c``:

1. Pull its observed prevalence ``p_obs(c) = (matrix[:, c] > 0).sum() / n_pixels``.
2. Pull its observed peak count ``k_c`` from ``ds.extra["consensus_n_peaks_per_channel"]``
   (recorded by the consensus operator upstream).
3. Simulate B permutations: place ``k_c`` peaks into ``n_pixels`` bins uniformly
   at random; count distinct bins; convert to prevalence ``p_perm``.
4. Empirical right-tail p-value with the +1/+1 stabilizer:
   ``p_value(c) = (#{p_perm >= p_obs} + 1) / (B + 1)``.
5. Benjamini–Hochberg adjustment across all channels.
6. Drop channels with ``q_value >= q_threshold``.

This is the empirical analog of "channel c is real if its peaks are more
broadly shared across the image than random placement would produce".

When to use
-----------

- After consensus alignment, on any PeakMatrix-backed dataset.
- Particularly useful when ``min_prevalence`` is hard to set: the FDR test
  adapts to ``k_c`` per channel, so a channel with many peaks needs higher
  prevalence to look "non-random" than one with few peaks. The same fixed
  ``min_prevalence`` over-rejects sparse channels and under-rejects dense
  ones.

Caveats
-------

- The null assumes peaks are placed independently across pixels. If your
  pipeline already drops noise channels via Moran's I, run that *after* this
  filter, not before — Moran's I needs spatially structured channels to
  detect, and this filter removes the spatially-unstructured ones first.
- For very small datasets (n_pixels < ~64) the null distribution is too
  coarse to discriminate; the operator falls through as a passthrough in that
  regime (same convention as ``morans_i_permutation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dapple.data.dataset import MSIDataset, PeakMatrix
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
                "occupancy distribution per channel. Default 499 gives p-value "
                "resolution of ~0.002 — comfortable for a BH-FDR cut across a "
                "few hundred channels. Drop to 99 for fast iteration; raise to "
                "1999 if the channel count exceeds ~1000 and you need tighter "
                "p-values."
            ),
        },
    )
    q_threshold: float = field(
        default=0.05,
        metadata={
            "label": "BH-FDR q threshold",
            "help": (
                "Channels with Benjamini-Hochberg-adjusted q-value above this "
                "threshold are dropped. Default 0.05 (the standard FDR floor). "
                "Raise to 0.1 to be permissive; lower to 0.01 to keep only "
                "channels whose prevalence is strongly above the random-placement "
                "expectation."
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


@register
class PrevalenceFdrFilter(Operator):
    """Drop consensus channels whose prevalence is indistinguishable from random."""

    name = "prevalence_fdr_filter"
    params_cls = PrevalenceFdrParams

    def default_params(self, ep: ExperimentParams) -> PrevalenceFdrParams:
        return PrevalenceFdrParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return []

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

        # Observed prevalence per channel.
        p_obs = (matrix > 0).sum(axis=0).astype(np.float64) / max(n_pixels, 1)

        # Per-channel total peak count (the null's ball count). Recorded by the
        # consensus operator upstream. If it's missing (e.g. legacy dataset),
        # fall back to using the observed pixel count — yields a conservative
        # test (the null with k_c = pixel-count produces approximately the same
        # prevalence as observed, so p-values will be conservatively near 0.5).
        k_per = ds.extra.get("consensus_n_peaks_per_channel")
        if k_per is None or len(k_per) != n_channels:
            k_per = np.maximum(
                (matrix > 0).sum(axis=0).astype(np.int64), 1
            )
            note = (
                "consensus_n_peaks_per_channel missing — falling back to "
                "observed pixel count; test is conservative."
            )
        else:
            k_per = np.asarray(k_per, dtype=np.int64)
            note = ""

        # Permutation null. For each channel c we sample B occupancy values
        # from the discrete distribution "k_c balls in n_pixels bins, count
        # distinct bins". Vectorized per channel: allocate a (B, n_pixels)
        # presence bool, scatter sampled pixel indices, sum along axis 1.
        B = int(params.n_permutations)
        seed = int(rng.integers(0, 2**31 - 1)) ^ int(params.rng_seed)
        prng = np.random.default_rng(seed)
        n_extreme = np.zeros(n_channels, dtype=np.int64)
        # For each channel, count how many of B perm samples produce a
        # prevalence >= the observed one.
        for c in range(n_channels):
            k = int(k_per[c])
            if k <= 0:
                # No peaks at all — the observed prevalence is 0, the null
                # prevalence is also 0. p-value = 1 (always extreme).
                n_extreme[c] = B
                continue
            # Sample (B, k) uniform pixel indices; mark presence; count distinct.
            # Memory-bounded chunking when B*k would exceed ~50M (e.g. very
            # peaky channels): split B into chunks.
            chunk = max(1, min(B, max(1, 50_000_000 // max(k, 1))))
            distinct = np.empty(B, dtype=np.int64)
            done = 0
            while done < B:
                b = min(chunk, B - done)
                pixels = prng.integers(0, n_pixels, size=(b, k), dtype=np.int64)
                # presence[row, pixel] = True; sum gives distinct pixel count per row.
                presence = np.zeros((b, n_pixels), dtype=bool)
                row_idx = np.repeat(np.arange(b), k)
                # Using boolean OR semantics via direct assignment is fine since
                # we only care about the True state — duplicate assignments are
                # idempotent.
                presence[row_idx, pixels.ravel()] = True
                distinct[done:done + b] = presence.sum(axis=1)
                done += b
            p_perm = distinct.astype(np.float64) / max(n_pixels, 1)
            n_extreme[c] = int(np.sum(p_perm >= p_obs[c]))

        # Empirical right-tail p-value with the +1/+1 stabilizer (so p_value
        # is never exactly 0, which would crash BH-FDR's downstream log).
        p_values = (n_extreme + 1).astype(np.float64) / (B + 1)
        q_values = _bh_fdr(p_values)
        keep = q_values < float(params.q_threshold)
        if not keep.any():
            raise RuntimeError(
                f"prevalence_fdr_filter: every channel rejected at q < "
                f"{params.q_threshold}. The dataset has no channels with "
                "prevalence significantly above random; consider lowering "
                "q_threshold or revisiting upstream peak picking / consensus."
            )

        # Trim the matrix to surviving channels.
        new_matrix = matrix[:, keep].astype(np.float32, copy=False)
        new_axis = np.asarray(pm.mz_axis[:])[keep].astype(np.float64, copy=False)
        new_pm = PeakMatrix(matrix=new_matrix, mz_axis=new_axis)

        # Subset companion arrays in extra to match.
        new_extra = {**ds.extra}
        for key in ("consensus_prevalence", "consensus_n_peaks_per_channel"):
            v = new_extra.get(key)
            if v is not None and len(v) == n_channels:
                new_extra[key] = np.asarray(v)[keep]
        new_extra["prevalence_fdr_dropped_n"] = int((~keep).sum())
        new_extra["prevalence_fdr_p_values"] = p_values
        new_extra["prevalence_fdr_q_values"] = q_values

        new_ds = ds.with_backend(new_pm).__class__(
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
        },
        payload={"note": np.array([note], dtype=object)},
    )
    record = merge_op_record(
        op_name=op_name, params=params, input_ds=ds, output_ds=ds, diagnostics=[diag]
    )
    return OpResult(dataset=ds.with_history(record), diagnostics=[diag])
