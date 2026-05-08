"""Spatial filtering: drop consensus channels whose pixel-level intensity pattern
is indistinguishable from random.

For each consensus channel ``c`` we compute Moran's I on the spatial intensity
distribution and compare the observed value to a permutation null. Channels
whose permutation-FDR-adjusted q-value is above the threshold are dropped —
they have no spatial coherence and are dominated by noise.

Moran's I (Cliff & Ord, 1981):

  I = (n / S0) · ((x − x̄)ᵀ W (x − x̄)) / ((x − x̄)ᵀ (x − x̄))

with W the binary adjacency matrix (queen or rook) and S0 = ΣᵢⱼWᵢⱼ. Under
spatial randomness E[I] ≈ −1/(n−1), so positive observed I indicates clustering
of similar values, negative indicates dispersion. We use a permutation null
because real MSI data has heavy-tailed intensity distributions and analytic
moments under spatial randomness assume symmetry that doesn't hold.

Output: an ``MSIDataset`` whose PeakMatrix has had non-coherent channels removed.
The per-channel Moran's I, p-value, and q-value land in the diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dapple.data.coords import coords_to_grid_index
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

Neighborhood = Literal["queen", "rook"]


@dataclass(frozen=True)
class MoransIParams(OpParams):
    n_permutations: int = field(
        default=499,
        metadata={
            "label": "Permutation count",
            "help": (
                "Number of label-permutation samples used to estimate the null "
                "distribution of Moran's I per channel. Default 499 gives "
                "p-value resolution of ~0.002 — enough headroom for an FDR cut "
                "across a few hundred channels. Drop to 99 for fast iteration; "
                "raise to 1999 for tighter p-values when N_channels > 1000."
            ),
        },
    )
    q_threshold: float = field(
        default=0.05,
        metadata={
            "label": "BH-FDR q threshold",
            "help": (
                "Channels with Benjamini–Hochberg-adjusted q-value above this "
                "threshold are dropped as spatially incoherent. Default 0.05 — "
                "the standard FDR floor. Raise to 0.1 to be permissive; lower "
                "to 0.01 to keep only channels with strong, near-certain "
                "spatial structure."
            ),
        },
    )
    neighborhood: Neighborhood = field(
        default="queen",
        metadata={
            "label": "Neighborhood",
            "help": (
                "queen: 8 neighbors per pixel (orthogonal + diagonal). rook: 4 "
                "(orthogonal only). Queen is the default — slightly more "
                "permissive at honoring fine diagonal tissue features. Rook is "
                "appropriate when pixels are anisotropic."
            ),
        },
    )
    min_pixels_for_test: int = field(
        default=64,
        metadata={
            "label": "Minimum pixels for the test",
            "help": (
                "Skip the test (keep every channel as-is) when fewer than this "
                "many pixels are populated — Moran's I is unstable on tiny "
                "images. Default 64 (an 8×8 region)."
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
class MoransIPermutation(Operator):
    name = "morans_i_permutation"
    params_cls = MoransIParams

    def default_params(self, ep: ExperimentParams) -> MoransIParams:
        return MoransIParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        # No instrument-specific contraindication.
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: MoransIParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakMatrix):
            raise RuntimeError(
                "morans_i_permutation requires a PeakMatrix backend (post "
                "consensus alignment)."
            )
        n_pixels = ds.n_pixels
        if n_pixels < int(params.min_pixels_for_test):
            return _passthrough(
                ds, params, self.name,
                f"only {n_pixels} pixels — below min_pixels_for_test ({params.min_pixels_for_test})",
            )

        # Build a sparse adjacency matrix W keyed by populated pixels.
        W = _build_adjacency(ds, params.neighborhood)
        S0 = float(W.sum())
        if S0 == 0:
            return _passthrough(ds, params, self.name, "no neighboring pixels")

        pm: PeakMatrix = ds.backend
        matrix = np.asarray(pm.matrix[:]).astype(np.float64, copy=True)
        n_channels = matrix.shape[1]

        # Vectorized observed Moran's I across all channels at once.
        x = matrix - matrix.mean(axis=0, keepdims=True)
        # numerator[c]   = (x[:,c]).T @ W @ x[:,c]
        # denominator[c] = (x[:,c]**2).sum()
        Wx = W @ x  # (n_pixels, n_channels)
        numerator = (x * Wx).sum(axis=0)
        denominator = (x ** 2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            I_obs = (n_pixels / S0) * numerator / np.where(denominator > 0, denominator, 1.0)
        I_obs = np.where(denominator > 0, I_obs, 0.0)

        # Permutation null: shuffle pixel labels, recompute I_perm. For each
        # permutation we shuffle the pixel ordering once and apply it to every
        # channel — that's a global pixel-label permutation, the standard null.
        B = int(params.n_permutations)
        seed = int(rng.integers(0, 2**31 - 1)) ^ int(params.rng_seed)
        prng = np.random.default_rng(seed)
        # Tail counts for the empirical p-value: number of permutations whose
        # |I_perm[c]| >= |I_obs[c]| (two-sided test).
        n_extreme = np.zeros(n_channels, dtype=np.int64)
        # We'll need x's variance per channel under permutation — but variance is
        # invariant to row permutation, so denominator stays constant.
        for _ in range(B):
            perm = prng.permutation(n_pixels)
            xp = x[perm, :]
            Wxp = W @ xp
            num_p = (xp * Wxp).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                I_perm = (n_pixels / S0) * num_p / np.where(denominator > 0, denominator, 1.0)
            I_perm = np.where(denominator > 0, I_perm, 0.0)
            n_extreme += np.abs(I_perm) >= np.abs(I_obs)

        # Empirical two-sided p-values, with the +1/+1 stabilizer.
        p_values = (n_extreme + 1) / (B + 1)
        # BH-FDR adjustment.
        q_values = _bh_fdr(p_values)
        keep = q_values < params.q_threshold

        if not keep.any():
            raise RuntimeError(
                f"morans_i_permutation: every channel rejected at q < "
                f"{params.q_threshold}. Either the data has no spatial structure "
                "(unlikely on tissue), the threshold is too strict, or the "
                "permutation count is too small to discriminate."
            )

        # Build a trimmed dataset.
        new_matrix = matrix[:, keep].astype(np.float32, copy=False)
        new_axis = np.asarray(pm.mz_axis[:])[keep].astype(np.float64, copy=False)
        new_pm = PeakMatrix(matrix=new_matrix, mz_axis=new_axis)
        new_extra = {**ds.extra}
        if "consensus_prevalence" in new_extra:
            old_prev = np.asarray(new_extra["consensus_prevalence"])
            if old_prev.shape == (n_channels,):
                new_extra["consensus_prevalence"] = old_prev[keep]
        new_extra["morans_i_dropped_n"] = int((~keep).sum())
        new_extra["morans_i_per_channel"] = I_obs.copy()
        new_extra["morans_i_p_values"] = p_values.copy()
        new_extra["morans_i_q_values"] = q_values.copy()
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

        diag = Diagnostic(
            name=self.name,
            summary={
                "n_channels_in": float(n_channels),
                "n_channels_out": float(int(keep.sum())),
                "n_dropped_by_fdr": float(int((~keep).sum())),
                "n_permutations": float(B),
                "I_min": float(I_obs.min()),
                "I_median": float(np.median(I_obs)),
                "I_max": float(I_obs.max()),
                "S0": S0,
                "q_threshold": float(params.q_threshold),
            },
            payload={
                "I_obs": I_obs,
                "p_values": p_values,
                "q_values": q_values,
                "channel_mz_in": np.asarray(pm.mz_axis[:]),
                "kept_mask": keep,
            },
            figure_hint="histogram:morans_i_with_threshold",
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


def _build_adjacency(ds: MSIDataset, neighborhood: Neighborhood):
    """Build a sparse symmetric adjacency matrix in scipy.sparse CSR format.

    Each populated pixel is a node; an edge exists between any two populated
    pixels that are 4- or 8-connected on the raster grid. Self-edges are
    excluded.
    """
    from scipy.sparse import csr_matrix

    h, w = ds.grid_shape
    flat = coords_to_grid_index(ds.coords, ds.grid_shape)
    grid_to_node = -np.ones(h * w, dtype=np.int64)
    grid_to_node[flat] = np.arange(ds.n_pixels, dtype=np.int64)
    iy, ix = divmod(flat, w)

    if neighborhood == "queen":
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for dy, dx in offsets:
        ny = iy + dy
        nx = ix + dx
        in_bounds = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        flat_n = np.where(in_bounds, ny * w + nx, 0)
        node_n = grid_to_node[flat_n]
        valid = in_bounds & (node_n >= 0)
        if not valid.any():
            continue
        rows.append(np.arange(ds.n_pixels)[valid])
        cols.append(node_n[valid])
    if not rows:
        return csr_matrix((ds.n_pixels, ds.n_pixels), dtype=np.float64)
    rr = np.concatenate(rows)
    cc = np.concatenate(cols)
    data = np.ones_like(rr, dtype=np.float64)
    W = csr_matrix((data, (rr, cc)), shape=(ds.n_pixels, ds.n_pixels))
    # The construction is already symmetric (each pair appears twice); drop any
    # accidental duplicates from boundary handling.
    W.sum_duplicates()
    W.eliminate_zeros()
    return W


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted q-values."""
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    multiplier = n / (np.arange(n, dtype=np.float64) + 1.0)
    q_sorted = np.minimum.accumulate((ranked * multiplier)[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    out = np.empty_like(q_sorted)
    out[order] = q_sorted
    return out


def _passthrough(
    ds: MSIDataset,
    params: MoransIParams,
    op_name: str,
    note: str,
) -> OpResult:
    diag = Diagnostic(
        name=op_name,
        summary={
            "n_channels_in": float(getattr(ds.backend, "n_peaks", 0)),
            "n_channels_out": float(getattr(ds.backend, "n_peaks", 0)),
            "n_dropped_by_fdr": 0.0,
            "n_permutations": float(params.n_permutations),
        },
        payload={"note": np.array([note], dtype=object)},
    )
    record = merge_op_record(
        op_name=op_name, params=params, input_ds=ds, output_ds=ds, diagnostics=[diag]
    )
    return OpResult(dataset=ds.with_history(record), diagnostics=[diag])
