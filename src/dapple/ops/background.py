"""Background subtraction operators.

The wizard lets the user mark some polygon ROIs as "background" — typically
off-tissue pixels, or a separate matrix-only region. This module consumes those
ROIs (or, equivalently, "everything outside any foreground ROI") to reject or
attenuate channels whose intensity in the background is comparable to the
foreground signal — i.e., contaminating peaks that aren't actually associated with
the analyte you care about.

Two modes:
  - ``reject_channels`` (default): drop consensus m/z channels whose background
    mean intensity is at least ``bg_to_fg_ratio_threshold`` of their foreground
    mean. Conservative and easy to interpret.
  - ``subtract``: replace each pixel's intensity in channel c with
    ``max(0, intensity - bg_mean[c])``. Preserves channel count but does mutate
    intensities; only meaningful when background is uniform.

Requires a ``PeakMatrix`` backend (post-consensus alignment) and at least one
foreground ROI. If no background ROI is supplied, the operator can fall back to
"everything outside the foreground ROIs" — controlled by ``use_outside_as_bg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dapple.data.coords import coords_to_grid_index
from dapple.data.dataset import MSIDataset, PeakMatrix
from dapple.data.metadata import ExperimentParams, RoiDef
from dapple.ops.base import (
    Diagnostic,
    OpParams,
    OpResult,
    Operator,
    merge_op_record,
    register,
)

BgMode = Literal["reject_channels", "subtract"]


@dataclass(frozen=True)
class BackgroundSubtractParams(OpParams):
    mode: BgMode = field(
        default="reject_channels",
        metadata={
            "label": "Mode",
            "help": (
                "reject_channels: drop consensus m/z channels whose mean intensity "
                "in background pixels is at least bg_to_fg_ratio_threshold of "
                "their foreground mean. Conservative — preserves the foreground "
                "intensities exactly; just removes confounded channels. "
                "subtract: per-pixel subtract the per-channel background mean "
                "(clipped at zero). Use only when background pixels are "
                "uniformly off-tissue and intensity is comparable across "
                "spatial regions."
            ),
        },
    )
    bg_to_fg_ratio_threshold: float = field(
        default=0.5,
        metadata={
            "label": "Background / foreground ratio threshold",
            "help": (
                "In reject_channels mode, drop channel c when "
                "mean(bg, c) / max(mean(fg, c), eps) ≥ this value. Default 0.5 "
                "rejects channels that are at least half as bright in the "
                "background — these are usually matrix peaks or contaminants. "
                "Lower toward 0.2–0.3 to be stricter (drop more channels); raise "
                "to 0.8 to keep most channels and only kill the most egregious."
            ),
        },
    )
    use_outside_as_bg: bool = field(
        default=True,
        metadata={
            "label": "Use outside-ROI pixels as background",
            "help": (
                "When no ROI is explicitly marked as background but at least one "
                "foreground ROI is drawn, treat 'everything outside the foreground' "
                "as the background population. Default ON. Disable when you only "
                "want explicit background ROIs to count (e.g. when off-tissue pixels "
                "still contain analyte signal you don't want subtracted)."
            ),
        },
    )
    eps: float = field(
        default=1e-9,
        metadata={
            "label": "Numerical floor",
            "help": (
                "Avoids division-by-zero when a foreground channel has zero mean. "
                "Default 1e-9; you should not need to change it."
            ),
        },
    )


@register
class BackgroundSubtract(Operator):
    name = "background_subtract"
    params_cls = BackgroundSubtractParams

    def default_params(self, ep: ExperimentParams) -> BackgroundSubtractParams:
        return BackgroundSubtractParams()

    def validate(self, ep: ExperimentParams) -> list[str]:
        return []

    def apply(
        self,
        ds: MSIDataset,
        params: BackgroundSubtractParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        if not isinstance(ds.backend, PeakMatrix):
            raise RuntimeError(
                "background_subtract requires a PeakMatrix backend (post-consensus "
                "alignment). Run kde_consensus_alignment first."
            )
        if not ds.rois:
            raise RuntimeError(
                "background_subtract requires at least one ROI on the dataset. Draw a "
                "foreground polygon (and optionally a background polygon) on the wizard's "
                "RoiPage before running."
            )
        fg_mask, bg_mask = _foreground_background_masks(
            ds, use_outside_as_bg=params.use_outside_as_bg
        )
        if not fg_mask.any():
            raise RuntimeError("no foreground pixels — every ROI is marked as background.")
        if not bg_mask.any():
            raise RuntimeError(
                "no background pixels. Either draw a background ROI or enable "
                "'Use outside-ROI pixels as background'."
            )

        pm: PeakMatrix = ds.backend
        matrix = np.asarray(pm.matrix[:])
        fg_mean = matrix[fg_mask, :].mean(axis=0).astype(np.float64)
        bg_mean = matrix[bg_mask, :].mean(axis=0).astype(np.float64)
        ratios = bg_mean / np.maximum(fg_mean, params.eps)

        if params.mode == "reject_channels":
            keep_mask = ratios < params.bg_to_fg_ratio_threshold
            n_dropped = int((~keep_mask).sum())
            if not keep_mask.any():
                raise RuntimeError(
                    f"every channel has bg/fg ratio ≥ {params.bg_to_fg_ratio_threshold}; "
                    "nothing would survive. Lower bg_to_fg_ratio_threshold or check your "
                    "ROI assignments."
                )
            new_matrix = matrix[:, keep_mask].astype(np.float32, copy=False)
            new_axis = np.asarray(pm.mz_axis[:])[keep_mask].astype(np.float64, copy=False)
            new_pm = PeakMatrix(matrix=new_matrix, mz_axis=new_axis)
            new_extra = {**ds.extra}
            # Trim the consensus-prevalence vector if it's present.
            if "consensus_prevalence" in new_extra:
                old_prev = np.asarray(new_extra["consensus_prevalence"])
                if old_prev.shape == (matrix.shape[1],):
                    new_extra["consensus_prevalence"] = old_prev[keep_mask]
            new_extra["bg_subtract_dropped_n"] = n_dropped
            new_extra["bg_subtract_dropped_mz"] = np.asarray(pm.mz_axis[:])[~keep_mask]
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
        else:  # subtract
            corrected = matrix.astype(np.float64, copy=True) - bg_mean[None, :]
            np.maximum(corrected, 0.0, out=corrected)
            new_pm = PeakMatrix(
                matrix=corrected.astype(np.float32, copy=False),
                mz_axis=pm.mz_axis,
            )
            new_extra = {**ds.extra, "bg_subtract_per_channel_bg_mean": bg_mean}
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
                "mode_reject_channels": float(params.mode == "reject_channels"),
                "n_fg_pixels": float(int(fg_mask.sum())),
                "n_bg_pixels": float(int(bg_mask.sum())),
                "n_channels_in": float(matrix.shape[1]),
                "n_channels_out": float(new_ds.backend.n_peaks),
                "ratio_median": float(np.median(ratios)),
                "ratio_max": float(ratios.max()),
                "bg_to_fg_ratio_threshold": float(params.bg_to_fg_ratio_threshold),
            },
            payload={
                "fg_mean": fg_mean,
                "bg_mean": bg_mean,
                "ratios": ratios,
                "channel_mz": np.asarray(pm.mz_axis[:]),
            },
            figure_hint="histogram:bg_to_fg_ratio_with_threshold_marker",
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


def _foreground_background_masks(
    ds: MSIDataset, *, use_outside_as_bg: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build boolean (n_pixels,) masks for foreground and background populations."""
    from skimage.draw import polygon as sk_polygon

    h, w = ds.grid_shape
    fg_grid = np.zeros((h, w), dtype=bool)
    bg_grid = np.zeros((h, w), dtype=bool)
    has_fg = False
    has_bg = False
    for roi in ds.rois:
        ys = np.asarray([v[0] for v in roi.vertices])
        xs = np.asarray([v[1] for v in roi.vertices])
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if roi.is_background:
            bg_grid[rr, cc] = True
            has_bg = True
        else:
            fg_grid[rr, cc] = True
            has_fg = True

    flat = coords_to_grid_index(ds.coords, ds.grid_shape)
    fg_mask = fg_grid.reshape(-1)[flat]
    bg_mask = bg_grid.reshape(-1)[flat]

    if has_fg and not has_bg and use_outside_as_bg:
        # Treat any populated pixel that isn't foreground as background.
        bg_mask = ~fg_mask
    elif has_bg and not has_fg:
        # If only a background ROI was drawn, treat the rest as foreground.
        fg_mask = ~bg_mask
    return fg_mask, bg_mask
