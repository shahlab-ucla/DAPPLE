"""Per-pixel projections (TIC, RMS, median, ...) rendered to napari (H, W) images.

The actual computation lives on `MSIDataset.project()`; this module is a thin registry
that names the projections, gives each a human-readable label and a sensible colormap,
and provides a single entry point the wizard's PreviewPage iterates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dapple.data.dataset import MSIDataset, ProjectionKind


@dataclass(frozen=True)
class ProjectionSpec:
    kind: "ProjectionKind"
    label: str
    description: str
    colormap: str


PROJECTIONS: tuple[ProjectionSpec, ...] = (
    ProjectionSpec(
        kind="tic",
        label="Total ion current (TIC)",
        description="Sum of intensities per pixel — most common general-purpose view.",
        colormap="viridis",
    ),
    ProjectionSpec(
        kind="rms",
        label="Root-mean-square intensity",
        description="Less sensitive to a single dominating peak than TIC.",
        colormap="magma",
    ),
    ProjectionSpec(
        kind="median",
        label="Median intensity",
        description="Robust per-pixel center; useful as a normalization reference.",
        colormap="inferno",
    ),
    ProjectionSpec(
        kind="base_peak",
        label="Base peak intensity",
        description="Maximum intensity per pixel — highlights the dominant analyte.",
        colormap="cividis",
    ),
    ProjectionSpec(
        kind="peak_count",
        label="Peak count",
        description="Number of peaks per pixel — good for detecting empty/sparse regions.",
        colormap="gray_r",
    ),
    ProjectionSpec(
        kind="mean_mz",
        label="Intensity-weighted mean m/z",
        description="Captures gross compositional shifts across the image.",
        colormap="turbo",
    ),
)


def project(ds: "MSIDataset", kind: "ProjectionKind") -> np.ndarray:
    """Compute the named projection — convenience wrapper around `MSIDataset.project`."""
    return ds.project(kind)


def projection_layer_name(spec: ProjectionSpec, dataset_stem: str) -> str:
    """Layer name for a projection in napari, e.g. 'jerboa-100825 · TIC'."""
    short = {
        "tic": "TIC",
        "rms": "RMS",
        "median": "median intensity",
        "base_peak": "base peak",
        "peak_count": "peak count",
        "mean_mz": "mean m/z",
    }.get(spec.kind, spec.kind)
    return f"{dataset_stem} · {short}"


def channel_layer_name(dataset_stem: str, mz: float, prevalence: float | None = None) -> str:
    """Layer name for a single consensus m/z channel in the hyperspectral browser."""
    base = f"{dataset_stem} · m/z {mz:.4f}"
    if prevalence is not None:
        return f"{base} · prev {prevalence:.0%}"
    return base


def percentile_contrast(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> tuple[float, float]:
    """Robust contrast bounds for a (possibly sparse) image layer.

    Computes percentile bounds on the *non-zero* values so that an empty-pixel
    background doesn't compress everything into a single visible color. Returns a
    pair (lo, hi) with `hi > lo`; if the image is degenerate, returns (min, max).
    """
    nz = img[img > 0] if (img > 0).any() else img
    if nz.size == 0:
        return float(img.min()), max(float(img.max()), float(img.min()) + 1e-12)
    lo = float(np.quantile(nz, lo_pct / 100.0))
    hi = float(np.quantile(nz, hi_pct / 100.0))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi
