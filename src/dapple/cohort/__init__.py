"""Cohort harmonization: align consensus channels across multiple datasets.

The single-dataset pipeline produces a ``PeakMatrix`` whose m/z axis was chosen
to fit *that dataset alone*. When several datasets are processed independently
and then compared, their consensus axes are subtly different — peak X may sit
at m/z 250.123 in dataset A and 250.121 in dataset B, even when they're the
same molecular ion. The cohort module produces a *shared* consensus axis used
by every dataset in the cohort, so post-hoc comparison is straightforward.

Approach: per-dataset reference detection and recalibration → pool every
post-recal peak from every pixel of every dataset → run a single KDE consensus
on the pooled cloud → for each dataset, build its own ``(n_pixels_d,
n_consensus)`` matrix on the *shared* axis. Each dataset retains its own
pixels and ROIs but they all use the same m/z columns.

This module exposes the algorithm as a function (``align_cohort``) rather than
a Pipeline operator because the pipeline DAG is currently single-input. The
``dapple-cohort-align`` CLI and napari Cohort harmonization widget provide
headless and interactive entry points.
"""

from dapple.cohort.align import (
    CohortAlignResult,
    CohortAlignParams,
    align_cohort,
    load_cohort_directory,
)

__all__ = [
    "CohortAlignParams",
    "CohortAlignResult",
    "align_cohort",
    "load_cohort_directory",
]
