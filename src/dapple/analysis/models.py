"""Typed result objects for ROI and anatomical-axis analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

InferenceStatus = Literal["ok", "partial", "insufficient_units", "insufficient_bins"]


def _one_dimensional(name: str, value: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (size,):
        raise ValueError(f"{name} shape {arr.shape} != ({size},)")
    return arr


@dataclass(frozen=True, eq=False)
class RoiEnrichmentResult:
    """Per-channel descriptive and Welch-inference results for one ROI contrast.

    ``log2_fold_change`` is numerator over denominator and uses trimmed raw-intensity
    centers plus a recorded, channel-adaptive pseudocount.  Welch inference is run on
    ``log2(intensity + pseudocount)``.  The two quantities are kept separate so the
    displayed effect retains a fold-change interpretation.
    """

    contrast_name: str
    numerator_names: tuple[str, ...]
    denominator_names: tuple[str, ...]
    mz: np.ndarray
    n_numerator: int
    n_denominator: int
    mean_numerator: np.ndarray
    mean_denominator: np.ndarray
    median_numerator: np.ndarray
    median_denominator: np.ndarray
    trimmed_mean_numerator: np.ndarray
    trimmed_mean_denominator: np.ndarray
    prevalence_numerator: np.ndarray
    prevalence_denominator: np.ndarray
    pseudocount: np.ndarray
    log2_fold_change: np.ndarray
    mean_log2_difference: np.ndarray
    welch_t: np.ndarray
    welch_df: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    test_valid: np.ndarray
    inference_status: InferenceStatus
    spatial_fingerprint: str = ""
    warnings: tuple[str, ...] = ()
    source_dataset_hash: str = ""
    overlap_policy: str = ""
    min_units_per_group: int = 3
    trim_fraction: float = 0.1
    pseudocount_quantile: float = 0.05

    def __post_init__(self) -> None:
        k = int(np.asarray(self.mz).size)
        _one_dimensional("mz", self.mz, k)
        for name in (
            "mean_numerator",
            "mean_denominator",
            "median_numerator",
            "median_denominator",
            "trimmed_mean_numerator",
            "trimmed_mean_denominator",
            "prevalence_numerator",
            "prevalence_denominator",
            "pseudocount",
            "log2_fold_change",
            "mean_log2_difference",
            "welch_t",
            "welch_df",
            "p_value",
            "q_value",
            "test_valid",
        ):
            _one_dimensional(name, getattr(self, name), k)

    def to_frame(self) -> pd.DataFrame:
        """Return one row per shared-m/z channel."""
        return pd.DataFrame(
            {
                "contrast": self.contrast_name,
                "mz": np.asarray(self.mz, dtype=np.float64),
                "n_numerator": self.n_numerator,
                "n_denominator": self.n_denominator,
                "mean_numerator": self.mean_numerator,
                "mean_denominator": self.mean_denominator,
                "median_numerator": self.median_numerator,
                "median_denominator": self.median_denominator,
                "trimmed_mean_numerator": self.trimmed_mean_numerator,
                "trimmed_mean_denominator": self.trimmed_mean_denominator,
                "prevalence_numerator": self.prevalence_numerator,
                "prevalence_denominator": self.prevalence_denominator,
                "prevalence_difference": (
                    self.prevalence_numerator - self.prevalence_denominator
                ),
                "pseudocount": self.pseudocount,
                "log2_fold_change": self.log2_fold_change,
                "mean_log2_difference": self.mean_log2_difference,
                "welch_t": self.welch_t,
                "welch_df": self.welch_df,
                "p_value": self.p_value,
                "q_value": self.q_value,
                "test_valid": self.test_valid,
                "inference_unit": "pixel",
                "inference_status": self.inference_status,
            }
        )

    def metadata_dict(self) -> dict[str, object]:
        """Small JSON-safe description; channel arrays belong in :meth:`to_frame`."""
        return {
            "kind": "roi_enrichment",
            "contrast_name": self.contrast_name,
            "numerator_names": list(self.numerator_names),
            "denominator_names": list(self.denominator_names),
            "n_numerator": int(self.n_numerator),
            "n_denominator": int(self.n_denominator),
            "n_channels": int(self.mz.size),
            "inference_unit": "pixel",
            "inference_status": self.inference_status,
            "source_dataset_hash": self.source_dataset_hash,
            "spatial_fingerprint": self.spatial_fingerprint,
            "overlap_policy": self.overlap_policy,
            "analysis_parameters": {
                "min_units_per_group": int(self.min_units_per_group),
                "trim_fraction": float(self.trim_fraction),
                "pseudocount_quantile": float(self.pseudocount_quantile),
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, eq=False)
class AxisProfileResult:
    """Binned abundance profiles and per-channel ordered-axis pattern statistics."""

    axis_name: str
    start_label: str
    end_label: str
    mz: np.ndarray
    bin_centers: np.ndarray
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    mean_intensity: np.ndarray  # (B, K)
    median_intensity: np.ndarray  # (B, K)
    mean_log2_intensity: np.ndarray  # (B, K)
    prevalence: np.ndarray  # (B, K)
    pseudocount: np.ndarray
    spearman_rho: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    endpoint_log2_enrichment: np.ndarray  # end over start
    peak_position: np.ndarray
    concentration: np.ndarray
    pattern_label: np.ndarray
    n_included_pixels: int
    n_start_pixels: int
    n_end_pixels: int
    n_permutations_requested: int
    n_permutations_effective: int
    inference_status: InferenceStatus
    spatial_fingerprint: str = ""
    warnings: tuple[str, ...] = ()
    source_dataset_hash: str = ""
    axis_start_yx: tuple[float, float] = (0.0, 0.0)
    axis_end_yx: tuple[float, float] = (0.0, 1.0)
    half_width_px: float | None = None
    min_pixels_per_bin: int = 2
    min_bins_for_trend: int = 4
    endpoint_fraction: float = 0.2
    min_endpoint_pixels: int = 3
    trim_fraction: float = 0.1
    pseudocount_quantile: float = 0.05
    q_threshold: float = 0.05
    trend_threshold: float = 0.5
    concentration_threshold: float = 0.25
    endpoint_effect_threshold: float = 1.0
    rng_seed: int = 0
    selection_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        k = int(np.asarray(self.mz).size)
        b = int(np.asarray(self.bin_centers).size)
        _one_dimensional("mz", self.mz, k)
        _one_dimensional("bin_centers", self.bin_centers, b)
        _one_dimensional("bin_edges", self.bin_edges, b + 1)
        _one_dimensional("bin_counts", self.bin_counts, b)
        for name in (
            "mean_intensity",
            "median_intensity",
            "mean_log2_intensity",
            "prevalence",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (b, k):
                raise ValueError(f"{name} shape {arr.shape} != ({b}, {k})")
        for name in (
            "pseudocount",
            "spearman_rho",
            "p_value",
            "q_value",
            "endpoint_log2_enrichment",
            "peak_position",
            "concentration",
            "pattern_label",
        ):
            _one_dimensional(name, getattr(self, name), k)

    def profile_frame(self) -> pd.DataFrame:
        """Return a long-form ``bin × channel`` profile table."""
        b, k = self.mean_intensity.shape
        return pd.DataFrame(
            {
                "axis": self.axis_name,
                "bin": np.repeat(np.arange(b, dtype=np.int64), k),
                "bin_center": np.repeat(self.bin_centers, k),
                "bin_start": np.repeat(self.bin_edges[:-1], k),
                "bin_end": np.repeat(self.bin_edges[1:], k),
                "bin_n_pixels": np.repeat(self.bin_counts, k),
                "mz": np.tile(self.mz, b),
                "mean_intensity": self.mean_intensity.reshape(-1),
                "median_intensity": self.median_intensity.reshape(-1),
                "mean_log2_intensity": self.mean_log2_intensity.reshape(-1),
                "prevalence": self.prevalence.reshape(-1),
            }
        )

    def statistics_frame(self) -> pd.DataFrame:
        """Return one row per channel with ordered-axis summary statistics."""
        return pd.DataFrame(
            {
                "axis": self.axis_name,
                "start_label": self.start_label,
                "end_label": self.end_label,
                "mz": self.mz,
                "pseudocount": self.pseudocount,
                "spearman_rho": self.spearman_rho,
                "p_value": self.p_value,
                "q_value": self.q_value,
                "endpoint_log2_enrichment": self.endpoint_log2_enrichment,
                "peak_position": self.peak_position,
                "concentration": self.concentration,
                "pattern_label": self.pattern_label,
                "inference_status": self.inference_status,
            }
        )

    def metadata_dict(self) -> dict[str, object]:
        """Small JSON-safe description; numeric arrays are exported as CSV."""
        return {
            "kind": "axis_profile",
            "axis_name": self.axis_name,
            "start_label": self.start_label,
            "end_label": self.end_label,
            "n_channels": int(self.mz.size),
            "n_bins": int(self.bin_centers.size),
            "bin_counts": self.bin_counts.astype(int).tolist(),
            "n_included_pixels": int(self.n_included_pixels),
            "n_start_pixels": int(self.n_start_pixels),
            "n_end_pixels": int(self.n_end_pixels),
            "n_permutations_requested": int(self.n_permutations_requested),
            "n_permutations_effective": int(self.n_permutations_effective),
            "permutation_unit": "ordered_bin",
            "inference_status": self.inference_status,
            "source_dataset_hash": self.source_dataset_hash,
            "spatial_fingerprint": self.spatial_fingerprint,
            "selection_names": list(self.selection_names),
            "axis_geometry": {
                "coordinate_system": "napari-data-yx-zero-based",
                "start_yx": [float(v) for v in self.axis_start_yx],
                "end_yx": [float(v) for v in self.axis_end_yx],
                "half_width_px": (
                    None if self.half_width_px is None else float(self.half_width_px)
                ),
            },
            "analysis_parameters": {
                "min_pixels_per_bin": int(self.min_pixels_per_bin),
                "min_bins_for_trend": int(self.min_bins_for_trend),
                "endpoint_fraction": float(self.endpoint_fraction),
                "min_endpoint_pixels": int(self.min_endpoint_pixels),
                "trim_fraction": float(self.trim_fraction),
                "pseudocount_quantile": float(self.pseudocount_quantile),
                "q_threshold": float(self.q_threshold),
                "trend_threshold": float(self.trend_threshold),
                "concentration_threshold": float(self.concentration_threshold),
                "endpoint_effect_threshold": float(self.endpoint_effect_threshold),
                "rng_seed": int(self.rng_seed),
            },
            "warnings": list(self.warnings),
        }
