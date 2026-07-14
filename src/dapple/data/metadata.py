"""Experiment parameters and ROI definitions.

`ExperimentParams` captures everything the wizard collects from the user about how the
data was acquired. It feeds `recommend_pipeline` and the operator defaults table.
`RoiDef` describes a polygon region of interest drawn on the image.

Both are frozen dataclasses so they hash cleanly into the pipeline DAG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

InstrumentFamily = Literal[
    "tof_axial",
    "tof_reflectron",
    "qtof",
    "orbitrap",
    "fticr",
    "unknown",
]
Ionization = Literal["maldi", "desi", "sims", "esi", "unknown"]
Mode = Literal["profile", "centroided", "unknown"]
Polarity = Literal["positive", "negative"]
SampleType = Literal["tissue", "cell_culture", "whole_organism", "other"]


@dataclass(frozen=True)
class ExperimentParams:
    instrument_family: InstrumentFamily
    ionization: Ionization
    profile_or_centroided: Mode
    polarity: Polarity
    mz_min: float
    mz_max: float
    pixel_size_um: float | None = None
    sample_type: SampleType | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.mz_min) or not math.isfinite(self.mz_max):
            raise ValueError("m/z bounds must be finite")
        if self.mz_max <= self.mz_min:
            raise ValueError(f"mz_max ({self.mz_max}) must exceed mz_min ({self.mz_min})")
        if self.pixel_size_um is not None and self.pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be positive, got {self.pixel_size_um}")


@dataclass(frozen=True)
class RoiDef:
    """A polygon ROI in zero-based napari ``(y, x)`` data coordinates."""

    name: str
    vertices: tuple[tuple[float, float], ...]  # ((y, x), ...) in pixel coords
    is_background: bool = False
    color: str = "#ff7f0e"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ROI name must not be empty")
        if len(self.vertices) < 3:
            raise ValueError(f"ROI '{self.name}' needs >=3 vertices, got {len(self.vertices)}")
        if any(len(vertex) != 2 for vertex in self.vertices):
            raise ValueError(f"ROI '{self.name}' vertices must be (y, x) pairs")
        if any(not math.isfinite(value) for vertex in self.vertices for value in vertex):
            raise ValueError(f"ROI '{self.name}' vertices must be finite")


@dataclass(frozen=True)
class DatasetIdentity:
    """Stable identifiers for a dataset."""

    source_path: str
    content_sha256: str
    declared_md5: str | None = None  # for imzML, the IMS:1000090 ibd MD5
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)
