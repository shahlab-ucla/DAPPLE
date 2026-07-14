"""CSV plus JSON-manifest export for analysis result objects."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from dapple.analysis.models import AxisProfileResult, RoiEnrichmentResult

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ExportedAnalysis:
    """Paths created by :func:`export_analysis_result`."""

    manifest_path: Path
    table_paths: tuple[Path, ...]


def export_analysis_result(
    result: RoiEnrichmentResult | AxisProfileResult,
    output_dir: Path | str,
    *,
    stem: str = "analysis",
) -> ExportedAnalysis:
    """Write stable CSV tables and a small JSON manifest.

    Numeric channel/profile data stay in CSV so downstream R, Python, and spreadsheet
    workflows can consume them directly. The JSON records result type, source
    fingerprint, endpoint geometry/orientation, scientific parameters, software
    versions, inference status, warnings, and the exact filenames.
    """
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename stem without directories")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(result, RoiEnrichmentResult):
        table_paths = (out / f"{stem}_roi_enrichment.csv",)
        result.to_frame().to_csv(table_paths[0], index=False)
    elif isinstance(result, AxisProfileResult):
        table_paths = (
            out / f"{stem}_axis_profiles.csv",
            out / f"{stem}_axis_statistics.csv",
        )
        result.profile_frame().to_csv(table_paths[0], index=False)
        result.statistics_frame().to_csv(table_paths[1], index=False)
    else:  # pragma: no cover - guarded by the public type contract
        raise TypeError(f"unsupported analysis result {type(result).__name__}")

    manifest = {
        "schema": "dapple-analysis",
        "schema_version": SCHEMA_VERSION,
        "software": _software_versions(),
        **result.metadata_dict(),
        "tables": [p.name for p in table_paths],
    }
    manifest_path = out / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExportedAnalysis(manifest_path=manifest_path, table_paths=table_paths)


def _software_versions() -> dict[str, str]:
    """Record the numerical/export environment without importing heavy packages."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": sys.implementation.name,
    }
    for package in ("dapple", "numpy", "scipy", "pandas", "scikit-image"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "source-tree"
    return versions
