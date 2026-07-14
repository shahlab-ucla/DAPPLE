"""``dapple-cohort-align``: shared consensus across a directory of datasets.

Usage::

    dapple-cohort-align ROOT_DIR [-o OUTPUT_DIR] [--pattern '*.imzML']
                        [--recursive] [--no-recalibrate]

For every imzML / CDF file matching ``--pattern`` under ``ROOT_DIR``, run the
single-dataset preprocessing (reference detection → tolerance fit →
recalibration → normalization → peak picking) and pool all peaks onto a single
shared consensus m/z axis. Each input dataset is written back as its own
PeakMatrix-backed imzML/ibd/axis-sidecar set and TIFF aligned to the shared axis, plus a single
``cohort_summary.json`` with the cross-dataset prevalence and run-level
diagnostics.

Exit codes:

- 0  success
- 1  load or pipeline error
- 2  argument error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

# UTF-8 stdout/stderr so unicode glyphs (✓ ⚠ ✗ ─) from the diagnostic formatter
# render on Windows consoles. No-op where stdout is already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

from dapple.cohort.align import (
    CohortAlignParams,
    align_cohort,
)
from dapple.data.dataset import MSIDataset
from dapple.io.imzml_reader import read_imzml
from dapple.io.imzml_writer import write_imzml
from dapple.io.tiff_writer import write_hyperspectral_tiff
from dapple.pipeline import COHORT_RUBRICS, format_flat_summary

logger = logging.getLogger(__name__)


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether *path* resolves inside *directory*."""

    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _discover_cohort_files(
    root: Path,
    *,
    pattern: str,
    recursive: bool,
    exclude_dir: Path | None = None,
) -> tuple[list[Path], int]:
    """Discover inputs without feeding prior harmonized outputs back in."""

    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    excluded = 0
    if exclude_dir is not None and exclude_dir.resolve() != root.resolve():
        kept = [f for f in files if not _is_within(f, exclude_dir)]
        excluded = len(files) - len(kept)
        files = kept
    # When users intentionally write into ROOT_DIR itself there is no directory
    # subtree to exclude. DAPPLE PeakMatrix imzML outputs carry this axis
    # sidecar and cannot be valid PeakList cohort inputs, so ignore them even
    # when output and input roots are identical.
    kept = [
        f for f in files
        if not f.with_suffix(".dapple-axis.json").is_file()
    ]
    excluded += len(files) - len(kept)
    files = kept
    if not files:
        suffix = " after excluding generated/output files" if excluded else ""
        raise FileNotFoundError(
            f"no files matching {pattern!r} under {root}{suffix}"
        )
    return files, excluded


def _load_cohort_files(files: Sequence[Path]) -> list[MSIDataset]:
    """Load an already-discovered, stable cohort file list."""

    datasets: list[MSIDataset] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in {".imzml", ".ibd"}:
            datasets.append(read_imzml(path))
        elif suffix in {".cdf", ".nc"}:
            from dapple.io.cdf_reader import read_cdf

            datasets.append(read_cdf(path))
        else:
            raise ValueError(f"don't know how to load {path}")
    return datasets


def _unique_output_bases(
    datasets: Sequence[MSIDataset], out_dir: Path
) -> list[Path]:
    """Choose deterministic basenames without overwriting duplicate stems."""

    used: set[str] = set()
    bases: list[Path] = []
    for index, dataset in enumerate(datasets):
        stem = Path(dataset.identity.source_path).stem or f"dataset_{index:02d}"
        candidate = f"{stem}_cohort"
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{suffix}_cohort"
            suffix += 1
        used.add(candidate.casefold())
        bases.append(out_dir / candidate)
    return bases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dapple-cohort-align",
        description=(
            "Run a unified DAPPLE pipeline across every dataset under ROOT_DIR. "
            "Each dataset gets its own preprocessing; a single KDE consensus "
            "on the pooled peaks produces a shared m/z axis used by every "
            "cohort member. Outputs one .imzML + .tif per dataset plus a "
            "cohort_summary.json."
        ),
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing the cohort's imzML / CDF files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write per-dataset outputs. Defaults to ROOT_DIR/dapple_cohort.",
    )
    parser.add_argument(
        "--pattern",
        default="*.imzML",
        help="Glob pattern for cohort members. Default '*.imzML'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Walk subdirectories of ROOT_DIR.",
    )
    parser.add_argument(
        "--no-recalibrate",
        action="store_true",
        help="Skip per-dataset recalibration before pooling.",
    )
    parser.add_argument(
        "--bandwidth-ppm",
        type=float,
        default=50.0,
        help="KDE bandwidth in ppm. Default 50.",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.05,
        help=(
            "Drop pooled-consensus peaks below this prevalence. The denominator "
            "is selected with --prevalence-basis. Default 0.05."
        ),
    )
    parser.add_argument(
        "--pool-weighting",
        choices=("sample", "intensity"),
        default="sample",
        help=(
            "Weight each dataset equally ('sample', recommended) or let raw "
            "pooled intensity determine its contribution ('intensity')."
        ),
    )
    parser.add_argument(
        "--prevalence-basis",
        choices=("pixel", "dataset"),
        default="pixel",
        help=(
            "Apply --min-prevalence to all cohort pixels ('pixel') or to the "
            "fraction of datasets carrying a channel ('dataset')."
        ),
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="RNG seed (deterministic re-runs).",
    )
    parser.add_argument(
        "--no-imzml",
        action="store_true",
        help="Skip per-dataset imzML output (TIFF only).",
    )
    parser.add_argument(
        "--no-tiff",
        action="store_true",
        help="Skip per-dataset TIFF output (imzML only).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
    )
    args = parser.parse_args(argv)

    log_level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(log_level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return _run(args)
    except Exception as e:  # noqa: BLE001
        print(f"dapple-cohort-align: {e}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    root: Path = args.root.expanduser().resolve()
    out_dir: Path = (args.output_dir or (root / "dapple_cohort")).expanduser().resolve()

    print(f"loading cohort from {root} (pattern={args.pattern!r}, recursive={args.recursive})")
    input_files, n_excluded = _discover_cohort_files(
        root,
        pattern=args.pattern,
        recursive=args.recursive,
        exclude_dir=out_dir,
    )
    if n_excluded:
        print(f"  ignored {n_excluded} matching generated/output file(s)")
    datasets = _load_cohort_files(input_files)
    print(f"  loaded {len(datasets)} dataset(s):")
    for i, d in enumerate(datasets):
        print(f"    [{i}] {Path(d.identity.source_path).name}: "
              f"{d.n_pixels} pixels, grid {d.grid_shape}")

    params = CohortAlignParams(
        bandwidth_ppm=float(args.bandwidth_ppm),
        min_prevalence=float(args.min_prevalence),
        recalibrate=not args.no_recalibrate,
        pool_weighting=args.pool_weighting,
        prevalence_basis=args.prevalence_basis,
        rng_seed=int(args.rng_seed),
    )
    print("aligning cohort...")
    result = align_cohort(datasets, params=params)

    # Pretty-printed cohort diagnostics with health rubric. Same format as
    # the per-node block from `dapple-apply-spec` for consistency.
    print("")
    for line in format_flat_summary(
        result.diagnostics,
        title="Cohort alignment diagnostics",
        extra_rubrics=COHORT_RUBRICS,
    ):
        print(line)
    print("")

    # Per-dataset outputs.
    out_dir.mkdir(parents=True, exist_ok=True)
    output_bases = _unique_output_bases(result.aligned_datasets, out_dir)
    written: list[Path] = []
    dataset_output_files: list[list[str]] = [
        [] for _ in result.aligned_datasets
    ]
    for di, (ds, base) in enumerate(zip(result.aligned_datasets, output_bases)):
        if not args.no_tiff:
            tiff_path = base.with_suffix(".tif")
            extra = {
                "cohort_prevalence": result.cohort_prevalence,
                "dataset_prevalence": result.dataset_prevalence,
                "per_dataset_prevalence": result.per_dataset_prevalence[di],
            }
            tw = write_hyperspectral_tiff(ds, tiff_path, extra_per_channel=extra)
            written.append(tw.tiff_path)
            written.append(tw.csv_path)
            dataset_output_files[di].extend((tw.tiff_path.name, tw.csv_path.name))
            print(f"  [{di}] wrote {tw.tiff_path.name} ({tw.n_pages} channels)")
        if not args.no_imzml:
            imz_path = base.with_suffix(".imzML")
            iw = write_imzml(ds, imz_path)
            written.extend(iw.artifact_paths)
            dataset_output_files[di].extend(path.name for path in iw.artifact_paths)
            companions = [path.name for path in iw.artifact_paths[1:]]
            print(
                f"  [{di}] wrote {iw.imzml_path.name} "
                f"(+ {', '.join(companions)})"
            )

    # Cohort-level summary JSON.
    summary_path = out_dir / "cohort_summary.json"
    selected_prevalence = (
        result.dataset_prevalence
        if params.prevalence_basis == "dataset"
        else result.cohort_prevalence
    )
    summary = {
        "summary_schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "discovery": {
            "root": str(root),
            "pattern": args.pattern,
            "recursive": bool(args.recursive),
            "input_files": [str(path) for path in input_files],
            "output_dir_excluded": str(out_dir) if out_dir != root else None,
            "n_matching_outputs_excluded": n_excluded,
        },
        "n_datasets": len(datasets),
        "n_channels": int(result.shared_consensus_mz.size),
        "output_files": [path.name for path in written] + [summary_path.name],
        "shared_consensus_mz": result.shared_consensus_mz.tolist(),
        "cohort_prevalence": result.cohort_prevalence.tolist(),
        "dataset_prevalence": result.dataset_prevalence.tolist(),
        "prevalence_filter": {
            "basis": params.prevalence_basis,
            "minimum": params.min_prevalence,
            "values": selected_prevalence.tolist(),
        },
        "per_dataset_prevalence": {
            str(k): v.tolist() for k, v in result.per_dataset_prevalence.items()
        },
        "datasets": [
            {
                "index": i,
                "source_path": str(d.identity.source_path),
                "content_sha256": d.identity.content_sha256,
                "declared_md5": d.identity.declared_md5,
                "output_basename": output_bases[i].name,
                "output_files": dataset_output_files[i],
                "n_pixels": int(d.n_pixels),
                "grid_shape": list(d.grid_shape),
                "instrument_family": d.metadata.instrument_family,
                "ionization": d.metadata.ionization,
                "polarity": d.metadata.polarity,
                "mode": d.metadata.profile_or_centroided,
            }
            for i, d in enumerate(datasets)
        ],
        "diagnostics": result.diagnostics,
        "params": asdict(params),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)
    print(f"\nwrote {summary_path}")
    print(f"done — {len(written)} output file(s) in {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
