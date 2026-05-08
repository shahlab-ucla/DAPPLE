"""``dapple-cohort-align``: shared consensus across a directory of datasets.

Usage::

    dapple-cohort-align ROOT_DIR [-o OUTPUT_DIR] [--pattern '*.imzML']
                        [--recursive] [--no-recalibrate]

For every imzML / CDF file matching ``--pattern`` under ``ROOT_DIR``, run the
single-dataset preprocessing (reference detection → tolerance fit →
recalibration → normalization → peak picking) and pool all peaks onto a single
shared consensus m/z axis. Each input dataset is written back as its own
PeakMatrix-backed imzML and TIFF aligned to the shared axis, plus a single
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
from pathlib import Path

# UTF-8 stdout/stderr so unicode glyphs (✓ ⚠ ✗ ─) from the diagnostic formatter
# render on Windows consoles. No-op where stdout is already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

import numpy as np

from dapple.cohort.align import (
    CohortAlignParams,
    align_cohort,
    load_cohort_directory,
)
from dapple.io.imzml_writer import write_imzml
from dapple.io.tiff_writer import write_hyperspectral_tiff
from dapple.pipeline import COHORT_RUBRICS, format_flat_summary

logger = logging.getLogger(__name__)


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
        help="Drop pooled-consensus peaks present in < this fraction of pixels.",
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
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading cohort from {root} (pattern={args.pattern!r}, recursive={args.recursive})")
    datasets = load_cohort_directory(root, pattern=args.pattern, recursive=args.recursive)
    print(f"  loaded {len(datasets)} dataset(s):")
    for i, d in enumerate(datasets):
        print(f"    [{i}] {Path(d.identity.source_path).name}: "
              f"{d.n_pixels} pixels, grid {d.grid_shape}")

    params = CohortAlignParams(
        bandwidth_ppm=float(args.bandwidth_ppm),
        min_prevalence=float(args.min_prevalence),
        recalibrate=not args.no_recalibrate,
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
    written: list[Path] = []
    for di, ds in enumerate(result.aligned_datasets):
        stem = Path(ds.identity.source_path).stem or f"dataset_{di:02d}"
        base = out_dir / f"{stem}_cohort"
        if not args.no_tiff:
            tiff_path = base.with_suffix(".tif")
            extra = {
                "cohort_prevalence": result.cohort_prevalence,
                "per_dataset_prevalence": result.per_dataset_prevalence[di],
            }
            tw = write_hyperspectral_tiff(ds, tiff_path, extra_per_channel=extra)
            written.append(tw.tiff_path)
            written.append(tw.csv_path)
            print(f"  [{di}] wrote {tw.tiff_path.name} ({tw.n_pages} channels)")
        if not args.no_imzml:
            imz_path = base.with_suffix(".imzML")
            iw = write_imzml(ds, imz_path)
            written.append(iw.imzml_path)
            written.append(iw.ibd_path)
            print(f"  [{di}] wrote {iw.imzml_path.name} (+ {iw.ibd_path.name})")

    # Cohort-level summary JSON.
    summary_path = out_dir / "cohort_summary.json"
    summary = {
        "n_datasets": len(datasets),
        "shared_consensus_mz": result.shared_consensus_mz.tolist(),
        "cohort_prevalence": result.cohort_prevalence.tolist(),
        "per_dataset_prevalence": {
            str(k): v.tolist() for k, v in result.per_dataset_prevalence.items()
        },
        "datasets": [
            {
                "index": i,
                "source_path": str(d.identity.source_path),
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
        "params": {
            "bandwidth_ppm": params.bandwidth_ppm,
            "bandwidth_scale": params.bandwidth_scale,
            "min_prevalence": params.min_prevalence,
            "min_prominence_quantile": params.min_prominence_quantile,
            "n_grid_points": params.n_grid_points,
            "default_tol_ppm": params.default_tol_ppm,
            "pool_normalize": params.pool_normalize,
            "recalibrate": params.recalibrate,
            "rng_seed": params.rng_seed,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)
    print(f"\nwrote {summary_path}")
    print(f"done — {len(written)} output file(s) in {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
