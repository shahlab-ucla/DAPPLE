"""``dapple-apply-spec``: apply a saved ``.spec.xml`` pipeline to an input dataset.

Usage::

    dapple-apply-spec INPUT SPEC [-o OUTPUT_BASE] [--rng-seed N]

INPUT may be an imzML file, a single .cdf file, or a directory of .cdf files
(multi-file MSI imaging). SPEC is a ``.spec.xml`` from an earlier run. The
loaded pipeline runs against INPUT; outputs land at ``OUTPUT_BASE.imzML/.ibd``,
``OUTPUT_BASE.tif`` (+ ``OUTPUT_BASE_channels.csv``), and a fresh
``OUTPUT_BASE.spec.xml`` recording this re-run.

Exit codes:

- 0  success
- 1  load or pipeline error (with diagnostic message on stderr)
- 2  argument error (the argparse default; printed by argparse itself)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 so unicode glyphs in the diagnostic
# formatter (✓ ⚠ ✗ ─) render correctly on Windows consoles whose default
# codepage is cp1252. No-op when stdout is already UTF-8 (Linux / macOS / new
# Windows terminal). Wrapped in a try/except because some non-tty wrappers
# don't expose ``reconfigure``.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

import numpy as np

from dapple.io.cdf_image_reader import read_cdf_image
from dapple.io.cdf_reader import read_cdf
from dapple.io.imzml_reader import read_imzml
from dapple.io.imzml_writer import write_imzml
from dapple.io.spec_xml import make_provenance, read_spec_xml, write_spec_xml
from dapple.io.tiff_writer import write_hyperspectral_tiff
from dapple.pipeline import PipelineRunner, format_diagnostics

logger = logging.getLogger(__name__)


def _load_input(path: Path):  # noqa: ANN201 — MSIDataset
    """Pick the right reader for `path` and return an MSIDataset."""
    if path.is_dir():
        return read_cdf_image(path)
    suffix = path.suffix.lower()
    if suffix in {".imzml", ".ibd"}:
        return read_imzml(path)
    if suffix in {".cdf", ".nc"}:
        return read_cdf(path)
    raise ValueError(
        f"unrecognized input path: {path}. Expected .imzML/.ibd, .cdf/.nc, "
        "or a directory of .cdf files."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dapple-apply-spec",
        description=(
            "Re-apply a saved DAPPLE pipeline (.spec.xml) to an input dataset. "
            "Writes harmonized .imzML, multipage .tif + CSV, and a fresh "
            ".spec.xml that records this run's provenance."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input dataset path (.imzML / .cdf file, or .cdf directory).",
    )
    parser.add_argument(
        "spec",
        type=Path,
        help="Saved .spec.xml file produced by a previous DAPPLE run.",
    )
    parser.add_argument(
        "-o",
        "--output-base",
        type=Path,
        default=None,
        help=(
            "Base path for outputs (no extension). Defaults to the input "
            "stem + '_dapple' next to the input."
        ),
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help=(
            "Override the RNG seed embedded in the .spec.xml. Useful for "
            "regenerating bootstrap CIs without re-running every node."
        ),
    )
    parser.add_argument(
        "--no-imzml",
        action="store_true",
        help="Skip imzML output (TIFF + .spec.xml only).",
    )
    parser.add_argument(
        "--no-tiff",
        action="store_true",
        help="Skip TIFF + CSV output (imzML + .spec.xml only).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v shows INFO; -vv shows DEBUG.",
    )
    args = parser.parse_args(argv)

    log_level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(log_level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        return _run(args)
    except Exception as e:  # noqa: BLE001 — top-level error → user-friendly exit
        print(f"dapple-apply-spec: {e}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    input_path: Path = args.input.expanduser()
    spec_path: Path = args.spec.expanduser()
    if not spec_path.exists():
        raise FileNotFoundError(f"spec.xml not found: {spec_path}")

    pipeline, ep, prov = read_spec_xml(spec_path)
    print(
        f"loaded spec {spec_path.name}: pipeline={len(pipeline.nodes)} nodes, "
        f"experiment={ep.instrument_family}/{ep.ionization}/"
        f"{ep.profile_or_centroided}/{ep.polarity}"
    )
    if args.rng_seed is not None:
        pipeline = pipeline.with_rng_seed(int(args.rng_seed))
        print(f"  rng_seed override: {args.rng_seed}")

    print(f"loading input {input_path}...")
    ds = _load_input(input_path)
    print(
        f"  loaded: {ds.n_pixels} pixels, grid {ds.grid_shape}, "
        f"backend={type(ds.backend).__name__}"
    )

    runner = PipelineRunner()

    def _progress(i: int, n: int, msg: str) -> None:
        print(f"  [{i}/{n}] {msg}")

    print("running pipeline...")
    result = runner.run(pipeline, ds, progress=_progress)

    # Pretty-printed per-node diagnostics with health rubric.
    print("")
    for line in format_diagnostics(result, title="Run diagnostics"):
        print(line)
    print("")

    # Resolve output base.
    out_base: Path = args.output_base or input_path.parent / (input_path.stem + "_dapple")

    out_files: list[Path] = []
    if not args.no_tiff:
        from dapple.data.dataset import PeakMatrix

        if not isinstance(result.output.backend, PeakMatrix):
            print(
                "warn: pipeline did not produce a PeakMatrix (no consensus alignment "
                "step?); skipping TIFF output.",
                file=sys.stderr,
            )
        else:
            tiff_path = out_base.with_suffix(".tif")
            tiff_result = write_hyperspectral_tiff(result.output, tiff_path)
            out_files.append(tiff_result.tiff_path)
            out_files.append(tiff_result.csv_path)
            print(
                f"wrote {tiff_result.tiff_path} ({tiff_result.n_pages} channels)"
                f" + {tiff_result.csv_path.name}"
            )

    if not args.no_imzml:
        imzml_path = out_base.with_suffix(".imzML")
        write_result = write_imzml(result.output, imzml_path)
        out_files.append(write_result.imzml_path)
        out_files.append(write_result.ibd_path)
        print(
            f"wrote {write_result.imzml_path} (+ {write_result.ibd_path.name}); "
            f"{write_result.n_spectra} spectra / {write_result.total_peaks} peaks"
        )

    # Always write a fresh .spec.xml: records this run's provenance, library
    # versions, and diagnostics.
    new_spec_path = out_base.with_suffix(".spec.xml")
    new_prov = make_provenance(
        plugin_version=_plugin_version(),
        input_dataset_hash=ds.hash(),
        declared_md5=ds.identity.declared_md5,
        notes=f"re-applied from {spec_path.name}",
    )
    write_spec_xml(
        new_spec_path,
        pipeline=pipeline,
        experiment_params=ep,
        provenance=new_prov,
        diagnostics=result.diagnostics,
    )
    out_files.append(new_spec_path)
    print(f"wrote {new_spec_path}")

    print(f"\ndone — {len(out_files)} output file(s) written.")
    return 0


def _plugin_version() -> str:
    try:
        from dapple import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
