"""Headless ROI-contrast and directed-axis analysis for harmonized MSI data.

The command deliberately consumes the persisted forms used by DAPPLE instead of
depending on a napari session: channel-aligned intensities are restored from a
harmonized imzML plus its ``.dapple-axis.json`` sidecar, while ROI polygons are
restored from a saved ``.spec.xml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from dapple.analysis import (
    DirectedAxis,
    analyze_axis_profiles,
    analyze_roi_enrichment,
    export_analysis_result,
    rasterize_rois,
)
from dapple.data.dataset import MSIDataset, PeakMatrix
from dapple.data.metadata import RoiDef
from dapple.io.imzml_reader import read_imzml
from dapple.io.spec_xml import read_spec_xml

_OVERLAP_POLICIES = ("error", "exclude", "first", "allow")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dapple-analyze-patterns",
        description=(
            "Analyze spatial enrichment in a harmonized DAPPLE imzML without "
            "opening napari. Use 'roi' for named ROI contrasts or 'axis' for "
            "patterns along a directed anatomical axis."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    roi = subparsers.add_parser(
        "roi",
        help="Compare one named ROI (or ROI union) with another.",
        description=(
            "Compute channel-wise descriptive summaries, log2 fold changes, "
            "pixel-level Welch tests, and FDR for two named ROI unions. ROI "
            "polygons are loaded from a DAPPLE .spec.xml."
        ),
    )
    _add_input_output_arguments(roi, default_stem_suffix="roi")
    roi.add_argument(
        "--roi-spec",
        type=Path,
        required=True,
        help="DAPPLE .spec.xml containing the ROI polygons.",
    )
    roi.add_argument(
        "--numerator",
        metavar="ROI",
        action="append",
        required=True,
        help="Numerator ROI name; repeat to analyze a union.",
    )
    roi.add_argument(
        "--denominator",
        metavar="ROI",
        action="append",
        required=True,
        help="Denominator ROI name; repeat to analyze a union.",
    )
    roi.add_argument(
        "--contrast-name",
        default=None,
        help="Optional label stored in the exported tables and manifest.",
    )
    _add_overlap_argument(roi)
    roi.add_argument(
        "--min-units",
        type=int,
        default=3,
        help="Minimum populated pixels per arm for Welch inference (default: 3).",
    )
    roi.add_argument(
        "--trim-fraction",
        type=float,
        default=0.1,
        help="Fraction trimmed from each intensity tail (default: 0.1).",
    )
    roi.add_argument(
        "--pseudocount-quantile",
        type=float,
        default=0.05,
        help="Positive-intensity quantile used for pseudocounts (default: 0.05).",
    )

    axis = subparsers.add_parser(
        "axis",
        help="Find gradients and localized patterns along a directed axis.",
        description=(
            "Project populated pixels onto a finite directed line, build binned "
            "channel profiles, and test monotonic trends by ordered-bin "
            "permutation. Coordinates use napari image order: Y X, zero-based."
        ),
    )
    _add_input_output_arguments(axis, default_stem_suffix="axis")
    axis.add_argument(
        "--start",
        nargs=2,
        type=float,
        metavar=("Y", "X"),
        required=True,
        help="Directed-axis start in zero-based image coordinates: Y X.",
    )
    axis.add_argument(
        "--end",
        nargs=2,
        type=float,
        metavar=("Y", "X"),
        required=True,
        help="Directed-axis end in zero-based image coordinates: Y X.",
    )
    axis.add_argument("--axis-name", default="developmental_axis")
    axis.add_argument("--start-label", default="start")
    axis.add_argument("--end-label", default="end")
    axis.add_argument(
        "--half-width",
        type=float,
        default=None,
        help="Optionally include only pixels this many pixels from the axis.",
    )
    axis.add_argument(
        "--roi-spec",
        type=Path,
        default=None,
        help="Optional .spec.xml supplying an ROI mask for the axis analysis.",
    )
    axis.add_argument(
        "--roi-name",
        metavar="ROI",
        action="append",
        default=None,
        help="Restrict to this ROI; repeat to use a union (requires --roi-spec).",
    )
    _add_overlap_argument(axis)
    axis.add_argument("--bins", type=int, default=20, help="Number of axis bins (default: 20).")
    axis.add_argument(
        "--min-pixels-per-bin",
        type=int,
        default=2,
        help="Minimum pixels for a bin to enter trend inference (default: 2).",
    )
    axis.add_argument(
        "--min-bins-for-trend",
        type=int,
        default=4,
        help="Minimum sufficiently populated bins for inference (default: 4).",
    )
    axis.add_argument(
        "--endpoint-fraction",
        type=float,
        default=0.2,
        help="Fraction of the axis assigned to each endpoint (default: 0.2).",
    )
    axis.add_argument(
        "--min-endpoint-pixels",
        type=int,
        default=3,
        help="Minimum pixels at each endpoint for enrichment labels (default: 3).",
    )
    axis.add_argument(
        "--permutations",
        type=int,
        default=999,
        help="Random bin orders; each receives a reversal partner (default: 999).",
    )
    axis.add_argument("--seed", type=int, default=0, help="Permutation RNG seed (default: 0).")
    axis.add_argument("--trim-fraction", type=float, default=0.1)
    axis.add_argument("--pseudocount-quantile", type=float, default=0.05)
    axis.add_argument("--q-threshold", type=float, default=0.05)
    axis.add_argument("--trend-threshold", type=float, default=0.5)
    axis.add_argument("--concentration-threshold", type=float, default=0.25)
    axis.add_argument("--endpoint-effect-threshold", type=float, default=1.0)
    return parser


def _add_input_output_arguments(
    parser: argparse.ArgumentParser, *, default_stem_suffix: str
) -> None:
    parser.add_argument(
        "input",
        type=Path,
        help="Harmonized .imzML or .ibd with its sibling .dapple-axis.json.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: INPUT_STEM_patterns beside the input).",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help=f"Output filename stem (default: INPUT_STEM_{default_stem_suffix}).",
    )
    parser.add_argument(
        "--chunk-channels",
        type=int,
        default=256,
        help="Channels processed together, useful for limiting memory (default: 256).",
    )


def _add_overlap_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overlap-policy",
        choices=_OVERLAP_POLICIES,
        default="error",
        help=(
            "How multiply covered pixels are handled: error (default), exclude, "
            "first ROI wins, or allow. ROI contrast arms must remain disjoint."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - command boundary gives concise diagnostics
        print(f"dapple-analyze-patterns: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    dataset = _load_harmonized(input_path)
    print(
        f"loaded {input_path.name}: {dataset.n_pixels} populated pixels, "
        f"{dataset.backend.n_peaks} harmonized channels, grid {dataset.grid_shape}"
    )

    if args.mode == "roi":
        result = _run_roi(dataset, args)
    elif args.mode == "axis":
        result = _run_axis(dataset, args)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown analysis mode {args.mode!r}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / f"{input_path.stem}_patterns"
    )
    stem = args.stem or f"{input_path.stem}_{args.mode}"
    exported = export_analysis_result(result, output_dir, stem=stem)

    print(f"analysis status: {result.inference_status}")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"manifest: {exported.manifest_path.resolve()}")
    for path in exported.table_paths:
        print(f"table: {path.resolve()}")
    return 0


def _load_harmonized(path: Path) -> MSIDataset:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if path.suffix.lower() not in {".imzml", ".ibd"}:
        raise ValueError(
            f"unsupported input {path}; expected a harmonized .imzML or .ibd file"
        )
    dataset = read_imzml(path)
    if not isinstance(dataset.backend, PeakMatrix):
        sidecar = path.with_suffix(".dapple-axis.json")
        raise RuntimeError(
            "input did not restore a harmonized PeakMatrix. Analyze the output of "
            "DAPPLE harmonization and keep its .imzML, .ibd, and "
            f"{sidecar.name} files together in the same directory."
        )
    return dataset


def _load_rois(spec_path: Path) -> tuple[RoiDef, ...]:
    path = spec_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ROI spec not found: {path}")
    try:
        _, _, provenance = read_spec_xml(path)
    except Exception as exc:  # noqa: BLE001 - add file-specific context
        raise ValueError(f"could not read ROI spec {path}: {exc}") from exc
    if not provenance.roi_definitions:
        raise ValueError(
            f"ROI spec {path} contains no ROI polygons. Save ROIs in DAPPLE and "
            "export a new .spec.xml."
        )
    return provenance.roi_definitions


def _validate_roi_names(rois: Sequence[RoiDef], requested: Sequence[str]) -> None:
    available = tuple(roi.name for roi in rois)
    unknown = tuple(dict.fromkeys(name for name in requested if name not in available))
    if unknown:
        raise ValueError(
            f"unknown ROI name(s): {', '.join(unknown)}. Available ROI names: "
            f"{', '.join(available)}"
        )


def _select_rois(rois: Sequence[RoiDef], requested: Sequence[str]) -> tuple[RoiDef, ...]:
    """Keep only requested definitions so unrelated overlaps cannot alter a run."""
    requested_set = set(requested)
    return tuple(roi for roi in rois if roi.name in requested_set)


def _run_roi(dataset: MSIDataset, args: argparse.Namespace):  # noqa: ANN202
    rois = _load_rois(args.roi_spec)
    requested = tuple(args.numerator) + tuple(args.denominator)
    _validate_roi_names(rois, requested)
    selected_rois = _select_rois(rois, requested)
    masks = rasterize_rois(
        dataset, selected_rois, overlap_policy=args.overlap_policy
    )
    counts = ", ".join(
        f"{name}={int(count)}"
        for name, count in zip(masks.names, masks.pixel_counts, strict=True)
    )
    print(f"ROI pixels ({args.overlap_policy} overlap policy): {counts}")
    return analyze_roi_enrichment(
        dataset,
        masks,
        tuple(args.numerator),
        tuple(args.denominator),
        contrast_name=args.contrast_name,
        min_units_per_group=args.min_units,
        trim_fraction=args.trim_fraction,
        pseudocount_quantile=args.pseudocount_quantile,
        chunk_channels=args.chunk_channels,
    )


def _run_axis(dataset: MSIDataset, args: argparse.Namespace):  # noqa: ANN202
    roi_names = tuple(args.roi_name or ())
    if bool(args.roi_spec) != bool(roi_names):
        raise ValueError(
            "axis ROI restriction requires both --roi-spec and at least one "
            "--roi-name; omit both to analyze all pixels on the axis"
        )

    include_mask: np.ndarray | None = None
    if args.roi_spec is not None:
        rois = _load_rois(args.roi_spec)
        _validate_roi_names(rois, roi_names)
        selected_rois = _select_rois(rois, roi_names)
        masks = rasterize_rois(
            dataset, selected_rois, overlap_policy=args.overlap_policy
        )
        include_mask = masks.union(roi_names)
        print(
            f"axis restricted to ROI union {', '.join(roi_names)}: "
            f"{int(include_mask.sum())} populated pixels "
            f"({args.overlap_policy} overlap policy)"
        )

    directed_axis = DirectedAxis(
        name=args.axis_name,
        start_yx=(float(args.start[0]), float(args.start[1])),
        end_yx=(float(args.end[0]), float(args.end[1])),
        start_label=args.start_label,
        end_label=args.end_label,
        half_width_px=args.half_width,
    )
    print(
        f"axis {directed_axis.name!r}: {directed_axis.start_label} "
        f"{directed_axis.start_yx} -> {directed_axis.end_label} "
        f"{directed_axis.end_yx}; bins={args.bins}, seed={args.seed}, "
        f"permutations={args.permutations}"
    )
    result = analyze_axis_profiles(
        dataset,
        directed_axis,
        include_mask=include_mask,
        selection_names=roi_names,
        n_bins=args.bins,
        min_pixels_per_bin=args.min_pixels_per_bin,
        min_bins_for_trend=args.min_bins_for_trend,
        endpoint_fraction=args.endpoint_fraction,
        min_endpoint_pixels=args.min_endpoint_pixels,
        n_permutations=args.permutations,
        rng_seed=args.seed,
        trim_fraction=args.trim_fraction,
        pseudocount_quantile=args.pseudocount_quantile,
        q_threshold=args.q_threshold,
        trend_threshold=args.trend_threshold,
        concentration_threshold=args.concentration_threshold,
        endpoint_effect_threshold=args.endpoint_effect_threshold,
        chunk_channels=args.chunk_channels,
    )
    print(
        f"included pixels={result.n_included_pixels}; bin counts="
        + ",".join(str(int(value)) for value in result.bin_counts)
    )
    return result


if __name__ == "__main__":
    sys.exit(main())
