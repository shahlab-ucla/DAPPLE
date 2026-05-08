"""Multi-file ANDI-MS NetCDF reader for MSI imaging.

When MSI data is exported from Xcalibur as one ANDI-MS .cdf per scan-line, the image
has to be stitched together from the directory. This is the case for the supplied
`Boone cdf/` DESI dataset: 140 files, each with 62 scans, forming a 62×140 image.

The reader's job:
    1. Locate and naturally-sort the .cdf files in a directory.
    2. Confirm they all expose ANDI-MS variables and have a consistent scan count
       (or raise a helpful error if not).
    3. Stream each file once, materializing into a flat CSR PeakList covering the
       whole image (so all downstream operators see a single MSIDataset).
    4. Auto-derive a `CdfLayout` (axis_along_files, scans_per_line, serpentine,
       start_index) and let callers override.
    5. Extract ExperimentParams from the FIRST file's global attributes (assume
       acquisition consistency across files; warn if mass-range or polarity differ).

The reader treats Finnigan ESI in this multi-file context as DESI by default — that's
the only way Xcalibur emits one ANDI-MS file per line for an MSI experiment. Users
can override on the wizard's ParamsPage.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dapple.data.dataset import MSIDataset, PeakList
from dapple.data.hashing import combine_hashes, sha256_file
from dapple.data.metadata import DatasetIdentity
from dapple.data.metadata_source import MetadataSource
from dapple.io.cdf_reader import (
    _attr_dict,
    _build_peaklist_from_andims,
    _extract_andims_params,
    _is_andims_format,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CdfLayout:
    """How the directory of .cdf files maps onto an MSI raster.

    `axis_along_files` — "y" means each .cdf is one row (scans run along x within a
        file, files run along y); "x" means each is one column.
    `serpentine` — alternate-file rows are reversed (scan back-and-forth raster).
    `start_index` — first integer suffix in filenames (usually 1).
    `scans_per_line` — pinned scan count; None means infer from min across files.
    `pattern` — glob to find files; default `*.cdf`.
    """

    pattern: str = "*.cdf"
    axis_along_files: Literal["x", "y"] = "y"
    serpentine: bool = False
    start_index: int = 1
    scans_per_line: int | None = None


_INT_SUFFIX_RE = re.compile(r"(\d+)(?=\D*$)")


def infer_layout(dir_path: Path | str) -> tuple[CdfLayout, list[str]]:
    """Look at a directory of .cdf files and propose a CdfLayout, plus warnings.

    Conservative defaults: assume `axis_along_files='y'`, no serpentine, and let the
    layout `scans_per_line` come out as the (constant) per-file scan count. The
    warnings list flags things the wizard's LoadPage should surface to the user.
    """
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(p)
    files = _list_sorted_cdf_files(p, pattern="*.cdf")
    if not files:
        raise FileNotFoundError(f"no .cdf files found under {p}")

    warnings_out: list[str] = []
    suffixes = [_extract_int_suffix(f) for f in files]
    if any(s is None for s in suffixes):
        warnings_out.append(
            "Some filenames lack an integer suffix; sorting by filename string instead. "
            "If files aren't in the expected order, rename them like `sample-1.cdf`, "
            "`sample-2.cdf`, ..."
        )
        start_index = 1
    else:
        ints = [int(s) for s in suffixes if s is not None]
        start_index = min(ints)
        if sorted(set(ints)) != list(range(start_index, start_index + len(ints))):
            warnings_out.append(
                f"Integer suffixes are not contiguous (gaps detected). Range: "
                f"{min(ints)}–{max(ints)}, count={len(ints)}; expected "
                f"{max(ints) - min(ints) + 1}."
            )

    # Inspect first file to pick scans_per_line.
    import netCDF4 as nc

    nc0 = nc.Dataset(str(files[0]), "r")
    try:
        if not _is_andims_format(nc0):
            raise ValueError(
                f"{files[0].name} is not in ANDI-MS format; cannot infer layout."
            )
        scans_per_line = len(nc0.dimensions["scan_number"])
    finally:
        nc0.close()

    layout = CdfLayout(
        pattern="*.cdf",
        axis_along_files="y",
        serpentine=False,
        start_index=start_index,
        scans_per_line=scans_per_line,
    )
    return layout, warnings_out


def read_cdf_image(
    dir_path: Path | str,
    layout: CdfLayout | None = None,
) -> MSIDataset:
    """Open a directory of .cdf files as a single 2-D MSI dataset.

    `layout=None` → call `infer_layout` and use the result.
    Returns an `MSIDataset` with a `PeakList` backend.
    """
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(p)

    if layout is None:
        layout, warns = infer_layout(p)
        for w in warns:
            warnings.warn(w, stacklevel=2)

    files = _list_sorted_cdf_files(p, pattern=layout.pattern)
    if not files:
        raise FileNotFoundError(f"no .cdf files found under {p} with pattern {layout.pattern}")

    n_files = len(files)
    expected_scans = layout.scans_per_line
    if expected_scans is None:
        # Infer from the minimum across files (conservative; we'll truncate to this).
        expected_scans = _min_scan_count(files)

    # First pass: gather per-file (n_scans, total_peaks) so we can size flat arrays.
    import netCDF4 as nc

    # We allocate big flat arrays then fill per file. Total peaks is sum of per-file
    # peak counts, but we may truncate per-file scans to expected_scans.
    per_file_peakcount: list[int] = []
    per_file_scancount: list[int] = []
    per_file_first_attrs: dict[str, Any] | None = None
    per_file_mz_extents: list[tuple[float, float]] = []
    per_file_polarities: list[str] = []

    for fi, fp in enumerate(files):
        nc_ds = nc.Dataset(str(fp), "r")
        try:
            if not _is_andims_format(nc_ds):
                raise ValueError(f"{fp.name} is not in ANDI-MS format.")
            ns = len(nc_ds.dimensions["scan_number"])
            if ns < expected_scans:
                warnings.warn(
                    f"{fp.name} has only {ns} scans (< expected {expected_scans}); "
                    "this row will be partial.",
                    stacklevel=2,
                )
            per_file_scancount.append(min(ns, expected_scans))
            point_count = np.asarray(nc_ds.variables["point_count"][:expected_scans], dtype=np.int64)
            per_file_peakcount.append(int(point_count.sum()))
            attrs = _attr_dict(nc_ds)
            if fi == 0:
                per_file_first_attrs = attrs
            per_file_polarities.append(str(attrs.get("test_ionization_polarity", "")).lower())
            mlo = float(attrs.get("global_mass_min", 0.0) or 0.0)
            mhi = float(attrs.get("global_mass_max", 0.0) or 0.0)
            per_file_mz_extents.append((mlo, mhi))
        finally:
            nc_ds.close()

    # Sanity: warn on heterogeneity.
    pol_set = {p_ for p_ in per_file_polarities if p_}
    if len(pol_set) > 1:
        warnings.warn(
            f"Files report different polarities: {sorted(pol_set)}. Using the first file's "
            "polarity for ExperimentParams; the wizard's ParamsPage can override.",
            stacklevel=2,
        )
    # Some vendors (e.g. older Xcalibur exports) report `global_mass_min=0` even though
    # the actual scan covers the configured mass range. Treat zero as "unset" rather
    # than as a real mass bound when checking heterogeneity.
    mzlows = {round(e[0], 3) for e in per_file_mz_extents if e[0] > 0}
    mzhighs = {round(e[1], 3) for e in per_file_mz_extents if e[1] > 0}
    if len(mzlows) > 1 or len(mzhighs) > 1:
        warnings.warn(
            f"Files report different m/z ranges (min set={sorted(mzlows)}, max set="
            f"{sorted(mzhighs)}). Using the union as the dataset bounds.",
            stacklevel=2,
        )

    n_pixels = int(sum(per_file_scancount))
    total_peaks = int(sum(per_file_peakcount))
    logger.info(
        "stitching %d .cdf files (%d scans/file expected) → %d pixels, %d total peaks",
        n_files,
        expected_scans,
        n_pixels,
        total_peaks,
    )

    mz_buf = np.empty(total_peaks, dtype=np.float64)
    int_buf = np.empty(total_peaks, dtype=np.float32)
    offsets = np.empty(n_pixels + 1, dtype=np.int64)
    offsets[0] = 0
    coords = np.empty((n_pixels, 2), dtype=np.int32)

    flat_peak_idx = 0
    flat_pix_idx = 0
    file_hashes: list[str] = []
    for fi, fp in enumerate(files):
        # Position of this file along the file-axis (1-indexed for imzML compatibility).
        line_idx = fi + 1
        nc_ds = nc.Dataset(str(fp), "r")
        try:
            ns = per_file_scancount[fi]
            point_count = np.asarray(nc_ds.variables["point_count"][:ns], dtype=np.int64)
            scan_index = np.asarray(nc_ds.variables["scan_index"][:ns], dtype=np.int64)
            # Read only the slice covering scans [0:ns].
            file_total = int(scan_index[ns - 1] + point_count[ns - 1]) if ns > 0 else 0
            mz_file = np.asarray(
                nc_ds.variables["mass_values"][:file_total], dtype=np.float64
            )
            int_file = np.asarray(
                nc_ds.variables["intensity_values"][:file_total], dtype=np.float32
            )

            # Walk scans in order — possibly reversed for serpentine.
            scan_order = list(range(ns))
            scan_along_axis_reversed = layout.serpentine and (fi % 2 == 1)
            if scan_along_axis_reversed:
                scan_order = list(reversed(scan_order))

            for scan_pos, src_scan in enumerate(scan_order):
                a = int(scan_index[src_scan])
                n = int(point_count[src_scan])
                b = a + n
                offsets[flat_pix_idx + 1] = offsets[flat_pix_idx] + n
                if n > 0:
                    mz_buf[flat_peak_idx : flat_peak_idx + n] = mz_file[a:b]
                    int_buf[flat_peak_idx : flat_peak_idx + n] = int_file[a:b]
                # Coords: scan_pos along the in-line axis, line_idx along the across-files axis.
                if layout.axis_along_files == "y":
                    coords[flat_pix_idx, 0] = scan_pos + 1  # x
                    coords[flat_pix_idx, 1] = line_idx  # y
                else:
                    coords[flat_pix_idx, 0] = line_idx  # x
                    coords[flat_pix_idx, 1] = scan_pos + 1  # y
                flat_peak_idx += n
                flat_pix_idx += 1
        finally:
            nc_ds.close()
        file_hashes.append(sha256_file(fp))

    if flat_peak_idx != total_peaks:
        raise RuntimeError(
            f"internal: peak buffer fill mismatch ({flat_peak_idx} != {total_peaks})"
        )
    if flat_pix_idx != n_pixels:
        raise RuntimeError(
            f"internal: pixel count mismatch ({flat_pix_idx} != {n_pixels})"
        )

    # Build dataset.
    backend = PeakList(mz=mz_buf, intensity=int_buf, offsets=offsets, n_pixels=n_pixels)

    # Grid shape from coords.
    if layout.axis_along_files == "y":
        grid_shape = (n_files, expected_scans)  # (H, W) = (rows=files, cols=scans)
    else:
        grid_shape = (expected_scans, n_files)

    # Build ExperimentParams from the first file (assume homogeneity already warned about).
    nc_first = __import__("netCDF4").Dataset(str(files[0]), "r")
    try:
        metadata, metadata_source = _extract_andims_params(nc_first, is_imaging=True)
    finally:
        nc_first.close()

    # Override mz range with the union we computed.
    union_lo = min(e[0] for e in per_file_mz_extents if e[0] > 0) if any(e[0] > 0 for e in per_file_mz_extents) else metadata.mz_min
    union_hi = max(e[1] for e in per_file_mz_extents if e[1] > 0) if any(e[1] > 0 for e in per_file_mz_extents) else metadata.mz_max
    from dataclasses import replace as dataclass_replace

    metadata = dataclass_replace(metadata, mz_min=float(union_lo), mz_max=float(union_hi))
    # The union came from the same andims attributes — keep the source tag.
    if metadata_source.get("mz_min") == "default":
        metadata_source["mz_min"] = "andims"
    if metadata_source.get("mz_max") == "default":
        metadata_source["mz_max"] = "andims"

    # Sidecar discovery: look for a metadata.json / experiment.json in the directory,
    # or a <dirname>.spec.xml next to the directory.
    metadata, metadata_source, sidecar_path = _apply_image_sidecars(
        p, metadata, metadata_source
    )

    identity = DatasetIdentity(
        source_path=str(p.resolve()),
        content_sha256=combine_hashes(*file_hashes),
        declared_md5=None,
        extra=(
            ("n_files", str(n_files)),
            ("scans_per_line", str(expected_scans)),
            ("axis_along_files", layout.axis_along_files),
            ("serpentine", str(layout.serpentine)),
        ),
    )

    return MSIDataset(
        coords=coords,
        grid_shape=grid_shape,
        metadata=metadata,
        backend=backend,
        identity=identity,
        history=(),
        rois=(),
        rng_seed=0,
        extra={
            "cdf_dir": str(p),
            "cdf_layout": layout,
            "is_imaging": True,
            "andims_first_file_attrs": per_file_first_attrs or {},
            "metadata_source": dict(metadata_source),
            "sidecar_path": str(sidecar_path) if sidecar_path else None,
        },
    )


def _apply_image_sidecars(
    dir_path: Path,
    metadata: Any,
    source: MetadataSource,
) -> tuple[Any, MetadataSource, Path | None]:
    """Look for sidecars next to a multi-file CDF imaging directory.

    Tries (in order):
      - ``<dir>/metadata.json``
      - ``<dir>/experiment.json``
      - ``<dir.parent>/<dir.name>.metadata.json``
      - ``<dir.parent>/<dir.name>.spec.xml``
    """
    from dapple.io._sidecar import (
        load_json_sidecar,
        load_spec_xml_sidecar,
    )

    json_candidates = [
        dir_path / "metadata.json",
        dir_path / "experiment.json",
        dir_path.parent / f"{dir_path.name}.metadata.json",
    ]
    for candidate in json_candidates:
        if candidate.exists():
            metadata, source = load_json_sidecar(candidate, metadata, source)
            return metadata, source, candidate
    spec_candidate = dir_path.parent / f"{dir_path.name}.spec.xml"
    if spec_candidate.exists():
        try:
            metadata, source = load_spec_xml_sidecar(spec_candidate, metadata, source)
            return metadata, source, spec_candidate
        except Exception:  # noqa: BLE001
            pass
    return metadata, source, None


def _list_sorted_cdf_files(dir_path: Path, *, pattern: str) -> list[Path]:
    """Glob for files and naturally-sort them by their last integer suffix.

    Files without an integer suffix sort lexicographically at the end.
    """
    files = sorted(dir_path.glob(pattern))
    keyed: list[tuple[tuple[int, int, str], Path]] = []
    for f in files:
        s = _extract_int_suffix(f)
        if s is None:
            keyed.append(((1, 0, f.name), f))
        else:
            keyed.append(((0, int(s), f.name), f))
    keyed.sort(key=lambda kp: kp[0])
    return [f for _, f in keyed]


def _extract_int_suffix(p: Path) -> str | None:
    m = _INT_SUFFIX_RE.search(p.stem)
    return m.group(1) if m else None


def _min_scan_count(files: list[Path]) -> int:
    """Minimum scan_number dimension across files (used when scans_per_line=None)."""
    import netCDF4 as nc

    counts: list[int] = []
    for fp in files:
        nc_ds = nc.Dataset(str(fp), "r")
        try:
            if "scan_number" in nc_ds.dimensions:
                counts.append(len(nc_ds.dimensions["scan_number"]))
        finally:
            nc_ds.close()
    if not counts:
        raise ValueError("no scan_number dimension found in any .cdf file")
    return min(counts)
