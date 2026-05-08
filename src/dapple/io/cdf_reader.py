"""Single-file ANDI-MS NetCDF reader.

ANDI-MS NetCDF (.cdf) files store mass spectra produced by Finnigan/Thermo Xcalibur
and other vendors. A single .cdf is almost always a 1-D run (LC-MS or GC-MS), not an
imaging dataset; the multi-file MSI variant lives in `cdf_image_reader.py`.

This reader is intentionally cautious: it loads the file as a 1-D `MSIDataset` (one
"pixel" per scan) and emits a warning that the file is not imaging. It is useful as
an I/O smoke test and as a way to inspect a single LC-MS scan in napari, but it is
not what you want for true MSI data.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from dapple.data.dataset import MSIDataset, PeakList
from dapple.data.hashing import sha256_file
from dapple.data.metadata import (
    DatasetIdentity,
    ExperimentParams,
    InstrumentFamily,
    Ionization,
    Mode,
    Polarity,
)
from dapple.data.metadata_source import MetadataSource, empty_source

logger = logging.getLogger(__name__)


def read_cdf(path: Path | str) -> MSIDataset:
    """Open a single ANDI-MS NetCDF file and return a 1-D MSIDataset.

    The returned dataset is annotated with `extra['is_imaging'] = False` and a console
    warning is emitted; downstream code should check this flag before treating it as
    a 2-D image.
    """
    import netCDF4 as nc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    nc_ds = nc.Dataset(str(p), "r")
    try:
        if not _is_andims_format(nc_ds):
            raise ValueError(
                f"{p.name} does not look like an ANDI-MS NetCDF "
                "(missing mass_values / intensity_values / scan_index variables)."
            )
        backend = _build_peaklist_from_andims(nc_ds)
        metadata, metadata_source = _extract_andims_params(nc_ds, is_imaging=False)
        attrs = _attr_dict(nc_ds)
    finally:
        nc_ds.close()

    n_scans = backend.n_pixels
    coords = np.stack(
        [
            np.arange(n_scans, dtype=np.int32),
            np.zeros(n_scans, dtype=np.int32),
        ],
        axis=1,
    )
    grid_shape = (1, n_scans)
    identity = DatasetIdentity(
        source_path=str(p.resolve()),
        content_sha256=sha256_file(p),
        declared_md5=None,
        extra=tuple((k, str(v)) for k, v in attrs.items() if isinstance(v, (str, int, float))),
    )

    warnings.warn(
        f"{p.name} is a single-file ANDI-MS NetCDF (likely LC-MS chromatography), not "
        "an MSI image. Loaded as a 1-D pseudo-dataset with one pixel per scan. For "
        "multi-file MSI imaging, use read_cdf_image() with a directory of .cdf files.",
        stacklevel=2,
    )

    # Sidecar discovery: <stem>.metadata.json or <stem>.spec.xml next to the .cdf.
    metadata, metadata_source, sidecar_path = _apply_cdf_sidecars(
        p, metadata, metadata_source
    )

    ds = MSIDataset(
        coords=coords,
        grid_shape=grid_shape,
        metadata=metadata,
        backend=backend,
        identity=identity,
        history=(),
        rois=(),
        rng_seed=0,
        extra={
            "cdf_path": str(p),
            "is_imaging": False,
            "andims_attrs": attrs,
            "metadata_source": dict(metadata_source),
            "sidecar_path": str(sidecar_path) if sidecar_path else None,
        },
    )
    logger.info("loaded ANDI-MS NetCDF %s: %d scans, 1-D pseudo-dataset", p.name, n_scans)
    return ds


def _is_andims_format(nc_ds: Any) -> bool:
    required = {"mass_values", "intensity_values", "scan_index"}
    return required.issubset(set(nc_ds.variables.keys()))


def _build_peaklist_from_andims(nc_ds: Any) -> PeakList:
    """Build a CSR PeakList from a single ANDI-MS file's flat mass/intensity arrays.

    `scan_index[i]` is the start position; the count for scan i is either
    `point_count[i]` (when present) or `scan_index[i+1] - scan_index[i]` (with the last
    scan ending at `len(mass_values)`).
    """
    mass = np.asarray(nc_ds.variables["mass_values"][:], dtype=np.float64)
    raw_intensity = nc_ds.variables["intensity_values"][:]
    intensity = np.asarray(raw_intensity, dtype=np.float32)

    scan_index = np.asarray(nc_ds.variables["scan_index"][:], dtype=np.int64)
    n_scans = scan_index.size

    if "point_count" in nc_ds.variables:
        point_count = np.asarray(nc_ds.variables["point_count"][:], dtype=np.int64)
    else:
        point_count = np.empty(n_scans, dtype=np.int64)
        point_count[:-1] = np.diff(scan_index)
        point_count[-1] = mass.size - int(scan_index[-1])

    offsets = np.empty(n_scans + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(point_count, out=offsets[1:])

    expected_total = int(offsets[-1])
    if mass.size != expected_total:
        raise ValueError(
            f"mass_values length {mass.size} != sum(point_count) {expected_total}"
        )

    return PeakList(mz=mass, intensity=intensity, offsets=offsets, n_pixels=n_scans)


def _extract_andims_params(
    nc_ds: Any, *, is_imaging: bool
) -> tuple[ExperimentParams, MetadataSource]:
    """Best-effort ExperimentParams from ANDI-MS global attributes.

    `is_imaging` flips the default ionization for ESI: in 1-D runs we keep "esi", but
    when this file is part of an imaging directory the same Xcalibur ESI string usually
    means DESI (desorption electrospray imaging).
    """
    source = empty_source()
    attrs = _attr_dict(nc_ds)

    instrument, src = _detect_instrument_andims(attrs)
    source["instrument_family"] = src
    ionization, src = _detect_ionization_andims(attrs, is_imaging=is_imaging)
    source["ionization"] = src
    mode, src = _detect_mode_andims(attrs)
    source["profile_or_centroided"] = src
    polarity, src = _detect_polarity_andims(attrs)
    source["polarity"] = src
    (mz_min, mz_max), src = _detect_mz_range_andims(nc_ds, attrs)
    source["mz_min"] = src
    source["mz_max"] = src
    source["pixel_size_um"] = "default"
    source["sample_type"] = "default"
    source["notes"] = "default"

    ep = ExperimentParams(
        instrument_family=instrument,
        ionization=ionization,
        profile_or_centroided=mode,
        polarity=polarity,
        mz_min=mz_min,
        mz_max=mz_max,
        pixel_size_um=None,
        sample_type=None,
        notes="",
    )
    return ep, source


def _apply_cdf_sidecars(
    cdf_path: Path,
    metadata: ExperimentParams,
    source: MetadataSource,
) -> tuple[ExperimentParams, MetadataSource, Path | None]:
    """Look for sibling sidecars next to a single .cdf file."""
    from dapple.io._sidecar import (
        load_json_sidecar,
        load_spec_xml_sidecar,
    )

    base = cdf_path.with_suffix("")
    json_candidates = [
        base.with_suffix(".metadata.json"),
        base.with_suffix(".experiment.json"),
    ]
    for candidate in json_candidates:
        if candidate.exists():
            metadata, source = load_json_sidecar(candidate, metadata, source)
            return metadata, source, candidate
    spec_candidate = base.with_suffix(".spec.xml")
    if spec_candidate.exists():
        try:
            metadata, source = load_spec_xml_sidecar(spec_candidate, metadata, source)
            return metadata, source, spec_candidate
        except Exception:  # noqa: BLE001
            pass
    return metadata, source, None


def _attr_dict(nc_ds: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in nc_ds.ncattrs():
        v = nc_ds.getncattr(k)
        if hasattr(v, "item") and v.size == 1:
            try:
                v = v.item()
            except Exception:  # noqa: BLE001
                v = str(v)
        out[k] = v
    return out


def _detect_instrument_andims(attrs: dict[str, Any]) -> tuple[InstrumentFamily, str]:
    """ANDI-MS doesn't encode analyzer family directly. Best-guess from source file format."""
    fmt = str(attrs.get("source_file_format", "")).lower()
    inlet = str(attrs.get("test_ms_inlet", "")).lower()
    detector = str(attrs.get("test_detector_type", "")).lower()
    blob = f"{fmt} {inlet} {detector}"
    if "orbitrap" in blob:
        return "orbitrap", "andims"
    if "fticr" in blob or "ion cyclotron" in blob:
        return "fticr", "andims"
    if "tof" in blob:
        return "tof_reflectron", "andims"
    return "unknown", "default"


def _detect_ionization_andims(
    attrs: dict[str, Any], *, is_imaging: bool
) -> tuple[Ionization, str]:
    s = str(attrs.get("test_ionization_mode", "")).lower()
    if "matrix-assisted" in s or "maldi" in s:
        return "maldi", "andims"
    if "desorption electrospray" in s or "desi" in s:
        return "desi", "andims"
    if "secondary ion" in s or "sims" in s:
        return "sims", "andims"
    if "electrospray" in s:
        # In MSI imaging context, plain "Electrospray Ionization" written by Xcalibur
        # almost always means DESI (the imaging variant); in 1-D LC-MS context it's
        # truly direct ESI.
        return ("desi" if is_imaging else "esi"), "andims"
    return "unknown", "default"


def _detect_mode_andims(attrs: dict[str, Any]) -> tuple[Mode, str]:
    s = str(attrs.get("experiment_type", "")).lower()
    if "centroided" in s or "centroid" in s:
        return "centroided", "andims"
    if "profile" in s or "continuum" in s:
        return "profile", "andims"
    return "unknown", "default"


def _detect_polarity_andims(attrs: dict[str, Any]) -> tuple[Polarity, str]:
    s = str(attrs.get("test_ionization_polarity", "")).lower()
    if "negative" in s:
        return "negative", "andims"
    if "positive" in s:
        return "positive", "andims"
    return "positive", "default"


def _detect_mz_range_andims(
    nc_ds: Any, attrs: dict[str, Any]
) -> tuple[tuple[float, float], str]:
    """Prefer per-scan `mass_range_min/max` if present; fall back to `global_mass_*`."""
    if "mass_range_min" in nc_ds.variables and "mass_range_max" in nc_ds.variables:
        try:
            lo = float(np.asarray(nc_ds.variables["mass_range_min"][:]).min())
            hi = float(np.asarray(nc_ds.variables["mass_range_max"][:]).max())
            if hi > lo > 0:
                return (lo, hi), "andims"
        except Exception:  # noqa: BLE001
            pass
    lo = float(attrs.get("global_mass_min", 0.0) or 0.0)
    hi = float(attrs.get("global_mass_max", 0.0) or 0.0)
    if hi > lo > 0:
        return (lo, hi), "andims"
    return (0.0, 1.0), "default"


# --- napari plugin entry point -----------------------------------------------------------


def napari_get_reader(path: str | list[str]) -> Any:
    """npe2 reader entry point — returns a callable or None.

    Accepts either a single .cdf file (1-D) or a directory of .cdf files (multi-file
    MSI imaging via cdf_image_reader.read_cdf_image).
    """
    if isinstance(path, list):
        path = path[0]
    p = Path(path)
    if p.is_dir():
        # Directory: multi-file MSI imaging path.
        if any(p.glob("*.cdf")) or any(p.glob("*.CDF")):
            return _napari_dir_reader
        return None
    if p.suffix.lower() in {".cdf", ".nc"}:
        return _napari_file_reader
    return None


def _napari_file_reader(path: str) -> list[tuple[Any, dict[str, Any], str]]:
    ds = read_cdf(Path(path))
    img = ds.project("tic")
    name = f"{Path(path).stem}:tic"
    metadata = {"msi_dataset": ds, "projection": "tic"}
    return [(img, {"name": name, "metadata": metadata, "colormap": "viridis"}, "image")]


def _napari_dir_reader(path: str) -> list[tuple[Any, dict[str, Any], str]]:
    from dapple.io.cdf_image_reader import read_cdf_image

    ds = read_cdf_image(Path(path))
    img = ds.project("tic")
    name = f"{Path(path).name}:tic"
    metadata = {"msi_dataset": ds, "projection": "tic"}
    return [(img, {"name": name, "metadata": metadata, "colormap": "viridis"}, "image")]
