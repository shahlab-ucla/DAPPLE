"""imzML reader.

Builds an MSIDataset with a PeakList backend from a paired .imzML/.ibd. Auto-extracts
ExperimentParams from CV terms in the XML when available; falls back to "unknown" so
the wizard can collect them.

We read .ibd contents through pyimzml's parsed offsets but build the CSR arrays
ourselves in one pass to avoid pyimzml's per-pixel decode-and-tuple overhead.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dapple.data.coords import coords_to_grid_index
from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix
from dapple.data.hashing import md5_file, sha256_file
from dapple.data.metadata import (
    DatasetIdentity,
    ExperimentParams,
    InstrumentFamily,
    Ionization,
    Mode,
    Polarity,
)
from dapple.data.metadata_source import MetadataSource, empty_source, merge_sources

logger = logging.getLogger(__name__)


def read_imzml(path: Path | str, *, lazy: bool = True) -> MSIDataset:
    """Open an imzML file and return an MSIDataset with a PeakList backend.

    `path` may point to either the .imzML or its .ibd sibling; the partner is located
    by matching base name.

    `lazy` is currently a placeholder — peaks are always materialized in RAM. An
    on-disk Zarr backing path is planned for datasets that exceed available memory.
    """
    from pyimzml.ImzMLParser import ImzMLParser

    imzml_path, ibd_path = _resolve_pair(Path(path))
    if not ibd_path.exists():
        raise FileNotFoundError(
            f"imzML at {imzml_path} but .ibd not found at {ibd_path}. "
            "Both files are required."
        )

    parser = ImzMLParser(str(imzml_path), parse_lib="lxml")
    try:
        coords_xy = _parse_coords(parser)
        grid_shape = _infer_grid_shape(parser, coords_xy)
        backend = _build_peaklist(parser)
        # pyimzml uses lxml.iterparse and *clears* spectrum elements from `parser.root`
        # as they're read. To recover per-spectrum CV terms (polarity, observed m/z range)
        # we reparse the (small) imzML XML once into a clean DOM.
        full_root = _parse_imzml_full(imzml_path)
        metadata, metadata_source = _extract_experiment_params(parser, full_root)
        declared_md5 = _read_declared_ibd_md5(parser, full_root)
    finally:
        # Close the .ibd file handle pyimzml keeps open as parser.m. Even though we've
        # materialized everything into RAM, leaving the handle dangling triggers
        # ResourceWarning on garbage collection on Windows.
        fh = getattr(parser, "m", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass

    actual_md5 = md5_file(ibd_path)
    if declared_md5 and declared_md5.lower() != actual_md5.lower():
        warnings.warn(
            f"declared .ibd MD5 ({declared_md5}) does not match actual ({actual_md5}). "
            "File may have been re-encoded or corrupted; loading anyway.",
            stacklevel=2,
        )

    identity = DatasetIdentity(
        source_path=str(imzml_path.resolve()),
        content_sha256=sha256_file(ibd_path),
        declared_md5=declared_md5,
        extra=(("ibd_size", str(ibd_path.stat().st_size)),),
    )

    # Discover any sibling metadata sidecar(s) and overlay them on the auto-detected
    # ExperimentParams. The wizard exposes the merged result with per-field source.
    metadata, metadata_source, sidecar_path = _apply_sidecars(
        imzml_path, metadata, metadata_source
    )

    # If this imzML was emitted by ``write_imzml`` from a PeakMatrix-backed
    # dataset, restore the dense matrix so the Channels Panel and Spectrum
    # Panel work on the reloaded data. Detection: a marker user-param in the
    # imzML XML, plus a sibling ``.dapple-axis.json`` carrying the shared axis.
    extra: dict[str, Any] = {
        "imzml_path": str(imzml_path),
        "ibd_path": str(ibd_path),
        "metadata_source": dict(metadata_source),
        "sidecar_path": str(sidecar_path) if sidecar_path else None,
    }
    if _is_dapple_harmonized_imzml(imzml_path, full_root):
        try:
            pm, prev = _restore_peakmatrix_from_peaklist(backend, imzml_path)
            backend = pm  # type: ignore[assignment]
            if prev is not None:
                extra["consensus_prevalence"] = prev
            logger.info(
                "detected DAPPLE-harmonized imzML: restored PeakMatrix backend "
                "(%d pixels × %d channels)",
                pm.n_pixels, pm.n_peaks,
            )
        except _HarmonizedRestoreError as e:
            warnings.warn(
                f"imzML is marked dapple-harmonized but PeakMatrix restoration "
                f"failed ({e}); falling back to PeakList backend.",
                stacklevel=2,
            )

    ds = MSIDataset(
        coords=coords_xy.astype(np.int32, copy=False),
        grid_shape=grid_shape,
        metadata=metadata,
        backend=backend,
        identity=identity,
        history=(),
        rois=(),
        rng_seed=0,
        extra=extra,
    )

    # Sanity check: every coord must map into the grid.
    coords_to_grid_index(coords_xy, grid_shape)

    if isinstance(backend, PeakList):
        total_peaks = int(np.asarray(backend.offsets[:])[-1])
    else:  # PeakMatrix (round-tripped from a harmonized save)
        total_peaks = int((np.asarray(backend.matrix[:]) > 0).sum())
    logger.info(
        "loaded imzML %s: %d spectra on %s grid, %d total peaks (%s)",
        imzml_path.name,
        ds.n_pixels,
        grid_shape,
        total_peaks,
        type(backend).__name__,
    )
    return ds


def _resolve_pair(path: Path) -> tuple[Path, Path]:
    """Given .imzML or .ibd, return (.imzML, .ibd) paths (case-insensitive ext match)."""
    suffix = path.suffix.lower()
    base = path.with_suffix("")
    if suffix == ".imzml":
        imzml = path
        ibd_candidates = [base.with_suffix(".ibd"), base.with_suffix(".IBD")]
    elif suffix == ".ibd":
        imzml_candidates = [base.with_suffix(".imzML"), base.with_suffix(".imzml")]
        for c in imzml_candidates:
            if c.exists():
                return c, path
        raise FileNotFoundError(f"no .imzML companion for {path}")
    else:
        raise ValueError(f"expected .imzML or .ibd, got {path}")
    for c in ibd_candidates:
        if c.exists():
            return imzml, c
    raise FileNotFoundError(f"no .ibd companion for {imzml}")


def _parse_coords(parser: Any) -> np.ndarray:
    """Extract (x, y) coords from a pyimzml parser; drop the z component."""
    coords = np.asarray(parser.coordinates, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(f"unexpected coords shape from pyimzml: {coords.shape}")
    return coords[:, :2]


def _infer_grid_shape(parser: Any, coords_xy: np.ndarray) -> tuple[int, int]:
    """Prefer the imzML CV-declared grid; fall back to coord max+1.

    imzML coords are 1-indexed, so grid height = max y, width = max x.
    """
    declared_x = _get_imaging_param(parser, "max count of pixels x")
    declared_y = _get_imaging_param(parser, "max count of pixels y")
    if declared_x is not None and declared_y is not None:
        try:
            return (int(declared_y), int(declared_x))
        except (TypeError, ValueError):
            pass
    return (int(coords_xy[:, 1].max()), int(coords_xy[:, 0].max()))


def _build_peaklist(parser: Any) -> PeakList:
    """Build CSR-style PeakList by reading the .ibd in one pass per pixel.

    pyimzml's parser exposes mzOffsets/intensityOffsets/mzLengths/intensityLengths
    plus an open self.m file handle. We use them directly to avoid the per-pixel
    tuple-creation overhead of `getspectrum`.
    """
    n = len(parser.coordinates)
    counts = np.asarray(parser.mzLengths, dtype=np.int64)
    if not np.array_equal(counts, np.asarray(parser.intensityLengths, dtype=np.int64)):
        raise ValueError("mzLengths != intensityLengths — malformed imzML")
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])

    mz_buf = np.empty(total, dtype=np.float64)
    int_buf = np.empty(total, dtype=np.float32)

    mz_dtype = _get_pyimzml_dtype(parser, "mz")
    int_dtype = _get_pyimzml_dtype(parser, "intensity")

    fh = parser.m  # raw file handle to .ibd
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if a == b:
            continue
        # m/z block
        fh.seek(int(parser.mzOffsets[i]))
        mz_raw = np.frombuffer(fh.read(int(parser.mzLengths[i]) * mz_dtype.itemsize), dtype=mz_dtype)
        mz_buf[a:b] = mz_raw.astype(np.float64, copy=False)
        # intensity block
        fh.seek(int(parser.intensityOffsets[i]))
        in_raw = np.frombuffer(
            fh.read(int(parser.intensityLengths[i]) * int_dtype.itemsize), dtype=int_dtype
        )
        int_buf[a:b] = in_raw.astype(np.float32, copy=False)

    return PeakList(mz=mz_buf, intensity=int_buf, offsets=offsets, n_pixels=n)


_PRECISION_ACCESSIONS: dict[str, np.dtype] = {
    "MS:1000519": np.dtype("<i4"),  # 32-bit integer
    "MS:1000521": np.dtype("<f4"),  # 32-bit float
    "MS:1000522": np.dtype("<i8"),  # 64-bit integer
    "MS:1000523": np.dtype("<f8"),  # 64-bit float
}


def _get_pyimzml_dtype(parser: Any, which: str) -> np.dtype:
    """Resolve the numpy dtype for m/z (`which='mz'`) or intensity arrays.

    Read directly from the referenceableParamGroup in the imzML XML — most reliable.
    Fall back to pyimzml's declared precision string if the group isn't found.
    """
    group_id = "mzArray" if which == "mz" else "intensities"
    try:
        ns = {"mz": "http://psi.hupo.org/ms/mzml"}
        for grp in parser.root.findall(".//mz:referenceableParamGroup", ns):
            if grp.get("id") not in (group_id, "intensityArray"):
                continue
            for cv in grp.findall("mz:cvParam", ns):
                acc = cv.get("accession")
                if acc in _PRECISION_ACCESSIONS:
                    return _PRECISION_ACCESSIONS[acc]
    except Exception:  # noqa: BLE001
        pass

    # Fallback: pyimzml's precision string attributes.
    attr = "mzPrecision" if which == "mz" else "intensityPrecision"
    prec = getattr(parser, attr, "")
    s = str(prec).lower()
    if "64" in s and "int" in s:
        return np.dtype("<i8")
    if "64" in s:
        return np.dtype("<f8")
    if "32" in s and "int" in s:
        return np.dtype("<i4")
    if "32" in s or s == "f":
        return np.dtype("<f4")
    raise ValueError(f"could not resolve dtype for {which!r} (precision={prec!r})")


def _read_declared_ibd_md5(parser: Any, full_root: Any | None = None) -> str | None:
    """The IMS:1000090 'ibd MD5' CV term, if present in the imzML header."""
    root = full_root if full_root is not None else parser.root
    try:
        ns = {"mz": "http://psi.hupo.org/ms/mzml"}
        for cv in root.findall(".//mz:fileDescription/mz:fileContent/mz:cvParam", ns):
            if cv.get("accession") == "IMS:1000090":
                return cv.get("value")
    except Exception:  # noqa: BLE001
        pass
    return None


def _get_imaging_param(parser: Any, name: str) -> Any:
    """Pull a CV term value from the scanSettings block, if present."""
    try:
        root = parser.root
        ns = {"mz": "http://psi.hupo.org/ms/mzml"}
        for cv in root.findall(".//mz:scanSettingsList//mz:cvParam", ns):
            if cv.get("name") == name:
                return cv.get("value")
    except Exception:  # noqa: BLE001
        return None
    return None


def _extract_experiment_params(
    parser: Any, full_root: Any | None = None
) -> tuple[ExperimentParams, MetadataSource]:
    """Best-effort extraction of ExperimentParams from imzML CV terms.

    `full_root` is a freshly parsed DOM that still contains spectrum elements (pyimzml
    drops those during its iterparse pass). Pass it in for accurate per-spectrum CV
    detection (polarity, observed m/z bounds).

    Returns the params plus a per-field source map: each field is tagged "imzml" if
    a CV term provided the value or "default" if we fell through to a placeholder.
    """
    source = empty_source()

    instrument, src = _detect_instrument_family(parser, full_root)
    source["instrument_family"] = src

    ionization, src = _detect_ionization(parser, full_root)
    source["ionization"] = src

    mode, src = _detect_mode(parser, full_root)
    source["profile_or_centroided"] = src

    polarity, src = _detect_polarity(parser, full_root)
    source["polarity"] = src

    (mz_min, mz_max), src = _detect_mz_range(parser, full_root)
    source["mz_min"] = src
    source["mz_max"] = src

    pixel_size, src = _detect_pixel_size(parser)
    source["pixel_size_um"] = src

    # sample_type and notes are never in the imzML file.
    source["sample_type"] = "default"
    source["notes"] = "default"

    ep = ExperimentParams(
        instrument_family=instrument,
        ionization=ionization,
        profile_or_centroided=mode,
        polarity=polarity,
        mz_min=mz_min,
        mz_max=mz_max,
        pixel_size_um=pixel_size,
        sample_type=None,
        notes="",
    )
    return ep, source


def _detect_instrument_family(
    parser: Any, full_root: Any | None = None
) -> tuple[InstrumentFamily, str]:
    has = _has_cv_factory(parser, full_root)
    # Most precise wins; fall through to families.
    if has("MS:1000484") or has("orbitrap"):  # MS:1000484 = orbitrap
        return "orbitrap", "imzml"
    if has("FT-ICR") or has("ion cyclotron resonance"):
        return "fticr", "imzml"
    has_tof = has("MS:1000084") or has("time-of-flight")
    has_reflectron = has("reflectron")
    if has_tof:
        if has("Q-TOF") or has("quadrupole time-of-flight"):
            return "qtof", "imzml"
        if has_reflectron:
            return "tof_reflectron", "imzml"
        if has("Bruker Daltonics flex series"):
            # Bruker flex line is reflectron-class for typical MSI usage.
            return "tof_reflectron", "imzml"
        return "tof_axial", "imzml"
    return "unknown", "default"


def _detect_ionization(
    parser: Any, full_root: Any | None = None
) -> tuple[Ionization, str]:
    has = _has_cv_factory(parser, full_root)
    if has("MS:1000075") or has("matrix-assisted laser desorption"):
        return "maldi", "imzml"
    if has("desorption electrospray"):
        return "desi", "imzml"
    if has("secondary ion"):
        return "sims", "imzml"
    if has("MS:1000073") or has("electrospray ionization"):
        return "esi", "imzml"
    return "unknown", "default"


def _detect_mode(parser: Any, full_root: Any | None = None) -> tuple[Mode, str]:
    has = _has_cv_factory(parser, full_root)
    if has("MS:1000128") or has("profile spectrum"):
        return "profile", "imzml"
    if has("MS:1000127") or has("centroid spectrum"):
        return "centroided", "imzml"
    return "unknown", "default"


def _detect_polarity(
    parser: Any, full_root: Any | None = None
) -> tuple[Polarity, str]:
    has = _has_cv_factory(parser, full_root)
    if has("MS:1000129") or has("negative scan"):
        return "negative", "imzml"
    if has("MS:1000130") or has("positive scan"):
        return "positive", "imzml"
    return "positive", "default"  # neither term present — confess to defaulting


def _detect_mz_range(
    parser: Any, full_root: Any | None = None
) -> tuple[tuple[float, float], str]:
    """Walk all spectra's lowest/highest observed m/z; aggregate to a global range."""
    root = full_root if full_root is not None else parser.root
    try:
        ns = {"mz": "http://psi.hupo.org/ms/mzml"}
        mins: list[float] = []
        maxs: list[float] = []
        for cv in root.findall(".//mz:spectrumList//mz:cvParam", ns):
            if cv.get("accession") == "MS:1000528":  # lowest observed m/z
                mins.append(float(cv.get("value", "nan")))
            elif cv.get("accession") == "MS:1000527":  # highest observed m/z
                maxs.append(float(cv.get("value", "nan")))
        if mins and maxs:
            return (float(min(mins)), float(max(maxs))), "imzml"
    except Exception:  # noqa: BLE001
        pass
    return (0.0, 1.0), "default"


def _detect_pixel_size(parser: Any) -> tuple[float | None, str]:
    val = _get_imaging_param(parser, "pixel size x")
    if val is None:
        return None, "default"
    try:
        return float(val), "imzml"
    except (TypeError, ValueError):
        return None, "default"


def _has_cv_factory(parser: Any, full_root: Any | None = None) -> Callable[[str], bool]:
    """Return a fast `has(needle)` for accession or name across every cvParam in the imzML.

    Prefers `full_root` (a freshly parsed DOM) since pyimzml's `parser.root` no longer
    contains spectrum elements after iterparse has cleared them.
    """
    root = full_root if full_root is not None else parser.root
    accessions: set[str] = set()
    names: set[str] = set()
    try:
        ns = {"mz": "http://psi.hupo.org/ms/mzml"}
        for cv in root.findall(".//mz:cvParam", ns):
            acc = cv.get("accession")
            nm = cv.get("name")
            if acc:
                accessions.add(acc)
            if nm:
                names.add(nm.lower())
    except Exception:  # noqa: BLE001
        pass

    def has(needle: str) -> bool:
        if needle in accessions:
            return True
        nl = needle.lower()
        return any(nl in n for n in names)

    return has


def _parse_imzml_full(imzml_path: Path) -> Any:
    """Parse an imzML file fully into a clean lxml DOM (with all spectrum elements).

    Used to recover per-spectrum CV terms (polarity, observed m/z bounds) that pyimzml
    discards during its memory-efficient iterparse pass.
    """
    from lxml import etree

    with imzml_path.open("rb") as f:
        return etree.parse(f).getroot()


def _apply_sidecars(
    imzml_path: Path,
    metadata: ExperimentParams,
    source: MetadataSource,
) -> tuple[ExperimentParams, MetadataSource, Path | None]:
    """Look for sibling metadata sidecars next to the .imzML and overlay them.

    Tries (in order): `<stem>.metadata.json`, `<stem>.experiment.json`,
    `<stem>.spec.xml`. The first one found wins. Sidecar fields override auto-detected
    values; auto-detected fields stay if the sidecar omits them.
    """
    from dapple.io._sidecar import (
        load_json_sidecar,
        load_spec_xml_sidecar,
    )

    base = imzml_path.with_suffix("")
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
        except Exception:  # noqa: BLE001 — corrupt or wrong-namespace spec; ignore
            pass

    return metadata, source, None


# --- DAPPLE-harmonized round-trip detection ---------------------------------------------


_HARMONIZED_AXIS_SIDECAR_SUFFIX = ".dapple-axis.json"
"""Filename suffix used by ``write_imzml`` for the axis sidecar. Kept literal here
(rather than imported from imzml_writer) so the reader doesn't pull the writer
into the load path."""


class _HarmonizedRestoreError(Exception):
    """Raised when an imzML is marked dapple-harmonized but the sidecar / data
    needed to reconstruct the PeakMatrix is missing or inconsistent."""


def _is_dapple_harmonized_imzml(imzml_path: Path, full_root: Any | None) -> bool:
    """True if the imzML carries DAPPLE's ``dapple-harmonized`` user-param marker.

    The marker is emitted by ``write_imzml`` on the spectrum referenceableParamGroup
    when the source dataset had a PeakMatrix backend. This check is fast (the
    full DOM is already parsed for ExperimentParams extraction).
    """
    if full_root is None:
        return False
    try:
        # Search for ``<userParam name="dapple-harmonized" value="true"/>``.
        # Using local-name() to avoid needing the mzML namespace prefix here.
        results = full_root.xpath(
            "//*[local-name()='userParam' "
            "and @name='dapple-harmonized' and @value='true']"
        )
        return bool(results)
    except Exception:  # noqa: BLE001
        return False


def _restore_peakmatrix_from_peaklist(
    backend: PeakList, imzml_path: Path
) -> tuple[PeakMatrix, np.ndarray | None]:
    """Reconstruct a dense (n_pixels, n_channels) PeakMatrix from the per-pixel
    sparse PeakList plus a sibling ``.dapple-axis.json`` sidecar.

    The sidecar carries the shared m/z axis; for each pixel we look up which
    channel each non-zero peak's m/z corresponds to and write its intensity into
    the matrix. If the sidecar is missing or inconsistent (channel count or
    ibd_md5 mismatch), raises ``_HarmonizedRestoreError``.
    """
    import json

    sidecar_path = imzml_path.with_suffix(_HARMONIZED_AXIS_SIDECAR_SUFFIX)
    if not sidecar_path.exists():
        raise _HarmonizedRestoreError(
            f"sidecar {sidecar_path.name} not found next to {imzml_path.name}"
        )
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise _HarmonizedRestoreError(
            f"could not parse {sidecar_path.name}: {e}"
        ) from e
    shared_axis = np.asarray(sidecar.get("shared_mz_axis", []), dtype=np.float64)
    if shared_axis.size == 0:
        raise _HarmonizedRestoreError(
            f"{sidecar_path.name} has no 'shared_mz_axis'"
        )
    n_axis = int(sidecar.get("n_channels", shared_axis.size))
    if n_axis != shared_axis.size:
        raise _HarmonizedRestoreError(
            f"{sidecar_path.name} 'n_channels' ({n_axis}) does not match "
            f"axis length ({shared_axis.size})"
        )

    # Build the dense matrix by binning each peak's m/z onto the shared axis.
    # The writer wrote per-pixel non-zero entries with mz values drawn from the
    # shared axis, so each peak's m/z should match an axis entry exactly. We use
    # ``searchsorted`` for an O(log N) lookup per peak; tolerate sub-ULP float
    # round-trip drift by allowing the nearest neighbour within 1e-9 relative.
    sorted_idx = np.argsort(shared_axis)
    sorted_axis = shared_axis[sorted_idx]
    peak_mz_all = np.asarray(backend.mz[:])
    peak_int_all = np.asarray(backend.intensity[:])
    offsets = np.asarray(backend.offsets[:])
    n_pixels = backend.n_pixels
    matrix = np.zeros((n_pixels, n_axis), dtype=np.float32)
    if peak_mz_all.size > 0:
        # For each peak, find the closest shared-axis entry. The writer wrote
        # axis values verbatim into pixel mz arrays, so right_idx-1 is usually
        # an exact match. Fall back to the closer of right/left if not.
        right = np.searchsorted(sorted_axis, peak_mz_all)
        right = np.clip(right, 0, n_axis - 1)
        left = np.clip(right - 1, 0, n_axis - 1)
        d_right = np.abs(sorted_axis[right] - peak_mz_all)
        d_left = np.abs(sorted_axis[left] - peak_mz_all)
        nearest = np.where(d_left <= d_right, left, right)
        # Map sorted-index back to original axis order.
        axis_idx = sorted_idx[nearest]
        # Per-peak pixel id (gather offsets → repeat).
        per_peak_pixel = np.repeat(
            np.arange(n_pixels, dtype=np.int64),
            np.diff(offsets).astype(np.int64),
        )
        np.maximum.at(matrix, (per_peak_pixel, axis_idx), peak_int_all.astype(np.float32, copy=False))

    pm = PeakMatrix(matrix=matrix, mz_axis=shared_axis)
    prev_list = sidecar.get("consensus_prevalence")
    prev = None
    if prev_list is not None:
        prev = np.asarray(prev_list, dtype=np.float64)
        if prev.size != n_axis:
            prev = None  # corrupted; ignore but keep the matrix
    return pm, prev


# --- napari plugin entry point -----------------------------------------------------------


def napari_get_reader(path: str | list[str]) -> Any:
    """npe2 reader entry point — returns a callable or None."""
    if isinstance(path, list):
        path = path[0]
    p = Path(path)
    if p.suffix.lower() not in {".imzml", ".ibd"}:
        return None
    return _napari_reader


def _napari_reader(path: str) -> list[tuple[Any, dict[str, Any], str]]:
    ds = read_imzml(Path(path))
    img = ds.project("tic")
    name = f"{Path(path).stem}:tic"
    metadata = {"msi_dataset": ds, "projection": "tic"}
    return [(img, {"name": name, "metadata": metadata, "colormap": "viridis"}, "image")]
