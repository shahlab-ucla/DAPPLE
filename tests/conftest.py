"""Shared pytest fixtures.

`make_synth_imzml` writes a tiny synthetic processed-mode imzML+ibd pair into a tmp_path
so I/O tests can run without committing binary fixtures.

`real_dataset_dir` and the `real_data` marker gate the integration tests that read the
user's actual datasets (jerboa, root1, Boone) — set MSI_REAL_DATA=1 to run them.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


@dataclass(frozen=True)
class SynthSpec:
    n_x: int = 5
    n_y: int = 5
    peaks_mz: tuple[float, ...] = (200.0, 250.0, 300.0, 350.0, 400.0)
    base_intensity: float = 1000.0
    jitter_ppm: float = 5.0
    seed: int = 42


@pytest.fixture
def real_data_enabled() -> bool:
    return os.environ.get("MSI_REAL_DATA", "0") == "1"


@pytest.fixture
def real_dataset_dir() -> Path:
    """Where the user's datasets live. Override via MSI_DATA_DIR."""
    p = Path(os.environ.get("MSI_DATA_DIR", r"C:\Users\pavak\MSI\datasets"))
    if not p.exists():
        pytest.skip(f"real dataset directory not found at {p}")
    return p


@pytest.fixture
def synth_centroided(tmp_path: Path) -> Path:
    """Write a tiny processed-mode imzML+ibd. Returns the .imzML path."""
    spec = SynthSpec()
    return write_synth_imzml(tmp_path, spec)


@pytest.fixture
def synth_lcms_cdf(tmp_path: Path) -> Path:
    """Write a tiny ANDI-MS NetCDF file emulating a 1-D LC-MS run.

    5 scans, each with 4 peaks at known m/z. Negative ESI to match the supplied
    real Finnigan file.
    """
    return _write_synth_andims_cdf(tmp_path / "synth_lcms.cdf", n_scans=5, peaks_per_scan=4)


@pytest.fixture
def synth_cdf_image_dir(tmp_path: Path) -> Path:
    """Write a tiny multi-file CDF image: 4 files × 6 scans.

    Each file uses the same set of 4 m/z values per scan; reading the directory should
    yield a 4×6 grid (axis_along_files='y'). Used to exercise cdf_image_reader without
    requiring the 150 MB Boone dataset.
    """
    out = tmp_path / "synth_image"
    out.mkdir()
    for i in range(1, 5):
        _write_synth_andims_cdf(
            out / f"line-{i}.cdf", n_scans=6, peaks_per_scan=4, label_offset=i * 0.001
        )
    return out


def _write_synth_andims_cdf(
    path: Path,
    *,
    n_scans: int,
    peaks_per_scan: int,
    label_offset: float = 0.0,
) -> Path:
    """Build an ANDI-MS NetCDF file with the variables our reader needs.

    `label_offset` is added to every m/z so different files in a multi-file fixture
    don't have identical content (helps catch hash collisions).
    """
    import netCDF4 as nc

    rng = np.random.default_rng(0xC0FFEE + int(label_offset * 1000))
    base_mz = np.array([200.0, 300.0, 400.0, 500.0], dtype=np.float64)[:peaks_per_scan]

    point_count = np.full(n_scans, peaks_per_scan, dtype=np.int32)
    scan_index = np.empty(n_scans, dtype=np.int32)
    scan_index[0] = 0
    np.cumsum(point_count[:-1], out=scan_index[1:])

    total = int(point_count.sum())
    mass_values = np.empty(total, dtype=np.float64)
    intensity_values = np.empty(total, dtype=np.int32)
    for s in range(n_scans):
        a, b = int(scan_index[s]), int(scan_index[s] + point_count[s])
        jitter = rng.normal(0, 5e-6, size=peaks_per_scan)
        mass_values[a:b] = base_mz + label_offset + base_mz * jitter
        intensity_values[a:b] = (1000 + rng.integers(-50, 50, size=peaks_per_scan)).astype(np.int32)

    nc_ds = nc.Dataset(str(path), "w", format="NETCDF3_CLASSIC")
    try:
        nc_ds.createDimension("scan_number", n_scans)
        nc_ds.createDimension("point_number", total)
        for dim in (2, 4, 8, 16, 32, 64, 80, 128, 256):
            nc_ds.createDimension(f"_{dim}_byte_string", dim)
        # Required variables.
        v = nc_ds.createVariable("scan_index", "i4", ("scan_number",))
        v[:] = scan_index
        v = nc_ds.createVariable("point_count", "i4", ("scan_number",))
        v[:] = point_count
        v = nc_ds.createVariable("scan_acquisition_time", "f8", ("scan_number",))
        v[:] = np.linspace(0.1, 1.0, n_scans)
        v = nc_ds.createVariable("total_intensity", "f8", ("scan_number",))
        v[:] = np.full(n_scans, 1000.0 * peaks_per_scan)
        v = nc_ds.createVariable("mass_range_min", "f8", ("scan_number",))
        v[:] = np.full(n_scans, float(base_mz.min()))
        v = nc_ds.createVariable("mass_range_max", "f8", ("scan_number",))
        v[:] = np.full(n_scans, float(base_mz.max()))
        v = nc_ds.createVariable("mass_values", "f8", ("point_number",))
        v[:] = mass_values
        v = nc_ds.createVariable("intensity_values", "i4", ("point_number",))
        v[:] = intensity_values
        # Global attributes that our extractor reads.
        nc_ds.setncattr("experiment_type", "Centroided Mass Spectrum")
        nc_ds.setncattr("test_ionization_mode", "Electrospray Ionization")
        nc_ds.setncattr("test_ionization_polarity", "Negative Polarity")
        nc_ds.setncattr("test_detector_type", "Conversion Dynode Electron Multiplier")
        nc_ds.setncattr("source_file_format", "Finnigan")
        nc_ds.setncattr("global_mass_min", float(base_mz.min()) - 1.0)
        nc_ds.setncattr("global_mass_max", float(base_mz.max()) + 1.0)
        nc_ds.setncattr("number_of_scans", n_scans)
        nc_ds.setncattr("dataset_completeness", "C1+C2")
    finally:
        nc_ds.close()
    return path


def write_synth_imzml(out_dir: Path, spec: SynthSpec) -> Path:
    """Generate a paired .imzML/.ibd pair under out_dir for testing.

    Each pixel gets a peak at every mz in `spec.peaks_mz`, with intensity drawn from
    a Gaussian centered on `base_intensity` (no zero-suppression — sparseness is added
    later by tests if needed). Mass jitter ~ N(0, jitter_ppm) per (pixel, peak).
    """
    rng = np.random.default_rng(spec.seed)

    base_name = "synth_centroided"
    imzml_path = out_dir / f"{base_name}.imzML"
    ibd_path = out_dir / f"{base_name}.ibd"

    n_pixels = spec.n_x * spec.n_y
    n_peaks = len(spec.peaks_mz)

    # Generate per-pixel arrays
    mz_arrays: list[np.ndarray] = []
    int_arrays: list[np.ndarray] = []
    for _ in range(n_pixels):
        jitter = rng.normal(0.0, spec.jitter_ppm * 1e-6, size=n_peaks)
        mz_pix = np.array(spec.peaks_mz, dtype=np.float64) * (1.0 + jitter)
        int_pix = rng.normal(spec.base_intensity, spec.base_intensity * 0.1, size=n_peaks).astype(
            np.float32
        )
        int_pix = np.clip(int_pix, 1.0, None)
        mz_arrays.append(mz_pix)
        int_arrays.append(int_pix)

    # Write .ibd:  16-byte UUID header + concatenated (mz, intensity) pairs.
    uuid_bytes = b"\x00" * 16  # zeroed UUID is fine for tests
    offsets_mz: list[int] = []
    offsets_int: list[int] = []
    lengths_mz: list[int] = []
    lengths_int: list[int] = []

    with ibd_path.open("wb") as f:
        f.write(uuid_bytes)
        for mz_arr, int_arr in zip(mz_arrays, int_arrays, strict=True):
            offsets_mz.append(f.tell())
            f.write(mz_arr.astype("<f8").tobytes())
            lengths_mz.append(mz_arr.size)
            offsets_int.append(f.tell())
            f.write(int_arr.astype("<f4").tobytes())
            lengths_int.append(int_arr.size)

    ibd_md5 = hashlib.md5(ibd_path.read_bytes(), usedforsecurity=False).hexdigest().upper()

    # Build minimal imzML XML.
    spectra_xml = []
    flat_idx = 0
    for iy in range(1, spec.n_y + 1):
        for ix in range(1, spec.n_x + 1):
            n = lengths_mz[flat_idx]
            spectra_xml.append(
                f"""    <spectrum id="spectrum={flat_idx}" index="{flat_idx}" defaultArrayLength="0">
      <referenceableParamGroupRef ref="spectrum"/>
      <cvParam accession="MS:1000129" cvRef="MS" name="negative scan"/>
      <scanList count="1">
        <cvParam accession="MS:1000795" cvRef="MS" name="no combination"/>
        <scan>
          <cvParam accession="IMS:1000050" cvRef="IMS" name="position x" value="{ix}"/>
          <cvParam accession="IMS:1000051" cvRef="IMS" name="position y" value="{iy}"/>
        </scan>
      </scanList>
      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="0">
          <referenceableParamGroupRef ref="mzArray"/>
          <cvParam accession="IMS:1000103" cvRef="IMS" name="external array length" value="{n}"/>
          <cvParam accession="IMS:1000104" cvRef="IMS" name="external encoded length" value="{n * 8}"/>
          <cvParam accession="IMS:1000102" cvRef="IMS" name="external offset" value="{offsets_mz[flat_idx]}"/>
          <binary/>
        </binaryDataArray>
        <binaryDataArray encodedLength="0">
          <referenceableParamGroupRef ref="intensities"/>
          <cvParam accession="IMS:1000103" cvRef="IMS" name="external array length" value="{n}"/>
          <cvParam accession="IMS:1000104" cvRef="IMS" name="external encoded length" value="{n * 4}"/>
          <cvParam accession="IMS:1000102" cvRef="IMS" name="external offset" value="{offsets_int[flat_idx]}"/>
          <binary/>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>"""
            )
            flat_idx += 1

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mzML version="1.1.0" xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd">
  <cvList count="3">
    <cv URI="https://purl.obolibrary.org/obo/ms.obo" fullName="PSI MS" id="MS" version="4.1.35"/>
    <cv URI="https://purl.obolibrary.org/obo/uo.obo" fullName="Unit Ontology" id="UO" version="2019-03-29"/>
    <cv URI="https://www.maldi-msi.org/download/imzml/imagingMS.obo" fullName="Imaging MS Ontology" id="IMS" version="0.9.1"/>
  </cvList>
  <fileDescription>
    <fileContent>
      <cvParam accession="IMS:1000080" cvRef="IMS" name="universally unique identifier" value="00000000-0000-0000-0000-000000000000"/>
      <cvParam accession="IMS:1000031" cvRef="IMS" name="processed"/>
      <cvParam accession="IMS:1000090" cvRef="IMS" name="ibd MD5" value="{ibd_md5}"/>
      <cvParam accession="MS:1000294" cvRef="MS" name="mass spectrum"/>
    </fileContent>
  </fileDescription>
  <referenceableParamGroupList count="3">
    <referenceableParamGroup id="mzArray">
      <cvParam accession="MS:1000514" cvRef="MS" name="m/z array" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z"/>
      <cvParam accession="MS:1000523" cvRef="MS" name="64-bit float"/>
      <cvParam accession="MS:1000576" cvRef="MS" name="no compression"/>
      <cvParam accession="IMS:1000101" cvRef="IMS" name="external data" value="true"/>
    </referenceableParamGroup>
    <referenceableParamGroup id="intensities">
      <cvParam accession="MS:1000515" cvRef="MS" name="intensity array" unitAccession="MS:1000131" unitCvRef="MS" unitName="number of detector counts"/>
      <cvParam accession="MS:1000521" cvRef="MS" name="32-bit float"/>
      <cvParam accession="MS:1000576" cvRef="MS" name="no compression"/>
      <cvParam accession="IMS:1000101" cvRef="IMS" name="external data" value="true"/>
    </referenceableParamGroup>
    <referenceableParamGroup id="spectrum">
      <cvParam accession="MS:1000294" cvRef="MS" name="mass spectrum"/>
      <cvParam accession="MS:1000127" cvRef="MS" name="centroid spectrum"/>
    </referenceableParamGroup>
  </referenceableParamGroupList>
  <scanSettingsList count="1">
    <scanSettings id="scanSettings0">
      <cvParam accession="IMS:1000042" cvRef="IMS" name="max count of pixels x" value="{spec.n_x}"/>
      <cvParam accession="IMS:1000043" cvRef="IMS" name="max count of pixels y" value="{spec.n_y}"/>
      <cvParam accession="IMS:1000046" cvRef="IMS" name="pixel size x" value="50"/>
      <cvParam accession="IMS:1000047" cvRef="IMS" name="pixel size y" value="50"/>
    </scanSettings>
  </scanSettingsList>
  <instrumentConfigurationList count="1">
    <instrumentConfiguration id="instrumentConfiguration0">
      <cvParam accession="MS:1000084" cvRef="MS" name="time-of-flight"/>
      <cvParam accession="MS:1000075" cvRef="MS" name="matrix-assisted laser desorption ionization"/>
    </instrumentConfiguration>
  </instrumentConfigurationList>
  <run defaultInstrumentConfigurationRef="instrumentConfiguration0" id="run0">
    <spectrumList count="{n_pixels}">
{chr(10).join(spectra_xml)}
    </spectrumList>
  </run>
</mzML>
"""
    imzml_path.write_text(xml, encoding="utf-8")
    # mark unused locals to satisfy linters in case struct is removed later
    _ = struct
    return imzml_path
