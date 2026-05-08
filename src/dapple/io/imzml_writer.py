"""imzML writer (processed mode).

Round-trips an `MSIDataset` back to an imzML/.ibd pair so harmonized data can be
re-loaded by other tools (Cardinal, METASPACE, etc.) and by us via `read_imzml`.

Currently supports:
- PeakList input (per-pixel variable-length peaks): write directly.
- PeakMatrix input (post-consensus dense): re-emit only nonzero entries per pixel
  so the file stays compact.

For PeakMatrix input the writer also emits a sibling ``<base>.dapple-axis.json``
sidecar carrying the shared m/z axis and per-channel prevalence, plus a
``<userParam name="dapple-harmonized">true</userParam>`` marker on each spectrum
group. ``read_imzml`` detects the marker and reconstructs the PeakMatrix backend
on load, so a saved-then-reloaded harmonized dataset reappears with its
consensus channels intact (the Channels Panel and Spectrum Panel both depend on
this).

Continuous-mode imzML output (one common m/z axis for every pixel) is not yet supported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent

import numpy as np

from dapple.data.dataset import MSIDataset, PeakList, PeakMatrix


# Sidecar filename suffix for the harmonized m/z axis.
_HARMONIZED_AXIS_SIDECAR_SUFFIX = ".dapple-axis.json"

# Marker user-param emitted on each spectrum's referenceableParamGroup so the
# reader can detect harmonized files even if the sidecar was lost in transit.
_HARMONIZED_USER_PARAM = (
    '<userParam name="dapple-harmonized" value="true"/>'
)


@dataclass(frozen=True)
class ImzMLWriteResult:
    imzml_path: Path
    ibd_path: Path
    n_spectra: int
    total_peaks: int
    ibd_md5: str


def write_imzml(
    ds: MSIDataset,
    path: Path | str,
    *,
    write_zero_intensity: bool = False,
) -> ImzMLWriteResult:
    """Write `ds` to a paired .imzML/.ibd at `path` (processed mode).

    `write_zero_intensity=False` (default) skips zero-intensity entries from a
    PeakMatrix so the .ibd doesn't bloat with empty cells. Set True to round-trip
    a dense matrix verbatim (useful for QA when comparing pre/post-write hashes).
    """
    base = Path(path)
    if base.suffix.lower() == ".imzml":
        imzml_path = base
        ibd_path = base.with_suffix(".ibd")
    else:
        imzml_path = base.with_suffix(".imzML")
        ibd_path = base.with_suffix(".ibd")

    coords = ds.coords  # (n_pixels, 2) — (x, y)
    n_pixels = coords.shape[0]
    is_harmonized = isinstance(ds.backend, PeakMatrix)

    # Build per-pixel arrays.
    pixel_mz: list[np.ndarray] = []
    pixel_int: list[np.ndarray] = []
    if isinstance(ds.backend, PeakList):
        for i in range(n_pixels):
            mz_i, int_i = ds.backend.pixel(i)
            pixel_mz.append(mz_i.astype(np.float64, copy=False))
            pixel_int.append(int_i.astype(np.float32, copy=False))
    elif isinstance(ds.backend, PeakMatrix):
        mz_axis = np.asarray(ds.backend.mz_axis[:])
        matrix = np.asarray(ds.backend.matrix[:])
        for i in range(n_pixels):
            row = matrix[i, :]
            if write_zero_intensity:
                pixel_mz.append(mz_axis.astype(np.float64, copy=False))
                pixel_int.append(row.astype(np.float32, copy=False))
            else:
                nz = np.flatnonzero(row)
                pixel_mz.append(mz_axis[nz].astype(np.float64, copy=False))
                pixel_int.append(row[nz].astype(np.float32, copy=False))
    else:  # pragma: no cover
        raise TypeError(f"unknown backend {type(ds.backend)}")

    # Write the .ibd with a 16-byte UUID header (zeroed; pyimzml accepts this) and
    # concatenated (mz, intensity) pairs. Track offsets/lengths for the XML.
    offsets_mz: list[int] = []
    offsets_int: list[int] = []
    lengths_mz: list[int] = []
    lengths_int: list[int] = []

    h = hashlib.md5(usedforsecurity=False)

    def _write(buf: bytes, fh: Any) -> None:
        fh.write(buf)
        h.update(buf)

    with ibd_path.open("wb") as f:
        uuid_bytes = b"\x00" * 16
        _write(uuid_bytes, f)
        for mz_i, int_i in zip(pixel_mz, pixel_int, strict=True):
            offsets_mz.append(f.tell())
            mzb = mz_i.astype("<f8", copy=False).tobytes()
            _write(mzb, f)
            lengths_mz.append(int(mz_i.size))
            offsets_int.append(f.tell())
            inb = int_i.astype("<f4", copy=False).tobytes()
            _write(inb, f)
            lengths_int.append(int(int_i.size))
    ibd_md5 = h.hexdigest().upper()

    # Determine grid shape & instrument-config CV terms from metadata.
    height, width = ds.grid_shape
    polarity_cv = (
        '<cvParam accession="MS:1000129" cvRef="MS" name="negative scan"/>'
        if ds.metadata.polarity == "negative"
        else '<cvParam accession="MS:1000130" cvRef="MS" name="positive scan"/>'
    )
    mode_ref = (
        '<cvParam accession="MS:1000127" cvRef="MS" name="centroid spectrum"/>'
        if ds.metadata.profile_or_centroided == "centroided"
        else '<cvParam accession="MS:1000128" cvRef="MS" name="profile spectrum"/>'
    )

    # Build the spectrum block list.
    spectra_xml_parts: list[str] = []
    for i in range(n_pixels):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        n_mz = lengths_mz[i]
        n_int = lengths_int[i]
        mz_block = (
            f'        <binaryDataArray encodedLength="0">\n'
            f'          <referenceableParamGroupRef ref="mzArray"/>\n'
            f'          <cvParam accession="IMS:1000103" cvRef="IMS" name="external array length" value="{n_mz}"/>\n'
            f'          <cvParam accession="IMS:1000104" cvRef="IMS" name="external encoded length" value="{n_mz * 8}"/>\n'
            f'          <cvParam accession="IMS:1000102" cvRef="IMS" name="external offset" value="{offsets_mz[i]}"/>\n'
            f"          <binary/>\n"
            f"        </binaryDataArray>"
        )
        int_block = (
            f'        <binaryDataArray encodedLength="0">\n'
            f'          <referenceableParamGroupRef ref="intensities"/>\n'
            f'          <cvParam accession="IMS:1000103" cvRef="IMS" name="external array length" value="{n_int}"/>\n'
            f'          <cvParam accession="IMS:1000104" cvRef="IMS" name="external encoded length" value="{n_int * 4}"/>\n'
            f'          <cvParam accession="IMS:1000102" cvRef="IMS" name="external offset" value="{offsets_int[i]}"/>\n'
            f"          <binary/>\n"
            f"        </binaryDataArray>"
        )
        spectrum_xml = (
            f'      <spectrum id="spectrum={i}" index="{i}" defaultArrayLength="0">\n'
            f'        <referenceableParamGroupRef ref="spectrum"/>\n'
            f"        {polarity_cv}\n"
            f"        <scanList count=\"1\">\n"
            f'          <cvParam accession="MS:1000795" cvRef="MS" name="no combination"/>\n'
            f"          <scan>\n"
            f'            <cvParam accession="IMS:1000050" cvRef="IMS" name="position x" value="{x}"/>\n'
            f'            <cvParam accession="IMS:1000051" cvRef="IMS" name="position y" value="{y}"/>\n'
            f"          </scan>\n"
            f"        </scanList>\n"
            f'        <binaryDataArrayList count="2">\n'
            f"{mz_block}\n"
            f"{int_block}\n"
            f"        </binaryDataArrayList>\n"
            f"      </spectrum>"
        )
        spectra_xml_parts.append(spectrum_xml)
    spectra_xml = "\n".join(spectra_xml_parts)

    pixel_size = (
        f'<cvParam accession="IMS:1000046" cvRef="IMS" name="pixel size x" value="{ds.metadata.pixel_size_um}"/>\n'
        f'      <cvParam accession="IMS:1000047" cvRef="IMS" name="pixel size y" value="{ds.metadata.pixel_size_um}"/>'
        if ds.metadata.pixel_size_um
        else ""
    )
    instrument_cv = _instrument_cv_block(ds)

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
      {mode_ref}
      {_HARMONIZED_USER_PARAM if is_harmonized else ""}
    </referenceableParamGroup>
  </referenceableParamGroupList>
  <scanSettingsList count="1">
    <scanSettings id="scanSettings0">
      <cvParam accession="IMS:1000042" cvRef="IMS" name="max count of pixels x" value="{width}"/>
      <cvParam accession="IMS:1000043" cvRef="IMS" name="max count of pixels y" value="{height}"/>
      {pixel_size}
    </scanSettings>
  </scanSettingsList>
  {instrument_cv}
  <run defaultInstrumentConfigurationRef="instrumentConfiguration0" id="run0">
    <spectrumList count="{n_pixels}">
{indent(spectra_xml, '')}
    </spectrumList>
  </run>
</mzML>
"""
    imzml_path.write_text(xml, encoding="utf-8")

    # If we wrote a PeakMatrix-backed dataset, drop a sidecar so the reader can
    # reconstruct the dense matrix on load. The marker user-param in the XML is
    # the primary signal; this sidecar carries the data needed to rebuild.
    if is_harmonized:
        pm: PeakMatrix = ds.backend  # type: ignore[assignment]
        prev = ds.extra.get("consensus_prevalence")
        sidecar = {
            "version": 1,
            "shared_mz_axis": np.asarray(pm.mz_axis[:]).tolist(),
            "n_channels": int(pm.n_peaks),
            "consensus_prevalence": (
                np.asarray(prev, dtype=np.float64).tolist()
                if prev is not None and len(prev) == pm.n_peaks
                else None
            ),
            "ibd_md5": ibd_md5,
        }
        sidecar_path = imzml_path.with_suffix(_HARMONIZED_AXIS_SIDECAR_SUFFIX)
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return ImzMLWriteResult(
        imzml_path=imzml_path,
        ibd_path=ibd_path,
        n_spectra=n_pixels,
        total_peaks=int(sum(lengths_mz)),
        ibd_md5=ibd_md5,
    )


def _instrument_cv_block(ds: MSIDataset) -> str:
    """A minimal instrumentConfiguration that records the family + ionization."""
    family_cv = {
        "tof_axial": '<cvParam accession="MS:1000084" cvRef="MS" name="time-of-flight"/>',
        "tof_reflectron": '<cvParam accession="MS:1000084" cvRef="MS" name="time-of-flight"/>',
        "qtof": '<cvParam accession="MS:1000084" cvRef="MS" name="time-of-flight"/>',
        "orbitrap": '<cvParam accession="MS:1000484" cvRef="MS" name="orbitrap"/>',
        "fticr": '<cvParam accession="MS:1000079" cvRef="MS" name="fourier transform ion cyclotron resonance mass spectrometer"/>',
        "unknown": "",
    }.get(ds.metadata.instrument_family, "")
    ion_cv = {
        "maldi": '<cvParam accession="MS:1000075" cvRef="MS" name="matrix-assisted laser desorption ionization"/>',
        "desi": '<cvParam accession="MS:1000406" cvRef="MS" name="desorption electrospray ionization"/>',
        "esi": '<cvParam accession="MS:1000073" cvRef="MS" name="electrospray ionization"/>',
        "sims": '<cvParam accession="MS:1000074" cvRef="MS" name="secondary ion mass spectrometry"/>',
        "unknown": "",
    }.get(ds.metadata.ionization, "")
    return (
        f'<instrumentConfigurationList count="1">\n'
        f'    <instrumentConfiguration id="instrumentConfiguration0">\n'
        f"      {ion_cv}\n"
        f"      {family_cv}\n"
        f"    </instrumentConfiguration>\n"
        f"  </instrumentConfigurationList>"
    )
