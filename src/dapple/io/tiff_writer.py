"""Multipage float32 TIFF writer for harmonized MSI hyperspectral cubes.

After consensus alignment we have a `PeakMatrix`-backed `MSIDataset`: one m/z channel
per consensus peak, each renderable as a (H, W) image. We persist this as:

- A multipage TIFF, one page per channel, dtype float32, photometric `minisblack`.
  Each page's `description` tag carries a JSON blob with the channel's m/z, ppm
  tolerance, prevalence, and a hash linking back to the operator chain.
- A sidecar CSV (`<base>_channels.csv`) with one row per page so spreadsheet tools
  can browse the channel table without parsing TIFF tags.

Reading back: the TIFF page-description JSON is enough for `tifffile` users to
recover the channel m/z; the CSV is a convenience.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from dapple.data.coords import project_to_grid
from dapple.data.dataset import MSIDataset, PeakMatrix


@dataclass(frozen=True)
class TiffWriteResult:
    tiff_path: Path
    csv_path: Path
    n_pages: int
    image_shape: tuple[int, int]


def write_hyperspectral_tiff(
    ds: MSIDataset,
    path: Path | str,
    *,
    compression: str | None = "deflate",
    write_csv: bool = True,
    extra_per_channel: dict[str, np.ndarray] | None = None,
) -> TiffWriteResult:
    """Write the dataset's PeakMatrix to a multipage float32 TIFF + CSV sidecar.

    `extra_per_channel` lets callers attach extra columns to the CSV / TIFF metadata
    (e.g. Moran's I values or cohort dataset prevalence). Each value must
    be a 1-D array of length `n_peaks`.
    """
    if not isinstance(ds.backend, PeakMatrix):
        raise ValueError(
            "write_hyperspectral_tiff requires a PeakMatrix-backed dataset (post "
            "consensus alignment); got a PeakList. Run kde_consensus_alignment first."
        )
    pm: PeakMatrix = ds.backend
    n_pixels = pm.n_pixels
    n_peaks = pm.n_peaks
    height, width = ds.grid_shape
    if n_peaks == 0:
        raise ValueError("PeakMatrix has zero peaks; nothing to write.")
    extra_per_channel = extra_per_channel or {}
    for k, v in extra_per_channel.items():
        if v.shape != (n_peaks,):
            raise ValueError(
                f"extra_per_channel[{k!r}] must have shape ({n_peaks},); got {v.shape}"
            )

    # Per-page metadata to embed in the TIFF description tag.
    prevalence = ds.extra.get("consensus_prevalence")
    if prevalence is None or len(prevalence) != n_peaks:
        prevalence = (np.asarray(pm.matrix[:]) > 0).sum(axis=0) / n_pixels

    op_history_hash = ds.hash()
    mz_axis = np.asarray(pm.mz_axis[:])
    matrix = np.asarray(pm.matrix[:]).astype(np.float32, copy=False)

    tiff_path = Path(path)
    csv_path = tiff_path.with_name(tiff_path.stem + "_channels.csv")

    # Stream pages; tifffile.TiffWriter handles BigTIFF auto-promotion.
    with tifffile.TiffWriter(str(tiff_path), bigtiff=True) as tw:
        for c in range(n_peaks):
            page = project_to_grid(matrix[:, c], ds.coords, ds.grid_shape).astype(
                np.float32, copy=False
            )
            description = json.dumps(
                {
                    "channel": int(c),
                    "mz": float(mz_axis[c]),
                    "prevalence": float(prevalence[c]),
                    "op_history_hash": op_history_hash,
                    **{k: _scalar(v[c]) for k, v in extra_per_channel.items()},
                },
                separators=(",", ":"),
            )
            tw.write(
                page,
                photometric="minisblack",
                description=description,
                compression=compression,
                metadata=None,  # we put structured info in `description` for portability
            )

    if write_csv:
        # The CSV is the human-readable channel manifest. We keep the op_history_hash
        # in the TIFF page description (machine-readable provenance) but leave it out
        # of the CSV — it's a long opaque string that adds no useful information for
        # the analyst reading the file in Excel or pandas.
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            cols = ["page", "mz", "prevalence"] + list(extra_per_channel.keys())
            writer.writerow(cols)
            for c in range(n_peaks):
                row = [
                    c,
                    float(mz_axis[c]),
                    float(prevalence[c]),
                ] + [_scalar(extra_per_channel[k][c]) for k in extra_per_channel]
                writer.writerow(row)

    return TiffWriteResult(
        tiff_path=tiff_path,
        csv_path=csv_path,
        n_pages=n_peaks,
        image_shape=(height, width),
    )


def read_hyperspectral_tiff_metadata(path: Path | str) -> list[dict[str, Any]]:
    """Read the per-page JSON description tags written by `write_hyperspectral_tiff`.

    Returns a list of dicts, one per page. The pixel arrays themselves are NOT
    returned — use `tifffile.imread(path)` for that.
    """
    p = Path(path)
    out: list[dict[str, Any]] = []
    with tifffile.TiffFile(str(p)) as tf:
        for page in tf.pages:
            desc = page.tags.get("ImageDescription")
            text = desc.value if desc is not None else ""
            try:
                out.append(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                out.append({"raw_description": text})
    return out


def _scalar(v: Any) -> Any:
    """Coerce numpy scalars to Python scalars so json/csv can serialize them."""
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except Exception:  # noqa: BLE001
            pass
    return v
