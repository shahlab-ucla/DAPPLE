"""Content-addressed hashing for datasets, parameters, and pipeline nodes.

We use SHA-256 throughout. Hashes are stable across runs and serve as the cache key
for pipeline node outputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CHUNK = 1 << 20  # 1 MB


def sha256_file(path: Path | str) -> str:
    """SHA-256 hex digest of a file, streamed in 1 MB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path | str) -> str:
    """MD5 hex digest of a file (for imzML .ibd verification only — not a security claim)."""
    h = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def hash_array_content(array: Any) -> str:
    """Return a stable SHA-256 fingerprint for an array-like value.

    Shape and dtype are included so byte-identical buffers with different
    interpretations cannot collide.  Numeric arrays are normalized to
    little-endian contiguous storage before hashing, which makes the result
    independent of host byte order and memory layout.  Object arrays use the
    same canonical JSON representation as :func:`hash_obj` rather than hashing
    process-specific object pointers.

    Array-like backends (for example a future zarr-backed dataset) are
    materialized once by this function.  The data-model classes cache the
    resulting digest, so ordinary dataset/cache fingerprinting remains O(1).
    """
    if isinstance(array, np.ndarray):
        arr = array
    else:
        try:
            arr = np.asarray(array[:])
        except (IndexError, TypeError):
            arr = np.asarray(array)

    canonical_dtype = arr.dtype if arr.dtype.hasobject else arr.dtype.newbyteorder("<")
    header = json.dumps(
        {"shape": list(arr.shape), "dtype": canonical_dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    h = hashlib.sha256()
    h.update(header)
    h.update(b"|")

    if arr.dtype.hasobject:
        payload = json.dumps(
            _canonicalize(arr.tolist()), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        h.update(payload)
        return h.hexdigest()

    # Canonicalize byte order without changing the logical dtype recorded in
    # the header.  ``newbyteorder`` is a no-op for one-byte and byte-order-free
    # dtypes.
    canonical = np.ascontiguousarray(arr, dtype=canonical_dtype)
    if canonical.size:
        h.update(memoryview(canonical).cast("B"))
    return h.hexdigest()


def _canonicalize(obj: Any) -> Any:
    """Convert dataclasses + tuples into a JSON-serializable structure with sorted keys.

    Floats/ints stay as numbers; numpy scalars are coerced to Python types.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _canonicalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray) or (
        hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "__getitem__")
    ):
        arr = obj if isinstance(obj, np.ndarray) else np.asarray(obj[:])
        if arr.ndim == 0:
            return _canonicalize(arr.item())
        return {
            "__array_sha256__": hash_array_content(arr),
            "dtype": (
                arr.dtype if arr.dtype.hasobject else arr.dtype.newbyteorder("<")
            ).str,
            "shape": list(arr.shape),
        }
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


def hash_obj(obj: Any) -> str:
    """SHA-256 of a deterministic JSON serialization of a (possibly nested) dataclass/dict."""
    payload = json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def combine_hashes(*hashes: str) -> str:
    """Combine multiple hex digests into a single SHA-256 (order-sensitive)."""
    h = hashlib.sha256()
    for x in hashes:
        h.update(x.encode("ascii"))
        h.update(b"|")
    return h.hexdigest()
