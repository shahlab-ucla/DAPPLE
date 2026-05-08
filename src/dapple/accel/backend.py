"""Acceleration backend detection — scaffolding for future GPU/DL operators.

`get_backend()` returns the most preferred available backend for ML/DL operators
(none of which exist yet; any DL operators added later must fit on the loaded
dataset rather than relying on bundled pre-trained weights). The detection itself
is exercised by `tests/test_accel_backend.py` to ensure the import paths don't break
on machines without GPU libraries.

Order of preference: CUDA → DirectML (Windows) → MPS (macOS) → CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendName = Literal["cuda", "directml", "mps", "cpu"]


@dataclass(frozen=True)
class Backend:
    name: BackendName
    detail: str  # human-readable: device name, version, etc.


def get_backend() -> Backend:
    """Probe in order of preference; return the first available."""
    cuda = _probe_cuda()
    if cuda is not None:
        return cuda
    dml = _probe_directml()
    if dml is not None:
        return dml
    mps = _probe_mps()
    if mps is not None:
        return mps
    return Backend(name="cpu", detail="no accelerator detected")


def get_onnx_providers() -> list[str]:
    """ONNX Runtime execution-provider list, filtered by what's actually installed."""
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return []
    available = set(ort.get_available_providers())
    preferred = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available]


def _probe_cuda() -> Backend | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if n > 0 else "cuda"
            return Backend(name="cuda", detail=f"{name} (devices={n})")
    except Exception:  # noqa: BLE001
        return None
    return None


def _probe_directml() -> Backend | None:
    try:
        import torch_directml  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        if torch_directml.is_available():
            n = torch_directml.device_count()
            name = torch_directml.device_name(0) if n > 0 else "directml"
            return Backend(name="directml", detail=f"{name} (devices={n})")
    except Exception:  # noqa: BLE001
        return None
    return None


def _probe_mps() -> Backend | None:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        backends = getattr(torch, "backends", None)
        if backends is not None and getattr(backends, "mps", None) is not None:
            if torch.backends.mps.is_available():
                return Backend(name="mps", detail="Apple Metal Performance Shaders")
    except Exception:  # noqa: BLE001
        return None
    return None
