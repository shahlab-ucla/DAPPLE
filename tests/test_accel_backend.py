"""Tests for the acceleration backend detection module.

These tests should pass on every supported platform regardless of which (if any) GPU
backends are installed. We just verify that the probes don't error, and that the
return value is one of the expected backend names.
"""

from __future__ import annotations

from dapple.accel import Backend, get_backend
from dapple.accel.backend import get_onnx_providers


def test_get_backend_returns_known_name():
    b = get_backend()
    assert isinstance(b, Backend)
    assert b.name in {"cuda", "directml", "mps", "cpu"}
    assert isinstance(b.detail, str) and b.detail


def test_get_onnx_providers_returns_list():
    providers = get_onnx_providers()
    assert isinstance(providers, list)
    # Always either empty (no onnxruntime) or a subset of the known providers.
    known = {
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    }
    assert all(p in known for p in providers)
