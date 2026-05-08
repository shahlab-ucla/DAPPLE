"""Acceleration backends (CUDA, DirectML, MPS) — scaffolding for future GPU operators."""

from dapple.accel.backend import Backend, get_backend

__all__ = ["Backend", "get_backend"]
