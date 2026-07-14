"""Per-field provenance for ExperimentParams.

A `MetadataSource` dict maps each `ExperimentParams` field name to a string saying
where the value came from:

    "imzml"      — read from imzML CV terms
    "andims"     — read from ANDI-MS NetCDF attributes
    "sidecar"    — loaded from a `*.metadata.json` or `*.experiment.json` next to the data
    "spec_xml"   — loaded from an accompanying `*.spec.xml`
    "default"    — fell through to a sensible default; user should verify
    "user"       — user-edited in the wizard

The wizard's ParamsPage renders a small badge per field based on this so the user can
tell which entries the plugin auto-detected vs. which were guesses.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Mapping, MutableMapping

from dapple.data.metadata import ExperimentParams

MetadataSource = MutableMapping[str, str]
"""dict-of-str — field name -> source label."""

KNOWN_SOURCES = ("imzml", "andims", "sidecar", "spec_xml", "default", "user")


def empty_source() -> MetadataSource:
    """All fields default — useful when constructing ExperimentParams from scratch."""
    return {f.name: "default" for f in fields(ExperimentParams)}


def merge_sources(*srcs: Mapping[str, str]) -> MetadataSource:
    """Right-most non-default wins; ties go to the rightmost mapping."""
    out: MetadataSource = empty_source()
    for src in srcs:
        for k, v in src.items():
            if v == "default":
                continue
            out[k] = v
    return out


def labels_for_source(source: str) -> tuple[str, str]:
    """Return (display_label, color_hint) for a UI badge."""
    return {
        "imzml": ("auto-detected (imzML)", "#2e7d32"),
        "andims": ("auto-detected (CDF)", "#2e7d32"),
        "sidecar": ("from sidecar metadata", "#1565c0"),
        "spec_xml": ("from .spec.xml", "#1565c0"),
        "user": ("user override", "#6a1b9a"),
        "default": ("default — please verify", "#b58a00"),
    }.get(source, (source, "#666"))
