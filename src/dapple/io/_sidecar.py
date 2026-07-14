"""Sidecar metadata file loading.

Two formats are supported next to a dataset:

1. JSON (``.metadata.json`` or ``.experiment.json``) — a flat object whose keys match
   ``ExperimentParams`` field names. Unknown keys are ignored. Numeric fields are
   coerced; ``sample_type`` may be an empty string for "(unset)".
2. XML (``.spec.xml``) — the dapple reproducibility manifest. The
   ``<experimentParams>`` block is read; the pipeline (if present) is currently used
   only to confirm the namespace.

Either form overrides auto-detected values without erasing fields it omits; the
returned ``MetadataSource`` map tags overridden fields with ``"sidecar"`` or
``"spec_xml"`` so the wizard can show the user where each value came from.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from dapple.data.metadata import ExperimentParams
from dapple.data.metadata_source import MetadataSource


def load_json_sidecar(
    path: Path,
    metadata: ExperimentParams,
    source: MetadataSource,
) -> tuple[ExperimentParams, MetadataSource]:
    """Overlay a flat JSON object onto ExperimentParams.

    Unknown keys are ignored with a debug log; numeric fields are coerced; the
    ``sample_type`` field accepts empty string as "unset".
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read metadata JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level JSON must be an object, got {type(raw).__name__}")
    return _apply_overrides(raw, metadata, source, "sidecar", context=str(path))


def load_spec_xml_sidecar(
    path: Path,
    metadata: ExperimentParams,
    source: MetadataSource,
) -> tuple[ExperimentParams, MetadataSource]:
    """Read the ExperimentParams from an accompanying ``*.spec.xml``.

    Only the ``<experimentParams>`` block is consumed; the pipeline portion is left to
    the user's next interactive step (the wizard's WorkflowPage), since they may want
    different processing parameters even if the experiment description matches.
    """
    from dapple.io.spec_xml import read_spec_xml

    _pipeline, ep, _prov = read_spec_xml(path)
    overrides: dict[str, Any] = {}
    for f in fields(ExperimentParams):
        val = getattr(ep, f.name, None)
        if val in (None, "unknown", ""):
            continue
        overrides[f.name] = val
    return _apply_overrides(overrides, metadata, source, "spec_xml", context=str(path))


def _apply_overrides(
    overrides: dict[str, Any],
    metadata: ExperimentParams,
    source: MetadataSource,
    tag: str,
    *,
    context: str,
) -> tuple[ExperimentParams, MetadataSource]:
    field_map = {f.name: f for f in fields(ExperimentParams)}
    kwargs: dict[str, Any] = {}
    new_source: MetadataSource = dict(source)
    for k, v in overrides.items():
        if k not in field_map:
            continue
        try:
            coerced = _coerce(v, field_map[k].type, field_name=k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid field {k!r}: {exc}") from exc
        kwargs[k] = coerced
        new_source[k] = tag
    if not kwargs:
        return metadata, new_source
    try:
        new_metadata = replace(metadata, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid experiment metadata: {exc}") from exc
    return new_metadata, new_source


_LITERAL_VALUES: dict[str, frozenset[str]] = {
    "instrument_family": frozenset(
        {"tof_axial", "tof_reflectron", "qtof", "orbitrap", "fticr", "unknown"}
    ),
    "ionization": frozenset({"maldi", "desi", "sims", "esi", "unknown"}),
    "profile_or_centroided": frozenset({"profile", "centroided", "unknown"}),
    "polarity": frozenset({"positive", "negative"}),
    "sample_type": frozenset({"tissue", "cell_culture", "whole_organism", "other"}),
}


def _coerce(value: Any, target_type: Any, *, field_name: str) -> Any:
    """Best-effort coercion to a Python value matching the dataclass field's type.

    ``target_type`` may be a type or a string (PEP 563 deferred annotations); we don't
    parse complex generics — just enough to pick float/int/None/str.
    """
    if value is None:
        if field_name in {"pixel_size_um", "sample_type"}:
            return None
        raise TypeError("null is not allowed")
    type_str = str(target_type).lower()
    if "float" in type_str or "int" in type_str:
        if isinstance(value, str) and value.strip() == "":
            if field_name == "pixel_size_um":
                return None
            raise TypeError("an empty string is not a number")
        if isinstance(value, bool):
            raise TypeError("boolean is not a number")
        try:
            f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"expected a number, got {type(value).__name__}") from exc
        if not math.isfinite(f):
            raise ValueError("number must be finite")
        return int(f) if "int" in type_str and "float" not in type_str else f
    if "bool" in type_str:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {type(value).__name__}")
    value = value.strip() if field_name != "notes" else value
    if field_name == "sample_type" and not value:
        return None
    allowed = _LITERAL_VALUES.get(field_name)
    if allowed is not None and value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"expected one of {choices}; got {value!r}")
    if not value and field_name != "notes":
        raise ValueError("value must not be empty")
    return value
