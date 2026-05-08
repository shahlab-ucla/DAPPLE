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
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level JSON must be an object, got {type(raw).__name__}")
    return _apply_overrides(raw, metadata, source, "sidecar")


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
    return _apply_overrides(overrides, metadata, source, "spec_xml")


def _apply_overrides(
    overrides: dict[str, Any],
    metadata: ExperimentParams,
    source: MetadataSource,
    tag: str,
) -> tuple[ExperimentParams, MetadataSource]:
    field_map = {f.name: f for f in fields(ExperimentParams)}
    kwargs: dict[str, Any] = {}
    new_source: MetadataSource = dict(source)
    for k, v in overrides.items():
        if k not in field_map:
            continue
        coerced = _coerce(v, field_map[k].type)
        if coerced is _UNCOERCIBLE:
            continue
        kwargs[k] = coerced
        new_source[k] = tag
    if not kwargs:
        return metadata, new_source
    new_metadata = replace(metadata, **kwargs)
    return new_metadata, new_source


_UNCOERCIBLE = object()


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion to a Python value matching the dataclass field's type.

    ``target_type`` may be a type or a string (PEP 563 deferred annotations); we don't
    parse complex generics — just enough to pick float/int/None/str.
    """
    if value is None:
        return None
    type_str = str(target_type).lower()
    if "float" in type_str or "int" in type_str:
        if isinstance(value, str) and value.strip() == "":
            return None
        try:
            f = float(value)
            return int(f) if "int" in type_str and "float" not in type_str else f
        except (TypeError, ValueError):
            return _UNCOERCIBLE
    if "bool" in type_str:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)
    # Strings + Literal aliases just round-trip.
    return None if (isinstance(value, str) and value.strip() == "") else value
