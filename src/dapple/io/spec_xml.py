"""Read and write `.spec.xml` reproducibility manifests.

A `.spec.xml` captures everything needed to reapply a pipeline to a new dataset:
- provenance (timestamp, plugin version, library versions, input dataset SHA-256)
- the source ExperimentParams
- the full pipeline DAG with operator names and parameters
- a flat scalar-only diagnostics summary (heavy diagnostic payloads stay out — they
  belong in a sibling Zarr if persisted)

The XML is human-readable and validates against `spec_xml.xsd` (currently a permissive
placeholder; a strict per-element schema is planned). The format is designed so
that `read_spec_xml(write_spec_xml(p)) == p` round-trips semantically, and `apply_spec`
can take the XML plus an input dataset and produce identical output.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree

from dapple.data.metadata import ExperimentParams
from dapple.ops.base import REGISTRY, Diagnostic, OpParams
from dapple.pipeline.pipeline import Node, Pipeline

NS = "urn:dapple:spec:v1"
NSMAP = {None: NS}
SPEC_VERSION = "1.0"


@dataclass(frozen=True)
class ProvenanceInfo:
    created_at: str
    plugin_version: str
    input_dataset_hash: str | None = None
    declared_md5: str | None = None
    notes: str = ""


def write_spec_xml(
    path: Path | str,
    *,
    pipeline: Pipeline,
    experiment_params: ExperimentParams,
    provenance: ProvenanceInfo,
    diagnostics: dict[str, list[Diagnostic]] | None = None,
) -> Path:
    """Serialize a Pipeline + ExperimentParams + provenance + diagnostics to XML.

    Returns the path written. Diagnostic payloads (numpy arrays) are NOT serialized;
    only scalar summaries are. Persist payloads to a sibling Zarr if you need them.
    """
    root = etree.Element("spec", nsmap=NSMAP, attrib={"version": SPEC_VERSION})
    _append_provenance(root, provenance, pipeline.library_versions)
    _append_experiment_params(root, experiment_params)
    _append_pipeline(root, pipeline)
    if diagnostics:
        _append_diagnostics_summary(root, diagnostics)

    out_path = Path(path)
    tree = etree.ElementTree(root)
    tree.write(str(out_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return out_path


def read_spec_xml(path: Path | str) -> tuple[Pipeline, ExperimentParams, ProvenanceInfo]:
    """Parse a .spec.xml back into a Pipeline + ExperimentParams + provenance.

    Operator parameters are reconstructed by importing each operator's `params_cls`
    from the registry and instantiating it from the serialized field values. Library
    versions in the document populate the resulting Pipeline's `library_versions` so
    the cache key is preserved.
    """
    p = Path(path)
    tree = etree.parse(str(p))
    root = tree.getroot()
    if etree.QName(root.tag).localname != "spec":
        raise ValueError(f"{p} is not a dapple spec (root element is {root.tag})")
    if etree.QName(root.tag).namespace != NS:
        raise ValueError(f"{p} uses unexpected namespace {etree.QName(root.tag).namespace!r}")

    provenance = _parse_provenance(root)
    library_versions = _parse_library_versions(root)
    experiment_params = _parse_experiment_params(root)
    nodes, rng_seed = _parse_pipeline(root)
    pipeline = Pipeline(
        nodes=tuple(nodes), rng_seed=rng_seed, library_versions=library_versions
    )
    return pipeline, experiment_params, provenance


def make_provenance(
    *,
    plugin_version: str,
    input_dataset_hash: str | None,
    declared_md5: str | None = None,
    notes: str = "",
    timestamp: _dt.datetime | None = None,
) -> ProvenanceInfo:
    ts = timestamp or _dt.datetime.now(tz=_dt.UTC)
    return ProvenanceInfo(
        created_at=ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        plugin_version=plugin_version,
        input_dataset_hash=input_dataset_hash,
        declared_md5=declared_md5,
        notes=notes,
    )


# ---- writers --------------------------------------------------------------------


def _append_provenance(
    root: etree._Element, prov: ProvenanceInfo, library_versions: dict[str, str]
) -> None:
    el = etree.SubElement(root, "provenance")
    etree.SubElement(el, "createdAt").text = prov.created_at
    etree.SubElement(el, "pluginVersion").text = prov.plugin_version
    libs = etree.SubElement(el, "libraryVersions")
    for name in sorted(library_versions):
        etree.SubElement(libs, "lib", attrib={"name": name, "version": library_versions[name]})
    if prov.input_dataset_hash is not None:
        etree.SubElement(el, "inputDatasetHash", attrib={"algo": "sha256"}).text = (
            prov.input_dataset_hash
        )
    if prov.declared_md5:
        etree.SubElement(el, "ibdHash", attrib={"algo": "md5"}).text = prov.declared_md5
    if prov.notes:
        etree.SubElement(el, "notes").text = prov.notes


def _append_experiment_params(root: etree._Element, ep: ExperimentParams) -> None:
    el = etree.SubElement(root, "experimentParams")
    for f in fields(ep):
        val = getattr(ep, f.name)
        if val is None:
            continue
        sub = etree.SubElement(el, f.name)
        sub.text = str(val)


def _append_pipeline(root: etree._Element, pipeline: Pipeline) -> None:
    el = etree.SubElement(root, "pipeline", attrib={"rngSeed": str(pipeline.rng_seed)})
    for node in pipeline.nodes:
        node_el = etree.SubElement(
            el, "node", attrib={"id": node.id, "op": node.op_name}
        )
        if node.upstream:
            ups = etree.SubElement(node_el, "upstream")
            for u in node.upstream:
                etree.SubElement(ups, "ref", attrib={"id": u})
        params_el = etree.SubElement(node_el, "params")
        _params_to_xml(params_el, node.params)


def _params_to_xml(parent: etree._Element, params: OpParams) -> None:
    for f in fields(params):
        val = getattr(params, f.name)
        if val is None:
            continue
        kind = _python_type_label(val)
        param_el = etree.SubElement(
            parent, "param", attrib={"name": f.name, "type": kind}
        )
        param_el.text = _value_to_text(val)


def _python_type_label(val: Any) -> str:
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "str"
    return "str"


def _value_to_text(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _append_diagnostics_summary(
    root: etree._Element, diagnostics: dict[str, list[Diagnostic]]
) -> None:
    el = etree.SubElement(root, "diagnosticsSummary")
    for node_id, diags in diagnostics.items():
        node_el = etree.SubElement(el, "node", attrib={"id": node_id})
        for d in diags:
            for k, v in d.summary.items():
                etree.SubElement(
                    node_el, "summary", attrib={"key": f"{d.name}.{k}"}
                ).text = _format_scalar(v)


def _format_scalar(v: Any) -> str:
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        v = int(v)
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "+Inf" if v > 0 else "-Inf"
        return repr(v)
    return str(v)


# ---- readers --------------------------------------------------------------------


def _ns(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _parse_provenance(root: etree._Element) -> ProvenanceInfo:
    el = root.find(_ns("provenance"))
    if el is None:
        raise ValueError("missing <provenance> element")
    created_at = (el.findtext(_ns("createdAt")) or "").strip()
    plugin_version = (el.findtext(_ns("pluginVersion")) or "").strip()
    input_hash = (el.findtext(_ns("inputDatasetHash")) or "").strip() or None
    declared_md5 = (el.findtext(_ns("ibdHash")) or "").strip() or None
    notes = (el.findtext(_ns("notes")) or "").strip()
    return ProvenanceInfo(
        created_at=created_at,
        plugin_version=plugin_version,
        input_dataset_hash=input_hash,
        declared_md5=declared_md5,
        notes=notes,
    )


def _parse_library_versions(root: etree._Element) -> dict[str, str]:
    el = root.find(f"{_ns('provenance')}/{_ns('libraryVersions')}")
    if el is None:
        return {}
    out: dict[str, str] = {}
    for lib in el.findall(_ns("lib")):
        name = lib.get("name")
        ver = lib.get("version")
        if name and ver:
            out[name] = ver
    return out


def _parse_experiment_params(root: etree._Element) -> ExperimentParams:
    el = root.find(_ns("experimentParams"))
    if el is None:
        raise ValueError("missing <experimentParams> element")
    raw: dict[str, Any] = {}
    for child in el:
        local = etree.QName(child.tag).localname
        raw[local] = (child.text or "").strip()
    # Coerce numeric fields.
    for key in ("mz_min", "mz_max", "pixel_size_um"):
        if key in raw and raw[key]:
            try:
                raw[key] = float(raw[key])
            except ValueError:
                pass
    if raw.get("pixel_size_um") in {"", None}:
        raw["pixel_size_um"] = None
    if raw.get("sample_type") in {"", None}:
        raw["sample_type"] = None
    kwargs: dict[str, Any] = {}
    for f in fields(ExperimentParams):
        if f.name in raw:
            kwargs[f.name] = raw[f.name]
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(
                f"<experimentParams> is missing required field {f.name!r}"
            )
    return ExperimentParams(**kwargs)


def _parse_pipeline(root: etree._Element) -> tuple[list[Node], int]:
    el = root.find(_ns("pipeline"))
    if el is None:
        raise ValueError("missing <pipeline> element")
    rng_seed = int(el.get("rngSeed") or 0)
    nodes: list[Node] = []
    for node_el in el.findall(_ns("node")):
        node_id = node_el.get("id")
        op_name = node_el.get("op")
        if not node_id or not op_name:
            raise ValueError("<node> missing 'id' or 'op'")
        upstream: list[str] = []
        ups_el = node_el.find(_ns("upstream"))
        if ups_el is not None:
            for ref in ups_el.findall(_ns("ref")):
                rid = ref.get("id")
                if rid:
                    upstream.append(rid)
        params = _parse_params(op_name, node_el.find(_ns("params")))
        nodes.append(Node(id=node_id, op_name=op_name, params=params, upstream=tuple(upstream)))
    return nodes, rng_seed


def _parse_params(op_name: str, params_el: etree._Element | None) -> OpParams:
    op_cls = REGISTRY.get(op_name)
    params_cls = op_cls.params_cls
    if params_el is None:
        return params_cls()  # type: ignore[call-arg]
    raw: dict[str, Any] = {}
    for p in params_el.findall(_ns("param")):
        name = p.get("name")
        kind = (p.get("type") or "str").lower()
        text = (p.text or "").strip()
        if name is None:
            continue
        raw[name] = _coerce_param(text, kind)
    # Filter to the fields the dataclass actually has.
    known = {f.name for f in fields(params_cls)}
    return params_cls(**{k: v for k, v in raw.items() if k in known})  # type: ignore[call-arg]


def _coerce_param(text: str, kind: str) -> Any:
    if kind == "bool":
        return text.lower() in {"true", "1", "yes"}
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    return text


# ---- napari plugin entry-point hook ---------------------------------------------


def napari_get_reader(path: str | list[str]) -> Any:
    """npe2 reader entry point. .spec.xml on its own can't materialize layers, so we
    return a callable that imports a no-op layer and prints the recovered pipeline,
    matching napari's expectation that a reader returns at least one layer."""
    if isinstance(path, list):
        path = path[0]
    p = Path(path)
    if p.suffix.lower() != ".xml" or not p.name.endswith(".spec.xml"):
        return None
    return _napari_spec_reader


def _napari_spec_reader(path: str) -> list[tuple[Any, dict[str, Any], str]]:
    pipeline, ep, prov = read_spec_xml(Path(path))
    summary = (
        f"dapple spec '{Path(path).name}'\n"
        f"  created_at:   {prov.created_at}\n"
        f"  plugin v:     {prov.plugin_version}\n"
        f"  pipeline:     {len(pipeline.nodes)} nodes — {' → '.join(n.op_name for n in pipeline.nodes)}\n"
        f"  experiment:   {ep.instrument_family} / {ep.ionization} / {ep.profile_or_centroided} / {ep.polarity}"
    )
    # Pure-spec reads don't add a layer; return a 1×1 image with metadata so napari
    # is happy and the wizard can pick the spec up via its sidecar metadata.
    img = np.zeros((1, 1), dtype=np.float32)
    metadata = {"msi_spec": {"pipeline": pipeline, "experiment_params": ep, "provenance": prov}}
    return [(img, {"name": Path(path).stem, "metadata": metadata, "visible": False}, "image")]


# Defensive: ensure ops are imported so REGISTRY contains every operator before
# anyone tries to read a .spec.xml that mentions them.
def _eager_import_ops() -> None:
    import_module("dapple.ops")


_eager_import_ops()
