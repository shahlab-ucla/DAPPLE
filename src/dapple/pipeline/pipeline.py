"""Pipeline DAG and Node dataclasses.

A `Pipeline` is a directed-acyclic graph of operator invocations. Each `Node` names
an operator (registry key), an `OpParams` instance, and an `upstream` list of node
ids that must execute first. The whole pipeline is content-addressed: every node has
a deterministic cache hash derived from its inputs, parameters, op name, and the
library versions it ran under, so reruns can short-circuit.

Pipelines round-trip through `.spec.xml` (see `io/spec_xml.py`) for reproducibility.
The runner that actually executes a pipeline lives in `pipeline/runner.py`.
"""

from __future__ import annotations

import importlib.metadata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from dapple.data.hashing import combine_hashes, hash_obj
from dapple.ops.base import OpParams


@dataclass(frozen=True)
class Node:
    """One invocation of an operator inside a pipeline.

    `id` — short name unique within the pipeline; used for upstream references and
        as the name a runner attaches to its output.
    `op_name` — registry key (matches `Operator.name`).
    `params` — frozen dataclass instance the operator expects.
    `upstream` — node ids whose outputs feed this one. Empty for source nodes that
        run on the original input.
    """

    id: str
    op_name: str
    params: OpParams
    upstream: tuple[str, ...] = ()

    def params_hash(self) -> str:
        return hash_obj(self.params)


@dataclass(frozen=True)
class Pipeline:
    """A topologically ordered list of nodes plus run-wide metadata.

    `nodes` is in topological order; the runner walks it left-to-right.
    `rng_seed` is the master seed; per-node RNGs derive from `(rng_seed, node.id)`.
    `library_versions` snapshots what was installed when the pipeline was built; it
    feeds the cache key, so swapping `numpy` major versions invalidates cached results.
    """

    nodes: tuple[Node, ...]
    rng_seed: int = 0
    library_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate node id(s): {[i for i in ids if ids.count(i) > 1]}")
        for node in self.nodes:
            for up in node.upstream:
                if up not in ids:
                    raise ValueError(
                        f"node {node.id!r} references unknown upstream {up!r}"
                    )
        # Verify topological order.
        seen: set[str] = set()
        for node in self.nodes:
            for up in node.upstream:
                if up not in seen:
                    raise ValueError(
                        f"node {node.id!r} depends on {up!r} which appears later — "
                        "nodes must be topologically sorted."
                    )
            seen.add(node.id)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def successors(self) -> dict[str, tuple[str, ...]]:
        """For each node, the ids of its direct downstream successors."""
        out: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes:
            for u in n.upstream:
                out[u].append(n.id)
        return {k: tuple(v) for k, v in out.items()}

    def with_rng_seed(self, seed: int) -> "Pipeline":
        return Pipeline(nodes=self.nodes, rng_seed=seed, library_versions=self.library_versions)

    def hash(self, *, input_hash: str) -> str:
        """A pipeline-wide hash, including the (caller-supplied) input dataset hash."""
        parts: list[str] = [
            input_hash,
            str(self.rng_seed),
            hash_obj(self.library_versions),
        ]
        for n in self.nodes:
            parts.extend([n.id, n.op_name, n.params_hash(), "|".join(n.upstream)])
        return combine_hashes(*parts)


def detect_library_versions(extra: Iterable[str] = ()) -> dict[str, str]:
    """Snapshot installed versions of dependencies that affect numerical behavior.

    Only includes packages that are actually installed; missing packages are omitted
    rather than reported as "unknown" so the resulting dict is order-independent and
    cleanly hashes.
    """
    pkgs = [
        "numpy",
        "scipy",
        "pyimzml",
        "lxml",
        "netCDF4",
        "scikit-image",
        "tifffile",
        "zarr",
        "dask",
        "dapple",
        *extra,
    ]
    out: dict[str, str] = {}
    for name in pkgs:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out
