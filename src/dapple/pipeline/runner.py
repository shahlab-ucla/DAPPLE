"""Pipeline runner: walk a topo-sorted Pipeline, execute each Node, gather diagnostics.

Cache strategy: each Node's output dataset is keyed by its `(input_hash, params_hash,
op_name, library_versions_hash)`. The runner exposes a simple in-memory cache;
on-disk Zarr caching for cross-session reuse is planned.

Execution model: a synchronous, single-process walk. The runner emits progress
callbacks so the napari `thread_worker` can keep the UI responsive without the
runner itself depending on Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dapple.data.dataset import MSIDataset
from dapple.data.hashing import combine_hashes, hash_obj
from dapple.ops.base import REGISTRY, Diagnostic, OpResult
from dapple.pipeline.pipeline import Node, Pipeline


ProgressCB = Callable[[int, int, str], None]


@dataclass
class RunResult:
    """The final output dataset plus all per-node diagnostics in execution order."""

    output: MSIDataset
    per_node_outputs: dict[str, MSIDataset] = field(default_factory=dict)
    diagnostics: dict[str, list[Diagnostic]] = field(default_factory=dict)


class PipelineRunner:
    """Executes a Pipeline against an input dataset.

    Holds an in-memory result cache keyed by node cache hash so reruns of the same
    `(input, pipeline)` combination skip work. Cache is per-runner instance — start
    a fresh one when you want a clean run.
    """

    def __init__(self) -> None:
        self._cache: dict[str, MSIDataset] = {}
        self._diag_cache: dict[str, list[Diagnostic]] = {}

    def run(
        self,
        pipeline: Pipeline,
        input_ds: MSIDataset,
        *,
        progress: ProgressCB | None = None,
        on_node_done: Callable[[str, OpResult], None] | None = None,
    ) -> RunResult:
        n = len(pipeline.nodes)
        per_node: dict[str, MSIDataset] = {}
        diags: dict[str, list[Diagnostic]] = {}

        for i, node in enumerate(pipeline.nodes):
            input_for_node = self._resolve_input(node, pipeline, per_node, input_ds)
            if progress is not None:
                progress(i, n, f"running {node.op_name}({node.id})")
            cache_key = _cache_key(node, pipeline, input_for_node)
            if cache_key in self._cache:
                out_ds = self._cache[cache_key]
                node_diags = self._diag_cache.get(cache_key, [])
            else:
                op_cls = REGISTRY.get(node.op_name)
                op = op_cls()
                # Per-node RNG: deterministic combination of master seed and node id.
                rng = np.random.default_rng(_derive_seed(pipeline.rng_seed, node.id))
                result = op.apply(input_for_node, node.params, rng=rng)
                out_ds = result.dataset
                node_diags = list(result.diagnostics)
                self._cache[cache_key] = out_ds
                self._diag_cache[cache_key] = node_diags
                if on_node_done is not None:
                    on_node_done(node.id, result)
            per_node[node.id] = out_ds
            diags[node.id] = node_diags

        if progress is not None:
            progress(n, n, "done")

        # Final output = output of the last node in topo order. (DAG with multiple
        # sinks is uncommon in our recommend_pipeline; if it ever happens, callers
        # use per_node[node.id] directly.)
        last = pipeline.nodes[-1].id
        return RunResult(output=per_node[last], per_node_outputs=per_node, diagnostics=diags)

    def _resolve_input(
        self,
        node: Node,
        pipeline: Pipeline,
        per_node: dict[str, MSIDataset],
        input_ds: MSIDataset,
    ) -> MSIDataset:
        """Pick the dataset to feed into a node.

        Source nodes (no upstream) get the original input. Other nodes consume the
        last-listed upstream's output by convention. Multi-input operators (planned)
        will need to negotiate which upstream to take; for now everything is a chain.
        """
        if not node.upstream:
            return input_ds
        return per_node[node.upstream[-1]]


def _derive_seed(master_seed: int, node_id: str) -> int:
    """Mix master seed and node id deterministically so each node has its own RNG."""
    h = hash_obj({"seed": int(master_seed), "node": node_id})
    return int(h[:16], 16) & 0xFFFFFFFF


def _cache_key(node: Node, pipeline: Pipeline, input_ds: MSIDataset) -> str:
    return combine_hashes(
        node.op_name,
        node.params_hash(),
        input_ds.hash(),
        hash_obj(pipeline.library_versions),
    )
