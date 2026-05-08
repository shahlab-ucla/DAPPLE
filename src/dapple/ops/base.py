"""Operator ABC and shared dataclasses.

An `Operator` takes an `MSIDataset` and an `OpParams` (a frozen dataclass), and returns
an `OpResult` consisting of a (possibly new) `MSIDataset` and a list of `Diagnostic`s.
Operators must be deterministic given a seeded `numpy.random.Generator`. Their
parameter dataclass and name are hashable into the pipeline DAG.

Every operator emits at least one diagnostic. Diagnostics carry a scalar summary dict
(persisted in `.spec.xml` and rendered in the wizard's RunPage) and an optional payload
dict of arrays (persisted to a sibling Zarr).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from dapple.data.dataset import MSIDataset
from dapple.data.metadata import ExperimentParams


@dataclass(frozen=True)
class OpParams:
    """Marker base class. Subclass with `@dataclass(frozen=True)` and ordinary fields."""


def field_label(field: Any) -> str:
    """Return the user-facing label for a dataclass field (falls back to its name)."""
    return field.metadata.get("label", field.name.replace("_", " ").capitalize())


def field_help(field: Any) -> str:
    """Return the tooltip text for a dataclass field, or an empty string if absent."""
    return field.metadata.get("help", "")


@dataclass
class Diagnostic:
    name: str
    summary: dict[str, float] = field(default_factory=dict)
    payload: dict[str, np.ndarray] | None = None
    figure_hint: str | None = None


@dataclass
class OpResult:
    dataset: MSIDataset
    diagnostics: list[Diagnostic] = field(default_factory=list)


class Operator(ABC):
    """Abstract base class for every processing operator.

    Subclasses set `name` (used as the registry key and serialized in `.spec.xml`) and
    implement `apply`. They may override `default_params` and `validate`.
    """

    name: ClassVar[str]
    params_cls: ClassVar[type[OpParams]]

    @abstractmethod
    def apply(
        self,
        ds: MSIDataset,
        params: OpParams,
        *,
        rng: np.random.Generator,
    ) -> OpResult:
        """Run the operator. Must not mutate `ds`. Must be deterministic given `rng`."""

    def default_params(self, ep: ExperimentParams) -> OpParams:
        """Produce the recommended parameters for the given experiment.

        Default implementation returns the params class with no arguments; subclasses
        override to do instrument-aware tuning.
        """
        return self.params_cls()  # type: ignore[call-arg]

    def validate(self, ep: ExperimentParams) -> list[str]:
        """Return human-readable warnings if `ep` looks unsuited to this operator."""
        return []


class _Registry:
    def __init__(self) -> None:
        self._by_name: dict[str, type[Operator]] = {}

    def register(self, cls: type[Operator]) -> type[Operator]:
        if not getattr(cls, "name", None):
            raise ValueError(f"{cls.__name__} has no name attribute")
        if cls.name in self._by_name and self._by_name[cls.name] is not cls:
            raise ValueError(f"Operator {cls.name!r} already registered")
        self._by_name[cls.name] = cls
        return cls

    def get(self, name: str) -> type[Operator]:
        if name not in self._by_name:
            raise KeyError(f"no operator registered under {name!r}")
        return self._by_name[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))


REGISTRY = _Registry()


def register(cls: type[Operator]) -> type[Operator]:
    """Decorator: register an Operator subclass under its `name`."""
    return REGISTRY.register(cls)


def diagnostic_summary_tuple(diags: list[Diagnostic]) -> tuple[tuple[str, dict[str, float]], ...]:
    """Compact, hashable view of a diagnostics list — for OpRecord and .spec.xml summary."""
    return tuple((d.name, dict(d.summary)) for d in diags)


def hash_op_result_inputs(
    op_name: str, params: OpParams, input_hash: str, lib_versions: dict[str, str]
) -> str:
    """The cache key for one node's output."""
    from dapple.data.hashing import combine_hashes, hash_obj

    return combine_hashes(
        op_name,
        hash_obj(params),
        input_hash,
        hash_obj(lib_versions),
    )


def merge_op_record(
    *,
    op_name: str,
    params: OpParams,
    input_ds: MSIDataset,
    output_ds: MSIDataset,
    diagnostics: list[Diagnostic],
    lib_versions: dict[str, str] | None = None,
) -> Any:
    """Build an `OpRecord` for `output_ds.with_history(...)`.

    `output_hash` is computed as the *cache key* for this op (deterministic in op_name,
    params, input dataset hash, and library versions) — NOT from `output_ds.hash()`,
    because that would equal `input_ds.hash()` since the record has not yet been
    appended to history. This gives every applied op a distinct hash even when the
    backend itself is unchanged (e.g. detect_reference_ions only attaches metadata).
    """
    from dapple.data.dataset import OpRecord
    from dapple.data.hashing import hash_obj

    libs = lib_versions or {}
    output_hash = hash_op_result_inputs(op_name, params, input_ds.hash(), libs)
    # Defensive use of output_ds to avoid unused-var lint complaints; future versions
    # may incorporate output_ds backend hashes too.
    _ = output_ds
    return OpRecord(
        op_name=op_name,
        params_hash=hash_obj(params),
        input_hash=input_ds.hash(),
        output_hash=output_hash,
        diagnostics_summary=diagnostic_summary_tuple(diagnostics),
    )
