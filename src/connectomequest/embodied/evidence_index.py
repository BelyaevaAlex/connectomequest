"""Deterministic receipt-to-plan dependency indexing.

The index contains no environment state.  It maps explicitly recorded evidence
and optimization dependencies to future plan positions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PlanDependency:
    step_index: int
    required_receipts: frozenset[str]
    optimization_keys: frozenset[str]
    required_assertions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EvidenceUpdate:
    withdrawn_receipts: frozenset[str]
    added_keys: frozenset[str]
    superseded_receipts: frozenset[str]


@dataclass(frozen=True)
class ReceiptIndex:
    receipt_to_steps: Mapping[str, tuple[int, ...]]
    optimization_key_to_steps: Mapping[str, tuple[int, ...]]
    step_indices: tuple[int, ...]
    plan_length: int

    @classmethod
    def from_dependencies(cls, dependencies: Iterable[PlanDependency]) -> ReceiptIndex:
        rows = tuple(dependencies)
        indices = tuple(row.step_index for row in rows)
        if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
            raise ValueError("step indexes must be unique non-negative integers")
        plan_length = max(indices, default=-1) + 1
        if tuple(sorted(indices)) != tuple(range(plan_length)):
            raise ValueError("dependency records must cover every plan step")

        receipt_steps: dict[str, set[int]] = defaultdict(set)
        key_steps: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            for receipt in row.required_receipts:
                receipt_steps[receipt].add(row.step_index)
            for key in row.optimization_keys:
                key_steps[key].add(row.step_index)
        return cls(
            receipt_to_steps=MappingProxyType(
                {key: tuple(sorted(value)) for key, value in receipt_steps.items()}
            ),
            optimization_key_to_steps=MappingProxyType(
                {key: tuple(sorted(value)) for key, value in key_steps.items()}
            ),
            step_indices=tuple(sorted(indices)),
            plan_length=plan_length,
        )

    def affected(self, update: EvidenceUpdate, cursor: int) -> tuple[int, ...]:
        if not 0 <= cursor <= self.plan_length:
            raise ValueError("cursor outside plan")
        affected: set[int] = set()
        for receipt in update.withdrawn_receipts | update.superseded_receipts:
            affected.update(self.receipt_to_steps.get(receipt, ()))
        for key in update.added_keys:
            affected.update(self.optimization_key_to_steps.get(key, ()))
        return tuple(sorted(index for index in affected if index >= cursor))

    def validate_complete(self, plan_length: int) -> None:
        expected = tuple(range(plan_length))
        if plan_length != self.plan_length or self.step_indices != expected:
            raise ValueError("incomplete dependency index")
