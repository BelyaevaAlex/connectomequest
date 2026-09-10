"""Prospective amortized validation audit on acquired RGB plans.

The module deliberately separates case construction from timing.  Generator
labels are used only to enroll eligible cases; timed methods receive the same
``MethodInput`` sequence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median
from typing import Any

from connectomequest.embodied.evidence_index import PlanDependency
from connectomequest.embodied.revalidation import FullScan, ReceiptIndex
from connectomequest.embodied.types import EvidenceEvent, MethodInput, canonical_json_bytes


@dataclass(frozen=True)
class AmortizedProtocol:
    stage: str
    layout_seeds: tuple[int, ...]
    update_counts: tuple[int, ...] = (1, 4, 8)
    update_seeds: tuple[int, ...] = (17, 29, 43)
    methods: tuple[str, ...] = ("complete_validation", "dependency_indexed")
    minimum_plan_length: int = 32
    warmups: int = 10
    repetitions: int = 40
    bootstrap_seed: int = 40_040
    bootstrap_draws: int = 10_000

    @classmethod
    def development(cls) -> AmortizedProtocol:
        return cls("development", tuple(range(40_030_000, 40_030_032)))

    @classmethod
    def confirmation(cls) -> AmortizedProtocol:
        return cls("confirmation", tuple(range(40_040_000, 40_040_256)))

    def __post_init__(self) -> None:
        if self.stage not in {"development", "confirmation"}:
            raise ValueError("unknown stage")
        if len(set(self.layout_seeds)) != len(self.layout_seeds):
            raise ValueError("layout seeds must be unique")
        if not self.layout_seeds or min(self.update_counts) <= 0:
            raise ValueError("non-empty positive protocol required")
        if self.warmups < 0 or self.repetitions <= 0:
            raise ValueError("invalid repetition counts")


@dataclass(frozen=True)
class TimingRow:
    method: str = ""
    layout_seed: int = 0
    update_seed: int = 0
    plan_length: int = 0
    update_count: int = 0
    index_build_ns: int = 0
    index_maintenance_ns: int = 0
    validation_ns: int = 0
    planning_ns: int = 0
    total_reasoning_ns: int = 0
    raw_repetitions_ns: tuple[int, ...] = ()
    method_input_sha256: str = ""
    decision_sha256: str = ""
    predicate_evaluations: int = 0
    plan_position_visits: int = 0
    index_lookups: int = 0
    posting_entries: int = 0

    def validate(self) -> None:
        components = (
            self.index_build_ns,
            self.index_maintenance_ns,
            self.validation_ns,
            self.planning_ns,
        )
        if any(value < 0 for value in components + self.raw_repetitions_ns):
            raise ValueError("negative timing")
        if self.total_reasoning_ns != sum(components):
            raise ValueError("incomplete total-reasoning accounting")
        if not self.raw_repetitions_ns:
            raise ValueError("raw repetitions are required")


def _ordered(values: Iterable[str], *, update_seed: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: hashlib.sha256(f"{update_seed}:{value}".encode()).digest(),
        )
    )


def prepare_case_stream(
    *,
    dependencies: tuple[PlanDependency, ...],
    active_receipts: tuple[str, ...],
    update_count: int,
    update_seed: int,
) -> tuple[EvidenceEvent, ...]:
    """Select paid-but-plan-irrelevant receipt withdrawals prospectively."""

    required = set().union(*(row.required_receipts for row in dependencies))
    candidates = _ordered(set(active_receipts) - required, update_seed=update_seed)
    if len(candidates) < update_count:
        raise ValueError("insufficient irrelevant receipts")
    return tuple(
        EvidenceEvent(
            action_index=offset,
            withdrawn_receipt_ids=(receipt,),
            superseded_receipt_ids=(),
            changed_semantic_keys=(),
            added_assertion_bindings=(),
            condition="irrelevant_receipt_withdrawal",
        )
        for offset, receipt in enumerate(candidates[:update_count])
    )


def stream_sha256(inputs: tuple[MethodInput, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(inputs)).hexdigest()


def decision_sha256(decisions: Iterable[Any]) -> str:
    rows = [
        {
            "authorized_action": row.authorized_action,
            "remaining_plan": row.remaining_plan,
            "replanned": row.replanned,
            "unsupported_authorization": row.unsupported_authorization,
            "authorization_receipts": row.authorization_receipts,
        }
        for row in decisions
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


class _NoReplan:
    tie_break_id = "not-invoked-for-irrelevant-withdrawals"

    def __call__(self, _method_input: MethodInput):
        raise AssertionError("irrelevant-withdrawal audit must not invoke the planner")


def time_case_stream(
    *,
    method: str,
    plan: tuple[Any, ...],
    dependencies: tuple[PlanDependency, ...],
    inputs: tuple[MethodInput, ...],
    layout_seed: int,
    update_seed: int,
    warmups: int,
    repetitions: int,
) -> tuple[TimingRow, str]:
    """Time one sequence, including index construction on every repetition."""

    strategy_type = {
        "complete_validation": FullScan,
        "dependency_indexed": ReceiptIndex,
    }.get(method)
    if strategy_type is None:
        raise ValueError(f"unknown method: {method}")
    if not inputs or repetitions <= 0 or warmups < 0:
        raise ValueError("invalid timing request")

    def execute():
        strategy = strategy_type(_NoReplan())
        strategy.initialize(plan, dependencies)
        decisions = []
        validation_ns = planning_ns = 0
        predicates = positions = lookups = postings = 0
        for method_input in inputs:
            decision = strategy.handle_update(method_input)
            decisions.append(decision)
            counter = decision.counters
            validation_ns += counter.revalidation_ns
            planning_ns += counter.planning_ns
            predicates += counter.predicate_evaluations
            positions += counter.plan_position_visits
            lookups += counter.index_lookups
            postings += counter.posting_entries
        build_ns = strategy.index_build_ns
        maintenance_ns = 0  # Withdrawals do not modify immutable plan postings.
        total = build_ns + maintenance_ns + validation_ns + planning_ns
        return (
            total,
            build_ns,
            maintenance_ns,
            validation_ns,
            planning_ns,
            predicates,
            positions,
            lookups,
            postings,
            decision_sha256(decisions),
        )

    for _ in range(warmups):
        execute()
    measured = [execute() for _ in range(repetitions)]
    hashes = {row[-1] for row in measured}
    if len(hashes) != 1:
        raise ValueError("nondeterministic decisions")

    medians = [int(median(row[index] for row in measured)) for index in range(9)]
    total_ns, build_ns, maintenance_ns, validation_ns, planning_ns = medians[:5]
    # Report the additive component total; retain directly observed totals raw.
    total_ns = build_ns + maintenance_ns + validation_ns + planning_ns
    row = TimingRow(
        method=method,
        layout_seed=int(layout_seed),
        update_seed=int(update_seed),
        plan_length=len(plan),
        update_count=len(inputs),
        index_build_ns=build_ns,
        index_maintenance_ns=maintenance_ns,
        validation_ns=validation_ns,
        planning_ns=planning_ns,
        total_reasoning_ns=total_ns,
        raw_repetitions_ns=tuple(int(value[0]) for value in measured),
        method_input_sha256=stream_sha256(inputs),
        decision_sha256=next(iter(hashes)),
        predicate_evaluations=medians[5],
        plan_position_visits=medians[6],
        index_lookups=medians[7],
        posting_entries=medians[8],
    )
    row.validate()
    return row, row.decision_sha256


def canonical_protocol(protocol: AmortizedProtocol) -> bytes:
    return canonical_json_bytes(protocol)


def protocol_sha256(protocol: AmortizedProtocol) -> str:
    return hashlib.sha256(canonical_protocol(protocol)).hexdigest()
