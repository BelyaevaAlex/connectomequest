"""Four access-matched full-episode revalidation strategies."""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol

from connectomequest.embodied.evidence_index import (
    EvidenceUpdate,
    PlanDependency,
)
from connectomequest.embodied.evidence_index import (
    ReceiptIndex as DependencyIndex,
)
from connectomequest.embodied.navigation import PlanStep
from connectomequest.embodied.search import WorkCounter
from connectomequest.embodied.types import EpisodeCounters, MethodInput


@dataclass(frozen=True)
class PlannedSuffix:
    plan: tuple[PlanStep, ...]
    dependencies: tuple[PlanDependency, ...]
    work: WorkCounter


class CommonPlanner(Protocol):
    tie_break_id: str

    def __call__(self, method_input: MethodInput) -> PlannedSuffix: ...


@dataclass(frozen=True)
class MethodDecision:
    authorized_action: str | None
    remaining_plan: tuple[PlanStep, ...]
    dependencies: tuple[PlanDependency, ...]
    replanned: bool
    unsupported_authorization: bool
    affected_positions: tuple[int, ...]
    counters: EpisodeCounters
    authorization_receipts: tuple[str, ...]


class RevalidationStrategy(Protocol):
    planner: CommonPlanner
    tie_break_id: str
    index_build_ns: int
    index_peak_bytes: int

    def initialize(
        self,
        plan: tuple[PlanStep, ...],
        dependencies: tuple[PlanDependency, ...],
    ) -> None: ...

    def handle_update(self, method_input: MethodInput) -> MethodDecision: ...


def _event_update(method_input: MethodInput) -> EvidenceUpdate:
    event = method_input.update
    return EvidenceUpdate(
        withdrawn_receipts=frozenset(event.withdrawn_receipt_ids),
        added_keys=frozenset(event.changed_semantic_keys),
        superseded_receipts=frozenset(event.superseded_receipt_ids),
    )


def _has_update(method_input: MethodInput) -> bool:
    event = method_input.update
    return bool(
        event.withdrawn_receipt_ids
        or event.superseded_receipt_ids
        or event.changed_semantic_keys
        or event.added_assertion_bindings
    )


def _support(
    dependency: PlanDependency,
    method_input: MethodInput,
) -> tuple[bool, int, tuple[str, ...]]:
    active = set(method_input.active_receipt_ids)
    bindings = dict(method_input.assertion_to_receipt_bindings)
    receipts: set[str] = set()
    if dependency.required_assertions:
        supported = True
        for assertion in dependency.required_assertions:
            live = active & set(bindings.get(assertion, ()))
            receipts.update(live)
            supported = supported and bool(live)
        return supported, len(dependency.required_assertions), tuple(sorted(receipts))
    live = active & set(dependency.required_receipts)
    return (
        dependency.required_receipts <= active,
        len(dependency.required_receipts),
        tuple(sorted(live)),
    )


def _build_index(
    dependencies: tuple[PlanDependency, ...],
) -> tuple[DependencyIndex, int, int]:
    owned_trace = not tracemalloc.is_tracing()
    if owned_trace:
        tracemalloc.start()
    before_current, before_peak = tracemalloc.get_traced_memory()
    started = perf_counter_ns()
    index = DependencyIndex.from_dependencies(dependencies)
    elapsed = perf_counter_ns() - started
    after_current, after_peak = tracemalloc.get_traced_memory()
    peak_bytes = max(0, after_peak - before_current)
    if owned_trace:
        tracemalloc.stop()
    return index, elapsed, peak_bytes


class _BaseStrategy:
    name = "base"

    def __init__(self, planner: CommonPlanner) -> None:
        if not callable(planner):
            raise TypeError("planner must be callable")
        if not hasattr(planner, "tie_break_id"):
            raise TypeError("planner must declare tie_break_id")
        self.planner = planner
        self.tie_break_id = str(planner.tie_break_id)
        self.plan: tuple[PlanStep, ...] = ()
        self.dependencies: tuple[PlanDependency, ...] = ()
        self.index_build_ns = 0
        self.index_peak_bytes = 0

    def initialize(
        self,
        plan: tuple[PlanStep, ...],
        dependencies: tuple[PlanDependency, ...],
    ) -> None:
        plan = tuple(plan)
        dependencies = tuple(dependencies)
        if tuple(row.step_index for row in dependencies) != tuple(range(len(plan))):
            raise ValueError("dependencies must cover the plan in order")
        self.plan = plan
        self.dependencies = dependencies
        self._initialize_auxiliary()

    def _initialize_auxiliary(self) -> None:
        return None

    def _replace_plan(self, outcome: PlannedSuffix) -> None:
        self.initialize(outcome.plan, outcome.dependencies)

    def _planner_call(self, method_input: MethodInput) -> tuple[PlannedSuffix, int]:
        started = perf_counter_ns()
        outcome = self.planner(method_input)
        elapsed = perf_counter_ns() - started
        if not isinstance(outcome, PlannedSuffix):
            raise TypeError("common planner must return PlannedSuffix")
        return outcome, elapsed

    def _decision(
        self,
        method_input: MethodInput,
        *,
        plan: tuple[PlanStep, ...],
        dependencies: tuple[PlanDependency, ...],
        replanned: bool,
        affected: tuple[int, ...],
        counters: EpisodeCounters,
    ) -> MethodDecision:
        action = plan[0].action if plan else None
        unsupported = False
        receipts: tuple[str, ...] = ()
        if plan and dependencies:
            supported, _, receipts = _support(dependencies[0], method_input)
            unsupported = not supported
        return MethodDecision(
            action,
            plan,
            dependencies,
            replanned,
            unsupported,
            affected,
            counters,
            receipts,
        )

    def _aligned_state(
        self, method_input: MethodInput
    ) -> tuple[tuple[PlanStep, ...], tuple[PlanDependency, ...]]:
        visible_plan = tuple(method_input.remaining_plan)
        if visible_plan != self.plan:
            raise ValueError("method input remaining plan differs from initialized plan")
        if method_input.cursor != 0:
            raise ValueError("remaining-plan inputs use a zero-relative cursor")
        return self.plan, self.dependencies


class FullReplan(_BaseStrategy):
    name = "full_replan"

    def handle_update(self, method_input: MethodInput) -> MethodDecision:
        plan, dependencies = self._aligned_state(method_input)
        started = perf_counter_ns()
        updated = _has_update(method_input)
        revalidation_ns = perf_counter_ns() - started
        if not updated:
            counters = EpisodeCounters(revalidation_ns=revalidation_ns)
            return self._decision(
                method_input,
                plan=plan,
                dependencies=dependencies,
                replanned=False,
                affected=(),
                counters=counters,
            )
        outcome, planning_ns = self._planner_call(method_input)
        self._replace_plan(outcome)
        counters = EpisodeCounters(
            predicate_evaluations=outcome.work.precondition_evaluations,
            successor_expansions=outcome.work.successor_evaluations,
            planner_calls=1,
            revalidation_ns=revalidation_ns,
            planning_ns=planning_ns,
        )
        return self._decision(
            method_input,
            plan=self.plan,
            dependencies=self.dependencies,
            replanned=True,
            affected=tuple(range(len(plan))),
            counters=counters,
        )


class FullScan(_BaseStrategy):
    name = "full_scan"

    def handle_update(self, method_input: MethodInput) -> MethodDecision:
        plan, dependencies = self._aligned_state(method_input)
        started = perf_counter_ns()
        affected = tuple(range(len(plan))) if _has_update(method_input) else ()
        predicates = 0
        supported = True
        for dependency in dependencies if affected else ():
            row_supported, row_predicates, _ = _support(dependency, method_input)
            predicates += row_predicates
            supported = supported and row_supported
        changed = set(method_input.update.changed_semantic_keys)
        reoptimization = any(
            changed & set(dependency.optimization_keys) for dependency in dependencies
        )
        revalidation_ns = perf_counter_ns() - started
        if supported and not reoptimization:
            counters = EpisodeCounters(
                predicate_evaluations=predicates,
                plan_position_visits=len(affected),
                revalidation_ns=revalidation_ns,
            )
            return self._decision(
                method_input,
                plan=plan,
                dependencies=dependencies,
                replanned=False,
                affected=affected,
                counters=counters,
            )
        outcome, planning_ns = self._planner_call(method_input)
        self._replace_plan(outcome)
        counters = EpisodeCounters(
            predicate_evaluations=predicates + outcome.work.precondition_evaluations,
            successor_expansions=outcome.work.successor_evaluations,
            plan_position_visits=len(affected),
            planner_calls=1,
            revalidation_ns=revalidation_ns,
            planning_ns=planning_ns,
        )
        return self._decision(
            method_input,
            plan=self.plan,
            dependencies=self.dependencies,
            replanned=True,
            affected=affected,
            counters=counters,
        )


class ReceiptIndex(_BaseStrategy):
    name = "receipt_index"

    def __init__(self, planner: CommonPlanner) -> None:
        super().__init__(planner)
        self._index: DependencyIndex | None = None

    def _initialize_auxiliary(self) -> None:
        index, elapsed, peak = _build_index(self.dependencies)
        self._index = index
        self.index_build_ns += elapsed
        self.index_peak_bytes = max(self.index_peak_bytes, peak)

    def handle_update(self, method_input: MethodInput) -> MethodDecision:
        plan, dependencies = self._aligned_state(method_input)
        if self._index is None:
            raise RuntimeError("strategy is not initialized")
        started = perf_counter_ns()
        update = _event_update(method_input)
        lookup_keys = (
            set(update.withdrawn_receipts)
            | set(update.superseded_receipts)
            | set(update.added_keys)
        )
        affected = self._index.affected(update, 0) if _has_update(method_input) else ()
        posting_entries = 0
        for receipt in update.withdrawn_receipts | update.superseded_receipts:
            posting_entries += len(self._index.receipt_to_steps.get(receipt, ()))
        for key in update.added_keys:
            posting_entries += len(self._index.optimization_key_to_steps.get(key, ()))
        predicates = 0
        supported = True
        for position in affected:
            row_supported, row_predicates, _ = _support(dependencies[position], method_input)
            predicates += row_predicates
            supported = supported and row_supported
        changed = set(update.added_keys)
        reoptimization = any(
            changed & set(dependencies[position].optimization_keys) for position in affected
        )
        revalidation_ns = perf_counter_ns() - started
        if supported and not reoptimization:
            counters = EpisodeCounters(
                predicate_evaluations=predicates,
                plan_position_visits=len(affected),
                index_lookups=len(lookup_keys),
                posting_entries=posting_entries,
                revalidation_ns=revalidation_ns,
            )
            return self._decision(
                method_input,
                plan=plan,
                dependencies=dependencies,
                replanned=False,
                affected=affected,
                counters=counters,
            )
        outcome, planning_ns = self._planner_call(method_input)
        self._replace_plan(outcome)
        counters = EpisodeCounters(
            predicate_evaluations=predicates + outcome.work.precondition_evaluations,
            successor_expansions=outcome.work.successor_evaluations,
            plan_position_visits=len(affected),
            index_lookups=len(lookup_keys),
            posting_entries=posting_entries,
            planner_calls=1,
            revalidation_ns=revalidation_ns,
            planning_ns=planning_ns,
        )
        return self._decision(
            method_input,
            plan=self.plan,
            dependencies=self.dependencies,
            replanned=True,
            affected=affected,
            counters=counters,
        )


class UncheckedReuse(_BaseStrategy):
    name = "unchecked_reuse"

    def handle_update(self, method_input: MethodInput) -> MethodDecision:
        plan, dependencies = self._aligned_state(method_input)
        started = perf_counter_ns()
        revalidation_ns = perf_counter_ns() - started
        return self._decision(
            method_input,
            plan=plan,
            dependencies=dependencies,
            replanned=False,
            affected=(),
            counters=EpisodeCounters(revalidation_ns=revalidation_ns),
        )
