"""Access-matched plan revalidation strategies over explicit receipt support."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from connectomequest.embodied.evidence_index import EvidenceUpdate, PlanDependency, ReceiptIndex
from connectomequest.embodied.navigation import PlanStep
from connectomequest.embodied.search import (
    WorkCounter,
    requirements_supported_counted,
    route_counted,
)

RevalidationMethod = Literal[
    "full_replan",
    "complete_revalidation",
    "dependency_unaware",
    "receipt_indexed",
    "unchecked_cache",
    "dependency_oracle",
]
METHODS = (
    "full_replan",
    "complete_revalidation",
    "dependency_unaware",
    "receipt_indexed",
    "unchecked_cache",
    "dependency_oracle",
)


@dataclass(frozen=True)
class EvidenceState:
    active_receipts: frozenset[str]
    assertion_receipts: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class RevalidationInput:
    plan: tuple[PlanStep, ...]
    cursor: int
    dependencies: tuple[PlanDependency, ...]
    update: EvidenceUpdate
    evidence: EvidenceState
    belief: object
    pose: tuple[tuple[int, int], int]
    goals: frozenset[tuple[tuple[int, int], int]]
    has_key: bool
    trusted_update: bool
    provenance_complete: bool
    sealed_plan_sha256: str
    oracle_affected_steps: tuple[int, ...] | None = None
    key_label: str | None = None


@dataclass(frozen=True)
class RevalidationResult:
    authorized: bool
    replan: bool
    selected_plan: tuple[PlanStep, ...]
    affected_steps: tuple[int, ...]
    unsupported_next_action: bool
    work: WorkCounter
    authorization_receipts: frozenset[str]


def plan_sha256(plan: tuple[PlanStep, ...]) -> str:
    payload = [
        {
            "action": step.action,
            "position": list(step.position),
            "direction": int(step.direction),
            "requirements": [[list(position), label] for position, label in step.requirements],
        }
        for step in plan
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _has_update(update: EvidenceUpdate) -> bool:
    return bool(update.withdrawn_receipts or update.superseded_receipts or update.added_keys)


def _dependency_supported(dependency: PlanDependency, evidence: EvidenceState) -> bool:
    if dependency.required_assertions:
        return all(
            bool(evidence.assertion_receipts.get(assertion, frozenset()) & evidence.active_receipts)
            for assertion in dependency.required_assertions
        )
    return dependency.required_receipts <= evidence.active_receipts


def _support_check(
    indices: tuple[int, ...],
    dependencies: tuple[PlanDependency, ...],
    evidence: EvidenceState,
) -> tuple[bool, WorkCounter]:
    preconditions = 0
    validations = 0
    by_index = {row.step_index: row for row in dependencies}
    for index in indices:
        validations += 1
        dependency = by_index[index]
        checks = dependency.required_assertions or dependency.required_receipts
        preconditions += len(checks)
        if not _dependency_supported(dependency, evidence):
            return False, WorkCounter(
                precondition_evaluations=preconditions,
                step_validations=validations,
            )
    return True, WorkCounter(
        precondition_evaluations=preconditions,
        step_validations=validations,
    )


def _step_assertions(step: PlanStep) -> frozenset[str]:
    return frozenset(
        f"cell:{int(position[0])},{int(position[1])}={label}"
        for position, label in step.requirements
    )


def _authorization_receipts(
    step: PlanStep,
    dependency: PlanDependency | None,
    evidence: EvidenceState,
) -> frozenset[str]:
    assertions = dependency.required_assertions if dependency else _step_assertions(step)
    receipts: set[str] = set()
    for assertion in assertions:
        receipts.update(evidence.assertion_receipts.get(assertion, frozenset()))
    if dependency and not assertions:
        receipts.update(dependency.required_receipts)
    return frozenset(receipts & evidence.active_receipts)


def revalidate(method: RevalidationMethod, request: RevalidationInput) -> RevalidationResult:
    if method not in METHODS:
        raise ValueError(f"unknown revalidation method: {method}")
    if not request.trusted_update:
        raise ValueError("untrusted evidence update")
    if plan_sha256(request.plan) != request.sealed_plan_sha256:
        raise ValueError("plan hash mismatch")

    index = ReceiptIndex.from_dependencies(request.dependencies)
    index.validate_complete(len(request.plan))
    if not 0 <= request.cursor < len(request.plan):
        raise ValueError("cursor outside plan")
    update_present = _has_update(request.update)
    future = tuple(range(request.cursor, len(request.plan)))
    index_work = WorkCounter()

    if method == "dependency_oracle":
        if request.oracle_affected_steps is None:
            raise ValueError("oracle affected steps are required")
        affected = tuple(sorted(set(request.oracle_affected_steps) & set(future)))
    elif method in ("complete_revalidation", "dependency_unaware") or method == "full_replan":
        affected = future if update_present else ()
    elif method == "receipt_indexed" and request.provenance_complete:
        affected = index.affected(request.update, request.cursor)
        index_work = WorkCounter(
            index_lookups=len(
                request.update.withdrawn_receipts
                | request.update.superseded_receipts
                | request.update.added_keys
            )
        )
    elif method == "receipt_indexed":
        affected = future if update_present else ()
    else:
        affected = ()

    support_ok = True
    validation_work = WorkCounter()
    if method not in ("unchecked_cache", "full_replan") and affected:
        support_ok, validation_work = _support_check(
            affected, request.dependencies, request.evidence
        )

    added_reoptimization = bool(request.update.added_keys) and bool(affected)
    do_replan = bool(
        update_present
        and (
            method == "full_replan"
            or (method != "unchecked_cache" and (not support_ok or added_reoptimization))
        )
    )
    selected = request.plan[request.cursor :]
    planner_work = WorkCounter()
    if do_replan:
        planned = route_counted(
            request.belief,
            tuple(request.pose[0]),
            int(request.pose[1]),
            goals=set(request.goals),
            has_key=request.has_key,
            key_label=request.key_label,
        )
        selected = () if planned.plan is None else planned.plan
        planner_work = planned.work

    original_dependency = None
    if selected and selected[0] == request.plan[request.cursor]:
        original_dependency = request.dependencies[request.cursor]
    receipts = (
        _authorization_receipts(selected[0], original_dependency, request.evidence)
        if selected
        else frozenset()
    )
    labels_supported = False
    if selected:
        labels_supported, _ = requirements_supported_counted(selected[:1], request.belief)
    evidence_required = bool(selected and selected[0].requirements)
    evidence_supported = bool(receipts) or not evidence_required
    next_supported = bool(selected) and labels_supported and evidence_supported
    authorized = bool(selected) if method == "unchecked_cache" else next_supported
    unsupported = authorized and not next_supported

    return RevalidationResult(
        authorized=authorized,
        replan=do_replan,
        selected_plan=tuple(selected),
        affected_steps=affected,
        unsupported_next_action=unsupported,
        work=index_work + validation_work + planner_work,
        authorization_receipts=receipts,
    )
