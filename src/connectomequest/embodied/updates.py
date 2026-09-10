"""Prospective, method-independent evidence-update schedules for V39."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from connectomequest.embodied.evidence_index import PlanDependency
from connectomequest.embodied.types import EvidenceEvent

CONDITIONS: Final = (
    "none",
    "irrelevant_receipt_withdrawal",
    "alternative_support",
    "final_support_loss",
    "semantic_correction",
    "sparse_burst",
)
EXCLUSION_REASONS: Final = frozenset(
    {
        "no_irrelevant_receipt",
        "no_alternative_support",
        "no_future_supported_precondition",
        "no_prefix_correction_receipt",
    }
)


@dataclass(frozen=True)
class CorrectionBinding:
    semantic_key: str
    old_assertion: str
    new_assertion: str
    receipt_id: str


@dataclass(frozen=True)
class SchedulePrefix:
    action_count: int
    active_receipt_ids: tuple[str, ...]
    assertion_to_receipt_bindings: tuple[tuple[str, tuple[str, ...]], ...]
    remaining_dependencies: tuple[PlanDependency, ...]
    correction_bindings: tuple[CorrectionBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.action_count < 0:
            raise ValueError("action_count must be non-negative")


@dataclass(frozen=True)
class UpdateAudit:
    action_index: int
    relevant: bool
    assertion: str | None
    affected_plan_positions: tuple[int, ...]
    generator_label: str


@dataclass(frozen=True)
class UpdateSchedule:
    events: tuple[EvidenceEvent, ...]
    audit: tuple[UpdateAudit, ...]
    exclusion_reason: str | None
    realized_relevant_fraction: float


def _layout_seed(layout_seed: object) -> int:
    if isinstance(layout_seed, int):
        return layout_seed
    spec = getattr(layout_seed, "spec", None)
    if spec is not None and hasattr(spec, "layout_seed"):
        return int(spec.layout_seed)
    if hasattr(layout_seed, "layout_seed"):
        return int(layout_seed.layout_seed)
    raise TypeError("layout_seed must be an integer or generated layout")


def _ordered(values, *, seed: int, tag: str):
    def key(value):
        payload = f"{seed}:{tag}:{value}".encode()
        return hashlib.sha256(payload).digest()

    return tuple(sorted(values, key=key))


def _empty(reason: str | None = None) -> UpdateSchedule:
    return UpdateSchedule((), (), reason, 0.0)


def _dependency_maps(prefix: SchedulePrefix):
    receipt_positions: dict[str, set[int]] = {}
    key_positions: dict[str, set[int]] = {}
    assertion_positions: dict[str, set[int]] = {}
    assertion_bindings = {
        assertion: tuple(receipts) for assertion, receipts in prefix.assertion_to_receipt_bindings
    }
    for dependency in prefix.remaining_dependencies:
        for receipt in dependency.required_receipts:
            receipt_positions.setdefault(receipt, set()).add(dependency.step_index)
        for semantic_key in dependency.optimization_keys:
            key_positions.setdefault(semantic_key, set()).add(dependency.step_index)
        for assertion in dependency.required_assertions:
            assertion_positions.setdefault(assertion, set()).add(dependency.step_index)
            for receipt in assertion_bindings.get(assertion, ()):
                receipt_positions.setdefault(receipt, set()).add(dependency.step_index)
    return receipt_positions, key_positions, assertion_positions


def _single_event(
    prefix: SchedulePrefix,
    *,
    seed: int,
    condition: str,
) -> tuple[EvidenceEvent, UpdateAudit] | str:
    active = set(prefix.active_receipt_ids)
    bindings = {key: tuple(value) for key, value in prefix.assertion_to_receipt_bindings}
    receipt_positions, key_positions, assertion_positions = _dependency_maps(prefix)
    action_index = prefix.action_count

    if condition == "irrelevant_receipt_withdrawal":
        candidates = active - set(receipt_positions)
        if not candidates:
            return "no_irrelevant_receipt"
        receipt = _ordered(candidates, seed=seed, tag=condition)[0]
        event = EvidenceEvent(action_index, (receipt,), (), (), (), condition)
        audit = UpdateAudit(action_index, False, None, (), condition)
        return event, audit

    if condition == "alternative_support":
        candidates = []
        for assertion, receipts in bindings.items():
            live = tuple(sorted(set(receipts) & active))
            positions = assertion_positions.get(assertion, set())
            if len(live) >= 2 and positions:
                candidates.append((assertion, live, tuple(sorted(positions))))
        if not candidates:
            return "no_alternative_support"
        assertion, live, positions = _ordered(candidates, seed=seed, tag=condition)[0]
        event = EvidenceEvent(action_index, (live[0],), (), (), (), condition)
        audit = UpdateAudit(action_index, True, assertion, positions, condition)
        return event, audit

    if condition == "final_support_loss":
        candidates = []
        for assertion, receipts in bindings.items():
            live = tuple(sorted(set(receipts) & active))
            positions = assertion_positions.get(assertion, set())
            if live and positions:
                candidates.append((assertion, live, tuple(sorted(positions))))
        if not candidates:
            return "no_future_supported_precondition"
        assertion, live, positions = min(candidates, key=lambda row: (row[2][0], row[0]))
        event = EvidenceEvent(action_index, live, (), (), (), condition)
        audit = UpdateAudit(action_index, True, assertion, positions, condition)
        return event, audit

    if condition == "semantic_correction":
        candidates = []
        for correction in prefix.correction_bindings:
            if correction.receipt_id not in active:
                continue
            old_receipts = tuple(sorted(set(bindings.get(correction.old_assertion, ())) & active))
            positions = tuple(sorted(assertion_positions.get(correction.old_assertion, set())))
            if old_receipts and positions:
                candidates.append((correction, old_receipts, positions))
        if not candidates:
            if not prefix.correction_bindings:
                return "no_prefix_correction_receipt"
            return "no_future_supported_precondition"
        correction, old_receipts, positions = _ordered(candidates, seed=seed, tag=condition)[0]
        event = EvidenceEvent(
            action_index,
            old_receipts,
            old_receipts,
            (correction.semantic_key,),
            ((correction.new_assertion, (correction.receipt_id,)),),
            condition,
        )
        audit = UpdateAudit(
            action_index,
            True,
            correction.old_assertion,
            positions,
            condition,
        )
        return event, audit

    raise ValueError(f"condition is not a single-event condition: {condition}")


def build_update_schedule(
    layout_seed,
    prefix: SchedulePrefix,
    update_seed: int,
    condition: str,
    count: int,
    relevant_fraction: float,
) -> UpdateSchedule:
    """Create a sealed schedule without accepting any evaluated-method outcome."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown update condition: {condition}")
    if count < 0:
        raise ValueError("count must be non-negative")
    if not 0.0 <= relevant_fraction <= 1.0:
        raise ValueError("relevant_fraction must be in [0, 1]")
    if condition == "none":
        if count != 0:
            raise ValueError("the none condition requires count=0")
        return _empty()
    if count == 0:
        raise ValueError("non-empty conditions require count>0")

    seed = _layout_seed(layout_seed) ^ (int(update_seed) << 1)
    if condition != "sparse_burst":
        if count != 1:
            raise ValueError("single-update conditions require count=1")
        built = _single_event(prefix, seed=seed, condition=condition)
        if isinstance(built, str):
            return _empty(built)
        event, audit = built
        return UpdateSchedule((event,), (audit,), None, float(audit.relevant))

    relevant_count = min(count, max(0, int(round(count * relevant_fraction))))
    irrelevant_count = count - relevant_count
    irrelevant = _single_event(prefix, seed=seed, condition="irrelevant_receipt_withdrawal")
    if irrelevant_count and isinstance(irrelevant, str):
        return _empty(irrelevant)
    receipt_positions, key_positions, _ = _dependency_maps(prefix)
    relevant_keys = tuple(key_positions)
    if relevant_count and not relevant_keys:
        return _empty("no_future_supported_precondition")
    irrelevant_receipts = tuple(
        receipt
        for receipt in _ordered(
            set(prefix.active_receipt_ids) - set(receipt_positions),
            seed=seed,
            tag="sparse-irrelevant",
        )
    )
    if irrelevant_count and not irrelevant_receipts:
        return _empty("no_irrelevant_receipt")

    labels = [True] * relevant_count + [False] * irrelevant_count
    labels = list(_ordered(labels, seed=seed, tag="sparse-order"))
    events = []
    audits = []
    for offset, is_relevant in enumerate(labels):
        action_index = prefix.action_count + offset
        if is_relevant:
            semantic_key = _ordered(relevant_keys, seed=seed + offset, tag="sparse-key")[0]
            positions = tuple(sorted(key_positions[semantic_key]))
            event = EvidenceEvent(
                action_index,
                (),
                (),
                (semantic_key,),
                (),
                "sparse_burst",
            )
            audit = UpdateAudit(
                action_index,
                True,
                None,
                positions,
                "sparse_relevant_semantic_key",
            )
        else:
            receipt = irrelevant_receipts[offset % len(irrelevant_receipts)]
            event = EvidenceEvent(
                action_index,
                (receipt,),
                (),
                (),
                (),
                "sparse_burst",
            )
            audit = UpdateAudit(
                action_index,
                False,
                None,
                (),
                "sparse_irrelevant_withdrawal",
            )
        events.append(event)
        audits.append(audit)
    return UpdateSchedule(
        tuple(events),
        tuple(audits),
        None,
        relevant_count / count,
    )
