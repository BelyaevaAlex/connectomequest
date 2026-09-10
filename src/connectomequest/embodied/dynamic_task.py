"""Public-observation utilities for dynamic UnlockPickupDist evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from connectomequest.embodied.evidence_index import PlanDependency
from connectomequest.embodied.indexed_revalidation import plan_sha256
from connectomequest.embodied.navigation import PlanStep

CONDITIONS = (
    "clean",
    "irrelevant_update",
    "alternative_support",
    "relevant_support_loss",
    "false_positive_correction",
    "compound_occlusion_correction",
    "sparse_update_burst",
)

ACCESS_FIELDS = (
    "seed",
    "corruption_seed",
    "rgb_receipts",
    "assertion_updates",
    "active_receipts",
    "assertion_receipts",
    "belief_cells",
    "pose",
    "inventory",
    "plan_sha256",
    "planner",
    "actions",
)


@dataclass(frozen=True)
class DynamicCase:
    seed: int
    corruption_seed: int
    selected: bool
    exclusion_reason: str | None
    initial_receipt: str
    rgb_receipts: tuple[str, ...]
    prefix: tuple[dict, ...]
    plan: tuple[PlanStep, ...]
    dependencies: tuple[PlanDependency, ...]
    active_receipts: frozenset[str]
    assertion_receipts: Mapping[str, frozenset[str]]
    belief_cells: tuple[tuple[tuple[int, int], str], ...]
    support: tuple[tuple[tuple[int, int], str], ...]
    pose: tuple[tuple[int, int], int]
    inventory: str | None
    goals: frozenset[tuple[tuple[int, int], int]]
    plan_sha256: str


class FrozenPerceptionCache:
    """Content-addressed cache bound to exactly one frozen checkpoint."""

    def __init__(self, perceptor, checkpoint_sha256: str):
        self._perceptor = perceptor
        self._checkpoint_sha256 = checkpoint_sha256
        self._predictions: dict[str, tuple] = {}

    @property
    def entries(self) -> int:
        return len(self._predictions)

    def verify_checkpoint(self, checkpoint_sha256: str) -> None:
        if checkpoint_sha256 != self._checkpoint_sha256:
            raise ValueError("checkpoint hash mismatch")

    def predict(self, rgb: np.ndarray) -> tuple:
        contiguous = np.ascontiguousarray(rgb)
        digest = hashlib.sha256(self._checkpoint_sha256.encode() + contiguous.tobytes()).hexdigest()
        if digest not in self._predictions:
            self._predictions[digest] = tuple(
                tuple(value for value in row) for row in self._perceptor.predict(contiguous)
            )
        return self._predictions[digest]


def _assertion_key(position: tuple[int, int], label: str) -> str:
    return f"cell:{int(position[0])},{int(position[1])}={label}"


def _optimization_key(position: tuple[int, int]) -> str:
    return f"cell:{int(position[0])},{int(position[1])}"


def plan_dependencies(
    plan: tuple[PlanStep, ...],
    *,
    support: Mapping[tuple[int, int], str],
) -> tuple[PlanDependency, ...]:
    """Bind every recorded symbolic requirement to its acquired RGB receipt."""

    rows = []
    for index, step in enumerate(plan):
        receipts = set()
        assertions = set()
        keys = set()
        for raw_position, label in step.requirements:
            position = tuple(raw_position)
            receipt = support.get(position)
            if not receipt:
                raise ValueError(f"missing receipt support for {position}")
            receipts.add(receipt)
            assertions.add(_assertion_key(position, label))
            keys.add(_optimization_key(position))
        rows.append(
            PlanDependency(
                step_index=index,
                required_receipts=frozenset(receipts),
                optimization_keys=frozenset(keys),
                required_assertions=frozenset(assertions),
            )
        )
    return tuple(rows)


def access_signature(row: Mapping[str, object]) -> str:
    """Hash only method-independent information available before a decision."""

    missing = [field for field in ACCESS_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing access fields: {', '.join(missing)}")
    payload = {field: row[field] for field in ACCESS_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _as_plan_step(step) -> PlanStep:
    requirements = ()
    if step.action not in ("left", "right"):
        front = (
            int(step.position[0]) + ((1, 0, -1, 0)[int(step.direction)]),
            int(step.position[1]) + ((0, 1, 0, -1)[int(step.direction)]),
        )
        requirements = ((front, step.front),)
    return PlanStep(step.action, tuple(step.position), int(step.direction), requirements)


def _endpoint(plan: tuple[PlanStep, ...], pose: tuple[tuple[int, int], int]):
    position, direction = tuple(pose[0]), int(pose[1])
    for step in plan:
        if (tuple(step.position), int(step.direction)) != (position, direction):
            raise ValueError("plan kinematics mismatch")
        if step.action == "left":
            direction = (direction - 1) % 4
        elif step.action == "right":
            direction = (direction + 1) % 4
        elif step.action == "forward":
            direction_vector = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
            position = (position[0] + direction_vector[0], position[1] + direction_vector[1])
        else:
            raise ValueError("enrollment requires a pure locomotion subplan")
    return position, direction


def enroll_case(
    seed: int,
    corruption_seed: int,
    perceptor,
    *,
    max_steps: int = 96,
) -> DynamicCase:
    """Enroll the first supported public-RGB locomotion subplan of length four."""

    from connectomequest.embodied.standard_benchmark import PublicDistractorTask
    from connectomequest.embodied.symbolic_controller import Controller

    env = PublicDistractorTask(seed)
    agent = Controller(perceptor, "persistent")
    initial_receipt = env.observation().receipt_id
    receipts = [initial_receipt]
    prefix: list[dict] = []
    try:
        while not env.observation().terminated and env.observation().step <= max_steps:
            obs = env.observation()
            action = agent.act(obs)
            current_requirements = tuple(
                (tuple(position), label)
                for position, label, receipt in agent.requirements
                if receipt is not None
            )
            current = PlanStep(
                action,
                tuple(obs.position),
                int(obs.direction),
                current_requirements,
            )
            proposed = (current,) + tuple(_as_plan_step(step) for step in agent.plan)
            locomotion_length = next(
                (
                    index
                    for index, step in enumerate(proposed)
                    if step.action not in ("left", "right", "forward")
                ),
                len(proposed),
            )
            candidate = proposed[:locomotion_length]
            pure = len(candidate) >= 4
            if pure:
                try:
                    dependencies = plan_dependencies(candidate, support=agent.support)
                    endpoint = _endpoint(candidate, (tuple(obs.position), int(obs.direction)))
                except ValueError:
                    dependencies = ()
                if dependencies:
                    assertion_receipts: dict[str, set[str]] = {}
                    for dependency in dependencies:
                        for assertion in dependency.required_assertions:
                            assertion_receipts.setdefault(assertion, set()).update(
                                dependency.required_receipts
                            )
                    return DynamicCase(
                        seed=seed,
                        corruption_seed=corruption_seed,
                        selected=True,
                        exclusion_reason=None,
                        initial_receipt=initial_receipt,
                        rgb_receipts=tuple(receipts),
                        prefix=tuple(prefix),
                        plan=candidate,
                        dependencies=dependencies,
                        active_receipts=frozenset(agent.support.values()),
                        assertion_receipts={
                            key: frozenset(value)
                            for key, value in sorted(assertion_receipts.items())
                        },
                        belief_cells=tuple(sorted(agent.cells.items())),
                        support=tuple(sorted(agent.support.items())),
                        pose=(tuple(obs.position), int(obs.direction)),
                        inventory=agent.inventory.label,
                        goals=frozenset({endpoint}),
                        plan_sha256=plan_sha256(candidate),
                    )
            if obs.step == max_steps:
                break
            after = env.step(action)
            prefix.append(
                {
                    "before": obs.receipt_id,
                    "action": action,
                    "after": after.receipt_id,
                    "acknowledged": bool(after.acknowledged),
                }
            )
            receipts.append(after.receipt_id)
    finally:
        env.close()

    return DynamicCase(
        seed=seed,
        corruption_seed=corruption_seed,
        selected=False,
        exclusion_reason="no_supported_locomotion_subplan",
        initial_receipt=initial_receipt,
        rgb_receipts=tuple(receipts),
        prefix=tuple(prefix),
        plan=(),
        dependencies=(),
        active_receipts=frozenset(),
        assertion_receipts={},
        belief_cells=(),
        support=(),
        pose=((0, 0), 0),
        inventory=None,
        goals=frozenset(),
        plan_sha256=plan_sha256(()),
    )


@dataclass(frozen=True)
class DynamicCondition:
    """Method-independent evidence state presented at the revalidation decision."""

    condition: str
    update: object
    active_receipts: frozenset[str]
    assertion_receipts: Mapping[str, frozenset[str]]
    belief_cells: tuple[tuple[tuple[int, int], str], ...]
    access_payload: Mapping[str, object]
    grounding: str


@dataclass(frozen=True)
class EpisodeRow:
    seed: int
    corruption_seed: int
    condition: str
    method: str
    access_sha256: str
    update_sha256: str
    plan_sha256: str
    authorized: bool
    replan: bool
    affected_steps: tuple[int, ...]
    unsupported_authorizations: int
    success: bool
    environment_actions: int
    primitive_evaluations: int
    precondition_evaluations: int
    successor_evaluations: int
    states_settled: int
    index_lookups: int
    step_validations: int
    planner_calls: int
    authorization_receipts: tuple[str, ...]
    exact_replay: bool
    trace_sha256: str
    grounding: str
    environment_receipts: tuple[str, ...]


class _Belief:
    def __init__(self, cells) -> None:
        self.cells = dict(cells)

    def label(self, position):
        return self.cells.get(tuple(position))


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _relevant_dependencies(case: DynamicCase) -> tuple[PlanDependency, ...]:
    return tuple(row for row in case.dependencies if row.required_receipts)


def build_condition(case: DynamicCase, condition: str) -> DynamicCondition:
    """Construct one predeclared update without consulting any method outcome."""

    from connectomequest.embodied.evidence_index import EvidenceUpdate

    if condition not in CONDITIONS:
        raise ValueError(f"unknown dynamic condition: {condition}")
    if not case.selected:
        raise ValueError("cannot build a condition for an excluded case")

    active = set(case.active_receipts)
    bindings = {key: set(value) for key, value in case.assertion_receipts.items()}
    cells = dict(case.belief_cells)
    relevant = _relevant_dependencies(case)
    if not relevant and condition not in ("clean", "irrelevant_update"):
        raise ValueError("condition requires at least one supported plan precondition")

    withdrawn: set[str] = set()
    superseded: set[str] = set()
    added_keys: set[str] = set()
    grounding = "controlled_assertion_update"

    if condition == "irrelevant_update":
        withdrawn.add(_json_sha256(["irrelevant", case.seed, case.corruption_seed]))
    elif condition == "alternative_support":
        dependency = relevant[0]
        old = sorted(dependency.required_receipts)[0]
        replacement = _json_sha256(["alternative", old, case.corruption_seed])
        withdrawn.add(old)
        active.discard(old)
        active.add(replacement)
        for assertion, receipts in bindings.items():
            if old in receipts:
                receipts.discard(old)
                receipts.add(replacement)
    elif condition in (
        "relevant_support_loss",
        "false_positive_correction",
        "compound_occlusion_correction",
        "sparse_update_burst",
    ):
        dependency = relevant[0]
        old = sorted(dependency.required_receipts)[0]
        withdrawn.add(old)
        active.discard(old)
        if condition in ("false_positive_correction", "compound_occlusion_correction"):
            superseded.add(old)
            grounding = "controlled_semantic_correction"
        for assertion in dependency.required_assertions:
            bindings.setdefault(assertion, set()).discard(old)
            if assertion.startswith("cell:") and "=" in assertion:
                coordinate, _ = assertion[5:].split("=", 1)
                x, y = (int(value) for value in coordinate.split(","))
                cells[(x, y)] = "wall"
        if condition == "compound_occlusion_correction":
            withdrawn.add(_json_sha256(["occluded", case.seed, case.corruption_seed]))
            grounding = "controlled_occlusion_withdrawal_plus_semantic_correction"
        if condition == "sparse_update_burst":
            withdrawn.add(_json_sha256(["irrelevant", case.seed, case.corruption_seed]))
            if len(relevant) > 1:
                added_keys.update(relevant[1].optimization_keys)
            grounding = "mixed_sparse_assertion_update"

    update = EvidenceUpdate(
        withdrawn_receipts=frozenset(withdrawn),
        added_keys=frozenset(added_keys),
        superseded_receipts=frozenset(superseded),
    )
    update_payload = {
        "withdrawn_receipts": sorted(update.withdrawn_receipts),
        "added_keys": sorted(update.added_keys),
        "superseded_receipts": sorted(update.superseded_receipts),
    }
    payload = {
        "seed": case.seed,
        "corruption_seed": case.corruption_seed,
        "rgb_receipts": list(case.rgb_receipts),
        "assertion_updates": [update_payload],
        "active_receipts": sorted(active),
        "assertion_receipts": {key: sorted(value) for key, value in sorted(bindings.items())},
        "belief_cells": [[list(position), label] for position, label in sorted(cells.items())],
        "pose": [list(case.pose[0]), case.pose[1]],
        "inventory": case.inventory,
        "plan_sha256": plan_sha256(case.plan),
        "planner": "common-counted-uniform-cost",
        "actions": ["left", "right", "forward", "pickup", "drop", "toggle"],
    }
    return DynamicCondition(
        condition=condition,
        update=update,
        active_receipts=frozenset(active),
        assertion_receipts={key: frozenset(value) for key, value in sorted(bindings.items())},
        belief_cells=tuple(sorted(cells.items())),
        access_payload=payload,
        grounding=grounding,
    )


def _oracle_affected(case: DynamicCase, condition: DynamicCondition) -> tuple[int, ...]:
    update = condition.update
    changed = update.withdrawn_receipts | update.superseded_receipts
    return tuple(
        dependency.step_index
        for dependency in case.dependencies
        if (
            dependency.required_receipts & changed
            or dependency.optimization_keys & update.added_keys
        )
    )


def _environment_replay(
    case: DynamicCase,
    plan: tuple[PlanStep, ...],
    *,
    authorized: bool,
) -> tuple[tuple[str, ...], bool]:
    """Replay public actions; synthetic unit fixtures have no environment trace."""

    if not case.prefix:
        return (), True
    from connectomequest.embodied.standard_benchmark import PublicDistractorTask

    env = PublicDistractorTask(case.seed)
    receipts = [env.observation().receipt_id]
    exact = receipts[0] == case.initial_receipt
    try:
        for item in case.prefix:
            observation = env.observation()
            exact = exact and observation.receipt_id == item["before"]
            observation = env.step(item["action"])
            receipts.append(observation.receipt_id)
            exact = exact and observation.receipt_id == item["after"]
        observation = env.observation()
        exact = exact and ((tuple(observation.position), int(observation.direction)) == case.pose)
        if authorized:
            for step in plan:
                exact = exact and (
                    (tuple(observation.position), int(observation.direction))
                    == (tuple(step.position), int(step.direction))
                )
                observation = env.step(step.action)
                receipts.append(observation.receipt_id)
    finally:
        env.close()
    return tuple(receipts), bool(exact)


def execute_case(
    case: DynamicCase,
    method: str,
    *,
    condition: str = "clean",
) -> EpisodeRow:
    """Execute one access-matched revalidation decision with deterministic replay."""

    from connectomequest.embodied.indexed_revalidation import (
        EvidenceState,
        RevalidationInput,
        revalidate,
    )

    condition_input = build_condition(case, condition)
    request = RevalidationInput(
        plan=case.plan,
        cursor=0,
        dependencies=case.dependencies,
        update=condition_input.update,
        evidence=EvidenceState(
            active_receipts=condition_input.active_receipts,
            assertion_receipts=condition_input.assertion_receipts,
        ),
        belief=_Belief(condition_input.belief_cells),
        pose=case.pose,
        goals=case.goals,
        has_key=bool(case.inventory and case.inventory.startswith("key:")),
        trusted_update=True,
        provenance_complete=method != "dependency_unaware",
        sealed_plan_sha256=plan_sha256(case.plan),
        oracle_affected_steps=(
            _oracle_affected(case, condition_input) if method == "dependency_oracle" else None
        ),
        key_label=case.inventory,
    )
    result = revalidate(method, request)
    replay = revalidate(method, request)
    environment_receipts, environment_exact = _environment_replay(
        case, result.selected_plan, authorized=result.authorized
    )
    exact_replay = replay == result and environment_exact

    trace_payload = {
        "authorized": result.authorized,
        "replan": result.replan,
        "affected_steps": list(result.affected_steps),
        "selected_plan_sha256": plan_sha256(result.selected_plan),
        "authorization_receipts": sorted(result.authorization_receipts),
    }
    update_payload = condition_input.access_payload["assertion_updates"][0]
    success = bool(result.authorized and not result.unsupported_next_action)
    work = result.work
    return EpisodeRow(
        seed=case.seed,
        corruption_seed=case.corruption_seed,
        condition=condition,
        method=method,
        access_sha256=access_signature(condition_input.access_payload),
        update_sha256=_json_sha256(update_payload),
        plan_sha256=plan_sha256(case.plan),
        authorized=result.authorized,
        replan=result.replan,
        affected_steps=result.affected_steps,
        unsupported_authorizations=int(result.unsupported_next_action),
        success=success,
        environment_actions=len(result.selected_plan) if result.authorized else 0,
        primitive_evaluations=work.primitive_evaluations,
        precondition_evaluations=work.precondition_evaluations,
        successor_evaluations=work.successor_evaluations,
        states_settled=work.states_settled,
        index_lookups=work.index_lookups,
        step_validations=work.step_validations,
        planner_calls=int(result.replan),
        authorization_receipts=tuple(sorted(result.authorization_receipts)),
        exact_replay=exact_replay,
        trace_sha256=_json_sha256(trace_payload),
        grounding=condition_input.grounding,
        environment_receipts=environment_receipts,
    )
