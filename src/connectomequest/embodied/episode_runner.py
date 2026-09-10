"""Complete access-matched V39 execution and exact environment replay."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import perf_counter_ns

from connectomequest.embodied.dynamic_task import FrozenPerceptionCache
from connectomequest.embodied.environment import (
    GeneratedLayout,
    LayoutSpec,
    classify_remaining_length,
    reset_layout,
)
from connectomequest.embodied.evidence_index import PlanDependency
from connectomequest.embodied.navigation import DIRS, PlanStep
from connectomequest.embodied.perturbations import view_to_world
from connectomequest.embodied.revalidation import (
    FullReplan,
    FullScan,
    PlannedSuffix,
    ReceiptIndex,
    UncheckedReuse,
)
from connectomequest.embodied.search import route_counted
from connectomequest.embodied.symbolic_controller import passable
from connectomequest.embodied.task_planner import plan_mission
from connectomequest.embodied.types import (
    EpisodeCounters,
    EpisodeProtocol,
    EpisodeResult,
    EvidenceEvent,
    MethodInput,
    canonical_json_bytes,
    digest_public_access,
)
from connectomequest.embodied.updates import (
    CorrectionBinding,
    SchedulePrefix,
    UpdateSchedule,
    build_update_schedule,
)

_METHODS = {
    "full_replan": FullReplan,
    "full_scan": FullScan,
    "receipt_index": ReceiptIndex,
    "unchecked_reuse": UncheckedReuse,
}


@dataclass(frozen=True)
class PreparedEpisode:
    layout: GeneratedLayout
    update_seed: int
    prefix_actions: tuple[str, ...]
    prefix_observation_hashes: tuple[str, ...]
    plan: tuple[PlanStep, ...]
    dependencies: tuple[PlanDependency, ...]
    active_receipt_ids: tuple[str, ...]
    assertion_to_receipt_bindings: tuple[tuple[str, tuple[str, ...]], ...]
    correction_bindings: tuple[CorrectionBinding, ...]
    schedule: UpdateSchedule
    pre_update_remaining_plan_length: int
    exclusion_reason: str | None


@dataclass(frozen=True)
class ReplayResult:
    exact: bool
    compared_observations: int
    first_mismatch: int | None


class _Belief:
    def __init__(self, cells: Mapping[tuple[int, int], str]):
        self.cells = dict(cells)

    def label(self, position):
        return self.cells.get(tuple(position))


class _Tracker:
    def __init__(self, perceptor: FrozenPerceptionCache):
        self.perceptor = perceptor
        self.cells: dict[tuple[int, int], str] = {}
        self.support: dict[tuple[int, int], str] = {}
        self.bindings: dict[str, set[str]] = {}
        self.receipt_assertions: dict[str, tuple[str, ...]] = {}
        self.inventory: str | None = None
        self.pending: tuple[str, str | None, str, tuple[int, int]] | None = None
        self.door_colors: dict[tuple[int, int], str] = {}
        self.disqualified_key_positions: set[tuple[int, int]] = set()

    @staticmethod
    def _assertion(position: tuple[int, int], label: str) -> str:
        return f"cell:{position[0]},{position[1]}={label}"

    def update(self, observation) -> tuple[str, ...]:
        previous_inventory = self.inventory
        if observation.carrying is None:
            if (
                observation.last_action == "drop"
                and observation.acknowledged
                and self.pending is not None
                and previous_inventory
                and not previous_inventory.startswith("key:")
            ):
                self.disqualified_key_positions.add(self.pending[3])
            self.inventory = None
        elif (
            observation.last_action == "pickup"
            and observation.acknowledged
            and self.pending is not None
        ):
            label = self.pending[1]
            if label and label.startswith(f"{observation.carrying}:"):
                self.inventory = label
            else:
                self.inventory = f"{observation.carrying}:unknown"
                if observation.carrying != "key":
                    self.disqualified_key_positions.add(self.pending[3])
        elif self.inventory is None or not self.inventory.startswith(f"{observation.carrying}:"):
            self.inventory = f"{observation.carrying}:unknown"

        assertions = []
        for x, y, label, confidence in self.perceptor.predict(observation.rgb):
            if (x, y) == (3, 6) or label == "unseen" or confidence < 0.45:
                continue
            position = view_to_world(
                observation.position,
                observation.direction,
                x,
                y,
            )
            if label.startswith("door_") and ":" in label:
                self.door_colors[position] = label.split(":", 1)[1]
            self.cells[position] = label
            self.support[position] = observation.receipt_id
            assertion = self._assertion(position, label)
            self.bindings.setdefault(assertion, set()).add(observation.receipt_id)
            assertions.append(assertion)

        position = tuple(observation.position)
        current_label = (
            f"door_open:{self.door_colors[position]}" if position in self.door_colors else "empty"
        )
        self.cells[position] = current_label
        self.support[position] = observation.receipt_id
        assertion = self._assertion(position, current_label)
        self.bindings.setdefault(assertion, set()).add(observation.receipt_id)
        assertions.append(assertion)
        unique = tuple(sorted(set(assertions)))
        self.receipt_assertions[observation.receipt_id] = unique
        return unique

    def commit(self, action: str, observation) -> None:
        front = (
            observation.position[0] + DIRS[observation.direction][0],
            observation.position[1] + DIRS[observation.direction][1],
        )
        self.pending = (
            action,
            self.cells.get(front),
            observation.receipt_id,
            front,
        )


def _digest(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pre_divergence_access_digest(access_digests: tuple[str, ...]) -> str:
    """Hash only the common method input immediately before divergence."""
    return _digest(tuple(access_digests[:1]))


def _endpoint(
    start_position: tuple[int, int],
    start_direction: int,
    plan: tuple[PlanStep, ...],
) -> tuple[tuple[int, int], int]:
    position = tuple(start_position)
    direction = int(start_direction)
    for step in plan:
        if step.action == "left":
            direction = (direction - 1) % 4
        elif step.action == "right":
            direction = (direction + 1) % 4
        elif step.action == "forward":
            position = (
                position[0] + DIRS[direction][0],
                position[1] + DIRS[direction][1],
            )
    return position, direction


def _route_to_objects(
    tracker: _Tracker,
    observation,
    positions: tuple[tuple[int, int], ...],
    terminal: str,
) -> tuple[PlanStep, ...] | None:
    """Plan to observed objects using only the public semantic belief."""

    goals = {
        ((position[0] - dx, position[1] - dy), direction)
        for position in positions
        for direction, (dx, dy) in enumerate(DIRS)
        if passable(
            tracker.cells.get(
                (position[0] - dx, position[1] - dy),
                "unknown",
            )
        )
    }
    if not goals:
        return None
    result = route_counted(
        _Belief(tracker.cells),
        tuple(observation.position),
        int(observation.direction),
        goals=goals,
        has_key=bool(tracker.inventory and tracker.inventory.startswith("key:")),
        terminal=terminal,
        key_label=tracker.inventory,
    )
    return result.plan


def _exploration_action(
    tracker: _Tracker,
    observation,
    scanned: set[tuple[tuple[int, int], int]],
) -> str:
    """Choose one deterministic action from the acquired RGB-only belief.

    This method-independent enrollment policy never receives the environment,
    a latent grid, a target location, or a reference trajectory.
    """

    locked = {
        position: label.split(":", 1)[1]
        for position, label in tracker.cells.items()
        if label.startswith("door_locked:") and label.split(":", 1)[1] != "unknown"
    }
    if tracker.inventory and tracker.inventory.startswith("key:"):
        key_color = tracker.inventory.split(":", 1)[1]
        doors = tuple(sorted(position for position, color in locked.items() if color == key_color))
        route = _route_to_objects(tracker, observation, doors, "toggle")
        if route:
            return route[0].action

    closed = tuple(
        sorted(
            position
            for position, label in tracker.cells.items()
            if label.startswith("door_closed:")
        )
    )
    route = _route_to_objects(tracker, observation, closed, "toggle")
    if route:
        return route[0].action

    if (
        tracker.inventory
        and (
            not tracker.inventory.startswith("key:")
            or not any(color == tracker.inventory.split(":", 1)[1] for color in locked.values())
        )
        and not closed
    ):
        door_positions = tuple(
            position for position, label in tracker.cells.items() if label.startswith("door_")
        )
        empty_positions = [
            position
            for position, label in tracker.cells.items()
            if label == "empty" and position != tuple(observation.position)
        ]
        empty_positions.sort(
            key=lambda position: (
                -min(
                    (
                        abs(position[0] - door[0]) + abs(position[1] - door[1])
                        for door in door_positions
                    ),
                    default=0,
                ),
                position,
            )
        )
        route = _route_to_objects(tracker, observation, tuple(empty_positions[:1]), "drop")
        if route:
            return route[0].action

    if tracker.inventory is None and locked:
        needed_colors = set(locked.values())
        keys = tuple(
            sorted(
                position
                for position, label in tracker.cells.items()
                if label.startswith("key:")
                and label.split(":", 1)[1] in needed_colors
                and position not in tracker.disqualified_key_positions
            )
        )
        route = _route_to_objects(tracker, observation, keys, "pickup")
        if route:
            return route[0].action

    visited_positions = {position for position, _ in scanned}
    position_goals = {
        (position, direction)
        for position, label in tracker.cells.items()
        if passable(label)
        for direction in range(4)
        if position not in visited_positions
    }
    belief = _Belief(tracker.cells)
    route = None
    for goals in (
        position_goals,
        {
            (position, direction)
            for position, label in tracker.cells.items()
            if passable(label)
            for direction in range(4)
            if (position, direction) not in scanned
        },
    ):
        route = route_counted(
            belief,
            tuple(observation.position),
            int(observation.direction),
            goals=goals,
            has_key=bool(tracker.inventory and tracker.inventory.startswith("key:")),
            key_label=tracker.inventory,
            allow_toggle=False,
        ).plan
        if route:
            break
    return route[0].action if route else None


def _candidate_stage(
    tracker: _Tracker,
    observation,
    requested_stratum: str,
):
    target_label = f"box:{observation.goals[observation.stage]}"
    targets = [position for position, label in tracker.cells.items() if label == target_label]
    if not targets:
        return None
    target = targets[0]
    candidates = [
        position
        for position, label in tracker.cells.items()
        if passable(label) and position != tuple(observation.position)
    ]
    candidates.sort(
        key=lambda position: (
            -abs(position[0] - target[0]) - abs(position[1] - target[1]),
            position,
        )
    )
    belief = _Belief(tracker.cells)
    for candidate in candidates[:1]:
        for desired_direction in range(4):
            route = route_counted(
                belief,
                tuple(observation.position),
                int(observation.direction),
                goals={(candidate, desired_direction)},
                has_key=bool(tracker.inventory and tracker.inventory.startswith("key:")),
                key_label=tracker.inventory,
                allow_toggle=False,
            )
            if route.plan is None:
                continue
            endpoint = _endpoint(
                tuple(observation.position),
                int(observation.direction),
                route.plan,
            )
            mission = plan_mission(
                tracker.cells,
                tracker.support,
                endpoint[0],
                endpoint[1],
                tracker.inventory,
                observation.mission,
                max_states=250_000,
            )
            if (
                mission.plan is not None
                and mission.dependencies[0].required_assertions
                and classify_remaining_length(len(mission.plan)) == requested_stratum
            ):
                return route.plan
    return None


def _bindings_for_dependencies(
    tracker: _Tracker,
    dependencies: tuple[PlanDependency, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    required = set().union(*(dependency.required_assertions for dependency in dependencies))
    return tuple(
        (assertion, tuple(sorted(tracker.bindings.get(assertion, set()))))
        for assertion in sorted(required)
    )


def _correction_bindings_for_dependencies(
    tracker: _Tracker,
    dependencies: tuple[PlanDependency, ...],
) -> tuple[CorrectionBinding, ...]:
    """Return only conflicting labels supported by distinct acquired receipts."""

    rows = set()
    for dependency in dependencies:
        for old_assertion in dependency.required_assertions:
            semantic_key, separator, _ = old_assertion.rpartition("=")
            if not separator:
                continue
            old_receipts = set(tracker.bindings.get(old_assertion, ()))
            for new_assertion, new_receipts in tracker.bindings.items():
                if new_assertion == old_assertion or not new_assertion.startswith(
                    f"{semantic_key}="
                ):
                    continue
                for receipt_id in set(new_receipts) - old_receipts:
                    rows.add(
                        CorrectionBinding(
                            semantic_key,
                            old_assertion,
                            new_assertion,
                            receipt_id,
                        )
                    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.semantic_key,
                row.old_assertion,
                row.new_assertion,
                row.receipt_id,
            ),
        )
    )


def _excluded(layout: GeneratedLayout, reason: str) -> PreparedEpisode:
    return PreparedEpisode(
        layout,
        0,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        UpdateSchedule((), (), None, 0.0),
        0,
        reason,
    )


def prepare_episode(
    layout: GeneratedLayout,
    perception_cache: FrozenPerceptionCache,
    *,
    update_seed: int,
    condition: str,
    update_count: int,
    relevant_fraction: float,
) -> PreparedEpisode:
    """Enroll using only a common RGB trace and the common symbolic planner."""

    if layout.exclusion_reason is not None:
        return _excluded(layout, layout.exclusion_reason)
    env = reset_layout(layout.spec)
    tracker = _Tracker(perception_cache)
    scanned: set[tuple[tuple[int, int], int]] = set()
    prefix_actions: list[str] = []
    prefix_hashes = [env.observation().receipt_id]
    try:
        while not env.observation().terminated and env.observation().step < layout.spec.horizon:
            observation = env.observation()
            tracker.update(observation)
            scanned.add((tuple(observation.position), int(observation.direction)))
            target_label = f"box:{observation.goals[observation.stage]}"
            if target_label in set(tracker.cells.values()):
                route = _candidate_stage(
                    tracker,
                    observation,
                    layout.requested_stratum,
                )
                if route is not None:
                    for step in route:
                        current = env.observation()
                        tracker.commit(step.action, current)
                        after = env.step(step.action)
                        prefix_actions.append(step.action)
                        prefix_hashes.append(after.receipt_id)
                        tracker.update(after)
                        if after.terminated:
                            return _excluded(layout, "staging_terminated_episode")
                    observation = env.observation()
                    mission = plan_mission(
                        tracker.cells,
                        tracker.support,
                        tuple(observation.position),
                        int(observation.direction),
                        tracker.inventory,
                        observation.mission,
                    )
                    if (
                        mission.plan is None
                        or classify_remaining_length(len(mission.plan)) != layout.requested_stratum
                    ):
                        return _excluded(layout, "length_outside_stratum")
                    bindings = _bindings_for_dependencies(tracker, mission.dependencies)
                    active = tuple(
                        sorted(
                            receipt
                            for receipts in tracker.bindings.values()
                            for receipt in receipts
                        )
                    )
                    corrections = _correction_bindings_for_dependencies(
                        tracker, mission.dependencies
                    )
                    schedule_prefix = SchedulePrefix(
                        action_count=len(prefix_actions),
                        active_receipt_ids=active,
                        assertion_to_receipt_bindings=bindings,
                        remaining_dependencies=mission.dependencies,
                        correction_bindings=corrections,
                    )
                    schedule = build_update_schedule(
                        layout,
                        schedule_prefix,
                        update_seed,
                        condition,
                        update_count,
                        relevant_fraction,
                    )
                    if schedule.exclusion_reason is not None:
                        return _excluded(layout, schedule.exclusion_reason)
                    return PreparedEpisode(
                        layout,
                        int(update_seed),
                        tuple(prefix_actions),
                        tuple(prefix_hashes),
                        mission.plan,
                        mission.dependencies,
                        active,
                        bindings,
                        corrections,
                        schedule,
                        len(mission.plan),
                        None,
                    )

            action = _exploration_action(tracker, observation, scanned)
            if action is None:
                return _excluded(layout, "no_public_full_mission_plan")
            tracker.commit(action, observation)
            after = env.step(action)
            prefix_actions.append(action)
            prefix_hashes.append(after.receipt_id)
        return _excluded(layout, "no_public_full_mission_plan")
    except (AssertionError, RuntimeError, ValueError):
        return _excluded(layout, "instrumentation_failure")
    finally:
        env.close()


def reschedule_episode(
    prepared: PreparedEpisode,
    *,
    update_seed: int,
    condition: str,
    update_count: int,
    relevant_fraction: float,
) -> PreparedEpisode:
    """Reuse one method-independent prefix under another prospective schedule."""

    if prepared.exclusion_reason is not None:
        return replace(prepared, update_seed=int(update_seed))
    prefix = SchedulePrefix(
        action_count=len(prepared.prefix_actions),
        active_receipt_ids=prepared.active_receipt_ids,
        assertion_to_receipt_bindings=prepared.assertion_to_receipt_bindings,
        remaining_dependencies=prepared.dependencies,
        correction_bindings=prepared.correction_bindings,
    )
    schedule = build_update_schedule(
        prepared.layout,
        prefix,
        update_seed,
        condition,
        update_count,
        relevant_fraction,
    )
    return replace(
        prepared,
        update_seed=int(update_seed),
        schedule=schedule,
        exclusion_reason=schedule.exclusion_reason,
    )


def _parse_cells(snapshot) -> dict[tuple[int, int], str]:
    cells = {}
    for key, label in snapshot:
        x, y = (int(value) for value in str(key).split(",", 1))
        cells[(x, y)] = str(label)
    return cells


def _support_from_input(method_input: MethodInput) -> dict[tuple[int, int], str]:
    active = set(method_input.active_receipt_ids)
    bindings = dict(method_input.assertion_to_receipt_bindings)
    support = {}
    for key, label in method_input.belief_snapshot:
        assertion = f"cell:{key}={label}"
        live = sorted(active & set(bindings.get(assertion, ())))
        if live:
            x, y = (int(value) for value in key.split(",", 1))
            support[(x, y)] = live[-1]
    return support


class _EpisodePlanner:
    tie_break_id = "action-order-v1"

    def __call__(self, method_input: MethodInput) -> PlannedSuffix:
        cells = _parse_cells(method_input.belief_snapshot)
        support = _support_from_input(method_input)
        outcome = plan_mission(
            cells,
            support,
            method_input.pose,
            method_input.direction,
            method_input.inventory,
            method_input.mission,
        )
        if outcome.plan is None:
            return PlannedSuffix((), (), outcome.work)
        return PlannedSuffix(outcome.plan, outcome.dependencies, outcome.work)


def _renumber(
    dependencies: tuple[PlanDependency, ...],
) -> tuple[PlanDependency, ...]:
    return tuple(
        replace(dependency, step_index=index) for index, dependency in enumerate(dependencies)
    )


def _method_input(
    tracker: _Tracker,
    active: set[str],
    bindings: Mapping[str, set[str]],
    plan: tuple[PlanStep, ...],
    event: EvidenceEvent,
) -> MethodInput:
    return MethodInput(
        pose=tuple(tracker.current_position),
        direction=int(tracker.current_direction),
        inventory=tracker.inventory,
        mission=tracker.current_mission,
        belief_snapshot=tuple(
            (f"{position[0]},{position[1]}", label)
            for position, label in sorted(tracker.cells.items())
        ),
        active_receipt_ids=tuple(sorted(active)),
        assertion_to_receipt_bindings=tuple(
            (assertion, tuple(sorted(receipts))) for assertion, receipts in sorted(bindings.items())
        ),
        remaining_plan=plan,
        cursor=0,
        update=event,
        public_action_set=("left", "right", "forward", "toggle", "pickup", "drop"),
    )


def _set_tracker_public_state(tracker: _Tracker, observation) -> None:
    tracker.current_position = tuple(observation.position)
    tracker.current_direction = int(observation.direction)
    tracker.current_mission = observation.mission


def _apply_event(
    event: EvidenceEvent,
    tracker: _Tracker,
    active: set[str],
    bindings: dict[str, set[str]],
    permanently_withdrawn: set[str],
) -> None:
    removed = set(event.withdrawn_receipt_ids) | set(event.superseded_receipt_ids)
    permanently_withdrawn.update(removed)
    active.difference_update(removed)
    for receipts in bindings.values():
        receipts.difference_update(removed)
    for assertion, receipts in event.added_assertion_bindings:
        bindings.setdefault(assertion, set()).update(set(receipts) & active)
        match = re.fullmatch(r"cell:(-?\d+),(-?\d+)=(.+)", assertion)
        if match:
            tracker.cells[(int(match.group(1)), int(match.group(2)))] = match.group(3)


def _next_input(
    tracker: _Tracker,
    active: set[str],
    bindings: dict[str, set[str]],
    plan: tuple[PlanStep, ...],
    action_index: int,
) -> MethodInput:
    return _method_input(
        tracker,
        active,
        bindings,
        plan,
        EvidenceEvent(action_index, (), (), (), (), "none"),
    )


def _supported(
    dependency: PlanDependency | None,
    active: set[str],
    bindings: Mapping[str, set[str]],
) -> bool:
    if dependency is None:
        return True
    if dependency.required_assertions:
        return all(
            bool(set(bindings.get(assertion, set())) & active)
            for assertion in dependency.required_assertions
        )
    return dependency.required_receipts <= active


def run_episode(
    protocol: EpisodeProtocol,
    prepared: PreparedEpisode,
    method: str,
    perception_cache: FrozenPerceptionCache,
) -> EpisodeResult:
    if prepared.exclusion_reason is not None:
        raise ValueError(f"cannot run excluded episode: {prepared.exclusion_reason}")
    if method not in _METHODS or method not in protocol.methods:
        raise ValueError(f"method not in protocol: {method}")
    perception_cache.verify_checkpoint(protocol.perception_sha256)
    env = reset_layout(prepared.layout.spec)
    tracker = _Tracker(perception_cache)
    actions: list[str] = []
    observations = [env.observation().receipt_id]
    wall_started = perf_counter_ns()
    try:
        for expected_receipt, action in zip(
            prepared.prefix_observation_hashes,
            prepared.prefix_actions,
        ):
            observation = env.observation()
            if observation.receipt_id != expected_receipt:
                raise AssertionError("prefix observation mismatch")
            _set_tracker_public_state(tracker, observation)
            tracker.update(observation)
            tracker.commit(action, observation)
            after = env.step(action)
            actions.append(action)
            observations.append(after.receipt_id)
        if tuple(observations) != prepared.prefix_observation_hashes:
            raise AssertionError("prefix trace mismatch")

        observation = env.observation()
        _set_tracker_public_state(tracker, observation)
        tracker.update(observation)
        active = set(prepared.active_receipt_ids)
        bindings = {
            assertion: set(receipts)
            for assertion, receipts in prepared.assertion_to_receipt_bindings
        }
        permanently_withdrawn: set[str] = set()
        plan = tuple(prepared.plan)
        dependencies = tuple(prepared.dependencies)
        planner = _EpisodePlanner()
        strategy = _METHODS[method](planner)
        counters = EpisodeCounters(environment_actions=len(actions))
        unsupported = 0
        illegal = 0
        update_digests = []
        access_digests = []
        plan_digests = [_digest(plan)]
        exposed = 0
        milestones = {
            "key_acquired": bool(tracker.inventory and tracker.inventory.startswith("key:")),
            "door_opened": any(label.startswith("door_open:") for label in tracker.cells.values()),
            "target_picked_up": False,
        }
        events = {event.action_index: event for event in prepared.schedule.events}

        while not env.observation().terminated and len(actions) < protocol.horizon:
            observation = env.observation()
            _set_tracker_public_state(tracker, observation)
            assertions = tracker.update(observation)
            if observation.receipt_id not in permanently_withdrawn:
                active.add(observation.receipt_id)
                for assertion in assertions:
                    bindings.setdefault(assertion, set()).add(observation.receipt_id)

            if not plan:
                planner_input = _next_input(tracker, active, bindings, (), len(actions))
                started = perf_counter_ns()
                outcome = planner(planner_input)
                planning_ns = perf_counter_ns() - started
                plan = outcome.plan
                dependencies = outcome.dependencies
                counters += EpisodeCounters(
                    predicate_evaluations=outcome.work.precondition_evaluations,
                    successor_expansions=outcome.work.successor_evaluations,
                    planner_calls=1,
                    planning_ns=planning_ns,
                )
                plan_digests.append(_digest(plan))

            event = events.get(len(actions))
            decision = None
            if event is not None:
                _apply_event(
                    event,
                    tracker,
                    active,
                    bindings,
                    permanently_withdrawn,
                )
                method_input = _method_input(
                    tracker,
                    active,
                    bindings,
                    plan,
                    event,
                )
                access_digests.append(digest_public_access(method_input))
                update_digests.append(_digest(event))
                exposed += 1
                strategy.initialize(plan, dependencies)
                decision = strategy.handle_update(method_input)
                counters += decision.counters
                plan = decision.remaining_plan
                dependencies = decision.dependencies
                unsupported += int(decision.unsupported_authorization)
                plan_digests.append(_digest(plan))

            if not plan:
                break
            dependency = dependencies[0] if dependencies else None
            if decision is None and not _supported(dependency, active, bindings):
                unsupported += 1
            action = plan[0].action
            tracker.commit(action, observation)
            after = env.step(action)
            actions.append(action)
            observations.append(after.receipt_id)
            counters += EpisodeCounters(environment_actions=1)
            illegal += int(not after.acknowledged)
            milestones["key_acquired"] = milestones["key_acquired"] or bool(after.carrying == "key")
            milestones["door_opened"] = milestones["door_opened"] or bool(
                action == "toggle" and after.acknowledged
            )
            milestones["target_picked_up"] = milestones["target_picked_up"] or bool(
                after.reward > 0
            )
            plan = tuple(plan[1:])
            dependencies = _renumber(tuple(dependencies[1:]))

        final = env.observation()
        elapsed = perf_counter_ns() - wall_started
        replay_probe = EpisodeResult(
            protocol_version=protocol.protocol_version,
            stage=protocol.stage,
            method=method,
            layout_seed=prepared.layout.spec.layout_seed,
            environment_seed=prepared.layout.spec.environment_seed,
            room_size=prepared.layout.spec.room_size,
            distractor_count=prepared.layout.spec.distractor_count,
            horizon=prepared.layout.spec.horizon,
            update_seed=prepared.update_seed,
            condition=(
                prepared.schedule.events[0].condition if prepared.schedule.events else "none"
            ),
            update_count=len(prepared.schedule.events),
            relevant_fraction=prepared.schedule.realized_relevant_fraction,
            plan_length_stratum=prepared.layout.requested_stratum,
            pre_update_remaining_plan_length=prepared.pre_update_remaining_plan_length,
            task_success=bool(final.reward > 0),
            terminated=bool(final.terminated and final.reward > 0),
            truncated=bool(final.terminated and final.reward <= 0),
            exposed_update_count=exposed,
            milestones=tuple(sorted(milestones.items())),
            unsupported_authorizations=unsupported,
            illegal_environment_actions=illegal,
            counters=counters,
            index_build_ns=strategy.index_build_ns,
            index_peak_bytes=strategy.index_peak_bytes,
            total_reasoning_ns=(
                counters.revalidation_ns + counters.planning_ns + strategy.index_build_ns
            ),
            total_wall_clock_ns=elapsed,
            trace_replay_ok=False,
            access_digest=_pre_divergence_access_digest(tuple(access_digests)),
            pre_update_action_digest=_digest(prepared.prefix_actions),
            action_trace=tuple(actions),
            observation_hashes=tuple(observations),
            update_digests=tuple(update_digests),
            plan_digests=tuple(plan_digests),
        )
    finally:
        env.close()
    replay = replay_episode(replay_probe)
    return replace(replay_probe, trace_replay_ok=replay.exact)


def replay_episode(result: EpisodeResult) -> ReplayResult:
    spec = LayoutSpec(
        result.layout_seed,
        result.environment_seed,
        result.room_size,
        result.distractor_count,
        result.horizon,
    )
    env = reset_layout(spec)
    observed = [env.observation().receipt_id]
    mismatch = None
    try:
        for index, action in enumerate(result.action_trace, start=1):
            if env.observation().terminated:
                mismatch = index
                break
            observed.append(env.step(action).receipt_id)
            if (
                index >= len(result.observation_hashes)
                or observed[-1] != result.observation_hashes[index]
            ):
                mismatch = index
                break
    finally:
        env.close()
    exact = (
        mismatch is None
        and tuple(observed) == result.observation_hashes
        and len(observed) == len(result.action_trace) + 1
    )
    return ReplayResult(exact, len(observed), mismatch)
