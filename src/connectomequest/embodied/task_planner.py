"""Deterministic full-mission planning over a public semantic belief map."""

from __future__ import annotations

import heapq
import re
from collections.abc import Mapping
from dataclasses import dataclass

from connectomequest.embodied.evidence_index import PlanDependency
from connectomequest.embodied.navigation import DIRS, PlanStep
from connectomequest.embodied.search import WorkCounter, route_counted

_PASSABLE = frozenset({"empty", "floor", "goal"})
_MOVABLE = frozenset({"key", "box", "ball"})
_ACTION_ORDER = ("left", "right", "forward", "toggle", "pickup", "drop")


class _CellBelief:
    def __init__(self, cells):
        self.cells = cells

    def label(self, position):
        return self.cells.get(tuple(position))


@dataclass(frozen=True)
class MissionPlan:
    plan: tuple[PlanStep, ...] | None
    dependencies: tuple[PlanDependency, ...]
    work: WorkCounter


def _target_from_mission(mission: str) -> str:
    match = re.fullmatch(
        r"pick up the (red|green|blue|purple|yellow|grey) (box|ball|key)",
        mission,
    )
    if not match:
        raise ValueError(f"unsupported mission: {mission}")
    return f"{match.group(2)}:{match.group(1)}"


def _passable(label: str | None) -> bool:
    return bool(label in _PASSABLE or (label and label.startswith("door_open:")))


def _object(label: str | None) -> bool:
    return bool(label and label.split(":", 1)[0] in _MOVABLE)


def _state_label(
    cells: Mapping[tuple[int, int], str],
    position: tuple[int, int],
    opened: frozenset[tuple[int, int]],
    removed: frozenset[tuple[int, int]],
    placed: tuple[tuple[tuple[int, int], str], ...],
) -> str | None:
    if position in opened:
        base = cells.get(position, "door_open:unknown")
        color = base.split(":", 1)[1] if ":" in base else "unknown"
        return f"door_open:{color}"
    placed_map = dict(placed)
    if position in placed_map:
        return placed_map[position]
    if position in removed:
        return "empty"
    return cells.get(position)


def _dependency(
    index: int,
    step: PlanStep,
    support: Mapping[tuple[int, int], str],
) -> PlanDependency | None:
    receipts = set()
    assertions = set()
    keys = set()
    for raw_position, label in step.requirements:
        position = tuple(raw_position)
        receipt = support.get(position)
        if receipt is None:
            return None
        receipts.add(str(receipt))
        assertions.add(f"cell:{position[0]},{position[1]}={label}")
        keys.add(f"cell:{position[0]},{position[1]}")
    return PlanDependency(
        index,
        frozenset(receipts),
        frozenset(keys),
        frozenset(assertions),
    )


def plan_mission(
    cells: Mapping[tuple[int, int], str],
    support: Mapping[tuple[int, int], str],
    position: tuple[int, int],
    direction: int,
    inventory: str | None,
    mission: str,
    max_states: int = 250_000,
) -> MissionPlan:
    """Return the fixed-order least-action plan using only the supplied belief."""

    target = _target_from_mission(mission)
    if target not in set(cells.values()):
        return MissionPlan(None, (), WorkCounter())
    target_positions = tuple(
        cell_position for cell_position, label in cells.items() if label == target
    )
    if inventory is None:
        direct_goals = {
            (
                (
                    target_position[0] - dx,
                    target_position[1] - dy,
                ),
                heading,
            )
            for target_position in target_positions
            for heading, (dx, dy) in enumerate(DIRS)
            if _passable(
                cells.get(
                    (
                        target_position[0] - dx,
                        target_position[1] - dy,
                    )
                )
            )
        }
        direct = route_counted(
            _CellBelief(cells),
            tuple(position),
            int(direction),
            goals=direct_goals,
            terminal="pickup",
        )
        if direct.plan is not None:
            direct_dependencies = tuple(
                _dependency(index, step, support) for index, step in enumerate(direct.plan)
            )
            if all(row is not None for row in direct_dependencies):
                return MissionPlan(
                    direct.plan,
                    tuple(direct_dependencies),
                    direct.work,
                )

    def lower_bound(current_position, carried_label):
        if carried_label == target:
            return 0
        return min(
            abs(current_position[0] - target_position[0])
            + abs(current_position[1] - target_position[1])
            for target_position in target_positions
        )

    locked_positions = {
        position
        for position, label in cells.items()
        if label.startswith("door_locked:") and label.split(":", 1)[1] != "unknown"
    }
    locked_key_labels = {f"key:{cells[position].split(':', 1)[1]}" for position in locked_positions}
    start = (
        tuple(position),
        int(direction),
        inventory,
        frozenset(),
        frozenset(),
        tuple(),
    )
    heap = [(lower_bound(position, inventory), 0, 0, start, tuple())]
    settled_states = set()
    serial = 0
    successor_evaluations = 0
    precondition_evaluations = 0

    while heap and len(settled_states) < max_states:
        _, distance, _, state, path = heapq.heappop(heap)
        if state in settled_states:
            continue
        settled_states.add(state)
        pos, heading, carried, opened, removed, placed = state
        if carried == target:
            dependencies = []
            for index, step in enumerate(path):
                row = _dependency(index, step, support)
                if row is None:
                    return MissionPlan(
                        None,
                        (),
                        WorkCounter(
                            precondition_evaluations,
                            successor_evaluations,
                            len(settled_states),
                        ),
                    )
                dependencies.append(row)
            return MissionPlan(
                path,
                tuple(dependencies),
                WorkCounter(
                    precondition_evaluations,
                    successor_evaluations,
                    len(settled_states),
                ),
            )

        front = (
            pos[0] + DIRS[heading][0],
            pos[1] + DIRS[heading][1],
        )
        front_label = _state_label(cells, front, opened, removed, placed)
        base_label = cells.get(front)
        precondition_evaluations += 1
        candidates = [
            ("left", (pos, (heading - 1) % 4, carried, opened, removed, placed)),
            ("right", (pos, (heading + 1) % 4, carried, opened, removed, placed)),
        ]
        if _passable(front_label):
            candidates.append(("forward", (front, heading, carried, opened, removed, placed)))
        if front_label and front_label.startswith("door_closed:"):
            candidates.append(
                ("toggle", (pos, heading, carried, opened | {front}, removed, placed))
            )
        elif front_label and front_label.startswith("door_locked:"):
            color = front_label.split(":", 1)[1]
            if carried == f"key:{color}":
                candidates.append(
                    (
                        "toggle",
                        (pos, heading, carried, opened | {front}, removed, placed),
                    )
                )
        useful_pickup = front_label == target or front_label in locked_key_labels
        pickup_is_acyclic = front not in dict(placed) or front_label == target
        if carried is None and useful_pickup and pickup_is_acyclic:
            placed_map = dict(placed)
            if front in placed_map:
                del placed_map[front]
                next_removed = removed
            else:
                next_removed = removed | {front}
            candidates.append(
                (
                    "pickup",
                    (
                        pos,
                        heading,
                        front_label,
                        opened,
                        next_removed,
                        tuple(sorted(placed_map.items())),
                    ),
                )
            )
        doors_ready = locked_positions <= set(opened)
        if (
            carried is not None
            and carried != target
            and doors_ready
            and front_label in ("empty", "floor")
        ):
            placed_map = dict(placed)
            placed_map[front] = carried
            candidates.append(
                (
                    "drop",
                    (
                        pos,
                        heading,
                        None,
                        opened,
                        removed,
                        tuple(sorted(placed_map.items())),
                    ),
                )
            )

        order = {action: index for index, action in enumerate(_ACTION_ORDER)}
        candidates.sort(key=lambda row: order[row[0]])
        successor_evaluations += len(candidates)
        for action, next_state in candidates:
            if next_state in settled_states:
                continue
            requirements = ()
            if action not in ("left", "right"):
                if base_label is not None and front not in removed:
                    requirements = ((front, base_label),)
            step = PlanStep(action, pos, heading, requirements)
            serial += 1
            heapq.heappush(
                heap,
                (
                    distance + 1 + lower_bound(next_state[0], next_state[2]),
                    distance + 1,
                    serial,
                    next_state,
                    path + (step,),
                ),
            )

    return MissionPlan(
        None,
        (),
        WorkCounter(
            precondition_evaluations,
            successor_evaluations,
            len(settled_states),
        ),
    )
