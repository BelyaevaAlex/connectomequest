"""Legacy-equivalent symbolic planning with deterministic work counters."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from connectomequest.embodied.navigation import DIRS, PASSABLE, PlanStep


def _passable_label(label: str | None) -> bool:
    return label in PASSABLE or bool(label and label.startswith("door_open:"))


def _closed_label(label: str | None) -> bool:
    return label == "door_closed" or bool(label and label.startswith("door_closed:"))


def _can_toggle_locked(label: str | None, has_key: bool, key_label: str | None) -> bool:
    if label == "door_locked":
        return has_key
    if not label or not label.startswith("door_locked:"):
        return False
    door_color = label.split(":", 1)[1]
    return door_color != "unknown" and key_label == f"key:{door_color}"


@dataclass(frozen=True)
class WorkCounter:
    precondition_evaluations: int = 0
    successor_evaluations: int = 0
    states_settled: int = 0
    index_lookups: int = 0
    step_validations: int = 0

    def __add__(self, other: WorkCounter) -> WorkCounter:
        return WorkCounter(
            self.precondition_evaluations + other.precondition_evaluations,
            self.successor_evaluations + other.successor_evaluations,
            self.states_settled + other.states_settled,
            self.index_lookups + other.index_lookups,
            self.step_validations + other.step_validations,
        )

    @property
    def primitive_evaluations(self) -> int:
        return self.precondition_evaluations + self.successor_evaluations


@dataclass(frozen=True)
class CountedPlan:
    plan: tuple[PlanStep, ...] | None
    work: WorkCounter


def route_counted(
    belief,
    position: tuple[int, int],
    direction: int,
    *,
    goals: set[tuple[tuple[int, int], int]],
    has_key: bool = False,
    terminal: str | None = None,
    key_label: str | None = None,
    allow_toggle: bool = True,
) -> CountedPlan:
    """Run the common uniform-cost planner and count shared primitives."""

    initial = (position, direction, frozenset())
    heap = [(0, 0, initial, ())]
    seen = {}
    serial = 0
    preconditions = 0
    successors = 0
    settled = 0
    while heap:
        cost, _, state, steps = heapq.heappop(heap)
        if state in seen:
            continue
        seen[state] = cost
        settled += 1
        pos, heading, opened = state
        if (pos, heading) in goals:
            if terminal is None:
                return CountedPlan(
                    tuple(steps),
                    WorkCounter(preconditions, successors, settled),
                )
            front = (pos[0] + DIRS[heading][0], pos[1] + DIRS[heading][1])
            label = belief.label(front)
            preconditions += 1
            terminal_step = PlanStep(terminal, pos, heading, ((front, label),))
            return CountedPlan(
                tuple(steps) + (terminal_step,),
                WorkCounter(preconditions, successors, settled),
            )

        options = [
            ("left", pos, (heading - 1) % 4, opened, (), 1),
            ("right", pos, (heading + 1) % 4, opened, (), 1),
        ]
        front = (pos[0] + DIRS[heading][0], pos[1] + DIRS[heading][1])
        label = belief.label(front)
        preconditions += 1
        if _passable_label(label) or front in opened:
            options.append(("forward", front, heading, opened, ((front, label),), 1))
        elif allow_toggle and (
            _closed_label(label) or _can_toggle_locked(label, has_key, key_label)
        ):
            options.append(("toggle", pos, heading, opened | {front}, ((front, label),), 1))
        successors += len(options)
        for action, next_position, next_heading, door_set, requirements, weight in options:
            nxt = (next_position, next_heading, frozenset(door_set))
            if nxt in seen:
                continue
            serial += 1
            heapq.heappush(
                heap,
                (
                    cost + weight,
                    serial,
                    nxt,
                    steps + (PlanStep(action, pos, heading, requirements),),
                ),
            )
    return CountedPlan(None, WorkCounter(preconditions, successors, settled))


def requirements_supported_counted(steps: Iterable[PlanStep], belief) -> tuple[bool, WorkCounter]:
    """Validate plan-step requirements and count every visited primitive."""

    preconditions = 0
    validations = 0
    for step in steps:
        validations += 1
        for position, expected in step.requirements:
            preconditions += 1
            if belief.label(tuple(position)) != expected:
                return False, WorkCounter(
                    precondition_evaluations=preconditions,
                    step_validations=validations,
                )
    return True, WorkCounter(
        precondition_evaluations=preconditions,
        step_validations=validations,
    )
