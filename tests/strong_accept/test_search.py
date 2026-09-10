from itertools import product

from connectomequest.embodied.navigation import PlanStep, route
from connectomequest.embodied.search import (
    requirements_supported_counted,
    route_counted,
)


class Belief:
    def __init__(self, cells: dict[tuple[int, int], str]):
        self.cells = dict(cells)

    def label(self, position: tuple[int, int]) -> str | None:
        return self.cells.get(tuple(position))


def corridor() -> Belief:
    return Belief({(0, 0): "empty", (1, 0): "empty", (2, 0): "empty"})


def test_counted_route_matches_legacy_plan_and_counts_work() -> None:
    belief = corridor()
    legacy = route(belief, (0, 0), 0, goals={((2, 0), 0)})

    counted = route_counted(belief, (0, 0), 0, goals={((2, 0), 0)})

    assert list(counted.plan or ()) == legacy
    assert counted.work.states_settled > 0
    assert counted.work.successor_evaluations >= counted.work.states_settled
    assert counted.work.precondition_evaluations > 0


def test_counted_route_matches_unreachable_legacy_result() -> None:
    belief = Belief({(0, 0): "empty", (1, 0): "wall"})

    legacy = route(belief, (0, 0), 0, goals={((1, 0), 0)})
    counted = route_counted(belief, (0, 0), 0, goals={((1, 0), 0)})

    assert legacy is None
    assert counted.plan is None
    assert counted.work.states_settled == 4


def test_locked_door_requires_key_in_both_planners() -> None:
    belief = Belief({(0, 0): "empty", (1, 0): "door_locked", (2, 0): "empty"})
    goals = {((2, 0), 0)}

    assert route(belief, (0, 0), 0, goals=goals, has_key=False) is None
    without_key = route_counted(belief, (0, 0), 0, goals=goals, has_key=False)
    assert without_key.plan is None

    legacy = route(belief, (0, 0), 0, goals=goals, has_key=True)
    counted = route_counted(belief, (0, 0), 0, goals=goals, has_key=True)
    assert list(counted.plan or ()) == legacy
    assert [step.action for step in counted.plan or ()] == ["toggle", "forward", "forward"]


def test_route_can_forbid_door_toggles_for_frontier_search() -> None:
    """A generic frontier search must not enumerate door-open subsets."""

    belief = Belief({(0, 0): "empty", (1, 0): "door_closed:red", (2, 0): "empty"})

    counted = route_counted(
        belief,
        (0, 0),
        0,
        goals={((2, 0), 0)},
        allow_toggle=False,
    )

    assert counted.plan is None
    assert counted.work.states_settled == 4


def test_terminal_action_and_requirements_match_legacy() -> None:
    belief = Belief({(0, 0): "empty", (1, 0): "box:red"})
    goals = {((0, 0), 0)}

    legacy = route(belief, (0, 0), 0, goals=goals, terminal="pickup")
    counted = route_counted(belief, (0, 0), 0, goals=goals, terminal="pickup")

    assert list(counted.plan or ()) == legacy
    assert counted.plan == (PlanStep("pickup", (0, 0), 0, (((1, 0), "box:red"),)),)


def test_requirement_checks_count_each_checked_precondition() -> None:
    belief = corridor()
    steps = (
        PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),
        PlanStep("forward", (1, 0), 0, (((2, 0), "empty"),)),
    )

    supported, work = requirements_supported_counted(steps, belief)

    assert supported is True
    assert work.precondition_evaluations == 2
    assert work.step_validations == 2
    assert work.successor_evaluations == 0


def test_requirement_checks_stop_at_first_mismatch() -> None:
    belief = Belief({(1, 0): "wall", (2, 0): "empty"})
    steps = (
        PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),
        PlanStep("forward", (1, 0), 0, (((2, 0), "empty"),)),
    )

    supported, work = requirements_supported_counted(steps, belief)

    assert supported is False
    assert work.precondition_evaluations == 1
    assert work.step_validations == 1


def test_small_grid_plans_exactly_match_legacy_search() -> None:
    positions = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 0))
    for labels in product(("empty", "wall"), repeat=len(positions)):
        cells = {(0, 0): "empty", **dict(zip(positions, labels, strict=True))}
        belief = Belief(cells)
        for direction in range(4):
            legacy = route(belief, (0, 0), direction, goals={((2, 0), 0)})
            counted = route_counted(belief, (0, 0), direction, goals={((2, 0), 0)})
            actual = None if counted.plan is None else list(counted.plan)
            assert actual == legacy


def test_colored_open_door_is_passable_in_embodied_belief() -> None:
    belief = Belief({(0, 0): "empty", (1, 0): "door_open:green", (2, 0): "empty"})

    result = route_counted(belief, (0, 0), 0, goals={((2, 0), 0)})

    assert result.plan is not None
    assert [step.action for step in result.plan] == ["forward", "forward"]


def test_colored_locked_door_requires_matching_key_label() -> None:
    belief = Belief({(0, 0): "empty", (1, 0): "door_locked:red", (2, 0): "empty"})

    wrong = route_counted(
        belief, (0, 0), 0, goals={((2, 0), 0)}, has_key=True, key_label="key:blue"
    )
    unknown = route_counted(
        belief, (0, 0), 0, goals={((2, 0), 0)}, has_key=True, key_label="key:unknown"
    )
    matching = route_counted(
        belief, (0, 0), 0, goals={((2, 0), 0)}, has_key=True, key_label="key:red"
    )

    assert wrong.plan is None
    assert unknown.plan is None
    assert matching.plan is not None
    assert [step.action for step in matching.plan][:2] == ["toggle", "forward"]
