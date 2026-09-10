from __future__ import annotations


def _world():
    cells = {}
    for x in range(0, 7):
        cells[(x, 0)] = "wall"
        cells[(x, 2)] = "wall"
    cells[(0, 1)] = "wall"
    cells[(6, 1)] = "wall"
    cells.update(
        {
            (1, 1): "empty",
            (2, 1): "key:red",
            (3, 1): "door_locked:red",
            (4, 1): "empty",
            (5, 1): "box:green",
        }
    )
    support = {position: f"rgb-{position[0]}-{position[1]}" for position in cells}
    return cells, support


def test_common_mission_planner_solves_key_door_release_target_sequence():
    from connectomequest.embodied.task_planner import plan_mission

    cells, support = _world()
    outcome = plan_mission(
        cells,
        support,
        position=(1, 1),
        direction=0,
        inventory=None,
        mission="pick up the green box",
    )
    assert outcome.plan is not None
    actions = tuple(step.action for step in outcome.plan)
    assert actions.count("pickup") >= 2
    assert "toggle" in actions
    assert "drop" in actions
    assert actions[-1] == "pickup"
    assert outcome.work.successor_evaluations > 0
    assert outcome.work.states_settled > 0


def test_common_mission_planner_is_deterministic_and_dependencies_are_complete():
    from connectomequest.embodied.task_planner import plan_mission

    cells, support = _world()
    kwargs = dict(
        cells=cells,
        support=support,
        position=(1, 1),
        direction=0,
        inventory=None,
        mission="pick up the green box",
    )
    first = plan_mission(**kwargs)
    second = plan_mission(**kwargs)
    assert first == second
    assert first.plan is not None
    assert tuple(row.step_index for row in first.dependencies) == tuple(range(len(first.plan)))
    assert all(row.required_receipts <= frozenset(support.values()) for row in first.dependencies)


def test_planner_uses_no_environment_or_oracle_argument():
    import inspect

    from connectomequest.embodied.task_planner import plan_mission

    names = set(inspect.signature(plan_mission).parameters)
    assert names == {
        "cells",
        "support",
        "position",
        "direction",
        "inventory",
        "mission",
        "max_states",
    }


def test_unknown_or_incomplete_belief_returns_no_plan():
    from connectomequest.embodied.task_planner import plan_mission

    outcome = plan_mission(
        {(1, 1): "empty"},
        {(1, 1): "r1"},
        position=(1, 1),
        direction=0,
        inventory=None,
        mission="pick up the green box",
    )
    assert outcome.plan is None
    assert outcome.dependencies == ()


def test_long_open_map_uses_target_lower_bound_without_changing_optimality():
    from connectomequest.embodied.task_planner import plan_mission

    cells = {
        (x, y): ("wall" if x in (0, 25) or y in (0, 8) else "empty")
        for x in range(26)
        for y in range(9)
    }
    cells[(3, 3)] = "key:red"
    cells[(12, 4)] = "door_locked:red"
    cells[(22, 4)] = "box:green"
    support = {position: f"r-{position}" for position in cells}
    outcome = plan_mission(
        cells,
        support,
        position=(2, 4),
        direction=0,
        inventory=None,
        mission="pick up the green box",
    )
    assert outcome.plan is not None
    assert len(outcome.plan) == 25
    assert outcome.work.states_settled < 2_000


def test_off_route_door_hypothesis_does_not_block_a_direct_target_route():
    from connectomequest.embodied.task_planner import plan_mission

    cells, support = _world()
    cells[(4, 0)] = "door_locked:blue"
    cells[(3, 1)] = "door_open:red"
    cells[(2, 1)] = "empty"
    support[(4, 0)] = "rgb-off-route-door"
    outcome = plan_mission(
        cells,
        support,
        position=(1, 1),
        direction=0,
        inventory=None,
        mission="pick up the green box",
    )
    assert outcome.plan is not None
    assert outcome.plan[-1].action == "pickup"
