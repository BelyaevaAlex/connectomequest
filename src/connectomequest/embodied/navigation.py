"""Common orientation-aware planning on observed labels; no environment access."""

from dataclasses import dataclass

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
PASSABLE = {"empty", "door_open", "floor", "goal"}


@dataclass(frozen=True)
class PlanStep:
    action: str
    position: tuple
    direction: int
    requirements: tuple


def route(belief, position, direction, *, goals, has_key=False, terminal=None):
    """Uniform-cost primitive actions; door toggle costs one action too."""
    import heapq

    initial = (position, direction, frozenset())
    heap = [(0, 0, initial, ())]
    seen = {}
    serial = 0
    while heap:
        cost, _, state, steps = heapq.heappop(heap)
        if state in seen:
            continue
        seen[state] = cost
        pos, heading, opened = state
        if (pos, heading) in goals:
            if terminal is None:
                return list(steps)
            front = (pos[0] + DIRS[heading][0], pos[1] + DIRS[heading][1])
            return list(steps) + (
                [PlanStep(terminal, pos, heading, ((front, belief.label(front)),))]
            )
        options = [
            ("left", pos, (heading - 1) % 4, opened, (), 1),
            ("right", pos, (heading + 1) % 4, opened, (), 1),
        ]
        front = (pos[0] + DIRS[heading][0], pos[1] + DIRS[heading][1])
        label = belief.label(front)
        if label in PASSABLE or front in opened:
            options.append(("forward", front, heading, opened, ((front, label),), 1))
        elif label == "door_closed" or (label == "door_locked" and has_key):
            options.append(("toggle", pos, heading, opened | {front}, ((front, label),), 1))
        for action, p, h, door_set, req, weight in options:
            nxt = (p, h, frozenset(door_set))
            if nxt in seen:
                continue
            serial += 1
            heapq.heappush(
                heap, (cost + weight, serial, nxt, steps + (PlanStep(action, pos, heading, req),))
            )
    return None


def adjacent_goals(positions, belief):
    return {
        ((p[0] - dx, p[1] - dy), d)
        for p in positions
        for d, (dx, dy) in enumerate(DIRS)
        if belief.label((p[0] - dx, p[1] - dy)) in PASSABLE
    }


def make_plan(obs, belief, scanned):
    target = f"box:{obs.goals[obs.stage]}"
    positions = [p for p, label in belief.cells.items() if label == target]
    door_cells = {p for p, label in belief.cells.items() if label.startswith("door_")}
    door_neighbours = {(p[0] + dx, p[1] + dy) for p in door_cells for dx, dy in DIRS}

    def drop_plan(avoid=()):
        avoid = set(avoid) | door_neighbours | door_cells
        empty = [
            p
            for p, label in belief.cells.items()
            if label == "empty" and p != obs.position and p not in avoid
        ]
        return route(
            belief,
            obs.position,
            obs.direction,
            goals=adjacent_goals(empty, belief),
            terminal="drop",
        )

    # Carrying an object prevents pickup, but retain the key until required doors open.
    if obs.carrying is not None:
        reachable = (
            route(
                belief,
                obs.position,
                obs.direction,
                goals=adjacent_goals(positions, belief),
                has_key=obs.carrying == "key",
                terminal="pickup",
            )
            if positions
            else None
        )
        if reachable is not None and not any(s.action == "toggle" for s in reachable):
            plan = drop_plan(p for step in reachable for p, _ in step.requirements)
            if plan:
                return plan, "drop"
        if obs.carrying != "key":
            # A completed/cleared box must also be put down when the next target
            # is not yet reachable; otherwise the agent cannot acquire a key.
            plan = drop_plan(p for step in (reachable or []) for p, _ in step.requirements)
            if plan:
                return plan, "free_inventory"
    if positions:
        plan = route(
            belief,
            obs.position,
            obs.direction,
            goals=adjacent_goals(positions, belief),
            has_key=obs.carrying == "key",
            terminal="pickup",
        )
        if plan:
            return plan, "target"
    blockers = [
        p
        for p, label in belief.cells.items()
        if p in door_neighbours and label.startswith("box:") and label != target
    ]
    if blockers:
        plan = (
            drop_plan()
            if obs.carrying is not None
            else route(
                belief,
                obs.position,
                obs.direction,
                goals=adjacent_goals(blockers, belief),
                terminal="pickup",
            )
        )
        if plan:
            return plan, "clear_doorway"
    if obs.carrying is None:
        keys = [p for p, label in belief.cells.items() if label.startswith("key:")]
        if keys:
            plan = route(
                belief,
                obs.position,
                obs.direction,
                goals=adjacent_goals(keys, belief),
                terminal="pickup",
            )
            if plan:
                return plan, "key"
    # An observed closed doorway is an acquisition opportunity even if no cell
    # behind it has been seen. Restrict this to publicly labeled openable doors.
    doors = [
        p
        for p, label in belief.cells.items()
        if label == "door_closed" or (label == "door_locked" and obs.carrying == "key")
    ]
    if doors:
        plan = route(
            belief,
            obs.position,
            obs.direction,
            goals=adjacent_goals(doors, belief),
            has_key=obs.carrying == "key",
            terminal="toggle",
        )
        if plan:
            return plan, "door"
    # Look in new directions near unknown cells; never use hidden map bounds.
    goals = {
        (p, d)
        for p, label in belief.cells.items()
        if label in PASSABLE
        for d in range(4)
        if (p, d) not in scanned
        and any((p[0] + dx, p[1] + dy) not in belief.cells for dx, dy in DIRS)
    }
    plan = (
        route(belief, obs.position, obs.direction, goals=goals, has_key=obs.carrying == "key")
        if goals
        else None
    )
    if plan:
        return plan, "explore"
    # No early abort while temporary perception faults can clear.
    return [PlanStep("right", obs.position, obs.direction, ())], "look"
