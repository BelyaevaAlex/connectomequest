"""Separately implemented nearest-subgoal reference over the same RGB belief.

Uses the BFS backend, not the persistent controller's subgoal chooser. Shared
sensor interpretation, legal-action definitions and trace harness are explicit.
"""

from connectomequest.embodied.symbolic_controller import DIRS, Step, allowed, passable


def choose(agent, obs):
    cells = agent.cells
    held = agent.inventory.label
    target = "box:" + obs.goals[obs.stage]
    known_doors = set(agent.door_sites)

    def adjacent(positions, action):
        return agent._object(obs, positions, action)

    def deposit(avoid=()):
        forbidden = (
            set(avoid) | known_doors | {(x + dx, y + dy) for x, y in known_doors for dx, dy in DIRS}
        )
        slots = [
            p for p, l in cells.items() if l == "empty" and p != obs.position and p not in forbidden
        ]
        return adjacent(slots, "drop")

    targets = [p for p, l in cells.items() if l == target]
    direct = adjacent(targets, "pickup") if targets else None
    if direct:
        if held is None:
            return direct, "reference_target"
        first_open = next((i for i, s in enumerate(direct) if s.action == "toggle"), None)
        if first_open is not None:
            return direct[: first_open + 1], "reference_unlock"
        path = deposit(s.position for s in direct)
        if path:
            return path, "reference_deposit"
    # Same inventory prerequisite repair, implemented in the reference chooser.
    for door in sorted(known_doors):
        if adjacent([door], "toggle") is not None:
            continue
        obstacles = [
            p
            for p, l in cells.items()
            if l.split(":")[0] in ("ball", "box", "key")
            and abs(p[0] - door[0]) + abs(p[1] - door[1]) == 1
        ]
        clear = adjacent(obstacles, "pickup") if obstacles else None
        if clear:
            if held:
                clear = deposit(s.position for s in clear)
            if clear:
                return clear, "reference_clear_inventory"
    needed = {l.split(":")[1] for l in cells.values() if l.startswith("door_locked:")}
    keep = held and held.startswith("key:") and held.split(":")[1] in needed
    if held and not keep:
        path = deposit()
        if path:
            return path, "reference_free_hand"
    possibilities = []
    for p, l in sorted(cells.items()):
        if allowed("toggle", l, held):
            route = adjacent([p], "toggle")
            if route:
                possibilities.append((len(route), p, route, "reference_door"))
        if held is None and l.startswith("key:") and l.split(":")[1] in needed:
            route = adjacent([p], "pickup")
            if route:
                possibilities.append((len(route), p, route, "reference_key"))
    if possibilities:
        _, _, path, phase = min(possibilities, key=lambda x: (x[0], x[1]))
        return path, phase
    # Unlike the persistent controller, clear a known doorway obstruction before
    # exhaustively inspecting every reachable orientation.
    if held is None:
        blockers = [
            p
            for p, l in cells.items()
            if l.split(":")[0] in ("ball", "box", "key")
            and l != target
            and any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in known_doors)
        ]
        path = adjacent(blockers, "pickup") if blockers else None
        if path:
            return path, "reference_clear"
    frontiers = {
        (p, d)
        for p, l in cells.items()
        if passable(l)
        for d in range(4)
        if (p, d) not in agent.scanned
    }
    path = agent._route(obs, frontiers)
    if path:
        return path, "reference_explore"
    if held is None:
        obstacles = [
            p
            for p, l in cells.items()
            if l.split(":")[0] in ("ball", "box", "key") and p not in agent.deposited
        ]
        clear = adjacent(obstacles, "pickup") if obstacles else None
        if clear:
            return clear, "reference_clear_reachability"
    return [Step("right", obs.position, obs.direction, "unknown")], "reference_rescan"
