"""RGB-only color-aware standard-task controls. No environment reference.

The added door decoder is renderer-supervised nearest-template perception,
not a claim of visual OOD learning. Search backends are independently coded;
high-level subgoal priorities and sensor interpretation are shared controls.
"""

import heapq
from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from connectomequest.embodied.perturbations import view_to_world
from connectomequest.embodied.tile_perception import split_rgb_tiles

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
PASSABLE = {"empty", "floor", "goal"}
METHODS = ("reference", "full_replan", "persistent")


def passable(label):
    return label in PASSABLE or label.startswith("door_open:")


def allowed(action, front, inventory):
    if action in ("left", "right"):
        return True
    if action == "forward":
        return passable(front)
    if action == "pickup":
        return inventory is None and front.split(":")[0] in ("key", "box", "ball")
    if action == "drop":
        return inventory is not None and front == "empty"
    if action == "toggle":
        if front.startswith("door_closed:"):
            return True
        if front.startswith("door_locked:"):
            color = front.split(":", 1)[1]
            return color != "unknown" and inventory == "key:" + color
    return False


class Inventory:
    def __init__(self):
        self.label = None
        self.proposed = None
        self.support = None

    def propose(self, action, front, receipt=None):
        self.proposed = (action, front, receipt)

    def update(self, obs):
        if obs.carrying is None:
            self.label = None
            self.support = None
        elif obs.last_action == "pickup" and obs.acknowledged:
            known = (
                self.proposed[1] if self.proposed and self.proposed[0] == "pickup" else "unknown"
            )
            self.label = (
                known if known.startswith(obs.carrying + ":") else obs.carrying + ":unknown"
            )
            self.support = self.proposed[2] if self.proposed else None
        elif self.label is None or not self.label.startswith(obs.carrying + ":"):
            self.label = obs.carrying + ":unknown"


class DoorColorDecoder:
    """36 synthetic 8x8 templates, generated independently of task seeds."""

    colors = ("red", "green", "blue", "purple", "yellow", "grey")

    def __init__(self):
        from minigrid.core.grid import Grid
        from minigrid.core.world_object import Door

        self.templates = {}
        for state, name in enumerate(("door_open", "door_closed", "door_locked")):
            self.templates[name] = np.stack(
                [
                    Grid.render_tile(
                        Door(c, is_open=state == 0, is_locked=state == 2), tile_size=8, highlight=h
                    ).astype(np.float32)
                    for c in self.colors
                    for h in (False, True)
                ]
            )

    def predict(self, tile, state):
        errors = ((self.templates[state] - tile.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
        order = np.argsort(errors)
        best = int(order[0])
        color = best // 2
        # Reject grossly mismatched appearances; threshold fixed before dev.
        if errors[best] > 1500:
            return "unknown"
        return self.colors[color]


class ColorPerceptor:
    def __init__(self, perceptor):
        self.base = perceptor
        self.decoder = DoorColorDecoder()

    def predict(self, rgb):
        tiles = split_rgb_tiles(rgb)
        rows = []
        for x, y, label, confidence in self.base.predict(rgb):
            if label in self.decoder.templates:
                label = label + ":" + self.decoder.predict(tiles[x * 7 + y], label)
            rows.append((x, y, label, confidence))
        return rows


@dataclass(frozen=True)
class Step:
    action: str
    position: tuple
    direction: int
    front: str


def search(cells, position, direction, goals, inventory, algorithm="uniform"):
    if not goals:
        return None
    if algorithm == "reference":
        return _reference_bfs(cells, position, direction, goals, inventory)
    if algorithm != "uniform":
        raise ValueError(algorithm)
    initial = (position, direction, frozenset())
    heap = [(0, 0, initial, ())]
    seen = set()
    serial = 0
    while heap:
        distance, _, state, path = heapq.heappop(heap)
        if state in seen:
            continue
        seen.add(state)
        pos, d, opened = state
        if (pos, d) in goals:
            return list(path)
        front = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
        label = cells.get(front, "unknown")
        options = [("left", (pos, (d - 1) % 4, opened)), ("right", (pos, (d + 1) % 4, opened))]
        if front in opened or passable(label):
            options.append(("forward", (front, d, opened)))
        elif allowed("toggle", label, inventory):
            options.append(("toggle", (pos, d, opened | {front})))
        for action, nxt in options:
            if nxt in seen:
                continue
            serial += 1
            heapq.heappush(heap, (distance + 1, serial, nxt, path + (Step(action, pos, d, label),)))
    return None


def _reference_bfs(cells, position, direction, goals, inventory):
    # Separate frontier, transition expansion and predecessor reconstruction.
    start = (position[0], position[1], direction, ())
    queue = deque([start])
    previous = {start: None}
    while queue:
        node = queue.popleft()
        x, y, d, opened = node
        if ((x, y), d) in goals:
            result = []
            while previous[node] is not None:
                parent, action, label = previous[node]
                result.append(Step(action, (parent[0], parent[1]), parent[2], label))
                node = parent
            return result[::-1]
        tx, ty = x + DIRS[d][0], y + DIRS[d][1]
        label = cells.get((tx, ty), "unknown")
        successors = [("left", (x, y, (d + 3) % 4, opened)), ("right", (x, y, (d + 1) % 4, opened))]
        if (tx, ty) in opened or label in PASSABLE or label.startswith("door_open:"):
            successors.append(("forward", (tx, ty, d, opened)))
        elif label.startswith("door_closed:") or (
            label.startswith("door_locked:")
            and label.split(":")[1] != "unknown"
            and inventory == "key:" + label.split(":")[1]
        ):
            successors.append(("toggle", (x, y, d, tuple(sorted((*opened, (tx, ty)))))))
        for action, nxt in successors:
            if nxt not in previous:
                previous[nxt] = (node, action, label)
                queue.append(nxt)
    return None


class Controller:
    def __init__(self, perceptor, method):
        if method not in METHODS:
            raise ValueError(method)
        self.perceptor = perceptor
        self.method = method
        self.cells = {}
        self.support = {}
        self.inventory = Inventory()
        self.scanned = set()
        self.visits = Counter()
        self.plan = []
        self.phase = "start"
        self.phases = Counter()
        self.requirements = ()
        self.guard_rejections = 0
        self.receipt_rejections = 0
        self.last_front = None
        self.stats = Counter()
        self.last_goal = None
        self.door_sites = {}
        self.deposited = set()

    def _route(self, obs, goals, terminal=None):
        route = search(
            self.cells,
            obs.position,
            obs.direction,
            goals,
            self.inventory.label,
            "reference" if self.method == "reference" else "uniform",
        )
        if route is None:
            return None
        if terminal:
            if route:
                last = route[-1]
                pos, d = last.position, last.direction
                if last.action == "forward":
                    pos = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
                elif last.action in ("left", "right"):
                    d = (d + (-1 if last.action == "left" else 1)) % 4
            else:
                pos, d = obs.position, obs.direction
            front = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
            route = route + [Step(terminal, pos, d, self.cells.get(front, "unknown"))]
        return route

    def _object(self, obs, positions, action):
        goals = {
            ((p[0] - dx, p[1] - dy), d)
            for p in positions
            for d, (dx, dy) in enumerate(DIRS)
            if passable(self.cells.get((p[0] - dx, p[1] - dy), "unknown"))
        }
        return self._route(obs, goals, action)

    def _drop(self, obs, avoid=()):
        doors = set(self.door_sites)
        excluded = set(avoid) | doors | {(p[0] + dx, p[1] + dy) for p in doors for dx, dy in DIRS}
        slots = [
            p
            for p, l in self.cells.items()
            if l == "empty" and p != obs.position and p not in excluded
        ]
        return self._object(obs, slots, "drop")

    def clearance(self, obs):
        """An occupied hand can block a prerequisite, even when it holds a key."""
        for door in sorted(self.door_sites):
            if self._object(obs, [door], "toggle") is not None:
                continue
            blockers = [
                p
                for p, l in self.cells.items()
                if l.split(":")[0] in ("ball", "box", "key")
                and abs(p[0] - door[0]) + abs(p[1] - door[1]) == 1
            ]
            path = self._object(obs, blockers, "pickup") if blockers else None
            if path:
                return self._drop(obs, [s.position for s in path]) if self.inventory.label else path
        return None

    def choose(self, obs):
        if self.method == "reference":
            from connectomequest.embodied.reference_policy import choose

            return choose(self, obs)
        target = "box:" + obs.goals[obs.stage]
        inv = self.inventory.label
        positions = [p for p, l in self.cells.items() if l == target]
        path = self._object(obs, positions, "pickup") if positions else None
        locked = {p: l.split(":")[1] for p, l in self.cells.items() if l.startswith("door_locked:")}
        if path:
            if inv is None:
                return path, "target"
            toggles = [i for i, s in enumerate(path) if s.action == "toggle"]
            if toggles:
                return path[: toggles[0] + 1], "unlock_for_target"
            drop = self._drop(obs, [s.position for s in path])
            if drop:
                return drop, "free_hand_for_target"
        clear = self.clearance(obs)
        if clear:
            return clear, "clear_with_free_hand" if inv else "clear_obstruction"
        # Keep a key only while an observed locked door needs that color.
        if inv is not None and (
            not inv.startswith("key:") or inv.split(":")[1] not in locked.values()
        ):
            drop = self._drop(obs)
            if drop:
                return drop, "free_hand"
        openable = [p for p, l in self.cells.items() if allowed("toggle", l, inv)]
        if openable:
            path = self._object(obs, openable, "toggle")
            if path:
                return path, "open_door"
        if inv is None and locked:
            keys = [
                p
                for p, l in self.cells.items()
                if l.startswith("key:") and l.split(":")[1] in locked.values()
            ]
            path = self._object(obs, keys, "pickup") if keys else None
            if path:
                return path, "matching_key"
        # All frontier poses are generated only from known cells; unknown map
        # dimensions and true target location never enter the planner.
        goals = {
            (p, d)
            for p, l in self.cells.items()
            if passable(l)
            for d in range(4)
            if (p, d) not in self.scanned
        }
        path = self._route(obs, goals)
        if path:
            return path, "explore"
        # Moving an observed object beside a door can expose an inaccessible room.
        if inv is None:
            blockers = [
                p
                for p, l in self.cells.items()
                if l.split(":")[0] in ("ball", "box", "key") and p not in self.deposited
            ]
            path = self._object(obs, blockers, "pickup") if blockers else None
            if path:
                return path, "clear_doorway"
        return [Step("right", obs.position, obs.direction, "unknown")], "rescan"

    def act(self, obs):
        started = perf_counter()
        self.inventory.update(obs)
        if obs.last_action == "drop" and obs.acknowledged and self.last_front is not None:
            self.deposited.add(self.last_front)
        predictions = self.perceptor.predict(obs.rgb)
        self.stats["inference_ms"] += (perf_counter() - started) * 1000
        changed = set()
        for x, y, label, confidence in predictions:
            if (x, y) == (3, 6):
                continue
            if label == "unseen" or confidence < 0.45:
                continue
            p = view_to_world(obs.position, obs.direction, x, y)
            if label.startswith("door_"):
                self.door_sites[p] = label.split(":")[1]
            if self.cells.get(p) != label:
                changed.add(p)
            self.cells[p] = label
            self.support[p] = obs.receipt_id
        self.cells[obs.position] = (
            "door_open:" + self.door_sites[obs.position]
            if obs.position in self.door_sites
            else "empty"
        )
        self.support[obs.position] = obs.receipt_id
        self.scanned.add((obs.position, obs.direction))
        self.visits[obs.position, obs.direction] += 1
        if obs.last_action and not obs.acknowledged:
            self.stats["ack_failures"] += 1
            self.plan = []
        started = perf_counter()
        # Opportunistic perception can change useful subgoals. Replan on newly
        # interpreted cells, failures, or exhausted plan; no implicit oracle.
        reuse = (
            self.method == "persistent"
            and self.plan
            and not changed
            and (self.plan[0].position, self.plan[0].direction) == (obs.position, obs.direction)
        )
        if not reuse:
            self.plan, self.phase = self.choose(obs)
            self.stats["plans_built"] += 1
        else:
            self.stats["reused_steps"] += 1
        step = self.plan.pop(0)
        front = (obs.position[0] + DIRS[obs.direction][0], obs.position[1] + DIRS[obs.direction][1])
        label = self.cells.get(front, "unknown")
        action = step.action
        if not allowed(action, label, self.inventory.label):
            self.guard_rejections += 1
            self.plan = []
            action = "right"
            self.phase = "guard_fallback"
        self.requirements = (
            () if action in ("left", "right") else ((front, label, self.support.get(front)),)
        )
        if any(
            not token or self.cells.get(tuple(p)) != l or self.support.get(tuple(p)) != token
            for p, l, token in self.requirements
        ):
            self.receipt_rejections += 1
            raise AssertionError("unsupported action")
        self.inventory.propose(action, label, self.support.get(front))
        self.phases[self.phase] += 1
        self.last_front = front
        self.stats["planning_ms"] += (perf_counter() - started) * 1000
        return action
