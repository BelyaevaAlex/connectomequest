"""Persistent plan execution with public-observation-only support monitoring."""

import hashlib
from time import perf_counter_ns

from connectomequest.embodied.dependent_evidence import SupportedMap
from connectomequest.embodied.navigation import DIRS, adjacent_goals, make_plan, route
from connectomequest.embodied.perturbations import view_to_world
from connectomequest.embodied.semantic_map import PlainMap

MODES = ("replan", "local", "full_history", "indexed", "targeted", "reset")


class DependentAgent:
    def __init__(self, perceptor, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.perceptor = perceptor
        self.mode = mode
        self.indexed_support = mode in {"full_history", "indexed", "targeted"}
        self.map = (
            SupportedMap("full" if mode == "full_history" else "indexed")
            if self.indexed_support
            else PlainMap()
        )
        self.scanned = set()
        self.plan = []
        self.suffix = []
        self.plan_index = 0
        self.serial = 0
        self.stage = -1
        self.last_target = None
        self.commitment = None
        self.stats = {
            "plans_built": 0,
            "plan_steps_reused": 0,
            "failed_acknowledgements": 0,
            "future_invalidations": 0,
            "reacquisition_actions": 0,
            "map_resets": 0,
            "planning_ms": 0.0,
            "inference_ms": 0.0,
            "max_plan_length": 0,
        }
        self.suspect = set()
        self.dependency_counts = {}
        self.reacquiring = False

    def _compile(self, obs, steps):
        if not self.indexed_support:
            self.plan = steps
            self.suffix = []
            self.plan_index = 0
            self.stats["plans_built"] += 1
            self.stats["max_plan_length"] = max(self.stats["max_plan_length"], len(steps))
            return
        g = self.map.graph
        self.serial += 1
        context = ("context", self.serial)
        with g.batch():
            g.assert_fact(f"context:{self.serial}", obs.receipt_id, context)
            heads = []
            for i, step in enumerate(steps):
                head = ("step", self.serial, i)
                premises = [context] + [self.map.fact(p, label) for p, label in step.requirements]
                g.justify(head, premises)
                heads.append(head)
            suffix = []
            for i in reversed(range(len(heads))):
                node = ("suffix", self.serial, i)
                g.justify(node, [heads[i]] + suffix[-1:])
                suffix.append(node)
        self.plan = steps
        self.suffix = list(reversed(suffix))
        self.plan_index = 0
        self.stats["plans_built"] += 1
        self.stats["max_plan_length"] = max(self.stats["max_plan_length"], len(steps))

    def act(self, obs):
        if self.plan and self.plan_index >= len(self.plan):
            self.commitment = None
        if obs.last_action is not None and not obs.acknowledged:
            self.stats["failed_acknowledgements"] += 1
            remaining = self.plan[self.plan_index :]
            self.suspect = {p for step in remaining for p, _ in step.requirements}
            self.dependency_counts = {
                p: sum(p in [q for q, _ in s.requirements] for s in remaining) for p in self.suspect
            }
            if self.last_target is not None:
                self.map.invalidate([self.last_target])
            if self.mode == "reset":
                self.map.invalidate(list(self.map.cells))
                self.scanned.clear()
                self.stats["map_resets"] += 1
            self.plan = []
            self.commitment = None
        start = perf_counter_ns()
        predictions = self.perceptor.predict(obs.rgb)
        self.stats["inference_ms"] += (perf_counter_ns() - start) / 1e6
        items = []
        visible = set()
        for x, y, label, confidence in predictions:
            if label == "unseen" or confidence < 0.45:
                continue
            p = view_to_world(obs.position, obs.direction, x, y)
            items.append((p, label))
            visible.add(p)
        items.append((obs.position, "empty"))
        self.map.observe(obs.receipt_id, hashlib.sha256(obs.rgb.tobytes()).hexdigest(), items)
        self.suspect -= visible
        self.scanned.add((obs.position, obs.direction))
        remaining = self.plan[self.plan_index :]
        invalid = bool(
            remaining
            and (
                not self.map.graph.supported(self.suffix[self.plan_index])
                if self.indexed_support
                else any(
                    self.map.label(p) != label
                    for step in remaining
                    for p, label in step.requirements
                )
            )
        )
        if invalid:
            self.stats["future_invalidations"] += 1
        mismatch = bool(
            remaining
            and (remaining[0].position != obs.position or remaining[0].direction != obs.direction)
        )
        impossible_pickup = bool(
            remaining and remaining[0].action == "pickup" and obs.carrying is not None
        )
        if obs.stage != self.stage or impossible_pickup:
            self.commitment = None
        immediate_invalid = bool(
            remaining and any(self.map.label(p) != label for p, label in remaining[0].requirements)
        )
        need_plan = (
            not remaining
            or mismatch
            or impossible_pickup
            or obs.stage != self.stage
            or self.mode == "replan"
            or (invalid and self.mode != "local")
            or immediate_invalid
        )
        if need_plan:
            start = perf_counter_ns()
            steps = None
            self.reacquiring = False
            if self.mode == "targeted" and self.suspect:
                # Public-history heuristic: dependency multiplicity per navigation cost.
                scored = []
                for p in sorted(self.suspect):
                    path = route(
                        self.map,
                        obs.position,
                        obs.direction,
                        goals=adjacent_goals([p], self.map),
                        has_key=obs.carrying == "key",
                    )
                    if path:
                        multiplicity = self.dependency_counts.get(p, 0)
                        scored.append((-(1 + multiplicity) / len(path), p, path))
                if scored:
                    steps = min(scored, key=lambda x: (x[0], x[1]))[2]
                    self.reacquiring = True
            if steps is None:
                if self.commitment is not None:
                    goal, terminal, target, label = self.commitment
                    if target is None or self.map.label(target) == label:
                        steps = route(
                            self.map,
                            obs.position,
                            obs.direction,
                            goals={goal},
                            has_key=obs.carrying == "key",
                            terminal=terminal,
                        )
                    if not steps:
                        self.commitment = None
                if not steps:
                    steps, _ = make_plan(obs, self.map, self.scanned)
                    last = steps[-1]
                    pos, direction = last.position, last.direction
                    terminal = last.action if last.action in {"pickup", "drop", "toggle"} else None
                    target = None
                    if terminal:
                        target = (pos[0] + DIRS[direction][0], pos[1] + DIRS[direction][1])
                    elif last.action == "forward":
                        pos = (pos[0] + DIRS[direction][0], pos[1] + DIRS[direction][1])
                    elif last.action in {"left", "right"}:
                        direction = (direction + (-1 if last.action == "left" else 1)) % 4
                    self.commitment = ((pos, direction), terminal, target, self.map.label(target))
            self._compile(obs, steps)
            self.stats["planning_ms"] += (perf_counter_ns() - start) / 1e6
        else:
            self.stats["plan_steps_reused"] += 1
        self.stage = obs.stage
        step = self.plan[self.plan_index]
        self.plan_index += 1
        self.last_target = (
            obs.position[0] + DIRS[obs.direction][0],
            obs.position[1] + DIRS[obs.direction][1],
        )
        if self.reacquiring:
            self.stats["reacquisition_actions"] += 1
        return step.action

    def metrics(self):
        return {
            **self.stats,
            **(self.map.graph.metrics() if self.indexed_support else self.map.metrics()),
        }
