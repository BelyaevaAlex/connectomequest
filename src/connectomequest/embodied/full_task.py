"""Full-task policy controls sharing observations, belief semantics and guard.

No class here receives an environment or hidden-grid reference. Frontier policy
uses separate high-level selection, but shares motion planning/recovery code.
"""

import hashlib
from time import perf_counter_ns

from connectomequest.embodied.evidence_revision import FrameWithdrawal
from connectomequest.embodied.navigation import DIRS, PASSABLE, PlanStep, adjacent_goals, route
from connectomequest.embodied.perturbations import view_to_world
from connectomequest.embodied.recheck_policy import BindingMap, ChoiceAgent, supported_plan
from connectomequest.embodied.sequence_recovery import RevisitAgent

METHODS = ("current", "full_replan", "frontier")
CONDITIONS = ("clean", "revision_short")
WINDOWS = ((8, 11), (40, 43))


class ProjectionChannel:
    """Harness-owned fault schedule; the policy sees predictions, not the regime."""

    def __init__(self, perceptor):
        self.perceptor = perceptor
        self.shift = 0
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        return [
            (x + self.shift, y, label, confidence)
            for x, y, label, confidence in self.perceptor.predict(rgb)
        ]


class FrontierPolicy:
    def __init__(self, perceptor):
        self.perceptor = perceptor
        self.map = BindingMap()
        self.scanned = set()
        self.visited = {}
        self.last_context = None
        self.rotation_count = 0
        self.last_target = None
        self.plan = []
        self.plan_index = 0
        self.commitment = None
        self.stats = {
            "rotation_recoveries": 0,
            "recovery_routes_considered": 0,
            "recovery_steps_planned": 0,
            "planning_ms": 0.0,
            "inference_ms": 0.0,
            "plans_built": 0,
            "failed_acknowledgements": 0,
        }

    def _to_object(self, obs, positions, terminal):
        return route(
            self.map,
            obs.position,
            obs.direction,
            goals=adjacent_goals(positions, self.map),
            has_key=obs.carrying == "key",
            terminal=terminal,
        )

    def choose_plan(self, obs):
        target = "box:" + obs.goals[obs.stage]
        cells = self.map.cells
        targets = [p for p, l in cells.items() if l == target]
        doors = {p for p, l in cells.items() if l.startswith("door_")}
        neighbours = {(p[0] + dx, p[1] + dy) for p in doors for dx, dy in DIRS}
        target_path = self._to_object(obs, targets, "pickup") if targets else None
        # Pick-up requires a free hand; retain a key while a locked door is needed.
        if obs.carrying and (
            obs.carrying != "key"
            or (target_path and not any(s.action == "toggle" for s in target_path))
        ):
            avoid = {p for s in (target_path or []) for p, _ in s.requirements} | doors | neighbours
            slots = [
                p for p, l in cells.items() if l == "empty" and p != obs.position and p not in avoid
            ]
            path = self._to_object(obs, slots, "drop")
            if path:
                return path
        if targets and obs.carrying is None and target_path:
            return target_path
        if obs.carrying == "key" and target_path and any(s.action == "toggle" for s in target_path):
            # Follow navigation/door opening, not an impossible pickup while carrying.
            first_toggle = next(i for i, s in enumerate(target_path) if s.action == "toggle")
            return target_path[: first_toggle + 1]
        if obs.carrying is None:
            keys = [p for p, l in cells.items() if l.startswith("key:")]
            # Do not repeatedly reacquire a discarded key when no locked door remains known.
            if keys and (any(l == "door_locked" for l in cells.values()) or not doors):
                path = self._to_object(obs, keys, "pickup")
                if path:
                    return path
        openable = [
            p
            for p in doors
            if cells[p] == "door_closed" or (cells[p] == "door_locked" and obs.carrying == "key")
        ]
        if openable:
            path = self._to_object(obs, openable, "toggle")
            if path:
                return path
        # Clear an observed object obstructing a doorway, using public labels only.
        if obs.carrying is None:
            blockers = [
                p
                for p, l in cells.items()
                if p in neighbours and l.startswith("box:") and l != target
            ]
            path = self._to_object(obs, blockers, "pickup") if blockers else None
            if path:
                return path
        goals = {
            (p, d)
            for p, l in cells.items()
            if l in PASSABLE
            for d in range(4)
            if (p, d) not in self.scanned
            and any((p[0] + dx, p[1] + dy) not in cells for dx, dy in DIRS)
        }
        path = (
            route(self.map, obs.position, obs.direction, goals=goals, has_key=obs.carrying == "key")
            if goals
            else None
        )
        return path or [PlanStep("right", obs.position, obs.direction, ())]

    def act(self, obs):
        if obs.last_action and not obs.acknowledged and self.last_target is not None:
            self.map.invalidate([self.last_target])
            self.stats["failed_acknowledgements"] += 1
        t = perf_counter_ns()
        predictions = self.perceptor.predict(obs.rgb)
        self.stats["inference_ms"] += (perf_counter_ns() - t) / 1e6
        items = [
            (view_to_world(obs.position, obs.direction, x, y), label)
            for x, y, label, c in predictions
            if label != "unseen" and c >= 0.45
        ]
        items.append((obs.position, "empty"))
        self.map.observe(obs.receipt_id, hashlib.sha256(obs.rgb.tobytes()).hexdigest(), items)
        self.scanned.add((obs.position, obs.direction))
        self.visited[obs.position, obs.direction] = obs.step
        context = (obs.position, obs.stage, obs.carrying)
        self.rotation_count = (
            self.rotation_count + 1
            if context == self.last_context and obs.last_action in ("left", "right")
            else 0
        )
        self.last_context = context
        t = perf_counter_ns()
        self.plan = self.choose_plan(obs)
        self.plan_index = 1
        self.stats["plans_built"] += 1
        if self.rotation_count >= 8:
            recovery = RevisitAgent.recovery_route(self, obs)
            if recovery:
                self.plan = recovery
                self.stats["rotation_recoveries"] += 1
                self.stats["recovery_steps_planned"] += len(recovery)
            self.rotation_count = 0
        self.stats["planning_ms"] += (perf_counter_ns() - t) / 1e6
        self.last_target = (
            obs.position[0] + DIRS[obs.direction][0],
            obs.position[1] + DIRS[obs.direction][1],
        )
        return self.plan[0].action


class MatchedController:
    """Common harness: fault exposure, public messages and final action guard."""

    def __init__(self, perceptor, method, condition):
        if method not in METHODS or condition not in CONDITIONS:
            raise ValueError("unknown configuration")
        self.channel = ProjectionChannel(perceptor)
        self.method = method
        self.condition = condition
        self.agent = (
            ChoiceAgent(self.channel, "replan")
            if method == "current"
            else RevisitAgent(self.channel, "replan")
            if method == "full_replan"
            else FrontierPolicy(self.channel)
        )
        self.agent.map = BindingMap()
        self.frames = {}
        self.events = []
        self.corrupted_frames = 0
        self.guard_rejections = 0
        self.last_requirements = ()
        self.last_certificate = ()
        self.last_proposal = None

    def act(self, obs):
        self.channel.shift = int(
            self.condition != "clean" and any(a <= obs.step <= b for a, b in WINDOWS)
        )
        self.corrupted_frames += self.channel.shift
        events = []
        if self.condition != "clean":
            for a, b in WINDOWS:
                if obs.step == b + 3:
                    event = FrameWithdrawal(
                        tuple(self.frames[t] for t in range(a, b + 1) if t in self.frames), obs.step
                    )
                    events.append(event)
                    self.events.append({"step": obs.step, "interpretations": event.interpretations})
        self.frames[obs.step] = obs.receipt_id
        calls = self.channel.calls
        if self.method == "current":
            action = self.agent.act(obs, events)
        else:
            for event in events:
                self.agent.map.withdraw(event, obs.step)
            action = self.agent.act(obs)
        assert self.channel.calls == calls + 1, "one inference per actual step"
        self.last_proposal = action
        pending = self.agent.plan[self.agent.plan_index - 1 :]
        proposed = (
            pending[0]
            if pending and pending[0].action == action
            else PlanStep(action, obs.position, obs.direction, ())
        )
        legal = supported_plan(self.agent.map, obs, [proposed])
        if legal is None:
            self.guard_rejections += 1
            legal = [PlanStep("right", obs.position, obs.direction, ())]
            self.agent.plan = legal
            self.agent.plan_index = 1
            self.agent.commitment = None
            if hasattr(self.agent, "goal"):
                self.agent.goal = None
        self.last_requirements = legal[0].requirements
        self.last_certificate = self.agent.map.certificate(self.last_requirements)
        if not self.agent.map.verify(self.last_requirements, self.last_certificate):
            raise AssertionError("unsupported accepted action")
        return legal[0].action

    def metrics(self):
        return {
            "guard_rejections": self.guard_rejections,
            "perception_calls": self.channel.calls,
            "corrupted_projection_frames": self.corrupted_frames,
            "revision_events": self.events,
            "rotation_recoveries": self.agent.stats["rotation_recoveries"],
        }
