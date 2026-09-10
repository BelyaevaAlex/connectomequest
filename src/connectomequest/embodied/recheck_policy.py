"""Choose supported continuation, observed alternative, or paid re-observation.

All geometry and labels come from acquired public history. Optimistic planning
with a withdrawn old label is explicitly hypothetical, never execution support.
"""

from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter_ns

from connectomequest.embodied.active_recheck import ActiveRecheckAgent, RecheckMap
from connectomequest.embodied.navigation import DIRS, PASSABLE, PlanStep, adjacent_goals, route

METHODS = ("replan", "always_recheck", "supported_first", "cost_aware")


@dataclass(frozen=True)
class BindingWithdrawal:
    bindings: tuple
    issued_step: int


class BindingMap(RecheckMap):
    """Revoke interpreted cell bindings without withdrawing an entire RGB frame."""

    def __init__(self):
        super().__init__()
        self.inactive_bindings = set()

    def _rebuild_position(self, position):
        self.cells.pop(position, None)
        self.sources.pop(position, None)
        for identity, (_, items) in reversed(list(self.frames.items())):
            if (
                identity not in self.inactive
                and (identity, position) not in self.inactive_bindings
                and position in items
            ):
                self._assign(position, items[position], identity)
                return

    def withdraw(self, event, current_step):
        if not isinstance(event, BindingWithdrawal):
            super().withdraw(event, current_step)
            for p in {p for _, p in self.inactive_bindings}:
                self._rebuild_position(p)
            return
        if not 0 <= event.issued_step <= current_step:
            raise ValueError("Future/invalid binding withdrawal")
        for identity, position in event.bindings:
            if identity.startswith(("pose:", "action-failure:")) or identity not in self.frames:
                raise ValueError("Unknown or non-visual interpretation")
            if position not in self.frames[identity][1]:
                raise ValueError("Binding never issued")
        new = set(event.bindings) - self.inactive_bindings
        if not new:
            return
        started = perf_counter_ns()
        positions = {p for _, p in new}
        before = {p: self.label(p) for p in positions}
        self.inactive_bindings.update(new)
        for p in positions:
            self._rebuild_position(p)
        self.stats["revision_events"] += 1
        self.stats["revision_changed_cells"] += sum(
            self.label(p) != label for p, label in before.items()
        )
        self.stats["revision_ns"] += perf_counter_ns() - started

    def verify(self, requirements, certificate):
        return super().verify(requirements, certificate) and all(
            (token.source, token.position) not in self.inactive_bindings for token in certificate
        )


def withdrawal_for_position(belief, position, step):
    """Only known active visual interpretations; never revoke trusted pose/barriers."""
    bindings = tuple(
        (identity, position)
        for identity, (_, items) in belief.frames.items()
        if identity not in belief.inactive
        and (identity, position) not in belief.inactive_bindings
        and not identity.startswith(("pose:", "action-failure:"))
        and position in items
    )
    return BindingWithdrawal(bindings, step)


@dataclass(frozen=True)
class Goal:
    position: tuple
    direction: int
    terminal: str | None
    target: tuple | None
    label: str | None
    stage: int


def goal_of(steps, stage):
    if not steps:
        return None
    last = steps[-1]
    pos, direction = last.position, last.direction
    if last.action in ("pickup", "drop", "toggle"):
        target = (pos[0] + DIRS[direction][0], pos[1] + DIRS[direction][1])
        return Goal(pos, direction, last.action, target, dict(last.requirements).get(target), stage)
    if last.action == "forward":
        pos = (pos[0] + DIRS[direction][0], pos[1] + DIRS[direction][1])
    elif last.action in ("left", "right"):
        direction = (direction + (-1 if last.action == "left" else 1)) % 4
    return Goal(pos, direction, None, None, None, stage)


def supported_plan(belief, obs, steps):
    """Validate pose sequence and symbolic effects, not stale literal synonyms.

    A verified door toggle predicts open-door passability for later planning.
    At execution each immediate action still needs newly current map support.
    Returns a normalized plan or None; never fills unknown cells with old labels.
    """
    if not steps:
        return None
    position, direction, carrying = obs.position, obs.direction, obs.carrying
    effects = {}
    normalized = []
    for step in steps:
        if (step.position, step.direction) != (position, direction):
            return None
        front = (position[0] + DIRS[direction][0], position[1] + DIRS[direction][1])
        label = effects.get(front, belief.label(front))
        requirements = ()
        if step.action == "forward":
            if label not in PASSABLE:
                return None
            requirements = ((front, label),)
            position = front
        elif step.action in ("left", "right"):
            direction = (direction + (-1 if step.action == "left" else 1)) % 4
        elif step.action == "toggle":
            if label not in ("door_closed", "door_locked") or (
                label == "door_locked" and carrying != "key"
            ):
                return None
            requirements = ((front, label),)
            effects[front] = "door_open"
        elif step.action == "pickup":
            if (
                carrying is not None
                or label is None
                or not label.startswith(("key:", "box:", "ball:"))
            ):
                return None
            expected = dict(step.requirements).get(front)
            if expected is not None and label != expected:
                return None
            requirements = ((front, label),)
            carrying = label.split(":")[0]
            effects[front] = "empty"
        elif step.action == "drop":
            if carrying is None or label != "empty":
                return None
            requirements = ((front, label),)
            effects[front] = "occupied-by-dropped-object"
            carrying = None
        else:
            return None
        normalized.append(replace(step, requirements=requirements))
    return normalized


def route_to_goal(belief, obs, goal):
    if goal is None or goal.stage != obs.stage:
        return None
    if goal.terminal and belief.label(goal.target) != goal.label:
        return None
    steps = route(
        belief,
        obs.position,
        obs.direction,
        goals={(goal.position, goal.direction)},
        has_key=obs.carrying == "key",
        terminal=goal.terminal,
    )
    return supported_plan(belief, obs, steps)


class PlanningHypothesis:
    """Labels-only planning view. Deliberately has no certificate/issuance API."""

    def __init__(self, belief):
        self.cells = dict(belief.cells)

    def label(self, position):
        return self.cells.get(position)


def choose_option(method, continuation, alternative, candidates):
    """The same options/costs feed every policy. Estimates are NOT calibrated VOI."""
    if method not in METHODS:
        raise ValueError(method)
    nearest = min(candidates, key=lambda c: (c["cost"], c["position"])) if candidates else None
    if method == "always_recheck" and nearest:
        return "recheck", nearest
    if method != "replan" and continuation:
        return "continue", continuation
    if method == "cost_aware" and alternative:
        useful = [
            c
            for c in candidates
            if c["occurrences"] > 0
            and c["completion_estimate"] is not None
            and c["completion_estimate"] < len(alternative)
        ]
        if useful:
            return "recheck", min(
                useful, key=lambda c: (c["completion_estimate"], c["cost"], c["position"])
            )
    if alternative:
        return "alternative", alternative
    if method != "replan":
        relevant = [c for c in candidates if c["occurrences"] > 0]
        if relevant:
            return "recheck", min(relevant, key=lambda c: (c["cost"], c["position"]))
    return "fallback", None


class ChoiceAgent(ActiveRecheckAgent):
    def __init__(self, perceptor, strategy):
        if strategy not in METHODS:
            raise ValueError(strategy)
        super().__init__(perceptor, "history_scan")
        self.strategy = strategy
        self.map = BindingMap()
        self.goal = None
        self.remembered = {}
        self.processed_receipt = None
        self.decisions = []
        self.stats.update(
            choice_checks=0,
            skipped_rechecks=0,
            continued_plans=0,
            alternative_routes=0,
            choice_recheck_actions=0,
            choice_ms=0.0,
            choice_guard_rejections=0,
        )

    def act(self, obs, withdrawals=()):
        started = perf_counter_ns()
        remaining = list(self.plan[self.plan_index :])
        previous_target = self.last_target
        if self.goal is not None:
            done = (
                obs.stage != self.goal.stage
                or (
                    self.goal.terminal is None
                    and (obs.position, obs.direction) == (self.goal.position, self.goal.direction)
                )
                or (
                    obs.acknowledged
                    and obs.last_action == self.goal.terminal
                    and previous_target == self.goal.target
                )
            )
            if done:
                self.goal = None
                self.remembered.clear()
        if self.goal is None:
            self.goal = goal_of(remaining, obs.stage)
        counts = Counter(p for s in remaining for p, _ in s.requirements)
        before = dict(self.map.cells)
        for event in withdrawals:
            self.map.withdraw(event, obs.step)
        if any(self.map.label(p) != label for p, label in before.items()):
            self.stats["support_loss_events"] += 1
        for p, old_label in before.items():
            if self.map.label(p) is None:
                self.remembered[p] = {
                    "position": p,
                    "label": old_label,
                    "occurrences": counts[p],
                    "attempts": 0,
                }
        if obs.receipt_id == self.processed_receipt:
            # Snapshot already contains this processed public frame. Do not
            # reinsert withdrawn bindings by reprocessing the same image.
            fallback = None
        else:
            super().act(obs)  # Common once-per-new-frame perception and recovery.
            fallback = list(self.plan[self.plan_index - 1 :])
            self.processed_receipt = obs.receipt_id
        for p, old_label in before.items():
            if self.map.label(p) is None and p not in self.remembered:
                self.remembered[p] = {
                    "position": p,
                    "label": old_label,
                    "occurrences": counts[p],
                    "attempts": 0,
                }
        if self.goal is None:
            self.goal = goal_of(fallback or remaining, obs.stage)
        for p in list(self.remembered):
            if self.map.label(p) is not None:
                del self.remembered[p]
        # A route to a sensing vantage is not a supported continuation of the
        # task goal. Reconsider it as soon as new evidence enables the goal.
        continuation = (
            supported_plan(self.map, obs, remaining)
            if goal_of(remaining, obs.stage) == self.goal
            else None
        )
        alternative = route_to_goal(self.map, obs, self.goal)
        candidates = []
        for p, item in sorted(self.remembered.items()):
            if item["attempts"] >= 2:
                continue
            path = route(
                self.map,
                obs.position,
                obs.direction,
                goals=adjacent_goals([p], self.map),
                has_key=obs.carrying == "key",
            )
            fresh_frame_turn = path == []
            if path == []:
                # A repeated computation is not a new observation. Pay two turns
                # to obtain a new frame at the original viewing direction.
                path = [
                    PlanStep("right", obs.position, obs.direction, ()),
                    PlanStep("left", obs.position, (obs.direction + 1) % 4, ()),
                ]
            path = supported_plan(self.map, obs, path)
            if not path:
                continue
            vantage = goal_of(path, obs.stage)
            imagined_obs = replace(obs, position=vantage.position, direction=vantage.direction)
            imagined_map = PlanningHypothesis(self.map)
            # Old public label only. This counterfactual never enters real map,
            # a certificate, an executed action or the planner's fallback.
            imagined_map.cells[p] = item["label"]
            for step in path:
                if step.action == "toggle":
                    front = (
                        step.position[0] + DIRS[step.direction][0],
                        step.position[1] + DIRS[step.direction][1],
                    )
                    imagined_map.cells[front] = "door_open"
            tail = route_to_goal(imagined_map, imagined_obs, self.goal)
            estimate = len(path) + len(tail) if tail else None
            if len(path) <= obs.budget - obs.step:
                candidates.append(
                    {
                        **item,
                        "path": path,
                        "cost": len(path),
                        "fresh_frame_turn": fresh_frame_turn,
                        "completion_estimate": estimate,
                    }
                )
        option, selected = choose_option(self.strategy, continuation, alternative, candidates)
        if option == "recheck":
            steps = selected["path"]
            if selected["fresh_frame_turn"]:
                self.remembered[selected["position"]]["attempts"] += 1
            self.stats["choice_recheck_actions"] += 1
        elif option == "continue":
            steps = selected
            self.stats["continued_plans"] += 1
            self.stats["skipped_rechecks"] += bool(candidates)
        elif option == "alternative":
            steps = selected
            self.stats["alternative_routes"] += 1
            self.stats["skipped_rechecks"] += bool(candidates)
        else:
            steps = supported_plan(self.map, obs, fallback)
            if not steps:
                from connectomequest.embodied.navigation import make_plan

                steps, _ = make_plan(obs, self.map, self.scanned)
        self.stats["choice_checks"] += 1
        self.decisions.append(
            {
                "step": obs.step,
                "option": option,
                "continuation_cost": len(continuation) if continuation else None,
                "alternative_cost": len(alternative) if alternative else None,
                "selected_target": selected["position"] if option == "recheck" else None,
                "candidates": [{k: v for k, v in c.items() if k != "path"} for c in candidates],
            }
        )
        first = supported_plan(self.map, obs, steps[:1])
        if first is None:
            first = [PlanStep("right", obs.position, obs.direction, ())]
            steps = first
            self.stats["choice_guard_rejections"] += 1
        else:
            steps = first + steps[1:]
        self._compile(obs, steps)
        self.plan_index = 1
        self.stage = obs.stage
        self.last_target = (
            obs.position[0] + DIRS[obs.direction][0],
            obs.position[1] + DIRS[obs.direction][1],
        )
        self.last_requirements = steps[0].requirements
        self.last_certificate = self.map.certificate(self.last_requirements)
        assert self.map.verify(self.last_requirements, self.last_certificate)
        self.stats["choice_ms"] += (perf_counter_ns() - started) / 1e6
        return steps[0].action
