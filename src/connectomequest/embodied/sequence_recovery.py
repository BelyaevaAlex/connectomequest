"""Common rotation-cycle escape for all v1 symbolic controllers.

This is a public-history liveness heuristic, not a receipt-specific advantage or
a completeness guarantee. It never reads fault schedules or simulator labels.
"""

from connectomequest.embodied.dependent_agent import DependentAgent
from connectomequest.embodied.navigation import route


class RevisitAgent(DependentAgent):
    def __init__(self, perceptor, mode, rotation_limit=8):
        super().__init__(perceptor, mode)
        self.rotation_limit = rotation_limit
        self.rotation_count = 0
        self.last_context = None
        self.visited = {}
        self.stats.update(
            rotation_recoveries=0, recovery_routes_considered=0, recovery_steps_planned=0
        )

    def recovery_route(self, obs):
        choices = []
        for (position, direction), last_seen in self.visited.items():
            if position == obs.position:
                continue
            self.stats["recovery_routes_considered"] += 1
            steps = route(
                self.map,
                obs.position,
                obs.direction,
                goals={(position, direction)},
                has_key=obs.carrying == "key",
            )
            if steps:
                choices.append((last_seen, len(steps), position, direction, steps))
        return min(choices, key=lambda row: row[:4])[-1] if choices else None

    def act(self, obs):
        context = (obs.position, obs.stage, obs.carrying)
        if context == self.last_context and obs.last_action in {"left", "right"}:
            self.rotation_count += 1
        else:
            self.rotation_count = 0
        self.last_context = context
        self.visited[obs.position, obs.direction] = obs.step
        counters_before = {
            key: self.stats[key] for key in ("plan_steps_reused", "reacquisition_actions")
        }
        action = super().act(obs)
        if self.rotation_count < self.rotation_limit:
            return action
        # The ordinary controller has already incorporated this public frame.
        # Its proposed action has NOT yet been executed in the environment.
        from time import perf_counter_ns

        started = perf_counter_ns()
        steps = self.recovery_route(obs)
        if steps:
            # The base proposal was not executed: do not count it as reuse/probing.
            for key, value in counters_before.items():
                self.stats[key] = value
            self._compile(obs, steps)
            # route's final pose is the selected previously observed vantage.
            last = steps[-1]
            from connectomequest.embodied.navigation import DIRS

            position, direction = last.position, last.direction
            if last.action == "forward":
                position = (position[0] + DIRS[direction][0], position[1] + DIRS[direction][1])
            elif last.action in {"left", "right"}:
                direction = (direction + (-1 if last.action == "left" else 1)) % 4
            self.commitment = ((position, direction), None, None, None)
            action = self.plan[0].action
            self.plan_index = 1
            self.reacquiring = False
            self.stats["rotation_recoveries"] += 1
            self.stats["recovery_steps_planned"] += len(steps)
        self.rotation_count = 0
        self.stats["planning_ms"] += (perf_counter_ns() - started) / 1e6
        return action
