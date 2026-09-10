"""Public-evidence rechecking; no simulator state or fault schedule is accessible."""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns

from connectomequest.embodied.evidence_revision import RevisionMap
from connectomequest.embodied.navigation import PlanStep, adjacent_goals, route
from connectomequest.embodied.sequence_recovery import RevisitAgent

METHODS = ("reset", "history_scan", "nearest", "dependency")


@dataclass(frozen=True)
class SupportToken:
    position: tuple
    label: str
    source: str
    digest: tuple


class RecheckMap(RevisionMap):
    """Shared retained history and current-support verifier for ALL methods.

    Interpretation withdrawal may reveal older support. A failed-action barrier
    instead masks ALL earlier support at that cell; it cannot itself be withdrawn.
    Neither operation claims that an unobserved physical world stayed unchanged.
    """

    def __init__(self):
        super().__init__("history_scan")
        self.visible = set()
        self.contradictions = set()

    def observe(self, receipt, digest, items):
        items = list(items)
        self.visible = {p for p, _ in items}
        self.contradictions = {
            p for p, label in items if self.label(p) is not None and self.label(p) != label
        }
        return super().observe(receipt, digest, items)

    def certificate(self, requirements):
        requirements = tuple(requirements)
        if any(dict(requirements)[p] != label for p, label in requirements):
            return None
        tokens = []
        for position, label in dict(requirements).items():
            source = self.sources.get(position)
            if label is None or self.label(position) != label or source is None:
                return None
            tokens.append(SupportToken(position, label, source, self.receipts[source]))
        return tuple(tokens)

    def verify(self, requirements, certificate):
        if certificate is None:
            return False
        requirements = tuple(requirements)
        expected = dict(requirements)
        if any(expected[p] != label for p, label in requirements):
            return False
        if not all(isinstance(t, SupportToken) for t in certificate):
            return False
        if len(certificate) != len(expected):
            return False
        if {t.position for t in certificate} != set(expected):
            return False
        return all(
            t.label is not None
            and expected[t.position] == t.label
            and t.source not in self.inactive
            and self.sources.get(t.position) == t.source
            and self.label(t.position) == t.label
            and self.receipts.get(t.source) == t.digest
            and self.frames.get(t.source, (None, {}))[1].get(t.position) == t.label
            for t in certificate
        )

    def reset_active(self):
        # Retain audit history, but never restore pre-reset facts during withdrawal.
        self.invalidate(list(self.cells))
        self.stats["map_resets"] += 1


def choose_recheck(candidates, method):
    """Same candidates/costs; dependency is the only selection ablation switch.

    Utility is a heuristic, NOT a calibrated expected information gain. Priority
    combines occurrence count and proximity in the invalidated plan suffix.
    Deterministic coordinate tie-breaking is common to both policies.
    """
    if method not in ("nearest", "dependency"):
        raise ValueError(method)
    if not candidates:
        return None

    def key(c):
        priority = (
            1.0 if method == "nearest" else 1.0 + c["occurrences"] + 1.0 / (1 + c["first_step"])
        )
        return (-priority / c["cost"], c["cost"], c["position"])

    return min(candidates, key=key)


class ActiveRecheckAgent(RevisitAgent):
    """Common planner/recovery; selective, paid navigation after support loss."""

    def __init__(self, perceptor, method):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(perceptor, "replan")
        self.method = method
        self.map = RecheckMap()
        self.pending = {}
        self.active_target = None
        self.recheck_trace = []
        self.last_requirements = ()
        self.last_certificate = ()
        self.stats.update(
            support_loss_events=0,
            pending_added=0,
            pending_resolved_by_observation=0,
            pending_resolved_by_alternative=0,
            recheck_decisions=0,
            recheck_navigation_actions=0,
            abandoned_recheck_targets=0,
            public_contradictions=0,
            support_checks=0,
            unsupported_proposals=0,
            unsupported_accepted=0,
            recheck_planning_ms=0.0,
            support_check_ms=0.0,
        )

    def _remember(self, remaining, positions, step):
        counts = Counter(p for s in remaining for p, _ in s.requirements)
        first = {}
        expected = {}
        for i, s in enumerate(remaining):
            for p, label in s.requirements:
                first.setdefault(p, i)
                expected.setdefault(p, label)
        # Include all unknown affected cells for BOTH selectors. Dependency gets
        # no extra candidate access, only publicly computed plan priority.
        for p in sorted(positions):
            if self.map.label(p) is not None:
                continue
            if p not in self.pending:
                self.stats["pending_added"] += 1
            self.pending[p] = {
                "position": p,
                "occurrences": counts[p],
                "first_step": first.get(p, len(remaining)),
                "expected": expected.get(p),
                "created_step": step,
                "stage": self.stage,
                "attempts": 0,
            }

    def _reset_controller(self):
        self.map.reset_active()
        self.scanned.clear()
        self.plan = []
        self.plan_index = 0
        self.commitment = None

    def act(self, obs, withdrawals=()):
        remaining = self.plan[self.plan_index :]
        requirements = [r for s in remaining for r in s.requirements]
        before_labels = dict(self.map.cells)
        previous_target = self.last_target
        for event in withdrawals:
            self.map.withdraw(event, obs.step)
        affected = {p for p, label in before_labels.items() if self.map.label(p) != label}
        failed = obs.last_action is not None and not obs.acknowledged
        if failed and self.last_target is not None:
            # Barrier, not withdrawal: do not resurrect an old "free" cell.
            self.map.invalidate([self.last_target])
            affected.add(self.last_target)
        if affected:
            self.stats["support_loss_events"] += 1
            self._remember(remaining, affected, obs.step)
            if self.method == "reset":
                self._reset_controller()
        # Parent handles RGB inference exactly once, map integration, common
        # planning and rotation-cycle recovery. Proposed action is not executed.
        recovery_before = self.stats["rotation_recoveries"]
        proposal = super().act(obs)
        self.stats["public_contradictions"] += len(self.map.contradictions)
        unexpected = self.map.contradictions & {p for p, _ in requirements}
        if obs.acknowledged and obs.last_action in ("pickup", "drop", "toggle"):
            unexpected.discard(previous_target)  # Expected effects are not faults.
        if self.method == "reset" and unexpected and not affected:
            # Preserve only the CURRENT public frame after clearing the old map.
            keep = [(p, self.map.label(p)) for p in self.map.visible]
            self._reset_controller()
            self.map.record(f"current-after-reset:{obs.receipt_id}", obs.receipt_id, keep)
            from connectomequest.embodied.navigation import make_plan

            steps, _ = make_plan(obs, self.map, self.scanned)
            self._compile(obs, steps)
            self.plan_index = 1
            proposal = steps[0].action
        for p in list(self.pending):
            if self.map.label(p) is not None:
                metric = (
                    "pending_resolved_by_observation"
                    if p in self.map.visible
                    else "pending_resolved_by_alternative"
                )
                self.stats[metric] += 1
                del self.pending[p]
                if self.active_target == p:
                    self.active_target = None
        # Parent has now updated self.stage, so stage-specific entries are
        # compared to the public stage recorded when they were created.
        self.pending = {p: item for p, item in self.pending.items() if item["stage"] == obs.stage}
        if self.active_target not in self.pending:
            self.active_target = None
        if (
            self.method in ("nearest", "dependency")
            and self.pending
            and self.stats["rotation_recoveries"] == recovery_before
        ):
            started = perf_counter_ns()
            candidates = []
            for p, item in sorted(self.pending.items()):
                if item["attempts"] >= 2:
                    continue
                path = route(
                    self.map,
                    obs.position,
                    obs.direction,
                    goals=adjacent_goals([p], self.map),
                    has_key=obs.carrying == "key",
                )
                if path == []:
                    # Arrived, but RGB did not resolve the cell. Do not declare
                    # success or pay zero cost for an uninformative recheck.
                    item["attempts"] += 1
                    self.stats["abandoned_recheck_targets"] += 1
                    if self.active_target == p:
                        self.active_target = None
                elif path:
                    candidates.append({**item, "cost": len(path), "path": path})
            selected = next((c for c in candidates if c["position"] == self.active_target), None)
            if selected is None:
                selected = choose_recheck(candidates, self.method)
                if selected:
                    self.active_target = selected["position"]
                    self.stats["recheck_decisions"] += 1
                    self.recheck_trace.append(
                        {
                            "step": obs.step,
                            "selected": self.active_target,
                            "candidates": [
                                {k: v for k, v in c.items() if k != "path"} for c in candidates
                            ],
                        }
                    )
            if selected:
                self._compile(obs, selected["path"])
                self.plan_index = 1
                self.commitment = None
                proposal = selected["path"][0].action
                self.stats["recheck_navigation_actions"] += 1
            self.stats["recheck_planning_ms"] += (perf_counter_ns() - started) / 1e6
        # Same support guard for ALL methods. Validity is relative to current
        # observed interpretations, never a promise of physical ground truth.
        step = self.plan[self.plan_index - 1]
        assert proposal == step.action
        started = perf_counter_ns()
        certificate = self.map.certificate(step.requirements)
        self.stats["support_checks"] += 1
        if not self.map.verify(step.requirements, certificate):
            self.stats["unsupported_proposals"] += 1
            step = PlanStep("right", obs.position, obs.direction, ())
            proposal = "right"
            certificate = ()
            self._compile(obs, [step])
            self.plan_index = 1
            self.commitment = None
        accepted = self.map.verify(step.requirements, certificate)
        self.stats["unsupported_accepted"] += int(not accepted)
        assert accepted
        self.stats["support_check_ms"] += (perf_counter_ns() - started) / 1e6
        self.last_requirements = step.requirements
        self.last_certificate = certificate
        return proposal
