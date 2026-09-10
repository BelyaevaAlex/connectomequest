"""Finite-model, costed certificate acquisition; no policy receives the actual world.

The public prior is intentionally richer than native connectome observations.
This is a controlled planning model, not a disguised full-graph baseline.
"""

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

METHODS = (
    "random",
    "cost_greedy",
    "myopic_goal_value",
    "commitment",
    "certificate_lookahead",
    "evidence_lookahead",
    "exact",
)


def random_success(n, m, k):
    if not 0 <= m <= n or not 0 <= k <= n or n < 1:
        raise ValueError("require N>=1, 0<=M,k<=N")
    return 1 - (math.comb(n - m, k) if k <= n - m else 0) / math.comb(n, k)


def acquired(witnesses, positive):
    return int(any(w & positive == w for w in witnesses))


@dataclass(frozen=True)
class Action:
    reveal: int
    cost: int
    location: int


@dataclass(frozen=True)
class Problem:
    n: int
    witnesses: tuple[int, ...]
    actions: tuple[Action, ...]
    worlds: tuple[int, ...]
    probabilities: tuple[float, ...]
    weights: tuple[float, ...]
    travel: int

    def __post_init__(self):
        if not self.worlds or len(self.worlds) != len(self.probabilities):
            raise ValueError("empty or inconsistent prior")
        if abs(sum(self.probabilities) - 1) > 1e-9 or any(p <= 0 for p in self.probabilities):
            raise ValueError("invalid prior")
        if any(a.cost <= 0 or not 0 < a.reveal < 2**self.n for a in self.actions):
            raise ValueError("invalid observation action")
        if self.travel < 0 or any(not 0 < w < 2**self.n for w in self.witnesses):
            raise ValueError("invalid witness or travel cost")


def make_problem(family, batch, travel, seed):
    clauses = {"2pt": (7, 56), "2i": (3, 12, 48), "shared": (7, 25, 49)}
    if family not in clauses or batch not in (1, 2):
        raise ValueError("unknown family/batch")
    # Batch/travel treatments do not change the hidden-world distribution.
    rng = random.Random(f"{family}:{seed}")
    probs = tuple(rng.choice((0.2, 0.5, 0.8)) for _ in range(6))
    costs = tuple(rng.choice((1, 2)) for _ in range(6))
    worlds = []
    masses = []
    for world in range(64):
        if not acquired(clauses[family], world):
            continue
        mass = math.prod(probs[i] if world & (1 << i) else 1 - probs[i] for i in range(6))
        worlds.append(world)
        masses.append(mass)
    total = sum(masses)
    actions = tuple(
        Action(sum(1 << j for j in range(i, i + batch)), max(costs[i : i + batch]), i // 2 + 1)
        for i in range(0, 6, batch)
    )
    return Problem(
        6, clauses[family], actions, tuple(worlds), tuple(p / total for p in masses), probs, travel
    )


class Solver:
    def __init__(self, problem):
        self.p = problem
        self.nodes = defaultdict(int)
        # Instance-local caches, not global lru caches retaining prior problems.
        for name in ("posterior", "outcomes", "value", "plan", "choose", "look"):
            setattr(self, name, lru_cache(maxsize=None)(getattr(self, "_" + name)))

    def clear(self):
        for name in ("posterior", "outcomes", "value", "plan", "choose", "look"):
            getattr(self, name).cache_clear()

    def _posterior(self, positive, negative):
        self.nodes["posterior"] += 1
        if positive & negative:
            return ()
        entries = [
            (w, p)
            for w, p in zip(self.p.worlds, self.p.probabilities)
            if w & positive == positive and not w & negative
        ]
        z = sum(p for _, p in entries)
        return tuple((w, p / z) for w, p in entries) if z else ()

    def cost(self, index, location):
        a = self.p.actions[index]
        return a.cost + self.p.travel * abs(location - a.location)

    def legal(self, positive, negative, location, budget):
        relevant = 0
        for witness in self.p.witnesses:
            if not witness & negative:
                relevant |= witness
        unknown = relevant & ~(positive | negative)
        return tuple(
            i
            for i, a in enumerate(self.p.actions)
            if a.reveal & unknown and self.cost(i, location) <= budget
        )

    def _outcomes(self, positive, negative, index):
        self.nodes["observation_model"] += 1
        mask = self.p.actions[index].reveal
        groups = defaultdict(float)
        for world, prob in self.posterior(positive, negative):
            groups[world & mask] += prob
        return tuple(
            (prob, positive | bits, negative | (mask & ~bits))
            for bits, prob in sorted(groups.items())
        )

    def _value(self, positive, negative, location, budget):
        self.nodes["exact_value"] += 1
        if acquired(self.p.witnesses, positive):
            return 1.0
        return max(
            (
                sum(
                    prob
                    * self.value(p, n, self.p.actions[i].location, budget - self.cost(i, location))
                    for prob, p, n in self.outcomes(positive, negative, i)
                )
                for i in self.legal(positive, negative, location, budget)
            ),
            default=0.0,
        )

    def _plan(self, positive, location, witness):
        """Minimum deterministic cost to acquire this witness assuming its facts true."""
        self.nodes["witness_cost"] += 1
        missing = witness & ~positive
        if not missing:
            return (0, -1)
        choices = []
        for i, a in enumerate(self.p.actions):
            if a.reveal & missing:
                rest, _ = self.plan(positive | a.reveal, a.location, witness)
                choices.append((self.cost(i, location) + rest, i))
        return min(choices) if choices else (math.inf, -1)

    def joint(self, positive, negative, witness):
        return sum(p for w, p in self.posterior(positive, negative) if w & witness == witness)

    def utility(self, positive, negative):
        """Myopic max-product goal utility; no claim of an ActiveVOO reproduction."""
        if acquired(self.p.witnesses, positive):
            return 1.0
        posterior = self.posterior(positive, negative)
        if not posterior:
            return 0.0
        marginal = [sum(p for w, p in posterior if w & (1 << j)) for j in range(self.p.n)]
        return max(
            (
                math.prod(
                    marginal[j]
                    for j in range(self.p.n)
                    if witness & (1 << j) and not positive & (1 << j)
                )
                for witness in self.p.witnesses
                if not witness & negative
            ),
            default=0.0,
        )

    def leaf(self, method, positive, negative, location, budget):
        if acquired(self.p.witnesses, positive):
            return 1.0
        if method == "evidence_lookahead":
            return positive.bit_count() / self.p.n
        return max(
            (
                self.joint(positive, negative, w)
                for w in self.p.witnesses
                if not w & negative and self.plan(positive, location, w)[0] <= budget
            ),
            default=0.0,
        )

    def _look(self, method, positive, negative, location, budget, depth):
        self.nodes["lookahead_value"] += 1
        if acquired(self.p.witnesses, positive):
            return 1.0
        if depth == 0:
            return self.leaf(method, positive, negative, location, budget)
        return max(
            (
                sum(
                    prob
                    * self.look(
                        method,
                        p,
                        n,
                        self.p.actions[i].location,
                        budget - self.cost(i, location),
                        depth - 1,
                    )
                    for prob, p, n in self.outcomes(positive, negative, i)
                )
                for i in self.legal(positive, negative, location, budget)
            ),
            default=0.0,
        )

    def _choose(self, method, positive, negative, location, budget, seed=17):
        self.nodes["decisions"] += 1
        legal = self.legal(positive, negative, location, budget)
        if acquired(self.p.witnesses, positive) or not legal:
            return None
        if method == "random":
            return min(
                legal,
                key=lambda i: hashlib.sha256(
                    f"{seed}:{positive}:{negative}:{location}:{budget}:{i}".encode()
                ).digest(),
            )
        if method == "commitment":
            plans = []
            for w in self.p.witnesses:
                if w & negative:
                    continue
                c, i = self.plan(positive, location, w)
                if i in legal:
                    plans.append((int(c <= budget), self.joint(positive, negative, w), -c, -i, i))
            if plans:
                return max(plans)[-1]
        scored = []
        for i in legal:
            c = self.cost(i, location)
            dest = self.p.actions[i].location
            outcomes = self.outcomes(positive, negative, i)
            if method == "exact":
                score = sum(prob * self.value(p, n, dest, budget - c) for prob, p, n in outcomes)
            elif method in ("certificate_lookahead", "evidence_lookahead"):
                score = sum(
                    prob * self.look(method, p, n, dest, budget - c, 1) for prob, p, n in outcomes
                )
            elif method == "myopic_goal_value":
                fresh = self.p.actions[i].reveal & ~(positive | negative)
                score = (
                    sum(prob * self.utility(p, n) for prob, p, n in outcomes)
                    - self.utility(positive, negative | fresh)
                ) / c
            elif method in ("cost_greedy", "commitment"):
                score = sum(prob * (p & ~positive).bit_count() for prob, p, n in outcomes) / c
            else:
                raise ValueError(method)
            scored.append((round(score, 14), -c, -i, i))
        return max(scored)[-1]


def run_world(solver, method, world, budget, seed=17):
    """Evaluator owns actual truth. choose() receives only public acquired state."""
    positive = negative = location = cost = 0
    trace = []
    while not acquired(solver.p.witnesses, positive):
        i = solver.choose(method, positive, negative, location, budget - cost, seed)
        if i is None:
            break
        assert i in solver.legal(positive, negative, location, budget - cost)
        a = solver.p.actions[i]
        paid = solver.cost(i, location)
        bits = world & a.reveal
        positive |= bits
        negative |= a.reveal & ~bits
        cost += paid
        location = a.location
        trace.append({"action": i, "positive_observation": bits, "charged": paid})
    witness = next((w for w in solver.p.witnesses if w & positive == w), None)
    # Replay independently, including cost and polarity; no posterior-as-evidence.
    rp = rn = rl = rc = 0
    for t in trace:
        a = solver.p.actions[t["action"]]
        assert t["positive_observation"] == world & a.reveal
        assert t["charged"] == a.cost + solver.p.travel * abs(rl - a.location)
        rp |= t["positive_observation"]
        rn |= a.reveal & ~t["positive_observation"]
        rc += t["charged"]
        rl = a.location
    assert (rp, rn, rl, rc) == (positive, negative, location, cost) and cost <= budget
    assert witness is None or (world & witness == witness and positive & witness == witness)
    return {
        "success": witness is not None,
        "cost": cost,
        "positive": positive,
        "negative": negative,
        "observations": len(trace),
        "trace": trace,
        "certificate": witness,
        "illegal_actions": 0,
        "exact_replay": True,
    }
