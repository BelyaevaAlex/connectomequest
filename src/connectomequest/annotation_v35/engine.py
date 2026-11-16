"""Two-goal certification engine for weighted category questions (V35).

A question has N unread cells. Paying cell ``j`` reveals one of ``K`` outcomes;
outcome ``k`` adds the nonnegative integer ``gp[j, k]`` toward a positive goal
``a0`` and ``gn[j, k]`` toward a negative goal ``b0``. The question is certified
positive once the positive goal is met and negative once the negative goal is met.

The weighted-threshold question "does category set A supply at least T of the
total weight W" is the instance with outcomes (member, nonmember, unresolved),
progress ``gp = (w, 0, 0)``, ``gn = (0, w, 0)``, ``a0 = T`` and ``b0 = W - T + 1``:
``L >= T`` is the positive certificate and ``U < T`` (confirmed nonmember weight
at least ``W - T + 1``) the negative one. An unresolved response adds nothing.

Proof arithmetic is exact integer arithmetic over the paid responses only.
Probabilities influence the order of payments, never a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEMBER, NONMEMBER, UNRESOLVED = 0, 1, 2


@dataclass(frozen=True)
class Instance:
    cells: np.ndarray
    weights: np.ndarray
    gp: np.ndarray
    gn: np.ndarray
    a0: int
    b0: int

    def __post_init__(self):
        cells, weights = np.asarray(self.cells), np.asarray(self.weights)
        gp, gn = np.asarray(self.gp), np.asarray(self.gn)
        n = len(cells)
        if cells.ndim != 1 or cells.dtype.kind not in "iu" or len(set(cells.tolist())) != n:
            raise ValueError("cells must be unique integer external IDs")
        if weights.shape != (n,) or weights.dtype.kind not in "iu" or np.any(weights <= 0):
            raise ValueError("weights must be positive integers")
        if (
            gp.ndim != 2
            or gp.shape != gn.shape
            or gp.shape[0] != n
            or gp.dtype.kind not in "iu"
            or gn.dtype.kind not in "iu"
            or np.any(gp < 0)
            or np.any(gn < 0)
        ):
            raise ValueError("progress must be aligned nonnegative integer N x K arrays")
        for goal in (self.a0, self.b0):
            if isinstance(goal, bool) or not isinstance(goal, (int, np.integer)) or goal <= 0:
                raise ValueError("goals must be positive integers")
        object.__setattr__(self, "cells", np.array(cells, dtype=np.int64))
        object.__setattr__(self, "weights", np.array(weights, dtype=np.int64))
        object.__setattr__(self, "gp", np.array(gp, dtype=np.int64))
        object.__setattr__(self, "gn", np.array(gn, dtype=np.int64))
        object.__setattr__(self, "a0", int(self.a0))
        object.__setattr__(self, "b0", int(self.b0))
        ranks = np.empty(n, dtype=np.intp)
        ranks[np.argsort(self.cells, kind="stable")] = np.arange(n)
        object.__setattr__(self, "ranks", ranks)
        for array in (self.cells, self.weights, self.gp, self.gn, ranks):
            array.setflags(write=False)

    @property
    def size(self):
        return len(self.cells)

    @property
    def outcomes(self):
        return self.gp.shape[1]


def threshold_goals(total, numerator, denominator):
    """Exact ``T = ceil(numerator * total / denominator)`` and ``Z = total - T + 1``."""
    for value in (total, numerator, denominator):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("integer threshold arithmetic required")
    total, numerator, denominator = int(total), int(numerator), int(denominator)
    if total <= 0 or denominator <= 0 or not 0 < numerator <= denominator:
        raise ValueError("supported questions need positive weight and 0 < tau <= 1")
    threshold = -((-numerator * total) // denominator)
    return threshold, total - threshold + 1


def threshold_instance(cells, weights, threshold):
    """Outcomes member/nonmember/unresolved for ``sum of member weight >= threshold``."""
    weights = np.asarray(weights, dtype=np.int64)
    total = int(weights.sum())
    if isinstance(threshold, bool) or not 1 <= int(threshold) <= total:
        raise ValueError("threshold must lie in [1, W]")
    zero = np.zeros_like(weights)
    return Instance(
        cells,
        weights,
        np.column_stack((weights, zero, zero)),
        np.column_stack((zero, weights, zero)),
        int(threshold),
        total - int(threshold) + 1,
    )


def check_probabilities(instance, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (
        p.shape != (instance.size, instance.outcomes)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("probabilities must be finite normalized N x K rows")
    return p


def check_outcomes(instance, outcomes):
    y = np.asarray(outcomes)
    if (
        y.shape != (instance.size,)
        or y.dtype.kind not in "iu"
        or np.any(y < 0)
        or np.any(y >= instance.outcomes)
    ):
        raise ValueError("one valid outcome index per cell required")
    return y.astype(np.intp)


def evaluate(instance, outcomes, order):
    """Exact verifier: cost and status of the shortest certifying prefix of ``order``.

    ``order`` lists distinct cell indices in payment order. Returns ``(cost, status)``
    where an uncertified session pays every listed cell.
    """
    y = check_outcomes(instance, outcomes)
    order = np.asarray(order, dtype=np.intp)
    if order.ndim != 1 or len(set(order.tolist())) != len(order):
        raise ValueError("payments must be distinct cells")
    if len(order) and (order.min() < 0 or order.max() >= instance.size):
        raise ValueError("payment outside the question")
    positive = np.cumsum(instance.gp[order, y[order]])
    negative = np.cumsum(instance.gn[order, y[order]])
    first_positive = int(np.searchsorted(positive, instance.a0, side="left"))
    first_negative = int(np.searchsorted(negative, instance.b0, side="left"))
    stop = min(first_positive, first_negative)
    if stop >= len(order):
        return len(order), "unresolved"
    if first_positive == first_negative:
        raise ArithmeticError("contradictory certificates")
    return stop + 1, "positive" if first_positive < first_negative else "negative"


def full_information_status(instance, outcomes):
    """Status after paying every cell; no order can certify an unresolved question."""
    return evaluate(instance, outcomes, np.arange(instance.size))[1]


def clairvoyant_cost(instance, outcomes):
    """Smallest certifying set given the answers; a diagnostic, not a policy."""
    y = check_outcomes(instance, outcomes)
    rows = np.arange(instance.size)
    best = None
    for gains, goal in ((instance.gp[rows, y], instance.a0), (instance.gn[rows, y], instance.b0)):
        cumulative = np.cumsum(np.sort(gains)[::-1])
        reached = int(np.searchsorted(cumulative, goal, side="left"))
        if reached < len(cumulative):
            best = reached + 1 if best is None else min(best, reached + 1)
    return best


def _descending(instance, score):
    """Order by decreasing score with ascending external-ID ties."""
    return np.lexsort((instance.ranks, -np.asarray(score, dtype=float)))


def expected_progress(instance, probabilities):
    p = check_probabilities(instance, probabilities)
    return (p * instance.gp).sum(1), (p * instance.gn).sum(1)


def weight_order(instance):
    return _descending(instance, instance.weights)


def random_order(instance, rng):
    return rng.permutation(instance.size)


def positive_order(instance, probabilities):
    return _descending(instance, expected_progress(instance, probabilities)[0])


def negative_order(instance, probabilities):
    return _descending(instance, expected_progress(instance, probabilities)[1])


def confident_order(instance, probabilities):
    """Largest expected progress toward whichever goal each cell serves best."""
    positive, negative = expected_progress(instance, probabilities)
    return _descending(instance, np.maximum(positive, negative))


def _forecast(order, progress, goal):
    reached = np.flatnonzero(np.cumsum(progress[order]) >= goal)
    return int(reached[0]) + 1 if reached.size else len(order) + 1


def direction_forecasts(instance, probabilities):
    positive, negative = expected_progress(instance, probabilities)
    po, no = _descending(instance, positive), _descending(instance, negative)
    return (po, _forecast(po, positive, instance.a0)), (no, _forecast(no, negative, instance.b0))


def direction_order(instance, probabilities):
    """Freeze the order whose expected progress reaches its goal sooner; tie positive."""
    (po, fp), (no, fn) = direction_forecasts(instance, probabilities)
    return po if fp <= fn else no


def ggn_static_order(instance, probabilities, *, constant=64, epsilon=0.15):
    """Non-adaptive order derived from Ghuge, Gupta and Nagarajan (arXiv:2111.05687v2).

    Unit costs. For budgets 1, 2, 4, ... take, for the negative then the positive
    goal, the greedy prefix of truncated expected reward at the first scale whose
    boundary density is at most ``epsilon / budget``; append first appearances.
    No approximation guarantee is claimed for estimated priors or paid unresolved
    responses.
    """
    p = check_probabilities(instance, probabilities)
    n = instance.size
    max_power = int(np.maximum(instance.gp, instance.gn).max(1).sum()).bit_length()
    gp, gn = instance.gp.astype(float), instance.gn.astype(float)
    remaining_mask = np.ones(n, dtype=bool)
    order, budget = [], 1

    def knapsack(remaining, gains):
        for exponent in range(max_power + 1):
            scale = float(1 << exponent)
            density = (p[remaining] * np.minimum(gains[remaining] / scale, 1.0)).sum(1)
            perm = np.lexsort((instance.ranks[remaining], -density))
            ranked = remaining[perm]
            cut = constant * budget - 1
            if cut >= len(ranked):
                selected, derivative = ranked, 0.0
            else:
                selected, derivative = ranked[: cut + 1], float(density[perm[cut]])
            if derivative <= epsilon / budget:
                return [int(j) for j in selected]
        raise ArithmeticError("no poor scale under finite scores")

    while remaining_mask.any():
        remaining = np.flatnonzero(remaining_mask)
        added = list(dict.fromkeys(knapsack(remaining, gn) + knapsack(remaining, gp)))
        order.extend(added)
        remaining_mask[added] = False
        budget *= 2
    return np.asarray(order, dtype=np.intp)


class _Adaptive:
    def __init__(self, instance, probabilities):
        self.instance = instance
        self.p = check_probabilities(instance, probabilities).copy()
        self._gp = instance.gp.astype(float)
        self._gn = instance.gn.astype(float)
        self._w = instance.weights.astype(float)
        self.fallbacks = 0

    def _gain(self, a, b):
        x, y = np.minimum(self._gp, float(a)), np.minimum(self._gn, float(b))
        return (self.p * (x * float(b) + float(a) * y - x * y)).sum(1)

    def _best(self, score, mask):
        indices = np.flatnonzero(mask)
        return int(indices[np.lexsort((self.instance.ranks[indices], -score[indices]))[0]])

    def _fallback(self, mask):
        self.fallbacks += 1
        return self._best(self._w, mask)

    def observe(self, cell, outcome):
        """Receive one paid response; static-prior policies ignore it."""


class AdaptiveGreedy(_Adaptive):
    """Maximum expected goal-utility gain (Golovin and Krause; Deshpande et al.)."""

    def choose(self, a, b, available):
        gain = self._gain(a, b)
        if not np.any(available & (gain > 0)):
            return self._fallback(available)
        return self._best(gain, available)


class AdaptiveDualGreedy(_Adaptive):
    """Adaptive Dual Greedy of Deshpande, Hellerstein and Kletenik with unit costs."""

    def __init__(self, instance, probabilities):
        super().__init__(instance, probabilities)
        self._residual = np.ones(instance.size)

    def choose(self, a, b, available):
        gain = self._gain(a, b)
        positive = available & (gain > 0)
        if not positive.any():
            return self._fallback(available)
        ratio = np.full(len(gain), np.inf)
        ratio[positive] = self._residual[positive] / gain[positive]
        j = self._best(-ratio, positive)
        updated = self._residual[available] - float(ratio[j]) * gain[available]
        if not np.isfinite(updated).all() or np.any(updated < -1e-9 * max(1, len(gain))):
            raise ArithmeticError("invalid dual residual")
        self._residual[available] = np.maximum(updated, 0.0)
        self._residual[j] = 0.0
        return j


class AdaptiveDirection(_Adaptive):
    """Pursue one goal at a time: the goal forecast to need fewer further payments.

    For each goal the capped expected progress of every unread cell is accumulated
    in decreasing order until the residual goal is met; a goal that the remaining
    cells cannot meet in expectation costs one more than their number. The policy
    pays the cell with the largest capped expected progress toward the cheaper goal
    (tie: positive) and re-forecasts after every response. Unlike the two-goal
    utility it earns nothing from progress toward a goal it does not pursue.
    """

    def _directional(self, a, b):
        positive = (self.p * np.minimum(self._gp, float(a))).sum(1)
        negative = (self.p * np.minimum(self._gn, float(b))).sum(1)
        return positive, negative

    @staticmethod
    def _needed(progress, goal):
        ranked = np.sort(progress)[::-1]
        reached = np.flatnonzero(np.cumsum(ranked) >= goal)
        return int(reached[0]) + 1 if reached.size else len(ranked) + 1

    def choose(self, a, b, available):
        positive, negative = self._directional(a, b)
        need_positive = self._needed(positive[available], float(a))
        need_negative = self._needed(negative[available], float(b))
        score = positive if need_positive <= need_negative else negative
        if not np.any(available & (score > 0)):
            return self._fallback(available)
        return self._best(score, available)


class _Recalibration:
    """Shrink beliefs toward the paid responses of the current session.

    Threshold questions only. Cells are grouped by their initial resolved membership
    probability. Within a group the member rate among resolved paid cells and the
    unresolved rate among all paid cells replace the prior with weight
    ``n / (n + strength)``. Only paid responses of the current session are used.
    """

    EDGES = (0.05, 0.25, 0.75, 0.95)

    def __init__(self, instance, probabilities, strength=4.0):
        super().__init__(instance, probabilities)
        if instance.outcomes != 3:
            raise ValueError("recalibration expects member/nonmember/unresolved outcomes")
        resolved = np.maximum(self.p[:, MEMBER] + self.p[:, NONMEMBER], 1e-12)
        self._prior_member = self.p[:, MEMBER] / resolved
        self._prior_unresolved = self.p[:, UNRESOLVED].copy()
        self._group = np.searchsorted(np.asarray(self.EDGES), self._prior_member, side="right")
        self._counts = np.zeros((len(self.EDGES) + 1, 3))
        self._strength = float(strength)
        self._all = np.zeros(3)

    def observe(self, cell, outcome):
        self._counts[self._group[cell], outcome] += 1
        self._all[outcome] += 1
        counts = self._counts[self._group]
        resolved = counts[:, MEMBER] + counts[:, NONMEMBER]
        total = resolved + counts[:, UNRESOLVED]
        session_unresolved = (self._all[UNRESOLVED] + 0.5) / (self._all.sum() + 1.0)
        keep = self._strength / (self._strength + resolved)
        member = keep * self._prior_member + (1 - keep) * counts[:, MEMBER] / np.maximum(resolved, 1)
        keep_u = self._strength / (self._strength + total)
        prior_u = 0.5 * self._prior_unresolved + 0.5 * session_unresolved
        unresolved = keep_u * prior_u + (1 - keep_u) * counts[:, UNRESOLVED] / np.maximum(total, 1)
        self.p[:, MEMBER] = (1 - unresolved) * member
        self.p[:, NONMEMBER] = (1 - unresolved) * (1 - member)
        self.p[:, UNRESOLVED] = unresolved


class RecalibratedGreedy(_Recalibration, AdaptiveGreedy):
    """Adaptive Greedy over beliefs recalibrated within the session."""


class RecalibratedDirection(_Recalibration, AdaptiveDirection):
    """Adaptive Direction over beliefs recalibrated within the session."""


def run_adaptive(instance, outcomes, policy):
    """Pay cells chosen by ``policy`` until a certificate or exhaustion; return the order."""
    y = check_outcomes(instance, outcomes)
    a, b = instance.a0, instance.b0
    available = np.ones(instance.size, dtype=bool)
    order = []
    while a > 0 and b > 0 and available.any():
        j = policy.choose(a, b, available)
        if j is None or not available[j]:
            raise RuntimeError("policy returned an unavailable cell")
        available[j] = False
        order.append(j)
        a -= int(instance.gp[j, y[j]])
        b -= int(instance.gn[j, y[j]])
        policy.observe(j, int(y[j]))
    return np.asarray(order, dtype=np.intp)
