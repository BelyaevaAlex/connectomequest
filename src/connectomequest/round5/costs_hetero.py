"""Heterogeneous check costs: some input cells are harder to type and cost ``c > 1`` units.

Difficulty proxies are known before any check and never use the reference answer:

* ``small``: the neuron is small - its total synapse count (in- plus out-strength in the
  thresholded graph of the connectome) is below the 25th percentile of that count over the
  distinct input cells of development targets (a graph statistic; fixed on the development
  split, so the held-out run reuses the same cut).
* ``unlabelled`` (FlyWire only): the earlier annotation release (v2.1.0) gives the neuron no
  name at all. MaleCNS has no earlier release in the repository, so it has no such proxy.

A flagged cell costs ``c`` (2 or 3) and every other cell 1. The cost of a session is the
sum of the costs of the distinct cells checked. Cost-aware Adaptive Greedy checks the cell
with the largest gain of Equation 3 per unit cost (fallback: weight per unit cost), and
cost-aware weight order sorts by ``w_v / c_v``; both reduce to the plain rules when every
cost is 1 (tested). ``clairvoyant_cost`` is the exact minimum cost of a certificate for
costs with two values (any cost vector with at most two distinct values).
"""

from __future__ import annotations

import numpy as np

from connectomequest.annotation_v35 import engine as e


class CostAwareGreedy(e.AdaptiveGreedy):
    """Adaptive Greedy maximizing expected utility gain per unit cost."""

    def __init__(self, instance, probabilities, cost=None):
        super().__init__(instance, probabilities)
        self._c = np.ones(instance.size) if cost is None else np.asarray(cost, dtype=float)

    def choose(self, a, b, available):
        gain = self._gain(a, b)
        if not np.any(available & (gain > 0)):
            self.fallbacks += 1
            return self._best(self._w / self._c, available)
        return self._best(gain / self._c, available)


class CostAwareRecalibratedGreedy(e._Recalibration, CostAwareGreedy):
    """Recalibrated AG maximizing expected utility gain per unit cost."""

    def __init__(self, instance, probabilities, cost=None):
        super().__init__(instance, probabilities)
        self._c = np.ones(instance.size) if cost is None else np.asarray(cost, dtype=float)


def weight_per_cost_order(instance, cost):
    return e._descending(instance, instance.weights / np.asarray(cost, dtype=float))


def run_cost_aware(instance, outcomes, p, cost, recalibrated=False):
    cls = CostAwareRecalibratedGreedy if recalibrated else CostAwareGreedy
    return e.run_adaptive(instance, outcomes, cls(instance, p, cost))


def _min_cost_cover(values, cost, goal):
    """Least total cost of cells whose ``values`` reach ``goal``; costs take at most two values."""
    values, cost = np.asarray(values, dtype=np.int64), np.asarray(cost, dtype=np.int64)
    levels = np.unique(cost)
    if len(levels) > 2:
        raise ValueError("exact bound implemented for at most two cost levels")
    if values.sum() < goal:
        return None
    lo = levels[0]
    hi = levels[-1]
    cheap = np.sort(values[cost == lo])[::-1]
    dear = np.sort(values[cost == hi])[::-1] if hi != lo else np.zeros(0, dtype=np.int64)
    pc = np.concatenate(([0], np.cumsum(cheap)))
    pd = np.concatenate(([0], np.cumsum(dear)))
    best = None
    for k2 in range(len(pd)):
        need = goal - pd[k2]
        if need <= 0:
            k1 = 0
        else:
            k1 = int(np.searchsorted(pc, need, side="left"))
            if k1 >= len(pc):
                continue
        total = int(lo) * k1 + int(hi) * k2
        best = total if best is None else min(best, total)
    return best


def clairvoyant_cost(instance, outcomes, cost):
    """Least total cost of any certificate given all answers (two cost levels; exact)."""
    y = e.check_outcomes(instance, outcomes)
    rows = np.arange(instance.size)
    found = [
        _min_cost_cover(instance.gp[rows, y], cost, instance.a0),
        _min_cost_cover(instance.gn[rows, y], cost, instance.b0),
    ]
    found = [x for x in found if x is not None]
    return min(found) if found else None


def paid_cost(order, cost_of_cell, count):
    """Total cost of the first ``count`` cells of ``order``."""
    return float(np.asarray(cost_of_cell, dtype=float)[np.asarray(order[:count], dtype=np.intp)].sum())
