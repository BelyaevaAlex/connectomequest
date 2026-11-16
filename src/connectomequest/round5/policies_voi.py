"""One-step value-of-information policies (round 5, workstream ``policies``).

Two cheap rules that run on every question. Both use the current beliefs ``q`` (the
static prior, or for the ``-recal`` variants the in-question recalibration of the
frozen engine) and break ties by the Adaptive Greedy gain, then by weight, then by
ascending neuron identifier.

* ``voi-complete`` - check the cell with the largest probability that *this* check
  completes a certificate, ``q+_v [w_v >= a] + q-_v [w_v >= b]``. While no single cell
  can complete one, the tie-break makes it Adaptive Greedy.
* ``voi-rollout`` - one-step lookahead with a rollout of weight order: check the cell
  that minimizes ``1 + sum_o q_v(o) H(R \\ v, a_o, b_o)``, where ``H(R, a, b)`` is the
  expected number of further checks if the cells of ``R`` were then checked in weight
  order, under independent answers with probabilities ``q``:

      H(R, a, b) = sum_{k=0}^{|R|-1} P(M_k < a) P(N_k < b),

  ``M_k`` and ``N_k`` the member and nonmember weight among the ``k`` heaviest cells of
  ``R``, each probability by a normal approximation with continuity correction (the
  product ignores the negative dependence of ``M_k`` and ``N_k``, which can only
  overstate ``H``). This is one step of policy improvement over weight order. To keep a
  step cheap the lookahead is evaluated for the 16 cells with the largest Adaptive
  Greedy gain and the heaviest unchecked cell; a step costs ``O(17 * 3 * |R|)``. Once
  neither goal can be reached any more, it takes weight order.

The forecast-based lookahead tried first on a development subset (expected-progress
forecasts of the single-goal rule) was inconsistent - removing a cell also removed
its expected progress from the forecast, so heavy cells looked costly - and was
dropped before any full run.

Probabilities only order checks; the frozen verifier decides every certificate.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from connectomequest.annotation_v35 import engine as e

CANDIDATES = 16


def _below(mean, var, goal):
    """P(S < goal) for an integer sum with the given mean and variance (normal, continuity-corrected)."""
    sd = np.sqrt(np.maximum(var, 0.0))
    z = (goal - 0.5 - mean) / np.where(sd > 0, sd, 1.0)
    return np.where(sd > 0, ndtr(z), (mean < goal).astype(float))


def rollout_cost(w_sorted, q_sorted, a, b):
    """Expected further checks of weight order over cells sorted by decreasing weight (``H`` above)."""
    if a <= 0 or b <= 0:
        return 0.0
    k = len(w_sorted)
    if k == 0:
        return 0.0
    wp = w_sorted * q_sorted[:, e.MEMBER]
    wn = w_sorted * q_sorted[:, e.NONMEMBER]
    w2 = w_sorted * w_sorted
    mean_m = np.concatenate(([0.0], np.cumsum(wp)[:-1]))
    mean_n = np.concatenate(([0.0], np.cumsum(wn)[:-1]))
    var_m = np.concatenate(([0.0], np.cumsum(w2 * q_sorted[:, e.MEMBER] * (1 - q_sorted[:, e.MEMBER]))[:-1]))
    var_n = np.concatenate(([0.0], np.cumsum(w2 * q_sorted[:, e.NONMEMBER] * (1 - q_sorted[:, e.NONMEMBER]))[:-1]))
    return float((_below(mean_m, var_m, a) * _below(mean_n, var_n, b)).sum())


class _VOI(e._Adaptive):
    rule = "rollout"

    def __init__(self, instance, probabilities):
        super().__init__(instance, probabilities)
        self._weight_order = e.weight_order(instance)
        self._total = float(instance.weights.sum())

    def choose(self, a, b, available):
        idx = np.flatnonzero(available)
        w = self._w[idx]
        remaining = float(w.sum())
        if a > remaining and b > remaining:  # no certificate reachable: the order no longer matters
            return int(next(j for j in self._weight_order if available[j]))
        gain = self._gain(a, b)[idx]
        q = self.p[idx]
        if self.rule == "complete":
            primary = q[:, e.MEMBER] * (w >= a) + q[:, e.NONMEMBER] * (w >= b)
            best = np.lexsort((self.instance.ranks[idx], -w, -gain, -np.round(primary, 12)))[0]
            return int(idx[best])
        order = np.asarray([int(j) for j in self._weight_order if available[j]])
        heaviest = int(order[0])
        top = idx[np.lexsort((self.instance.ranks[idx], -w, -gain))[:CANDIDATES]]
        cand = np.asarray(list(dict.fromkeys([int(j) for j in top] + [heaviest])))
        values = rollout_lookahead(self._w[order], self.p[order], order, cand, self._w[cand], self.p[cand], a, b)
        values = np.round(values, 12)
        gain_c = self._gain(a, b)[cand]
        best = np.lexsort((self.instance.ranks[cand], -self._w[cand], -gain_c, values))[0]
        return int(cand[best])


def rollout_lookahead(ws, qs, order, cand, wc, qc, a, b):
    """``1 + sum_o q_v(o) H(R \\ v, a_o, b_o)`` for every candidate ``v`` at once (vectorized ``rollout_cost``)."""
    k = len(ws)
    rank = np.empty(int(order.max()) + 1, dtype=np.intp)
    rank[order] = np.arange(k)
    pos = rank[cand]  # position of each candidate in weight order of R
    pm, pn = qs[:, e.MEMBER], qs[:, e.NONMEMBER]
    terms = (ws * pm, ws * pn, ws * ws * pm * (1 - pm), ws * ws * pn * (1 - pn))
    sums = [np.concatenate(([0.0], np.cumsum(t))) for t in terms]  # S[j] = sum over the j heaviest of R
    j = np.arange(k - 1)  # terms j = 0 .. |R \ v| - 1
    after = j[None, :] >= pos[:, None]
    def without(S, x):
        return np.where(after, S[None, j + 1] - x[:, None], S[None, j])
    own = (wc * qc[:, e.MEMBER], wc * qc[:, e.NONMEMBER], wc * wc * qc[:, e.MEMBER] * (1 - qc[:, e.MEMBER]), wc * wc * qc[:, e.NONMEMBER] * (1 - qc[:, e.NONMEMBER]))
    mean_m, mean_n, var_m, var_n = (without(S, x) for S, x in zip(sums, own))
    def h(goal_a, goal_b):
        out = (_below(mean_m, var_m, goal_a[:, None]) * _below(mean_n, var_n, goal_b[:, None])).sum(1)
        return np.where((goal_a <= 0) | (goal_b <= 0), 0.0, out)
    fa, fb = np.full(len(cand), float(a)), np.full(len(cand), float(b))
    value = 1.0 + qc[:, e.MEMBER] * h(fa - wc, fb) + qc[:, e.NONMEMBER] * h(fa, fb - wc) + qc[:, e.UNRESOLVED] * h(fa, fb)
    return value


class Rollout(_VOI):
    rule = "rollout"


class Complete(_VOI):
    rule = "complete"


class RecalRollout(e._Recalibration, Rollout):
    pass


class RecalComplete(e._Recalibration, Complete):
    pass


POLICIES = {"voi-rollout": Rollout, "voi-complete": Complete, "voi-rollout-recal": RecalRollout, "voi-complete-recal": RecalComplete}
