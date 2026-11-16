"""Batch cost model: one check confirms the release type of every input cell that shares it.

A *batch* of a question is the set of its input cells with one release type (MaleCNS
``type``, FlyWire v3.1.0 ``cell_type``); an input cell without a release type is a batch
of its own. Checking a batch costs one unit and reveals the reference answer of all its
cells. A batch is a unit with weight ``W_B`` (sum of its cells' weights) and realized
progress ``M_B`` (member weight) toward *yes* and ``N_B`` (nonmember weight) toward *no*.

The batch question is an ``engine.Instance`` with four outcomes per unit:
``0`` all member (gp = W_B), ``1`` all nonmember (gn = W_B), ``2`` all unresolved and
``3`` mixed (gp = M_B, gn = N_B). The realized outcome of a mixed batch is 3; the prior
puts no mass on 3. The exact verifier, the clairvoyant bound, weight order, member-first,
direction-fixed and Adaptive Greedy of the frozen engine then apply unchanged. The prior of
a batch is the weight-averaged prior of its cells, ``p_B = sum_v w_v p_v / W_B`` (exactly
``p_v`` for a one-cell batch), so its expected progress is the expected progress of all its
cells and its Adaptive Greedy gain is Equation 3 of the paper for one unit of weight ``W_B``
whose cells share one answer. ``SumGreedy`` instead adds the per-cell gains of Equation 3
(cells answering independently, no interaction inside the batch); ``BatchRecalibratedGreedy``
recalibrates per checked batch (one observation split over its cells' answers). With every
batch a single cell all policies coincide with the frozen per-cell policies (tested).
"""

from __future__ import annotations

import numpy as np

from connectomequest.annotation_v35 import engine as e

MIXED = 3


def batch_keys(release):
    """Batch key of every cell: its release type, or a unique negative key if it has none."""
    release = np.asarray(release)
    return np.where(release >= 0, release, -(np.arange(len(release)) + 2))


class BatchQuestion:
    """Batches of one question: the engine instance plus cell-to-batch bookkeeping."""

    def __init__(self, cells, weights, outcomes, threshold, keys):
        cells, weights, outcomes = np.asarray(cells), np.asarray(weights, dtype=np.int64), np.asarray(outcomes)
        n = len(cells)
        if not (len(weights) == len(outcomes) == len(keys) == n):
            raise ValueError("aligned cell arrays required")
        if np.any(np.diff(cells) <= 0):
            raise ValueError("cells must be sorted by identifier")
        _, first, inverse = np.unique(np.asarray(keys), return_index=True, return_inverse=True)
        # Batch order = order of first appearance, i.e. ascending smallest cell identifier.
        rank = np.empty(len(first), dtype=np.intp)
        rank[np.argsort(first, kind="stable")] = np.arange(len(first))
        self.batch_of = rank[inverse]
        nb = len(first)
        self.first_cell = cells[np.sort(first)]
        self.size = np.bincount(self.batch_of, minlength=nb)
        W = np.bincount(self.batch_of, weights=weights, minlength=nb).astype(np.int64)
        M = np.bincount(self.batch_of, weights=weights * (outcomes == e.MEMBER), minlength=nb).astype(np.int64)
        N = np.bincount(self.batch_of, weights=weights * (outcomes == e.NONMEMBER), minlength=nb).astype(np.int64)
        self.weights, self.member, self.nonmember = W, M, N
        self.max_weight = np.zeros(nb, dtype=np.int64)
        np.maximum.at(self.max_weight, self.batch_of, weights)
        zero = np.zeros(nb, dtype=np.int64)
        self.instance = e.Instance(
            self.first_cell, W, np.column_stack((W, zero, zero, M)), np.column_stack((zero, W, zero, N)), int(threshold), int(W.sum()) - int(threshold) + 1
        )
        outcome = np.full(nb, MIXED, dtype=np.intp)
        outcome[M == W] = e.MEMBER
        outcome[N == W] = e.NONMEMBER
        outcome[(M == 0) & (N == 0)] = e.UNRESOLVED
        self.outcomes = outcome
        counts = np.zeros((nb, 3))
        np.add.at(counts, (self.batch_of, np.asarray(outcomes, dtype=np.intp)), 1.0)
        self.shares = counts / self.size[:, None]
        self.cell_weights, self.cell_outcomes = weights, np.asarray(outcomes, dtype=np.intp)

    @property
    def units(self):
        return len(self.weights)

    def probabilities(self, p):
        """Weight-averaged prior of every batch (exactly the cell prior for one-cell batches)."""
        p = np.asarray(p, dtype=float)
        nb = self.units
        out = np.zeros((nb, 4))
        for k in range(3):
            out[:, k] = np.bincount(self.batch_of, weights=self.cell_weights * p[:, k], minlength=nb) / self.weights
        single = np.flatnonzero(self.size == 1)
        cell = np.empty(nb, dtype=np.intp)
        cell[self.batch_of] = np.arange(len(self.batch_of))  # a cell of each batch (the only one if single)
        out[single, :3] = p[cell[single]]
        return out

    def evaluate(self, order):
        return e.evaluate(self.instance, self.outcomes, order)

    def status(self):
        return e.full_information_status(self.instance, self.outcomes)

    def clairvoyant(self):
        return e.clairvoyant_cost(self.instance, self.outcomes)


def weight_order(bq):
    """Release types (batches) ranked by total weight."""
    return e.weight_order(bq.instance)


def heaviest_cell_order(bq):
    """The per-cell weight order with every batch checked at its heaviest cell."""
    return e._descending(bq.instance, bq.max_weight)


def member_order(bq, p):
    return e.positive_order(bq.instance, bq.probabilities(p))


def direction_order(bq, p):
    return e.direction_order(bq.instance, bq.probabilities(p))


def run_greedy(bq, p):
    return e.run_adaptive(bq.instance, bq.outcomes, e.AdaptiveGreedy(bq.instance, bq.probabilities(p)))


class SumGreedy(e.AdaptiveGreedy):
    """Batch gain = sum of the per-cell gains of Equation 3 at the current residual goals."""

    def __init__(self, bq, p):
        super().__init__(bq.instance, bq.probabilities(p))
        self._bq = bq
        self._cp = np.asarray(p, dtype=float)
        self._cw = bq.cell_weights.astype(float)

    def _gain(self, a, b):
        x, y = np.minimum(self._cw, float(a)), np.minimum(self._cw, float(b))
        cell = self._cp[:, e.MEMBER] * x * float(b) + self._cp[:, e.NONMEMBER] * float(a) * y
        return np.bincount(self._bq.batch_of, weights=cell, minlength=self._bq.units)


def run_sum_greedy(bq, p):
    return e.run_adaptive(bq.instance, bq.outcomes, SumGreedy(bq, p))


class BatchRecalibratedGreedy(e.AdaptiveGreedy):
    """Recalibrated AG over batches; a checked batch is one observation split over its cells.

    Same bins (edges 0.05, 0.25, 0.75, 0.95 of the batch's resolved membership probability),
    strength 4 and session-pooled unresolved rate as ``engine.RecalibratedGreedy``; the
    observation of a checked batch adds the shares of member, nonmember and unresolved
    answers among its cells (a one-hot vector for a one-cell batch).
    """

    EDGES = e._Recalibration.EDGES

    def __init__(self, bq, p, strength=4.0):
        super().__init__(bq.instance, bq.probabilities(p))
        resolved = np.maximum(self.p[:, e.MEMBER] + self.p[:, e.NONMEMBER], 1e-12)
        self._prior_member = self.p[:, e.MEMBER] / resolved
        self._prior_unresolved = self.p[:, e.UNRESOLVED].copy()
        self._group = np.searchsorted(np.asarray(self.EDGES), self._prior_member, side="right")
        self._counts = np.zeros((len(self.EDGES) + 1, 3))
        self._strength = float(strength)
        self._all = np.zeros(3)
        self._shares = bq.shares

    def observe(self, cell, outcome):
        share = self._shares[cell]
        self._counts[self._group[cell]] += share
        self._all += share
        counts = self._counts[self._group]
        resolved = counts[:, e.MEMBER] + counts[:, e.NONMEMBER]
        total = resolved + counts[:, e.UNRESOLVED]
        session_unresolved = (self._all[e.UNRESOLVED] + 0.5) / (self._all.sum() + 1.0)
        keep = self._strength / (self._strength + resolved)
        member = keep * self._prior_member + (1 - keep) * counts[:, e.MEMBER] / np.maximum(resolved, 1)
        keep_u = self._strength / (self._strength + total)
        prior_u = 0.5 * self._prior_unresolved + 0.5 * session_unresolved
        unresolved = keep_u * prior_u + (1 - keep_u) * counts[:, e.UNRESOLVED] / np.maximum(total, 1)
        self.p[:, e.MEMBER] = (1 - unresolved) * member
        self.p[:, e.NONMEMBER] = (1 - unresolved) * (1 - member)
        self.p[:, e.UNRESOLVED] = unresolved
        self.p[:, MIXED] = 0.0


def run_recalibrated(bq, p):
    return e.run_adaptive(bq.instance, bq.outcomes, BatchRecalibratedGreedy(bq, p))
