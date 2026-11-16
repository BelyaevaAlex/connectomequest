"""Betting tests with sampling *without* replacement (round 5, workstream ``policies``).

The frozen sampling test (``annotation_v35.statistical``) and the hybrid test
(``annotation_v35.posthoc``) draw cells with replacement. Here cells are drawn
without replacement, following Waudby-Smith and Ramdas, "Confidence sequences for
sampling without replacement" (NeurIPS 2020): the null mean of the next draw is the
*conditional* mean of the not-yet-sampled population given the draws so far, which
shrinks the null set as the sample grows. Their observation model draws units
uniformly; ours draws a cell with probability proportional to its weight among the
cells not yet drawn (successive sampling, implemented with Efraimidis-Spirakis keys),
so that a draw is a uniformly random *synapse* of the unchecked cells and the whole
cell is then checked.

Claims, with ``L`` the member weight, ``V`` the member-or-unresolved weight and ``R`` the
weight of the unchecked cells before draw ``t``:

* *yes* (``L* >= T``). Null ``L* <= T - 1``. Then
  ``E[X_t | past] = (L* - L) / R <= m_t := (T - 1 - L) / R`` for ``X_t = 1[member]``
  (an unresolved draw counts against *yes*), and the wealth
  ``prod_t (1 + lam_t (X_t - m_t))`` with predictable ``lam_t in [0, 0.9 / m_t]`` is a
  nonnegative supermartingale under the null.
* *no* (``U* <= T - 1``, ``U* = W - nonmember weight``). Null ``U* >= T``. Then
  ``E[Y_t | past] >= m'_t := (T - V) / R`` for ``Y_t = 1[not nonmember]`` (an unresolved
  draw counts against *no*), and ``prod_t (1 + lam'_t (m'_t - Y_t))`` with
  ``lam'_t in [0, 0.9 / (1 - m'_t)]`` is a nonnegative supermartingale under the null.

A claim is certified when its wealth reaches ``1 / alpha`` (Ville's inequality), so a
wrong *yes* and a wrong *no* each have probability at most ``alpha``. A bet is placed
only while ``0 < m < 1``; outside that range the claim is already decided by the
exact bounds or cannot be reached. Bets are the Kelly fraction of the frozen test for
a plug-in mean of the next draw: the Des Raj estimate of the population total
(``z_t = L + X_t R``, unbiased under successive sampling, with one pseudo-observation
of half the weight) minus what was found, over ``R``. The exact bounds over the cells
drawn are checked after every draw, as in the frozen tests; every draw is a new
cell, so the cost is the number of draws, at most ``|V|``.

The hybrid variant checks the weight-order prefix carrying half of ``W`` exactly
(it depends on the weights only) and samples the rest without replacement; the
martingales start after the prefix, with ``L``, ``V`` and ``R`` including it.
"""

from __future__ import annotations

import math

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35.statistical import _kelly

CAP = 0.9


def pps_order(weights, rng, cells=None):
    """Successive sampling without replacement, probability proportional to weight."""
    w = np.asarray(weights, dtype=float)
    keys = rng.exponential(size=len(w)) / w
    return np.argsort(keys, kind="stable") if cells is None else np.asarray(cells)[np.argsort(keys, kind="stable")]


def heavy_prefix(instance, stratum=(1, 2)):
    """Shortest prefix of weight order carrying at least ``stratum`` of the weight (as the frozen hybrid test)."""
    order = e.weight_order(instance)
    num, den = stratum
    size = int(np.searchsorted(np.cumsum(instance.weights[order]) * den, int(instance.weights.sum()) * num, side="left")) + 1
    return order[:size]


def wor_certificate(instance, outcomes, *, alpha=0.05, rng, prefix=None):
    """One audit of a threshold question. Returns ``(cells_checked, status, exact)``.

    ``prefix``: cells checked exactly before sampling starts (the hybrid's stratum).
    """
    y = e.check_outcomes(instance, outcomes)
    w = instance.weights
    n, total = instance.size, int(w.sum())
    threshold, goal_no = instance.a0, instance.b0
    if goal_no != total - threshold + 1 or not 0 < alpha < 1:
        raise ValueError("threshold questions only")
    prefix = np.zeros(0, dtype=np.intp) if prefix is None else np.asarray(prefix, dtype=np.intp)
    member = nonmember = upper = 0  # checked member, nonmember and member-or-unresolved weight
    remaining = total
    checked = 0
    for j in prefix:
        wj = int(w[j])
        member += wj if y[j] == e.MEMBER else 0
        nonmember += wj if y[j] == e.NONMEMBER else 0
        upper += wj if y[j] != e.NONMEMBER else 0
        remaining -= wj
        checked += 1
        if member >= threshold:
            return checked, "positive", True
        if nonmember >= goal_no:
            return checked, "negative", True
    rest = np.setdiff1d(np.arange(n), prefix, assume_unique=False)
    sequence = pps_order(w[rest], rng, rest).tolist()
    log_limit = math.log(1.0 / alpha)
    wealth_yes = wealth_no = 0.0
    z_member = z_upper = 0.0
    drawn = 0
    wl, yl = w.tolist(), y.tolist()
    for j in sequence:
        wj = wl[j]
        x = 1.0 if yl[j] == e.MEMBER else 0.0
        v = 1.0 if yl[j] != e.NONMEMBER else 0.0
        r = float(remaining)
        # Predictable plug-in means of the next draw (Des Raj totals with one pseudo-observation of W / 2).
        est_member = min(1.0, max(0.0, ((0.5 * total + z_member) / (drawn + 1.0) - member) / r))
        est_upper = min(1.0, max(0.0, ((0.5 * total + z_upper) / (drawn + 1.0) - upper) / r))
        m_yes = (threshold - 1 - member) / r
        m_no = (threshold - upper) / r
        if 0.0 < m_yes < 1.0:
            wealth_yes += math.log1p(_kelly(est_member, m_yes, CAP / m_yes) * (x - m_yes))
        if 0.0 < m_no < 1.0:
            wealth_no += math.log1p(_kelly(1.0 - est_upper, 1.0 - m_no, CAP / (1.0 - m_no)) * (m_no - v))
        z_member += member + x * r
        z_upper += upper + v * r
        drawn += 1
        checked += 1
        member += wj if x else 0
        upper += wj if v else 0
        nonmember += 0 if v else wj
        remaining -= wj
        if member >= threshold:
            return checked, "positive", True
        if nonmember >= goal_no:
            return checked, "negative", True
        if wealth_yes >= log_limit:
            return checked, "positive", False
        if wealth_no >= log_limit:
            return checked, "negative", False
    return n, "unresolved", True


def wor_hybrid_certificate(instance, outcomes, *, alpha=0.05, rng, stratum=(1, 2)):
    return wor_certificate(instance, outcomes, alpha=alpha, rng=rng, prefix=heavy_prefix(instance, stratum))
