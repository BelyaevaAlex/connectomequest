"""Additions made after the held-out run (the frozen engine is untouched).

* Three-way stopping. "Cannot be certified" is itself provable from the checked
  cells: *yes* is out of reach once checked nonmember plus checked unresolved weight
  exceeds ``W - T`` and *no* is out of reach once checked member plus checked
  unresolved weight reaches ``T``. When both hold the question is undecidable with
  this reference, whatever the unchecked cells are.
* A round-robin order alternating between the member-first and nonmember-first
  orders (Kaplan, Kushilevitz and Mansour; Allen et al.).
* A hybrid sampling test: the heaviest cells are checked exactly in weight order
  and only the remaining tail is sampled.

Threshold questions only (outcomes member / nonmember / unresolved).
"""

from __future__ import annotations

import numpy as np

from . import engine as e
from .statistical import _kelly


def evaluate_three_way(instance, outcomes, order):
    """Cost and status with the third certificate, undecidability.

    Returns ``(cost, status)`` with status ``positive``, ``negative``, ``undecidable``
    or ``open`` (``order`` ended before any of the three could be proven).
    """
    y = e.check_outcomes(instance, outcomes)
    order = np.asarray(order, dtype=np.intp)
    if order.ndim != 1 or len(set(order.tolist())) != len(order):
        raise ValueError("payments must be distinct cells")
    member = np.cumsum(instance.gp[order, y[order]])
    nonmember = np.cumsum(instance.gn[order, y[order]])
    unchecked = int(instance.weights.sum()) - np.cumsum(instance.weights[order])
    yes = member >= instance.a0
    no = nonmember >= instance.b0
    neither = (member + unchecked < instance.a0) & (nonmember + unchecked < instance.b0)
    stopped = np.flatnonzero(yes | no | neither)
    if not stopped.size:
        return len(order), "open"
    k = int(stopped[0])
    if yes[k] and no[k]:
        raise ArithmeticError("contradictory certificates")
    return k + 1, "positive" if yes[k] else "negative" if no[k] else "undecidable"


def undecidability_lower_bound(instance, outcomes):
    """Lower bound on the checks that prove an uncertifiable question undecidable.

    Any proof needs checked member-or-unresolved weight of at least ``T`` and checked
    nonmember-or-unresolved weight of at least ``Z``; the bound is the larger of the
    two smallest cell counts. ``None`` when the question is certifiable.
    """
    y = e.check_outcomes(instance, outcomes)
    if e.full_information_status(instance, y) != "unresolved":
        return None
    weights = instance.weights
    best = 0
    for blocked, goal in ((e.NONMEMBER, instance.a0), (e.MEMBER, instance.b0)):
        usable = np.sort(weights[y != blocked])[::-1]
        reached = int(np.searchsorted(np.cumsum(usable), goal, side="left"))
        if reached >= len(usable):
            raise ArithmeticError("an uncertifiable question must admit an undecidability proof")
        best = max(best, reached + 1)
    return best


def round_robin_order(instance, probabilities):
    """Alternate between the member-first and the nonmember-first order; member-first starts."""
    lists = (e.positive_order(instance, probabilities), e.negative_order(instance, probabilities))
    pointers = [0, 0]
    seen = np.zeros(instance.size, dtype=bool)
    order = []
    turn = 0
    while len(order) < instance.size:
        source = lists[turn]
        while seen[source[pointers[turn]]]:
            pointers[turn] += 1
        cell = int(source[pointers[turn]])
        seen[cell] = True
        order.append(cell)
        turn = 1 - turn
    return np.asarray(order, dtype=np.intp)


def hybrid_certificate(instance, outcomes, *, alpha=0.05, rng, stratum=(1, 2), max_draw_factor=50):
    """Exact checks of the heaviest cells, then a weighted sampling test of the tail.

    The certainty stratum is the shortest prefix of the weight order that carries at
    least ``stratum`` of the input weight; it depends on weights only. Its cells are
    checked exactly, stopping at an exact certificate. The remaining cells are then
    drawn with replacement in proportion to weight, and two one-sided betting tests
    run on the residual claims: with ``M``, ``R`` the member and unresolved weight of
    the stratum and ``W'`` the tail weight, *yes* needs tail member weight of at least
    ``T - M`` and *no* needs tail not-nonmember weight of at most ``T - M - R - 1``.
    Returns ``(distinct_cells_checked, status, exact)``.
    """
    y = e.check_outcomes(instance, outcomes)
    w = instance.weights
    total = int(w.sum())
    threshold, negative_goal = instance.a0, instance.b0
    order = e.weight_order(instance)
    num, den = stratum
    size = int(np.searchsorted(np.cumsum(w[order]) * den, total * num, side="left")) + 1
    lower = nonmember = unresolved = 0
    for position, j in enumerate(order[:size]):
        if y[j] == e.MEMBER:
            lower += int(w[j])
        elif y[j] == e.NONMEMBER:
            nonmember += int(w[j])
        else:
            unresolved += int(w[j])
        if lower >= threshold:
            return position + 1, "positive", True
        if nonmember >= negative_goal:
            return position + 1, "negative", True
    rest = order[size:]
    if not len(rest):
        return instance.size, "unresolved", True
    tail = int(w[rest].sum())
    need_yes = threshold - lower
    need_no = threshold - lower - unresolved
    test_yes = 1 <= need_yes <= tail and need_yes > 1
    test_no = 1 <= need_no <= tail and need_no < tail
    null_yes = (need_yes - 1) / tail if test_yes else 0.0
    null_no = need_no / tail if test_no else 1.0
    cap_yes = 0.9 / null_yes if test_yes else 0.0
    cap_no = 0.9 / (1.0 - null_no) if test_no else 0.0
    log_limit = np.log(1.0 / alpha)
    wealth_yes = wealth_no = 0.0
    seen = np.zeros(len(rest), dtype=bool)
    distinct = size
    sum_member = sum_upper = 0.0
    draws = rng.choice(len(rest), size=max_draw_factor * len(rest) + 64, p=w[rest] / tail)
    for t, i in enumerate(draws):
        j = rest[i]
        member = y[j] == e.MEMBER
        upper = y[j] != e.NONMEMBER
        estimate_member = (sum_member + 0.5) / (t + 1.0)
        estimate_upper = (sum_upper + 0.5) / (t + 1.0)
        if test_yes:
            wealth_yes += np.log1p(_kelly(estimate_member, null_yes, cap_yes) * (float(member) - null_yes))
        if test_no:
            wealth_no += np.log1p(_kelly(1.0 - estimate_upper, 1.0 - null_no, cap_no) * (null_no - float(upper)))
        sum_member += float(member)
        sum_upper += float(upper)
        if not seen[i]:
            seen[i] = True
            distinct += 1
            lower += int(w[j]) if member else 0
            nonmember += int(w[j]) if not upper else 0
        if lower >= threshold:
            return distinct, "positive", True
        if nonmember >= negative_goal:
            return distinct, "negative", True
        if test_yes and wealth_yes >= log_limit:
            return distinct, "positive", False
        if test_no and wealth_no >= log_limit:
            return distinct, "negative", False
        if distinct == instance.size:
            break
    return instance.size, "unresolved", True
