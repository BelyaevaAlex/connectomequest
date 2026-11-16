"""Synthetic priors with correlated errors (round 5, workstream ``priors``).

The synthetic priors of the paper (``annotation_v35/sweep.py``) make errors that are independent
of synapse weight and of each other. Two departures, each in the calibrated form (Gaussian score,
posterior membership probability, base-rate unresolved mass, as the ``base`` variant of the sweep)
and in the confident form (noisy oracle with 0.9 on the named answer, as the ``oracle`` variant):

* weight-correlated errors. Let ``r`` be the within-question weight percentile of a cell (0 for the
  lightest, 1 for the heaviest, ties averaged). ``wplus``: heavy cells are more accurate, the
  separation of a cell is ``d_v = c * exp(BETA * (r - 1/2))`` (calibrated) or its flip probability
  ``eps_v = min(1, c * exp(-BETA * (r - 1/2)))`` (confident); ``wminus``: the opposite sign. The
  calibrated posterior uses the cell's own ``d_v``, so the probabilities stay calibrated.
* errors shared within release type (``rtype``): every input cell of one release type (the
  release's own type name; a cell without one is its own group) receives the same noise value
  (calibrated) or the same flip decision and the same wrong answer (confident).

The scale ``c`` is chosen per connectome, form and level so that the realized within-question
AUROC, averaged over question families and draws exactly as in the paper, equals the value the
independent synthetic prior of the sweep realized on the same families. With ``BETA = 0`` the
weight-correlated priors are exactly the sweep's priors (same random numbers).
"""

from __future__ import annotations

from math import log

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import sweep as sw

BETA = log(4.0)  # heaviest vs lightest cell: separation (or flip probability) differs by a factor of 4
CORRELATIONS = ("wplus", "wminus", "rtype")
FORMS = ("base", "oracle")  # calibrated with base-rate unresolved mass / confident noisy oracle
CONFIDENCE = 0.9


def weight_percentile(weights):
    """Within-question weight percentile in [0, 1], ties averaged; 1/2 for a single cell."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n <= 1:
        return np.full(n, 0.5)
    order = np.argsort(w, kind="stable")
    ranks = np.empty(n)
    sw_sorted = w[order]
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and sw_sorted[stop + 1] == sw_sorted[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0
        start = stop + 1
    return ranks / (n - 1)


def modulation(percentile, correlation):
    """Multiplier of the separation per cell (the flip probability uses its inverse)."""
    if correlation == "wplus":
        return np.exp(BETA * (np.asarray(percentile) - 0.5))
    if correlation == "wminus":
        return np.exp(-BETA * (np.asarray(percentile) - 0.5))
    if correlation in ("rtype", "independent"):
        return np.ones(len(percentile))
    raise ValueError(correlation)


def calibrated(outcomes, noise, separation, member_rate, unresolved_rate):
    """Gaussian-score posterior with a per-cell separation ``d_v``; equals ``sweep.synthetic_prior`` for constant d."""
    outcomes = np.asarray(outcomes)
    d = np.asarray(separation, dtype=float)
    score = d * (outcomes == e.MEMBER) + np.asarray(noise, dtype=float)
    if np.ndim(member_rate) == 0:  # the sweep's own arithmetic, bit for bit
        prior_log_odds = log(float(member_rate) / (1.0 - float(member_rate)))
    else:  # one base rate per cell (vectorised scale fitting)
        rate = np.asarray(member_rate, dtype=float)
        prior_log_odds = np.log(rate / (1.0 - rate))
    log_odds = prior_log_odds + d * score - d * d / 2.0
    plus = 1.0 / (1.0 + np.exp(-log_odds))
    return sw._with_rate(plus, unresolved_rate)


def confident(outcomes, flip, alternative, flip_probability, confidence=CONFIDENCE):
    """Noisy oracle with a per-cell flip probability; equals ``sweep.oracle_prior`` for a constant one."""
    outcomes = np.asarray(outcomes)
    eps = np.clip(np.asarray(flip_probability, dtype=float), 0.0, 1.0)
    named = np.where(np.asarray(flip) < eps, (outcomes + 1 + (np.asarray(alternative) >= 0.5)) % 3, outcomes)
    p = np.full((len(outcomes), 3), (1.0 - confidence) / 2.0)
    p[np.arange(len(outcomes)), named] = confidence
    return p


def group_index(groups):
    """Dense group index per cell; a negative code (no release type) is a group of its own."""
    groups = np.asarray(groups, dtype=np.int64)
    key = np.where(groups >= 0, groups, -1 - np.arange(len(groups)))
    _, index = np.unique(key, return_inverse=True)
    return index


def random_numbers(family, draw, n, groups, correlation):
    """Noise (standard normal) and two uniforms per cell; the sweep's own streams unless errors are shared by type."""
    if correlation != "rtype":
        noise = np.random.default_rng([29, family, draw]).standard_normal(n)
        uniform = np.random.default_rng([31, family, draw]).random((2, n))
        return noise, uniform[0], uniform[1]
    g = group_index(groups)
    k = int(g.max()) + 1 if n else 0
    noise = np.random.default_rng([43, family, draw]).standard_normal(k)[g]
    uniform = np.random.default_rng([47, family, draw]).random((2, k))[:, g]
    return noise, uniform[0], uniform[1]


def prior(form, correlation, scale, outcomes, percentile, numbers, member_rate, unresolved_rate):
    """One synthetic prior; ``scale`` is the separation (calibrated) or the base flip probability (confident)."""
    noise, flip, alternative = numbers
    mod = modulation(percentile, correlation)
    if form == "base":
        return calibrated(outcomes, noise, scale * mod, member_rate, unresolved_rate)
    if form == "oracle":
        return confident(outcomes, flip, alternative, scale / mod)
    raise ValueError(form)


def grouped_auroc(score, positive, group):
    """Within-group AUROC (ties averaged) for every group; NaN where a group lacks a class."""
    score, positive, group = np.asarray(score, dtype=float), np.asarray(positive, dtype=bool), np.asarray(group)
    order = np.lexsort((score, group))
    gs, ss, ps = group[order], score[order], positive[order]
    n = len(order)
    if n == 0:
        return np.zeros(0)
    gstart = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    gid = np.cumsum(np.r_[0, (gs[1:] != gs[:-1]).astype(int)])
    pos_in_group = np.arange(n) - gstart[gid]
    run = np.r_[True, (gs[1:] != gs[:-1]) | (ss[1:] != ss[:-1])]
    rid = np.cumsum(run) - 1
    run_start = np.flatnonzero(run)
    run_len = np.diff(np.r_[run_start, n])
    avg = pos_in_group[run_start] + (run_len - 1) / 2.0 + 1.0
    ranks = avg[rid]
    k = gid[-1] + 1
    npos = np.bincount(gid, weights=ps, minlength=k)
    size = np.bincount(gid, minlength=k).astype(float)
    nneg = size - npos
    rsum = np.bincount(gid, weights=ranks * ps, minlength=k)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    out = np.full(int(gs[-1]) + 1, np.nan)  # groups are non-negative integers
    out[gs[gstart]] = auc
    return out


def fit_scale(realized, target, lo, hi, increasing, iterations=40):
    """Bisection for the scale whose realized mean AUROC equals ``target``."""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = realized(mid)
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
