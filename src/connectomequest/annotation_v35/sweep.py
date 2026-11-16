"""Priors of controlled quality (added in the second round of review; the frozen engine is untouched).

A *synthetic prior* knows the true answers of a question only through a noisy score.
Every input cell receives ``z = d * [member] + noise`` with standard normal noise, so
that members are ranked above other cells with an area under the ROC curve of
``Phi(d / sqrt 2)``; the separation ``d`` is chosen for a target value. The
membership probability is the posterior of this model given the base rate ``pi`` of
members, so the prior is calibrated whenever ``pi`` is. For the unresolved answer
there are two variants: no mass (as the connectivity priors have) or the base rate
``r`` of unresolved answers among cells that are not members, which is again the
posterior of the model because the score carries no information about it.

``with_unresolved_rate`` gives an existing prior the same base-rate treatment of the
unresolved answer, keeping its membership probabilities.

A *noisy-oracle prior* is confident instead of calibrated, like a prior built from an
earlier annotation: it names one of the three answers for every cell and puts 0.9 on
it and 0.05 on each other answer. The named answer is the true one except with flip
probability ``eps``, in which case it is one of the two others with equal chance;
members are then ranked above other cells with an area under the ROC curve of
``1 - 3 eps / 4``.
"""

from __future__ import annotations

from math import log, sqrt
from statistics import NormalDist

import numpy as np

from . import engine as e


def separation(auroc):
    """Mean shift of the member scores that gives the requested area under the ROC curve."""
    if not 0.5 <= auroc < 1.0:
        raise ValueError("the area under the ROC curve must lie in [0.5, 1)")
    return sqrt(2.0) * NormalDist().inv_cdf(auroc)


def synthetic_prior(outcomes, noise, auroc, member_rate, unresolved_rate=None):
    """Probabilities (member, nonmember, unresolved) of a synthetic prior; rows sum to one."""
    outcomes = np.asarray(outcomes)
    noise = np.asarray(noise, dtype=float)
    if outcomes.shape != noise.shape:
        raise ValueError("one noise value per cell")
    if not 0.0 < member_rate < 1.0:
        raise ValueError("the member rate must lie in (0, 1)")
    d = separation(auroc)
    score = d * (outcomes == e.MEMBER) + noise
    log_odds = log(member_rate / (1.0 - member_rate)) + d * score - d * d / 2.0
    plus = 1.0 / (1.0 + np.exp(-log_odds))
    return _with_rate(plus, unresolved_rate)


def flip_probability(auroc):
    """Flip probability of the noisy oracle that gives the requested area under the ROC curve."""
    if not 0.5 <= auroc <= 1.0:
        raise ValueError("the area under the ROC curve must lie in [0.5, 1]")
    return 4.0 * (1.0 - auroc) / 3.0


def oracle_prior(outcomes, flip, alternative, auroc, confidence=0.9):
    """Noisy-oracle prior; ``flip`` and ``alternative`` are uniform numbers in [0, 1), one per cell."""
    outcomes = np.asarray(outcomes)
    flip, alternative = np.asarray(flip, dtype=float), np.asarray(alternative, dtype=float)
    if not (outcomes.shape == flip.shape == alternative.shape):
        raise ValueError("one pair of uniform numbers per cell")
    named = np.where(flip < flip_probability(auroc), (outcomes + 1 + (alternative >= 0.5)) % 3, outcomes)
    p = np.full((len(outcomes), 3), (1.0 - confidence) / 2.0)
    p[np.arange(len(outcomes)), named] = confidence
    return p


def with_unresolved_rate(probabilities, unresolved_rate):
    """Keep the membership probabilities and put a base rate on the unresolved answer."""
    p = np.asarray(probabilities, dtype=float)
    return _with_rate(np.clip(p[:, e.MEMBER], 0.0, 1.0), unresolved_rate)


def _with_rate(plus, unresolved_rate):
    if unresolved_rate is None:
        unknown = np.zeros_like(plus)
    else:
        if not 0.0 <= unresolved_rate < 1.0:
            raise ValueError("the unresolved rate must lie in [0, 1)")
        unknown = unresolved_rate * (1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def member_auroc(plus, outcomes):
    """Area under the ROC curve of ``plus`` for members against all other cells; None without both classes."""
    outcomes = np.asarray(outcomes)
    positive = outcomes == e.MEMBER
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(plus, kind="stable")
    ranks = np.empty(len(plus), dtype=float)
    sorted_values = np.asarray(plus)[order]
    start = 0
    while start < len(sorted_values):
        stop = start
        while stop + 1 < len(sorted_values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
