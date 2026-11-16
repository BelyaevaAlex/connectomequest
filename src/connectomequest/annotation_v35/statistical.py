"""Risk-limited sequential certificate for a weighted category fraction (V35).

Cells are drawn with replacement with probability proportional to weight, so each
draw is an independent Bernoulli observation of a weight fraction. Two one-sided
betting tests (Waudby-Smith and Ramdas style test supermartingales with a
predictable plug-in Kelly bet) run in parallel:

* positive claim ``L_full >= T``: null ``L_full <= T - 1``; an unresolved response
  counts as a nonmember;
* negative claim ``U_full <= T - 1``: null ``U_full >= T``; an unresolved response
  counts as a member.

Either null is rejected when its wealth reaches ``1 / alpha`` (Ville's inequality),
so a wrong certificate has probability at most ``alpha``. The exact integer bounds
over the distinct paid cells are tracked as well and certify without risk. The cost
is the number of distinct cells drawn.
"""

from __future__ import annotations

import numpy as np

from .engine import MEMBER, NONMEMBER


def _kelly(estimate, null, upper):
    """Predictable Kelly fraction for a Bernoulli mean ``estimate`` against ``null``."""
    if null <= 0.0:
        return upper
    if null >= 1.0:
        return 0.0
    return float(np.clip((estimate - null) / (null * (1.0 - null)), 0.0, upper))


def sequential_certificate(weights, outcomes, threshold, *, alpha=0.05, rng, max_draw_factor=50):
    """Return ``(distinct_cells_paid, status, exact)`` for one random audit."""
    w = np.asarray(weights, dtype=np.int64)
    y = np.asarray(outcomes)
    total = int(w.sum())
    n = len(w)
    if n == 0 or np.any(w <= 0) or not 1 <= threshold <= total or not 0 < alpha < 1:
        raise ValueError("invalid audit")
    null_positive = (threshold - 1) / total
    null_negative = threshold / total
    cap_positive = 0.9 / null_positive if null_positive > 0 else np.inf
    cap_negative = 0.9 / (1.0 - null_negative) if null_negative < 1 else np.inf
    log_limit = np.log(1.0 / alpha)
    wealth_positive = wealth_negative = 0.0
    seen = np.zeros(n, dtype=bool)
    lower, nonmember = 0, 0
    distinct = 0
    sum_member = sum_upper = 0.0
    draws = rng.choice(n, size=max_draw_factor * n + 64, p=w / total)
    for t, j in enumerate(draws):
        member = y[j] == MEMBER
        upper = y[j] != NONMEMBER
        # Bets use only earlier draws; the 1/2 pseudo-observation keeps them predictable.
        estimate_member = (sum_member + 0.5) / (t + 1.0)
        estimate_upper = (sum_upper + 0.5) / (t + 1.0)
        if null_positive > 0:
            bet = _kelly(estimate_member, null_positive, cap_positive)
            wealth_positive += np.log1p(bet * (float(member) - null_positive))
        if null_negative < 1:
            bet = _kelly(1.0 - estimate_upper, 1.0 - null_negative, cap_negative)
            wealth_negative += np.log1p(bet * (null_negative - float(upper)))
        sum_member += float(member)
        sum_upper += float(upper)
        if not seen[j]:
            seen[j] = True
            distinct += 1
            lower += int(w[j]) if member else 0
            nonmember += int(w[j]) if not upper else 0
        if lower >= threshold:
            return distinct, "positive", True
        if nonmember >= total - threshold + 1:
            return distinct, "negative", True
        if null_positive > 0 and wealth_positive >= log_limit:
            return distinct, "positive", False
        if null_negative < 1 and wealth_negative >= log_limit:
            return distinct, "negative", False
        if distinct == n:
            break
    return n, "unresolved", True
