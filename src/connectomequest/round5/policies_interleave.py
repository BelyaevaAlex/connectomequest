"""Safe interleaving of a learned policy with weight order (round 5, workstream ``policies``).

``SafeInterleave`` alternates between a *given* policy ``P`` and weight order,
starting with ``P`` and skipping cells already checked. ``P`` runs on its own
history only (its *shadow* run): whenever it is ``P``'s turn, ``P`` is advanced
along exactly the trajectory it would follow alone - a cell it asks for that weight
order has already checked is answered from the record at no cost - and the first
cell of that trajectory not yet checked is checked. Because answers are fixed, the
shadow run is ``P``'s stand-alone run, so after ``2k`` checks the first ``k`` cells of
both stand-alone orders are checked; certificates are monotone in the checked set,
hence the cost is at most ``2 min(k_P, k_W)`` (proof:
``outputs/round5/policies/safe_interleave.md``).

``informed=True`` is a diagnostic without that guarantee: ``P`` also observes the
answers of the cells checked on weight order's turns.

Probabilities only order checks; every cost is decided by the frozen verifier.
"""

from __future__ import annotations

import numpy as np

from connectomequest.annotation_v35 import engine as e


def merge(primary, secondary, size):
    """Alternate between two payment orders (primary first), skipping cells already listed.

    When one order is exhausted the other continues. Returns a list of distinct cells.
    """
    seen = np.zeros(size, dtype=bool)
    out = []
    lists = (list(map(int, primary)), list(map(int, secondary)))
    pointers = [0, 0]

    def advance(t):
        i = pointers[t]
        while i < len(lists[t]) and seen[lists[t][i]]:
            i += 1
        pointers[t] = i
        return i < len(lists[t])

    turn = 0
    while len(out) < size:
        if not advance(turn):
            turn = 1 - turn
            if not advance(turn):
                break
        cell = lists[turn][pointers[turn]]
        seen[cell] = True
        out.append(cell)
        turn = 1 - turn
    return np.asarray(out, dtype=np.intp)


def safe_interleave(instance, outcomes, policy_order):
    """Shadow interleave: ``policy_order`` is P's stand-alone payment order on these answers."""
    return merge(policy_order, e.weight_order(instance), instance.size)


def informed_interleave(instance, outcomes, policy):
    """Diagnostic: P chooses on its turns from every answer seen so far (no 2x guarantee)."""
    y = e.check_outcomes(instance, outcomes)
    weight = e.weight_order(instance)
    a, b = instance.a0, instance.b0
    available = np.ones(instance.size, dtype=bool)
    order, pointer, turn = [], 0, 0
    while a > 0 and b > 0 and available.any():
        if turn == 0:
            j = policy.choose(a, b, available)
        else:
            while not available[weight[pointer]]:
                pointer += 1
            j = int(weight[pointer])
        if not available[j]:
            raise RuntimeError("policy returned an unavailable cell")
        available[j] = False
        order.append(j)
        a -= int(instance.gp[j, y[j]])
        b -= int(instance.gn[j, y[j]])
        policy.observe(j, int(y[j]))
        turn = 1 - turn
    return np.asarray(order, dtype=np.intp)
