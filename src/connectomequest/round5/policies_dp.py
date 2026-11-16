"""Bayes-optimal checking under an independent prior, by exact dynamic programming.

Model: the answers of the input cells are independent with the probabilities of the
prior, ``(p+, p-, p_unres)`` per cell. A policy checks cells one at a time and stops at
the first certificate (``L >= T`` or confirmed nonmember weight ``>= Z``); if neither
can be reached any more it checks every remaining cell, as every policy of the paper
does (the two-way verifier), so such a branch costs the number of unchecked cells.

State: ``(checked set S, member weight found, nonmember weight found)``; the expected
remaining checks ``V`` of a state depend only on it, because answers are independent.
``V`` is 0 for a certified state, ``|R|`` for a state from which no certificate can be
reached, and otherwise ``min_v 1 + sum_o p_v(o) V(child)``. The solver is a depth-first
search with a hash table over the reachable undecided states (numba). Ties between
optimal cells are broken toward weight order (heavier first, then ascending neuron
identifier), which fixes the realized order on the true answers.

``evaluate_stateful`` computes the exact expected cost under the prior of any policy
whose choice depends only on the state (weight order, AG, static orders, the one-step
lookahead); ``evaluate_tree`` enumerates the policy tree of a history-dependent policy
(recalibrated AG), carrying a copy of the policy along each branch.
"""

from __future__ import annotations

import copy

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is part of the environment
    njit = None

from connectomequest.annotation_v35 import engine as e

EMPTY = np.int64(-1)
TOL = 1e-12


def _encode_bits(n, total):
    wbits = int(total).bit_length() + 1
    if n + 2 * wbits > 62:
        raise ValueError("question too large for the state encoding")
    return wbits


if njit is not None:

    @njit(cache=True)
    def _probe(keys, key, mask_bits):
        h = (key * np.int64(-7046029254386353131)) & mask_bits
        while keys[h] != -1 and keys[h] != key:
            h = (h + 1) & mask_bits
        return h

    @njit(cache=True)
    def _grow(keys, vals, acts):
        cap = len(keys) * 2
        nk = np.full(cap, -1, dtype=np.int64)
        nv = np.zeros(cap, dtype=np.float64)
        na = np.full(cap, -1, dtype=np.int8)
        mb = np.int64(cap - 1)
        for i in range(len(keys)):
            if keys[i] != -1:
                h = _probe(nk, keys[i], mb)
                nk[h] = keys[i]
                nv[h] = vals[i]
                na[h] = acts[i]
        return nk, nv, na

    @njit(cache=True)
    def _solve(w, p, T, Z, rank, wbits, capacity, root):
        """Optimal expected checks from the empty state; returns (value, keys, vals, acts, states)."""
        n = len(w)
        nmask = 1 << n
        subset = np.zeros(nmask, dtype=np.int64)
        pop = np.zeros(nmask, dtype=np.int64)
        for m in range(1, nmask):
            low = m & (-m)
            b = 0
            while (1 << b) != low:
                b += 1
            subset[m] = subset[m ^ low] + w[b]
            pop[m] = pop[m ^ low] + 1
        total = subset[nmask - 1]
        keys = np.full(capacity, -1, dtype=np.int64)
        vals = np.zeros(capacity, dtype=np.float64)
        acts = np.full(capacity, -1, dtype=np.int8)
        mb = np.int64(capacity - 1)
        used = 0
        shift_m = n
        shift_n = n + wbits
        stack = np.zeros(64, dtype=np.int64)
        top = 0
        stack[0] = root
        top = 1
        while top > 0:
            key = stack[top - 1]
            h = _probe(keys, key, mb)
            if keys[h] == key:
                top -= 1
                continue
            mask = key & (nmask - 1)
            wm = (key >> shift_m) & ((1 << wbits) - 1)
            wn = key >> shift_n
            rem_cells = n - pop[mask]
            missing = False
            best = np.inf
            best_rank = n + 1
            best_v = -1
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                m2 = mask | (1 << v)
                rem2 = total - subset[m2]
                q = 1.0
                for o in range(3):
                    pr = p[v, o]
                    if pr <= 0.0:
                        continue
                    a2 = wm + (w[v] if o == 0 else 0)
                    b2 = wn + (w[v] if o == 1 else 0)
                    if a2 >= T or b2 >= Z:
                        continue
                    if a2 + rem2 < T and b2 + rem2 < Z:
                        q += pr * (rem_cells - 1)
                        continue
                    ck = m2 | (a2 << shift_m) | (b2 << shift_n)
                    hc = _probe(keys, ck, mb)
                    if keys[hc] == ck:
                        q += pr * vals[hc]
                    else:
                        missing = True
                        if top >= len(stack):
                            ns = np.zeros(len(stack) * 2, dtype=np.int64)
                            ns[: len(stack)] = stack
                            stack = ns
                        stack[top] = ck
                        top += 1
                if not missing:
                    tol = TOL * max(1.0, q)
                    if q < best - tol or (q <= best + tol and rank[v] < best_rank):
                        if q < best:
                            best = q
                        best_rank, best_v = rank[v], v
            if missing:
                continue
            top -= 1
            h = _probe(keys, key, mb)
            keys[h] = key
            vals[h] = best
            acts[h] = best_v
            used += 1
            if used * 2 > len(keys):
                keys, vals, acts = _grow(keys, vals, acts)
                mb = np.int64(len(keys) - 1)
        h = _probe(keys, root, mb)
        return vals[h], keys, vals, acts, used

    @njit(cache=True)
    def _lookup(keys, vals, acts, key):
        h = _probe(keys, key, np.int64(len(keys) - 1))
        if keys[h] != key:
            return np.nan, -1
        return vals[h], acts[h]


class Solution:
    """Optimal policy of one question under one prior."""

    def __init__(self, instance, probabilities, capacity=1 << 16):
        if njit is None:
            raise RuntimeError("numba is required for the exact solver")
        self.instance = instance
        self.p = e.check_probabilities(instance, probabilities)
        n = instance.size
        if n > 20:
            raise ValueError("exact solver limited to 20 cells")
        self.total = int(instance.weights.sum())
        self.wbits = _encode_bits(n, self.total)
        rank = np.empty(n, dtype=np.int64)
        rank[e.weight_order(instance)] = np.arange(n)
        self.rank = rank
        self._capacity = capacity
        value, self.keys, self.vals, self.acts, self.states = self._run(np.int64(0))
        self.value = float(value)
        self.resolves = 0

    def _run(self, root):
        inst = self.instance
        return _solve(inst.weights.astype(np.int64), self.p, np.int64(inst.a0), np.int64(inst.b0), self.rank, np.int64(self.wbits),
                      np.int64(self._capacity), np.int64(root))

    def key(self, mask, wm, wn):
        return np.int64(mask | (wm << self.instance.size) | (wn << (self.instance.size + self.wbits)))

    def action(self, mask, wm, wn):
        """Optimal cell and expected remaining checks at an undecided state.

        A state of probability zero under the prior (an answer the prior excludes) is
        solved on demand from that state: the optimal continuation of the remaining
        question, which is well defined because the answers are independent.
        """
        key = self.key(mask, wm, wn)
        value, act = _lookup(self.keys, self.vals, self.acts, key)
        if act < 0:
            self.resolves += 1
            _, keys, vals, acts, _ = self._run(key)
            value, act = _lookup(keys, vals, acts, key)
            if act < 0:
                raise KeyError("state is decided")
            self.keys, self.vals, self.acts = keys, vals, acts  # the sub-solution covers every later state of this path
        return int(act), float(value)

    def order(self, outcomes):
        """Payment order of the optimal policy on the given answers (to the certificate or exhaustion)."""
        inst = self.instance
        y = e.check_outcomes(inst, outcomes)
        mask = wm = wn = 0
        rem = self.total
        out = []
        n = inst.size
        while wm < inst.a0 and wn < inst.b0 and len(out) < n:
            if wm + rem < inst.a0 and wn + rem < inst.b0:
                rest = [int(j) for j in e.weight_order(inst) if not (mask >> int(j)) & 1]
                out.extend(rest)
                break
            v, _ = self.action(mask, wm, wn)
            out.append(v)
            mask |= 1 << v
            rem -= int(inst.weights[v])
            wm += int(inst.weights[v]) if y[v] == e.MEMBER else 0
            wn += int(inst.weights[v]) if y[v] == e.NONMEMBER else 0
        return np.asarray(out, dtype=np.intp)


# --------------------------------------------------------------------------- evaluation of given policies


def fork(policy):
    """Copy of an engine policy whose mutable arrays are independent (cheaper than ``deepcopy``)."""
    new = copy.copy(policy)
    for name, value in vars(policy).items():
        if isinstance(value, np.ndarray) and value.flags.writeable:
            setattr(new, name, value.copy())
    return new


def evaluate_stateful(instance, probabilities, choose):
    """Exact expected checks under the prior of a policy ``choose(available, a, b) -> cell``.

    ``choose`` may depend only on the state (static prior). Memoized over reachable states.
    """
    inst = instance
    p = e.check_probabilities(inst, probabilities)
    n, total, w = inst.size, int(inst.weights.sum()), inst.weights.astype(np.int64)
    memo = {}

    def value(mask, wm, wn, rem):
        if wm >= inst.a0 or wn >= inst.b0:
            return 0.0
        cells_left = n - bin(mask).count("1")
        if wm + rem < inst.a0 and wn + rem < inst.b0:
            return float(cells_left)
        key = (mask, wm, wn)
        if key in memo:
            return memo[key]
        available = np.array([not (mask >> i) & 1 for i in range(n)])
        v = int(choose(available, inst.a0 - wm, inst.b0 - wn))
        if not available[v]:
            raise RuntimeError("policy chose a checked cell")
        q = 1.0
        for o in range(3):
            if p[v, o] > 0:
                q += p[v, o] * value(mask | (1 << v), wm + (w[v] if o == e.MEMBER else 0), wn + (w[v] if o == e.NONMEMBER else 0), rem - w[v])
        memo[key] = q
        return q

    return value(0, 0, 0, total), len(memo)


def evaluate_tree(instance, probabilities, policy, limit=5_000_000):
    """Exact expected checks under the prior of an engine policy object (history-dependent allowed).

    The policy tree is enumerated depth first; each branch carries a deep copy of the policy,
    which receives ``observe`` exactly as in ``engine.run_adaptive``.
    """
    inst = instance
    p = e.check_probabilities(inst, probabilities)
    n, w = inst.size, inst.weights.astype(np.int64)
    nodes = [0]

    def value(pol, available, a, b, rem):
        if a <= 0 or b <= 0:
            return 0.0
        left = int(available.sum())
        if a > rem and b > rem:
            return float(left)
        nodes[0] += 1
        if nodes[0] > limit:
            raise RuntimeError("policy tree too large")
        v = int(pol.choose(a, b, available))
        q = 1.0
        nxt = available.copy()
        nxt[v] = False
        outcomes = [o for o in range(3) if p[v, o] > 0]
        for i, o in enumerate(outcomes):
            child = pol if i == len(outcomes) - 1 else fork(pol)
            child.observe(v, o)
            q += p[v, o] * value(child, nxt, a - int(inst.gp[v, o]), b - int(inst.gn[v, o]), rem - w[v])
        return q

    start = fork(policy)
    return value(start, np.ones(n, dtype=bool), inst.a0, inst.b0, int(w.sum())), nodes[0]
