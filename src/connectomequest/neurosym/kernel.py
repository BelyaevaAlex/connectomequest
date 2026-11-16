"""Pilot B, round 2: "same type => same connectivity" as a similarity kernel.

The kernel between two presynaptic cells ``j, k`` of one question is the cosine of
their connectivity profiles in the view graph (outputs and inputs concatenated,
square-root synapse counts, L2-normalized), thresholded at ``tau`` and raised to
``gamma``: ``K(j, k) = cos(j, k) ** gamma`` if ``cos >= tau`` else 0, ``K(j, j) = 0``.
It uses neither view labels nor the question.

* **B4** (two-way coupling) - ``KernelGreedy`` / ``KernelRecalibrated`` are the frozen
  Adaptive Greedy / Recalibrated Greedy whose ``observe`` also spreads the paid
  response to the unpaid cells. With the base beliefs ``m_k`` (member probability
  among resolved responses) and ``u_k`` (unresolved probability) - the static prior
  for AG, the in-session recalibration for recal - the policy uses

      m'_k = (m_k + lam * sum_{paid resolved j} K(j,k) [y_j = member]) / (1 + lam * sum_{paid resolved j} K(j,k))
      u'_k = (u_k + lam * sum_{paid j} K(j,k) [y_j = unresolved])   / (1 + lam * sum_{paid j} K(j,k))

  (a kernel-weighted pseudo-count update; cells without a similar paid cell keep the
  base belief bit for bit). With ``lam = 0`` the policies are the frozen ones.
* **B3'** (offline constraint) - a product of experts over similar cells of the same
  question on the full mlp_s0 distribution (all 5551 classes, ``__OTHER__`` included):
  ``log q_j = log q0_j + lam * sum_k K(j,k) log q0_k`` (+ optionally the B2 field on the
  typed classes), normalized; the question prior is then projected as in v35_build.

Probabilities only order payments; every cost is checked by the exact verifier.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import constraints as k


@lru_cache(maxsize=None)
def profiles(view):
    """CSR (N, 2N): row j = [sqrt outputs of j, sqrt inputs of j], L2-normalized (view graph only)."""
    g = c.view_data(view)["graph"].tocsr().astype(np.float64)
    prof = sparse.hstack([g.sqrt(), g.T.tocsr().sqrt()]).tocsr()
    norm = np.sqrt(np.asarray(prof.multiply(prof).sum(1)).ravel())
    prof = sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ prof
    return prof.tocsr()


def cosine(view, nodes):
    """Dense cosine similarity of the profiles of ``nodes`` (view node indices)."""
    p = profiles(view)[np.asarray(nodes)]
    return np.asarray((p @ p.T).todense(), dtype=np.float64)


def kernel(similarity, gamma, tau):
    """Thresholded, powered kernel with a zero diagonal."""
    s = np.clip(np.asarray(similarity, dtype=np.float64), 0.0, 1.0)
    out = np.where(s >= tau, s**gamma, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


_TARGET_CACHE = {}


def question_kernel(view, fam, gamma, tau):
    """Kernel of the presynaptic cells of ``fam`` (rows aligned with ``fam['cells']``)."""
    key = (view, fam["target"])
    if key not in _TARGET_CACHE:
        if len(_TARGET_CACHE) > 8:
            _TARGET_CACHE.clear()
        _TARGET_CACHE[key] = cosine(view, k.rows_of(view, fam["cells"]))
    return kernel(_TARGET_CACHE[key], gamma, tau)


# --------------------------------------------------------------------------- controls of the B4 mechanism


def cosine_without(view, nodes, column):
    """Cosine after zeroing one profile coordinate (e.g. the output to the question's target) and renormalizing."""
    p = profiles(view)[np.asarray(nodes)].tocsr(copy=True)
    p.data[p.indices == column] = 0.0
    p.eliminate_zeros()
    norm = np.sqrt(np.asarray(p.multiply(p).sum(1)).ravel())
    p = sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ p
    return np.asarray((p @ p.T).todense(), dtype=np.float64)


def permuted(kernel_matrix, seed, salt):
    """K[pi, pi] for a random permutation of the cells: same weight distribution, cell identity destroyed."""
    pi = np.random.default_rng([int(seed), int(salt)]).permutation(len(kernel_matrix))
    return kernel_matrix[np.ix_(pi, pi)]


def density(similarity, tau):
    """Fraction of off-diagonal pairs with similarity >= tau."""
    n = len(similarity)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, 1)
    return float(np.mean(similarity[iu] >= tau))


def weight_kernel(weights, sigma, share):
    """exp(-|log w_j - log w_k| / sigma) on the ``share`` most weight-similar pairs (ties by index), else 0."""
    lw = np.log(np.asarray(weights, dtype=np.float64))
    n = len(lw)
    out = np.zeros((n, n))
    if n < 2:
        return out
    iu = np.triu_indices(n, 1)
    s = np.exp(-np.abs(lw[iu[0]] - lw[iu[1]]) / sigma)
    m = int(round(share * len(s)))
    keep = np.argsort(-s, kind="stable")[:m]
    out[iu[0][keep], iu[1][keep]] = s[keep]
    return out + out.T


def oracle_kernel(labels):
    """ORACLE (uses reference labels): 1 if both cells are resolved and share the hemibrain type, else 0."""
    labels = np.asarray(labels)
    same = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
    out = same.astype(np.float64)
    np.fill_diagonal(out, 0.0)
    return out


SIDES = dict(flywire=("left", "right"), malecns=("L", "R"))


def contralateral(view, nodes):
    """Boolean matrix: one cell on the left and the other on the right (release annotation; diagnostics only)."""
    d = c.view_data(view)
    names = d["names"]["side"]
    left, right = (names.index(x) for x in SIDES[view])
    s = d["side_code"][np.asarray(nodes)]
    a, b = s[:, None], s[None, :]
    return ((a == left) & (b == right)) | ((a == right) & (b == left))


_CONTROL_CACHE = {}


def control_kernel(view, fam, cfg):
    """Kernel of a control (``cfg['kind']`` in cos / weight / oracle) for the cells of ``fam``."""
    key = (view, fam["target"], cfg.get("no_target", False))
    if key not in _CONTROL_CACHE:
        if len(_CONTROL_CACHE) > 8:
            _CONTROL_CACHE.clear()
        nodes = k.rows_of(view, fam["cells"])
        if cfg.get("no_target", False):
            target = int(k.rows_of(view, np.array([fam["target_cell"]]))[0])
            _CONTROL_CACHE[key] = cosine_without(view, nodes, target)  # output block: column = post node index
        else:
            _CONTROL_CACHE[key] = cosine(view, nodes)
    sim = _CONTROL_CACHE[key]
    kind = cfg["kind"]
    if kind == "cos":
        kern = kernel(sim, cfg["gamma"], cfg["tau"])
        if cfg.get("shuffle") is not None:
            kern = permuted(kern, cfg["shuffle"], fam["target"])
        if cfg.get("ipsilateral_only", False):
            kern = np.where(contralateral(view, k.rows_of(view, fam["cells"])), 0.0, kern)
        return kern
    if kind == "weight":
        return weight_kernel(fam["weights"], cfg["sigma"], density(sim, cfg["tau"]))
    if kind == "oracle":
        return oracle_kernel(fam["labels"])
    raise ValueError(kind)


# --------------------------------------------------------------------------- B4 policies


class _KernelCoupling:
    """Mixin: spread every paid response to unpaid cells through ``K``."""

    def _setup(self, kernel_matrix, lam):
        n = self.instance.size
        self._K = np.asarray(kernel_matrix, dtype=np.float64)
        if self._K.shape != (n, n):
            raise ValueError("kernel must be N x N")
        self._lam = float(lam)
        self._member = np.zeros(n)
        self._resolved = np.zeros(n)
        self._unresolved = np.zeros(n)
        self._paid = np.zeros(n)

    def _couple(self, cell, outcome, base):
        row = self._K[cell]
        self._paid += row
        if outcome == e.UNRESOLVED:
            self._unresolved += row
        else:
            self._resolved += row
            if outcome == e.MEMBER:
                self._member += row
        touched = np.flatnonzero(self._paid > 0)
        p = base.copy()
        if touched.size:
            b = base[touched]
            resolved = np.maximum(b[:, e.MEMBER] + b[:, e.NONMEMBER], 1e-12)
            m = b[:, e.MEMBER] / resolved
            u = b[:, e.UNRESOLVED]
            lam = self._lam
            m = (m + lam * self._member[touched]) / (1.0 + lam * self._resolved[touched])
            u = (u + lam * self._unresolved[touched]) / (1.0 + lam * self._paid[touched])
            p[touched, e.MEMBER] = (1 - u) * m
            p[touched, e.NONMEMBER] = (1 - u) * (1 - m)
            p[touched, e.UNRESOLVED] = u
        self.p = p


class KernelGreedy(_KernelCoupling, e.AdaptiveGreedy):
    """Adaptive Greedy whose beliefs about unpaid cells follow similar paid cells."""

    def __init__(self, instance, probabilities, kernel_matrix, lam):
        e.AdaptiveGreedy.__init__(self, instance, probabilities)
        self._setup(kernel_matrix, lam)
        self._base = self.p.copy()

    def observe(self, cell, outcome):
        if self._lam == 0:
            return
        self._couple(cell, outcome, self._base)


class KernelRecalibrated(_KernelCoupling, e.RecalibratedGreedy):
    """Recalibrated Greedy plus kernel coupling on top of the in-session recalibration."""

    def __init__(self, instance, probabilities, kernel_matrix, lam):
        e.RecalibratedGreedy.__init__(self, instance, probabilities)
        self._setup(kernel_matrix, lam)

    def observe(self, cell, outcome):
        e.RecalibratedGreedy.observe(self, cell, outcome)
        if self._lam == 0:
            return
        self._couple(cell, outcome, self.p)


def policy(base, instance, probabilities, kernel_matrix, lam):
    cls = KernelGreedy if base == "ag" else KernelRecalibrated
    return cls(instance, probabilities, kernel_matrix, lam)


# --------------------------------------------------------------------------- B3' offline prior


def pooled_posterior(q0, kernel_matrix, lam, field=None):
    """Product of experts over similar cells: log q_j = log q0_j + lam * sum_k K(j,k) log q0_k (+ field on c >= 1)."""
    q0 = np.asarray(q0, dtype=np.float64)
    if lam == 0 and field is None:
        return q0
    logq = np.log(np.maximum(q0, 1e-300))
    z = logq.copy()
    if lam != 0:
        z += lam * (np.asarray(kernel_matrix, dtype=np.float64) @ logq)
    if field is not None:
        z[:, 1:] += np.asarray(field, dtype=np.float64)[:, 1:]
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def family_priors(view, families, gamma, tau, lam, b2=None):
    """{family: (n, 3) prior} under B3' (and optionally the B2 field ``b2 = dict(beta, field, rows)``)."""
    d, q_full = c.benchmark(view), k.full_mlp(view)
    targets = {}
    for f in families:
        targets.setdefault(int(d["fam_target"][f]), []).append(int(f))
    out = {}
    for t, fams in targets.items():
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        nodes = k.rows_of(view, d["pre_cell"][lo:hi])
        field = None
        if b2 is not None and b2["beta"] != 0:
            field = b2["beta"] * b2["field"][b2["rows"][nodes]].astype(np.float64)
        kern = kernel(cosine(view, nodes), gamma, tau) if lam != 0 else None
        q = pooled_posterior(q_full[nodes], kern, lam, field)
        for f in fams:
            codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
            out[f] = k.prior_columns(*k.project(q, codes))
    return out
