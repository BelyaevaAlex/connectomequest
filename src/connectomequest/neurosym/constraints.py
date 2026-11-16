"""Pilot B: soft hemibrain connectivity constraints on the frozen type predictor.

Let ``q0_j`` be the full distribution of the frozen ``mlp_s0`` over the 5551
hemibrain classes for cell ``j`` (class 0 = ``__OTHER__``). Every constraint is a
log-factor ``phi_j(c)`` on the typed classes ``c >= 1``; the posterior is

    q_j(c) = S_j * q0_j(c) exp(phi_j(c)) / sum_{c' >= 1} q0_j(c') exp(phi_j(c'))   (c >= 1)
    q_j(0) = q0_j(0)                                                               (unchanged)

with ``S_j = sum_{c >= 1} q0_j(c)``. The ``__OTHER__`` mass (the source of the
``unknown`` column, i.e. the unresolved forecast) is kept as is, so no constraint
can collapse it: hemibrain rules say nothing about crosswalk failures.

Rules (all from the whole hemibrain type-to-type matrix ``by[pre_type, post_type]``;
the evaluation views' labels are never used):

* B1 "pre type -> post type": the question's target type ``t`` is known, so
  ``phi_j(c) = beta * log(eps + R(c | t))`` with ``R(c | t)`` the share of the input
  weight of type ``t`` that comes from type ``c`` (``form="share"``, the declared
  rule), or ``R(t | c)`` the share of the output weight of type ``c`` that goes to
  ``t`` (``form="lik"``, the likelihood without the double-counted class prior).
* B2 neighbour consistency (mean field over the view's graph): every in-neighbour
  ``i`` and out-neighbour ``k`` of ``j`` sends a message built from its own
  ``q`` (``q0`` on the first sweep) and the hemibrain rule of that direction;
  messages are averaged with synapse weights per direction and summed over the
  two directions. ``form="mf"`` is the mean-field expectation
  ``sum_c' q_i(c') log(eps + R(c' | c))``; ``form="bp"`` is the sum-product
  message ``log sum_c' q_i(c') (eps + R(c' | c)) / (eps + R(c'))`` with ``R(c')``
  the hemibrain marginal of that direction (neutral when the neighbour is
  uninformative). A second sweep recomputes the messages from ``q1``.

From ``q_j`` the question prior is built exactly as in ``v35_build``:
``plus = sum_{c in A} q_j(c)``, ``unknown = q_j(0)``, ``nonmember = 1 - plus - unknown``
(stored as float32 and clipped as in ``common.family``). With ``beta = 0`` the
prior equals the frozen ``mlp_s0`` prior.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

import numpy as np
from scipy import sparse

from connectomequest.neurosym import common as c

V35 = c.ROOT / "scripts"


def _v35_build():
    if str(V35) not in sys.path:
        sys.path.insert(0, str(V35))
    import v35_build

    return v35_build


# --------------------------------------------------------------------------- full frozen MLP


def full_path(view):
    return c.CACHE / f"mlp-full-{view}.npy"


FROZEN_BLAS_THREADS = 16  # float32 GEMM rounding depends on the thread split; 16 reproduces outputs/v35 bit for bit


def build_full(view):
    """Exact frozen mlp_s0 probabilities (N x 5551 float32) for every node, as in v35_build."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(FROZEN_BLAS_THREADS):
        _build_full(view)


def _build_full(view):
    from connectomequest.annotation_v32.training import StructuralMLP

    b = _v35_build()
    ids = c.view_data(view)["ids"]
    with np.load(c.ROOT / b.VIEWS[view]["features"] / "features.npz", allow_pickle=False) as z:
        feature_ids, features = z["node_ids"], z["features"]
    features = features[b.rows_of(feature_ids, ids)]
    model = StructuralMLP.load(c.ROOT / b.MLP / "model.npz")
    temperature = json.loads((c.ROOT / b.MLP / "report.json").read_text())["temperature"]
    full = b.full_probabilities(model, temperature, features)
    path = full_path(view)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, full)
    os.replace(tmp, path)


@lru_cache(maxsize=None)
def full_mlp(view):
    """Memory-mapped (N, 5551) float32 frozen mlp_s0 distribution, rows aligned with ``view_data(view)['ids']``."""
    if not full_path(view).exists():
        build_full(view)
    return np.load(full_path(view), mmap_mode="r")


def rows_of(view, cells):
    return _v35_build().rows_of(c.view_data(view)["ids"], np.asarray(cells))


def check_full(view, families=None):
    """Largest |plus|/|unknown| difference to the stored mlp_s0 priors (float32 rounding expected: 0)."""
    d, q = c.benchmark(view), full_mlp(view)
    families = range(len(d["fam_target"])) if families is None else families
    worst_plus = worst_unknown = 0.0
    for f in families:
        t = int(d["fam_target"][f])
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        rows = rows_of(view, d["pre_cell"][lo:hi])
        codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
        plus = q[np.ix_(rows, codes)].astype(np.float64).sum(1).astype(np.float32)
        start = d["fam_plus_offset"][f]
        worst_plus = max(worst_plus, float(np.abs(plus - d["plus_mlp_s0"][start : start + hi - lo]).max()))
        worst_unknown = max(worst_unknown, float(np.abs(q[rows, 0] - d["unknown_mlp_s0"][lo:hi]).max()))
    return worst_plus, worst_unknown


# --------------------------------------------------------------------------- hemibrain rules


@lru_cache(maxsize=1)
def type_matrix():
    """Dense float64 ``by[pre_type, post_type]`` of the whole hemibrain (row/column 0 are zero)."""
    return np.asarray(c.hemibrain()["by_type"].todense(), dtype=np.float64)


@lru_cache(maxsize=2)
def shares(direction):
    """``R[c', c]``: share of type c's input (``"in"``) or output (``"out"``) weight that involves type c'.

    Types with no hemibrain weight in that direction back off to the hemibrain
    marginal of that direction. Returns (R, marginal) with columns summing to 1 over c' >= 1.
    """
    by = type_matrix()
    m = by if direction == "in" else by.T  # m[c', c]: weight between partner type c' and type c
    total = m.sum(0)
    marginal = m.sum(1) / m.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(total > 0, m / np.where(total > 0, total, 1.0), marginal[:, None])
    r.setflags(write=False)
    marginal.setflags(write=False)
    return r, marginal


def b1_logfactor(target_type, beta, eps, form="share"):
    """B1 log-factor over the 5551 classes for a question whose target has hemibrain type ``target_type``."""
    if beta == 0:
        return None
    r_in, _ = shares("in")
    r_out, _ = shares("out")
    # share: R(c | t) = share of t's input from c;  lik: R(t | c) = share of c's output going to t.
    rule = r_in[:, target_type] if form == "share" else r_out[target_type, :]
    phi = beta * np.log(eps + rule)
    phi[0] = 0.0
    return phi


# --------------------------------------------------------------------------- posterior and prior


def posterior(q0, phi):
    """Rows of q0 (float64) reweighted by log-factor ``phi`` (n x V or V) on c >= 1, keeping q(0)."""
    q0 = np.asarray(q0, dtype=np.float64)
    if phi is None:
        return q0
    phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), q0.shape)
    typed = q0[:, 1:]
    s = typed.sum(1, keepdims=True)
    logits = phi[:, 1:]
    w = typed * np.exp(logits - logits.max(1, keepdims=True))
    z = w.sum(1, keepdims=True)
    q = np.empty_like(q0)
    q[:, 0] = q0[:, 0]
    q[:, 1:] = np.where(z > 0, w / np.where(z > 0, z, 1.0) * s, typed)
    return q


def project(q, codes):
    """(plus, unknown) as stored by v35_build: float32 sums."""
    plus = q[:, np.asarray(codes)].sum(1).astype(np.float32)
    unknown = q[:, 0].astype(np.float32)
    return plus, unknown


def prior_columns(plus, unknown):
    """(member, nonmember, unresolved) rows exactly as ``common.family`` builds them."""
    plus = np.clip(np.asarray(plus, dtype=np.float32).astype(float), 0, 1)
    unknown = np.clip(np.asarray(unknown, dtype=np.float32).astype(float), 0, 1 - plus)
    return np.column_stack((plus, 1 - plus - unknown, unknown))


# --------------------------------------------------------------------------- B2 neighbour messages


def _normalized(adjacency):
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    deg = np.asarray(a.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)
    return sparse.diags(inv) @ a


@lru_cache(maxsize=None)
def neighbour_operators(view):
    """Row-normalized synapse-weighted (in, out) neighbour operators of the view graph (no labels)."""
    g = c.view_data(view)["graph"]
    return _normalized(g.T.tocsr()), _normalized(g.tocsr())


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _kernel(direction, form, eps):
    r, marginal = shares(direction)
    if form == "mf":
        return np.log(eps + r)
    if form == "bp":
        return (eps + r) / (eps + marginal[:, None])
    raise ValueError(form)


def _messages(q, direction, form, eps, chunk=16384):
    """Per-node message (N x V float32) from beliefs ``q`` (torch, N x V) in one direction."""
    import torch

    k = torch.as_tensor(_kernel(direction, form, eps), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)
    for s in range(0, q.shape[0], chunk):
        m = q[s : s + chunk] @ k
        out[s : s + chunk] = m if form == "mf" else torch.log(torch.clamp(m, min=1e-30))
    return out


def _spmm(op, dense):
    import torch

    op = op.tocsr().astype(np.float32)
    t = torch.sparse_csr_tensor(torch.as_tensor(op.indptr, dtype=torch.int64), torch.as_tensor(op.indices, dtype=torch.int64),
                                torch.as_tensor(op.data), size=op.shape, device=dense.device)
    return t @ dense


def b2_field(view, form, eps, beta=None, sweeps=1, rows=None):
    """Neighbour field ``G`` (len(rows) x V float32): the B2 log-factor is ``beta * G``.

    ``sweeps=1`` uses the neighbours' frozen q0; ``sweeps=2`` first updates every node
    with ``beta * G1`` (same beta) and recomputes the messages from those beliefs.
    """
    import torch

    torch.set_num_threads(min(10, torch.get_num_threads()))
    dev = _device()
    a_in, a_out = neighbour_operators(view)
    q = torch.as_tensor(np.ascontiguousarray(full_mlp(view)), dtype=torch.float32, device=dev)
    for sweep in range(sweeps):
        last = sweep == sweeps - 1
        ops = (a_in, a_out) if not last or rows is None else (a_in[rows], a_out[rows])
        g = _spmm(ops[0], _messages(q, "in", form, eps))
        g += _spmm(ops[1], _messages(q, "out", form, eps))
        if last:
            break
        if beta is None:
            raise ValueError("two sweeps need beta")
        q0 = torch.as_tensor(np.ascontiguousarray(full_mlp(view)), dtype=torch.float32, device=dev)
        logits = beta * g[:, 1:]
        w = q0[:, 1:] * torch.exp(logits - logits.max(1, keepdim=True).values)
        s = q0[:, 1:].sum(1, keepdim=True)
        q = q0.clone()
        q[:, 1:] = w / torch.clamp(w.sum(1, keepdim=True), min=1e-30) * s
        del q0, w, logits
    out = g.cpu().numpy()
    del g, q
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- priors of a set of families


def family_priors(view, families, b1=None, b2=None, field=None, field_rows=None):
    """{family: (n, 3) prior} under the constraints.

    ``b1 = dict(beta, eps, form)``; ``b2 = dict(beta)`` with ``field`` the B2 field rows
    (``field_rows`` maps view node index -> row of ``field``).
    """
    d, q_full = c.benchmark(view), full_mlp(view)
    targets = {}
    for f in families:
        targets.setdefault(int(d["fam_target"][f]), []).append(int(f))
    out = {}
    for t, fams in targets.items():
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        nodes = rows_of(view, d["pre_cell"][lo:hi])
        phi = None
        if b1 is not None and b1["beta"] != 0:
            phi = b1_logfactor(int(d["target_type"][t]), b1["beta"], b1["eps"], b1.get("form", "share"))
        if b2 is not None and b2["beta"] != 0:
            g = b2["beta"] * field[field_rows[nodes]].astype(np.float64)
            phi = g if phi is None else g + phi[None, :]
        q = posterior(q_full[nodes], phi)
        for f in fams:
            codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
            out[f] = prior_columns(*project(q, codes))
    return out


# --------------------------------------------------------------------------- within-question AUROC


def auroc(score, positive):
    """Mann-Whitney AUROC with average ranks over ties (as in scripts/v35_analyze.auroc)."""
    from scipy.stats import rankdata

    score, positive = np.asarray(score, dtype=float), np.asarray(positive) > 0
    ranks = rankdata(score)
    n_pos = positive.sum()
    n_neg = len(score) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_auroc(view, families, priors):
    """Per-family AUROC of plus (member vs not member, as v35_analyze.calibration) and member vs nonmember only.

    Returns (families_with_both_classes, auroc_vs_rest, auroc_member_vs_nonmember or nan).
    """
    d = c.benchmark(view)
    fams, rest, strict = [], [], []
    for f in families:
        t = int(d["fam_target"][f])
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        labels = d["pre_label"][lo:hi]
        codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
        member = np.isin(labels, codes)
        p = priors[int(f)][:, 0]
        if 0 < member.sum() < len(member):
            fams.append(int(f))
            rest.append(auroc(p, member))
            resolved = labels >= 0
            m = member[resolved]
            strict.append(auroc(p[resolved], m) if 0 < m.sum() < len(m) else np.nan)
    return np.asarray(fams), np.asarray(rest), np.asarray(strict)


def mean_auroc_ci(view, fams, values, seed=20261112, reps=2000):
    """Mean of per-family values with a type-cluster bootstrap 95 % interval."""
    d = c.benchmark(view)
    ok = np.isfinite(values)
    fams, values = fams[ok], values[ok]
    _, index = np.unique(d["target_type"][d["fam_target"][fams]], return_inverse=True)
    k = int(index.max()) + 1
    s = np.bincount(index, weights=values, minlength=k)
    n = np.bincount(index, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(r, minlength=k) for r in rng.integers(0, k, size=(reps, k))]).astype(float)
    boot = (counts @ s) / (counts @ n)
    return dict(value=float(values.mean()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)), n=int(len(values)))


def paired_auroc_diff(view, fams, new, old, seed=20261112, reps=2000):
    """Mean paired per-family AUROC difference new - old with a type-cluster bootstrap 95 % interval."""
    ok = np.isfinite(new) & np.isfinite(old)
    return mean_auroc_ci(view, fams[ok], (new - old)[ok], seed, reps)
