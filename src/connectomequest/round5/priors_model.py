"""Seeded in-connectome type predictor (round 5, workstream ``priors``).

Scenario: a curator has confirmed the reference type (a hemibrain type, or *outside the
vocabulary*) of the seed neurons, a fraction f of the evaluation connectome chosen by a fixed
pseudo-random function of the neuron id (``priors_common.seed_uniform``). For every other neuron
the predictor gives a distribution over the hemibrain vocabulary plus the extra class 0.

Evidence sources, each turned into a candidate list and per-candidate scores:

* ``raw``    label-free connectivity: the most similar seeds by cosine of sqrt synapse counts to
             and from every neuron (shared partner neurons);
* ``typed``  typed-partner profile (CBLAST-like): sqrt synapse counts from and to seeds of each
             type, idf-weighted, cosine against seeds;
* ``hemi``   the same typed profile against the type-level profiles of the hemibrain (all
             hemibrain neurons are typed), which can propose types that have no seed;
* ``degree`` the twelve degree/strength features (nearest seeds) and the frozen topology MLP;
* ``soma``   soma position, as is and folded at the midline (nearest seeds) — modality (ii);
* ``nt``     predicted neurotransmitter: log-likelihood ratio under the seeds of each type — (ii).

A small listwise scorer (softmax over the candidates of a neuron plus one *miss* slot) is trained
on 80% of the seeds, its temperature fitted on the other 20% (the calibration seeds). The miss
mass is spread over non-candidate classes in proportion to the frozen topology MLP. A seed's own
label never enters its own features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from connectomequest.neurosym import common as nc

from . import priors_common as pc
from . import priors_data as pdata

NONCONN = ("soma", "somaf", "nt")  # sources removed in the connectivity-only variant
K_RAW, K_TYPED, K_SOMA, K_DEG = 30, 30, 20, 20
TAKE = dict(raw=5, typed=5, hemi=5, soma=3, somaf=3, degree=3, mlp=3)
MAX_CAND = 40
SEED_CONFIDENCE = 0.99


@dataclass
class Seeds:
    view: str
    fraction: float
    ids: np.ndarray
    label: np.ndarray  # reference class per neuron (0 = outside the vocabulary); used only where ``is_seed``
    is_seed: np.ndarray
    is_calib: np.ndarray  # held-aside seeds (calibration)

    @property
    def is_train(self):
        return self.is_seed & ~self.is_calib


def seeds(view, fraction):
    ids = nc.view_data(view)["ids"]
    u = pc.seed_uniform(view, ids)
    is_seed = u < fraction
    is_calib = is_seed & (pc.calibration_uniform(view, ids) < pc.CALIBRATION_SHARE)
    return Seeds(view, fraction, ids, pdata.view_labels(view), is_seed, is_calib)


# --------------------------------------------------------------------------- evidence


def typed_profiles(graph, label, is_seed, vocab):
    """Rows: idf-weighted sqrt synapse counts from (in) and to (out) seeds of each class; class 0 kept apart."""
    n = graph.shape[0]
    rows = np.flatnonzero(is_seed)
    y = sparse.csr_matrix((np.ones(len(rows)), (rows, label[rows])), shape=(n, vocab))
    g = graph.tocsc().astype(np.float64)
    p_in = (g.T @ y).tocsr()  # synapses from seeds of class c onto v
    p_out = (g.tocsr() @ y).tocsr()  # synapses from v onto seeds of class c
    other_in = np.asarray(p_in[:, 0].todense()).ravel()
    other_out = np.asarray(p_out[:, 0].todense()).ravel()
    tot_in = np.asarray(p_in.sum(1)).ravel()
    tot_out = np.asarray(p_out.sum(1)).ravel()
    blocks, idfs = [], []
    for p in (p_in, p_out):
        p = _drop_class0(p)
        p.data = np.sqrt(p.data)
        idf = np.log((n + 1.0) / (1.0 + np.bincount(p.indices, minlength=vocab)))
        idfs.append(idf)
        blocks.append(p @ sparse.diags(idf))
    x = sparse.hstack(blocks).tocsr()
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    x = (sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ x).tocsr()
    return dict(x=x.astype(np.float32), other_in=other_in, other_out=other_out, tot_in=tot_in, tot_out=tot_out, idf=idfs)


def _drop_class0(p):
    p = p.tocsc(copy=True)
    p.data[p.indptr[0] : p.indptr[1]] = 0.0
    p.eliminate_zeros()
    return p.tocsr()


def sparse_topk(xq, xs, k, exclude=None, chunk=1000):
    """Top-k columns of xq @ xs.T per row (cosine for normalised rows); ``exclude[i]`` = column to skip."""
    n = xq.shape[0]
    xst = xs.T.tocsr()
    index = np.full((n, k), -1, dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        s = (xq[start:stop] @ xst).toarray()
        if exclude is not None:
            ex = exclude[start:stop]
            ok = ex >= 0
            s[np.flatnonzero(ok), ex[ok]] = -1.0
        kk = min(k, s.shape[1])
        top = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        val = np.take_along_axis(s, top, 1)
        order = np.argsort(-val, axis=1, kind="stable")
        index[start:stop, :kk] = np.take_along_axis(top, order, 1)
        sim[start:stop, :kk] = np.take_along_axis(val, order, 1)
    return index, np.maximum(sim, 0.0)


def hemibrain_profiles(vocab, idf_in, idf_out):
    """Type-level typed profiles of the hemibrain (mean per neuron of the type), same weighting as the view."""
    h = nc.hemibrain()
    by_type = h["by_type"].tocsc().astype(np.float64)  # [pre_type, post_type]
    counts = np.bincount(h["labels"][h["labels"] >= 0], minlength=vocab).astype(float)
    inv = sparse.diags(np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0))
    h_in = (inv @ by_type.T).tocsr()  # row t: synapses from class c onto a neuron of type t
    h_out = (inv @ by_type).tocsr()  # row t: synapses from a neuron of type t onto class c
    blocks = []
    for p, idf in ((h_in, idf_in), (h_out, idf_out)):
        p = _drop_class0(p)
        p.data = np.sqrt(p.data)
        blocks.append(p @ sparse.diags(idf))
    x = sparse.hstack(blocks).tocsr()
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    x = (sparse.diags(np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)) @ x).tocsr()
    x = (sparse.diags((np.arange(x.shape[0]) > 0).astype(float)) @ x).tocsr()  # the extra class has no profile
    x.eliminate_zeros()
    return x.astype(np.float32), counts


def kd_neighbours(points, query, k, exclude=None):
    """k nearest points (Euclidean) per query; ``exclude[i]`` = point index to skip (self)."""
    tree = cKDTree(points)
    kk = k + (1 if exclude is not None else 0)
    dist, idx = tree.query(query, k=kk, workers=1)
    if exclude is not None:
        drop = idx == exclude[:, None]
        keep = np.argsort(drop, axis=1, kind="stable")[:, :k]  # first k non-self
        dist, idx = np.take_along_axis(dist, keep, 1), np.take_along_axis(idx, keep, 1)
    return idx.astype(np.int64), dist


def first_distinct(labels, weights, m):
    """Top-m classes per row by summed neighbour weight (ties: earlier neighbour); -1 padded."""
    n, k = labels.shape
    out = np.full((n, m), -1, dtype=np.int64)
    if k == 0:
        return out
    valid = labels >= 0
    eq = (labels[:, :, None] == labels[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    vote = (eq * weights[:, None, :]).sum(2)  # vote of the class of neighbour j
    earlier = np.tril(np.ones((k, k), dtype=bool), -1)
    first = valid & ~(eq & earlier[None]).any(2)
    score = np.where(first, vote + 1e-9 * (k - np.arange(k))[None], -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :m]
    chosen = np.take_along_axis(labels, order, 1)
    ok = np.take_along_axis(score, order, 1) > -np.inf
    out[:, : order.shape[1]] = np.where(ok, chosen, -1)
    return out


def candidate_stats(cand, labels, weights):
    """Per candidate: weighted vote share, log(1 + first rank) (K+1 if absent), top-1 indicator."""
    n, k = labels.shape
    out = np.zeros(cand.shape + (3,), dtype=np.float32)
    if k == 0:
        out[..., 1] = np.log1p(1)
        return out
    total = np.maximum(weights.sum(1), 1e-12)
    eq = (cand[:, :, None] == labels[:, None, :]) & (cand[:, :, None] >= 0)
    out[..., 0] = (eq * weights[:, None, :]).sum(2) / total[:, None]
    rank = np.where(eq.any(2), eq.argmax(2), k)
    out[..., 1] = np.log1p(rank)
    out[..., 2] = eq[:, :, 0]
    return out


@dataclass
class Evidence:
    """Neighbour lists (seed rows) and query-level statistics for every neuron of the view."""

    view: str
    fraction: float
    sources: dict = field(default_factory=dict)  # name -> (labels (n,k), weights (n,k)) or extra arrays
    query: dict = field(default_factory=dict)
    hemi: tuple = None  # (index (n,k) classes, sim (n,k))
    nt: tuple = None  # (code per neuron, log P(nt|class) table (vocab, codes+1), log P(nt) (codes+1,))


def build_evidence(view, s, log=print):
    started = time.time()
    data = nc.view_data(view)
    graph = data["graph"]
    n = len(s.ids)
    vocab = len(nc.vocabulary())
    seed_rows = np.flatnonzero(s.is_seed)
    seed_pos = np.full(n, -1, dtype=np.int64)
    seed_pos[seed_rows] = np.arange(len(seed_rows))
    ev = Evidence(view, s.fraction)

    # raw connectivity: precomputed top-200 among all neurons, filtered to seeds
    raw = pdata.raw_neighbours(view)
    idx, sim = raw["index"].astype(np.int64), raw["sim"].astype(np.float64)
    keep = s.is_seed[idx]
    order = np.argsort(~keep, axis=1, kind="stable")[:, :K_RAW]
    r_idx = np.take_along_axis(idx, order, 1)
    r_ok = np.take_along_axis(keep, order, 1)
    r_sim = np.where(r_ok, np.take_along_axis(sim, order, 1), 0.0)
    ev.sources["raw"] = (np.where(r_ok, s.label[r_idx], -1), r_sim**2)
    ev.query["raw_max"] = r_sim[:, 0]
    ev.query["raw_mean5"] = r_sim[:, :5].mean(1)
    log(f"[{view} f={s.fraction}] raw neighbours ({time.time() - started:.0f} s)")

    # typed-partner profiles
    tp = typed_profiles(graph, s.label, s.is_seed, vocab)
    x = tp["x"]
    t_idx, t_sim = sparse_topk(x, x[seed_rows], K_TYPED, exclude=seed_pos)
    t_ok = t_sim > 0
    t_lab = np.where(t_ok, s.label[seed_rows[np.maximum(t_idx, 0)]], -1)
    ev.sources["typed"] = (t_lab, t_sim.astype(np.float64) ** 2)
    ev.query["typed_max"] = t_sim[:, 0]
    tot = tp["tot_in"] + tp["tot_out"]
    ev.query["seed_syn"] = np.log1p(tot)
    ev.query["other_share"] = (tp["other_in"] + tp["other_out"]) / np.maximum(tot, 1)
    ev.query["other_share_in"] = tp["other_in"] / np.maximum(tp["tot_in"], 1)
    log(f"[{view} f={s.fraction}] typed neighbours ({time.time() - started:.0f} s)")

    # hemibrain transfer
    hx, _ = hemibrain_profiles(vocab, *tp["idf"])
    h_idx, h_sim = sparse_topk(x, hx, 20)
    ev.hemi = (np.where(h_sim > 0, h_idx, -1), h_sim.astype(np.float64))
    log(f"[{view} f={s.fraction}] hemibrain transfer ({time.time() - started:.0f} s)")

    # degree/strength features (standardised) and the frozen topology MLP
    deg = pdata.degree_features(view)
    deg = (deg - deg.mean(0)) / np.maximum(deg.std(0), 1e-9)
    d_idx, d_dist = kd_neighbours(deg[seed_rows], deg, K_DEG, exclude=seed_pos)
    ev.sources["degree"] = (s.label[seed_rows[d_idx]], 1.0 / (1.0 + d_dist))
    ev.query["indeg"] = deg[:, 1]
    ev.query["outdeg"] = deg[:, 0]
    ev.query["instr"] = deg[:, 3]
    log(f"[{view} f={s.fraction}] degree neighbours ({time.time() - started:.0f} s)")

    # soma position, raw and folded at the midline
    mod = pdata.modalities(view)
    soma = pdata.soma_nm(view)
    has = np.isfinite(soma).all(1)
    ev.query["has_soma"] = has.astype(float)
    sp_rows = seed_rows[has[seed_rows]]
    sp_pos = np.full(n, -1, dtype=np.int64)
    sp_pos[sp_rows] = np.arange(len(sp_rows))
    mid = float(np.median(soma[has, 0]))
    scale = float(np.median(np.abs(soma[has] - np.median(soma[has], 0))))
    for name, pts in (("soma", soma), ("somaf", np.column_stack((np.abs(soma[:, 0] - mid), soma[:, 1], soma[:, 2])))):
        lab = np.full((n, K_SOMA), -1, dtype=np.int64)
        w = np.zeros((n, K_SOMA))
        q = np.flatnonzero(has)
        i, dist = kd_neighbours(pts[sp_rows], pts[q], K_SOMA, exclude=sp_pos[q])
        lab[q] = s.label[sp_rows[i]]
        w[q] = 1.0 / (1.0 + dist / (0.02 * scale))
        ev.sources[name] = (lab, w)
        if name == "soma":
            near = np.zeros(n)
            near[q] = np.log1p(dist[:, 0] / scale)
            ev.query["soma_near"] = near
    log(f"[{view} f={s.fraction}] soma neighbours ({time.time() - started:.0f} s)")

    # predicted neurotransmitter: counts of (class, transmitter) among the training seeds; the features use
    # log P(nt | class) with a Dirichlet(1) prior against the class-free log P(nt), leave-one-out for training rows
    nt = np.where(mod["nt"] >= 0, mod["nt"], len(pdata.NT_NAMES))
    codes = len(pdata.NT_NAMES) + 1
    tr = np.flatnonzero(s.is_train)
    table = np.zeros((vocab, codes))
    np.add.at(table, (s.label[tr], nt[tr]), 1.0)
    marginal = np.bincount(nt[tr], minlength=codes) + 1.0
    marginal /= marginal.sum()
    ev.nt = (nt, table, np.log(marginal))
    conf = mod["nt_conf"]
    ev.query["nt_conf"] = np.where(np.isfinite(conf), conf, 0.0)

    # class frequencies among the training seeds; a training row's own label (for leave-one-out features)
    ev.query["_seed_counts"] = np.bincount(s.label[tr], minlength=vocab).astype(float)
    own = np.full(n, -1, dtype=np.int64)
    own[tr] = s.label[tr]
    ev.query["_own"] = own
    log(f"[{view} f={s.fraction}] evidence ready ({time.time() - started:.0f} s)")
    return ev


# --------------------------------------------------------------------------- candidates and features

QUERY_CONN = ("raw_max", "raw_mean5", "typed_max", "seed_syn", "other_share", "other_share_in", "indeg", "outdeg", "instr")
QUERY_NONCONN = ("has_soma", "soma_near", "nt_conf")
KNN_SOURCES = ("raw", "typed", "degree", "soma", "somaf")


def source_names(conn_only):
    return [x for x in KNN_SOURCES if not (conn_only and x in NONCONN)]


def candidates(ev, rows, mlp_top, conn_only):
    """Candidate classes of the given neurons: union of the top classes of every source, plus class 0."""
    parts = [np.zeros((len(rows), 1), dtype=np.int64)]
    for name in source_names(conn_only):
        lab, w = ev.sources[name]
        parts.append(first_distinct(lab[rows], w[rows], TAKE[name]))
    parts.append(ev.hemi[0][rows, : TAKE["hemi"]])
    parts.append(mlp_top[rows, : TAKE["mlp"]].astype(np.int64))
    c = np.concatenate(parts, 1)
    c = np.sort(c, axis=1)
    dup = np.zeros_like(c, dtype=bool)
    dup[:, 1:] = c[:, 1:] == c[:, :-1]
    c = np.where(dup, -1, c)
    # class 0 first, then valid classes, then padding
    key = np.where(c == 0, -1, np.where(c < 0, np.iinfo(np.int64).max, c))
    c = np.take_along_axis(c, np.argsort(key, axis=1, kind="stable"), 1)
    return c[:, :MAX_CAND]


def features(ev, rows, cand, mlp_full, conn_only):
    """(rows, C, F) candidate features and (rows, Q) query features."""
    blocks = []
    for name in source_names(conn_only):
        lab, w = ev.sources[name]
        blocks.append(candidate_stats(cand, lab[rows], w[rows]))
    h_idx, h_sim = ev.hemi
    hi, hs = h_idx[rows], h_sim[rows]
    eq = (cand[:, :, None] == hi[:, None, :]) & (cand[:, :, None] > 0)
    hemi_sim = (eq * hs[:, None, :]).max(2)
    hemi_rank = np.where(eq.any(2), eq.argmax(2), hi.shape[1])
    blocks.append(np.stack((hemi_sim, np.log1p(hemi_rank)), -1))
    safe = np.maximum(cand, 0)
    m = np.take_along_axis(np.asarray(mlp_full[np.sort(rows)])[np.argsort(np.argsort(rows))], safe, 1)
    counts = ev.query["_seed_counts"]
    own = ev.query["_own"][rows]
    self_hit = ((cand == own[:, None]) & (own[:, None] >= 0)).astype(float)  # a training row never counts itself
    extra = [np.log(np.maximum(m, 1e-8)), np.log1p(counts[safe] - self_hit), (cand == 0).astype(float)]
    if not conn_only:
        nt, table, log_marginal = ev.nt
        mine = nt[rows][:, None]
        num = table[safe, mine] + 1.0 - self_hit
        den = table[safe].sum(-1) + table.shape[1] - self_hit
        extra.append(np.log(num / den) - log_marginal[mine])
    blocks.append(np.stack(extra, -1))
    cf = np.concatenate(blocks, -1).astype(np.float32)
    cf[cand < 0] = 0.0
    names = list(QUERY_CONN) + ([] if conn_only else list(QUERY_NONCONN))
    qf = np.column_stack([ev.query[k][rows] for k in names] + [(cand >= 0).sum(1)]).astype(np.float32)
    return cf, qf


# --------------------------------------------------------------------------- scorer


class Scorer:
    """Listwise softmax over candidates + one miss slot; a two-layer MLP on [candidate, query] features."""

    def __init__(self, hidden=64, epochs=40, lr=2e-3, weight_decay=1e-4, batch=1024, seed=0, threads=4):
        self.hidden, self.epochs, self.lr, self.wd, self.batch, self.seed, self.threads = hidden, epochs, lr, weight_decay, batch, seed, threads
        self.temperature = 1.0

    def _nets(self, fc, fq):
        import torch

        torch.manual_seed(self.seed)
        h = self.hidden
        cand = torch.nn.Sequential(torch.nn.Linear(fc + fq, h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        miss = torch.nn.Sequential(torch.nn.Linear(fq, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
        return cand, miss

    def _logits(self, cf, qf, mask):
        import torch

        b, c, _ = cf.shape
        z = torch.cat((cf, qf[:, None, :].expand(b, c, qf.shape[1])), -1)
        s = self.cand(z).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        return torch.cat((s, self.miss(qf)), 1)

    def _norm(self, cf, qf):
        return (cf - self.mu_c) / self.sd_c, (qf - self.mu_q) / self.sd_q

    def fit(self, cf, qf, mask, target, log=print):
        import torch

        torch.set_num_threads(self.threads)
        valid = mask.reshape(-1)
        flat = cf.reshape(-1, cf.shape[-1])[valid]
        self.mu_c, self.sd_c = flat.mean(0), np.maximum(flat.std(0), 1e-6)
        self.mu_q, self.sd_q = qf.mean(0), np.maximum(qf.std(0), 1e-6)
        cfn, qfn = self._norm(cf, qf)
        cfn[~mask] = 0.0
        self.cand, self.miss = self._nets(cf.shape[-1], qf.shape[-1])
        params = list(self.cand.parameters()) + list(self.miss.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        tc, tq, tm, tt = (torch.as_tensor(cfn), torch.as_tensor(qfn), torch.as_tensor(mask), torch.as_tensor(target))
        rng = np.random.default_rng(self.seed)
        n = len(target)
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch):
                j = torch.as_tensor(perm[start : start + self.batch])
                loss = torch.nn.functional.cross_entropy(self._logits(tc[j], tq[j], tm[j]), tt[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(j)
            sched.step()
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                log(f"  epoch {epoch}: loss {total / n:.4f}")
        return self

    def logits(self, cf, qf, mask, chunk=20000):
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(qf), chunk):
                c, q = self._norm(cf[start : start + chunk], qf[start : start + chunk])
                m = mask[start : start + chunk]
                c = np.where(m[..., None], c, 0.0).astype(np.float32)
                out.append(self._logits(torch.as_tensor(c), torch.as_tensor(q.astype(np.float32)), torch.as_tensor(m)).numpy())
        return np.concatenate(out) if out else np.zeros((0, cf.shape[1] + 1), dtype=np.float32)

    def fit_temperature(self, logits, target):
        """Temperature minimising the negative log-likelihood on held-aside rows (grid then refine)."""
        def nll(t):
            z = logits / t
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(target)), target].mean()

        grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
        best = grid[int(np.argmin([nll(t) for t in grid]))]
        fine = best * np.exp(np.linspace(-0.05, 0.05, 21))
        self.temperature = float(fine[int(np.argmin([nll(t) for t in fine]))])
        return self.temperature

    def probabilities(self, logits):
        z = logits / self.temperature
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)


def targets(cand, label):
    """Index of the true class among the candidates, or C (the miss slot)."""
    hit = cand == label[:, None]
    return np.where(hit.any(1), hit.argmax(1), cand.shape[1])


@dataclass
class Prediction:
    """Model distribution of every neuron: candidate classes, their probabilities, miss mass."""

    cand: np.ndarray
    prob: np.ndarray
    miss: np.ndarray
    temperature: float
    report: dict


def predict_view(view, fraction, conn_only, log=print, threads=4):
    s = seeds(view, fraction)
    ev = build_evidence(view, s, log=log)
    mlp = nc.mlp_topk(view)
    mlp_full = np.load(pc.ROOT / f"outputs/neurosym/cache/mlp-full-{view}.npy", mmap_mode="r")
    n = len(s.ids)
    all_rows = np.arange(n)
    started = time.time()
    cand = np.concatenate([candidates(ev, all_rows[a : a + 20000], mlp["top"], conn_only) for a in range(0, n, 20000)])
    cf_parts, qf_parts = [], []
    for a in range(0, n, 20000):
        rows = all_rows[a : a + 20000]
        cf, qf = features(ev, rows, cand[rows], mlp_full, conn_only)
        cf_parts.append(cf)
        qf_parts.append(qf)
    cf, qf = np.concatenate(cf_parts), np.concatenate(qf_parts)
    mask = cand >= 0
    log(f"[{view} f={fraction} conn={conn_only}] features {cf.shape} ({time.time() - started:.0f} s)")
    tr, ca = np.flatnonzero(s.is_train), np.flatnonzero(s.is_calib)
    y = targets(cand, s.label)
    scorer = Scorer(threads=threads).fit(cf[tr], qf[tr], mask[tr], y[tr], log=log)
    lg = scorer.logits(cf, qf, mask)
    scorer.fit_temperature(lg[ca], y[ca])
    p = scorer.probabilities(lg)
    report = dict(
        seeds=int(s.is_seed.sum()), train=int(len(tr)), calibration=int(len(ca)), temperature=scorer.temperature,
        candidate_recall_train=float((y[tr] < cand.shape[1]).mean()), candidate_recall_calibration=float((y[ca] < cand.shape[1]).mean()),
        mean_candidates=float(mask.sum(1).mean()),
    )
    log(f"[{view} f={fraction} conn={conn_only}] fitted; T={scorer.temperature:.3f} ({time.time() - started:.0f} s)")
    return s, Prediction(cand=cand, prob=p[:, :-1].astype(np.float32), miss=p[:, -1].astype(np.float32), temperature=scorer.temperature, report=report)


def top1(pred):
    """Most probable class (the miss slot never wins: its mass is spread over many classes)."""
    p = np.where(pred.cand >= 0, pred.prob, -1.0)
    j = p.argmax(1)
    return pred.cand[np.arange(len(j)), j], p[np.arange(len(j)), j]


def class_mass(pred, rows, codes, mlp_full):
    """Model mass on the class set ``codes`` for each neuron in ``rows`` (candidates + spread miss mass)."""
    codes = np.asarray(codes, dtype=np.int64)
    cand, prob, miss = pred.cand[rows], pred.prob[rows], pred.miss[rows]
    inset = np.isin(cand, codes) & (cand >= 0)
    direct = (prob * inset).sum(1)
    m = np.asarray(mlp_full[rows])
    safe = np.maximum(cand, 0)
    cand_mlp = np.where((cand > 0), np.take_along_axis(m, safe, 1), 0.0).sum(1)
    rest = np.maximum(1.0 - m[:, 0] - cand_mlp, 1e-12)
    set_mlp = m[:, codes[codes > 0]].sum(1) - np.where(inset & (cand > 0), np.take_along_axis(m, safe, 1), 0.0).sum(1)
    return direct + miss * np.clip(set_mlp / rest, 0.0, 1.0)
