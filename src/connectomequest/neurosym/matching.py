"""Pilot C: candidate hemibrain types for unresolved release types, checked by the frozen verifier.

Every input cell gets a candidate set ``S`` of hemibrain vocabulary codes, possibly
with ``BOTTOM`` (= "not a type of the hemibrain vocabulary", a nonmember of every
partner set), or ``None`` (no constraint). Reading a cell for a partner set ``A``
returns member if ``S`` is a subset of ``A`` (``BOTTOM`` not in ``S``), nonmember if
``S`` and ``A`` are disjoint and unresolved otherwise. If the true type is in ``S``
each such answer equals the true answer or is unresolved, so a certificate built from
candidate answers never contradicts the certificate built from the true answers.

Sources of constraints, in priority order:

1. ``declared``: the author hemibrain field of the cell, also when it is multi-valued
   ("A,B"), names hemibrain bodies ("hb123" or a bare id: their hemibrain label, BOTTOM if the body
   carries no vocabulary type, no constraint if it is absent from the graph) or names a
   type outside the vocabulary (its vocabulary subtypes ``name_x``/``namex`` if any,
   else BOTTOM). Names are mapped with the hemibrain ontology, then the release one.
2. ``transitive``: release type -> the other release (MaleCNS ``flywireType`` ->
   FlyWire ``cell_type``; FlyWire ``cell_type`` -> MaleCNS cells with that
   ``flywireType``) -> the reference label or the ``declared`` set of those cells.
3. ``connectivity``: cosine similarity of type-level input and output profiles over
   resolved partner types (view) against ``by_type`` (hemibrain), within-type edges
   and the candidate's own coordinate excluded. Candidates are the top ``k`` types
   within ``margin`` of the best; below ``floor`` either ``{BOTTOM}`` ("nothing is
   similar", variant ``full``) or no constraint (variant ``confident``).

Only development families are evaluated; thresholds are fitted on half 0.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from scipy import sparse

from connectomequest.annotation_v32.data import COMMON_GROUP_MAPPINGS, ontology_type
from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c

BOTTOM = -2
HEMIBRAIN = "hemibrain:v1.2.1"
OTHER = dict(malecns="flywire", flywire="malecns")
_SUBTYPE = re.compile(r"^(?:[_\-][A-Za-z0-9]|[A-Za-z])")
_BODY = re.compile(r"(?:hb)?(\d{6,})")  # hemibrain body id, with or without the "hb" prefix


# ---------------------------------------------------------------- answers

def answer(candidates, codes):
    """Response of a cell with candidate set ``candidates`` for the partner set ``codes``."""
    if candidates is None:
        return e.UNRESOLVED
    codes = set(int(x) for x in codes)
    if BOTTOM not in candidates and candidates <= codes:
        return e.MEMBER
    if not (candidates & codes):
        return e.NONMEMBER
    return e.UNRESOLVED


def answers(sets, codes):
    return np.array([answer(s, codes) for s in sets], dtype=np.intp)


def contains(candidates, label):
    """True type ``label`` (vocabulary code, or BOTTOM) lies in the candidate set."""
    return candidates is not None and int(label) in candidates


# ---------------------------------------------------------------- source 1: declared field

@lru_cache(maxsize=1)
def _lookup():
    return {name: i for i, name in enumerate(c.vocabulary()) if i}


@lru_cache(maxsize=1)
def _raw_names():
    """Vocabulary code -> hemibrain names it stands for (itself and coarsened group members)."""
    lookup = _lookup()
    out = {code: [name] for name, code in lookup.items()}
    for raw, group in COMMON_GROUP_MAPPINGS[HEMIBRAIN].items():
        if group in lookup:
            out[lookup[group]].append(raw)
    return out


@lru_cache(maxsize=None)
def subtypes(name):
    """Vocabulary codes whose hemibrain name refines ``name`` (``PAM06`` -> ``PAM06_a``, ``PAM06_b``)."""
    hits = set()
    for code, raws in _raw_names().items():
        for raw in raws:
            if raw.startswith(name) and len(raw) > len(name) and _SUBTYPE.match(raw[len(name):]):
                hits.add(code)
    return frozenset(hits)


def _name_set(name, release):
    lookup = _lookup()
    for mapped in (ontology_type(name, HEMIBRAIN), ontology_type(name, release)):
        if mapped in lookup:
            return frozenset({lookup[mapped]}), "in-vocabulary"
    kids = subtypes(ontology_type(name, HEMIBRAIN))
    if kids:
        return kids, "subtypes"
    return frozenset({BOTTOM}), "outside-vocabulary"


@lru_cache(maxsize=1)
def _hb_index():
    return {int(x): i for i, x in enumerate(c.hemibrain()["ids"].tolist())}


@lru_cache(maxsize=None)
def parse_declared(text, release):
    """(candidate set or None, kind) for one author hemibrain field."""
    if not text:
        return None, "none"
    tokens = [t for t in re.split(r"[,()\s]+", text) if t]
    bodies = [_BODY.fullmatch(t).group(1) for t in tokens if _BODY.fullmatch(t)]
    if bodies and len(bodies) == len(tokens):
        hb, index = c.hemibrain(), _hb_index()
        out = set()
        for body in bodies:
            if int(body) not in index:
                return None, "hemibrain-body-absent"
            label = int(hb["labels"][index[int(body)]])
            out.add(BOTTOM if label < 0 else label)
        return frozenset(out), "hemibrain-body"
    parts = [x.strip() for x in text.split(",") if x.strip()]
    out, kinds = set(), set()
    for part in parts:
        s, kind = _name_set(part, release)
        out |= s
        kinds.add(kind)
    if len(parts) > 1:
        return frozenset(out), "multi-valued"
    return frozenset(out), kinds.pop()


@lru_cache(maxsize=None)
def author_sets(view):
    """Per node of the view: {reference label} if resolved, else the declared-field set (or None); and the kind."""
    d = c.view_data(view)
    names = d["names"]["declared"]
    release = c.RELEASE[view]
    parsed = [parse_declared(n, release) for n in names]
    sets, kinds = [], []
    for label, code in zip(d["labels"].tolist(), d["declared_code"].tolist()):
        if label >= 0:
            sets.append(frozenset({label}))
            kinds.append("reference")
        elif code >= 0:
            sets.append(parsed[code][0])
            kinds.append(parsed[code][1])
        else:
            sets.append(None)
            kinds.append("none")
    return sets, kinds


# ---------------------------------------------------------------- source 2: transitive

def _union(items):
    known = [s for s in items if s is not None]
    return frozenset().union(*known) if known else None


def _split_names(text, known):
    if text in known:
        return [text]
    return [x.strip() for x in text.split(",") if x.strip()]


@lru_cache(maxsize=None)
def transitive_sets(view):
    """Release code -> union of author sets of the corresponding cells in the other release (None if none known)."""
    d, o = c.view_data(view), c.view_data(OTHER[view])
    other_sets, _ = author_sets(OTHER[view])
    if view == "malecns":  # flywireType -> FlyWire cell_type
        by_name = {}
        fw_names = o["names"]["release"]
        for code, s in zip(o["release_code"].tolist(), other_sets):
            if code >= 0:
                by_name.setdefault(fw_names[code], []).append(s)
        by_name = {k: _union(v) for k, v in by_name.items()}
        names, known = d["names"]["other_release"], set(by_name)
        per_type = {}
        for r, x in zip(d["release_code"].tolist(), d["other_release_code"].tolist()):
            if r < 0 or x < 0:
                continue
            parts = [by_name.get(n) for n in _split_names(names[x], known)]
            per_type.setdefault(r, []).append(None if any(p is None for p in parts) else _union(parts))
    else:  # FlyWire cell_type -> MaleCNS cells whose flywireType names it
        fw_names = d["names"]["release"]
        known = set(fw_names)
        mc_names = o["names"]["other_release"]
        by_name = {}
        for x, s in zip(o["other_release_code"].tolist(), other_sets):
            if x >= 0:
                for n in _split_names(mc_names[x], known):
                    by_name.setdefault(n, []).append(s)
        by_name = {k: _union(v) for k, v in by_name.items()}
        per_type = {}
        for r in np.unique(d["release_code"][d["release_code"] >= 0]).tolist():
            per_type[r] = [by_name.get(fw_names[r])]
    return {r: _union(v) for r, v in per_type.items() if _union(v) is not None}


# ---------------------------------------------------------------- source 3: connectivity

@lru_cache(maxsize=None)
def _profiles(view, hide=0.0, seed=0):
    """Type-level input/output profiles of release types (view) and hemibrain types over a shared coordinate set.

    ``hide`` > 0 is a stress test: that share of the resolved partner labels is hidden at random.
    """
    d, hb = c.view_data(view), c.hemibrain()
    v = len(c.vocabulary())
    g = d["graph"].tocoo()
    label, release = d["labels"], d["release_code"]
    if hide > 0:
        label = np.where(np.random.default_rng(seed).random(len(label)) < hide, -1, label)
    pre, post, w = g.row, g.col, g.data.astype(np.float64)
    cross = ~((release[pre] >= 0) & (release[pre] == release[post]))  # drop within-type edges
    nr = len(d["names"]["release"])

    def accumulate(typed_end, partner_end):
        keep = cross & (release[typed_end] >= 0) & (label[partner_end] > 0)
        return sparse.csr_matrix((w[keep], (release[typed_end][keep], label[partner_end][keep])), shape=(nr, v))

    q_in, q_out = accumulate(post, pre), accumulate(pre, post)
    present = np.zeros(v, dtype=bool)
    present[np.unique(label[label > 0])] = True
    present[0] = False
    by_type = hb["by_type"].tocsr().astype(np.float64)
    by_type.setdiag(0)
    by_type.eliminate_zeros()
    h_in, h_out = by_type.T.tocsr(), by_type.tocsr()  # row h: input over pre types / output over post types
    mask = sparse.diags(present.astype(np.float64))
    return dict(q=(q_in @ mask, q_out @ mask), h=(h_in @ mask, h_out @ mask), present=present)


def _cosine_block(q, h, rows, transform):
    """Cosine of query rows against every hemibrain type with coordinate h excluded for candidate h."""
    qb = transform(q[rows]).toarray()
    hm = transform(h)
    dot = (hm @ qb.T).T  # (rows, V)
    diag = np.asarray(hm.diagonal()).ravel()  # h[h, h]
    hn2 = np.asarray(hm.multiply(hm).sum(1)).ravel() - diag**2
    qn2 = (qb**2).sum(1)[:, None] - qb**2
    dot = dot - qb * diag[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / np.sqrt(np.maximum(qn2, 0) * np.maximum(hn2, 0)[None, :])
    has_q = qn2 > 0
    cos = np.where(has_q & (hn2[None, :] > 0), cos, 0.0)
    return np.nan_to_num(cos), has_q


def _sqrt(m):
    m = m.copy().astype(np.float64)
    m.data = np.sqrt(m.data)
    return m


@lru_cache(maxsize=None)
def similarity_top(view, release_codes, top=20, hide=0.0):
    """Top ``top`` hemibrain candidates by profile similarity for each release code: (codes, sims)."""
    p = _profiles(view, hide)
    rows = np.asarray(release_codes, dtype=np.int64)
    out_codes = np.zeros((len(rows), top), dtype=np.int32)
    out_sims = np.zeros((len(rows), top), dtype=np.float32)
    for start in range(0, len(rows), 512):
        block = rows[start : start + 512]
        sims, counts = 0.0, 0.0
        for q, h in zip(p["q"], p["h"]):
            cos, has_q = _cosine_block(q, h, block, _sqrt)
            sims = sims + cos
            counts = counts + has_q
        sims = sims / np.maximum(counts, 1)
        sims[:, 0] = -1.0  # __OTHER__ is never a candidate; every other hemibrain type is, present in the view or not
        idx = np.argpartition(-sims, top, axis=1)[:, :top]
        val = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-val, axis=1)
        out_codes[start : start + len(block)] = np.take_along_axis(idx, order, 1)
        out_sims[start : start + len(block)] = np.take_along_axis(val, order, 1)
    return out_codes, out_sims


def connectivity_set(codes, sims, k, margin, floor, below=frozenset({BOTTOM})):
    """Candidate set from ranked similarities: top ``k`` within ``margin`` of the best; ``below`` under ``floor``."""
    if sims[0] < floor:
        return below
    keep = [int(h) for h, s in zip(codes[:k], sims[:k]) if s >= sims[0] - margin]
    return frozenset(keep)


# ---------------------------------------------------------------- cells of the benchmark

@lru_cache(maxsize=None)
def rows_of(view):
    d = c.view_data(view)
    return {int(x): i for i, x in enumerate(d["ids"].tolist())}


def presynaptic(view, families):
    """Rows (in the view), weights and reference labels of the presynaptic cells of the families' targets."""
    b = c.benchmark(view)
    targets = np.unique(b["fam_target"][np.asarray(families)])
    idx = np.concatenate([np.arange(b["target_offset"][t], b["target_offset"][t + 1]) for t in targets])
    index = rows_of(view)
    rows = np.array([index[int(x)] for x in b["pre_cell"][idx]], dtype=np.int64)
    return rows, b["pre_weight"][idx].astype(np.int64), b["pre_label"][idx]


# ---------------------------------------------------------------- sessions

def statuses(fam, outcome_sets):
    """Full-information status per rho for each named outcome vector."""
    out = []
    for rho_index, threshold in c.thresholds(fam):
        inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
        out.append({name: e.full_information_status(inst, y) for name, y in outcome_sets.items()})
    return out


def session_costs(fam, outcomes, prior="mlp_s0"):
    """Per rho: (status, weight-order cost, recal cost) under ``outcomes``."""
    p = fam["priors"][prior]
    out = []
    for rho_index, threshold in c.thresholds(fam):
        inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
        status = e.full_information_status(inst, outcomes)
        weight, s1 = e.evaluate(inst, outcomes, e.weight_order(inst))
        recal, s2 = e.evaluate(inst, outcomes, e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p)))
        if not (s1 == s2 == status):
            raise AssertionError("order contradicts the full-information status")
        out.append((status, weight, recal))
    return out
