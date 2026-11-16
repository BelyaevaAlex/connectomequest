#!/usr/bin/env python3
"""Pilot C: cross-connectome type matching with hard constraints (development split only).

1. Fit the connectivity rule (floor for BOTTOM, top-k, margin) on masked resolved
   presynaptic types of half-0 development families.
2. Masked validation on half 1: every resolved presynaptic cell is read through its
   candidate set (sources 2-3 only, its own label hidden); coverage, |S| and wrong
   certificates against the certificates from the true answers. Also for the stratum
   of cells resolved through the author hemibrain field (release name not in the
   vocabulary), which is closer to the really unresolved types. A second masked
   validation reads the cells with a declared type outside the vocabulary (truth
   BOTTOM) through connectivity alone.
3. Really unresolved cells on half 1: share of uncertifiable sessions and recal cost
   (mlp_s0 prior) before / after, against the lenient reading of v35_lenient.

Three variants: ``author`` (sources 1-2), ``confident`` (1-3, connectivity used only
above the fitted floor, no constraint below it) and ``full`` (1-3, ``{BOTTOM}`` below
the floor: "nothing is similar"). A stress test hides a random share of the resolved
partner labels to see whether scarce resolved partners explain the low similarity of
the really unresolved types.
"""

from __future__ import annotations

import argparse
import json
import re
from multiprocessing import get_context

import numpy as np

from connectomequest.annotation_v32.data import ontology_type
from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import matching as m
from v35_diagnostics import VIEWS, annotation_rows, unresolved_reason

OUT = c.ROOT / "outputs/neurosym/C"
VARIANTS = ("author", "confident", "full")
HIDE = (0.3, 0.5)
FLOOR_QUANTILE = 0.025  # false-BOTTOM budget on masked half-0 weight
COVERAGE_TARGET = 0.95  # half-0 coverage the (k, margin) rule must reach
KS, MARGINS = (1, 2, 3, 5), (0.0, 0.02, 0.05, 0.1, 0.2, 1.0)
_G = {}


def wquantile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(np.asarray(w, float)[order]) / np.sum(w)
    return float(np.asarray(x)[order][min(np.searchsorted(cw, q), len(x) - 1)])


def masked_cells(view, half, hide=0.0):
    """Resolved presynaptic cells (with a release type) of the half's families and their ranked candidates."""
    d = c.view_data(view)
    rows, w, lab = m.presynaptic(view, c.development_families(view, half))
    rel = d["release_code"][rows]
    keep = (lab >= 0) & (rel >= 0)
    rows, w, lab, rel = rows[keep], w[keep], lab[keep], rel[keep]
    types = tuple(np.unique(rel).tolist())
    codes, sims = m.similarity_top(view, types, hide=hide)
    pos = {r: i for i, r in enumerate(types)}
    i = np.array([pos[int(r)] for r in rel])
    lookup = m._lookup()
    names = d["names"]["release"]
    by_name = np.array([ontology_type(names[r], c.RELEASE[view]) in lookup for r in rel])
    return dict(rows=rows, w=w, lab=lab, rel=rel, codes=codes[i], sims=sims[i], by_name=by_name)


def fit(view):
    """Connectivity rule on half 0 only."""
    x = masked_cells(view, 0)
    floor = wquantile(x["sims"][:, 0], x["w"], FLOOR_QUANTILE)
    grid = []
    for k in KS:
        for margin in MARGINS:
            sets = [m.connectivity_set(cs, ss, k, margin, floor) for cs, ss in zip(x["codes"], x["sims"])]
            cov = np.array([m.contains(s, l) for s, l in zip(sets, x["lab"])])
            size = np.array([len(s) for s in sets])
            grid.append(dict(k=k, margin=margin, coverage=float((x["w"] * cov).sum() / x["w"].sum()), mean_size=float((x["w"] * size).sum() / x["w"].sum())))
    ok = [g for g in grid if g["coverage"] >= COVERAGE_TARGET] or [max(grid, key=lambda g: g["coverage"])]
    best = min(ok, key=lambda g: (g["mean_size"], -g["coverage"]))
    return dict(floor=floor, k=best["k"], margin=best["margin"], half0=best, grid=grid)


def connectivity_sets(view, types, rule, confident):
    """Release code -> connectivity set; under the floor {BOTTOM} or, if ``confident``, no entry (no constraint)."""
    codes, sims = m.similarity_top(view, tuple(types))
    below = None if confident else frozenset({m.BOTTOM})
    out = {}
    for r, cs, ss in zip(types, codes, sims):
        s = m.connectivity_set(cs, ss, rule["k"], rule["margin"], rule["floor"], below)
        if s is not None:
            out[int(r)] = s
    return out


def source_of_type(view, r, variant, conn, trans):
    if r < 0:
        return None, "no-type"
    if int(r) in trans:
        return trans[int(r)], "transitive"
    if variant != "author" and int(r) in conn:
        return conn[int(r)], "connectivity"
    return None, "none"


def lenient_flags(view, rows):
    """v35_lenient: unresolved cells with a release-specific type and no correspondence read as nonmembers."""
    d = c.view_data(view)
    ann = annotation_rows(view)
    lookup = m._lookup()
    out = {}
    for r in rows.tolist():
        cell = int(d["ids"][r])
        if d["labels"][r] < 0 and cell not in out:
            out[cell] = unresolved_reason(*ann.get(cell, (None, None)), VIEWS[view]["release"], lookup) == "release-specific type without correspondence"
    return out


def build_sets(view, rule):
    """Per presynaptic cell id: candidate sets (real, per variant), masked sets (per variant), sources, lenient flag."""
    d = c.view_data(view)
    fams = c.development_families(view, 1)
    rows, w, lab = m.presynaptic(view, fams)
    rel = d["release_code"][rows]
    trans = m.transitive_sets(view)
    types = sorted(set(rel[rel >= 0].tolist()))
    conn = {False: connectivity_sets(view, types, rule, False), True: connectivity_sets(view, types, rule, True)}  # key: confident
    author, kind = m.author_sets(view)
    lenient = lenient_flags(view, rows)
    real = {v: {} for v in VARIANTS}
    masked = {v: {} for v in VARIANTS}
    source = {v: {} for v in VARIANTS}
    declared_only = {}  # ablation: source 1 alone
    bottom_cells = {}  # declared BOTTOM cells read by connectivity alone: a masked validation whose truth is BOTTOM
    for r, rr in zip(rows.tolist(), rel.tolist()):
        cell = int(d["ids"][r])
        if cell in real["author"]:
            continue
        declared_only[cell] = frozenset({int(d["labels"][r])}) if d["labels"][r] >= 0 else author[r]
        if d["labels"][r] < 0 and author[r] == frozenset({m.BOTTOM}) and rr >= 0:
            bottom_cells[cell] = dict(confident=conn[True].get(int(rr)), full=conn[False].get(int(rr)))
        for v in VARIANTS:
            cmap = conn[v == "confident"]
            s, src = source_of_type(view, rr, v, cmap, trans)
            if d["labels"][r] >= 0:
                real[v][cell] = frozenset({int(d["labels"][r])})
                source[v][cell] = "reference"
                masked[v][cell] = s
            else:
                masked[v][cell] = None
                if author[r] is not None:
                    real[v][cell], source[v][cell] = author[r], "declared:" + kind[r]
                else:
                    real[v][cell], source[v][cell] = s, src
    names = d["names"]["release"]
    lookup = m._lookup()
    hemibrain_field = {int(d["ids"][r]): bool(d["labels"][r] >= 0 and rr >= 0 and ontology_type(names[rr], c.RELEASE[view]) not in lookup)
                       for r, rr in zip(rows.tolist(), rel.tolist())}
    return dict(real=real, masked=masked, source=source, declared_only=declared_only, bottom_cells=bottom_cells, lenient=lenient, hemibrain_field=hemibrain_field, rows=rows, w=w, lab=lab, rel=rel)


def family_job(f):
    view, sets = _G["view"], _G["sets"]
    fam = c.family(view, f)
    codes = set(int(x) for x in fam["codes"])
    strict = fam["outcomes"]
    cells = fam["cells"].tolist()
    outcome = {"strict": strict}
    lenient = np.array([e.NONMEMBER if (y == e.UNRESOLVED and sets["lenient"].get(cid, False)) else y for cid, y in zip(cells, strict)], dtype=np.intp)
    outcome["lenient"] = lenient
    for v in VARIANTS:
        outcome["real:" + v] = m.answers([sets["real"][v][cid] for cid in cells], codes)
        mk = m.answers([sets["masked"][v][cid] for cid in cells], codes)
        resolved = strict != e.UNRESOLVED
        outcome["masked:" + v] = np.where(resolved, mk, e.UNRESOLVED)
        field = np.array([sets["hemibrain_field"][cid] for cid in cells])
        outcome["field:" + v] = np.where(field, mk, strict)
    outcome["ablation:declared"] = m.answers([sets["declared_only"][cid] for cid in cells], codes)
    bottom = sets["bottom_cells"]
    touched = np.array([cid in bottom for cid in cells])
    outcome["bottom:truth"] = np.where(touched, e.NONMEMBER, strict)
    for v in ("confident", "full"):
        read = m.answers([bottom[cid][v] if cid in bottom else None for cid in cells], codes)
        outcome["bottom:" + v] = np.where(touched, read, strict)
    status = m.statuses(fam, {k: y for k, y in outcome.items() if k.startswith(("masked:", "field:", "ablation:", "bottom:"))})
    costs = {k: m.session_costs(fam, outcome[k]) for k in ("strict", "lenient", *("real:" + v for v in VARIANTS))}
    rows = []
    for i in range(len(c.RHOS)):
        row = dict(status={**status[i], **{k: costs[k][i][0] for k in costs}}, touched_bottom=bool(touched.any()))
        row["weight"] = {k: costs[k][i][1] for k in costs}
        row["recal"] = {k: costs[k][i][2] for k in costs}
        rows.append(row)
    return rows


def run_sessions(view, sets, workers):
    fams = c.development_families(view, 1)
    _G.update(view=view, sets=sets)
    c.benchmark(view)
    with get_context("fork").Pool(workers) as pool:
        out = pool.map(family_job, list(map(int, fams)), chunksize=8)
    return fams, [r for g in out for r in g]


def size_stats(sets, w):
    size = np.array([len(s) for s in sets])
    return dict(median=float(np.median(size)), weighted_median=wquantile(size, w, 0.5), mean=float(np.mean(size)), share_le3=float((size <= 3).mean()))


def masked_report(view, rule, sets):
    """Coverage and |S| on half-1 masked cells, per source and per variant."""
    x = masked_cells(view, 1)
    trans = m.transitive_sets(view)
    out = {}
    conn_plain = connectivity_sets(view, sorted(set(x["rel"].tolist())), rule, False)
    conn_conf = connectivity_sets(view, sorted(set(x["rel"].tolist())), rule, True)
    per = dict(
        transitive=[trans.get(int(r)) for r in x["rel"]],
        connectivity=[conn_plain[int(r)] for r in x["rel"]],
        connectivity_above_floor=[conn_conf.get(int(r)) for r in x["rel"]],
    )
    for v in VARIANTS:
        per["variant:" + v] = [sets["masked"][v][int(c.view_data(view)["ids"][r])] for r in x["rows"]]
    for name, s in per.items():
        for stratum, mask in (("all", np.ones(len(s), bool)), ("hemibrain_field", ~x["by_name"])):
            have = np.array([q is not None for q in s]) & mask
            cov = np.array([m.contains(q, l) for q, l in zip(s, x["lab"])]) & mask
            bottom_only = np.array([q == frozenset({m.BOTTOM}) for q in s]) & mask
            w = x["w"]
            out.setdefault(name, {})[stratum] = dict(
                available_weight=float(w[have].sum() / w[mask].sum()),
                coverage_given_available=float(w[cov].sum() / max(w[have].sum(), 1)),
                bottom_only_weight=float(w[bottom_only].sum() / max(w[have].sum(), 1)),
                size=size_stats([q for q, hv in zip(s, have) if hv], w[have]) if have.any() else None,
            )
    # per-type coverage: mean over release types of the weight share covered
    for name, s in per.items():
        types = {}
        for q, l, r, w in zip(s, x["lab"], x["rel"], x["w"]):
            t = types.setdefault(int(r), [0, 0, 0])
            t[0] += w
            t[1] += w * m.contains(q, l)
            t[2] = t[2] or (q is not None)
        have = [t for t in types.values() if t[2]]
        out[name]["types"] = dict(n=len(types), available=len(have), coverage_unweighted=float(np.mean([t[1] / t[0] for t in have])) if have else 0.0)
    return out


def similarity_shift(view, sets):
    x = masked_cells(view, 1)
    real_rows = [(r, w, rr) for r, w, rr, l in zip(sets["rows"], sets["w"], sets["rel"], sets["lab"])
                 if l < 0 and sets["source"]["full"][int(c.view_data(view)["ids"][r])] == "connectivity"]
    types = tuple(sorted({int(rr) for _, _, rr in real_rows}))
    codes, sims = m.similarity_top(view, types)
    pos = {t: i for i, t in enumerate(types)}
    best = np.array([sims[pos[int(rr)], 0] for _, _, rr in real_rows])
    w = np.array([w for _, w, _ in real_rows])
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    return dict(masked_best_similarity={str(q): wquantile(x["sims"][:, 0], x["w"], q) for q in qs},
                unresolved_best_similarity={str(q): wquantile(best, w, q) for q in qs} if len(best) else None)


def partner_fraction(view, hide=0.0):
    """Release code -> share of its cross-type partner synapses whose partner is resolved."""
    d, p = c.view_data(view), m._profiles(view, hide)
    g, rel = d["graph"].tocoo(), d["release_code"]
    cross = ~((rel[g.row] >= 0) & (rel[g.row] == rel[g.col]))
    total = np.zeros(len(d["names"]["release"]))
    for end in (g.col, g.row):
        keep = cross & (rel[end] >= 0)
        np.add.at(total, rel[end][keep], g.data[keep])
    mass = np.asarray(p["q"][0].sum(1)).ravel() + np.asarray(p["q"][1].sum(1)).ravel()
    return mass / np.maximum(total, 1)


def stress_test(view, rule, sets):
    """Masked half-1 cells with a random share of resolved partner labels hidden (connectivity rule fixed)."""
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    real = [(rr, w) for r, w, rr, l in zip(sets["rows"], sets["w"], sets["rel"], sets["lab"])
            if l < 0 and sets["source"]["full"][int(c.view_data(view)["ids"][r])] == "connectivity"]
    rr, rw = np.array([a for a, _ in real]), np.array([b for _, b in real])
    out = dict(unresolved_connectivity_types=dict(resolved_partner_fraction={str(q): wquantile(partner_fraction(view)[rr], rw, q) for q in qs}))
    for hide in (0.0, *HIDE):
        x = masked_cells(view, 1, hide)
        s = [m.connectivity_set(cs, ss, rule["k"], rule["margin"], rule["floor"]) for cs, ss in zip(x["codes"], x["sims"])]
        cov = np.array([m.contains(q, l) for q, l in zip(s, x["lab"])])
        bottom = np.array([q == frozenset({m.BOTTOM}) for q in s])
        w = x["w"]
        out[f"masked_hide_{hide}"] = dict(
            resolved_partner_fraction={str(q): wquantile(partner_fraction(view, hide)[x["rel"]], w, q) for q in qs},
            best_similarity={str(q): wquantile(x["sims"][:, 0], w, q) for q in qs},
            coverage=float(w[cov].sum() / w.sum()), bottom_weight=float(w[bottom].sum() / w.sum()),
            coverage_above_floor=float(w[cov & ~bottom].sum() / max(w[~bottom].sum(), 1)),
        )
    return out


def _real_connectivity(view, sets, half=1):
    """Best similarity and weight of the really unresolved cells that only connectivity can read (half's families)."""
    d = c.view_data(view)
    trans, (author, _) = m.transitive_sets(view), m.author_sets(view)
    rows, w, lab = m.presynaptic(view, c.development_families(view, half))
    rel = d["release_code"][rows]
    keep = np.array([l < 0 and author[r] is None and rr >= 0 and int(rr) not in trans for r, l, rr in zip(rows, lab, rel)])
    types = tuple(sorted(set(rel[keep].tolist())))
    codes, sims = m.similarity_top(view, types)
    pos = {t: i for i, t in enumerate(types)}
    i = np.array([pos[int(r)] for r in rel[keep]], dtype=np.int64)
    return codes[i], sims[i], w[keep]


def shift_corrected(view, rule, sets):
    """Masked coverage of connectivity above the floor, reweighted by best-similarity bins to the really unresolved cells.

    Covariate-shift correction on the one covariate the rule uses; it still assumes that the
    unresolved cells above the floor carry a hemibrain type and behave like masked cells of
    equal similarity. Also the (k, margin) grid under this weighting on both halves.
    """
    floor = rule["floor"]
    edges = np.unique(np.r_[floor, np.arange(0.5, 1.0, 0.05)[np.arange(0.5, 1.0, 0.05) > floor], 1.01])
    out = dict(edges=edges.tolist(), grid=[])
    for half in (0, 1):
        x = masked_cells(view, half)
        _, rsims, rw = _real_connectivity(view, sets, half)
        above = rsims[:, 0] >= floor
        share = np.array([rw[above][(np.digitize(rsims[above, 0], edges) - 1) == b].sum() for b in range(len(edges) - 1)]) / rw[above].sum()
        bins = np.digitize(x["sims"][:, 0], edges) - 1
        for k in KS:
            for margin in sorted({0.0, 0.1, 1.0, rule["margin"]}):
                s = [m.connectivity_set(cs, ss, k, margin, floor, None) for cs, ss in zip(x["codes"], x["sims"])]
                hit = np.array([m.contains(q, l) for q, l in zip(s, x["lab"])])
                per_bin = [float((x["w"][bins == b] * hit[bins == b]).sum() / max(x["w"][bins == b].sum(), 1)) for b in range(len(edges) - 1)]
                cov = float(np.dot(share, per_bin))
                entry = dict(half=half, k=k, margin=margin, coverage=cov)
                if half == 1:
                    rc, rs, _ = _real_connectivity(view, sets, 1)
                    sizes = [len(m.connectivity_set(a, b, k, margin, floor, None)) for a, b in zip(rc[rs[:, 0] >= floor], rs[rs[:, 0] >= floor])]
                    entry["deployed_median_size"] = float(np.median(sizes)) if sizes else None
                out["grid"].append(entry)
                if half == 1 and k == rule["k"] and margin == rule["margin"]:
                    out.update(fitted_rule_coverage=cov, per_bin_coverage=per_bin, unresolved_bin_share=share.tolist())
    return out


def source_agreement(view, rule, sets):
    """Agreement of connectivity top-1 (above the floor) with the author-level sets, on masked vs really unresolved cells."""
    d = c.view_data(view)
    trans, (author, kind) = m.transitive_sets(view), m.author_sets(view)
    rows, w, lab = m.presynaptic(view, c.development_families(view, 1))
    rel = d["release_code"][rows]
    types = tuple(sorted(set(rel[rel >= 0].tolist())))
    codes, sims = m.similarity_top(view, types)
    pos = {t: i for i, t in enumerate(types)}
    acc = {}

    def add(key, weight, ok):
        a = acc.setdefault(key, [0, 0])
        a[0] += int(weight)
        a[1] += int(weight) * bool(ok)

    for r, ww, l, rr in zip(rows, w, lab, rel):
        if rr < 0:
            continue
        i = pos[int(rr)]
        top, above = int(codes[i, 0]), bool(sims[i, 0] >= rule["floor"])
        where = "masked" if l >= 0 else "unresolved"
        t = trans.get(int(rr))
        if t is not None and m.BOTTOM not in t and above:
            add(f"{where}: connectivity top-1 in transitive set", ww, top in t)
        if l < 0 and author[r] is not None:
            s = author[r]
            if m.BOTTOM not in s and above:
                add(f"unresolved: connectivity top-1 in declared set ({kind[r]})", ww, top in s)
            if s == frozenset({m.BOTTOM}):
                add("unresolved: declared BOTTOM but connectivity above the floor", ww, above)
            if t is not None and m.BOTTOM not in s and m.BOTTOM not in t:
                add("unresolved: transitive set intersects declared set", ww, s & t)
    return {k: dict(weight=a, rate=b / a) for k, (a, b) in sorted(acc.items())}


def sources_report(view, sets):
    """How the really unresolved presynaptic cells of half 1 were read: weight, cells and release types per source."""
    d = c.view_data(view)
    out = {}
    for v in VARIANTS:
        acc = {}
        for r, w, l, rr in zip(sets["rows"], sets["w"], sets["lab"], sets["rel"]):
            if l >= 0:
                continue
            cid = int(d["ids"][r])
            s = sets["real"][v][cid]
            src = sets["source"][v][cid]
            if s is None:
                src = "unresolved:" + src
            elif s == frozenset({m.BOTTOM}):
                src += " -> BOTTOM"
            a = acc.setdefault(src, dict(weight=0, cells=set(), types=set()))
            a["weight"] += int(w)
            a["cells"].add(cid)
            a["types"].add(int(rr))
        total = sum(a["weight"] for a in acc.values())
        out[v] = {k: dict(weight_share=a["weight"] / total, cells=len(a["cells"]), release_types=len(a["types"])) for k, a in sorted(acc.items())}
    return out


def session_report(view, fams, rows):
    truth = h.truth(view, fams)
    mine = np.array([h.STATUS[r["status"]["strict"]] for r in rows])
    if not np.array_equal(truth, mine):
        raise AssertionError("strict statuses differ from the frozen development run")
    frozen_recal, frozen_weight = h.baseline(view, fams, "mlp_s0:recal"), h.baseline(view, fams, "weight")
    recal_strict = np.array([r["recal"]["strict"] for r in rows])
    weight_strict = np.array([r["weight"]["strict"] for r in rows])
    if not (np.array_equal(recal_strict, frozen_recal) and np.array_equal(weight_strict, frozen_weight)):
        raise AssertionError("recomputed strict costs differ from the frozen development run")
    st = {k: np.array([r["status"][k] for r in rows]) for k in rows[0]["status"]}
    n = len(rows)
    out = dict(sessions=n, families=len(fams))
    # masked validation
    val = {}
    for key in [k for k in st if k.startswith(("masked:", "field:"))]:
        cand, true = st[key], st["strict"]
        cert = cand != "unresolved"
        wrong = cert & (cand != true)
        opposite = cert & (true != "unresolved") & (cand != true)
        val[key] = dict(certified=float(cert.mean()), wrong=int(wrong.sum()), wrong_share_of_certified=float(wrong.sum() / max(cert.sum(), 1)),
                        wrong_share_of_sessions=float(wrong.sum() / n), opposite=int(opposite.sum()),
                        certified_given_true_certified=float((cert & (true != "unresolved")).sum() / max((true != "unresolved").sum(), 1)))
    touched = np.array([r["touched_bottom"] for r in rows])
    for v in ("confident", "full"):
        cand, true = st["bottom:" + v], st["bottom:truth"]
        cert = cand != "unresolved"
        wrong = cert & (cand != true)
        val["bottom:" + v] = dict(sessions_touched=int(touched.sum()), certified=float(cert.mean()), wrong=int(wrong.sum()),
                                  wrong_share_of_certified=float(wrong.sum() / max(cert.sum(), 1)),
                                  wrong_share_of_touched_certified=float(wrong.sum() / max((cert & touched).sum(), 1)),
                                  opposite=int((cert & (true != "unresolved") & (cand != true)).sum()))
    out["masked_validation"] = val
    # really unresolved
    keys = ("strict", "lenient", *("real:" + v for v in VARIANTS))
    out["uncertifiable_share"] = {k: float((st[k] == "unresolved").mean()) for k in (*keys, "ablation:declared")}
    out["answers"] = {k: {s: float((st[k] == s).mean()) for s in ("positive", "negative", "unresolved")} for k in keys}
    out["contradictions_with_strict"] = {k: int(((st["strict"] != "unresolved") & (st[k] != "unresolved") & (st[k] != st["strict"])).sum()) for k in keys}
    out["disagreements_with_lenient"] = {k: dict(
        opposite=int(((st["lenient"] != "unresolved") & (st[k] != "unresolved") & (st[k] != st["lenient"])).sum()),
        lenient_only=int(((st["lenient"] != "unresolved") & (st[k] == "unresolved")).sum()),
        ours_only=int(((st["lenient"] == "unresolved") & (st[k] != "unresolved")).sum())) for k in keys if k.startswith("real:")}
    costs = {}
    for k in keys:
        x = np.array([r["recal"][k] for r in rows])
        y = np.array([r["weight"][k] for r in rows])
        mask = st[k] != "unresolved"
        costs[k] = dict(recal_to_weight=h.ratio(view, fams, x, y, mask), mean_recal_certified=float(x[mask].mean()), mean_weight_certified=float(y[mask].mean()), certified=int(mask.sum()))
    common = np.all([st[k] != "unresolved" for k in keys], axis=0)
    for k in keys:
        x = np.array([r["recal"][k] for r in rows])
        costs[k]["mean_recal_on_common_certified"] = float(x[common].mean())
    out["costs"] = costs
    out["common_certified_sessions"] = int(common.sum())
    return out


VARIANT_TEXT = dict(author="declared + transitive", confident="declared + transitive + connectivity above the floor (no constraint below it)",
                    full="declared + transitive + connectivity, {BOTTOM} below the floor")


# ---------------------------------------------------------------- verification (review round): no threshold changes

DRAWS = 40
BOOTSTRAP = 2000
NOBOTTOM_KINDS = ("multi-valued", "subtypes")  # author_nobottom: these declared kinds plus transitive, every set containing BOTTOM dropped


def _no_bottom(s):
    return None if s is None or m.BOTTOM in s else s


def nobottom_sets(view, sets):
    """author_nobottom: multi-valued fields, subtypes and transitive sets only; no BOTTOM-producing source."""
    d = c.view_data(view)
    trans, (author, kind) = m.transitive_sets(view), m.author_sets(view)
    real, masked, source = {}, {}, {}
    for r, rr in zip(sets["rows"].tolist(), sets["rel"].tolist()):
        cell = int(d["ids"][r])
        if cell in real:
            continue
        t = _no_bottom(trans.get(int(rr))) if rr >= 0 else None
        if d["labels"][r] >= 0:
            real[cell], source[cell], masked[cell] = frozenset({int(d["labels"][r])}), "reference", t
            continue
        masked[cell] = None
        if author[r] is not None and kind[r] in NOBOTTOM_KINDS and _no_bottom(author[r]) is not None:
            real[cell], source[cell] = author[r], "declared:" + kind[r]
        elif author[r] is None and t is not None:
            real[cell], source[cell] = t, "transitive"
        else:
            real[cell], source[cell] = None, "none"
    return real, masked, source


def _bins(floor):
    return np.unique(np.r_[-1.0, 0.2, 0.3, 0.4, floor, np.arange(0.5, 1.0, 0.05)[np.arange(0.5, 1.0, 0.05) > floor], 1.01])


def shift_draws(view, rule, sets, nb_source, seed=20261112, mass=False):
    """Per variant: DRAWS random maskings of resolved half-1 types whose similarity mix matches the really unresolved cells.

    The really unresolved cells read by transitive (resp. connectivity) in a variant set the
    target weight per best-similarity bin. A masked type is read by transitive with
    probability p_tr(bin) (only if the transitive set exists) and otherwise by connectivity
    alone with probability p_conn(bin); one common scale makes the expected masked weight per
    (source, bin) proportional to the target. With ``mass`` the scale is 1 instead: the
    expected masked weight per (source, bin) equals the target weight, capped at what the
    masked types offer, so a session reads about as much candidate weight as in deployment
    at the price of under-representing the scarce bins. Returns draws (list of {release
    code: set}) and the target share that no masked type can represent.
    """
    d = c.view_data(view)
    x = masked_cells(view, 1)
    trans = m.transitive_sets(view)
    edges = _bins(rule["floor"])
    types, first = np.unique(x["rel"], return_index=True)
    tw = np.bincount(np.searchsorted(types, x["rel"]), weights=x["w"])
    tbin = np.digitize(x["sims"][first, 0], edges) - 1
    codes, sims = x["codes"][first], x["sims"][first]
    nbin = len(edges) - 1
    real_rows = [(r, w, rr) for r, w, rr, l in zip(sets["rows"], sets["w"], sets["rel"], sets["lab"]) if l < 0 and rr >= 0]
    rtypes = tuple(sorted({int(rr) for _, _, rr in real_rows}))
    _, rsims = m.similarity_top(view, rtypes)
    rpos = {t: i for i, t in enumerate(rtypes)}
    out = {}
    for v in (*VARIANTS, "author_nobottom"):
        src = nb_source if v == "author_nobottom" else sets["source"][v]
        target = np.zeros((2, nbin))  # rows: transitive, connectivity
        for r, w, rr in real_rows:
            s = src[int(d["ids"][r])]
            if s in ("transitive", "connectivity"):
                target[int(s == "connectivity"), np.digitize(rsims[rpos[int(rr)], 0], edges) - 1] += w
        tr_avail = np.array([(_no_bottom(trans.get(int(t))) if v == "author_nobottom" else trans.get(int(t))) is not None for t in types])
        mass_tr = np.bincount(tbin[tr_avail], weights=tw[tr_avail], minlength=nbin)
        mass_all = np.bincount(tbin, weights=tw, minlength=nbin)
        uncovered = float((target[0][mass_tr == 0].sum() + target[1][mass_all == 0].sum()) / max(target.sum(), 1))
        target[0][mass_tr == 0] = 0
        target[1][mass_all == 0] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rt = np.where(mass_tr > 0, target[0] / mass_tr, 0.0)
            rc = np.where(mass_all > 0, target[1] / mass_all, 0.0)
        scale = 1.0 if mass else 1.0 / max(float(np.max(rt + rc)), 1e-12)
        p_tr, p_conn = scale * rt, scale * rc
        over = np.maximum(p_tr + p_conn, 1.0)  # cap: a type is masked at most once per draw
        p_tr, p_conn = p_tr / over, p_conn / over
        represented = float((np.minimum(p_tr * mass_tr, target[0]).sum() + np.minimum(p_conn * mass_all, target[1]).sum()) / max(target.sum(), 1))
        conf = v == "confident"
        reading_c = [m.connectivity_set(cs, ss, rule["k"], rule["margin"], rule["floor"], None if conf else frozenset({m.BOTTOM})) for cs, ss in zip(codes, sims)]
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(DRAWS):
            u = rng.random(len(types))
            draw = {}
            for i, t in enumerate(types.tolist()):
                b = tbin[i]
                if tr_avail[i] and u[i] < p_tr[b]:
                    s = trans[int(t)]
                    draw[int(t)] = _no_bottom(s) if v == "author_nobottom" else s
                elif u[i] < (p_tr[b] if tr_avail[i] else 0.0) + p_conn[b]:
                    draw[int(t)] = reading_c[i]
            draws.append(draw)
        masked_share = float(np.mean([sum(tw[np.searchsorted(types, t)] for t in dr) for dr in draws]) / tw.sum())
        out[v] = dict(draws=draws, uncovered_target_share=uncovered, mean_masked_weight_share=masked_share,
                      target_weight=dict(transitive=float(target[0].sum()), connectivity=float(target[1].sum())),
                      target_weight_share_of_resolved=float(target.sum() / tw.sum()), expected_masked_over_target=represented * float(target.sum() > 0))
    return out


def verify_job(f):
    view, g = _G["view"], _G["verify"]
    fam = c.family(view, f)
    codes = set(int(x) for x in fam["codes"])
    strict = fam["outcomes"]
    cells = fam["cells"].tolist()
    resolved = strict != e.UNRESOLVED
    rel = g["rel_of"][np.array([g["row_of"][cid] for cid in cells])]
    outcome = {"real:author_nobottom": m.answers([g["nb_real"][cid] for cid in cells], codes)}
    mk = m.answers([g["nb_masked"][cid] for cid in cells], codes)
    field = np.array([g["field"][cid] for cid in cells])
    outcome["masked:author_nobottom"] = np.where(resolved, mk, e.UNRESOLVED)
    outcome["field:author_nobottom"] = np.where(field, mk, strict)
    touched = {}
    for v, draws in g["draws"].items():
        for i, draw in enumerate(draws):
            hit = resolved & np.array([int(r) in draw for r in rel])
            touched[(v, i)] = bool(hit.any())
            if hit.any():
                read = m.answers([draw[int(r)] if h else None for r, h in zip(rel, hit)], codes)
                outcome[f"shift:{v}:{i}"] = np.where(hit, read, strict)
    status = m.statuses(fam, {**outcome, "strict": strict})
    return [dict(status=s, touched=touched) for s in status]


def opposite_cases(view, sets):
    """Sessions whose candidate certificate is opposite to the true one (masked:*, field:*, bottom:*) and the cells that flipped."""
    d = c.view_data(view)
    voc = c.vocabulary()
    rows_of, rel_names = m.rows_of(view), d["names"]["release"]
    name = lambda s: None if s is None else sorted("BOTTOM" if x == m.BOTTOM else voc[x] for x in s)
    out = []
    for f in c.development_families(view, 1).tolist():
        fam = c.family(view, int(f))
        codes = set(int(x) for x in fam["codes"])
        strict, cells = fam["outcomes"], fam["cells"].tolist()
        resolved = strict != e.UNRESOLVED
        field = np.array([sets["hemibrain_field"][cid] for cid in cells])
        bottom = sets["bottom_cells"]
        touched = np.array([cid in bottom for cid in cells])
        cand = {}
        for v in VARIANTS:
            mk = m.answers([sets["masked"][v][cid] for cid in cells], codes)
            cand["masked:" + v] = (np.where(resolved, mk, e.UNRESOLVED), [sets["masked"][v][cid] for cid in cells])
            cand["field:" + v] = (np.where(field, mk, strict), [sets["masked"][v][cid] for cid in cells])
        truth_b = np.where(touched, e.NONMEMBER, strict)
        for v in ("confident", "full"):
            read = [bottom[cid][v] if cid in bottom else None for cid in cells]
            cand["bottom:" + v] = (np.where(touched, m.answers(read, codes), strict), read)
        for rho, threshold in c.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            for key, (y, read) in cand.items():
                ref = truth_b if key.startswith("bottom:") else strict
                s, t = e.full_information_status(inst, y), e.full_information_status(inst, ref)
                if s != "unresolved" and t != "unresolved" and s != t:
                    flipped = [dict(release_type=rel_names[d["release_code"][rows_of[cid]]], true=("BOTTOM" if key.startswith("bottom:") else voc[int(fam["labels"][j])]),
                                    candidates=name(read[j]), weight=int(fam["weights"][j]), read=["member", "nonmember", "unresolved"][int(y[j])])
                               for j, cid in enumerate(cells) if y[j] != ref[j]]
                    out.append(dict(key=key, family=int(f), rho=rho, target_type=voc[fam["target_type"]], partner_set=sorted(voc[x] for x in codes),
                                    true=t, candidate=s, threshold=int(threshold), total=int(fam["weights"].sum()), flipped_cells=flipped))
    return out


def _cluster_rate(view, fams, wrong, cert):
    index, counts, k = h._clusters(view, tuple(int(f) for f in fams))
    sw = np.bincount(index, weights=wrong, minlength=k)
    sc = np.bincount(index, weights=cert, minlength=k)
    boot = (counts @ sw) / np.maximum(counts @ sc, 1)
    return dict(value=float(sw.sum() / max(sc.sum(), 1)), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)),
                wrong=int(sw.sum()), certified_touched=int(sc.sum()))


def _one_edit(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def bottom_sources_check(view, sets):
    """What can be checked about the BOTTOM sources of 'author' on the really unresolved half-1 cells (weight shares)."""
    import pyarrow.parquet as pq

    d = c.view_data(view)
    author, kind = m.author_sets(view)
    vocab = [n for n in c.vocabulary()[1:]]
    hbn = pq.read_table(c.ROOT / "data/raw/hemibrain/neurons.parquet", columns=["bodyId", "type", "instance"]).to_pandas().set_index("bodyId")
    acc = {}

    def add(key, w):
        acc[key] = acc.get(key, 0) + int(w)

    names = d["names"]["declared"]
    typo = {}
    bottom = frozenset({m.BOTTOM})
    for r, w, l in zip(sets["rows"], sets["w"], sets["lab"]):
        if l >= 0:
            continue
        cid = int(d["ids"][r])
        if sets["source"]["author"][cid] == "transitive" and sets["real"]["author"][cid] == bottom:
            add("transitive -> BOTTOM (other release's declared BOTTOM)", w)
        if author[r] != bottom:
            continue
        text = names[d["declared_code"][r]]
        if kind[r] == "hemibrain-body":
            bodies = [int(b) for b in re.findall(r"\d{6,}", text)]
            types = hbn.loc[bodies, "type"]
            inst = hbn.loc[bodies, "instance"]
            add("hemibrain body -> BOTTOM: neuPrint type null", w if types.isna().all() else 0)
            add("hemibrain body -> BOTTOM: neuPrint type null, provisional instance name", w if (types.isna().all() and inst.notna().any()) else 0)
            add("hemibrain body -> BOTTOM: total", w)
        else:
            parts = [x.strip() for x in text.split(",") if x.strip()]
            if text not in typo:
                typo[text] = any(_one_edit(p, v) for p in parts for v in vocab)
            add("name outside vocabulary -> BOTTOM: total", w)
            add("name outside vocabulary -> BOTTOM: one edit from a vocabulary name (possible typo)", w if typo[text] else 0)
            sensory = all(p.startswith(("ORN_", "HRN_", "TRN_")) for p in parts)
            add("name outside vocabulary -> BOTTOM: sensory ORN/HRN/TRN", w if sensory else 0)
            add("name outside vocabulary -> BOTTOM: possible typo, not sensory", w if typo[text] and not sensory else 0)
    unresolved_weight = int(sets["w"][sets["lab"] < 0].sum())
    out = {k: dict(weight=v, share_of_unresolved_weight=v / unresolved_weight) for k, v in sorted(acc.items())}
    out["possible_typo_names"] = sorted(k for k, v in typo.items() if v)
    return out


def hemibrain_label_none():
    """Meaning of label None in the prepared hemibrain labels, checked against neuPrint type/instance."""
    import pyarrow.parquet as pq

    lab = pq.read_table(c.ROOT / "inputs/v32/prepared-common-v1/hemibrain/private/labels.parquet").to_pandas()
    hbn = pq.read_table(c.ROOT / "data/raw/hemibrain/neurons.parquet", columns=["bodyId", "type", "instance"]).to_pandas()
    j = lab.merge(hbn, left_on="cell_id", right_on="bodyId", how="left")
    none, typed = j[j["label"].isna()], j[j["label"].notna()]
    vocab = c.vocabulary()
    from connectomequest.annotation_v32.data import ontology_type

    mapped = {ontology_type(t, m.HEMIBRAIN) for t in hbn["type"].dropna() if t}
    return dict(
        bodies=len(j), label_none=len(none), label_zero=int((j["label"] == 0).sum()), unmatched_in_neuprint=int(j["bodyId"].isna().sum()),
        label_none_with_type_null=int(none["type"].isna().sum()), label_none_with_instance=int(none["instance"].notna().sum()),
        label_none_instance_examples=none["instance"].dropna().head(8).tolist(),
        typed_label_equals_mapped_type=int(sum(ontology_type(t, m.HEMIBRAIN) == vocab[int(l)] for t, l in zip(typed["type"], typed["label"]))),
        typed_bodies=len(typed), neuprint_types_in_graph=len(mapped), neuprint_types_outside_vocabulary=len(mapped - set(vocab)),
        source_accounting="v35_build.source_statistics: by_type (the partner-set weight) counts typed presynaptic bodies only, while source_total "
                          "(the denominator of every threshold) counts all presynaptic bodies, untyped ones included; an untyped body thus enters the "
                          "source questions exactly as a nonmember of every partner set.",
    )


def verify(workers):
    path = OUT / "matching.json"
    result = json.loads(path.read_text())
    result["protocol"]["variants"] = VARIANT_TEXT
    ver = dict(note="Review round: four measurements added without changing any threshold; the original verdict is kept as pre-specified.", views={})
    for view in ("malecns", "flywire"):
        rule = fit(view)
        stored = result["views"][view]["rule"]
        if (rule["k"], rule["margin"], round(rule["floor"], 9)) != (stored["k"], stored["margin"], round(stored["floor"], 9)):
            raise AssertionError("refitted rule differs from the stored one")
        sets = build_sets(view, rule)
        nb_real, nb_masked, nb_source = nobottom_sets(view, sets)
        draws = {f"{mode}:{v}": x for mode, flag in (("distribution", False), ("mass", True))
                 for v, x in shift_draws(view, rule, sets, nb_source, mass=flag).items()}
        d = c.view_data(view)
        _G.update(view=view, verify=dict(nb_real=nb_real, nb_masked=nb_masked, field=sets["hemibrain_field"], rel_of=d["release_code"],
                                          row_of=m.rows_of(view), draws={v: x["draws"] for v, x in draws.items()}))
        fams = c.development_families(view, 1)
        c.benchmark(view)
        with get_context("fork").Pool(workers) as pool:
            rows = [r for g in pool.map(verify_job, list(map(int, fams)), chunksize=8) for r in g]
        strict = np.array([r["status"]["strict"] for r in rows])
        if not np.array_equal(np.array([h.STATUS[s] for s in strict]), h.truth(view, fams)):
            raise AssertionError("strict statuses differ from the frozen development run")
        mv = dict(result["views"][view]["sessions"]["masked_validation"])
        for key in ("masked:author_nobottom", "field:author_nobottom"):
            cand = np.array([r["status"][key] for r in rows])
            cert = cand != "unresolved"
            wrong = cert & (cand != strict)
            mv[key] = dict(certified=float(cert.mean()), wrong=int(wrong.sum()), wrong_share_of_certified=float(wrong.sum() / max(cert.sum(), 1)),
                           opposite=int((cert & (strict != "unresolved") & (cand != strict)).sum()))
        real_nb = np.array([r["status"]["real:author_nobottom"] for r in rows])
        shift = {}
        for v, x in draws.items():
            w_s, c_s, o_s = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
            for i in range(DRAWS):
                for j, r in enumerate(rows):
                    if not r["touched"][(v, i)]:
                        continue
                    s = r["status"][f"shift:{v}:{i}"]
                    if s != "unresolved":
                        c_s[j] += 1
                        w_s[j] += s != strict[j]
                        o_s[j] += strict[j] != "unresolved" and s != strict[j]
            shift[v] = dict(**_cluster_rate(view, fams, w_s, c_s), opposite=int(o_s.sum()), draws=DRAWS,
                            **{k: x[k] for k in ("uncovered_target_share", "mean_masked_weight_share", "target_weight", "target_weight_share_of_resolved", "expected_masked_over_target")})
        sessions = result["views"][view]["sessions"]
        unc = dict(sessions["uncertifiable_share"], **{"real:author_nobottom": float((real_nb == "unresolved").mean())})
        mk_nb = masked_report_nobottom(view, nb_masked)
        table = {}
        for v in (*VARIANTS, "author_nobottom"):
            keys = [k for k in (f"masked:{v}", f"field:{v}", f"bottom:{v}") if k in mv]
            table[v] = dict(opposite={k: mv[k]["opposite"] for k in keys}, opposite_total=int(sum(mv[k]["opposite"] for k in keys)),
                            wrong_masked=max(mv[f"masked:{v}"]["wrong_share_of_certified"], mv[f"field:{v}"]["wrong_share_of_certified"]),
                            wrong_bottom=mv[f"bottom:{v}"]["wrong_share_of_touched_certified"] if f"bottom:{v}" in mv else None,
                            wrong_shift_corrected=shift[f"distribution:{v}"], wrong_shift_mass_matched=shift[f"mass:{v}"],
                            uncertifiable_drop_pp=100 * (unc["strict"] - unc[f"real:{v}"]),
                            bottom_validation="session-level validation possible" if f"bottom:{v}" in mv else
                            ("no BOTTOM source" if v == "author_nobottom" else "not validated at session level (no independent truth for author-declared BOTTOM)"))
        table["author_nobottom"].update(coverage=min(mk_nb["all"]["coverage_given_available"], mk_nb["hemibrain_field"]["coverage_given_available"]),
                                        median_size=mk_nb["all"]["size"]["median"], masked=mk_nb)
        ver["views"][view] = dict(table=table, uncertifiable_share=unc, masked_validation_author_nobottom={k: mv[k] for k in ("masked:author_nobottom", "field:author_nobottom")},
                                  sources_author_nobottom=_count_sources(view, sets, nb_real, nb_source), bottom_sources=bottom_sources_check(view, sets),
                                  opposite_cases=opposite_cases(view, sets))
        print(view, json.dumps({v: dict(opp=t["opposite_total"], wrong=round(t["wrong_masked"], 4), shift=round(t["wrong_shift_corrected"]["value"], 4),
                                        mass=round(t["wrong_shift_mass_matched"]["value"], 4)) for v, t in table.items()}), flush=True)
    ver["hemibrain_label_none"] = hemibrain_label_none()
    ver["verdicts"] = verification_verdicts(result, ver)
    result["verification"] = ver
    path.write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(ver["verdicts"], indent=1))


def masked_report_nobottom(view, nb_masked):
    x = masked_cells(view, 1)
    ids = c.view_data(view)["ids"]
    s = [nb_masked[int(ids[r])] for r in x["rows"]]
    out = {}
    for stratum, mask in (("all", np.ones(len(s), bool)), ("hemibrain_field", ~x["by_name"])):
        have = np.array([q is not None for q in s]) & mask
        cov = np.array([m.contains(q, l) for q, l in zip(s, x["lab"])]) & mask
        out[stratum] = dict(available_weight=float(x["w"][have].sum() / x["w"][mask].sum()), coverage_given_available=float(x["w"][cov].sum() / max(x["w"][have].sum(), 1)),
                            size=size_stats([q for q, hv in zip(s, have) if hv], x["w"][have]))
    return out


def _count_sources(view, sets, real, source):
    ids = c.view_data(view)["ids"]
    acc = {}
    for r, w, l in zip(sets["rows"], sets["w"], sets["lab"]):
        if l < 0:
            k = source[int(ids[r])]
            acc[k] = acc.get(k, 0) + int(w)
    total = sum(acc.values())
    return {k: v / total for k, v in sorted(acc.items())}


def verification_verdicts(result, ver):
    """Original pre-specified verdict, the same criterion with the shift-corrected wrong share, and the strict 'sound' verdict (no opposite)."""
    out = {}
    for v in (*VARIANTS, "author_nobottom"):
        per = {}
        for view, x in ver["views"].items():
            t = x["table"][v]
            if v == "author_nobottom":
                cov, size = t["coverage"], t["median_size"]
            else:
                ch = result["verdict"][v]["checks"][view]
                cov, size = ch["coverage"], ch["median_size"]
            wrong = max(t["wrong_masked"], t["wrong_bottom"] or 0.0)
            wrong_shift = max(wrong, t["wrong_shift_corrected"]["value"], t["wrong_shift_mass_matched"]["value"])
            drop = t["uncertifiable_drop_pp"]
            per[view] = dict(coverage=cov, median_size=size, wrong=wrong, wrong_shift_corrected=t["wrong_shift_corrected"]["value"],
                             wrong_shift_mass_matched=t["wrong_shift_mass_matched"]["value"], drop_pp=drop,
                             opposite=t["opposite_total"], shift_opposite=t["wrong_shift_corrected"]["opposite"] + t["wrong_shift_mass_matched"]["opposite"],
                             passes=bool(cov >= 0.90 and size <= 3 and wrong <= 0.01 and drop >= 5),
                             passes_shift=bool(cov >= 0.90 and size <= 3 and wrong_shift <= 0.01 and drop >= 5),
                             bottom_validated=t["bottom_validation"] != "not validated at session level (no independent truth for author-declared BOTTOM)")
        original = result["verdict"][v]["verdict"] if v in result["verdict"] else ("DO" if all(p["passes"] for p in per.values()) else "DO NOT")
        out[v] = dict(per_view=per, verdict=original,
                      verdict_shift_corrected="DO" if all(p["passes_shift"] for p in per.values()) else "DO NOT",
                      sound_verdict="PASS" if all(p["opposite"] == 0 for p in per.values()) else "FAIL")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--verify", action="store_true", help="review round: add the 'verification' key to the existing matching.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify(args.workers)
        return
    result = dict(
        protocol=dict(split="development only; fit on half 0, evaluate on half 1", floor_quantile=FLOOR_QUANTILE, coverage_target=COVERAGE_TARGET, ks=KS, margins=MARGINS,
                      variants=VARIANT_TEXT,
                      masking="every resolved presynaptic cell of half-1 families read through sources 2-3 with its own label hidden; really unresolved cells stay unresolved"),
        caveats=[
            "Masked types are easy: most are resolved because the release name equals a hemibrain name, so the other release usually carries the same name and the "
            "transitive source is near-trivial for them; the really unresolved types are mostly release-specific (optic lobe, ascending, sensory) and differ in "
            "distribution (see similarity and stress). The 'hemibrain_field' stratum (release name not in the vocabulary) is the closer analogue and enters the verdict.",
            "Masking never has BOTTOM as the truth: it measures false BOTTOM on hemibrain-typed cells, not whether BOTTOM is right for the really unresolved types. "
            "In variant 'full' most of the gain comes from connectivity -> BOTTOM ('nothing is similar'), which is therefore not validated by this pilot.",
            "Source 1 (author hemibrain field: multi-valued, subtypes of a coarse name, hemibrain bodies, names outside the vocabulary -> BOTTOM) is trusted like "
            "the frozen crosswalk and not validated by masking; names outside the vocabulary are mostly ORN/HRN/TRN sensory types, but a few look like typos (e.g. PLVP059).",
            "Profiles are built from the annotation of the whole release (labels and author fields of every cell, the same source the verifier reads), including cells "
            "that are inputs of held-out targets; no held-out session or family is evaluated or used for fitting. Thresholds are fitted on development half 0 only.",
            "Partner types are shared between halves (the halves split target types), so half 1 is not held out with respect to partner types; only 3 parameters are fitted.",
            "Masked validation masks all resolved presynaptic cells at once but builds each type's profile from the true labels of the other types (leave-one-type-out).",
            "Raw masked coverage of connectivity is dominated by high-similarity types; 'shift_corrected' reweights it by best-similarity bins to the really "
            "unresolved cells and enters the verdict. It still assumes those cells carry a hemibrain type; 'agreement' compares connectivity with the author-level "
            "sets on the really unresolved cells themselves.",
            "Cells whose author field names a type outside the vocabulary (truth BOTTOM, mostly ORN/HRN/TRN sensory types) read through connectivity alone form a "
            "second masked validation with BOTTOM as the truth ('bottom:*', wrong share among certified sessions that contain such a cell); they are one specific "
            "kind of BOTTOM type, not a sample of the really unresolved ones.",
        ],
        views={},
    )
    for view in ("malecns", "flywire"):
        rule = fit(view)
        sets = build_sets(view, rule)
        fams, rows = run_sessions(view, sets, args.workers)
        sessions = session_report(view, fams, rows)
        result["views"][view] = dict(rule={k: v for k, v in rule.items() if k != "grid"}, fit_grid=rule["grid"], masked=masked_report(view, rule, sets),
                                     shift_corrected=shift_corrected(view, rule, sets), agreement=source_agreement(view, rule, sets),
                                     similarity=similarity_shift(view, sets), stress=stress_test(view, rule, sets), sources=sources_report(view, sets), sessions=sessions)
        print(view, json.dumps(dict(rule=result["views"][view]["rule"], uncertifiable=sessions["uncertifiable_share"]), indent=1), flush=True)
    result["verdict"] = verdict(result)
    (OUT / "matching.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(result["verdict"], indent=1))


def verdict(result):
    """DO only if coverage >= 0.90 with median |S| <= 3, wrong certificates <= 1 % and uncertifiable share down >= 5 pp."""
    out = {}
    for v in VARIANTS:
        checks = {}
        for view, r in result["views"].items():
            mk = r["masked"]
            # coverage of what is actually used on the unresolved cells: the variant's masked rule and, if used, connectivity alone
            covs = [mk["variant:" + v]["all"]["coverage_given_available"], mk["variant:" + v]["hemibrain_field"]["coverage_given_available"]]
            source = dict(confident="connectivity_above_floor", full="connectivity").get(v)
            if source:
                covs += [mk[source]["all"]["coverage_given_available"], mk[source]["hemibrain_field"]["coverage_given_available"],
                         r["shift_corrected"]["fitted_rule_coverage"]]
            size = mk["variant:" + v]["all"]["size"]["median"]
            mv = r["sessions"]["masked_validation"]
            wrong = max(mv["masked:" + v]["wrong_share_of_certified"], mv["field:" + v]["wrong_share_of_certified"],
                        mv["bottom:" + v]["wrong_share_of_touched_certified"] if "bottom:" + v in mv else 0.0)
            drop = r["sessions"]["uncertifiable_share"]["strict"] - r["sessions"]["uncertifiable_share"]["real:" + v]
            checks[view] = dict(coverage=min(covs), median_size=size, wrong_certificates=wrong, uncertifiable_drop_pp=100 * drop,
                                passes=bool(min(covs) >= 0.90 and size <= 3 and wrong <= 0.01 and drop >= 0.05))
        out[v] = dict(checks=checks, verdict="DO" if all(x["passes"] for x in checks.values()) else "DO NOT")
    return out


if __name__ == "__main__":
    main()
