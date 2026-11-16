#!/usr/bin/env python3
"""Pilot B round 2: similarity-kernel coupling (B4) and offline pooling (B3') with two-way cross-fitting.

``fit --view V --half H`` scans the grids of the four variants on half H of the
development types and picks, per variant, the (setting, base policy) with the
smallest cost ratio to the better of mlp_s0:recal and mlp_s0:ag on that half.
``eval`` replays every pick on the other half and writes ``outputs/neurosym/B/round2.json``
with the pre-declared verdict. Variants:

* ``B4``         mlp_s0 prior, kernel-coupled AG / recal;
* ``B3p``        B3' pooled prior, frozen AG / recal;
* ``B3p+B2``     B3' pooled prior plus the round-1 B2 field (bp, eps=1e-8, one sweep);
* ``B3p(+B2)+B4`` the better of the two pooled priors with kernel-coupled AG / recal.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as c
from connectomequest.neurosym import constraints as k
from connectomequest.neurosym import harness as h
from connectomequest.neurosym import kernel as kk

OUT = c.ROOT / "outputs/neurosym/B/round2"
WORKERS = int(__import__("os").environ.get("NEUROSYM_WORKERS", "10"))
VARIANTS = ("B4", "B3p", "B3p+B2", "B3p(+B2)+B4")
B4_GRID = dict(lam=(0.5, 2.0, 8.0), gamma=(1.0, 3.0), tau=(0.3, 0.5, 0.7))
B3_GRID = dict(lam=(0.1, 0.3, 1.0), gamma=(1.0, 3.0), tau=(0.3, 0.5))
B2_FIELD = dict(form="bp", eps=1e-8, sweeps=1)
B2_BETAS = (0.3, 1.0)
B4_SCALES = (0.5, 1.0, 2.0)
_RUN = dict(view=None, configs={}, priors={})


def arms(fam, inst, outcomes, rho_index):
    """AG and recal of every registered config (module level, inherited by the fork pool)."""
    view, out = _RUN["view"], {}
    for name, cfg in _RUN["configs"].items():
        prior = fam["priors"]["mlp_s0"] if cfg["prior"] is None else _RUN["priors"][cfg["prior"]][fam["family"]]
        b4 = cfg.get("b4")
        if b4 is None or b4["lam"] == 0:
            out[name + ":ag"] = e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, prior))
            out[name + ":recal"] = e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, prior))
        else:
            kern = kk.question_kernel(view, fam, b4["gamma"], b4["tau"])
            for base in ("ag", "recal"):
                out[f"{name}:{base}"] = e.run_adaptive(inst, outcomes, kk.policy(base, inst, prior, kern, b4["lam"]))
    return out


def dev_nodes(view):
    d = c.benchmark(view)
    targets = np.unique(d["fam_target"][c.development_families(view)])
    cells = np.concatenate([d["pre_cell"][d["target_offset"][t] : d["target_offset"][t + 1]] for t in targets])
    return np.unique(k.rows_of(view, cells))


_FIELD = {}


def b2_field(view):
    if view not in _FIELD:
        nodes = dev_nodes(view)
        index = np.full(len(c.view_data(view)["ids"]), -1, dtype=np.int64)
        index[nodes] = np.arange(len(nodes))
        _FIELD[view] = (k.b2_field(view, B2_FIELD["form"], B2_FIELD["eps"], sweeps=B2_FIELD["sweeps"], rows=nodes), index)
    return _FIELD[view]


def build_prior(view, families, spec):
    """spec: None (mlp_s0) or dict(gamma, tau, lam, beta_b2)."""
    if spec is None:
        return None
    b2 = None
    if spec.get("beta_b2", 0):
        field, rows = b2_field(view)
        b2 = dict(beta=spec["beta_b2"], field=field, rows=rows)
    return kk.family_priors(view, families, spec["gamma"], spec["tau"], spec["lam"], b2=b2)


def key_of(cfg):
    return json.dumps(cfg, sort_keys=True)


def costs_of(view, families, configs):
    """{key: {ag, recal}} costs on the sessions of ``families`` for every config (dict(prior=spec|None, b4=..|None))."""
    _RUN.update(view=view, configs={}, priors={})
    for cfg in configs:
        name = key_of(cfg)
        pname = None
        if cfg["prior"] is not None:
            pname = key_of(cfg["prior"])
            if pname not in _RUN["priors"]:
                _RUN["priors"][pname] = build_prior(view, families, cfg["prior"])
        _RUN["configs"][name] = dict(prior=pname, b4=cfg.get("b4"))
    kk.profiles(view), k.rows_of(view, c.benchmark(view)["pre_cell"][:1])  # build once before the fork pool
    out = h.run(view, families, arms, workers=WORKERS)
    priors = dict(_RUN["priors"])
    _RUN.update(configs={}, priors={})
    return {key_of(cfg): {b: out[f"{key_of(cfg)}:{b}"] for b in ("ag", "recal")} for cfg in configs}, priors


def best_baseline(view, families):
    """The better (smaller total on certifiable sessions) of mlp_s0:recal and mlp_s0:ag on these families."""
    mask = h.truth(view, families) > 0
    base = {b: h.baseline(view, families, f"mlp_s0:{b}") for b in ("ag", "recal")}
    name = min(base, key=lambda b: base[b][mask].sum())
    return name, base


def scan(view, families, variant, configs, rows, batch=6):
    best_name, base = best_baseline(view, families)
    for i in range(0, len(configs), batch):
        part = configs[i : i + batch]
        t = time.time()
        res, _ = costs_of(view, families, part)
        for cfg in part:
            r = res[key_of(cfg)]
            row = dict(variant=variant, config=cfg,
                       ag=h.ratio(view, families, r["ag"], base[best_name])["value"],
                       recal=h.ratio(view, families, r["recal"], base[best_name])["value"])
            rows.append(row)
            print(f"  {variant} {key_of(cfg)}: ag {row['ag']:.4f} recal {row['recal']:.4f}", flush=True)
        print(f"  batch {time.time() - t:.0f}s", flush=True)
    return rows


def pick(rows, variant):
    """(config, base) with the smallest ratio to the better baseline on the fitting half."""
    cand = [(r[b], b, r["config"]) for r in rows if r["variant"] == variant for b in ("ag", "recal")]
    value, base, cfg = min(cand, key=lambda x: (round(x[0], 6), x[1]))
    return dict(config=cfg, base=base, fit_ratio=value)


def fit(view, half):
    families = c.development_families(view, half)
    best_name, _ = best_baseline(view, families)
    out = dict(view=view, half=half, families=int(len(families)), best_baseline=best_name, rows=[], picks={},
               rule="min cost ratio to the better of mlp_s0:recal/ag on certifiable sessions of the fitting half")
    path = OUT / f"fit-{view}-h{half}.json"
    if path.exists() and set(json.loads(path.read_text()).get("picks", {})) >= set(VARIANTS):
        print(view, half, "already fitted", flush=True)  # resumable: a finished fit is never redone
        return json.loads(path.read_text())

    def save():
        tmp = path.with_name(f"{path.name}.{__import__('os').getpid()}.tmp")
        tmp.write_text(json.dumps(out, indent=1, default=float) + "\n")
        tmp.replace(path)
    rows = out["rows"]

    b4 = [dict(prior=None, b4=dict(lam=l, gamma=g, tau=t)) for l, g, t in product(*B4_GRID.values())]
    scan(view, families, "B4", b4, rows)
    out["picks"]["B4"] = pick(rows, "B4")
    save()

    b3 = [dict(prior=dict(lam=l, gamma=g, tau=t), b4=None) for l, g, t in product(*B3_GRID.values())]
    scan(view, families, "B3p", b3, rows)
    out["picks"]["B3p"] = pick(rows, "B3p")
    save()

    p3 = out["picks"]["B3p"]["config"]["prior"]
    scan(view, families, "B3p+B2", [dict(prior=dict(p3, beta_b2=b), b4=None) for b in B2_BETAS], rows)
    out["picks"]["B3p+B2"] = pick(rows, "B3p+B2")
    save()

    priors = [out["picks"][v] for v in ("B3p", "B3p+B2")]
    prior = min(priors, key=lambda p: p["fit_ratio"])["config"]["prior"]
    k4 = out["picks"]["B4"]["config"]["b4"]
    combo = [dict(prior=prior, b4=dict(k4, lam=k4["lam"] * s)) for s in B4_SCALES]
    scan(view, families, "B3p(+B2)+B4", combo, rows)
    out["picks"]["B3p(+B2)+B4"] = pick(rows, "B3p(+B2)+B4")
    save()
    return out


def clairvoyant(view, families):
    """Clairvoyant cost (diagnostic, not a policy) per session, aligned with harness.rows."""
    out = []
    for f in families:
        fam = c.family(view, f)
        for _, threshold in c.thresholds(fam):
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            b = e.clairvoyant_cost(inst, fam["outcomes"])
            out.append(inst.size if b is None else b)
    return np.asarray(out, dtype=np.int64)


def evaluate(views):
    out = dict(protocol=dict(
        cross_fit="fit on half H, evaluate on half 1-H, both directions; a variant counts only if it passes in both",
        primary="cost ratio to the better of mlp_s0:recal and mlp_s0:ag on the evaluation half (paired, type-cluster bootstrap)",
        do="ratio <= 0.95 with CI upper < 1 on both views and both directions; no significant harm anywhere",
        partial="the same on one view only", harm="CI of new/old AG or recal entirely > 1, or CI of delta AUROC entirely < 0"),
        results={}, clairvoyant={})
    for view in views:
        for half in (0, 1):
            fitted = json.loads((OUT / f"fit-{view}-h{half}.json").read_text())
            families = c.development_families(view, 1 - half)
            best_name, base = best_baseline(view, families)
            weight = h.baseline(view, families, "weight")
            configs = [p["config"] for p in fitted["picks"].values()]
            res, priors = costs_of(view, families, configs)
            stored = {int(f): c.family(view, f)["priors"]["mlp_s0"] for f in families}
            fams0, rest0, _ = k.within_auroc(view, families, stored)
            block = dict(fit_half=half, eval_half=1 - half, families=int(len(families)), best_baseline=best_name,
                         baseline_to_weight={b: h.ratio(view, families, base[b], weight) for b in base})
            for variant, p in fitted["picks"].items():
                r = res[key_of(p["config"])]
                row = dict(config=p["config"], base=p["base"], fit_ratio=p["fit_ratio"],
                           primary=h.ratio(view, families, r[p["base"]], base[best_name]),
                           to_weight=h.ratio(view, families, r[p["base"]], weight),
                           ag_new_over_old=h.ratio(view, families, r["ag"], base["ag"]),
                           recal_new_over_old=h.ratio(view, families, r["recal"], base["recal"]))
                if p["config"]["prior"] is not None:
                    pr = priors[key_of(p["config"]["prior"])]
                    fams, rest, _ = k.within_auroc(view, families, pr)
                    assert np.array_equal(fams, fams0)
                    row["auroc"] = k.mean_auroc_ci(view, fams, rest)
                    row["auroc_minus_mlp_s0"] = k.paired_auroc_diff(view, fams, rest, rest0)
                else:
                    row["auroc_minus_mlp_s0"] = dict(value=0.0, low=0.0, high=0.0, note="prior unchanged (mlp_s0)")
                row["harm"] = bool(row["ag_new_over_old"]["low"] > 1 or row["recal_new_over_old"]["low"] > 1 or row["auroc_minus_mlp_s0"]["high"] < 0)
                row["pass"] = bool(row["primary"]["value"] <= 0.95 and row["primary"]["high"] < 1 and not row["harm"])
                block[variant] = row
                print(view, f"h{half}->h{1 - half}", variant, {x: round(row["primary"][x], 4) for x in ("value", "low", "high")}, "harm", row["harm"], flush=True)
            out["results"][f"{view}:fit{half}->eval{1 - half}"] = block
        fams = c.development_families(view)
        cv = clairvoyant(view, fams)
        best_name, base = best_baseline(view, fams)
        out["clairvoyant"][view] = dict(to_weight=h.ratio(view, fams, cv, h.baseline(view, fams, "weight")),
                                        to_best_baseline=h.ratio(view, fams, cv, base[best_name]), sessions="all development, certifiable")
    out["verdict"] = verdict(out["results"], views)
    out["looks_on_evaluation_half"] = "each direction: 4 picks (one per variant) evaluated once on the other half"
    (OUT.parent / "round2.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def verdict(results, views):
    rows = {}
    variants = sorted({v for b in results.values() for v in b if v.startswith("B")})
    for variant in variants:
        passes = {view: all(results[f"{view}:fit{hh}->eval{1 - hh}"][variant]["pass"] for hh in (0, 1)) for view in views}
        harms = {view: any(results[f"{view}:fit{hh}->eval{1 - hh}"][variant]["harm"] for hh in (0, 1)) for view in views}
        if any(harms.values()):
            v = "DO NOT"
        elif all(passes.values()):
            v = "DO"
        elif any(passes.values()):
            v = "PARTIAL"
        else:
            v = "DO NOT"
        rows[variant] = dict(passes_both_directions=passes, harm_any_direction=harms, verdict=v)
    order = ["DO", "PARTIAL", "DO NOT"]
    rows["overall"] = min((r["verdict"] for r in rows.values()), key=order.index)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "eval"))
    p.add_argument("--view", nargs="+", default=["flywire", "malecns"])
    p.add_argument("--half", nargs="+", type=int, default=[0, 1])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "fit":
        for view in a.view:
            for half in a.half:
                fit(view, half)
    else:
        evaluate(a.view)


if __name__ == "__main__":
    main()
