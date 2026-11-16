#!/usr/bin/env python3
"""Pilot B: hemibrain connectivity constraints in the type predictor (development split only).

Stage ``fit`` scans B1 / B2 / B1+B2 settings on half 0 of the development types and
picks, per view and variant, the setting with the smallest total recal cost on the
certifiable sessions of half 0 (ties: larger within-question AUROC). Stage ``eval``
replays the picked settings once on half 1 and writes the comparison with the frozen
mlp_s0. Outputs: ``outputs/neurosym/B/fit-{view}.json`` and ``outputs/neurosym/B/eval.json``.
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

OUT = c.ROOT / "outputs/neurosym/B"
WORKERS = 10
VIEWS = ("flywire", "malecns")
B1_GRID = dict(form=("share", "lik"), eps=(1e-4, 1e-3, 1e-2), beta=(0.25, 0.5, 1.0, 2.0))
B2_GRID = dict(mf=dict(eps=(1e-4, 1e-2), beta=(0.03, 0.1, 0.3, 1.0)), bp=dict(eps=(1e-8, 1e-6, 1e-4), beta=(0.1, 0.3, 1.0, 2.0)))
COMBO_SCALES = (0.5, 1.0)
_PRIORS = {}


def arms(fam, inst, outcomes, rho_index):
    """AG and recal over every prior set registered in ``_PRIORS`` (module level for the fork pool)."""
    out = {}
    for name, priors in _PRIORS.items():
        p = priors[fam["family"]]
        out[name + ":ag"] = e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, p))
        out[name + ":recal"] = e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p))
    return out


def stored_priors(view, families):
    return {int(f): c.family(view, f)["priors"]["mlp_s0"] for f in families}


def brier(view, families, priors):
    d = c.benchmark(view)
    s, n = 0.0, 0
    for f in families:
        t = int(d["fam_target"][f])
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
        member = np.isin(d["pre_label"][lo:hi], codes).astype(float)
        s += float(((priors[int(f)][:, 0] - member) ** 2).sum())
        n += hi - lo
    return s / n


def score(view, families, sets, base_auroc=None):
    """Costs (AG, recal) via the exact verifier plus AUROC/Brier for every prior set in ``sets``."""
    _PRIORS.clear()
    _PRIORS.update(sets)
    costs = h.run(view, families, arms, workers=WORKERS)
    weight = h.baseline(view, families, "weight")
    out = {}
    for name, priors in sets.items():
        fams, rest, strict = k.within_auroc(view, families, priors)
        row = dict(
            recal=h.ratio(view, families, costs[name + ":recal"], weight)["value"],
            ag=h.ratio(view, families, costs[name + ":ag"], weight)["value"],
            auroc=float(rest.mean()),
            auroc_member_vs_nonmember=float(np.nanmean(strict)),
            brier=brier(view, families, priors),
        )
        out[name] = (row, costs[name + ":recal"], costs[name + ":ag"], (fams, rest, strict))
    _PRIORS.clear()
    return out


def dev_nodes(view):
    """View node indices of every presynaptic cell of a development target (both halves)."""
    d = c.benchmark(view)
    targets = np.unique(d["fam_target"][c.development_families(view)])
    cells = np.concatenate([d["pre_cell"][d["target_offset"][t] : d["target_offset"][t + 1]] for t in targets])
    return np.unique(k.rows_of(view, cells))


def field_of(view, form, eps, beta=None, sweeps=1, cache={}):
    key = (view, form, eps, beta if sweeps > 1 else None, sweeps)
    if key not in cache:
        nodes = dev_nodes(view)
        index = np.full(len(c.view_data(view)["ids"]), -1, dtype=np.int64)
        index[nodes] = np.arange(len(nodes))
        t = time.time()
        cache[key] = (k.b2_field(view, form, eps, beta=beta, sweeps=sweeps, rows=nodes), index)
        print(f"  field {key} {time.time() - t:.1f}s", flush=True)
    return cache[key]


def name_of(cfg):
    return "|".join(f"{a}={b}" for a, b in sorted(cfg.items()) if not isinstance(b, dict)) + (
        "" if "b1" not in cfg else f"|b1={name_of(cfg['b1'])}") + ("" if "b2" not in cfg else f"|b2={name_of(cfg['b2'])}")


def priors_for(view, families, variant, cfg):
    if variant == "B1":
        return k.family_priors(view, families, b1=cfg)
    if variant == "B2":
        field, index = field_of(view, cfg["form"], cfg["eps"], cfg["beta"], cfg["sweeps"])
        return k.family_priors(view, families, b2=dict(beta=cfg["beta"]), field=field, field_rows=index)
    if variant == "B1+B2":
        b1, b2 = cfg["b1"], cfg["b2"]
        field, index = field_of(view, b2["form"], b2["eps"], b2["beta"], b2["sweeps"])
        return k.family_priors(view, families, b1=b1, b2=dict(beta=b2["beta"]), field=field, field_rows=index)
    raise ValueError(variant)


def pick(rows):
    return min(rows, key=lambda r: (round(r["recal"], 6), -r["auroc"]))


def extend(view, families, variant, rows, top, log):
    """One extension step: if the picked beta is the largest of its grid, try 2x and 4x for that setting."""
    best = pick(rows)["config"]
    if best["beta"] != top(best):
        return rows
    return rows + scan(view, families, variant, [dict(best, beta=best["beta"] * m) for m in (2, 4)], log)


def scan(view, families, variant, configs, log):
    rows = []
    for i in range(0, len(configs), 4):  # four prior sets per verifier pass keeps memory modest
        batch = configs[i : i + 4]
        sets = {name_of(cfg): priors_for(view, families, variant, cfg) for cfg in batch}
        for cfg, (name, (row, *_)) in zip(batch, score(view, families, sets).items()):
            rows.append(dict(variant=variant, config=cfg, **row))
            print(f"  {variant} {name}: recal {row['recal']:.4f} ag {row['ag']:.4f} auroc {row['auroc']:.4f}", flush=True)
        log()
    return rows


def fit(view):
    families = c.development_families(view, 0)
    out = dict(view=view, half=0, families=int(len(families)), rule="min recal cost ratio to weight on certifiable half-0 sessions; tie: AUROC")
    base, *_ = score(view, families, {"mlp_s0": stored_priors(view, families)})["mlp_s0"]
    frozen = h.ratio(view, families, h.baseline(view, families, "mlp_s0:recal"), h.baseline(view, families, "weight"))["value"]
    assert abs(base["recal"] - frozen) < 1e-12, "stored mlp_s0 priors must reproduce the frozen recal run"
    out["mlp_s0"] = base
    path = OUT / f"fit-{view}.json"
    log = lambda: path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(view, "mlp_s0", base, flush=True)

    b1 = [dict(form=f, eps=x, beta=b) for f, x, b in product(*B1_GRID.values())]
    out["B1"] = scan(view, families, "B1", b1, log)
    out["B1"] = extend(view, families, "B1", out["B1"], lambda cfg: max(B1_GRID["beta"]), log)
    best1 = pick(out["B1"])["config"]

    b2 = [dict(form=f, eps=x, beta=b, sweeps=1) for f, g in B2_GRID.items() for x, b in product(g["eps"], g["beta"])]
    out["B2"] = scan(view, families, "B2", b2, log)
    out["B2"] = extend(view, families, "B2", out["B2"], lambda cfg: max(B2_GRID[cfg["form"]]["beta"]), log)
    best2 = pick(out["B2"])["config"]
    two = [dict(best2, beta=b, sweeps=2) for b in sorted({best2["beta"] / 2, best2["beta"]})]
    out["B2"] += scan(view, families, "B2", two, log)
    best2 = pick(out["B2"])["config"]
    # The criterion is an OR (AUROC or cost): B2 also gets its best-AUROC setting of half 0.
    out["B2_auroc_pick"] = max(out["B2"], key=lambda r: (round(r["auroc"], 6), -r["recal"]))["config"]

    combos = [dict(b1=dict(best1, beta=best1["beta"] * s1), b2=dict(best2, beta=best2["beta"] * s2)) for s1, s2 in product(COMBO_SCALES, COMBO_SCALES)]
    out["B1+B2"] = scan(view, families, "B1+B2", combos, log)

    # Where does B1's information come from? (half 0 only) Flatten its factor inside A or outside A.
    out["B1_diagnostic"] = b1_diagnostic(view, families, best1)
    out["picked"] = {v: pick(out[v])["config"] for v in ("B1", "B2", "B1+B2")}
    if out["B2_auroc_pick"] != out["picked"]["B2"]:
        out["picked"]["B2:auroc"] = out["B2_auroc_pick"]
    log()
    return out


def b1_diagnostic(view, families, cfg):
    """B1 with its log-factor replaced by its mean inside A ("flat_in_A") or outside A ("flat_out_A")."""
    d = c.benchmark(view)
    sets = {"flat_in_A": {}, "flat_out_A": {}, "A_only_constant": {}}
    q_full = k.full_mlp(view)
    for f in families:
        t = int(d["fam_target"][f])
        lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
        nodes = k.rows_of(view, d["pre_cell"][lo:hi])
        codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
        phi = k.b1_logfactor(int(d["target_type"][t]), cfg["beta"], cfg["eps"], cfg["form"])
        inside = np.zeros(len(phi), dtype=bool)
        inside[codes] = True
        outside = ~inside
        outside[0] = False
        q0 = q_full[nodes]
        for name, mask in (("flat_in_A", inside), ("flat_out_A", outside)):
            p = phi.copy()
            p[mask] = phi[mask].mean()
            sets[name][int(f)] = k.prior_columns(*k.project(k.posterior(q0, p), codes))
        p = np.where(inside, phi[inside].mean(), phi[outside].mean())  # only the A / not-A split
        p[0] = 0.0
        sets["A_only_constant"][int(f)] = k.prior_columns(*k.project(k.posterior(q0, p), codes))
    res = score(view, families, sets)
    return {n: row for n, (row, *_) in res.items()}


def picks(view):
    """Recal picks of the fit plus, for B1 and B2, the best-AUROC setting of half 0 (the criterion is an OR)."""
    fitted = json.loads((OUT / f"fit-{view}.json").read_text())
    out = {v: cfg for v, cfg in fitted["picked"].items() if ":" not in v}
    for variant in ("B1", "B2"):
        best = max(fitted[variant], key=lambda r: (round(r["auroc"], 6), -r["recal"]))["config"]
        if best != out[variant]:
            out[variant + ":auroc"] = best
    return out


def evaluate(views):
    picked = {v: picks(v) for v in views}
    out = dict(half=1, looks_on_half1_per_view={v: len(p) for v, p in picked.items()}, picked=picked, views={})
    for view in views:
        families = c.development_families(view, 1)
        weight = h.baseline(view, families, "weight")
        old = dict(recal=h.baseline(view, families, "mlp_s0:recal"), ag=h.baseline(view, families, "mlp_s0:ag"))
        sets = {"mlp_s0": stored_priors(view, families)}
        for variant, cfg in picked[view].items():
            sets[variant] = priors_for(view, families, variant.split(":")[0], cfg)
        res = score(view, families, sets)
        base_row, base_recal, base_ag, (fams0, rest0, strict0) = res["mlp_s0"]
        assert np.array_equal(base_recal, old["recal"]) and np.array_equal(base_ag, old["ag"]), "mlp_s0 must match the frozen run"
        block = dict(families=int(len(families)), sessions=int(len(weight)), certifiable=int((h.truth(view, families) > 0).sum()),
                     mlp_s0=dict(recal_to_weight=h.ratio(view, families, old["recal"], weight), ag_to_weight=h.ratio(view, families, old["ag"], weight),
                                 auroc=k.mean_auroc_ci(view, fams0, rest0), auroc_member_vs_nonmember=k.mean_auroc_ci(view, fams0, strict0),
                                 brier=base_row["brier"]))
        for variant in picked[view]:
            row, recal, ag, (fams, rest, strict) = res[variant]
            assert np.array_equal(fams, fams0)
            block[variant] = dict(
                config=picked[view][variant],
                recal_to_weight=h.ratio(view, families, recal, weight),
                ag_to_weight=h.ratio(view, families, ag, weight),
                recal_new_over_mlp_s0=h.ratio(view, families, recal, old["recal"]),
                ag_new_over_mlp_s0=h.ratio(view, families, ag, old["ag"]),
                auroc=k.mean_auroc_ci(view, fams, rest),
                auroc_minus_mlp_s0=k.paired_auroc_diff(view, fams, rest, rest0),
                auroc_member_vs_nonmember=k.mean_auroc_ci(view, fams, strict),
                auroc_member_vs_nonmember_minus_mlp_s0=k.paired_auroc_diff(view, fams, strict, strict0),
                brier=row["brier"],
            )
        out["views"][view] = block
        print(json.dumps({view: block}, indent=1, default=float), flush=True)
    out["verdict"] = verdict(out["views"])
    (OUT / "eval.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def verdict(views):
    """DO if on some view a variant gains AUROC >= 0.03 or cuts recal >= 3 % with CI upper < 1, without harming the other view."""
    rows = {}
    names = sorted({v for b in views.values() for v in b if v.startswith("B")})
    for variant in names:
        good, harm = {}, {}
        for view, b in views.items():
            if variant not in b:  # e.g. B2:auroc picked the same setting as B2 on this view
                b = dict(b, **{variant: b[variant.split(":")[0]]})
            r = b[variant]
            auc_gain = r["auroc_minus_mlp_s0"]["value"] >= 0.03
            cost_gain = r["recal_new_over_mlp_s0"]["value"] <= 0.97 and r["recal_new_over_mlp_s0"]["high"] < 1.0
            good[view] = bool(auc_gain or cost_gain)
            harm[view] = bool(r["recal_new_over_mlp_s0"]["low"] > 1.0 or r["auroc_minus_mlp_s0"]["high"] < 0)
        ok = any(good[v] and not any(harm[w] for w in views if w != v) for v in views)
        rows[variant] = dict(meets=good, harms=harm, do=bool(ok))
    rows["overall"] = "DO" if any(r["do"] for r in rows.values()) else "DO NOT"
    return rows


def verdict_strict(views):
    """Stricter reading added after review: a gain must be significant and no view may be harmed.

    gain on a view: AUROC +0.03 or more with CI low > 0, or recal <= 0.97 with CI high < 1;
    harm on any view (the credited one included): AUROC CI high < 0, or recal or AG CI low > 1.
    """
    rows = {}
    names = sorted({v for b in views.values() for v in b if v.startswith("B")})
    for variant in names:
        good, harm = {}, {}
        for view, b in views.items():
            r = b[variant] if variant in b else b[variant.split(":")[0]]
            auc = r["auroc_minus_mlp_s0"]
            recal, ag = r["recal_new_over_mlp_s0"], r["ag_new_over_mlp_s0"]
            good[view] = bool((auc["value"] >= 0.03 and auc["low"] > 0) or (recal["value"] <= 0.97 and recal["high"] < 1.0))
            harm[view] = bool(auc["high"] < 0 or recal["low"] > 1.0 or ag["low"] > 1.0)
        rows[variant] = dict(meets=good, harms=harm, do=bool(any(good.values()) and not any(harm.values())))
    rows["overall"] = "DO" if any(r["do"] for k, r in rows.items() if k != "overall") else "DO NOT"
    return rows


def reverdict():
    path = OUT / "eval.json"
    out = json.loads(path.read_text())
    out["verdict_strict"] = verdict_strict(out["views"])
    out.setdefault("notes", [])
    note = "verdict_strict (added after independent review): same-view harm disqualifies, AG harm counts, the AUROC branch needs CI low > 0; the L/R homologue constraint (B3) was not implemented"
    if note not in out["notes"]:
        out["notes"].append(note)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out["verdict_strict"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "eval", "reverdict"))
    p.add_argument("--views", nargs="+", default=list(VIEWS))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "fit":
        for view in a.views:
            fit(view)
    elif a.stage == "reverdict":
        print(json.dumps(reverdict(), indent=1, ensure_ascii=False))
    else:
        evaluate(a.views)


if __name__ == "__main__":
    main()
