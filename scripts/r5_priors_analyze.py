#!/usr/bin/env python3
"""Analysis of the round-5 workstream ``priors`` for one split.

* Seeded priors: quality (from the build records) and the cost of member-first, direction-fixed,
  AG and recalibrated AG relative to weight order on certifiable questions, with the frozen topology
  and earlier-release arms on the same questions for reference; contrasts with intervals.
* Correlated errors: cost relative to weight order against the AUROC level for errors that are
  independent (frozen sweep), correlated with weight (``wplus``, ``wminus``) or shared within release
  type (``rtype``), calibrated (``base``) or confident (``oracle``); the AUROC at which a rule starts
  to pay (its cost ratio stays at or below 1, 0.95 or 0.90 from there on), with intervals and the
  shift against independent errors.
* The frozen development sweep stratified by connectome and number of input cells (no new runs).

Intervals: 2,000 resamples of target cell types (seed 20261112), as in the paper. The script only
reads results; it refuses to overwrite its output, and the held-out split (whose prior quality reads
the answers of held-out questions) needs ``outputs/round5/record.json`` like the runners.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from connectomequest.neurosym import common as nc
from connectomequest.round5 import priors_common as pc
from connectomequest.round5 import priors_layout as pl
from connectomequest.round5 import priors_model as pm

LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)
THRESHOLDS = (1.0, 0.95, 0.9)
SIZE_BINS = ((10, 50, "10-49"), (50, 150, "50-149"), (150, 500, "150-499"), (500, np.inf, ">=500"))
SWEEP_VARIANTS = ("none", "base", "oracle")
SWEEP_POLICIES = ("member", "direction", "rr", "ag", "recal")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def frozen_runs(view, split):
    meta = json.loads((pc.ROOT / f"outputs/v35/runs/{view}-{split}.json").read_text())
    d = load(pc.ROOT / f"outputs/v35/runs/{view}-{split}.npz")
    return {(int(f), int(r)): i for i, (f, r) in enumerate(zip(d["family"], d["rho"]))}, {n: i for i, n in enumerate(meta["arms"])}, d


def session_clusters(view, family):
    return pc.Clusters(pc.family_types(view, family))


# --------------------------------------------------------------------------- crossing


def crossing(levels, ratios, threshold, anchor):
    """Smallest AUROC from which the piecewise-linear cost curve stays at or below ``threshold``.

    ``anchor``: prepend (0.5, 1.0) (calibrated priors without separation are weight order exactly).
    Returns (value, censoring): censoring -1 if already at or below at the first level, +1 if never.
    """
    x, y = list(levels), list(ratios)
    if anchor:
        x, y = [0.5] + x, [1.0] + y
    above = [i for i, v in enumerate(y) if v > threshold + 1e-12]
    if not above:
        return float(x[0]), -1
    i = above[-1]
    if i == len(y) - 1:
        return float("nan"), 1
    return float(x[i] + (y[i] - threshold) * (x[i + 1] - x[i]) / (y[i] - y[i + 1])), 0


def curve_summary(clusters, costs, weight, anchor):
    """costs: (sessions, levels). Ratio per level with CI and crossings with bootstrap CI."""
    sy = clusters.sums(weight)
    sx = np.stack([clusters.sums(costs[:, j]) for j in range(costs.shape[1])])  # levels x clusters
    point = sx.sum(1) / sy.sum()
    boot = (clusters.counts @ sx.T) / (clusters.counts @ sy)[:, None]  # B x levels
    out = dict(ratio={f"{lvl:g}": dict(value=float(point[j]), low=float(np.quantile(boot[:, j], 0.025)), high=float(np.quantile(boot[:, j], 0.975)))
                      for j, lvl in enumerate(LEVELS)}, crossing={})
    for t in THRESHOLDS:
        value, censored = crossing(LEVELS, point, t, anchor)
        reps = np.array([crossing(LEVELS, b, t, anchor) for b in boot])
        vals = reps[:, 0]
        out["crossing"][f"{t:g}"] = dict(value=value, censored=censored, low=float(np.nanquantile(vals, 0.025)) if np.isfinite(vals).any() else None,
                                         high=float(np.nanquantile(vals, 0.975)) if np.isfinite(vals).any() else None,
                                         share_never=float((reps[:, 1] == 1).mean()), share_below_first=float((reps[:, 1] == -1).mean()))
    return out, boot


def crossing_shift(boot_a, boot_b, anchor, threshold):
    """Bootstrap distribution of crossing(a) - crossing(b), replicate by replicate."""
    a = np.array([crossing(LEVELS, r, threshold, anchor)[0] for r in boot_a])
    b = np.array([crossing(LEVELS, r, threshold, anchor)[0] for r in boot_b])
    d = a - b
    ok = np.isfinite(d)
    if not ok.any():
        return None
    return dict(low=float(np.quantile(d[ok], 0.025)), high=float(np.quantile(d[ok], 0.975)), median=float(np.median(d[ok])), share_defined=float(ok.mean()))


# --------------------------------------------------------------------------- seeded priors


def seeded(view, split, base=pc.OUT):
    meta = json.loads((base / f"runs/{view}-{split}.json").read_text())
    d = load(base / f"runs/{view}-{split}.npz")
    arms = {n: i for i, n in enumerate(meta["arms"])}
    key, farms, fd = frozen_runs(view, split)
    rows = np.array([key[int(f), int(r)] for f, r in zip(d["family"], d["rho"])])
    if not np.array_equal(d["cost"][:, arms["weight"]], fd["cost"][rows, farms["weight"]]):
        raise SystemExit("weight order does not reproduce the frozen runs")
    if len(rows) != int((fd["truth"] > 0).sum()):
        raise SystemExit("seeded runs do not cover the certifiable questions")
    c = session_clusters(view, d["family"])
    weight = d["cost"][:, arms["weight"]].astype(float)
    cost = {n: d["cost"][:, i].astype(float) for n, i in arms.items()}
    references = [p for p in ("mlp_s0", "provisional") if f"{p}:ag" in farms]
    for p in references:
        for q in meta["policies"]:
            cost[f"{p}:{q}"] = fd["cost"][rows, farms[f"{p}:{q}"]].astype(float)
    cost["clairvoyant"] = d["clairvoyant"].astype(float)
    out = dict(sessions=len(rows), types=c.k, ratio_to_weight={n: c.ratio(v, weight) for n, v in cost.items() if n != "weight"}, contrasts={},
               quality={ref: prior_quality(view, split, None, ref) for ref in references}, mechanism={})
    for p in meta["priors"]:
        for a, b in (("ag", "direction"), ("recal", "ag"), ("recal", "direction"), ("direction", "member")):
            out["contrasts"][f"{p}:{a}/{p}:{b}"] = c.ratio(cost[f"{p}:{a}"], cost[f"{p}:{b}"])
        if not p.endswith("conn") and f"{p}conn:ag" in cost:
            for q in meta["policies"]:
                out["contrasts"][f"{p}:{q}/{p}conn:{q}"] = c.ratio(cost[f"{p}:{q}"], cost[f"{p}conn:{q}"])
        for ref in references:
            for q in ("ag", "recal", "direction"):
                out["contrasts"][f"{p}:{q}/{ref}:{q}"] = c.ratio(cost[f"{p}:{q}"], cost[f"{ref}:{q}"])
        out["contrasts"][f"{p}:recal/clairvoyant"] = c.ratio(cost[f"{p}:recal"], cost["clairvoyant"])
        out["quality"][p] = prior_quality(view, split, base / f"{p}-{view}.npz", p)
        positive = d["truth"] == 1
        for q in meta["policies"]:
            k = arms[f"{p}:{q}"]
            paid = d["cost"][positive, k].sum()
            out["mechanism"][f"{p}:{q}"] = dict(member=float(d["paid_member"][positive, k].sum() / paid), unresolved=float(d["paid_unresolved"][positive, k].sum() / paid))
        per = d["cost"][:, arms[f"{p}:recal"]] / weight
        out.setdefault("per_question", {})[f"{p}:recal"] = dict(fewer=float((per < 1).mean()), tied=float((per == 1).mean()), more=float((per > 1).mean()), at_least_double=float((per >= 2).mean()))
    return out


def prior_quality(view, split, path, name):
    """Within-question AUROC, Brier, ECE (all input cells; non-seed cells; by whether the target is a seed) and top-1."""
    bench = nc.benchmark(view)
    families = nc.split_families(view, split)
    if path is None:
        plus, unknown = bench["plus_" + name], bench["unknown_" + name]
        return dict(questions_all_cells=pl.quality(view, plus, unknown, families))
    with np.load(path, allow_pickle=False) as z:
        plus, unknown, seed = z["plus_" + name], z["unknown_" + name], z["seed_" + name]
        top = z["top1_" + name] if "top1_" + name in z.files else None
    entries = np.zeros(len(seed), dtype=bool)
    for t in np.unique(bench["fam_target"][families]):
        entries[bench["target_offset"][t] : bench["target_offset"][t + 1]] = True
    _, first = np.unique(bench["pre_cell"][entries & ~seed], return_index=True)
    idx = np.flatnonzero(entries & ~seed)[first]  # one entry per distinct non-seed input cell
    truth = np.maximum(bench["pre_label"][idx], 0)
    resolved = truth > 0
    out = dict(
        questions_all_cells=pl.quality(view, plus, unknown, families),
        questions_nonseed_cells=pl.quality(view, plus, unknown, families, keep=~seed),
        seed_share_of_input_entries=float(seed[entries].mean()),
    )
    fraction = int(re.fullmatch(r"seed(?:ed|mlp)(\d\d)(?:conn)?", name).group(1)) / 100.0
    s = pm.seeds(view, fraction)
    target_rows = pl.node_rows(view, bench["target_cell"])
    target_seed = s.is_seed[target_rows][bench["fam_target"][families]]
    for key, mask in (("target_seed", target_seed), ("target_nonseed", ~target_seed)):
        out[f"questions_nonseed_cells_{key}"] = dict(families=int(mask.sum()), **pl.quality(view, plus, unknown, families[mask], keep=~seed))
    if top is not None:
        out["input_nonseed"] = dict(cells=int(len(idx)), top1=float((top[idx] == truth).mean()), top1_resolved=float((top[idx][resolved] == truth[resolved]).mean()),
                                    top1_outside=float((top[idx][~resolved] == 0).mean()))
    return out


# --------------------------------------------------------------------------- correlated errors


def correlated(view, split):
    meta = json.loads((pc.OUT / f"corrsweep/{view}-{split}.json").read_text())
    d = load(pc.OUT / f"corrsweep/{view}-{split}.npz")
    sweep_meta = json.loads((pc.ROOT / f"outputs/v35/sweep/{view}-{split}.json").read_text())
    sw = load(pc.ROOT / f"outputs/v35/sweep/{view}-{split}.npz")
    if not (np.array_equal(d["family"], sw["family"]) and np.array_equal(d["rho"], sw["rho"]) and np.array_equal(d["weight"], sw["weight"])):
        raise SystemExit("correlated sweep is not aligned with the frozen sweep")
    if sweep_meta["result_sha256"] != meta["frozen_sweep_sha256"]:
        raise SystemExit("frozen sweep changed")
    c = session_clusters(view, d["family"])
    weight = d["weight"].astype(float)
    cost = d["cost"].astype(float).mean(1)  # mean over draws: (sessions, levels, correlations, forms, policies)
    frozen = sw["cost"].astype(float).mean(1)  # (sessions, levels, variants, policies)
    out = dict(sessions=len(weight), types=c.k, beta=meta["beta"], scales=meta["scales"], curves={}, shifts={}, versus_independent={}, realized_auroc={})
    _, first = np.unique(d["family"], return_index=True)
    boots = {}
    for m, form in enumerate(meta["forms"]):
        anchor = form == "base"
        for policy in meta["policies"]:
            pi = SWEEP_POLICIES.index(policy)
            curves = {"independent": frozen[:, :, SWEEP_VARIANTS.index(form), pi]}
            for k, corr in enumerate(meta["correlations"]):
                curves[corr] = cost[:, :, k, m, meta["policies"].index(policy)]
            for corr, x in curves.items():
                summary, boot = curve_summary(c, x, weight, anchor)
                out["curves"][f"{form}/{corr}/{policy}"] = summary
                boots[form, corr, policy] = boot
            for corr in meta["correlations"]:
                out["versus_independent"][f"{form}/{corr}/{policy}"] = {f"{lvl:g}": c.ratio(curves[corr][:, j], curves["independent"][:, j]) for j, lvl in enumerate(LEVELS)}
                out["shifts"][f"{form}/{corr}/{policy}"] = {f"{t:g}": crossing_shift(boots[form, corr, policy], boots[form, "independent", policy], anchor, t) for t in THRESHOLDS}
        for k, corr in enumerate(meta["correlations"]):
            out["realized_auroc"][f"{form}/{corr}"] = {f"{lvl:g}": float(np.nanmean(d["auroc"][first][:, :, j, k, m])) for j, lvl in enumerate(LEVELS)}
        out["realized_auroc"][f"{form}/independent"] = {f"{lvl:g}": float(np.nanmean(sw["auroc"][first][:, :, j, SWEEP_VARIANTS.index(form)])) for j, lvl in enumerate(LEVELS)}
    return out


# --------------------------------------------------------------------------- strata of the frozen sweep


def sweep_strata(view, split):
    sw = load(pc.ROOT / f"outputs/v35/sweep/{view}-{split}.npz")
    c = session_clusters(view, sw["family"])
    weight = sw["weight"].astype(float)
    cost = sw["cost"].astype(float).mean(1)
    size = sw["size"]
    out = {}
    for lo, hi, label in ((10, np.inf, "all"),) + SIZE_BINS:
        mask = (size >= lo) & (size < hi)
        sub = pc.Clusters(pc.family_types(view, sw["family"][mask]))
        entry = dict(sessions=int(mask.sum()), types=sub.k, weight_checks=float(weight[mask].sum()), share_of_weight_checks=float(weight[mask].sum() / weight.sum()), curves={})
        for k, variant in enumerate(SWEEP_VARIANTS):
            for m, policy in enumerate(SWEEP_POLICIES):
                summary, _ = curve_summary(sub, cost[mask][:, :, k, m], weight[mask], anchor=variant != "oracle")
                entry["curves"][f"{variant}/{policy}"] = summary
        out[label] = entry
    return out


def predictor_beyond_seeds(view, split):
    """seeded<ff> against seedmlp<ff> (same confirmed seeds, topology prior elsewhere), question by question."""
    a_meta = json.loads((pc.OUT / f"runs/{view}-{split}.json").read_text())
    b_meta = json.loads((pc.OUT / f"controls/runs/{view}-{split}.json").read_text())
    a, b = load(pc.OUT / f"runs/{view}-{split}.npz"), load(pc.OUT / f"controls/runs/{view}-{split}.npz")
    if not (np.array_equal(a["family"], b["family"]) and np.array_equal(a["rho"], b["rho"])):
        raise SystemExit("seeded and control runs are not aligned")
    ia, ib = {n: i for i, n in enumerate(a_meta["arms"])}, {n: i for i, n in enumerate(b_meta["arms"])}
    c = session_clusters(view, a["family"])
    out = {}
    for f in pc.FRACTIONS:
        tag = pc.fraction_tag(f)
        for q in ("member", "direction", "ag", "recal"):
            out[f"seeded{tag}:{q}/seedmlp{tag}:{q}"] = c.ratio(a["cost"][:, ia[f"seeded{tag}:{q}"]], b["cost"][:, ib[f"seedmlp{tag}:{q}"]])
    return out


def synthetic_equivalent(curves, ratio):
    """AUROC at which the calibrated independent AG curve has the given cost ratio (linear interpolation)."""
    x = [0.5] + list(LEVELS)
    y = [1.0] + [curves["ratio"][f"{lvl:g}"]["value"] for lvl in LEVELS]
    for i in range(len(y) - 1):
        if (y[i] - ratio) * (y[i + 1] - ratio) <= 0 and y[i] != y[i + 1]:
            return x[i] + (y[i] - ratio) * (x[i + 1] - x[i]) / (y[i] - y[i + 1])
    return None if ratio > y[0] else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=pc.SPLITS)
    parser.add_argument("--output", type=Path, default=pc.OUT / "analysis")
    args = parser.parse_args()
    code = pc.guard(args.split, __file__) or pc.code_identity(__file__)
    out_path = args.output / f"{args.split}.json"
    if out_path.exists():
        raise FileExistsError(out_path)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "priors", "analyze", out_path)
    data = [p for sub in ("", "extra/", "controls/") for p in sorted((pc.OUT / sub).glob("seed*-*.npz"))]
    pc.guard_data(args.split, data)
    result = {}
    for view in pc.VIEWS:
        r = {}
        if (pc.OUT / f"runs/{view}-{args.split}.npz").exists():
            r["seeded"] = seeded(view, args.split)
        for key, sub in (("seeded_extra", "extra"), ("seeded_controls", "controls")):
            if (pc.OUT / f"{sub}/runs/{view}-{args.split}.npz").exists():
                r[key] = seeded(view, args.split, base=pc.OUT / sub)
        if (pc.OUT / f"corrsweep/{view}-{args.split}.npz").exists():
            r["correlated"] = correlated(view, args.split)
        if (pc.ROOT / f"outputs/v35/sweep/{view}-{args.split}.npz").exists():
            r["sweep_strata"] = sweep_strata(view, args.split)
        if "seeded" in r and "seeded_controls" in r:
            r["seeded_vs_controls"] = predictor_beyond_seeds(view, args.split)
        for key in ("seeded", "seeded_extra", "seeded_controls"):
            if key in r and "sweep_strata" in r:
                curves = r["sweep_strata"]["all"]["curves"]
                r[key]["synthetic_equivalent_auroc"] = {
                    n: synthetic_equivalent(curves[f"base/{n.split(':')[1]}"], v["value"])
                    for n, v in r[key]["ratio_to_weight"].items() if n.split(":")[-1] in ("direction", "ag", "recal") and ":" in n}
        result[view] = r
    pc.write_json_new(out_path, dict(split=args.split, views=result, code=code, data={str(p.relative_to(pc.ROOT)): pc.hash_file(p) for p in data}))
    print_summary(result)


def fmt(x):
    return f"{x['value']:.3f} [{x['low']:.3f}, {x['high']:.3f}]"


def print_summary(result):
    for view, r in result.items():
        print("=" * 20, view)
        for key in ("seeded", "seeded_extra", "seeded_controls"):
            if key not in r:
                continue
            s = r[key]
            print(f" seeded priors, {s['sessions']} certifiable questions, {s['types']} types")
            for name, v in s["ratio_to_weight"].items():
                print(f"   {name:28s} {fmt(v)}")
            for name, q in s["quality"].items():
                al = q["questions_all_cells"]
                if "questions_nonseed_cells" not in q or "input_nonseed" not in q:
                    print(f"   quality {name:16s} AUROC(all) {al['member_auroc_within_question']:.3f} Brier {al['member_brier']:.3f} ECE {al['member_ece']:.3f}")
                    continue
                a, b = q["questions_nonseed_cells"], q["input_nonseed"]
                print(f"   quality {name:16s} AUROC(nonseed) {a['member_auroc_within_question']:.3f} AUROC(all) {al['member_auroc_within_question']:.3f} "
                      f"Brier {a['member_brier']:.3f} ECE {a['member_ece']:.3f} top1 {b['top1']:.3f} top1-resolved {b['top1_resolved']:.3f}")
        if "correlated" in r:
            cc = r["correlated"]
            for key, v in cc["curves"].items():
                ratios = " ".join(f"{v['ratio'][f'{lvl:g}']['value']:.3f}" for lvl in LEVELS)
                cr = v["crossing"]
                print(f"   {key:26s} {ratios} | pays(<=1) {cr['1']['value']:.3f} [{cr['1']['low']}, {cr['1']['high']}] <=0.95 {cr['0.95']['value']:.3f} [{cr['0.95']['low']}, {cr['0.95']['high']}]")
        if "sweep_strata" in r:
            for label, e in r["sweep_strata"].items():
                ag = e["curves"]["base/ag"]
                print(f"   strata {label:8s} n={e['sessions']:5d} share {e['share_of_weight_checks']:.2f} base/ag " + " ".join(f"{ag['ratio'][f'{lvl:g}']['value']:.3f}" for lvl in LEVELS)
                      + f" | pays {ag['crossing']['1']['value']:.3f} <=0.95 {ag['crossing']['0.95']['value']:.3f}")


if __name__ == "__main__":
    main()
