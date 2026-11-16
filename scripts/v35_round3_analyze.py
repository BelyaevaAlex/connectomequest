#!/usr/bin/env python3
"""Analyses requested in the third round of review, for one split.

* Earlier-release prior built in other ways (FlyWire): cost relative to weight order
  and contrasts between policies for each construction.
* Ablation of the recalibration: cost of each variant relative to weight order and to
  Adaptive Greedy with the same prior, for the topology prior, the earlier-release
  prior and the confident synthetic priors at an AUROC of 0.6 and 0.8.
* Prior quality: the within-question AUROC of the topology and NTAC priors, and the
  cost that the calibrated synthetic priors of the second round predict for them, once
  from the mean AUROC and once question by question (interpolating each question's
  synthetic costs at that question's AUROC; AUROC 0.5 is weight order, which a
  constant prior reproduces exactly).
Arms that exist in the frozen or second-round results must reproduce them exactly.
Intervals resample target cell types, as in the frozen analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v35_round2_analyze import Clusters, paired

ROOT = Path(__file__).resolve().parents[1]
GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def load(kind, view, split):
    base = ROOT / "outputs/v35"
    meta = json.loads((base / f"{kind}/{view}-{split}.json").read_text())
    with np.load(base / f"{kind}/{view}-{split}.npz", allow_pickle=False) as z:
        return meta, {k: z[k] for k in z.files}


def align(rows_from, rows_to):
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rows_to["family"], rows_to["rho"]))}
    return np.array([key[int(a), int(b)] for a, b in zip(rows_from["family"], rows_from["rho"])])


def interpolate(a, curve):
    """Row-wise linear interpolation of ``curve`` (questions x grid) at ``a``, clamped to the grid."""
    a = np.clip(a, GRID[0], GRID[-1])
    j = np.clip(np.searchsorted(GRID, a, side="right") - 1, 0, len(GRID) - 2)
    lo, hi = np.asarray(GRID)[j], np.asarray(GRID)[j + 1]
    t = (a - lo) / (hi - lo)
    rows = np.arange(len(a))
    return (1 - t) * curve[rows, j] + t * curve[rows, j + 1]


def analyse(view, split):
    meta, d = load("round3", view, split)
    fmeta, f = load("runs", view, split)
    smeta, sw = load("sweep", view, split)
    arm = {n: i for i, n in enumerate(fmeta["arms"])}
    rf, rs = align(d, f), align(d, sw)
    weight = d["weight"].astype(float)
    frozen = lambda name: f["cost"][rf, arm[name]].astype(float)
    # Integrity: rows, weight order, and every arm that reproduces an earlier one.
    problems = []
    if len(rf) != int((f["truth"] > 0).sum()) or not np.array_equal(weight, frozen("weight")):
        problems.append("rows or weight order")
    variants, constructions, policies = meta["variants"], meta["constructions"], meta["policies"]
    vi = variants.index("frozen")
    for p, prior in enumerate(meta["real_priors"]):
        if f"{prior}:recal" in arm and not np.array_equal(d["recal_real"][:, p, vi], frozen(f"{prior}:recal")):
            problems.append(f"{prior} recal")
    level_index = [smeta["aurocs"].index(level) for level in meta["synthetic_levels"]]
    oracle, recal_policy = smeta["variants"].index("oracle"), smeta["policies"].index("recal")
    for j, level in enumerate(level_index):
        if not np.array_equal(d["recal_syn"][:, :, j, vi], sw["cost"][rs, :, level, oracle, recal_policy]):
            problems.append(f"synthetic recal at level {meta['synthetic_levels'][j]}")
    if "provisional:member" in arm:
        a = constructions.index("a0.9")
        for m, policy in enumerate(policies):
            if not np.array_equal(d["earlier"][:, a, m], frozen(f"provisional:{policy}")):
                problems.append(f"earlier {policy}")
    if problems:
        raise SystemExit(f"third-round arms do not reproduce earlier results: {problems}")
    c = Clusters(view, d["family"])
    out = dict(questions=len(rf), development_class_rates=meta["development_class_rates"])
    # Earlier-release prior built in other ways.
    if "provisional:member" in arm:
        e = {}
        for a, name in enumerate(constructions):
            x = {p: d["earlier"][:, a, m].astype(float) for m, p in enumerate(policies)}
            e[name] = dict(
                ratio_to_weight={p: c.ratio(v, weight) for p, v in x.items()},
                contrasts={"ag/direction": c.ratio(x["ag"], x["direction"]), "recal/direction": c.ratio(x["recal"], x["direction"]),
                           "recal/ag": c.ratio(x["recal"], x["ag"])},
                recal_vs_direction=paired(x["recal"], x["direction"]),
            )
        out["earlier"] = e
    # Ablation of the recalibration.
    ab = {}
    for p, prior in enumerate(meta["real_priors"]):
        if f"{prior}:ag" not in arm:
            continue
        ag = frozen(f"{prior}:ag")
        ab[prior] = {v: dict(ratio_to_weight=c.ratio(d["recal_real"][:, p, k], weight), over_ag=c.ratio(d["recal_real"][:, p, k], ag))
                     for k, v in enumerate(variants)}
    ag_policy = smeta["policies"].index("ag")
    for j, level in enumerate(level_index):
        ag = sw["cost"][rs, :, level, oracle, ag_policy].astype(float).mean(1)
        tag = f"oracle{meta['synthetic_levels'][j]:g}"
        ab[tag] = {v: dict(ratio_to_weight=c.ratio(d["recal_syn"][:, :, j, k].astype(float).mean(1), weight),
                           over_ag=c.ratio(d["recal_syn"][:, :, j, k].astype(float).mean(1), ag)) for k, v in enumerate(variants)}
    out["ablation"] = ab
    # Prior quality.
    base = smeta["variants"].index("base")
    quality = {}
    for k, prior in enumerate(meta["auroc_priors"]):
        a = d["auroc"][:, k]
        if np.all(np.isnan(a)):
            continue
        defined = ~np.isnan(a)
        summary = dict(mean=float(np.nanmean(a)), sd=float(np.nanstd(a)), quantiles={str(q): float(np.nanquantile(a, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                       undefined=float((~defined).mean()), at_least_0_8=float((a[defined] >= 0.8).mean()), at_least_0_9=float((a[defined] >= 0.9).mean()),
                       below_0_5=float((a[defined] < 0.5).mean()))
        quality[prior] = dict(auroc=summary)
        if f"{prior}:base:ag" not in smeta["real_arms"]:
            continue
        filled = np.where(defined, a, np.nanmean(a))
        for policy in ("ag", "recal"):
            m = smeta["policies"].index(policy)
            levels = [smeta["aurocs"].index(g) for g in GRID[1:]]
            curve = np.column_stack([weight] + [sw["cost"][rs, :, level, base, m].astype(float).mean(1) for level in levels])
            real = sw["real"][rs, smeta["real_arms"].index(f"{prior}:base:{policy}")].astype(float)
            per_question = interpolate(filled, curve)
            aggregate = curve.sum(0) / weight.sum()
            mean_level = float(np.nanmean(a))
            quality[prior][policy] = dict(
                real=c.ratio(real, weight),
                predicted_per_question=c.ratio(per_question, weight),
                predicted_from_mean=float(np.interp(mean_level, GRID, aggregate)),
                real_over_predicted=c.ratio(real, per_question),
                synthetic_curve={f"{g:g}": float(v) for g, v in zip(GRID, aggregate)},
            )
    out["quality"] = quality
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    args = parser.parse_args()
    result = {view: analyse(view, args.split) for view in ("malecns", "flywire")}
    path = ROOT / f"outputs/v35/analysis/round3-{args.split}.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for view, r in result.items():
        print("=====", view, "questions", r["questions"])
        for name, x in r.get("earlier", {}).items():
            print(f"  earlier {name:10s}", {p: round(v["value"], 3) for p, v in x["ratio_to_weight"].items()},
                  {k: f"{v['value']:.3f} [{v['low']:.3f},{v['high']:.3f}]" for k, v in x["contrasts"].items()})
        for prior, x in r["ablation"].items():
            print(f"  ablation {prior:11s}", {v: (round(y["ratio_to_weight"]["value"], 3), round(y["over_ag"]["value"], 3)) for v, y in x.items()})
        for prior, x in r["quality"].items():
            print(f"  quality {prior}", {k: round(v, 3) if isinstance(v, float) else v for k, v in x["auroc"].items()})
            for policy in ("ag", "recal"):
                if policy in x:
                    q = x[policy]
                    print(f"     {policy}: real {q['real']['value']:.3f} per-question {q['predicted_per_question']['value']:.3f} from-mean {q['predicted_from_mean']:.3f} real/pred {q['real_over_predicted']['value']:.3f}")


if __name__ == "__main__":
    main()
