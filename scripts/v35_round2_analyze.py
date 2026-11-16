#!/usr/bin/env python3
"""Analyses requested in the second round of review, for one split.

* Priors of controlled quality (``v35_prior_sweep.py``): realized area under the ROC
  curve, cost of every policy relative to weight order, and contrasts between
  policies, for each level and each treatment of the unresolved answer; the
  connectivity priors with a base-rate unresolved mass. Arms that exist in the
  frozen results must reproduce them exactly.
* Robustness of the frozen results: paired per-question comparisons, a single pass
  over the four retention levels of a question family for orders that do not depend
  on the threshold, and the endpoint without questions that one check can settle.
Intervals resample target cell types, as in the frozen analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = 2000


class Clusters:
    def __init__(self, view, family):
        with np.load(ROOT / f"outputs/v35/benchmark/{view}.npz", allow_pickle=False) as z:
            target_type, fam_target = z["target_type"], z["fam_target"]
        clusters, self.index = np.unique(target_type[fam_target[family]], return_inverse=True)
        self.k = len(clusters)
        rng = np.random.default_rng(20261112)
        self.counts = np.stack([np.bincount(row, minlength=self.k) for row in rng.integers(0, self.k, size=(BOOTSTRAP, self.k))]).astype(float)

    def ratio(self, x, y, mask=None):
        mask = np.ones(len(x), dtype=bool) if mask is None else mask
        sx = np.bincount(self.index[mask], weights=np.asarray(x, dtype=float)[mask], minlength=self.k)
        sy = np.bincount(self.index[mask], weights=np.asarray(y, dtype=float)[mask], minlength=self.k)
        boot = (self.counts @ sx) / (self.counts @ sy)
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def paired(c, ref):
    r = c / ref
    return dict(better=float((c < ref).mean()), tied=float((c == ref).mean()), worse=float((c > ref).mean()),
                geometric_mean=float(np.exp(np.log(r).mean())), at_least_double=float((r >= 2).mean()), half_or_less=float((r <= 0.5).mean()))


def sweep(view, split):
    base = ROOT / "outputs/v35"
    meta = json.loads((base / f"sweep/{view}-{split}.json").read_text())
    with np.load(base / f"sweep/{view}-{split}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    frozen_meta = json.loads((base / f"runs/{view}-{split}.json").read_text())
    with np.load(base / f"runs/{view}-{split}.npz", allow_pickle=False) as z:
        f = {k: z[k] for k in ("family", "rho", "truth", "cost")}
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(f["family"], f["rho"]))}
    rows = np.array([key[int(a), int(b)] for a, b in zip(d["family"], d["rho"])])
    arm = {n: i for i, n in enumerate(frozen_meta["arms"])}
    weight = d["weight"].astype(float)
    checks = {"weight": weight}
    for name, frozen in (("mlp_s0:as_is:ag", "mlp_s0:ag"), ("mlp_s0:as_is:recal", "mlp_s0:recal"), ("ntac:as_is:ag", "ntac:ag"), ("ntac:as_is:recal", "ntac:recal")):
        checks[name] = f["cost"][rows, arm[frozen]]
    real = {n: d["real"][:, i].astype(float) for i, n in enumerate(meta["real_arms"])}
    mismatched = [n for n in checks if not np.array_equal(real.get(n, weight) if n != "weight" else weight, f["cost"][rows, arm["weight"]] if n == "weight" else checks[n])]
    if mismatched or len(rows) != int((f["truth"] > 0).sum()):
        raise SystemExit(f"sweep does not reproduce the frozen costs or rows: {mismatched}")
    c = Clusters(view, d["family"])
    cost = d["cost"].astype(float).mean(1)  # average over noise draws: (questions, levels, variants, policies)
    levels, variants, policies = meta["aurocs"], meta["variants"], meta["policies"]
    _, first = np.unique(d["family"], return_index=True)
    auroc = d["auroc"][first]
    out = dict(rates=meta["rates"], questions=len(rows), realized_auroc={}, ratio_to_weight={}, contrasts={}, per_question={})
    for j, level in enumerate(levels):
        out["realized_auroc"][f"{level:g}"] = {variant: float(np.nanmean(auroc[:, :, j, k])) for k, variant in enumerate(variants)}
        for k, variant in enumerate(variants):
            tag = f"{level:g}/{variant}"
            x = {p: cost[:, j, k, m] for m, p in enumerate(policies)}
            out["ratio_to_weight"][tag] = {p: c.ratio(v, weight) for p, v in x.items()}
            out["contrasts"][tag] = {
                "ag/direction": c.ratio(x["ag"], x["direction"]),
                "recal/ag": c.ratio(x["recal"], x["ag"]),
                "recal/direction": c.ratio(x["recal"], x["direction"]),
                "rr/direction": c.ratio(x["rr"], x["direction"]),
                "rr/best": c.ratio(x["rr"], np.minimum(x["member"], x["direction"])),
                "best/recal": c.ratio(np.minimum(x["direction"], x["ag"]), x["recal"]),
            }
            out["per_question"][tag] = {"recal_vs_direction": paired(x["recal"], x["direction"]), "ag_vs_direction": paired(x["ag"], x["direction"])}
    out["real_priors"] = {n: c.ratio(v, weight) for n, v in real.items()}
    out["real_contrasts"] = {}
    for prior in ("mlp_s0", "ntac"):
        out["real_contrasts"][prior] = {
            "base_ag/as_is_ag": c.ratio(real[f"{prior}:base:ag"], real[f"{prior}:as_is:ag"]),
            "as_is_recal/as_is_ag": c.ratio(real[f"{prior}:as_is:recal"], real[f"{prior}:as_is:ag"]),
            "base_recal/base_ag": c.ratio(real[f"{prior}:base:recal"], real[f"{prior}:base:ag"]),
        }
    return out


def robustness(view, split):
    base = ROOT / "outputs/v35"
    meta = json.loads((base / f"runs/{view}-{split}.json").read_text())
    with np.load(base / f"runs/{view}-{split}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in ("family", "rho", "truth", "cost", "clairvoyant", "size")}
    arm = {n: i for i, n in enumerate(meta["arms"])}
    cert = d["truth"] > 0
    c = Clusters(view, d["family"])
    cost = {n: d["cost"][:, i].astype(float) for n, i in arm.items()}
    cost["stat"] = np.mean([cost[f"stat_r{r}"] for r in range(5)], 0)
    weight = cost["weight"]
    priors = ["mlp_s0", "linear", "ntac"] + (["provisional"] if "provisional:member" in arm else [])
    out = {}
    # Paired per-question comparisons.
    pairs = [("mlp_s0:recal", "mlp_s0:ag"), ("mlp_s0:recal", "mlp_s0:direction")]
    if "provisional:recal" in arm:
        pairs += [("provisional:recal", "provisional:direction"), ("provisional:ag", "provisional:direction"), ("provisional:recal", "provisional:ag")]
    out["paired"] = {f"{a}/{b}": dict(ratio_of_totals=c.ratio(cost[a], cost[b], cert), **paired(cost[a][cert], cost[b][cert])) for a, b in pairs}
    # A single pass over the levels of a family, for threshold-free orders.
    families = np.unique(d["family"][cert])
    position = {int(f): i for i, f in enumerate(families)}
    group = np.array([position.get(int(f), -1) for f in d["family"]])
    single = {}
    for name in ["weight"] + [f"{p}:{o}" for p in priors for o in ("member", "nonmember")]:
        best = np.zeros(len(families))
        np.maximum.at(best, group[cert], cost[name][cert])
        single[name] = best
    fam_cluster = Clusters(view, families)
    out["single_pass"] = dict(
        families=len(families),
        weight_inflation=float(weight[cert].sum() / single["weight"].sum()),
        ratio_to_weight={n: fam_cluster.ratio(v, single["weight"]) for n, v in single.items() if n != "weight"},
        per_question_ratio_to_weight={n: c.ratio(cost[n], weight, cert) for n in single if n != "weight"},
    )
    # Without the questions that one check can settle.
    trivial = cert & (d["clairvoyant"] == 1)
    rest = cert & ~trivial
    names = ["stat", "mlp_s0:member", "mlp_s0:direction", "mlp_s0:ag", "mlp_s0:adg", "mlp_s0:recal", "linear:ag", "ntac:ag"]
    names += [f"provisional:{p}" for p in ("member", "direction", "ag", "recal", "adir")] if "provisional:recal" in arm else []
    clair = np.where(d["clairvoyant"] < 0, d["size"], d["clairvoyant"]).astype(float)
    out["nontrivial"] = dict(
        share_of_certifiable=float(trivial.sum() / cert.sum()),
        weight_settles_trivial_in_one=float((weight[trivial] == 1).mean()),
        ratio_to_weight={n: c.ratio(cost[n], weight, rest) for n in names},
        clairvoyant=c.ratio(clair, weight, rest),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    args = parser.parse_args()
    result = {view: dict(sweep=sweep(view, args.split), robustness=robustness(view, args.split)) for view in ("malecns", "flywire")}
    path = ROOT / f"outputs/v35/analysis/round2-{args.split}.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for view, r in result.items():
        s = r["sweep"]
        print("=====", view, "questions", s["questions"], "rates", {k: v for k, v in s["rates"].items()})
        print(" realized AUROC", {k: {a: round(b, 3) for a, b in v.items()} for k, v in s["realized_auroc"].items()})
        for tag, x in s["ratio_to_weight"].items():
            con = s["contrasts"][tag]
            print(f"  {tag:10s}", {p: round(v["value"], 3) for p, v in x.items()}, "| ag/dir", round(con["ag/direction"]["value"], 3), "recal/ag", round(con["recal/ag"]["value"], 3),
                  "recal/dir", f"{con['recal/direction']['value']:.3f} [{con['recal/direction']['low']:.3f},{con['recal/direction']['high']:.3f}]", "best/recal", round(con["best/recal"]["value"], 3))
        print(" real", {k: round(v["value"], 3) for k, v in s["real_priors"].items()})
        print(" real contrasts", {p: {k: round(v["value"], 3) for k, v in x.items()} for p, x in s["real_contrasts"].items()})
        rb = r["robustness"]
        print(" paired", {k: {a: (round(b["value"], 3) if isinstance(b, dict) else round(b, 3)) for a, b in v.items()} for k, v in rb["paired"].items()})
        sp = rb["single_pass"]
        print(" single pass: inflation", round(sp["weight_inflation"], 3), {n: (round(v["value"], 3), round(sp["per_question_ratio_to_weight"][n]["value"], 3)) for n, v in sp["ratio_to_weight"].items()})
        nt = rb["nontrivial"]
        print(" nontrivial: share trivial", round(nt["share_of_certifiable"], 3), "weight settles in one", round(nt["weight_settles_trivial_in_one"], 3), {n: round(v["value"], 3) for n, v in nt["ratio_to_weight"].items()}, "clair", round(nt["clairvoyant"]["value"], 3))


if __name__ == "__main__":
    main()
