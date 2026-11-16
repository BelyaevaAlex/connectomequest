#!/usr/bin/env python3
"""Analyse the controls added after the held-out run, and per-question distributions.

Reads the frozen per-question results and the post hoc results of one split. Checks
that every arm present in both has identical costs, then reports
* the no-prediction priors and the round-robin order (certifiable questions);
* costs under the three-way stopping rule (uncertifiable and all questions);
* the hybrid sampling test;
* per-question distributions of the frozen arms (median, geometric mean, shares of
  questions better / tied / worse than weight order, totals without large neurons).
Intervals resample target cell types, as in the frozen analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = 2000


class Split:
    def __init__(self, view, split):
        base = ROOT / "outputs/v35"
        self.view = view
        self.frozen_meta = json.loads((base / f"runs/{view}-{split}.json").read_text())
        self.post_meta = json.loads((base / f"posthoc/{view}-{split}.json").read_text())
        with np.load(base / f"runs/{view}-{split}.npz", allow_pickle=False) as z:
            self.frozen = {k: z[k] for k in z.files}
        with np.load(base / f"posthoc/{view}-{split}.npz", allow_pickle=False) as z:
            self.post = {k: z[k] for k in z.files}
        with np.load(base / f"benchmark/{view}.npz", allow_pickle=False) as z:
            target_type, fam_target, fam_kind = z["target_type"], z["fam_target"], z["fam_kind"]
        for key in ("family", "rho", "size", "truth"):
            if not np.array_equal(self.frozen[key], self.post[key]):
                raise ValueError(f"post hoc rows are not aligned with the frozen rows ({key})")
        self.farm = {n: i for i, n in enumerate(self.frozen_meta["arms"])}
        self.parm = {n: i for i, n in enumerate(self.post_meta["arms"])}
        self.truth = self.frozen["truth"]
        self.size = self.frozen["size"].astype(float)
        self.kind = fam_kind[self.frozen["family"]]
        self.n = len(self.truth)
        cluster = target_type[fam_target[self.frozen["family"]]]
        clusters, self.index = np.unique(cluster, return_inverse=True)
        self.k = len(clusters)
        rng = np.random.default_rng(20261112)
        self.counts = np.stack([np.bincount(row, minlength=self.k) for row in rng.integers(0, self.k, size=(BOOTSTRAP, self.k))]).astype(float)

    def sums(self, values, mask):
        return np.bincount(self.index[mask], weights=values[mask], minlength=self.k)

    def ratio(self, x, y, mask):
        sx, sy = self.sums(x, mask), self.sums(y, mask)
        boot = (self.counts @ sx) / (self.counts @ sy)
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    def frozen_cost(self, name):
        d, a = self.frozen, self.farm
        if name == "stat":
            return np.mean([d["cost"][:, a[f"stat_r{r}"]] for r in range(5)], 0)
        if name == "clairvoyant":
            return np.where(d["clairvoyant"] < 0, d["size"], d["clairvoyant"]).astype(float)
        return d["cost"][:, a[name]].astype(float)

    def post_cost(self, name, three_way=False):
        return self.post["cost3" if three_way else "cost"][:, self.parm[name]].astype(float)


def analyse(view, split):
    s = Split(view, split)
    certifiable, uncertifiable, everything = s.truth > 0, s.truth == 0, np.ones(s.n, dtype=bool)
    weight = s.frozen_cost("weight")
    out = dict(view=view, split=split, sessions=s.n, post_result_sha256=s.post_meta["result_sha256"])
    # Integrity: arms present in both result sets must agree exactly.
    common = [n for n in s.parm if n in s.farm]
    mismatched = [n for n in common if not np.array_equal(s.post["cost"][:, s.parm[n]], s.frozen["cost"][:, s.farm[n]])]
    out["integrity"] = dict(common_arms=len(common), mismatched=mismatched)
    if mismatched:
        raise SystemExit(f"post hoc costs differ from the frozen results: {mismatched}")
    # No-prediction priors and round robin.
    controls = {}
    for name in s.parm:
        if name.startswith("const_") or name.endswith(":rr"):
            c = s.post_cost(name)
            controls[name] = dict(
                ratio_to_weight=s.ratio(c, weight, certifiable),
                share_identical_to_weight=float((c[certifiable] == weight[certifiable]).mean()),
                largest_difference=float(np.abs(c[certifiable] - weight[certifiable]).max()),
            )
    for prior in ("mlp_s0", "ntac", "provisional"):
        if f"{prior}:rr" in s.parm:
            controls[f"{prior}:rr/direction"] = s.ratio(s.post_cost(f"{prior}:rr"), s.post_cost(f"{prior}:direction"), certifiable)
            controls[f"{prior}:rr/recal"] = s.ratio(s.post_cost(f"{prior}:rr"), s.post_cost(f"{prior}:recal"), certifiable)
    out["controls"] = controls
    # Three-way stopping.
    weight3 = s.post_cost("weight", True)
    bound3 = np.where(s.post["bound3"] < 0, weight3, s.post["bound3"]).astype(float)
    three = dict(
        uncertifiable_questions=int(uncertifiable.sum()),
        weight_fraction_of_inputs=s.ratio(weight3, s.size, uncertifiable),
        lower_bound_over_weight=s.ratio(bound3, weight3, uncertifiable),
        share_of_weight_checks_on_uncertifiable_exhaustive=float(weight[uncertifiable].sum() / weight.sum()),
        share_of_weight_checks_on_uncertifiable_three_way=float(weight3[uncertifiable].sum() / weight3.sum()),
        uncertifiable_ratio_to_weight={n: s.ratio(s.post_cost(n, True), weight3, uncertifiable) for n in s.parm if n != "weight"},
        all_questions_ratio_to_weight={n: s.ratio(s.post_cost(n, True), weight3, everything) for n in s.parm if n != "weight"},
        all_questions_ratio_exhaustive={n: s.ratio(s.post_cost(n), weight, everything) for n in s.parm if n != "weight"},
    )
    out["three_way"] = three
    # Hybrid sampling test.
    hybrid = s.post["hybrid_cost"].mean(1)
    status = s.post["hybrid_status"]
    wrong = (status != 0) & (status != s.truth[:, None])
    strata = {}
    for lo, hi, label in ((10, 50, "10-49"), (50, 150, "50-149"), (150, 500, "150-499"), (500, np.inf, ">=500")):
        m = certifiable & (s.size >= lo) & (s.size < hi)
        strata[label] = dict(questions=int(m.sum()), hybrid=s.ratio(hybrid, weight, m), sampling=s.ratio(s.frozen_cost("stat"), weight, m))
    out["hybrid"] = dict(
        ratio_to_weight=s.ratio(hybrid, weight, certifiable),
        ratio_to_sampling=s.ratio(hybrid, s.frozen_cost("stat"), certifiable),
        strata=strata,
        audits=int(status.size),
        wrong_rate=float(wrong.mean()),
        wrong_rate_certifiable=float(wrong[certifiable].mean()),
        wrong_rate_uncertifiable=float(wrong[uncertifiable].mean()),
        share_using_risk=float((~s.post["hybrid_exact"]).mean()),
        share_identical_to_weight=float((s.post["hybrid_cost"][certifiable] == weight[certifiable, None]).mean()),
    )
    # Per-question distributions of the frozen arms.
    arms = ["clairvoyant", "stat", "mlp_s0:member", "mlp_s0:direction", "mlp_s0:ag", "mlp_s0:adg", "mlp_s0:recal", "ntac:member", "linear:member", "ntac:ag", "linear:ag"]
    if view == "flywire":
        arms += ["provisional:member", "provisional:direction", "provisional:ag", "provisional:recal", "provisional:adir"]
    small = certifiable & (s.size < 500)
    per = {}
    for name in arms:
        c = s.frozen_cost(name)
        r = c[certifiable] / weight[certifiable]
        per[name] = dict(
            ratio_of_totals=s.ratio(c, weight, certifiable),
            ratio_of_totals_below_500_inputs=s.ratio(c, weight, small),
            median=float(np.median(r)),
            geometric_mean=float(np.exp(np.log(r).mean())),
            mean=float(r.mean()),
            better=float((c[certifiable] < weight[certifiable]).mean()),
            tied=float((c[certifiable] == weight[certifiable]).mean()),
            worse=float((c[certifiable] > weight[certifiable]).mean()),
            at_least_double=float((r >= 2).mean()),
            largest=float(r.max()),
        )
    large = certifiable & (s.size >= 500)
    out["per_question"] = dict(
        arms=per,
        weight_median_checks=float(np.median(weight[certifiable])),
        weight_mean_checks=float(weight[certifiable].mean()),
        weight_equals_clairvoyant=float((weight[certifiable] == s.frozen_cost("clairvoyant")[certifiable]).mean()),
        large_share_of_questions=float(large.sum() / certifiable.sum()),
        large_share_of_weight_checks=float(weight[large].sum() / weight[certifiable].sum()),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    args = parser.parse_args()
    result = {view: analyse(view, args.split) for view in ("malecns", "flywire")}
    path = ROOT / f"outputs/v35/analysis/posthoc-{args.split}.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for view, r in result.items():
        print("=====", view, r["integrity"])
        for name, c in r["controls"].items():
            x = c.get("ratio_to_weight", c)
            print(f"  control {name:28s} {x['value']:.3f} [{x['low']:.3f},{x['high']:.3f}]", {k: round(v, 4) for k, v in c.items() if isinstance(v, float)})
        t = r["three_way"]
        print("  three-way: weight checks/inputs", t["weight_fraction_of_inputs"], "LB/weight", t["lower_bound_over_weight"])
        print("   uncertifiable ratios", {k: round(v["value"], 3) for k, v in t["uncertifiable_ratio_to_weight"].items()})
        print("   all-question ratios (3-way)", {k: round(v["value"], 3) for k, v in t["all_questions_ratio_to_weight"].items()})
        h = r["hybrid"]
        print("  hybrid", h["ratio_to_weight"], "vs sampling", h["ratio_to_sampling"]["value"], "wrong", h["wrong_rate"], h["wrong_rate_certifiable"], h["wrong_rate_uncertifiable"], "risk", h["share_using_risk"], "identical", h["share_identical_to_weight"])
        print("   strata", {k: (v["questions"], round(v["hybrid"]["value"], 3), round(v["hybrid"]["low"], 3), round(v["hybrid"]["high"], 3)) for k, v in h["strata"].items()})
        q = r["per_question"]
        print("  per-question: weight median", q["weight_median_checks"], "mean", round(q["weight_mean_checks"], 1), "=clairvoyant", round(q["weight_equals_clairvoyant"], 3), "large share q/checks", round(q["large_share_of_questions"], 3), round(q["large_share_of_weight_checks"], 3))
        for name, a in q["arms"].items():
            print(f"   {name:24s} totals {a['ratio_of_totals']['value']:.3f} <500 {a['ratio_of_totals_below_500_inputs']['value']:.3f} median {a['median']:.3f} geo {a['geometric_mean']:.3f} mean {a['mean']:.3f} better/tied/worse {a['better']:.3f}/{a['tied']:.3f}/{a['worse']:.3f} >=2x {a['at_least_double']:.3f}")


if __name__ == "__main__":
    main()
