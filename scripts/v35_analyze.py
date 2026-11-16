#!/usr/bin/env python3
"""Aggregate one split of the V35 benchmark into the numbers the manuscript reports.

All uncertainty statements resample target cell types with replacement (cluster
bootstrap): both endpoints of a type, both partner sets and all four thresholds
move together. Cost ratios are ratios of total paid observations. The same code is
used unchanged for the development and the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = 2000
STRONG = {"malecns": [], "flywire": ["provisional"]}
POLICIES = ("member", "nonmember", "confident", "direction", "ggn", "ag", "adg", "recal", "adir", "adir-recal")


def load(view, split, runs, benchmark):
    meta = json.loads((runs / f"{view}-{split}.json").read_text())
    with np.load(runs / f"{view}-{split}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(benchmark / f"{view}.npz", allow_pickle=False) as z:
        bench = {k: z[k] for k in z.files}
    return meta, data, bench


class Split:
    def __init__(self, view, split, runs, benchmark):
        self.view = view
        self.meta, self.d, self.bench = load(view, split, runs, benchmark)
        self.manifest = json.loads((benchmark / "manifest.json").read_text())
        self.arms = {n: i for i, n in enumerate(self.meta["arms"])}
        d = self.d
        self.n = len(d["size"])
        self.size = d["size"].astype(float)
        self.truth = d["truth"]
        self.cluster = self.bench["target_type"][self.bench["fam_target"][d["family"]]]
        self.kind = self.bench["fam_kind"][d["family"]]
        self.priors = [p for p in self.manifest["views"][[v["view"] for v in self.manifest["views"]].index(view)]["priors"]]
        self.rng = np.random.default_rng(20261112)
        clusters, self.cluster_index = np.unique(self.cluster, return_inverse=True)
        self.n_clusters = len(clusters)
        self.draws = self.rng.integers(0, self.n_clusters, size=(BOOTSTRAP, self.n_clusters))
        self.draw_counts = np.stack([np.bincount(row, minlength=self.n_clusters) for row in self.draws]).astype(float)

    def cost(self, name):
        d, a = self.d, self.arms
        if name == "random":
            return np.mean([d["cost"][:, a[f"random_r{r}"]] for r in range(5)], 0)
        if name == "stat":
            return np.mean([d["cost"][:, a[f"stat_r{r}"]] for r in range(5)], 0)
        if name == "clairvoyant":
            return np.where(d["clairvoyant"] < 0, d["size"], d["clairvoyant"]).astype(float)
        if name.endswith(":oracle-direction"):
            p = name.split(":")[0]
            return np.minimum(d["cost"][:, a[p + ":member"]], d["cost"][:, a[p + ":nonmember"]]).astype(float)
        if name.startswith("mlp:"):
            policy = name.split(":")[1]
            return np.mean([d["cost"][:, a[f"mlp_s{s}:{policy}"]] for s in range(5)], 0)
        return d["cost"][:, a[name]].astype(float)

    def sums(self, values, mask):
        return np.bincount(self.cluster_index[mask], weights=values[mask], minlength=self.n_clusters)

    def ratio(self, x, y, mask):
        """Ratio of totals with a type-clustered percentile interval."""
        sx, sy = self.sums(x, mask), self.sums(y, mask)
        boot = (self.draw_counts @ sx) / (self.draw_counts @ sy)
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    def mean(self, x, mask):
        sx, sn = self.sums(x, mask), self.sums(np.ones(self.n), mask)
        boot = (self.draw_counts @ sx) / (self.draw_counts @ sn)
        return dict(value=float(sx.sum() / sn.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def method_names(s):
    names = ["weight", "random", "clairvoyant", "stat"]
    for prior in ["mlp_s0", "linear", "ntac"] + STRONG[s.view]:
        names += [f"{prior}:{p}" for p in POLICIES] + [f"{prior}:oracle-direction"]
    return names


def margin_bins(s):
    """Relative distance of the full-information member share from the threshold."""
    d = s.d
    share, threshold = d["member"] / d["total"], d["threshold"] / d["total"]
    relative = (share - threshold) / threshold
    return relative


def calibration(s, split_types):
    """Within-question ranking and pooled calibration of the projected member probability."""
    b = s.bench
    types = b["target_type"][b["fam_target"]]
    families = np.flatnonzero(np.isin(types, split_types))
    lengths = np.diff(b["target_offset"])[b["fam_target"]]
    plus_offset = np.concatenate(([0], np.cumsum(lengths)))
    out = {}
    for prior in s.priors:
        aurocs, brier_sum, count = [], 0.0, 0
        bins = np.zeros((10, 3))
        unresolved_auc = []
        mean_unknown, unresolved_rate, cells = 0.0, 0.0, 0
        for f in families:
            t = int(b["fam_target"][f])
            lo, hi = int(b["target_offset"][t]), int(b["target_offset"][t + 1])
            labels = b["pre_label"][lo:hi]
            codes = b["fam_codes"][b["fam_code_offset"][f] : b["fam_code_offset"][f + 1]]
            member = np.isin(labels, codes).astype(float)
            p = b["plus_" + prior][plus_offset[f] : plus_offset[f] + hi - lo].astype(float)
            unknown = b["unknown_" + prior][lo:hi].astype(float)
            brier_sum += float(((p - member) ** 2).sum())
            count += len(p)
            idx = np.minimum((p * 10).astype(int), 9)
            np.add.at(bins, (idx, 0), 1)
            np.add.at(bins, (idx, 1), p)
            np.add.at(bins, (idx, 2), member)
            if 0 < member.sum() < len(member):
                aurocs.append(auroc(p, member))
            missing = (labels < 0).astype(float)
            if b["fam_kind"][f] == 0:
                mean_unknown += float(unknown.sum())
                unresolved_rate += float(missing.sum())
                cells += len(missing)
                if 0 < missing.sum() < len(missing):
                    unresolved_auc.append(auroc(unknown, missing))
        nonempty = bins[:, 0] > 0
        ece = float((np.abs(bins[nonempty, 1] - bins[nonempty, 2]).sum()) / count)
        out[prior] = dict(
            member_auroc_within_question=float(np.mean(aurocs)),
            questions_with_both_classes=len(aurocs),
            member_brier=brier_sum / count,
            member_ece=ece,
            unresolved_auroc_within_target=float(np.mean(unresolved_auc)) if unresolved_auc else None,
            mean_predicted_unresolved=mean_unknown / cells,
            observed_unresolved_rate=unresolved_rate / cells,
        )
    return out


def auroc(score, positive):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks over ties.
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        stop = start
        while stop + 1 < len(score) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    n_pos = positive.sum()
    n_neg = len(positive) - n_pos
    return float((ranks[positive > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def analyze(view, split, runs, benchmark):
    s = Split(view, split, runs, benchmark)
    d = s.d
    everything = np.ones(s.n, dtype=bool)
    certifiable = s.truth > 0
    masks = dict(all=everything, certifiable=certifiable, positive=s.truth == 1, negative=s.truth == 2)
    weight = s.cost("weight")
    methods = {}
    for name in method_names(s):
        c = s.cost(name)
        methods[name] = {
            key: dict(ratio_to_weight=s.ratio(c, weight, m), mean_fraction_read=s.mean(c / s.size, m), total=float(c[m].sum()))
            for key, m in masks.items()
        }
    for policy in ("member", "nonmember", "direction", "ag"):
        per_seed = [float(d["cost"][certifiable, s.arms[f"mlp_s{k}:{policy}"]].sum() / weight[certifiable].sum()) for k in range(5)]
        methods[f"mlp:{policy}"] = dict(
            certifiable=dict(ratio_to_weight=s.ratio(s.cost(f"mlp:{policy}"), weight, certifiable)),
            seed_ratios=per_seed,
            seed_mean=float(np.mean(per_seed)),
            seed_sd=float(np.std(per_seed, ddof=1)),
        )
    contrasts = {}

    def contrast(key, x, y, mask=certifiable):
        contrasts[key] = s.ratio(s.cost(x), s.cost(y), mask)

    for prior in ["mlp_s0", "linear", "ntac"] + STRONG[view]:
        contrast(f"{prior}:ag/direction", f"{prior}:ag", f"{prior}:direction")
        contrast(f"{prior}:ag/member", f"{prior}:ag", f"{prior}:member")
        contrast(f"{prior}:ag/oracle-direction", f"{prior}:ag", f"{prior}:oracle-direction")
        contrast(f"{prior}:adg/ag", f"{prior}:adg", f"{prior}:ag")
        contrast(f"{prior}:recal/ag", f"{prior}:recal", f"{prior}:ag")
        contrast(f"{prior}:adir/ag", f"{prior}:adir", f"{prior}:ag")
        contrast(f"{prior}:adir-recal/ag", f"{prior}:adir-recal", f"{prior}:ag")
        contrast(f"{prior}:adir-recal/direction", f"{prior}:adir-recal", f"{prior}:direction")
        contrast(f"{prior}:adir-recal/member", f"{prior}:adir-recal", f"{prior}:member")
        contrast(f"{prior}:adir-recal/oracle-direction", f"{prior}:adir-recal", f"{prior}:oracle-direction")
        contrast(f"{prior}:adir-recal/clairvoyant", f"{prior}:adir-recal", "clairvoyant")
        for key, m in (("positive", masks["positive"]), ("negative", masks["negative"])):
            contrasts[f"{prior}:ag/direction@{key}"] = s.ratio(s.cost(f"{prior}:ag"), s.cost(f"{prior}:direction"), m)
            contrasts[f"{prior}:adir-recal/ag@{key}"] = s.ratio(s.cost(f"{prior}:adir-recal"), s.cost(f"{prior}:ag"), m)
    contrast("linear:ag/mlp_s0:ag", "linear:ag", "mlp_s0:ag")
    contrast("ntac:ag/mlp_s0:ag", "ntac:ag", "mlp_s0:ag")
    contrast("linear:adir-recal/mlp_s0:adir-recal", "linear:adir-recal", "mlp_s0:adir-recal")
    contrast("stat/weight", "stat", "weight")
    contrast("stat/mlp_s0:adir-recal", "stat", "mlp_s0:adir-recal")
    contrast("random/weight", "random", "weight")
    contrast("clairvoyant/weight", "clairvoyant", "weight")
    # Strata.
    rel = margin_bins(s)
    strata = {}
    focus = ["weight", "clairvoyant", "stat", "mlp_s0:ag", "mlp_s0:recal", "mlp_s0:direction", "mlp_s0:member"] + [f"{p}:{q}" for p in STRONG[view] for q in ("member", "direction", "ag", "recal")]

    def stratum(key, mask):
        if mask.sum() == 0:
            return
        strata[key] = dict(
            sessions=int(mask.sum()),
            positive=int((s.truth[mask] == 1).sum()),
            negative=int((s.truth[mask] == 2).sum()),
            unresolved=int((s.truth[mask] == 0).sum()),
            mean_size=float(s.size[mask].mean()),
            methods={n: s.ratio(s.cost(n), weight, mask & certifiable) for n in focus} if (mask & certifiable).sum() else {},
            weight_fraction_read=s.mean(weight / s.size, mask & certifiable) if (mask & certifiable).sum() else None,
        )

    for r, (num, den) in enumerate(s.meta["rhos"]):
        stratum(f"rho={num}/{den}", d["rho"] == r)
    stratum("family=top1", s.kind == 0)
    stratum("family=coalition", s.kind == 1)
    for lo, hi, label in ((0, 0.5, "margin<0.5"), (0.5, 1, "0.5-1"), (1, 3, "1-3"), (3, np.inf, ">=3")):
        stratum(f"positive margin {label}", (s.truth == 1) & (rel >= lo) & (rel < hi))
    for lo, hi, label in ((10, 50, "10-49"), (50, 150, "50-149"), (150, 500, "150-499"), (500, np.inf, ">=500")):
        stratum(f"inputs {label}", (s.size >= lo) & (s.size < hi))
    # Statistical certificate.
    stat_cols = [s.arms[f"stat_r{r}"] for r in range(5)]
    stat_status = d["status"][:, stat_cols]
    wrong = (stat_status != 0) & (stat_status != s.truth[:, None])
    risk_used = ~d["exact"][:, stat_cols]
    statistical = dict(
        alpha=s.meta["alpha"],
        audits=int(stat_status.size),
        wrong_certificates=int(wrong.sum()),
        wrong_rate=float(wrong.mean()),
        certificates_using_risk=int(risk_used.sum()),
        share_using_risk=float(risk_used.mean()),
        certified_on_unresolvable=int(((stat_status != 0) & (s.truth[:, None] == 0)).sum()),
        unresolvable_audits=int((s.truth == 0).sum() * 5),
        wrong_rate_among_unresolvable=float(((stat_status != 0) & (s.truth[:, None] == 0)).sum() / max(1, (s.truth == 0).sum() * 5)),
        mean_cost_ratio_to_weight_all=s.ratio(s.cost("stat"), weight, everything),
    )
    # Composition and conservation.
    composition = dict(
        sessions=s.n,
        target_types=s.n_clusters,
        targets=int(len(np.unique(s.bench["fam_target"][d["family"]]))),
        families=int(len(np.unique(d["family"]))),
        positive=int((s.truth == 1).sum()),
        negative=int((s.truth == 2).sum()),
        unresolved=int((s.truth == 0).sum()),
        mean_inputs=float(s.size.mean()),
        median_inputs=float(np.median(s.size)),
        unresolved_cell_fraction=float(d["unresolved_cells"].sum() / d["size"].sum()),
        unresolved_weight_fraction=float(d["unresolved"].sum() / d["total"].sum()),
        total_weight_order_observations=float(weight.sum()),
        total_budget=float(s.size.sum()),
    )
    conservation = {}
    for r, (num, den) in enumerate(s.meta["rhos"]):
        for k, label in ((0, "top1"), (1, "coalition")):
            m = (d["rho"] == r) & (s.kind == k)
            conservation[f"{label}@rho={num}/{den}"] = dict(
                questions=int(m.sum()),
                positive=float((s.truth[m] == 1).mean()),
                negative=float((s.truth[m] == 2).mean()),
                unresolved=float((s.truth[m] == 0).mean()),
            )
    # Where payments go: share of paid responses by kind in questions with a positive answer.
    mechanism = {}
    positive = s.truth == 1
    for prior in ["mlp_s0"] + STRONG[view]:
        for policy in ("member", "direction", "ag", "recal"):
            k = s.arms[f"{prior}:{policy}"]
            paid = float(d["cost"][positive, k].sum())
            mechanism[f"{prior}:{policy}"] = dict(
                member=float(d["paid_member"][positive, k].sum() / paid),
                unresolved=float(d["paid_unresolved"][positive, k].sum() / paid),
                nonmember=float(1 - (d["paid_member"][positive, k].sum() + d["paid_unresolved"][positive, k].sum()) / paid),
            )
    k = s.arms["weight"]
    paid = float(d["cost"][positive, k].sum())
    mechanism["weight"] = dict(
        member=float(d["paid_member"][positive, k].sum() / paid),
        unresolved=float(d["paid_unresolved"][positive, k].sum() / paid),
        nonmember=float(1 - (d["paid_member"][positive, k].sum() + d["paid_unresolved"][positive, k].sum()) / paid),
    )
    split_types = [t["code"] for t in s.manifest["types"] if t["split"] == split]
    return dict(
        view=view,
        split=split,
        result_sha256=s.meta["result_sha256"],
        composition=composition,
        methods=methods,
        contrasts=contrasts,
        strata=strata,
        statistical=statistical,
        conservation=conservation,
        mechanism=mechanism,
        calibration=calibration(s, split_types),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/v35/runs")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs/v35/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/analysis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {view: analyze(view, args.split, args.runs, args.benchmark) for view in ("malecns", "flywire")}
    (args.output / f"{args.split}.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for view, r in result.items():
        print("=====", view, json.dumps(r["composition"]))
        for name, m in r["methods"].items():
            if "all" in m:
                c = m["certifiable"]["ratio_to_weight"]
                print(f"  {name:30s} cert {c['value']:.3f} [{c['low']:.3f},{c['high']:.3f}]  all {m['all']['ratio_to_weight']['value']:.3f}  pos {m['positive']['ratio_to_weight']['value']:.3f}  neg {m['negative']['ratio_to_weight']['value']:.3f}")
            else:
                print(f"  {name:30s} cert {m['certifiable']['ratio_to_weight']['value']:.3f} seeds sd {m['seed_sd']:.4f} {['%.3f' % x for x in m['seed_ratios']]}")


if __name__ == "__main__":
    main()
