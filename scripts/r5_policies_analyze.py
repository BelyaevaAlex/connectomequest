#!/usr/bin/env python3
"""Round 5, workstream ``policies``: analysis of one split.

Reads ``outputs/round5/policies/{runs,dp,wor}`` and the frozen / stored arms, and writes
one JSON with
* ``runs``: every new arm relative to weight order and to the frozen reference arms of
  the same prior (AG, recalibrated AG, direction-fixed, stored propagation), certifiable
  questions, ratio of totals with a 95 % type-cluster bootstrap interval; for the safe
  interleaves also the ratio to the cheaper of base and weight order per question and
  the largest per-question ratio (which the theorem bounds by 2);
* ``dp``: on questions with at most ``n`` input cells, expected checks under the prior
  relative to the Bayes-optimal policy, and realized checks relative to weight order and
  to the optimum;
* ``wor``: cost of the without-replacement tests relative to weight order by number of
  input cells, against the frozen with-replacement tests; wrong-certificate rates on the
  questions and at adversarial thresholds (Clopper-Pearson 95 % intervals);
* ``voi_selection``: the pre-declared choice of one VOI variant (development only).

``--expectations`` (held-out only) evaluates the expected outcomes stored in the record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.round5 import policies_data as pdata  # noqa: E402

STRATA = ((10, 50, "10-49"), (50, 150, "50-149"), (150, 500, "150-499"), (500, np.inf, ">=500"))
REFERENCE = ("ag", "recal", "direction", "adir-recal")
VOI = ("voi-rollout", "voi-complete", "voi-rollout-recal", "voi-complete-recal")
VOI_RULE = ("smallest mean, over the development (view, prior) pairs mlp_s0/malecns, mlp_s0/flywire and provisional/flywire, "
            "of the ratio of total checks to weight order on certifiable questions")


def clopper(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return dict(rate=k / n if n else float("nan"), low=lo, high=hi, k=int(k), n=int(n))


def load(path):
    meta = Path(path).with_suffix(".json")
    if meta.exists():  # results computed with another version of a prior file are refused
        m = json.loads(meta.read_text())
        if "prior_identity" in m and m["prior_identity"] != pdata.prior_identity(m["view"], m["prior"]):
            raise SystemExit(f"{path}: computed with another version of the prior {m['prior']} ({m['view']})")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def frozen_rows(view, split, family, rho):
    fr = pdata.frozen(view, split)
    return None if fr is None else np.array([fr["key"][(int(f), int(r))] for f, r in zip(family, rho)])


def stored_aligned(view, split, arm, family, rho):
    cost = pdata.stored_cost(view, split, arm)
    if cost is None:
        return None
    if arm.endswith(("prop-recal", "prop-ag")):
        prop = pdata.frozen_propagation(view, split)
        key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(prop["family"], prop["rho"]))}
        return cost[[key[(int(f), int(r))] for f, r in zip(family, rho)]]
    return cost[frozen_rows(view, split, family, rho)]


def analyse_runs(split):
    out = {}
    for path in sorted((pdata.OUT / "runs").glob(f"*-{split}-*.npz")):
        view, _, prior = path.stem.split("-", 2)
        z = load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        cl = pdata.Clusters(view, z["family"])
        m = z["truth"] > 0
        w = z["cost__weight"]
        arms = {k[6:]: v for k, v in z.items() if k.startswith("cost__") and k != "cost__weight"}
        ref = {}
        for name in REFERENCE + ("prop-recal",):
            v = stored_aligned(view, split, f"{prior}:{name}", z["family"], z["rho"])
            if v is not None:
                ref[name] = v
        for name in ("recal", "prop-recal"):
            if name in arms and name not in ref:
                ref[name] = arms[name]  # recomputed base without a stored copy
        arms_all = dict(arms)
        if "prop-recal" in ref:
            arms_all.setdefault("prop-recal", ref["prop-recal"])
        res = dict(sessions=int(len(m)), certifiable=int(m.sum()), meta_checks=meta.get("checks"), arms={})
        for name, cost in sorted(arms_all.items()):
            row = dict(to_weight=cl.ratio(cost, w, m), stored=name not in arms)
            for r, v in ref.items():
                if r != name:
                    row[f"to_{r}"] = cl.ratio(cost, v, m)
            if name.endswith("+wo"):
                base = ref[name[:-3]]
                cheaper = np.minimum(base, w)
                row["to_cheaper"] = cl.ratio(cost, cheaper, m)
                row["max_per_question_over_cheaper"] = float((cost[m] / cheaper[m]).max())
                row["share_above_both"] = float((cost[m] > np.maximum(base, w)[m]).mean())
                row["to_base"] = cl.ratio(cost, base, m)
            row["better_tied_worse_vs_weight"] = [float((cost[m] < w[m]).mean()), float((cost[m] == w[m]).mean()), float((cost[m] > w[m]).mean())]
            res["arms"][name] = row
        res["reference_to_weight"] = {r: cl.ratio(v, w, m) for r, v in ref.items()}
        out[f"{view}:{prior}"] = res
    return out


def cross_prior(split):
    """Propagation with a prior versus propagation without prediction (``const_half``), same questions."""
    out = {}
    paths = {tuple(p.stem.split("-", 2)[::2]): p for p in (pdata.OUT / "runs").glob(f"*-{split}-*.npz")}
    for (view, prior), path in sorted(paths.items()):
        base = paths.get((view, "const_half"))
        if prior == "const_half" or base is None:
            continue
        z, b = load(path), load(base)
        if not (np.array_equal(z["family"], b["family"]) and np.array_equal(z["rho"], b["rho"])):
            raise AssertionError("runs are not aligned")
        cl = pdata.Clusters(view, z["family"])
        m = z["truth"] > 0
        x = z.get("cost__prop-recal")
        if x is None:
            x = stored_aligned(view, split, f"{prior}:prop-recal", z["family"], z["rho"])
        if x is None or "cost__prop-recal" not in b:
            continue
        out[f"{view}:{prior}"] = dict(prop_over_prop_const_half=cl.ratio(x, b["cost__prop-recal"], m))
    return out


def voi_selection(runs):
    pairs = [("malecns", "mlp_s0"), ("flywire", "mlp_s0"), ("flywire", "provisional")]
    scores = {}
    for name in VOI:
        vals = [runs[f"{v}:{p}"]["arms"][name]["to_weight"]["value"] for v, p in pairs if f"{v}:{p}" in runs and name in runs[f"{v}:{p}"]["arms"]]
        if len(vals) == len(pairs):
            scores[name] = float(np.mean(vals))
    if not scores:
        return None
    chosen = min(scores, key=lambda k: (scores[k], k))
    return dict(rule=VOI_RULE, scores=scores, chosen=chosen)


def analyse_dp(split):
    out = {}
    for path in sorted((pdata.OUT / "dp").glob(f"*-{split}-*.npz")):
        view, _, rest = path.stem.split("-", 2)
        prior, ncap = rest.rsplit("-n", 1)
        z = load(path)
        if not len(z["family"]):
            continue
        cl = pdata.Clusters(view, z["family"])
        names = [k[len("expected__") :] for k in z if k.startswith("expected__")]
        everything = np.ones(len(z["family"]), dtype=bool)
        m = z["truth"] > 0
        res = dict(max_cells=int(ncap), sessions=int(len(m)), certifiable=int(m.sum()), families=int(len(np.unique(z["family"]))),
                   types=int(cl.k), sizes={int(s): int((z["size"] == s).sum()) for s in np.unique(z["size"])},
                   expected_over_opt={}, expected_over_opt_certifiable={}, realized_to_weight={}, realized_to_opt={}, realized_opt_better_tied_worse={},
                   share_ag_optimal_in_expectation=float((np.abs(z["expected__ag"] - z["expected__opt"]) <= 1e-9 * np.maximum(1, z["expected__opt"])).mean()),
                   expected_opt_to_weight=cl.ratio(z["expected__opt"], z["expected__weight"], everything))
        for name in names:
            res["expected_over_opt"][name] = cl.ratio(z[f"expected__{name}"], z["expected__opt"], everything)
            res["expected_over_opt_certifiable"][name] = cl.ratio(z[f"expected__{name}"], z["expected__opt"], m)
            res["realized_to_weight"][name] = cl.ratio(z[f"cost__{name}"], z["cost__weight"], m)
            res["realized_to_opt"][name] = cl.ratio(z[f"cost__{name}"], z["cost__opt"], m)
        c = z["cost__opt"][m]
        wgt = z["cost__weight"][m]
        res["realized_opt_better_tied_worse"] = [float((c < wgt).mean()), float((c == wgt).mean()), float((c > wgt).mean())]
        res["largest_expected_gap"] = {name: float((z[f"expected__{name}"] / z["expected__opt"]).max()) for name in names}
        out[f"{view}:{prior}:n{ncap}"] = res
    return out


def analyse_wor(split):
    out = {}
    for path in sorted((pdata.OUT / "wor").glob(f"*-{split}.npz")):
        view = path.stem.split("-")[0]
        z = load(path)
        cl = pdata.Clusters(view, z["family"])
        m = z["truth"] > 0
        u = z["truth"] == 0
        w = z["weight"]
        size = z["size"]
        costs = {"wor": z["wor_cost"].mean(1), "worhyb": z["worhyb_cost"].mean(1)}
        fr = pdata.frozen(view, split)
        rows = frozen_rows(view, split, z["family"], z["rho"])
        if fr is not None and "stat_r0" in fr["arms"]:
            repeats = sorted(a for a in fr["arms"] if a.startswith("stat_r"))
            assert repeats == [f"stat_r{r}" for r in range(5)], repeats  # the five frozen with-replacement repeats
            costs["wr"] = np.mean([fr["cost"][rows, fr["arms"][a]] for a in repeats], 0)
        ph = pdata.frozen_posthoc(view, split)
        if ph is not None:
            costs["wrhyb"] = ph["hybrid_cost"][rows].mean(1)
        res = dict(sessions=int(len(m)), certifiable=int(m.sum()), overall={}, strata={}, errors={}, contrasts={})
        for name, c in costs.items():
            res["overall"][name] = cl.ratio(c, w, m)
        for lo, hi, label in STRATA:
            s = m & (size >= lo) & (size < hi)
            res["strata"][label] = dict(questions=int(s.sum()), **{name: cl.ratio(c, w, s) for name, c in costs.items()})
        res["contrasts"]["wor_over_wr"] = cl.ratio(costs["wor"], costs["wr"], m) if "wr" in costs else None
        res["contrasts"]["worhyb_over_wrhyb"] = cl.ratio(costs["worhyb"], costs["wrhyb"], m) if "wrhyb" in costs else None
        big = m & (size >= 500)
        res["contrasts"]["worhyb_over_wrhyb_ge500"] = cl.ratio(costs["worhyb"], costs["wrhyb"], big) if "wrhyb" in costs and big.any() else None
        for name in ("wor", "worhyb"):
            status, exact = z[f"{name}_status"], z[f"{name}_exact"]
            wrong = (status != 0) & (status != z["truth"][:, None])
            res["errors"][name] = dict(certifiable=clopper(int(wrong[m].sum()), int(wrong[m].size)),
                                       uncertifiable=clopper(int(wrong[u].sum()), int(wrong[u].size)),
                                       share_statistical=float((~exact).mean()), share_statistical_certifiable=float((~exact[m]).mean()))
        if "adv_family" in z:
            adv = {}
            for name in ("wor", "worhyb"):
                for kind in ("yes", "no"):
                    sel = (z["adv_test"] == name) & (z["adv_kind"] == kind)
                    n = int(z["adv_repeats"][sel].sum())
                    key = "wrong_yes" if kind == "yes" else "wrong_no"
                    adv[f"{name}:{kind}"] = dict(claim_error=clopper(int(z[f"adv_{key}"][sel].sum()), n),
                                                 any_error=clopper(int((z["adv_wrong_yes"][sel] + z["adv_wrong_no"][sel]).sum()), n),
                                                 statistical_share=float(z["adv_statistical"][sel].sum() / max(n, 1)), thresholds=int(sel.sum()))
            res["adversarial"] = adv
        out[view] = res
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default = pdata.OUT / f"analysis-{args.split}.json"
    out = pdata.default_output(args.output or default, default, args.split)
    pdata.require_record(__file__, args.split)  # the held-out analysis must be the recorded one
    pdata.fresh(out)
    if args.split == "test":  # the round-5 one-shot gate (after every import and the output check)
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "policies", "analyze", out)
    runs = analyse_runs(args.split)
    result = dict(split=args.split, runs=runs, cross_prior=cross_prior(args.split), dp=analyse_dp(args.split), wor=analyse_wor(args.split))
    if args.split == "development":
        result["voi_selection"] = voi_selection(runs)
    if args.split == "test":
        result["expectations"] = evaluate_expectations(recorded_expectations(), result)
    pdata.save_json(out, result)
    print(out)


def recorded_expectations():
    """The expected outcomes fixed in the record: the file named by ``record["expectations"]["policies"]``
    (its sha256 is checked by the held-out gate), else a list under ``record["policies"]["expectations"]``."""
    record = json.loads(pdata.RECORD.read_text())
    names = record.get("expectations", {}).get("policies")
    if isinstance(names, str):
        names = [names]
    if names:  # one path or a list of paths; every expectations file of this workstream counts
        return [x for name in names for x in json.loads((pdata.ROOT / name).read_text())["expectations"]]
    exp = record.get("policies", {}).get("expectations") if isinstance(record.get("policies"), dict) else None
    if not exp:
        raise SystemExit("the record lists no expected outcomes for policies")
    return exp


def lookup(result, path):
    node = result
    for part in path:
        node = node[part]
    return node


def evaluate_expectations(expectations, result):
    """Each expectation: ``{"id", "path": [...], "field": "value"|"low"|"high", "op": "<="|">="|"<"|">", "threshold"}``."""
    ops = {"<=": np.less_equal, ">=": np.greater_equal, "<": np.less, ">": np.greater}
    out = []
    for x in expectations:
        try:
            v = float(lookup(result, x["path"])[x.get("field", "value")] if x.get("field") else lookup(result, x["path"]))
            holds = bool(ops[x["op"]](v, x["threshold"]))
        except (KeyError, TypeError):
            v, holds = None, False
        out.append(dict(x, observed=v, holds=holds))
    return dict(items=out, held=sum(i["holds"] for i in out), of=len(out))


if __name__ == "__main__":
    main()
