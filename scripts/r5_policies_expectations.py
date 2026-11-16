#!/usr/bin/env python3
"""Round 5, workstream ``policies``: expected held-out outcomes, fixed from the development split.

Every expectation names a number of ``analysis-test.json`` (``path`` + ``field``), a comparison
and a threshold set by hand from the development analysis (the development value and interval
are stored next to it). The script refuses to write ``outputs/round5/policies/expectations.json``
unless every expectation can be looked up in ``analysis-development.json`` and holds there, so a
threshold can never be one that development already contradicts. ``r5_policies_record.py`` copies
the file into the record proposal; ``r5_policies_analyze.py --split test`` evaluates it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.round5 import policies_data as pdata  # noqa: E402

ENV = "cd <repo> && export PYTHONPATH=src:.v32-deps:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1"
CORE_PRIORS = {"malecns": ("mlp_s0", "const_half", "const_share"), "flywire": ("mlp_s0", "provisional", "const_half", "const_share")}
SEEDED = ("seeded10", "seeded10conn", "seeded30", "seeded30conn", "seeded50", "seeded50conn")


def commands():
    run = [f".venv/bin/python scripts/r5_policies_run.py --view {v} --split test --prior {p} --workers 16" for v, ps in CORE_PRIORS.items() for p in ps]
    dp = [f".venv/bin/python scripts/r5_policies_dp.py --view {v} --split test --prior {p} --max-cells 14 --workers 16" for v, ps in CORE_PRIORS.items() for p in ps]
    wor = [f".venv/bin/python scripts/r5_policies_wor.py --view {v} --split test --adversarial --repeats 10 --workers 16" for v in CORE_PRIORS]
    seeded = [f".venv/bin/python scripts/r5_policies_run.py --view {v} --split test --prior {p} --workers 16" for v in CORE_PRIORS for p in SEEDED]
    # the exact DP with the seeded priors on MaleCNS only (FlyWire: about 70 min per prior on 16 CPUs)
    seeded += [f".venv/bin/python scripts/r5_policies_dp.py --view malecns --split test --prior {p} --max-cells 14 --workers 16" for p in SEEDED]
    final = [".venv/bin/python scripts/r5_policies_analyze.py --split test", ".venv/bin/python scripts/r5_policies_report.py --split test"]
    return dict(environment=ENV, order="all commands of 'core' and 'seeded' in any order (they are independent), then 'final' once",
                core=run + dp + wor, seeded=seeded, final=final,
                note="run once; every runner refuses an existing output, so a repeated command fails instead of overwriting")


def E(id_, claim, path, op, threshold, field="value"):
    return dict(id=id_, claim=claim, path=list(path), field=field, op=op, threshold=threshold)


def arm(view, prior, name, ref="to_weight"):
    return ("runs", f"{view}:{prior}", "arms", name, ref)


def expectations():
    x = []
    # 1. active-learning baselines
    for v, t in (("malecns", 1.30), ("flywire", 1.30)):
        x.append(E(f"us-{v}", "uncertainty sampling costs at least 1.3x weight order (mlp_s0)", arm(v, "mlp_s0", "us"), ">=", t))
    x.append(E("wus-malecns", "weight-scaled uncertainty costs at least 1.05x weight order (mlp_s0)", arm("malecns", "mlp_s0", "wus"), ">=", 1.05))
    x.append(E("wus-flywire", "weight-scaled uncertainty does not beat weight order (mlp_s0)", arm("flywire", "mlp_s0", "wus"), ">=", 1.00))
    for v in ("malecns", "flywire"):
        x.append(E(f"wus-recal-{v}", "recalibrated weight-scaled uncertainty does not beat recalibrated AG (mlp_s0)", arm(v, "mlp_s0", "wus-recal", "to_recal"), ">=", 1.00))
    x.append(E("wus-provisional", "weight-scaled uncertainty costs at least 1.05x recalibrated AG (provisional)", arm("flywire", "provisional", "wus", "to_recal"), ">=", 1.05))
    # 2. DP optimality gap and VOI
    for v in ("malecns", "flywire"):
        for p in ("const_half", "const_share"):
            x.append(E(f"dp-const-{v}-{p}", "weight order is Bayes-optimal under a no-prediction prior on every DP question",
                       ("dp", f"{v}:{p}:n14", "largest_expected_gap", "weight"), "<=", 1 + 1e-9, field=None))
        x.append(E(f"dp-ag-{v}", "AG is within 0.5% of the Bayes optimum in expectation (mlp_s0)", ("dp", f"{v}:mlp_s0:n14", "expected_over_opt", "ag"), "<=", 1.005))
        x.append(E(f"dp-recal-{v}", "recalibrated AG is within 0.5% of the Bayes optimum in expectation (mlp_s0)", ("dp", f"{v}:mlp_s0:n14", "expected_over_opt", "recal"), "<=", 1.005))
        x.append(E(f"dp-weight-{v}", "weight order is within 1% of the Bayes optimum in expectation (mlp_s0)", ("dp", f"{v}:mlp_s0:n14", "expected_over_opt", "weight"), "<=", 1.01))
        x.append(E(f"dp-realized-{v}", "on the true answers the Bayes policy checks fewer cells than weight order (mlp_s0)", ("dp", f"{v}:mlp_s0:n14", "realized_to_weight", "opt"), "<", 1.00))
    x.append(E("dp-provisional-ag", "AG is within 3% of the Bayes optimum in expectation (provisional)", ("dp", "flywire:provisional:n14", "expected_over_opt", "ag"), "<=", 1.03))
    x.append(E("dp-provisional-weight", "weight order is at least 8% above the Bayes optimum in expectation (provisional)",
               ("dp", "flywire:provisional:n14", "expected_over_opt", "weight"), ">=", 1.08))
    x.append(E("dp-provisional-realized-opt", "on the true answers the Bayes policy checks at most 0.8x weight order (provisional)",
               ("dp", "flywire:provisional:n14", "realized_to_weight", "opt"), "<=", 0.80))
    x.append(E("dp-provisional-realized-ag", "on the true answers AG checks at least 1.1x the Bayes policy, although it is within 3% in expectation (provisional)",
               ("dp", "flywire:provisional:n14", "realized_to_opt", "ag"), ">=", 1.10))
    for v in ("malecns", "flywire"):
        x.append(E(f"voi-{v}", "the chosen VOI policy (voi-complete-recal) does not beat recalibrated AG (mlp_s0)", arm(v, "mlp_s0", "voi-complete-recal", "to_recal"), ">=", 1.00))
        x.append(E(f"voi-weight-{v}", "the chosen VOI policy costs at most 1.01x weight order (mlp_s0)", arm(v, "mlp_s0", "voi-complete-recal"), "<=", 1.01))
    x.append(E("voi-provisional", "the chosen VOI policy costs at least 1.05x recalibrated AG (provisional)", arm("flywire", "provisional", "voi-complete-recal", "to_recal"), ">=", 1.05))
    x.append(E("voi-rollout-const", "the rollout VOI policy costs more than weight order under the no-prediction prior", arm("malecns", "const_half", "voi-rollout"), ">=", 1.03))
    # 3. safe interleave
    for v, p in (("malecns", "mlp_s0"), ("flywire", "mlp_s0"), ("flywire", "provisional"), ("malecns", "const_half"), ("flywire", "const_half")):
        for a in ("recal+wo", "prop-recal+wo"):
            x.append(E(f"interleave-bound-{v}-{p}-{a}", "safe interleave never exceeds twice the cheaper of base and weight order on a question",
                       ("runs", f"{v}:{p}", "arms", a, "max_per_question_over_cheaper"), "<=", 2.0, field=None))
    for v in ("malecns", "flywire"):
        x.append(E(f"interleave-recal-{v}", "recalibrated AG + weight order costs at most weight order (mlp_s0)", arm(v, "mlp_s0", "recal+wo"), "<=", 1.00))
        x.append(E(f"interleave-recal-insurance-{v}", "the interleave costs at least its base: insurance, not savings (mlp_s0)", arm(v, "mlp_s0", "recal+wo", "to_base"), ">=", 1.00))
        x.append(E(f"interleave-prop-{v}", "propagation + weight order costs at most 0.92x weight order (mlp_s0)", arm(v, "mlp_s0", "prop-recal+wo"), "<=", 0.92))
        x.append(E(f"interleave-prop-insurance-{v}", "propagation + weight order costs at least 1.05x propagation alone (mlp_s0)", arm(v, "mlp_s0", "prop-recal+wo", "to_base"), ">=", 1.05))
    x.append(E("interleave-provisional", "recalibrated AG + weight order costs at most 0.85x weight order (provisional)", arm("flywire", "provisional", "recal+wo"), "<=", 0.85))
    x.append(E("interleave-provisional-insurance", "the interleave costs at least 1.05x its base (provisional)", arm("flywire", "provisional", "recal+wo", "to_base"), ">=", 1.05))
    # propagation without prediction (base of the second interleave)
    for v in ("malecns", "flywire"):
        x.append(E(f"prop-const-{v}", "propagation without any prediction costs at most 0.85x weight order", arm(v, "const_half", "prop-recal"), "<=", 0.85))
        x.append(E(f"prop-topology-{v}", "the topology prior saves at most 5% on top of propagation", ("cross_prior", f"{v}:mlp_s0", "prop_over_prop_const_half"), ">=", 0.95))
    # seeded in-connectome priors (command group "seeded")
    for v in ("malecns", "flywire"):
        x.append(E(f"prop-seeded30-{v}", "propagation with the 30% seeded prior costs at most 0.9x propagation without prediction",
                   ("cross_prior", f"{v}:seeded30", "prop_over_prop_const_half"), "<=", 0.90))
        x.append(E(f"prop-vs-recal-seeded30-{v}", "with the 30% seeded prior, propagation no longer beats recalibrated AG alone",
                   arm(v, "seeded30", "prop-recal", "to_recal"), ">=", 1.00))
        x.append(E(f"voi-seeded30-{v}", "with the 30% seeded prior the chosen VOI policy costs at least 1.1x recalibrated AG",
                   arm(v, "seeded30", "voi-complete-recal", "to_recal"), ">=", 1.10))
        for a in ("recal+wo", "prop-recal+wo"):
            x.append(E(f"interleave-bound-{v}-seeded30-{a}", "safe interleave never exceeds twice the cheaper of base and weight order on a question",
                       ("runs", f"{v}:seeded30", "arms", a, "max_per_question_over_cheaper"), "<=", 2.0, field=None))
    x.append(E("dp-seeded30-ag", "with the 30% seeded prior AG is at least 1% above the Bayes optimum in expectation (MaleCNS)",
               ("dp", "malecns:seeded30:n14", "expected_over_opt", "ag"), ">=", 1.01))
    x.append(E("dp-seeded30-ag-bound", "... and at most 8% above it", ("dp", "malecns:seeded30:n14", "expected_over_opt", "ag"), "<=", 1.08))
    x.append(E("dp-seeded30-weight", "with the 30% seeded prior weight order is at least 10% above the Bayes optimum in expectation (MaleCNS)",
               ("dp", "malecns:seeded30:n14", "expected_over_opt", "weight"), ">=", 1.10))
    x.append(E("dp-seeded30-realized", "with the 30% seeded prior AG checks at least 3% more cells than the Bayes policy on the true answers (MaleCNS)",
               ("dp", "malecns:seeded30:n14", "realized_to_opt", "ag"), ">=", 1.03))
    # 4. sampling tests without replacement
    for v in ("malecns", "flywire"):
        for t in ("wor", "worhyb"):
            x.append(E(f"{t}-error-{v}", "wrong certificates on certifiable audits below 0.5% (upper 95% bound)", ("wor", v, "errors", t, "certifiable"), "<=", 0.005, field="high"))
            x.append(E(f"{t}-error-uncertifiable-{v}", "wrong certificates on uncertifiable audits below 1% (upper 95% bound)", ("wor", v, "errors", t, "uncertifiable"), "<=", 0.01, field="high"))
            for c in ("yes", "no"):
                x.append(E(f"{t}-adversarial-{c}-{v}", f"wrong '{c}' claims at adversarial thresholds below alpha = 5% (upper 95% bound)",
                           ("wor", v, "adversarial", f"{t}:{c}", "claim_error"), "<=", 0.05, field="high"))
        x.append(E(f"wor-vs-wr-{v}", "sampling without replacement costs at most 0.98x sampling with replacement", ("wor", v, "contrasts", "wor_over_wr"), "<=", 0.98))
        x.append(E(f"worhyb-vs-wrhyb-{v}", "the WoR hybrid costs at most the with-replacement hybrid", ("wor", v, "contrasts", "worhyb_over_wrhyb"), "<=", 1.00))
        x.append(E(f"worhyb-overall-{v}", "the WoR hybrid costs at most weight order", ("wor", v, "overall", "worhyb"), "<=", 1.00))
        x.append(E(f"wor-large-{v}", "on questions with >= 500 input cells WoR costs at most 0.6x weight order", ("wor", v, "strata", ">=500", "wor"), "<=", 0.60))
        x.append(E(f"wor-small-{v}", "on questions with 10-49 input cells WoR costs at least 1.2x weight order", ("wor", v, "strata", "10-49", "wor"), ">=", 1.20))
    return x


def lookup(result, path, field):
    node = result
    for part in path:
        node = node[part]
    return node if field is None else node[field]


def main():
    dev = json.loads((pdata.OUT / "analysis-development.json").read_text())
    ops = {"<=": float.__le__, ">=": float.__ge__, "<": float.__lt__, ">": float.__gt__}
    items, bad = [], []
    for x in expectations():
        try:
            value = float(lookup(dev, x["path"], x["field"]))
        except (KeyError, TypeError) as err:
            bad.append(f"{x['id']}: not in the development analysis ({err!r})")
            continue
        node = lookup(dev, x["path"], None)
        interval = [node["low"], node["high"]] if isinstance(node, dict) and "low" in node else None
        if x["threshold"] is None:
            bad.append(f"{x['id']}: threshold not set (development value {value:.4f}, interval {interval})")
            continue
        holds = ops[x["op"]](value, float(x["threshold"]))
        print(f"{'ok ' if holds else 'NO '} {x['id']:44s} dev {value:.4f} {interval and [round(i, 4) for i in interval]} {x['op']} {x['threshold']}")
        if not holds:
            bad.append(f"{x['id']}: fails on development ({value} {x['op']} {x['threshold']})")
        items.append(dict(x, development=dict(value=value, interval=interval)))
    if bad:
        raise SystemExit("not written:\n  " + "\n  ".join(bad))
    out = pdata.save_json(pdata.OUT / "expectations.json", dict(
        schema="round5-policies-expectations-1", commands=commands(), expectations=items,
        rule="thresholds set by hand from analysis-development.json; each holds on development; evaluated once on analysis-test.json"))
    print(out, len(items), "expectations")


if __name__ == "__main__":
    main()
