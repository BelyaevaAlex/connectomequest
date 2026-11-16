#!/usr/bin/env python3
"""Round 5, workstream ``costs``: ratios for the batch and heterogeneous cost models.

Reads outputs/round5/costs/runs/{view}-{split}.npz (from r5_costs_run.py) and the frozen
per-cell costs of outputs/v35/runs. All ratios are ratios of totals over certifiable
questions with 95% percentile intervals from 2,000 resamples of cell types (seed 20261112,
as in scripts/v35_analyze.py).

A1 (batch cost): every batch arm relative to batch weight order, to per-cell weight order and
to the same rule with per-cell cost; contrasts between rules in both models; rankings.
A2 (heterogeneous cost): total cost of every arm relative to plain weight order under the
same cost model, and cost-aware / plain contrasts.

With ``--expectations FILE`` (see ``costs_common.check_expectations``), every listed quantity
is looked up in the result and the verdict is stored; on the held-out split the file must be
listed in the record too. The held-out split is refused
unless outputs/round5/record.json lists the sha256 of this script and of every round-5 module
it imports; outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from connectomequest.neurosym import common as nc
from connectomequest.round5 import costs_common as cc

CELL_POLICIES = ("member", "direction", "ag", "recal")
PRIORS = dict(malecns=("mlp_s0",), flywire=("mlp_s0", "provisional"))


def load(view, split, runs):
    meta = json.loads((runs / f"{view}-{split}.json").read_text())
    with np.load(runs / f"{view}-{split}.npz", allow_pickle=False) as z:
        mine = {k: z[k] for k in z.files}
    if cc.hash_file(runs / f"{view}-{split}.npz") != meta["result_sha256"]:
        raise ValueError("run file differs from its metadata")
    return meta, mine


def analyse(view, split, runs):
    meta, r = load(view, split, runs)
    fr = cc.frozen(view, split)
    order = r["frozen_row"]
    if not np.array_equal(fr["family"][order], r["family"]) or not np.array_equal(fr["truth"][order], r["truth"]):
        raise ValueError("run rows do not align with the frozen run")
    boot = cc.Bootstrap(cc.cluster_of_rows(view, r["family"]))
    cert = r["truth"] > 0
    farm = fr["arms"]

    def frozen_cost(name):
        if name == "clairvoyant":
            c = np.where(fr["clairvoyant"] < 0, fr["size"], fr["clairvoyant"])
        else:
            c = fr["cost"][:, farm[name]]
        return c[order].astype(float)

    barm = {n: i for i, n in enumerate(meta["batch_arms"])}
    bcost = {n: r["batch_cost"][:, i].astype(float) for n, i in barm.items()}
    cell_weight, batch_weight = frozen_cost("weight"), bcost["b:weight"]
    out = dict(view=view, split=split, sessions=int(len(cert)), certifiable=int(cert.sum()), types=int(boot.n))
    kind = nc.benchmark(view)["fam_kind"][r["family"]]
    strata = {
        "certifiable": cert,
        "inputs<500": cert & (r["size"] < 500),
        "inputs>=500": cert & (r["size"] >= 500),
        "top1": cert & (kind == 0),
        "coalition": cert & (kind == 1),
    }
    out["strata_sizes"] = {k: int(m.sum()) for k, m in strata.items()}
    out["large_share_of_cell_weight_checks"] = float(cell_weight[strata["inputs>=500"]].sum() / cell_weight[cert].sum())
    # ---------------------------------------------------------------- A1
    cell_names = ["clairvoyant"] + [f"{p}:{x}" for p in PRIORS[view] for x in CELL_POLICIES]

    def batch_block(mask):
        a1 = dict(
            composition=dict(
                units_per_cell_total=float(r["units"][mask].sum() / r["size"][mask].sum()),
                units_per_cell_mean=boot.mean(r["units"] / r["size"], mask),
                mixed_batch_share=float(r["mixed"][mask].sum() / r["units"][mask].sum()),
                batch_weight_over_cell_weight_median=float(np.median(batch_weight[mask] / cell_weight[mask])),
                batch_weight_fewer_share=float((batch_weight[mask] < cell_weight[mask]).mean()),
            ),
            cell_model={n: boot.ratio(frozen_cost(n), cell_weight, mask) for n in cell_names},
            batch_model={n: boot.ratio(bcost[n], batch_weight, mask) for n in barm if n != "b:weight"},
            batch_over_cell_weight={n: boot.ratio(bcost[n], cell_weight, mask) for n in barm},
            batch_over_same_cell_rule={},
            contrasts={},
        )
        for n in barm:
            same = n[2:] if n[2:] in ("weight", "clairvoyant") or n.split(":")[-1] in CELL_POLICIES else None
            if same is not None:
                a1["batch_over_same_cell_rule"][n] = boot.ratio(bcost[n], frozen_cost(same), mask)
        for p in PRIORS[view]:
            for x, y in (("ag", "direction"), ("recal", "ag"), ("recal", "direction"), ("ag", "member"), ("member", "direction")):
                a1["contrasts"][f"cell:{p}:{x}/{y}"] = boot.ratio(frozen_cost(f"{p}:{x}"), frozen_cost(f"{p}:{y}"), mask)
                a1["contrasts"][f"batch:{p}:{x}/{y}"] = boot.ratio(bcost[f"b:{p}:{x}"], bcost[f"b:{p}:{y}"], mask)
            a1["contrasts"][f"batch:{p}:ag-sum/ag"] = boot.ratio(bcost[f"b:{p}:ag-sum"], bcost[f"b:{p}:ag"], mask)
        a1["contrasts"]["batch:heaviest/weight"] = boot.ratio(bcost["b:heaviest"], batch_weight, mask)
        for p in PRIORS[view]:
            cell_rank = sorted(["weight"] + [f"{p}:{x}" for x in CELL_POLICIES], key=lambda n: 1.0 if n == "weight" else a1["cell_model"][n]["value"])
            batch_rank = sorted(["b:weight"] + [f"b:{p}:{x}" for x in CELL_POLICIES], key=lambda n: 1.0 if n == "b:weight" else a1["batch_model"][n]["value"])
            a1[f"ranking:{p}"] = dict(cell=cell_rank, batch=[n[2:] for n in batch_rank])
        return a1

    out["batch"] = batch_block(cert)
    out["batch_strata"] = {k: batch_block(m) for k, m in strata.items() if k != "certifiable" and m.any()}
    # ---------------------------------------------------------------- A2
    harm = {n: i for i, n in enumerate(meta["hetero_arms"])}

    def hetero_block(mask):
        a2 = {}
        for mi, model in enumerate(meta["cost_models"]):
            level = float(model.split("@")[1])
            cost = {a.split("|", 1)[1]: r["hetero_cost"][:, i] for a, i in harm.items() if a.startswith(model + "|")}
            count = {a.split("|", 1)[1]: r["hetero_count"][:, i].astype(float) for a, i in harm.items() if a.startswith(model + "|")}
            base = cost["weight"]
            flagged_checked = (cost["weight"] - count["weight"]) / (level - 1)
            m = dict(
                flagged_share_of_inputs=float(r["flagged"][mask, mi].sum() / r["size"][mask].sum()),
                flagged_share_of_weight_checks=float(flagged_checked[mask].sum() / count["weight"][mask].sum()),
                cost_over_count_weight=float(base[mask].sum() / count["weight"][mask].sum()),
                ratio_to_weight={a: boot.ratio(c, base, mask) for a, c in cost.items() if a != "weight"},
                count_ratio_to_weight={a: boot.ratio(c, count["weight"], mask) for a, c in count.items() if a not in ("weight", "clairvoyant/c")},
                aware_over_plain={},
            )
            m["aware_over_plain"]["weight"] = boot.ratio(cost["weight/c"], cost["weight"], mask)
            for p in PRIORS[view]:
                for x in ("ag", "recal"):
                    m["aware_over_plain"][f"{p}:{x}"] = boot.ratio(cost[f"{p}:{x}/c"], cost[f"{p}:{x}"], mask)
            m["ranking"] = sorted(["weight"] + [a for a in cost if a not in ("weight", "clairvoyant/c")], key=lambda a: 1.0 if a == "weight" else m["ratio_to_weight"][a]["value"])
            a2[model] = m
        return a2

    out["hetero"] = hetero_block(cert)
    out["hetero_strata"] = {"inputs<500": hetero_block(strata["inputs<500"])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=cc.SPLITS)
    parser.add_argument("--runs", default=str(cc.OUT / "runs"))
    parser.add_argument("--output", default=str(cc.OUT))
    parser.add_argument("--expectations", type=Path)
    args = parser.parse_args()
    cc.guard(args.split, __file__, extra=[args.expectations] if args.expectations else [])
    out = Path(args.output) / f"analysis-{args.split}.json"
    if out.exists():
        raise FileExistsError(out)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "costs", "analyze", out)
    result = {view: analyse(view, args.split, Path(args.runs)) for view in cc.VIEWS}
    if args.expectations is not None:
        result["expectations"] = cc.check_expectations(result, json.loads(args.expectations.read_text()))
        result["expectations_sha256"] = cc.hash_file(args.expectations)
    result["code"] = {p: cc.hash_file(cc.ROOT / p) for p in cc.imported_round5_files(__file__)}
    cc.write_json_new(out, result)
    fmt = lambda x: f"{x['value']:.3f} [{x['low']:.3f}, {x['high']:.3f}]"
    for view in cc.VIEWS:
        v = result[view]
        print(f"===== {view}: {v['certifiable']} certifiable of {v['sessions']} questions, {v['types']} types")
        b = v["batch"]
        print("  batches per cell", b["composition"]["units_per_cell_total"], "mixed", b["composition"]["mixed_batch_share"])
        for n, x in b["cell_model"].items():
            print(f"  cell  {n:24s} {fmt(x)}")
        for n, x in b["batch_model"].items():
            print(f"  batch {n:24s} {fmt(x)}   / cell weight {fmt(b['batch_over_cell_weight'][n])}")
        print(f"  batch b:weight / cell weight {fmt(b['batch_over_cell_weight']['b:weight'])}")
        for n, x in b["contrasts"].items():
            print(f"  contrast {n:34s} {fmt(x)}")
        for k in b:
            if k.startswith("ranking"):
                print(" ", k, b[k])
        print("  strata sizes", v["strata_sizes"], "large share of cell weight checks", round(v["large_share_of_cell_weight_checks"], 3))
        for sname, sb in v["batch_strata"].items():
            print(f"  [{sname}] b:weight/cell weight {fmt(sb['batch_over_cell_weight']['b:weight'])} median {sb['composition']['batch_weight_over_cell_weight_median']:.3f}")
            for n, x in sb["batch_model"].items():
                cn = n[2:]
                cell = sb["cell_model"].get(cn)
                print(f"     batch {n:24s} {fmt(x)}" + (f"   cell {fmt(cell)}" if cell else ""))
            for k in sb:
                if k.startswith("ranking"):
                    print("    ", k, sb[k])
        for model, m in v["hetero"].items():
            print(f"  -- {model}: flagged inputs {m['flagged_share_of_inputs']:.3f}, of weight checks {m['flagged_share_of_weight_checks']:.3f}")
            for a, x in m["ratio_to_weight"].items():
                print(f"     {a:24s} {fmt(x)}")
            for a, x in m["aware_over_plain"].items():
                print(f"     aware/plain {a:16s} {fmt(x)}")
            print("     ranking", m["ranking"])
    for item in result.get("expectations", []):
        print(item["name"], round(item["observed"], 4), item["op"], item["value"], "HOLDS" if item["holds"] else "FAILS")


if __name__ == "__main__":
    main()
