#!/usr/bin/env python3
"""Round 5, workstream ``costs``: batch and heterogeneous cost models on one split.

For every question (target, partner set, retention level) of the split this runner

A1  checks *batches* (all input cells of one release type; one unit each) with weight order
    (release types by total weight), heaviest-cell order, member-first, direction-fixed,
    Adaptive Greedy (batch gain, and the sum of per-cell gains), recalibrated AG and the
    clairvoyant bound, for the topology prior and (FlyWire) the earlier-release prior;
A2  re-runs the pre-specified per-cell orders (weight, member-first, direction-fixed, AG,
    recalibrated AG) only to obtain the cells they pay - their number of checks must equal the
    frozen cost of outputs/v35/runs question by question - and prices them under
    heterogeneous costs; then runs the cost-aware variants (weight per cost, AG and
    recalibrated AG with gain per cost) and the exact clairvoyant cost.

The cells paid by the re-run per-cell arms are cached in outputs/round5/cache (reused when
present). The held-out split is refused unless outputs/round5/record.json lists the sha256
of this script and of every round-5 module it imports; outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as nc
from connectomequest.round5 import costs_batch as cb
from connectomequest.round5 import costs_common as cc
from connectomequest.round5 import costs_hetero as ch

PRIORS = dict(malecns=("mlp_s0",), flywire=("mlp_s0", "provisional"))
CELL_POLICIES = ("member", "direction", "ag", "recal")
BATCH_POLICIES = ("member", "direction", "ag", "ag-sum", "recal")
LEVELS = (2, 3)
PROXIES = dict(malecns=("small",), flywire=("small", "unlabelled"))
SMALL_QUANTILE = 0.25
WORKERS = int(os.environ.get("R5_COSTS_WORKERS", "6"))
_D = {}


def batch_arms(view):
    names = ["b:weight", "b:heaviest", "b:clairvoyant"]
    for prior in PRIORS[view]:
        names += [f"b:{prior}:{p}" for p in BATCH_POLICIES]
    return names


def cell_arms(view):
    return ["weight"] + [f"{prior}:{p}" for prior in PRIORS[view] for p in CELL_POLICIES]


def cost_models(view):
    return [f"{proxy}@{c}" for proxy in PROXIES[view] for c in LEVELS]


def aware_arms(view):
    return ["weight/c", "clairvoyant/c"] + [f"{prior}:{p}/c" for prior in PRIORS[view] for p in ("ag", "recal")]


def hetero_arms(view):
    return [f"{m}|{a}" for m in cost_models(view) for a in cell_arms(view) + aware_arms(view)]


def small_cut(view):
    """25th percentile of total synapse count over distinct input cells of development targets."""
    d = nc.benchmark(view)
    targets = np.unique(d["fam_target"][nc.development_families(view)])
    index = np.concatenate([np.arange(d["target_offset"][t], d["target_offset"][t + 1]) for t in targets])
    rows = np.unique(cc.input_rows(view)[index])
    return float(np.quantile(cc.node_strength(view)[rows], SMALL_QUANTILE))


def difficulty_flags(view):
    """Per benchmark input cell: dict proxy -> bool flag (known before any check)."""
    flags = dict(small=cc.node_strength(view)[cc.input_rows(view)] < small_cut(view))
    if view == "flywire":
        table = pq.read_table(nc.VIEW_ROOTS["flywire"] / "public/nodes.parquet", columns=["cell_id", "provisional"]).to_pandas()
        known = dict(zip(table["cell_id"].tolist(), table["provisional"].isna().tolist()))
        flags["unlabelled"] = np.array([known[c] for c in nc.benchmark(view)["pre_cell"].tolist()], dtype=bool)
    return flags


def run_family(f):
    d, view = _D["bench"], _D["view"]
    fam = nc.family(view, f)
    t = fam["target"]
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    cells, weights, outcomes = fam["cells"], fam["weights"], fam["outcomes"]
    keys = cb.batch_keys(_D["release"][lo:hi])
    flags = {k: v[lo:hi] for k, v in _D["flags"].items()}
    priors = {p: fam["priors"][p] for p in PRIORS[view]}
    barms, harms = _D["batch_arms"], _D["hetero_arms"]
    carms = cell_arms(view)
    models = cost_models(view)
    cached = _D.get("paid", {}).get(int(f))
    rows = []
    for rho, threshold in nc.thresholds(fam):
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        frozen_row = _D["frozen_row"][(int(f), rho)]
        # ---------------- A1: batches
        bq = cb.BatchQuestion(cells, weights, outcomes, threshold, keys)
        if bq.status() != truth:
            raise AssertionError("batches change the full-information status")
        bbound = bq.clairvoyant()
        bcost = np.zeros(len(barms), dtype=np.int32)

        def record_batch(name, order):
            c, s = bq.evaluate(order)
            if s != truth or (truth == "unresolved" and c != bq.units) or (bbound is not None and c < bbound):
                raise AssertionError(f"batch arm {name} contradicts the exact status")
            bcost[barms[name]] = c

        record_batch("b:weight", cb.weight_order(bq))
        record_batch("b:heaviest", cb.heaviest_cell_order(bq))
        bcost[barms["b:clairvoyant"]] = bq.units if bbound is None else bbound
        for prior, p in priors.items():
            record_batch(f"b:{prior}:member", cb.member_order(bq, p))
            record_batch(f"b:{prior}:direction", cb.direction_order(bq, p))
            record_batch(f"b:{prior}:ag", cb.run_greedy(bq, p))
            record_batch(f"b:{prior}:ag-sum", cb.run_sum_greedy(bq, p))
            record_batch(f"b:{prior}:recal", cb.run_recalibrated(bq, p))
        # ---------------- A2: per-cell orders of the frozen arms (paid cells only)
        paid = {}
        for name in carms:
            if cached is not None:
                order = cached[(rho, name)]
            else:
                if name == "weight":
                    order = e.weight_order(inst)
                else:
                    prior, policy = name.split(":")
                    p = priors[prior]
                    order = dict(
                        member=lambda: e.positive_order(inst, p),
                        direction=lambda: e.direction_order(inst, p),
                        ag=lambda: e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, p)),
                        recal=lambda: e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p)),
                    )[policy]()
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or c != int(_D["frozen_cost"][frozen_row, _D["frozen_arms"][name]]):
                raise AssertionError(f"re-run of {name} differs from its frozen cost")
            paid[name] = np.asarray(order[:c], dtype=np.int32)
        hcost = np.zeros(len(harms))
        hcount = np.zeros(len(harms), dtype=np.int32)
        budget = np.zeros(len(models))
        for mi, model in enumerate(models):
            proxy, level = model.split("@")
            cost = np.where(flags[proxy], float(level), 1.0)
            budget[mi] = cost.sum()
            for name in carms:
                k = harms[f"{model}|{name}"]
                hcount[k] = len(paid[name])
                hcost[k] = ch.paid_cost(paid[name], cost, len(paid[name]))
            aware = {"weight/c": ch.weight_per_cost_order(inst, cost)}
            for prior, p in priors.items():
                aware[f"{prior}:ag/c"] = ch.run_cost_aware(inst, outcomes, p, cost)
                aware[f"{prior}:recal/c"] = ch.run_cost_aware(inst, outcomes, p, cost, recalibrated=True)
            for name, order in aware.items():
                c, s = e.evaluate(inst, outcomes, order)
                if s != truth or (bound is not None and c < bound):
                    raise AssertionError(f"cost-aware arm {name} contradicts the exact status")
                k = harms[f"{model}|{name}"]
                hcount[k], hcost[k] = c, ch.paid_cost(order, cost, c)
            k = harms[f"{model}|clairvoyant/c"]
            best = ch.clairvoyant_cost(inst, outcomes, cost)
            hcost[k] = budget[mi] if best is None else best
            hcount[k] = -1
            if best is not None and best > min(hcost[harms[f"{model}|{a}"]] for a in carms + list(aware)):
                raise AssertionError("clairvoyant cost above a policy")
        rows.append(
            dict(
                family=int(f), rho=rho, frozen_row=frozen_row, truth=dict(unresolved=0, positive=1, negative=2)[truth], size=inst.size,
                units=bq.units, mixed=int((bq.outcomes == cb.MIXED).sum()), batch_cost=bcost, hetero_cost=hcost, hetero_count=hcount,
                budget=budget, flagged=np.array([int(flags[m.split("@")[0]].sum()) for m in models]),
                paid={name: paid[name] for name in carms},
            )
        )
    return rows


def paid_cache_name(view, split):
    return f"costs-paid-{view}-{split}.npz"


def load_paid_cache(view, split, arms):
    path = cc.CACHE / paid_cache_name(view, split)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        fam, rho = z["family"], z["rho"]
        out = {}
        for name in arms:
            off, idx = z[f"{name}__offset"], z[f"{name}__cells"]
            for r in range(len(fam)):
                out.setdefault(int(fam[r]), {})[(int(rho[r]), name)] = idx[off[r] : off[r + 1]]
    return out


def save_paid_cache(view, split, rows, arms):
    def build():
        arrays = dict(family=np.array([r["family"] for r in rows]), rho=np.array([r["rho"] for r in rows]))
        for name in arms:
            parts = [r["paid"][name] for r in rows]
            arrays[f"{name}__offset"] = np.concatenate(([0], np.cumsum([len(p) for p in parts]))).astype(np.int64)
            arrays[f"{name}__cells"] = np.concatenate(parts).astype(np.int32)
        return arrays

    cc.cached_npz(paid_cache_name(view, split), build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=cc.VIEWS)
    parser.add_argument("--split", required=True, choices=cc.SPLITS)
    parser.add_argument("--limit", type=int, help="development smoke runs only")
    parser.add_argument("--output", default=str(cc.OUT / "runs"))
    args = parser.parse_args()
    cc.guard(args.split, __file__)
    if args.split == "test" and args.limit is not None:
        raise ValueError("the held-out split runs completely")
    out = Path(args.output) / f"{args.view}-{args.split}{'' if args.limit is None else f'-limit{args.limit}'}.npz"
    meta_path = out.with_suffix(".json")
    if out.exists() or meta_path.exists():
        raise FileExistsError(out)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "costs", args.view, out)
    view = args.view
    frozen = cc.frozen(view, args.split)
    _D.update(
        view=view,
        bench=nc.benchmark(view),
        release=cc.input_release(view),
        flags=difficulty_flags(view),
        batch_arms={n: i for i, n in enumerate(batch_arms(view))},
        hetero_arms={n: i for i, n in enumerate(hetero_arms(view))},
        frozen_cost=frozen["cost"],
        frozen_arms=frozen["arms"],
        frozen_row={(int(f), int(r)): i for i, (f, r) in enumerate(zip(frozen["family"], frozen["rho"]))},
    )
    families = cc.split_families(view, args.split)
    if args.limit is not None:
        families = families[: args.limit]
    if args.limit is None:
        paid = load_paid_cache(view, args.split, cell_arms(view))
        if paid is not None:
            _D["paid"] = paid
    started = time.time()
    with Pool(WORKERS) as pool:
        rows = [r for group in pool.map(run_family, families.tolist(), chunksize=8) for r in group]
    if args.limit is None and "paid" not in _D:
        save_paid_cache(view, args.split, rows, cell_arms(view))
    if args.limit is None and len(rows) != len(frozen["family"]):
        raise AssertionError("the split does not cover the frozen run")
    scalars = ("family", "rho", "frozen_row", "truth", "size", "units", "mixed")
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in scalars}
    for k in ("batch_cost", "hetero_cost", "hetero_count", "budget", "flagged"):
        arrays[k] = np.stack([r[k] for r in rows])
    cc.save_npz_new(out, **arrays)
    b = _D["bench"]
    inputs = np.concatenate([np.arange(b["target_offset"][t], b["target_offset"][t + 1]) for t in np.unique(b["fam_target"][families])])
    meta = dict(
        view=view, split=args.split, limit=args.limit, sessions=len(rows), families=len(families), batch_arms=batch_arms(view),
        hetero_arms=hetero_arms(view), cost_models=cost_models(view), cell_arms=cell_arms(view), small_cut=small_cut(view),
        small_quantile=SMALL_QUANTILE, flagged_share_of_input_cells={k: float(v[inputs].mean()) for k, v in _D["flags"].items()},
        workers=WORKERS, wall_seconds=time.time() - started, code={p: cc.hash_file(cc.ROOT / p) for p in cc.imported_round5_files(__file__)},
        result_sha256=cc.hash_file(out),
    )
    cc.write_json_new(meta_path, meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("hetero_arms", "code")}, indent=1))


if __name__ == "__main__":
    main()
