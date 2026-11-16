#!/usr/bin/env python3
"""Round 5, workstream ``policies``: optimality gap on small questions by exact DP.

For every question of one split with at most ``--max-cells`` input cells and one prior:

* the Bayes-optimal adaptive policy under the independent prior (``policies_dp``), its
  expected number of checks ``opt`` and its realized cost on the true answers;
* the exact expected checks under the same prior of weight order, Adaptive Greedy,
  weight-scaled uncertainty sampling and the static one-step VOI rules (state-based
  policy evaluation), and of recalibrated AG (policy-tree enumeration; history-dependent);
* the realized costs of the same policies on the true answers (asserted equal to the
  frozen arms where those exist).

Work is split into shards of families written atomically under ``<output>/shards``;
a restarted job reuses finished shards whose code identity matches, so a preempted
job never redoes them. The merged file is never overwritten. The held-out split is
refused unless ``outputs/round5/record.json`` lists the sha256 of this runner and of
every round-5 module it imports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.annotation_v35 import engine as e  # noqa: E402
from connectomequest.round5 import policies_baselines as bl  # noqa: E402
from connectomequest.round5 import policies_data as pdata  # noqa: E402
from connectomequest.round5 import policies_dp as dp  # noqa: E402
from connectomequest.round5 import policies_voi as voi  # noqa: E402

STATIC = ("weight", "ag", "wus", "voi-complete", "voi-rollout")
TREE = ("recal",)
POLICIES = ("opt",) + STATIC + TREE
SHARD = 8
_RUN = {}


def static_policy(name, inst, p):
    if name == "weight":
        order = e.weight_order(inst)
        return lambda av, a, b: next(int(j) for j in order if av[j])
    if name == "wus":
        order = bl.weighted_uncertainty_order(inst, p)
        return lambda av, a, b: next(int(j) for j in order if av[j])
    pol = {"ag": e.AdaptiveGreedy, "voi-complete": voi.Complete, "voi-rollout": voi.Rollout}[name](inst, p)
    return lambda av, a, b: pol.choose(a, b, av)


def realized(name, inst, y, p, sol):
    if name == "opt":
        return sol.order(y)
    if name == "weight":
        return e.weight_order(inst)
    if name == "wus":
        return bl.weighted_uncertainty_order(inst, p)
    cls = {"ag": e.AdaptiveGreedy, "recal": e.RecalibratedGreedy, "voi-complete": voi.Complete, "voi-rollout": voi.Rollout}[name]
    return e.run_adaptive(inst, y, cls(inst, p))


def session_job(item):
    f, rho = item
    view, prior_name = _RUN["view"], _RUN["prior"]
    fam = pdata.family(view, f)
    p = pdata.prior(view, fam, prior_name)
    y = fam["outcomes"]
    threshold = dict(pdata.thresholds(fam))[rho]
    inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
    truth = e.full_information_status(inst, y)
    started = time.time()
    sol = dp.Solution(inst, p)
    expected = {"opt": sol.value}
    states = {"opt": sol.states}
    for name in STATIC:
        expected[name], states[name] = dp.evaluate_stateful(inst, p, static_policy(name, inst, p))
    expected["recal"], states["recal"] = dp.evaluate_tree(inst, p, e.RecalibratedGreedy(inst, p))
    cost = {}
    bound = e.clairvoyant_cost(inst, y)
    for name in POLICIES:
        c, s = e.evaluate(inst, y, realized(name, inst, y, p, sol))
        if s != truth or (bound is not None and c < bound):
            raise AssertionError(f"{name} contradicts the exact status")
        cost[name] = c
    for name in POLICIES:
        if expected[name] < expected["opt"] - 1e-9:
            raise AssertionError(f"{name} beats the optimum in expectation")
    return dict(family=int(f), rho=int(rho), size=inst.size, truth=pdata.STATUS[truth], expected=expected, cost=cost, states=states,
                resolves=sol.resolves, seconds=time.time() - started)


def sessions(view, split, max_cells, limit=None):
    families = pdata.split_families(view, split)
    if limit is not None:
        families = families[:limit]
    d = pdata.benchmark(view)
    sizes = np.diff(d["target_offset"])[d["fam_target"][families]]
    return [(int(f), rho) for f, n in zip(families, sizes) if n <= max_cells for rho in range(len(pdata.RHOS))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--prior", required=True)
    parser.add_argument("--max-cells", type=int, default=14)
    parser.add_argument("--output", type=Path, default=pdata.OUT / "dp")
    parser.add_argument("--workers", type=int, default=pdata.workers())
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.split == "test" and args.limit is not None:
        raise SystemExit("the held-out split runs completely")
    if args.prior not in pdata.available_priors(args.view):
        raise SystemExit(f"prior {args.prior} not available for {args.view}")
    output = pdata.default_output(args.output, pdata.OUT / "dp", args.split)
    pdata.preload(args.view)
    pdata.benchmark(args.view)
    code = pdata.require_record(__file__, args.split) or pdata.code_identity(__file__)
    stem = f"{args.view}-{args.split}-{args.prior}-n{args.max_cells}"
    out = pdata.fresh(output / f"{stem}.npz")
    pdata.fresh(out.with_suffix(".json"))
    if args.split == "test":  # the round-5 one-shot gate (after every import and the output check)
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "policies", f"dp-{args.view}-{args.prior}", out)
    shard_dir = output / "shards" / stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    # a shard is reused only with the same code, questions and prior file
    pin = pdata.pin_prior(args.view, args.prior)
    identity = ":".join([pdata.hash_file(Path(__file__)), *sorted(code.values()), pdata.hash_file(pdata.benchmark_path(args.view)),
                         json.dumps(dict(pin), sort_keys=True)])
    items = sessions(args.view, args.split, args.max_cells, args.limit)
    _RUN.update(view=args.view, prior=args.prior)
    chunks = [items[i : i + SHARD] for i in range(0, len(items), SHARD)]
    todo = []
    for i, chunk in enumerate(chunks):
        path = shard_dir / f"{i:05d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity or [tuple(x) for x in saved["items"]] != chunk:
                raise SystemExit(f"stale shard {path}: delete the shard directory of an older code version")
            continue
        todo.append((i, chunk))
    started = time.time()
    print(f"{stem}: {len(items)} sessions, {len(chunks)} shards, {len(todo)} to do, {args.workers} workers", flush=True)
    # longest first: large questions dominate the wall time
    sizes = {f: int(np.diff(pdata.benchmark(args.view)["target_offset"])[pdata.benchmark(args.view)["fam_target"][f]]) for f, _ in items}
    todo.sort(key=lambda t: -max(sizes[f] for f, _ in t[1]))
    with get_context("fork").Pool(args.workers) as pool:
        flat = [item for _, chunk in todo for item in chunk]
        where = {item: (i, chunk) for i, chunk in todo for item in chunk}
        pending = {i: {} for i, _ in todo}
        done = 0
        for result in pool.imap_unordered(session_job, flat, chunksize=1):
            key = (result["family"], result["rho"])
            i, chunk = where[key]
            pending[i][key] = result
            if len(pending[i]) == len(chunk):
                path = shard_dir / f"{i:05d}.json"
                tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(dict(identity=identity, items=chunk, rows=[pending[i][k] for k in chunk])))
                tmp.replace(path)
                del pending[i]
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} shards, {time.time() - started:.0f}s", flush=True)
    rows = []
    for i, chunk in enumerate(chunks):
        rows += json.loads((shard_dir / f"{i:05d}.json").read_text())["rows"]
    arrays = {k: np.asarray([r[k] for r in rows]) for k in ("family", "rho", "size", "truth", "resolves", "seconds")}
    for name in POLICIES:
        arrays[f"expected__{name}"] = np.asarray([r["expected"][name] for r in rows], dtype=float)
        arrays[f"cost__{name}"] = np.asarray([r["cost"][name] for r in rows], dtype=np.int64)
        arrays[f"states__{name}"] = np.asarray([r["states"][name] for r in rows], dtype=np.int64)
    checks = {}
    fr = pdata.frozen(args.view, args.split)
    if fr is not None and len(rows):
        index = np.array([fr["key"][(int(f), int(r))] for f, r in zip(arrays["family"], arrays["rho"])])
        for name in ("weight", "ag", "recal"):
            arm = "weight" if name == "weight" else f"{args.prior}:{name}"
            stored = pdata.stored_cost(args.view, args.split, arm)
            if stored is not None:
                same = bool(np.array_equal(stored[index], arrays[f"cost__{name}"]))
                checks[f"{name}_equals_stored"] = same
                if not same:
                    raise AssertionError(f"realized {name} differs from the stored arm")
    prior_ident = pin.check()
    pdata.save_npz(out, **arrays)
    meta = dict(view=args.view, split=args.split, prior=args.prior, prior_identity=prior_ident, max_cells=args.max_cells,
                policies=list(POLICIES), sessions=len(rows), limit=args.limit, checks=checks, code=code,
                benchmark_sha256=pdata.hash_file(pdata.benchmark_path(args.view)), result_sha256=pdata.hash_file(out),
                wall_seconds=round(time.time() - started, 1), cpu_seconds=float(arrays["seconds"].sum()) if len(rows) else 0.0, workers=args.workers)
    pdata.save_json(out.with_suffix(".json"), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "code"}), flush=True)


if __name__ == "__main__":
    main()
