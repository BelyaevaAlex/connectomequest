#!/usr/bin/env python3
"""Cost of weight order, member-first, direction-fixed, AG and recalibrated AG with the seeded priors.

For every certifiable question of one split this runs the frozen engine with the priors built by
``scripts/r5_priors_build.py`` (``outputs/round5/priors/seeded*-<view>.npz``). Questions that cannot
be certified cost every input cell under every policy and are not run. Each arm is checked to reach
the exact full-information status and never to undercut the clairvoyant bound.

The held-out split refuses to run unless ``outputs/round5/record.json`` lists the sha256 of this
script and of every round-5 module it imports; results are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.neurosym import common as nc
from connectomequest.round5 import priors_common as pc

POLICIES = ("member", "direction", "ag", "recal")
DEFAULT_PRIORS = tuple(f"seeded{pc.fraction_tag(f)}{v}" for f in pc.FRACTIONS for v in ("", "conn"))
_D = {}


def probabilities(plus, unknown):
    plus = np.clip(plus.astype(float), 0.0, 1.0)
    unknown = np.clip(unknown.astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def load_priors(view, names, bench, directory=pc.OUT):
    out, files = {}, {}
    for name in names:
        path = Path(directory) / f"{name}-{view}.npz"
        with np.load(path, allow_pickle=False) as z:
            for ref in ("pre_cell", "fam_target", "target_offset"):
                if not np.array_equal(z[ref], bench[ref]):
                    raise ValueError(f"{path.name}: {ref} differs from the benchmark")
            plus, unknown = z["plus_" + name], z["unknown_" + name]
        if plus.shape != (int(bench["fam_plus_offset"][-1]),) or unknown.shape != bench["pre_cell"].shape:
            raise ValueError(f"{path.name}: layout differs from the benchmark")
        out[name] = (plus, unknown)
        files[name] = pc.hash_file(path)
    return out, files


def arm_names(priors):
    return ["weight"] + [f"{p}:{q}" for p in priors for q in POLICIES]


def run_family(family):
    d = _D
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    start = int(d["fam_plus_offset"][family])
    prob = {name: probabilities(plus[start : start + hi - lo], unknown[lo:hi]) for name, (plus, unknown) in d["priors"].items()}
    arms = d["arms"]
    rows = []
    for rho_index, (num, den) in enumerate(nc.RHOS):
        threshold, _ = e.threshold_goals(total, num * int(d["fam_source_weight"][family]), den * int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        if truth == "unresolved":
            continue
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(arms), dtype=np.int32)
        paid_member = np.zeros(len(arms), dtype=np.int32)
        paid_unresolved = np.zeros(len(arms), dtype=np.int32)

        def record(name, order):
            c, status = e.evaluate(inst, outcomes, order)
            if status != truth or c < bound:
                raise AssertionError("arm contradicts the exact full-information status")
            k = arms[name]
            cost[k] = c
            paid = outcomes[np.asarray(order[:c], dtype=np.intp)]
            paid_member[k] = int((paid == e.MEMBER).sum())
            paid_unresolved[k] = int((paid == e.UNRESOLVED).sum())

        record("weight", e.weight_order(inst))
        for name, p in prob.items():
            record(f"{name}:member", e.positive_order(inst, p))
            record(f"{name}:direction", e.direction_order(inst, p))
            record(f"{name}:ag", e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, p)))
            record(f"{name}:recal", e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p)))
        rows.append(dict(family=family, rho=rho_index, size=inst.size, truth=1 if truth == "positive" else 2, clairvoyant=bound,
                         cost=cost, paid_member=paid_member, paid_unresolved=paid_unresolved))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=pc.VIEWS)
    parser.add_argument("--split", required=True, choices=pc.SPLITS)
    parser.add_argument("--priors", nargs="+", default=list(DEFAULT_PRIORS))
    parser.add_argument("--prior-dir", type=Path, default=pc.OUT, help="directory of the prior files (default outputs/round5/priors)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=pc.OUT / "runs")
    parser.add_argument("--limit", type=int, help="first N families only (development smoke tests)")
    args = parser.parse_args()
    if args.split == "test" and args.limit is not None:
        raise ValueError("the held-out split runs completely")
    code = pc.guard(args.split, __file__) or pc.code_identity(__file__)
    pc.guard_data(args.split, [args.prior_dir / f"{name}-{args.view}.npz" for name in args.priors])
    out = args.output / f"{args.view}-{args.split}.npz"
    meta_path = args.output / f"{args.view}-{args.split}.json"
    if args.limit is not None:
        out, meta_path = out.with_name(f"{args.view}-{args.split}-limit{args.limit}.npz"), meta_path.with_name(f"{args.view}-{args.split}-limit{args.limit}.json")
    if out.exists() or meta_path.exists():
        raise FileExistsError(out)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "priors", f"{args.view}-{args.prior_dir.resolve().name}", out)
    manifest = nc.manifest()
    view = next(v for v in manifest["views"] if v["view"] == args.view)
    if pc.hash_file(nc.BENCH / f"{args.view}.npz") != view["sha256"]:
        raise ValueError("benchmark file differs from its manifest")
    bench = nc.benchmark(args.view)
    _D.update(bench)
    _D["priors"], prior_files = load_priors(args.view, args.priors, bench, args.prior_dir)
    names = arm_names(args.priors)
    _D["arms"] = {n: i for i, n in enumerate(names)}
    families = nc.split_families(args.view, args.split).tolist()
    if args.limit is not None:
        families = families[: args.limit]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=4)
    rows = [r for group in results for r in group]
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in ("family", "rho", "size", "truth", "clairvoyant")}
    arrays.update({k: np.stack([r[k] for r in rows]) for k in ("cost", "paid_member", "paid_unresolved")})
    pc.save_npz_new(out, **arrays)
    meta = dict(view=args.view, split=args.split, arms=names, policies=list(POLICIES), priors=list(args.priors), prior_sha256=prior_files,
                sessions=len(rows), families=len(families), code=code, benchmark_sha256=view["sha256"],
                result_sha256=pc.hash_file(out), wall_seconds=time.time() - started, limit=args.limit, workers=args.workers)
    pc.write_json_new(meta_path, meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "arms")}))


if __name__ == "__main__":
    main()
