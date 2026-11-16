#!/usr/bin/env python3
"""Run every V35 arm on one split of the conserved-partner benchmark.

One session is one question: (target neuron, partner category set, rho). The
question asks whether the set supplies at least ``rho`` times its hemibrain share
of the target's included input. Sessions are independent and start with no paid
facts. Every arm pays cells in its own order until the exact verifier certifies or
every cell is paid; an uncertified session therefore costs its full budget.

The test split refuses to run without a freeze record whose hashes match the
benchmark and the code, and refuses to overwrite an existing result.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35.statistical import sequential_certificate

ROOT = Path(__file__).resolve().parents[1]
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
FULL_POLICIES = ("member", "nonmember", "confident", "direction", "ggn", "ag", "adg", "recal", "adir", "adir-recal")
SEED_POLICIES = ("member", "nonmember", "direction", "ag")
RANDOM_REPEATS = 5
STAT_REPEATS = 5
ALPHA = 0.05
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
CODE_FILES = (
    "src/connectomequest/annotation_v35/engine.py",
    "src/connectomequest/annotation_v35/statistical.py",
    "scripts/v35_run.py",
)
_DATA = {}


def arm_names(priors):
    names = ["weight"] + [f"random_r{r}" for r in range(RANDOM_REPEATS)]
    for prior in priors:
        seeded = prior.startswith("mlp_s") and prior != "mlp_s0"
        names += [f"{prior}:{policy}" for policy in (SEED_POLICIES if seeded else FULL_POLICIES)]
    return names + [f"stat_r{r}" for r in range(STAT_REPEATS)]


def code_identity():
    return {name: hash_file(ROOT / name) for name in CODE_FILES}


def probabilities(plus, unknown):
    plus = np.clip(plus.astype(float), 0.0, 1.0)
    unknown = np.clip(unknown.astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def run_family(family):
    d = _DATA
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    member = int(weights[outcomes == e.MEMBER].sum())
    unresolved = int(weights[outcomes == e.UNRESOLVED].sum())
    priors, arms = d["priors"], d["arms"]
    prob = {p: probabilities(d["plus_" + p][d["fam_plus_offset"][family] : d["fam_plus_offset"][family] + hi - lo], d["unknown_" + p][lo:hi]) for p in priors}
    rows = []
    for rho_index, (num, den) in enumerate(RHOS):
        threshold, _ = e.threshold_goals(
            total, num * int(d["fam_source_weight"][family]), den * int(d["fam_source_total"][family])
        )
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(arms), dtype=np.int32)
        status = np.zeros(len(arms), dtype=np.int8)
        exact = np.ones(len(arms), dtype=bool)
        paid_member = np.zeros(len(arms), dtype=np.int32)
        paid_unresolved = np.zeros(len(arms), dtype=np.int32)

        def record(name, order):
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or (truth == "unresolved" and c != inst.size) or (bound is not None and c < bound):
                raise AssertionError("arm contradicts the exact full-information status")
            k = arms[name]
            cost[k], status[k] = c, STATUS[s]
            paid = outcomes[np.asarray(order[:c], dtype=np.intp)]
            paid_member[k] = int((paid == e.MEMBER).sum())
            paid_unresolved[k] = int((paid == e.UNRESOLVED).sum())

        record("weight", e.weight_order(inst))
        for r in range(RANDOM_REPEATS):
            record(f"random_r{r}", e.random_order(inst, np.random.default_rng([11, family, rho_index, r])))
        for prior in priors:
            p = prob[prior]
            seeded = prior.startswith("mlp_s") and prior != "mlp_s0"
            record(f"{prior}:member", e.positive_order(inst, p))
            record(f"{prior}:nonmember", e.negative_order(inst, p))
            record(f"{prior}:direction", e.direction_order(inst, p))
            record(f"{prior}:ag", e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, p)))
            if not seeded:
                record(f"{prior}:confident", e.confident_order(inst, p))
                record(f"{prior}:ggn", e.ggn_static_order(inst, p))
                record(f"{prior}:adg", e.run_adaptive(inst, outcomes, e.AdaptiveDualGreedy(inst, p)))
                record(f"{prior}:recal", e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p)))
                record(f"{prior}:adir", e.run_adaptive(inst, outcomes, e.AdaptiveDirection(inst, p)))
                record(f"{prior}:adir-recal", e.run_adaptive(inst, outcomes, e.RecalibratedDirection(inst, p)))
        for r in range(STAT_REPEATS):
            c, s, x = sequential_certificate(
                weights, outcomes, threshold, alpha=ALPHA, rng=np.random.default_rng([13, family, rho_index, r])
            )
            k = arms[f"stat_r{r}"]
            cost[k], status[k], exact[k] = c, STATUS[s], x
        rows.append(
            dict(
                family=family,
                rho=rho_index,
                size=inst.size,
                total=total,
                threshold=threshold,
                member=member,
                unresolved=unresolved,
                unresolved_cells=int((outcomes == e.UNRESOLVED).sum()),
                truth=STATUS[truth],
                clairvoyant=-1 if bound is None else bound,
                cost=cost,
                status=status,
                exact=exact,
                paid_member=paid_member,
                paid_unresolved=paid_unresolved,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs/v35/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/runs")
    parser.add_argument("--view", required=True, choices=("malecns", "flywire"))
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    view = next(v for v in manifest["views"] if v["view"] == args.view)
    path = args.benchmark / f"{args.view}.npz"
    if hash_file(path) != view["sha256"]:
        raise ValueError("benchmark file differs from its manifest")
    if args.split == "test":
        if args.freeze is None or args.limit is not None:
            raise ValueError("the test split needs a freeze record and runs completely")
        freeze = json.loads(args.freeze.read_text())
        if (
            freeze["code"] != code_identity()
            or freeze["benchmark_manifest_sha256"] != hash_file(args.benchmark / "manifest.json")
            or freeze["benchmark"][args.view] != view["sha256"]
        ):
            raise ValueError("code or benchmark changed after the freeze")
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.view}-{args.split}.npz"
    if out.exists():
        raise FileExistsError(out)
    with np.load(path, allow_pickle=False) as z:
        _DATA.update({k: z[k] for k in z.files})
    priors = view["priors"]
    names = arm_names(priors)
    _DATA["priors"], _DATA["arms"] = priors, {n: i for i, n in enumerate(names)}
    lengths = np.diff(_DATA["target_offset"])[_DATA["fam_target"]]
    _DATA["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    split_of = {t["code"]: t["split"] for t in manifest["types"]}
    families = [
        f for f in range(len(_DATA["fam_target"])) if split_of[int(_DATA["target_type"][_DATA["fam_target"][f]])] == args.split
    ]
    if args.limit is not None:
        families = families[: args.limit]
    started, cpu = time.time(), time.process_time()
    with Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=4)
    rows = [row for group in results for row in group]
    fields = ("family", "rho", "size", "total", "threshold", "member", "unresolved", "unresolved_cells", "truth", "clairvoyant")
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in fields}
    arrays.update(
        cost=np.stack([r["cost"] for r in rows]),
        status=np.stack([r["status"] for r in rows]),
        exact=np.stack([r["exact"] for r in rows]),
        paid_member=np.stack([r["paid_member"] for r in rows]),
        paid_unresolved=np.stack([r["paid_unresolved"] for r in rows]),
    )
    with out.open("xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(
        view=args.view,
        split=args.split,
        arms=names,
        rhos=RHOS,
        alpha=ALPHA,
        sessions=len(rows),
        families=len(families),
        code=code_identity(),
        benchmark_sha256=view["sha256"],
        result_sha256=hash_file(out),
        wall_seconds=time.time() - started,
        limit=args.limit,
    )
    (args.output / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("arms", "code")}))


if __name__ == "__main__":
    main()
