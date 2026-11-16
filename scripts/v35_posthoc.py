#!/usr/bin/env python3
"""Controls added after the held-out run (requested in review; nothing is tuned).

For every question of one split this runs
* priors without any prediction: a coin (p+ = 1/2) and the hemibrain share of the
  partner set (p+ = s), both with no unresolved mass, under Adaptive Greedy, Adaptive
  Dual Greedy and recalibrated AG;
* a round-robin of the member-first and nonmember-first orders for each real prior;
* the main frozen arms again, so that every arm also receives its cost under the
  three-way stopping rule (stop as soon as "cannot be certified" is proven);
* a hybrid sampling test (exact heavy half, sampled tail).

Costs of arms that exist in the frozen results must equal them exactly; the analysis
script checks this. The frozen runner and its results are not modified.
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
from connectomequest.annotation_v35 import posthoc as p

ROOT = Path(__file__).resolve().parents[1]
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
HYBRID_REPEATS = 5
ALPHA = 0.05
STATUS3 = {"undecidable": 0, "positive": 1, "negative": 2}
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
CODE_FILES = (
    "src/connectomequest/annotation_v35/engine.py",
    "src/connectomequest/annotation_v35/statistical.py",
    "src/connectomequest/annotation_v35/posthoc.py",
    "scripts/v35_posthoc.py",
)
CONSTANT = ("const_half", "const_share")
_DATA = {}


def arm_names(view):
    names = ["weight", "random_r0"]
    names += [f"{prior}:{policy}" for prior in CONSTANT for policy in ("ag", "adg", "recal")]
    real = ["mlp_s0", "ntac"] + (["provisional"] if view == "flywire" else [])
    for prior in real:
        names += [f"{prior}:{policy}" for policy in ("member", "nonmember", "direction", "rr", "ag", "adg", "recal", "adir")]
    return names


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
    arms, n = d["arms"], hi - lo
    share = float(np.clip(int(d["fam_source_weight"][family]) / int(d["fam_source_total"][family]), 0.01, 0.99))
    prob = {
        "const_half": np.tile((0.5, 0.5, 0.0), (n, 1)),
        "const_share": np.tile((share, 1.0 - share, 0.0), (n, 1)),
    }
    for prior in d["real_priors"]:
        prob[prior] = probabilities(d["plus_" + prior][d["fam_plus_offset"][family] : d["fam_plus_offset"][family] + n], d["unknown_" + prior][lo:hi])
    rows = []
    for rho_index, (num, den) in enumerate(RHOS):
        threshold, _ = e.threshold_goals(total, num * int(d["fam_source_weight"][family]), den * int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        cost = np.zeros(len(arms), dtype=np.int32)
        cost3 = np.zeros(len(arms), dtype=np.int32)

        def record(name, order):
            c, s = e.evaluate(inst, outcomes, order)
            c3, s3 = p.evaluate_three_way(inst, outcomes, order)
            expected = "undecidable" if truth == "unresolved" else truth
            if s != truth or s3 != expected or c3 > c or (truth != "unresolved" and c3 != c):
                raise AssertionError("arm contradicts the exact status")
            cost[arms[name]], cost3[arms[name]] = c, c3

        record("weight", e.weight_order(inst))
        record("random_r0", e.random_order(inst, np.random.default_rng([11, family, rho_index, 0])))
        for prior, q in prob.items():
            record(f"{prior}:ag", e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, q)))
            record(f"{prior}:adg", e.run_adaptive(inst, outcomes, e.AdaptiveDualGreedy(inst, q)))
            record(f"{prior}:recal", e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, q)))
            if prior in CONSTANT:
                continue
            record(f"{prior}:member", e.positive_order(inst, q))
            record(f"{prior}:nonmember", e.negative_order(inst, q))
            record(f"{prior}:direction", e.direction_order(inst, q))
            record(f"{prior}:rr", p.round_robin_order(inst, q))
            record(f"{prior}:adir", e.run_adaptive(inst, outcomes, e.AdaptiveDirection(inst, q)))
        hybrid_cost = np.zeros(HYBRID_REPEATS, dtype=np.int32)
        hybrid_status = np.zeros(HYBRID_REPEATS, dtype=np.int8)
        hybrid_exact = np.ones(HYBRID_REPEATS, dtype=bool)
        for r in range(HYBRID_REPEATS):
            c, s, x = p.hybrid_certificate(inst, outcomes, alpha=ALPHA, rng=np.random.default_rng([17, family, rho_index, r]))
            hybrid_cost[r], hybrid_status[r], hybrid_exact[r] = c, STATUS[s], x
        bound3 = p.undecidability_lower_bound(inst, outcomes)
        rows.append(
            dict(
                family=family,
                rho=rho_index,
                size=inst.size,
                truth=STATUS[truth],
                bound3=-1 if bound3 is None else bound3,
                cost=cost,
                cost3=cost3,
                hybrid_cost=hybrid_cost,
                hybrid_status=hybrid_status,
                hybrid_exact=hybrid_exact,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs/v35/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/posthoc")
    parser.add_argument("--view", required=True, choices=("malecns", "flywire"))
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--record", type=Path, help="post hoc record; required for the held-out split")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    view = next(v for v in manifest["views"] if v["view"] == args.view)
    path = args.benchmark / f"{args.view}.npz"
    if hash_file(path) != view["sha256"]:
        raise ValueError("benchmark file differs from its manifest")
    code = {name: hash_file(ROOT / name) for name in CODE_FILES}
    if args.split == "test":
        if args.record is None or args.limit is not None:
            raise ValueError("the held-out split needs the post hoc record and runs completely")
        if json.loads(args.record.read_text())["code"] != code:
            raise ValueError("code changed after the post hoc record")
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.view}-{args.split}.npz"
    if out.exists():
        raise FileExistsError(out)
    with np.load(path, allow_pickle=False) as z:
        _DATA.update({k: z[k] for k in z.files})
    names = arm_names(args.view)
    _DATA["arms"] = {n: i for i, n in enumerate(names)}
    _DATA["real_priors"] = ["mlp_s0", "ntac"] + (["provisional"] if args.view == "flywire" else [])
    lengths = np.diff(_DATA["target_offset"])[_DATA["fam_target"]]
    _DATA["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    split_of = {t["code"]: t["split"] for t in manifest["types"]}
    families = [f for f in range(len(_DATA["fam_target"])) if split_of[int(_DATA["target_type"][_DATA["fam_target"][f]])] == args.split]
    if args.limit is not None:
        families = families[: args.limit]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=4)
    rows = [row for group in results for row in group]
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in ("family", "rho", "size", "truth", "bound3")}
    arrays.update({k: np.stack([r[k] for r in rows]) for k in ("cost", "cost3", "hybrid_cost", "hybrid_status", "hybrid_exact")})
    with out.open("xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(view=args.view, split=args.split, arms=names, sessions=len(rows), families=len(families), alpha=ALPHA, code=code,
                benchmark_sha256=view["sha256"], result_sha256=hash_file(out), wall_seconds=time.time() - started, limit=args.limit)
    (args.output / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("arms", "code")}))


if __name__ == "__main__":
    main()
