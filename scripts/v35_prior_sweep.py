#!/usr/bin/env python3
"""Priors of controlled quality (second round of review; nothing is tuned).

For every certifiable question of one split this runs
* synthetic priors that separate members from other input cells with an area under
  the ROC curve of 0.6, 0.7, 0.8, 0.9 or 0.95, in three variants: calibrated
  membership probabilities with no unresolved mass ("none") or with the base-rate
  unresolved mass ("base"), and a confident noisy oracle ("oracle"); three noise
  draws each, under member-first, direction-fixed, round-robin, Adaptive Greedy and
  recalibrated Adaptive Greedy;
* the topology and NTAC priors with their unresolved mass replaced by the base rate,
  under Adaptive Greedy and recalibrated Adaptive Greedy, and the same two rules with
  the priors unchanged (these must reproduce the frozen costs).

Base rates come from the development split only: the share of members among the input
cells of a question, for each kind of partner set, and the share of unresolved cells
among the input cells that are not members. The noise of a cell depends on the draw,
the question family and the cell, not on the level, so levels are compared on common
noise. Frozen code and results are not modified.
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
from connectomequest.annotation_v35 import sweep as s

ROOT = Path(__file__).resolve().parents[1]
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
AUROCS = (0.6, 0.7, 0.8, 0.9, 0.95)
VARIANTS = ("none", "base", "oracle")
POLICIES = ("member", "direction", "rr", "ag", "recal")
DRAWS = 3
REAL = tuple(f"{prior}:{variant}:{policy}" for prior in ("mlp_s0", "ntac") for variant in ("as_is", "base") for policy in ("ag", "recal"))
CODE_FILES = (
    "src/connectomequest/annotation_v35/engine.py",
    "src/connectomequest/annotation_v35/statistical.py",
    "src/connectomequest/annotation_v35/posthoc.py",
    "src/connectomequest/annotation_v35/sweep.py",
    "scripts/v35_prior_sweep.py",
)
_D = {}


def probabilities(plus, unknown):
    plus = np.clip(plus.astype(float), 0.0, 1.0)
    unknown = np.clip(unknown.astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def family_data(d, family):
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    labels = d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    return target, lo, hi, outcomes


def base_rates(d, families):
    """Member share per kind of partner set and unresolved share among non-members, over input cells."""
    members = {0: [0, 0], 1: [0, 0]}
    unresolved = [0, 0]
    for family in families:
        _, _, _, y = family_data(d, family)
        kind = int(d["fam_kind"][family])
        members[kind][0] += int((y == e.MEMBER).sum())
        members[kind][1] += len(y)
        unresolved[0] += int((y == e.UNRESOLVED).sum())
        unresolved[1] += int((y != e.MEMBER).sum())
    return dict(member={str(k): v[0] / v[1] for k, v in members.items()}, unresolved=unresolved[0] / unresolved[1])


def run_family(family):
    d = _D
    target, lo, hi, outcomes = family_data(d, family)
    cells, weights = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi]
    total, n = int(weights.sum()), hi - lo
    rates = d["rates"]
    member_rate = rates["member"][str(int(d["fam_kind"][family]))]
    noise = [np.random.default_rng([29, family, draw]).standard_normal(n) for draw in range(DRAWS)]
    uniform = [np.random.default_rng([31, family, draw]).random((2, n)) for draw in range(DRAWS)]
    synthetic = {}
    auroc = np.full((DRAWS, len(AUROCS), len(VARIANTS)), np.nan)
    for i in range(DRAWS):
        for j, level in enumerate(AUROCS):
            for k, variant in enumerate(VARIANTS):
                if variant == "oracle":
                    synthetic[i, j, k] = s.oracle_prior(outcomes, uniform[i][0], uniform[i][1], level)
                else:
                    synthetic[i, j, k] = s.synthetic_prior(outcomes, noise[i], level, member_rate, None if variant == "none" else rates["unresolved"])
                value = s.member_auroc(synthetic[i, j, k][:, e.MEMBER], outcomes)
                auroc[i, j, k] = np.nan if value is None else value
    real = {}
    for prior in ("mlp_s0", "ntac"):
        q = probabilities(d["plus_" + prior][d["fam_plus_offset"][family] : d["fam_plus_offset"][family] + n], d["unknown_" + prior][lo:hi])
        real[prior, "as_is"] = q
        real[prior, "base"] = s.with_unresolved_rate(q, rates["unresolved"])
    rows = []
    for rho_index, (num, den) in enumerate(RHOS):
        threshold, _ = e.threshold_goals(total, num * int(d["fam_source_weight"][family]), den * int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        if truth == "unresolved":
            continue

        def cost_of(order):
            c, status = e.evaluate(inst, outcomes, order)
            if status != truth:
                raise AssertionError("arm contradicts the exact status")
            return c

        def orders(q):
            return dict(
                member=lambda: e.positive_order(inst, q),
                direction=lambda: e.direction_order(inst, q),
                rr=lambda: p.round_robin_order(inst, q),
                ag=lambda: e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, q)),
                recal=lambda: e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, q)),
            )

        cost = np.zeros((DRAWS, len(AUROCS), len(VARIANTS), len(POLICIES)), dtype=np.int32)
        for (i, j, k), q in synthetic.items():
            make = orders(q)
            for m, policy in enumerate(POLICIES):
                cost[i, j, k, m] = cost_of(make[policy]())
        real_cost = np.zeros(len(REAL), dtype=np.int32)
        for m, name in enumerate(REAL):
            prior, variant, policy = name.split(":")
            real_cost[m] = cost_of(orders(real[prior, variant])[policy]())
        rows.append(dict(family=family, rho=rho_index, size=inst.size, truth=1 if truth == "positive" else 2, weight=cost_of(e.weight_order(inst)),
                         cost=cost, real=real_cost, auroc=auroc))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs/v35/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/sweep")
    parser.add_argument("--view", required=True, choices=("malecns", "flywire"))
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--record", type=Path, help="second post hoc record; required for the held-out split")
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
            raise ValueError("the held-out split needs the second post hoc record and runs completely")
        if json.loads(args.record.read_text())["code"]["prior_sweep"] != code:
            raise ValueError("code changed after the second post hoc record")
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.view}-{args.split}.npz"
    if out.exists():
        raise FileExistsError(out)
    with np.load(path, allow_pickle=False) as z:
        _D.update({k: z[k] for k in z.files})
    lengths = np.diff(_D["target_offset"])[_D["fam_target"]]
    _D["fam_plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    split_of = {t["code"]: t["split"] for t in manifest["types"]}
    family_split = np.array([split_of[int(_D["target_type"][_D["fam_target"][f]])] for f in range(len(_D["fam_target"]))])
    _D["rates"] = base_rates(_D, np.flatnonzero(family_split == "development"))
    families = np.flatnonzero(family_split == args.split).tolist()
    if args.limit is not None:
        families = families[: args.limit]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=2)
    rows = [row for group in results for row in group]
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in ("family", "rho", "size", "truth", "weight")}
    arrays.update({k: np.stack([r[k] for r in rows]) for k in ("cost", "real", "auroc")})
    with out.open("xb") as stream:
        np.savez(stream, **arrays)
    meta = dict(view=args.view, split=args.split, aurocs=list(AUROCS), variants=list(VARIANTS), policies=list(POLICIES), draws=DRAWS, real_arms=list(REAL),
                rates=_D["rates"], sessions=len(rows), families=len(families), code=code, benchmark_sha256=view["sha256"], result_sha256=hash_file(out),
                wall_seconds=time.time() - started, limit=args.limit)
    (args.output / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("code",)}))


if __name__ == "__main__":
    main()
