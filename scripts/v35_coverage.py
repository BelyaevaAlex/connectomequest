#!/usr/bin/env python3
"""Post hoc analysis: how complete must earlier labels be?

FlyWire only. For a coverage level c, a fixed pseudo-random fraction c of all neurons
keeps its earlier-release prior and every other neuron falls back to the topology
prior (nested: a neuron covered at a lower level stays covered at a higher one).
c = 0 is the topology prior and c = 1 the earlier-release prior of the paper. The
question is where a committing order overtakes the hedging rule. Specified after
the held-out analysis and first run on the development split; tunes nothing. The
held-out split was added in the second round of review and needs the second post
hoc record, whose code hashes must match.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v35 import engine as e

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/v35/benchmark"
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
POLICIES = ("member", "direction", "ag", "adg", "recal", "adir")
BOOTSTRAP = 2000
CODE_FILES = ("src/connectomequest/annotation_v35/engine.py", "scripts/v35_coverage.py")
_D = {}


def covered(cells, level):
    mixed = (cells.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return (mixed % np.uint64(10_000)).astype(np.int64) < int(round(level * 10_000))


def probabilities(plus, unknown):
    plus = np.clip(plus.astype(float), 0.0, 1.0)
    unknown = np.clip(unknown.astype(float), 0.0, 1.0 - plus)
    return np.column_stack((plus, 1.0 - plus - unknown, unknown))


def run_family(f):
    d = _D
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    span = slice(d["plus_offset"][f], d["plus_offset"][f] + hi - lo)
    weak = probabilities(d["plus_mlp_s0"][span], d["unknown_mlp_s0"][lo:hi])
    strong = probabilities(d["plus_provisional"][span], d["unknown_provisional"][lo:hi])
    priors = [np.where(covered(cells, level)[:, None], strong, weak) for level in LEVELS]
    rows = []
    for num, den in RHOS:
        threshold, _ = e.threshold_goals(total, num * int(d["fam_source_weight"][f]), den * int(d["fam_source_total"][f]))
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        if truth == "unresolved":
            continue
        cost = np.zeros((len(LEVELS), len(POLICIES)), dtype=np.int64)
        for i, p in enumerate(priors):
            orders = dict(
                member=e.positive_order(inst, p),
                direction=e.direction_order(inst, p),
                ag=e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, p)),
                adg=e.run_adaptive(inst, outcomes, e.AdaptiveDualGreedy(inst, p)),
                recal=e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, p)),
                adir=e.run_adaptive(inst, outcomes, e.AdaptiveDirection(inst, p)),
            )
            for j, name in enumerate(POLICIES):
                c, status = e.evaluate(inst, outcomes, orders[name])
                if status != truth:
                    raise AssertionError("policy contradicts the exact answer")
                cost[i, j] = c
        rows.append((int(d["target_type"][t]), e.evaluate(inst, outcomes, e.weight_order(inst))[0], e.clairvoyant_cost(inst, outcomes), cost))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="development", choices=("development", "test"))
    parser.add_argument("--record", type=Path, help="second post hoc record; required for the held-out split")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/analysis")
    args = parser.parse_args()
    out = args.output / f"coverage-{args.split}.json"
    if args.split == "test":
        code = {name: hash_file(ROOT / name) for name in CODE_FILES}
        if args.record is None or json.loads(args.record.read_text())["code"]["coverage"] != code:
            raise ValueError("the held-out split needs the second post hoc record with matching code hashes")
        if out.exists():
            raise FileExistsError(out)
    manifest = json.loads((BENCH / "manifest.json").read_text())
    with np.load(BENCH / "flywire.npz", allow_pickle=False) as z:
        _D.update({k: z[k] for k in z.files})
    lengths = np.diff(_D["target_offset"])[_D["fam_target"]]
    _D["plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    selected = np.isin(_D["target_type"], [t["code"] for t in manifest["types"] if t["split"] == args.split])
    families = np.flatnonzero(selected[_D["fam_target"]]).tolist()
    with Pool(26) as pool:
        rows = [row for group in pool.map(run_family, families, chunksize=4) for row in group]
    cluster = np.unique(np.array([r[0] for r in rows]), return_inverse=True)[1]
    n = int(cluster.max()) + 1
    weight = np.array([r[1] for r in rows], dtype=float)
    bound = np.array([r[2] for r in rows], dtype=float)
    cost = np.stack([r[3] for r in rows]).astype(float)
    rng = np.random.default_rng(20261112)
    counts = np.stack([np.bincount(row, minlength=n) for row in rng.integers(0, n, size=(BOOTSTRAP, n))]).astype(float)

    def ratio(x, y):
        sx, sy = np.bincount(cluster, weights=x, minlength=n), np.bincount(cluster, weights=y, minlength=n)
        boot = (counts @ sx) / (counts @ sy)
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    cells = _D["pre_cell"]
    result = dict(
        questions=len(rows),
        levels=list(LEVELS),
        policies=list(POLICIES),
        covered_share_of_input_cells=[float(covered(cells, level).mean()) for level in LEVELS],
        clairvoyant=ratio(bound, weight),
        ratio_to_weight={f"{level:g}": {name: ratio(cost[:, i, j], weight) for j, name in enumerate(POLICIES)} for i, level in enumerate(LEVELS)},
        ag_over_direction={f"{level:g}": ratio(cost[:, i, POLICIES.index("ag")], cost[:, i, POLICIES.index("direction")]) for i, level in enumerate(LEVELS)},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for level in LEVELS:
        print(level, {k: round(v["value"], 3) for k, v in result["ratio_to_weight"][f"{level:g}"].items()}, "ag/dir", round(result["ag_over_direction"][f"{level:g}"]["value"], 3))
    print("clairvoyant", round(result["clairvoyant"]["value"], 3), "questions", len(rows))


if __name__ == "__main__":
    main()
