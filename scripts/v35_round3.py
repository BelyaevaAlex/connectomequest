#!/usr/bin/env python3
"""Third round of review (post hoc; nothing is tuned on held-out questions).

For every certifiable question of one split this runs
* the earlier-release prior (FlyWire) built in other ways: mixture weight 0.8, 0.95 or
  0.99 instead of 0.9, or, on labelled cells, the shares of member, nonmember and
  unresolved answers that each class of earlier label (partner type, outside the
  vocabulary, other hemibrain type) has on the development split; under member-first,
  direction-fixed, Adaptive Greedy and recalibrated Adaptive Greedy. Weight 0.9 is the
  stored prior and must reproduce the frozen costs;
* recalibrated Adaptive Greedy with one choice changed at a time (strength 1, 16 or 64
  instead of 4; one, three or eight groups instead of five; unresolved rate shrunk
  toward the prior or toward the session rate only), for the topology prior, the
  earlier-release prior and the confident synthetic priors of the second round at an
  AUROC of 0.6 and 0.8 (same noise as there). The frozen choices must reproduce the
  frozen and second-round costs;
* the within-question AUROC of the topology, NTAC and earlier-release priors, for the
  analysis of prior quality.
Frozen code and results are not modified.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v35 import ablation as ab
from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import sweep as s

ROOT = Path(__file__).resolve().parents[1]
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
CONSTRUCTIONS = ("a0.8", "a0.9", "a0.95", "a0.99", "calibrated")
POLICIES = ("member", "direction", "ag", "recal")
VARIANTS = tuple(ab.RECAL_VARIANTS)
REAL_PRIORS = ("mlp_s0", "provisional")
SYN_LEVELS = (0.6, 0.8)
DRAWS = 3
AUROC_PRIORS = ("mlp_s0", "ntac", "provisional")
CODE_FILES = (
    "src/connectomequest/annotation_v35/engine.py",
    "src/connectomequest/annotation_v35/sweep.py",
    "src/connectomequest/annotation_v35/ablation.py",
    "scripts/v35_round3.py",
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
    return lo, hi, outcomes


def raw_prior(d, name, family, lo, hi):
    start = d["fam_plus_offset"][family]
    return d["plus_" + name][start : start + hi - lo], d["unknown_" + name][lo:hi]


def earlier_parts(d, family, lo, hi):
    pe, ue = raw_prior(d, "provisional", family, lo, hi)
    pt, ut = raw_prior(d, "mlp_s0", family, lo, hi)
    return ab.earlier_release_parts(pe, ue, pt, ut)


def development_rates(d, families):
    classes, outcomes = [], []
    for family in families:
        lo, hi, y = family_data(d, family)
        labelled, cls = earlier_parts(d, family, lo, hi)
        classes.append(cls[labelled])
        outcomes.append(y[labelled])
    return ab.class_rates(np.concatenate(classes), np.concatenate(outcomes))


def run_family(family):
    d = _D
    lo, hi, outcomes = family_data(d, family)
    cells, weights = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi]
    total, n = int(weights.sum()), hi - lo
    flywire = "plus_provisional" in d
    priors = {name: probabilities(*raw_prior(d, name, family, lo, hi)) for name in ("mlp_s0", "ntac") + (("provisional",) if flywire else ())}
    auroc = np.full(len(AUROC_PRIORS), np.nan)
    for k, name in enumerate(AUROC_PRIORS):
        if name in priors:
            value = s.member_auroc(priors[name][:, e.MEMBER], outcomes)
            auroc[k] = np.nan if value is None else value
    built = {}
    if flywire:
        labelled, cls = earlier_parts(d, family, lo, hi)
        rebuilt = ab.earlier_release_prior(priors["mlp_s0"], labelled, cls, alpha=ab.FROZEN_ALPHA)
        if np.max(np.abs(rebuilt - priors["provisional"])) > 1e-5:
            raise AssertionError("earlier-release prior not recovered")
        for name in CONSTRUCTIONS:
            if name == "a0.9":
                built[name] = priors["provisional"]
            elif name == "calibrated":
                built[name] = ab.earlier_release_prior(priors["mlp_s0"], labelled, cls, rates=d["rates"])
            else:
                built[name] = ab.earlier_release_prior(priors["mlp_s0"], labelled, cls, alpha=float(name[1:]))
    uniform = [np.random.default_rng([31, family, draw]).random((2, n)) for draw in range(DRAWS)]
    synthetic = {(i, j): s.oracle_prior(outcomes, uniform[i][0], uniform[i][1], level) for i in range(DRAWS) for j, level in enumerate(SYN_LEVELS)}
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

        def recal(q, variant):
            return cost_of(e.run_adaptive(inst, outcomes, ab.AblatedRecalibration(inst, q, **ab.RECAL_VARIANTS[variant])))

        earlier = np.full((len(CONSTRUCTIONS), len(POLICIES)), -1, dtype=np.int32)
        for a, name in enumerate(built):
            q = built[name]
            earlier[a] = (
                cost_of(e.positive_order(inst, q)),
                cost_of(e.direction_order(inst, q)),
                cost_of(e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, q))),
                cost_of(e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, q))),
            )
        real = np.full((len(REAL_PRIORS), len(VARIANTS)), -1, dtype=np.int32)
        for a, name in enumerate(REAL_PRIORS):
            if name in priors:
                real[a] = [recal(priors[name], v) for v in VARIANTS]
        syn = np.zeros((DRAWS, len(SYN_LEVELS), len(VARIANTS)), dtype=np.int32)
        for (i, j), q in synthetic.items():
            syn[i, j] = [recal(q, v) for v in VARIANTS]
        rows.append(dict(family=family, rho=rho_index, size=inst.size, truth=1 if truth == "positive" else 2, weight=cost_of(e.weight_order(inst)),
                         earlier=earlier, recal_real=real, recal_syn=syn, auroc=auroc))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs/v35/benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/round3")
    parser.add_argument("--view", required=True, choices=("malecns", "flywire"))
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--record", type=Path, help="third post hoc record; required for the held-out split")
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
            raise ValueError("the held-out split needs the third post hoc record and runs completely")
        if json.loads(args.record.read_text())["code"]["round3"] != code:
            raise ValueError("code changed after the third post hoc record")
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
    if "plus_provisional" in _D:
        _D["rates"] = development_rates(_D, np.flatnonzero(family_split == "development"))
    families = np.flatnonzero(family_split == args.split).tolist()
    if args.limit is not None:
        families = families[: args.limit]
    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_family, families, chunksize=2)
    rows = [row for group in results for row in group]
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in ("family", "rho", "size", "truth", "weight")}
    arrays.update({k: np.stack([r[k] for r in rows]) for k in ("earlier", "recal_real", "recal_syn", "auroc")})
    with out.open("xb") as stream:
        np.savez(stream, **arrays)
    rates = _D.get("rates")
    meta = dict(view=args.view, split=args.split, constructions=list(CONSTRUCTIONS), policies=list(POLICIES), variants=list(VARIANTS),
                variant_settings={k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()} for k, v in ab.RECAL_VARIANTS.items()},
                real_priors=list(REAL_PRIORS), synthetic_levels=list(SYN_LEVELS), draws=DRAWS, auroc_priors=list(AUROC_PRIORS),
                development_class_rates=None if rates is None else {c: list(map(float, rates[k])) for k, c in enumerate(ab.CLASSES)},
                sessions=len(rows), families=len(families), code=code, benchmark_sha256=view["sha256"], result_sha256=hash_file(out),
                wall_seconds=time.time() - started, limit=args.limit)
    (args.output / f"{args.view}-{args.split}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "variant_settings")}))


if __name__ == "__main__":
    main()
