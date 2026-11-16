#!/usr/bin/env python3
"""Synthetic priors with correlated errors at matched within-question AUROC (round 5, ``priors``).

For every certifiable question of one split this runs direction-fixed, Adaptive Greedy and
recalibrated AG with the priors of ``connectomequest.round5.priors_corr``: errors correlated with
synapse weight (heavy cells more accurate: ``wplus``; less accurate: ``wminus``) and errors shared
within release type (``rtype``), each calibrated (``base``) and confident (``oracle``), at the five
AUROC levels of the paper's sweep and with three noise draws. The independent arms already exist in
``outputs/v35/sweep`` and are not recomputed.

Matching: for each connectome, form, correlation and level, the scale of the prior is found by
bisection so that the realized within-question AUROC, averaged over the question families and
draws of the split exactly as in the paper, equals the value realized by the independent prior of
the frozen sweep on the same families. Base rates of members and unresolved answers come from the
development split, as in the sweep.

The held-out split refuses to run unless ``outputs/round5/record.json`` lists the sha256 of this
script and of every round-5 module it imports; results are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connectomequest.annotation_v35 import engine as e
from connectomequest.annotation_v35 import sweep as sw
from connectomequest.neurosym import common as nc
from connectomequest.round5 import priors_common as pc
from connectomequest.round5 import priors_corr as cr

sys.path.insert(0, str(pc.ROOT / "scripts"))
from v35_prior_sweep import base_rates, family_data  # noqa: E402  (frozen; imported, not modified)

LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)
POLICIES = ("direction", "ag", "recal")
DRAWS = 3
SWEEP_VARIANT = {"base": 1, "oracle": 2}  # index in the frozen sweep's variants (none, base, oracle)
_D = {}


def frozen_sweep(view, split):
    meta = json.loads((pc.ROOT / f"outputs/v35/sweep/{view}-{split}.json").read_text())
    with np.load(pc.ROOT / f"outputs/v35/sweep/{view}-{split}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in ("family", "auroc")}
    if tuple(meta["aurocs"]) != LEVELS or meta["variants"] != ["none", "base", "oracle"] or meta["draws"] != DRAWS:
        raise ValueError("frozen sweep layout changed")
    families, first = np.unique(d["family"], return_index=True)
    return families, d["auroc"][first], meta


def fit_scales(families, targets, rates, log=print):
    """Scale per (form, correlation, level) matching the frozen sweep's realized mean AUROC on these families."""
    outcomes, percentile, member_rate, group = [], [], [], []
    numbers = {c: [[], [], []] for c in cr.CORRELATIONS}
    for g, f in enumerate(int(x) for x in families):
        _, lo, hi, y = family_data(_D, f)
        outcomes.append(y)
        percentile.append(cr.weight_percentile(_D["pre_weight"][lo:hi]))
        member_rate.append(np.full(hi - lo, rates["member"][str(int(_D["fam_kind"][f]))]))
        group.append(np.full(hi - lo, g))
        for c in cr.CORRELATIONS:
            for draw in range(DRAWS):
                numbers[c][draw].append(cr.random_numbers(f, draw, hi - lo, _D["release"][lo:hi], c))
    outcomes, percentile, member_rate, group = (np.concatenate(x) for x in (outcomes, percentile, member_rate, group))
    numbers = {c: [tuple(np.concatenate([t[i] for t in numbers[c][draw]]) for i in range(3)) for draw in range(DRAWS)] for c in cr.CORRELATIONS}
    positive = outcomes == e.MEMBER

    def realized(form, correlation, scale):
        values = [cr.grouped_auroc(cr.prior(form, correlation, scale, outcomes, percentile, numbers[correlation][draw], member_rate, rates["unresolved"])[:, e.MEMBER],
                                   positive, group) for draw in range(DRAWS)]
        return float(np.nanmean(np.stack(values)))

    scales = {}
    for form in cr.FORMS:
        for correlation in cr.CORRELATIONS:
            for j, level in enumerate(LEVELS):
                target = targets[form][j]
                if form == "base":
                    lo, hi, increasing = 0.0, 4.0 * sw.separation(level), True
                else:
                    lo, hi, increasing = 0.0, 1.0, False
                scale = cr.fit_scale(lambda c: realized(form, correlation, c), target, lo, hi, increasing, iterations=30)
                got = realized(form, correlation, scale)
                scales[f"{form}/{correlation}/{level:g}"] = dict(scale=scale, target=target, realized=got)
                log(f"  {form:6s} {correlation:6s} {level:.2f}: scale {scale:.4f} target {target:.4f} realized {got:.4f}")
    return scales


def run_family(family):
    d = _D
    target, lo, hi, outcomes = family_data(d, family)
    cells, weights = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi]
    total, n = int(weights.sum()), hi - lo
    rates = d["rates"]
    member_rate = rates["member"][str(int(d["fam_kind"][family]))]
    percentile = cr.weight_percentile(weights)
    groups = d["release"][lo:hi]
    priors, auroc = {}, np.full((DRAWS, len(LEVELS), len(cr.CORRELATIONS), len(cr.FORMS)), np.nan)
    for i in range(DRAWS):
        for k, correlation in enumerate(cr.CORRELATIONS):
            numbers = cr.random_numbers(family, i, n, groups, correlation)
            for m, form in enumerate(cr.FORMS):
                for j, level in enumerate(LEVELS):
                    scale = d["scales"][f"{form}/{correlation}/{level:g}"]["scale"]
                    q = cr.prior(form, correlation, scale, outcomes, percentile, numbers, member_rate, rates["unresolved"])
                    priors[i, j, k, m] = q
                    value = sw.member_auroc(q[:, e.MEMBER], outcomes)
                    auroc[i, j, k, m] = np.nan if value is None else value
    rows = []
    for rho_index, (num, den) in enumerate(nc.RHOS):
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

        cost = np.zeros((DRAWS, len(LEVELS), len(cr.CORRELATIONS), len(cr.FORMS), len(POLICIES)), dtype=np.int32)
        for (i, j, k, m), q in priors.items():
            cost[i, j, k, m, 0] = cost_of(e.direction_order(inst, q))
            cost[i, j, k, m, 1] = cost_of(e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, q)))
            cost[i, j, k, m, 2] = cost_of(e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, q)))
        rows.append(dict(family=family, rho=rho_index, size=inst.size, truth=1 if truth == "positive" else 2, weight=cost_of(e.weight_order(inst)), cost=cost, auroc=auroc))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=pc.VIEWS)
    parser.add_argument("--split", required=True, choices=pc.SPLITS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=pc.OUT / "corrsweep")
    parser.add_argument("--limit", type=int, help="first N families only (development smoke tests)")
    args = parser.parse_args()
    if args.split == "test" and args.limit is not None:
        raise ValueError("the held-out split runs completely")
    code = pc.guard(args.split, __file__) or pc.code_identity(__file__)
    stem = f"{args.view}-{args.split}" + ("" if args.limit is None else f"-limit{args.limit}")
    out, meta_path = args.output / f"{stem}.npz", args.output / f"{stem}.json"
    if out.exists() or meta_path.exists():
        raise FileExistsError(out)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "priors", f"corrsweep-{args.view}", out)
    manifest = nc.manifest()
    view = next(v for v in manifest["views"] if v["view"] == args.view)
    if pc.hash_file(nc.BENCH / f"{args.view}.npz") != view["sha256"]:
        raise ValueError("benchmark file differs from its manifest")
    _D.update(nc.benchmark(args.view))
    vd = nc.view_data(args.view)
    rows_of = np.searchsorted(vd["ids"], _D["pre_cell"])
    if not np.array_equal(vd["ids"][rows_of], _D["pre_cell"]):
        raise ValueError("input cells absent from the view cache")
    _D["release"] = vd["release_code"][rows_of].astype(np.int64)
    split_of = {t["code"]: t["split"] for t in manifest["types"]}
    family_split = np.array([split_of[int(_D["target_type"][_D["fam_target"][f]])] for f in range(len(_D["fam_target"]))])
    _D["rates"] = base_rates(_D, np.flatnonzero(family_split == "development"))
    families, frozen_auroc, sweep_meta = frozen_sweep(args.view, args.split)
    if args.limit is not None:
        families, frozen_auroc = families[: args.limit], frozen_auroc[: args.limit]
    targets = {form: [float(np.nanmean(frozen_auroc[:, :, j, SWEEP_VARIANT[form]])) for j in range(len(LEVELS))] for form in cr.FORMS}
    started = time.time()
    print(f"fitting scales on {len(families)} families", flush=True)
    _D["scales"] = fit_scales(families, targets, _D["rates"], log=lambda m: print(m, flush=True))
    fit_seconds = time.time() - started
    with Pool(args.workers) as pool:
        results = pool.map(run_family, [int(f) for f in families], chunksize=2)
    rows = [r for group in results for r in group]
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in ("family", "rho", "size", "truth", "weight")}
    arrays.update({k: np.stack([r[k] for r in rows]) for k in ("cost", "auroc")})
    pc.save_npz_new(out, **arrays)
    meta = dict(view=args.view, split=args.split, levels=list(LEVELS), correlations=list(cr.CORRELATIONS), forms=list(cr.FORMS), policies=list(POLICIES),
                draws=DRAWS, beta=cr.BETA, scales=_D["scales"], targets=targets, rates=_D["rates"], sessions=len(rows), families=len(families),
                frozen_sweep_sha256=sweep_meta["result_sha256"], code=code, benchmark_sha256=view["sha256"],
                result_sha256=pc.hash_file(out), fit_seconds=fit_seconds, wall_seconds=time.time() - started, limit=args.limit, workers=args.workers)
    pc.write_json_new(meta_path, meta)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "scales")}))


if __name__ == "__main__":
    main()
