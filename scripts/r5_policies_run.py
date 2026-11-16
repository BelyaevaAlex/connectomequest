#!/usr/bin/env python3
"""Round 5, workstream ``policies``: new checking policies on one split, one prior.

Arms (``prior:policy``):

* ``us``, ``wus``, ``us-recal``, ``wus-recal`` - generic active-learning baselines
  (uncertainty sampling, weight-scaled; static or recalibrated beliefs);
* ``voi-rollout``, ``voi-complete`` and their ``-recal`` versions - one-step
  value-of-information rules;
* ``prop-recal`` - recalibrated AG with answer propagation (lambda 8, gamma 3, tau 0.3);
  needed as the base of an interleave and compared with the stored propagation run
  where one exists (topology prior);
* ``recal+wo``, ``prop-recal+wo`` - safe interleave of the base with weight order
  (shadow run, at most twice the cheaper of the two); ``...+wo-informed`` - diagnostic
  variant in which the base also sees weight order's answers (no guarantee).

Costs of arms that already exist (frozen ``recal``; stored ``prop-recal`` for mlp_s0) are
recomputed only as far as an interleave needs their orders, asserted equal to the
stored costs, and not written again. Weight order is recomputed and asserted equal to
the frozen run where it exists. Every order is checked by the frozen verifier against
the status of the fully checked question and the clairvoyant bound.

The held-out split is refused unless ``outputs/round5/record.json`` lists the sha256
of this runner and of every round-5 module it imports; outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
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
from connectomequest.round5 import policies_interleave as il  # noqa: E402
from connectomequest.round5 import policies_voi as voi  # noqa: E402

PROPAGATION = dict(lam=8.0, gamma=3.0, tau=0.3)
GROUPS = {
    "baselines": ["us", "wus", "us-recal", "wus-recal"],
    "voi": ["voi-rollout", "voi-complete", "voi-rollout-recal", "voi-complete-recal"],
    "interleave": ["recal+wo", "recal+wo-informed"],
    "propagation": ["prop-recal", "prop-recal+wo", "prop-recal+wo-informed"],
}
_RUN = {}


def propagation_policy(view, fam, inst, p):
    from connectomequest.neurosym import kernel as kk

    K = pdata.question_kernel(view, fam, PROPAGATION["gamma"], PROPAGATION["tau"])
    return kk.KernelRecalibrated(inst, p, K, PROPAGATION["lam"])


def family_job(f):
    view, prior_name, arms = _RUN["view"], _RUN["prior"], _RUN["arms"]
    fam = pdata.family(view, f)
    p = pdata.prior(view, fam, prior_name)
    y = fam["outcomes"]
    rows = []
    for rho, threshold in pdata.thresholds(fam):
        inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
        truth = e.full_information_status(inst, y)
        bound = e.clairvoyant_cost(inst, y)
        costs, orders = {}, {}

        def record(name, order):
            cost, status = e.evaluate(inst, y, order)
            if status != truth or (truth == "unresolved" and cost != inst.size) or (bound is not None and cost < bound):
                raise AssertionError(f"{prior_name}:{name} contradicts the exact full-information status")
            costs[name] = cost
            orders[name] = order

        record("weight", e.weight_order(inst))
        need_recal = any(a.startswith("recal+") for a in arms)
        need_prop = any(a.startswith("prop-recal") for a in arms)
        if need_recal:
            record("recal", e.run_adaptive(inst, y, e.RecalibratedGreedy(inst, p)))
        if need_prop:
            record("prop-recal", e.run_adaptive(inst, y, propagation_policy(view, fam, inst, p)))
        for name in arms:
            if name in costs:
                continue
            if name in bl.ORDERS or name in bl.ADAPTIVE:
                record(name, bl.run(name, inst, y, p))
            elif name in voi.POLICIES:
                record(name, e.run_adaptive(inst, y, voi.POLICIES[name](inst, p)))
            elif name.endswith("+wo"):
                record(name, il.safe_interleave(inst, y, orders[name[:-3]]))
                k_p, k_w = costs[name[:-3]], costs["weight"]
                if costs[name] > 2 * min(k_p, k_w) and truth != "unresolved":
                    raise AssertionError("safe interleave exceeded twice the cheaper policy")
            elif name.endswith("+wo-informed"):
                base = name[: -len("+wo-informed")]
                pol = e.RecalibratedGreedy(inst, p) if base == "recal" else propagation_policy(view, fam, inst, p)
                record(name, il.informed_interleave(inst, y, pol))
            else:
                raise ValueError(f"unknown arm {name}")
        rows.append((int(f), rho, inst.size, pdata.STATUS[truth], int(threshold), costs))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--prior", required=True, help="const_half, const_share, a benchmark prior (mlp_s0, provisional, ...) or a file prior")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--arms", nargs="*", help="explicit arm list (overrides --groups)")
    parser.add_argument("--output", type=Path, default=pdata.OUT / "runs")
    parser.add_argument("--workers", type=int, default=pdata.workers())
    parser.add_argument("--limit", type=int, help="development only: first N families (smoke tests)")
    args = parser.parse_args()
    if args.split == "test" and args.limit is not None:
        raise SystemExit("the held-out split runs completely")
    arms = args.arms or [a for g in args.groups for a in GROUPS[g]]
    output = pdata.default_output(args.output, pdata.OUT / "runs", args.split)
    pdata.preload(args.view)
    if args.prior not in pdata.available_priors(args.view):
        raise SystemExit(f"prior {args.prior} not available for {args.view}: {pdata.available_priors(args.view)}")
    # every import happens before the record check, so the check sees every round-5 module the run uses
    pdata.benchmark(args.view)
    if any(a.startswith("prop") for a in arms):
        pdata.preload_kernel(args.view)
    code = pdata.require_record(__file__, args.split) or pdata.code_identity(__file__)
    out = pdata.fresh(output / f"{args.view}-{args.split}-{args.prior}.npz")
    pdata.fresh(out.with_suffix(".json"))
    if args.split == "test":  # the round-5 one-shot gate (after every import and the output check)
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "policies", f"{args.view}-{args.prior}", out)
    families = pdata.split_families(args.view, args.split)
    if args.limit is not None:
        families = families[: args.limit]
    _RUN.update(view=args.view, prior=args.prior, arms=arms)
    pin = pdata.pin_prior(args.view, args.prior)
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(family_job, list(map(int, families)), chunksize=2)
    flat = [r for group in results for r in group]
    names = sorted({n for *_, costs in flat for n in costs})
    arrays = dict(family=np.array([r[0] for r in flat]), rho=np.array([r[1] for r in flat]), size=np.array([r[2] for r in flat]),
                  truth=np.array([r[3] for r in flat]), threshold=np.array([r[4] for r in flat]))
    cost = {n: np.array([r[5][n] for r in flat], dtype=np.int64) for n in names}
    checks = {}
    fr = pdata.frozen(args.view, args.split)
    if fr is not None:
        rows = np.array([fr["key"][(f, r)] for f, r in zip(arrays["family"], arrays["rho"])])
        if not np.array_equal(fr["truth"][rows], arrays["truth"]):
            raise AssertionError("truth differs from the frozen run")
        checks["weight_equals_frozen"] = bool(np.array_equal(fr["cost"][rows, fr["arms"]["weight"]], cost["weight"]))
        if not checks["weight_equals_frozen"]:
            raise AssertionError("weight order differs from the frozen run")
        for base in ("recal", "prop-recal"):
            stored = pdata.stored_cost(args.view, args.split, f"{args.prior}:{base}")
            if base in cost and stored is not None:
                prop = pdata.frozen_propagation(args.view, args.split) if base == "prop-recal" else None
                if prop is not None:
                    key = {(int(f), int(r)): i for i, (f, r) in enumerate(zip(prop["family"], prop["rho"]))}
                    stored = stored[[key[(f, r)] for f, r in zip(arrays["family"], arrays["rho"])]]
                else:
                    stored = stored[rows]
                same = bool(np.array_equal(stored, cost[base]))
                checks[f"{base}_equals_stored"] = same
                if not same:
                    raise AssertionError(f"recomputed {base} differs from the stored arm")
    stored_elsewhere = {n for n in cost if n != "weight" and checks.get(f"{n}_equals_stored")}
    keep = {n: v for n, v in cost.items() if n not in stored_elsewhere}  # bases without a stored copy are new arms
    arrays.update({f"cost__{n}": v for n, v in keep.items()})
    identity = pin.check()
    pdata.save_npz(out, **arrays)
    meta = dict(view=args.view, split=args.split, prior=args.prior, prior_identity=identity, arms=sorted(keep),
                recomputed_not_stored=sorted(set(cost) - set(keep)), sessions=len(flat), families=len(families), limit=args.limit,
                propagation=PROPAGATION, checks=checks, code=code, benchmark_sha256=pdata.hash_file(pdata.benchmark_path(args.view)),
                result_sha256=pdata.hash_file(out), wall_seconds=round(time.time() - started, 1), workers=args.workers)
    pdata.save_json(out.with_suffix(".json"), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "code"}))


if __name__ == "__main__":
    main()
