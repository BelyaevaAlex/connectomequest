#!/usr/bin/env python3
"""Round 5, workstream ``policies``: betting tests that sample without replacement.

For every question of one split this runs, five times each,
* ``wor``: weight-proportional sampling without replacement, two one-sided betting tests
  with the conditional (without-replacement) null mean (``round5.policies_wor``);
* ``worhyb``: the hybrid variant - the weight-order prefix carrying half of ``W`` checked
  exactly, the rest sampled without replacement;
with alpha = 0.05 per claim and the exact-bound early stop. The with-replacement tests
are not rerun: their per-question costs are the frozen ``stat_r*`` arms and the first
post hoc round's ``hybrid_cost``.

``--adversarial`` adds the error-rate audit of these two tests (not of the frozen
with-replacement ones) on the same question
families: thresholds just beyond the truth, ``T = L* + 1`` (a *yes* would be wrong) and
``T = U*`` (a *no* would be wrong), ``--repeats`` audits per threshold and test. These
thresholds are not questions of the benchmark; the audit only counts wrong certificates.

The held-out split is refused unless ``outputs/round5/record.json`` lists the sha256 of
this runner and of every round-5 module it imports; outputs are never overwritten.
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
from connectomequest.round5 import policies_data as pdata  # noqa: E402
from connectomequest.round5 import policies_wor as wor  # noqa: E402

ALPHA = 0.05
REPEATS = 5
TESTS = {"wor": (wor.wor_certificate, 31), "worhyb": (wor.wor_hybrid_certificate, 37)}
_RUN = {}


def family_job(f):
    view = _RUN["view"]
    fam = pdata.family(view, f)
    y = fam["outcomes"]
    rows = []
    for rho, threshold in pdata.thresholds(fam):
        inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
        truth = e.full_information_status(inst, y)
        out = dict(family=int(f), rho=rho, size=inst.size, truth=pdata.STATUS[truth], weight=e.evaluate(inst, y, e.weight_order(inst))[0])
        for name, (fn, salt) in TESTS.items():
            cost, status, exact = [], [], []
            for r in range(REPEATS):
                c, s, x = fn(inst, y, alpha=ALPHA, rng=np.random.default_rng([salt, int(f), rho, r]))
                if x and s != truth:
                    raise AssertionError("exact certificate contradicts the truth")
                cost.append(c)
                status.append(pdata.STATUS[s])
                exact.append(x)
            out[f"{name}_cost"], out[f"{name}_status"], out[f"{name}_exact"] = cost, status, exact
        rows.append(out)
    adversarial = None
    if _RUN["adversarial"]:
        w = fam["weights"].astype(np.int64)
        total = int(w.sum())
        member = int(w[y == e.MEMBER].sum())
        upper = total - int(w[y == e.NONMEMBER].sum())
        adversarial = []
        for kind, threshold in (("yes", member + 1), ("no", upper)):
            if not 1 <= threshold <= total:
                continue
            inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
            truth = e.full_information_status(inst, y)
            for name, (fn, salt) in TESTS.items():
                wrong_yes = wrong_no = stat = 0
                for r in range(_RUN["repeats"]):
                    c, s, x = fn(inst, y, alpha=ALPHA, rng=np.random.default_rng([salt + 100, int(f), 0 if kind == "yes" else 1, r]))
                    wrong_yes += s == "positive" and truth != "positive"
                    wrong_no += s == "negative" and truth != "negative"
                    stat += not x
                adversarial.append(dict(family=int(f), kind=kind, test=name, truth=pdata.STATUS[truth], size=inst.size, repeats=_RUN["repeats"],
                                        wrong_yes=int(wrong_yes), wrong_no=int(wrong_no), statistical=int(stat)))
    return rows, adversarial


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True)
    parser.add_argument("--split", required=True, choices=("development", "test"))
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--repeats", type=int, default=10, help="audits per adversarial threshold and test")
    parser.add_argument("--output", type=Path, default=pdata.OUT / "wor")
    parser.add_argument("--workers", type=int, default=pdata.workers())
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.split == "test" and args.limit is not None:
        raise SystemExit("the held-out split runs completely")
    output = pdata.default_output(args.output, pdata.OUT / "wor", args.split)
    pdata.preload(args.view)
    pdata.benchmark(args.view)
    code = pdata.require_record(__file__, args.split) or pdata.code_identity(__file__)
    out = pdata.fresh(output / f"{args.view}-{args.split}.npz")
    pdata.fresh(out.with_suffix(".json"))
    if args.split == "test":  # the round-5 one-shot gate (after every import and the output check)
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "policies", f"wor-{args.view}", out)
    families = pdata.split_families(args.view, args.split)
    if args.limit is not None:
        families = families[: args.limit]
    _RUN.update(view=args.view, adversarial=args.adversarial, repeats=args.repeats)
    pdata.benchmark(args.view)
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(family_job, list(map(int, families)), chunksize=2)
    rows = [r for group, _ in results for r in group]
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    fr = pdata.frozen(args.view, args.split)
    checks = {}
    if fr is not None:
        index = np.array([fr["key"][(int(f), int(r))] for f, r in zip(arrays["family"], arrays["rho"])])
        checks["truth_equals_frozen"] = bool(np.array_equal(fr["truth"][index], arrays["truth"]))
        checks["weight_equals_frozen"] = bool(np.array_equal(fr["cost"][index, fr["arms"]["weight"]], arrays["weight"]))
        if not all(checks.values()):
            raise AssertionError(checks)
    if args.adversarial:
        adv = [a for _, group in results for a in (group or [])]
        for key in adv[0]:
            arrays[f"adv_{key}"] = np.asarray([a[key] for a in adv])
    pdata.save_npz(out, **arrays)
    meta = dict(view=args.view, split=args.split, alpha=ALPHA, repeats=REPEATS, tests=list(TESTS), adversarial=args.adversarial,
                adversarial_repeats=args.repeats if args.adversarial else None, sessions=len(rows), families=len(families), limit=args.limit,
                checks=checks, code=code, benchmark_sha256=pdata.hash_file(pdata.benchmark_path(args.view)), result_sha256=pdata.hash_file(out),
                wall_seconds=round(time.time() - started, 1), workers=args.workers)
    pdata.save_json(out.with_suffix(".json"), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "code"}))


if __name__ == "__main__":
    main()
