#!/usr/bin/env python3
"""Run the sanity arms on one split of the BANC view (round 5, workstream ``banc``).

Arms: weight order; Adaptive Greedy and recalibrated AG with the transferred topology prior
(``mlp_s0``); recalibrated AG with the coin prior and AG with the share prior (no prediction:
both must reproduce weight order). Every question also records the clairvoyant bound. Sessions,
thresholds and the exact verifier are those of ``scripts/v35_run.py`` (frozen engine); the output
has the layout of ``outputs/v35/runs/{view}-{split}.npz`` / ``.json``.

The held-out split (``--split test``) is refused unless ``outputs/round5/record.json`` lists the
sha256 of this script, of every round-5 module it imports and of the BANC benchmark and manifest.
Existing results are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v35 import engine as e
from connectomequest.round5 import banc as b

ROOT = b.ROOT
ARMS = ("weight", "mlp_s0:ag", "mlp_s0:recal", "const_half:recal", "const_share:ag")
STATUS = {"unresolved": 0, "positive": 1, "negative": 2}
MAX_WORKERS = 6
CODE_FILES = ("scripts/r5_banc_run.py", "src/connectomequest/round5/banc.py", "src/connectomequest/annotation_v35/engine.py")


def orders(name, inst, outcomes, prior):
    if name == "weight":
        return e.weight_order(inst)
    if name.endswith(":ag"):
        return e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, prior))
    if name.endswith(":recal"):
        return e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, prior))
    raise ValueError(name)


def run_family(f):
    fam = b.family(f)
    cells, weights, outcomes = fam["cells"], fam["weights"], fam["outcomes"]
    total = int(weights.sum())
    rows = []
    for rho, threshold in b.thresholds(fam):
        inst = e.threshold_instance(cells, weights, threshold)
        truth = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, outcomes)
        cost = np.zeros(len(ARMS), dtype=np.int32)
        status = np.zeros(len(ARMS), dtype=np.int8)
        paid_member = np.zeros(len(ARMS), dtype=np.int32)
        paid_unresolved = np.zeros(len(ARMS), dtype=np.int32)
        for k, name in enumerate(ARMS):
            prior = None if name == "weight" else fam["priors"][name.split(":")[0]]
            order = orders(name, inst, outcomes, prior)
            c, s = e.evaluate(inst, outcomes, order)
            if s != truth or (truth == "unresolved" and c != inst.size) or (bound is not None and c < bound):
                raise AssertionError(f"{name}: arm contradicts the exact full-information status")
            paid = outcomes[np.asarray(order[:c], dtype=np.intp)]
            cost[k], status[k] = c, STATUS[s]
            paid_member[k], paid_unresolved[k] = int((paid == e.MEMBER).sum()), int((paid == e.UNRESOLVED).sum())
        rows.append(dict(family=int(f), rho=rho, size=inst.size, total=total, threshold=threshold,
                         member=int(weights[outcomes == e.MEMBER].sum()), unresolved=int(weights[outcomes == e.UNRESOLVED].sum()),
                         unresolved_cells=int((outcomes == e.UNRESOLVED).sum()), truth=STATUS[truth],
                         clairvoyant=-1 if bound is None else bound, cost=cost, status=status,
                         paid_member=paid_member, paid_unresolved=paid_unresolved))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=b.SPLITS)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=b.RUNS)
    args = parser.parse_args()
    b.guard(args.split, __file__, extra=(b.BENCH, b.MANIFEST))
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"at most {MAX_WORKERS} local workers")
    out, meta_path = args.output / f"{b.VIEW}-{args.split}.npz", args.output / f"{b.VIEW}-{args.split}.json"
    for path in (out, meta_path):
        if path.exists():
            raise FileExistsError(path)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "banc", "run", out)
    families = b.split_families(args.split)
    b.benchmark()  # load and check before forking
    started = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_family, list(map(int, families)), chunksize=4)
    rows = [r for group in results for r in group]
    fields = ("family", "rho", "size", "total", "threshold", "member", "unresolved", "unresolved_cells", "truth", "clairvoyant")
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in fields}
    for k in ("cost", "status", "paid_member", "paid_unresolved"):
        arrays[k] = np.stack([r[k] for r in rows])
    b.save_new(out, lambda s: np.savez(s, **arrays))
    meta = dict(view=b.VIEW, split=args.split, arms=list(ARMS), rhos=b.RHOS, sessions=len(rows), families=len(families),
                code={name: hash_file(ROOT / name) for name in CODE_FILES}, benchmark_sha256=hash_file(b.BENCH),
                manifest_sha256=hash_file(b.MANIFEST), result_sha256=hash_file(out), wall_seconds=time.time() - started,
                workers=args.workers, argv=sys.argv[1:])
    b.save_new(meta_path, lambda s: s.write((json.dumps(meta, indent=1, sort_keys=True) + "\n").encode()))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("arms", "code")}))


if __name__ == "__main__":
    main()
