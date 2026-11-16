#!/usr/bin/env python3
"""Control priors for the seeded scenario: confirmed seeds without the new predictor.

``seedmlp<ff>-<view>.npz`` (in outputs/round5/priors/controls): seeds of fraction f keep 0.99 on their
confirmed label (+ 0.01 x the topology prior), every other cell keeps the frozen topology prior
``mlp_s0`` unchanged. Against ``seeded<ff>`` it separates what the confirmed labels save by themselves
from what the in-connectome predictor adds for the other cells. Same layout and seed sets as the
seeded priors; never overwrites.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from connectomequest.neurosym import common as nc
from connectomequest.round5 import priors_common as pc
from connectomequest.round5 import priors_data as pdata
from connectomequest.round5 import priors_layout as pl
from connectomequest.round5 import priors_model as pm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=pc.VIEWS)
    parser.add_argument("--fraction", required=True, type=float, choices=pc.FRACTIONS)
    parser.add_argument("--output", type=Path, default=pc.OUT / "controls")
    args = parser.parse_args()
    name = f"seedmlp{pc.fraction_tag(args.fraction)}"
    out = args.output / f"{name}-{args.view}.npz"
    if out.exists():
        raise FileExistsError(out)
    s = pm.seeds(args.view, args.fraction)
    mlp = np.load(pc.ROOT / f"outputs/neurosym/cache/mlp-full-{args.view}.npy", mmap_mode="r")
    mass = lambda rows, codes: np.asarray(mlp[rows])[:, codes].sum(1)
    plus, unknown, seed = pl.benchmark_arrays(args.view, mass, np.asarray(mlp[:, 0]), s.is_seed, pdata.view_labels(args.view), pm.SEED_CONFIDENCE)
    d = nc.benchmark(args.view)
    nonseed_plus = ~seed[np.concatenate([np.arange(d["target_offset"][t], d["target_offset"][t + 1]) for t in d["fam_target"]])]
    if not np.allclose(plus[nonseed_plus], d["plus_mlp_s0"][nonseed_plus], atol=1e-5) or not np.allclose(unknown[~seed], d["unknown_mlp_s0"][~seed], atol=1e-6):
        raise ArithmeticError("non-seed cells must keep the frozen topology prior")
    pc.save_npz_new(out, **{"plus_" + name: plus, "unknown_" + name: unknown, "seed_" + name: seed,
                            "pre_cell": d["pre_cell"], "fam_target": d["fam_target"], "target_offset": d["target_offset"]})
    pc.write_json_new(out.with_suffix(".json"), dict(name=name, view=args.view, fraction=args.fraction, seed_confidence=pm.SEED_CONFIDENCE,
                                                    code=pc.code_identity(__file__), result_sha256=pc.hash_file(out)))
    print(json.dumps(dict(name=name, view=args.view, seeds_in_inputs=float(seed.mean()))))


if __name__ == "__main__":
    main()
