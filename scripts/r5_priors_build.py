#!/usr/bin/env python3
"""Build the seeded in-connectome priors of round 5 (workstream ``priors``).

For one connectome, seed fraction and variant this trains the seeded type predictor
(``connectomequest.round5.priors_model``) and writes, in the layout of the frozen benchmark,
``outputs/round5/priors/seeded<ff>[conn]-<view>.npz`` with ``plus_<name>`` (per question family and
input cell) and ``unknown_<name>`` (per input cell), where ``<name>`` is ``seeded<ff>`` or
``seeded<ff>conn`` (connectivity only: no soma position, no predicted transmitter). Seeds keep 0.99
on their confirmed label. Priors are computed for every neuron; they use the labels of seed neurons
only, never an answer of a non-seed cell. Quality statistics are computed on development question
families and on the held-aside calibration seeds only. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from connectomequest.neurosym import common as nc
from connectomequest.round5 import priors_common as pc
from connectomequest.round5 import priors_layout as pl
from connectomequest.round5 import priors_model as pm


def prior_name(fraction, conn_only):
    return f"seeded{pc.fraction_tag(fraction)}" + ("conn" if conn_only else "")


def development_input_rows(view, is_seed):
    d = nc.benchmark(view)
    targets = np.unique(d["fam_target"][nc.development_families(view)])
    cells = np.unique(np.concatenate([d["pre_cell"][d["target_offset"][t] : d["target_offset"][t + 1]] for t in targets]))
    rows = pl.node_rows(view, cells)
    return rows[~is_seed[rows]]


def development_entries(d, families):
    """Mask over ``pre_cell`` entries of the targets of the given families."""
    mask = np.zeros(len(d["pre_cell"]), dtype=bool)
    for t in np.unique(d["fam_target"][families]):
        mask[d["target_offset"][t] : d["target_offset"][t + 1]] = True
    return mask


def accuracy(top, label, rows):
    resolved = rows[label[rows] > 0]
    return dict(cells=int(len(rows)), top1=float((top[rows] == label[rows]).mean()), resolved_cells=int(len(resolved)),
                top1_resolved=float((top[resolved] == label[resolved]).mean()), outside_cells=int(len(rows) - len(resolved)),
                top1_outside=float((top[rows][label[rows] == 0] == 0).mean()) if len(rows) > len(resolved) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=pc.VIEWS)
    parser.add_argument("--fraction", required=True, type=float, choices=pc.FRACTIONS)
    parser.add_argument("--conn-only", action="store_true", help="drop soma position and predicted transmitter")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=pc.OUT)
    args = parser.parse_args()
    name = prior_name(args.fraction, args.conn_only)
    out = args.output / f"{name}-{args.view}.npz"
    meta_path = args.output / f"{name}-{args.view}.json"
    if out.exists() or meta_path.exists():
        raise FileExistsError(out)
    started = time.time()
    s, pred = pm.predict_view(args.view, args.fraction, args.conn_only, threads=args.threads)
    mlp_full = np.load(pc.ROOT / f"outputs/neurosym/cache/mlp-full-{args.view}.npy", mmap_mode="r")
    other = (pred.prob * (pred.cand == 0)).sum(1)
    plus, unknown, seed = pl.benchmark_arrays(args.view, lambda r, c: pm.class_mass(pred, r, c, mlp_full), other, s.is_seed, s.label, pm.SEED_CONFIDENCE)
    top, top_p = pm.top1(pred)
    d = nc.benchmark(args.view)
    rows = pl.node_rows(args.view, d["pre_cell"])
    dev_families = nc.development_families(args.view)
    dev_rows = development_input_rows(args.view, s.is_seed)
    calib = np.flatnonzero(s.is_calib)
    quality = dict(
        calibration_seeds=accuracy(top, s.label, calib),
        development_input_nonseed=dict(**accuracy(top, s.label, dev_rows), candidate_recall=float((pred.cand[dev_rows] == s.label[dev_rows][:, None]).any(1).mean())),
        development_questions_all_cells=pl.quality(args.view, plus, unknown, dev_families),
        development_questions_nonseed_cells=pl.quality(args.view, plus, unknown, dev_families, keep=~seed),
        development_seed_share_of_input_cells=float(seed[development_entries(d, dev_families)].mean()),
    )
    arrays = {
        "plus_" + name: plus,
        "unknown_" + name: unknown,
        "seed_" + name: seed,
        "top1_" + name: top[rows].astype(np.int32),
        "top1p_" + name: top_p[rows].astype(np.float32),
        "pre_cell": d["pre_cell"],
        "fam_target": d["fam_target"],
        "target_offset": d["target_offset"],
    }
    pc.save_npz_new(out, **arrays)
    meta = dict(
        name=name, view=args.view, fraction=args.fraction, conn_only=args.conn_only, seed_confidence=pm.SEED_CONFIDENCE,
        calibration_share=pc.CALIBRATION_SHARE, model=pred.report, quality=quality, code=pc.code_identity(__file__),
        benchmark_sha256=pc.hash_file(nc.BENCH / f"{args.view}.npz"), result_sha256=pc.hash_file(out), wall_seconds=time.time() - started,
        sources=dict(modalities=json.loads(str(np.load(pc.CACHE / f"priors-modalities-{args.view}.npz")["sources"]))),
    )
    pc.write_json_new(meta_path, meta)
    print(json.dumps(dict(name=name, view=args.view, quality=quality, model=pred.report, wall=meta["wall_seconds"]), indent=1))


if __name__ == "__main__":
    main()
