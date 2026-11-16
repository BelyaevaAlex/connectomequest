#!/usr/bin/env python3
"""Retrain the frozen hemibrain MLP recipe under additional initialization seeds.

Same features, split, architecture (12-64-5551 ReLU), Adam 0.01, batch 256, 40
epochs, earliest-best calibration checkpoint and temperature grid as the frozen
seed-zero model. Only the torch/minibatch seed changes. CPU only; no other
connectome is read. Seed zero is refit as a recipe check against the frozen report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from connectomequest.annotation_v32.predictors import fit_temperature, score_probabilities
from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v32.training import train_mlp

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/v32/predictors/59cd92072106eb705ce5ae288aa891a79ee47de74bbba3cc5dce30ae4fc38ea3"
FEATURES = ROOT / "outputs/v32/predictors/_features/9a056257416855aeccb501931445423fdda4d74de9e0e60553bfb86e29df1761"
SOURCE = ROOT / "inputs/v32/prepared-common-v1/hemibrain"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/mlp-seeds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads((MODEL / "split.json").read_text())
    with np.load(FEATURES / "features.npz", allow_pickle=False) as z:
        node_ids, features = z["node_ids"], z["features"]
    table = pq.read_table(SOURCE / "private/labels.parquet").to_pylist()
    labels = {int(r["cell_id"]): r["label"] for r in table}
    vocabulary = json.loads((SOURCE / "public/manifest.json").read_text())["vocabulary"]

    def subset(cells):
        rows = np.searchsorted(node_ids, cells)
        if not np.array_equal(node_ids[rows], cells):
            raise ValueError("split cell missing from the feature cache")
        return features[rows], np.array([labels[int(c)] for c in cells], dtype=np.int64)

    xt, yt = subset(np.asarray(split["train"]))
    xc, yc = subset(np.asarray(split["calibration"]))
    xa, ya = subset(np.asarray(split["audit"]))
    for seed in args.seeds:
        target = args.output / f"seed{seed}"
        if (target / "report.json").exists():
            print(json.dumps(json.loads((target / "report.json").read_text())["summary"]))
            continue
        model, report = train_mlp(xt, yt, xc, yc, label_count=len(vocabulary), seed=seed, epochs=40, device="cpu")
        temperature = fit_temperature(model.logits(xc), yc)
        probabilities = score_probabilities(model.logits(xa), temperature)
        summary = dict(
            seed=seed,
            temperature=temperature,
            best_epoch=report["best_epoch"],
            calibration_nll=min(report["calibration_nll_trajectory"]),
            audit_accuracy=float((probabilities.argmax(1) == ya).mean()),
            audit_nll=float(-np.log(np.maximum(probabilities[np.arange(len(ya)), ya], 1e-300)).mean()),
        )
        target.mkdir()
        model.save(target / "model.npz")
        (target / "report.json").write_text(
            json.dumps(
                dict(
                    summary=summary,
                    training=report,
                    split_sha256=hash_file(MODEL / "split.json"),
                    features_sha256=hash_file(FEATURES / "features.npz"),
                    recipe="frozen V32 MLP recipe; device cpu; only the seed differs",
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
