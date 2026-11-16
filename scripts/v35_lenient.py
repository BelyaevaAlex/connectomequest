#!/usr/bin/env python3
"""Post hoc sensitivity analysis: a lenient reference.

The declared reference returns "unresolved" for every input cell without a
hemibrain-vocabulary label. Most of these cells carry a release-specific type with
no author-declared hemibrain correspondence. This analysis asks what changes if
exactly those cells are read as nonmembers of every hemibrain partner set (cells
with several candidate hemibrain types, a correspondence outside the vocabulary or
no type stay unresolved). It was specified after the test split was analysed, was first
run on the development split and changes nothing in the frozen protocol. The held-out
split was added in the second round of review and needs the second post hoc record,
whose code hashes must match.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.data import ontology_type
from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.annotation_v35 import engine as e
from v35_diagnostics import VIEWS, annotation_rows, split_masks, unresolved_reason

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/v35/benchmark"
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))
BOOTSTRAP = 2000
CODE_FILES = ("src/connectomequest/annotation_v35/engine.py", "src/connectomequest/annotation_v32/data.py", "scripts/v35_diagnostics.py", "scripts/v35_lenient.py")
_D = {}


def run_family(f):
    d = _D
    t = int(d["fam_target"][f])
    lo, hi = int(d["target_offset"][t]), int(d["target_offset"][t + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    codes = d["fam_codes"][d["fam_code_offset"][f] : d["fam_code_offset"][f + 1]]
    strict = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    lenient = np.where((labels < 0) & d["release_specific"][lo:hi], e.NONMEMBER, strict)
    total = int(weights.sum())
    prob = {}
    for prior in d["priors"]:
        plus = np.clip(d["plus_" + prior][d["plus_offset"][f] : d["plus_offset"][f] + hi - lo].astype(float), 0, 1)
        unknown = np.clip(d["unknown_" + prior][lo:hi].astype(float), 0, 1 - plus)
        prob[prior] = np.column_stack((plus, 1 - plus - unknown, unknown))
    rows = []
    for num, den in RHOS:
        threshold, _ = e.threshold_goals(total, num * int(d["fam_source_weight"][f]), den * int(d["fam_source_total"][f]))
        inst = e.threshold_instance(cells, weights, threshold)
        row = dict(cluster=int(d["target_type"][t]), size=inst.size)
        for name, outcomes in (("strict", strict), ("lenient", lenient)):
            row[name] = e.full_information_status(inst, outcomes)
        bound = e.clairvoyant_cost(inst, lenient)
        cost = dict(weight=e.evaluate(inst, lenient, e.weight_order(inst))[0], clairvoyant=inst.size if bound is None else bound)
        for prior, p in prob.items():
            cost[prior + ":member"] = e.evaluate(inst, lenient, e.positive_order(inst, p))[0]
            cost[prior + ":direction"] = e.evaluate(inst, lenient, e.direction_order(inst, p))[0]
            cost[prior + ":ag"] = e.evaluate(inst, lenient, e.run_adaptive(inst, lenient, e.AdaptiveGreedy(inst, p)))[0]
            cost[prior + ":recal"] = e.evaluate(inst, lenient, e.run_adaptive(inst, lenient, e.RecalibratedGreedy(inst, p)))[0]
        row["cost"] = cost
        rows.append(row)
    return rows


def analyse(view, manifest, lookup, split):
    with np.load(BENCH / f"{view}.npz", allow_pickle=False) as z:
        _D.clear()
        _D.update({k: z[k] for k in z.files})
    rows = annotation_rows(view)
    reason = {}
    flags = np.zeros(len(_D["pre_cell"]), dtype=bool)
    for i, (cell, label) in enumerate(zip(_D["pre_cell"].tolist(), _D["pre_label"].tolist())):
        if label >= 0:
            continue
        if cell not in reason:
            reason[cell] = unresolved_reason(*rows.get(cell, (None, None)), VIEWS[view]["release"], lookup) == "release-specific type without correspondence"
        flags[i] = reason[cell]
    _D["release_specific"] = flags
    _D["priors"] = ["mlp_s0"] + (["provisional"] if view == "flywire" else [])
    lengths = np.diff(_D["target_offset"])[_D["fam_target"]]
    _D["plus_offset"] = np.concatenate(([0], np.cumsum(lengths)))
    selected = split_masks(_D, manifest, split)
    families = np.flatnonzero(selected[_D["fam_target"]]).tolist()
    with Pool(24) as pool:
        results = [row for group in pool.map(run_family, families, chunksize=4) for row in group]
    status = {name: np.array([r[name] for r in results]) for name in ("strict", "lenient")}
    cluster = np.unique(np.array([r["cluster"] for r in results]), return_inverse=True)[1]
    n_clusters = int(cluster.max()) + 1
    rng = np.random.default_rng(20261112)
    counts = np.stack([np.bincount(row, minlength=n_clusters) for row in rng.integers(0, n_clusters, size=(BOOTSTRAP, n_clusters))]).astype(float)
    certifiable = status["lenient"] != "unresolved"
    weight = np.array([r["cost"]["weight"] for r in results], dtype=float)

    def ratio(name):
        x = np.array([r["cost"][name] for r in results], dtype=float)
        sx = np.bincount(cluster[certifiable], weights=x[certifiable], minlength=n_clusters)
        sy = np.bincount(cluster[certifiable], weights=weight[certifiable], minlength=n_clusters)
        boot = (counts @ sx) / (counts @ sy)
        return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))

    return dict(
        sessions=len(results),
        families=len(families),
        answers={name: {k: float((s == k).mean()) for k in ("positive", "negative", "unresolved")} for name, s in status.items()},
        moved_from_unresolved=dict(
            to_positive=float(((status["strict"] == "unresolved") & (status["lenient"] == "positive")).sum() / max(1, (status["strict"] == "unresolved").sum())),
            to_negative=float(((status["strict"] == "unresolved") & (status["lenient"] == "negative")).sum() / max(1, (status["strict"] == "unresolved").sum())),
        ),
        contradictions=int(((status["strict"] != "unresolved") & (status["strict"] != status["lenient"])).sum()),
        ratio_to_weight={name: ratio(name) for name in results[0]["cost"] if name != "weight"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="development", choices=("development", "test"))
    parser.add_argument("--record", type=Path, help="second post hoc record; required for the held-out split")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v35/analysis")
    args = parser.parse_args()
    out = args.output / f"lenient-{args.split}.json"
    if args.split == "test":
        code = {name: hash_file(ROOT / name) for name in CODE_FILES}
        if args.record is None or json.loads(args.record.read_text())["code"]["lenient"] != code:
            raise ValueError("the held-out split needs the second post hoc record with matching code hashes")
        if out.exists():
            raise FileExistsError(out)
    manifest = json.loads((BENCH / "manifest.json").read_text())
    vocabulary = json.loads((ROOT / "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    result = {view: analyse(view, manifest, lookup, args.split) for view in VIEWS}
    args.output.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
