#!/usr/bin/env python3
"""Summarize a BANC run (round 5, workstream ``banc``): composition and cost ratios with 95% intervals.

Ratios are totals of cells checked over certifiable questions relative to weight order, with the
type-clustered percentile bootstrap of the paper (2,000 resamples of cell types, seed 20261112).
The same summary is computed from the frozen MaleCNS and FlyWire runs of the same split for
comparison. The held-out split is refused unless the round-5 record lists this script, the round-5
modules it imports and the benchmark files; existing reports are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from connectomequest.annotation_v32.runtime import hash_file
from connectomequest.round5 import banc as b

ROOT = b.ROOT
BOOTSTRAP, SEED = 2000, 20261112
RATIOS = (("clairvoyant", "weight"), ("mlp_s0:ag", "weight"), ("mlp_s0:recal", "weight"), ("mlp_s0:recal", "mlp_s0:ag"))


def load(view, split):
    if view == b.VIEW:
        d = dict(b.frozen(split))
        bench = b.benchmark()
    else:
        meta = json.loads((ROOT / f"outputs/v35/runs/{view}-{split}.json").read_text())
        with np.load(ROOT / f"outputs/v35/runs/{view}-{split}.npz", allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["arms"] = {n: i for i, n in enumerate(meta["arms"])}
        with np.load(ROOT / f"outputs/v35/benchmark/{view}.npz", allow_pickle=False) as z:
            bench = {k: z[k] for k in ("fam_target", "target_type", "target_offset")}
    d["type"] = bench["target_type"][bench["fam_target"][d["family"]]]
    d["n_targets"] = len(np.unique(bench["fam_target"][d["family"]]))
    return d


def column(d, name):
    if name == "clairvoyant":
        return d["clairvoyant"].astype(float)
    return d["cost"][:, d["arms"][name]].astype(float)


def bootstrap_ratio(x, y, clusters, mask):
    _, index = np.unique(clusters, return_inverse=True)
    k = int(index.max()) + 1
    rng = np.random.default_rng(SEED)
    counts = np.stack([np.bincount(row, minlength=k) for row in rng.integers(0, k, size=(BOOTSTRAP, k))]).astype(float)
    sx = np.bincount(index[mask], weights=x[mask], minlength=k)
    sy = np.bincount(index[mask], weights=y[mask], minlength=k)
    if sy.sum() <= 0:
        raise ValueError("no reference cost in the selected sessions")
    den = counts @ sy
    boot = (counts @ sx)[den > 0] / den[den > 0]
    return dict(value=float(sx.sum() / sy.sum()), low=float(np.quantile(boot, 0.025)), high=float(np.quantile(boot, 0.975)))


def summarize(view, split):
    d = load(view, split)
    cert = d["truth"] > 0
    w = column(d, "weight")
    out = dict(
        types=int(len(np.unique(d["type"]))), targets=int(d["n_targets"]), families=int(len(np.unique(d["family"]))),
        questions=int(len(d["truth"])), certifiable=int(cert.sum()),
        uncertifiable_share=float((d["truth"] == 0).mean()), yes_share=float((d["truth"] == 1).mean()), no_share=float((d["truth"] == 2).mean()),
        unresolved_cells_share=float(d["unresolved_cells"].sum() / d["size"].sum()),
        unresolved_synapses_share=float(d["unresolved"].sum() / d["total"].sum()),
        median_input_cells=float(np.median(d["size"])), questions_500_plus=int((d["size"] >= 500).sum()),
        weight_median_checks=float(np.median(w[cert])), weight_mean_fraction_read=float((w[cert] / d["size"][cert]).mean()),
        ratios={f"{x}/{y}": bootstrap_ratio(column(d, x), column(d, y), d["type"], cert) for x, y in RATIOS},
    )
    for name in ("const_half:recal", "const_share:ag"):
        if name in d["arms"]:
            out[f"{name}==weight"] = float((column(d, name) == w).mean())
    return out


def auroc(score, member):
    from scipy.stats import rankdata

    pos = int(member.sum())
    neg = len(member) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[member].sum() - pos * (pos + 1) / 2) / (pos * neg))


def prior_quality(view):
    """Within-question AUROC of the mlp_s0 membership probability (development families with both classes)."""
    from connectomequest.neurosym import common as c

    families, get = (b.split_families("development"), b.family) if view == b.VIEW else (c.development_families(view), lambda f: c.family(view, f))
    values = []
    for f in families:
        fam = get(int(f))
        a = auroc(fam["priors"]["mlp_s0"][:, 0], fam["outcomes"] == 0)
        if a is not None:
            values.append(a)
    values = np.asarray(values)
    return dict(families=int(len(values)), mean=float(values.mean()), sd=float(values.std()), at_least_0_8=float((values >= 0.8).mean()),
                below_0_5=float((values < 0.5).mean()))


def strata(view, split):
    d = load(view, split)
    cert, w = d["truth"] > 0, column(d, "weight")
    out = {}
    for lo, hi in ((10, 30), (30, 100), (100, 500), (500, None)):
        m = cert & (d["size"] >= lo) & (d["size"] < (hi or np.inf))
        if m.sum():
            out[f"{lo}-{hi or 'max'}"] = dict(n=int(m.sum()), **{f"{x}/weight": float(column(d, x)[m].sum() / w[m].sum())
                                                                 for x in ("clairvoyant", "mlp_s0:ag", "mlp_s0:recal")})
    return out


def crosswalk_variants():
    """Development only: certifiability of BANC questions under two readings the declared rule does not use."""
    from connectomequest.annotation_v35 import engine as e

    v = b.view_data()
    names = v["names"]
    lookup = {n: i for i, n in enumerate(b.vocabulary()) if i}
    strip = lambda s: None if s is None else ",".join(x.strip()[5:] if x.strip().startswith("auto:") else x for x in s.split(","))
    release = [names["release"][k] if k >= 0 else None for k in v["release_code"]]
    declared = [names["declared"][k] if k >= 0 else None for k in v["declared_code"]]
    variants = dict(
        declared=v["labels"],
        chained_flywire=np.where(v["labels"] >= 0, v["labels"], v["chained_flywire"]),
        auto_stripped=np.array([b.crosswalk(r, strip(s), lookup)[0] for r, s in zip(release, declared)], dtype=np.int32),
    )
    out = {}
    for name, labels in variants.items():
        cells = resolved = uncert = questions = 0
        for f in b.split_families("development"):
            fam = b.family(int(f))
            lab = labels[b.rows_of(fam["cells"])]
            if name == "declared" and not np.array_equal(lab, fam["labels"]):
                raise AssertionError("view labels differ from the benchmark labels")
            y = np.where(lab < 0, e.UNRESOLVED, np.where(np.isin(lab, fam["codes"]), e.MEMBER, e.NONMEMBER))
            cells += len(lab)
            resolved += int((lab >= 0).sum())
            for _, threshold in b.thresholds(fam):
                inst = e.threshold_instance(fam["cells"], fam["weights"], threshold)
                questions += 1
                uncert += e.full_information_status(inst, y) == "unresolved"
        out[name] = dict(neurons_resolved=int((labels >= 0).sum()), input_cells_resolved=resolved / cells, uncertifiable_share=uncert / questions)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="development", choices=b.SPLITS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = b.RUNS / f"{b.VIEW}-{args.split}.npz"
    b.guard(args.split, __file__, extra=(b.BENCH, b.MANIFEST))  # the run file is checked against its own json by b.frozen
    output = args.output or b.OUT / f"report-{args.split}.json"
    if output.exists():
        raise FileExistsError(output)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "banc", "report", output)
    report = dict(split=args.split, banc=summarize(b.VIEW, args.split),
                  frozen={view: summarize(view, args.split) for view in ("malecns", "flywire")},
                  inputs={str(p.relative_to(ROOT)): hash_file(p) for p in (run, b.BENCH)})
    report["strata"] = {view: strata(view, args.split) for view in (b.VIEW, "malecns", "flywire")}
    if args.split == "development":  # diagnostics read answers of development questions only
        report["prior_quality_development"] = {view: prior_quality(view) for view in (b.VIEW, "malecns", "flywire")}
        report["crosswalk_variants_development"] = crosswalk_variants()
    b.save_new(output, lambda s: s.write((json.dumps(report, indent=1, sort_keys=True) + "\n").encode()))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
