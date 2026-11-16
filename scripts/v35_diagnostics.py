#!/usr/bin/env python3
"""Descriptive diagnostics of the V35 benchmark inputs; no method is tuned here.

A. Why reference labels are unresolved (per view, presynaptic inputs of test targets).
B. How the earlier FlyWire annotation release relates to the reference release.
C. Development split only: which cells Adaptive Greedy, its recalibrated variant and
   the direction-fixed order pay for under the provisional prior, split by what the
   earlier release said about the cell.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as pq

from connectomequest.annotation_v32.data import ontology_type
from connectomequest.annotation_v35 import engine as e

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "outputs/v35/benchmark"
VIEWS = dict(
    malecns=dict(release="male-cns:v1.0", annotations="data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather"),
    flywire=dict(release="flywire:v3.1.0", annotations="inputs/v32/flywire/cq-flywire-v3.1.0.tsv"),
)
RHOS = ((1, 4), (1, 2), (3, 4), (1, 1))


def annotation_rows(view):
    cfg = VIEWS[view]
    if view == "flywire":
        with (ROOT / cfg["annotations"]).open(newline="") as stream:
            return {int(r["root_id"]): (r["cell_type"], r["hemibrain_type"]) for r in csv.DictReader(stream, delimiter="\t")}
    table = feather.read_table(ROOT / cfg["annotations"], columns=["bodyId", "type", "hemibrainType"])
    return {int(r["bodyId"]): (r["type"], r["hemibrainType"]) for r in table.to_pylist()}


def unresolved_reason(primary, secondary, release, lookup):
    names = [ontology_type(x.strip(), release) for x in (secondary or "").split(",") if x.strip()]
    if len(names) > 1:
        return "several candidate hemibrain types"
    if len(names) == 1:
        return "correspondence outside the vocabulary"
    if primary:
        return "release-specific type without correspondence"
    return "no type in the release"


def split_masks(bench, manifest, split):
    codes = {t["code"] for t in manifest["types"] if t["split"] == split}
    return np.isin(bench["target_type"], list(codes))


def part_a_b(view, manifest, lookup):
    bench = np.load(BENCH / f"{view}.npz", allow_pickle=False)
    rows = annotation_rows(view)
    targets = np.flatnonzero(split_masks(bench, manifest, "test"))
    offsets = bench["target_offset"]
    index = np.concatenate([np.arange(offsets[t], offsets[t + 1]) for t in targets])
    cells, labels, weights = bench["pre_cell"][index], bench["pre_label"][index], bench["pre_weight"][index]
    reasons, reason_weight = {}, {}
    cache = {}
    for cell, label, weight in zip(cells.tolist(), labels.tolist(), weights.tolist()):
        if label >= 0:
            continue
        if cell not in cache:
            cache[cell] = unresolved_reason(*rows.get(cell, (None, None)), VIEWS[view]["release"], lookup)
        reasons[cache[cell]] = reasons.get(cache[cell], 0) + 1
        reason_weight[cache[cell]] = reason_weight.get(cache[cell], 0) + weight
    unresolved = int((labels < 0).sum())
    out = dict(
        input_pairs=len(cells),
        unresolved_pairs=unresolved,
        unresolved_pair_share=unresolved / len(cells),
        unresolved_weight_share=float(weights[labels < 0].sum() / weights.sum()),
        reasons={k: dict(pairs=v, share_of_unresolved=v / unresolved, weight_share_of_unresolved=reason_weight[k] / float(weights[labels < 0].sum())) for k, v in sorted(reasons.items())},
    )
    if view == "flywire":
        table = pq.read_table(ROOT / "inputs/v32/prepared-common-v1-hints/flywire/public/nodes.parquet", columns=["cell_id", "provisional"]).to_pylist()
        hints = {int(r["cell_id"]): (-1 if r["provisional"] is None else int(r["provisional"])) for r in table}
        hint = np.array([hints[c] for c in cells.tolist()])
        typed_hint, resolved = hint > 0, labels >= 0
        out["provisional"] = dict(
            share_without_any_hint=float((hint < 0).mean()),
            share_hint_outside_vocabulary=float((hint == 0).mean()),
            share_hint_typed=float(typed_hint.mean()),
            agreement_when_both_typed=float((hint[typed_hint & resolved] == labels[typed_hint & resolved]).mean()),
            reference_unresolved_given_typed_hint=float((~resolved[typed_hint]).mean()),
            reference_resolved_given_no_typed_hint=float(resolved[~typed_hint].mean()),
            typed_hint_given_reference_resolved=float(typed_hint[resolved].mean()),
            hint_equals_reference_overall=float(((hint == labels) & resolved).mean()),
        )
    return out


def part_c(manifest, sample=700):
    view = "flywire"
    bench = np.load(BENCH / f"{view}.npz", allow_pickle=False)
    table = pq.read_table(ROOT / "inputs/v32/prepared-common-v1-hints/flywire/public/nodes.parquet", columns=["cell_id", "provisional"]).to_pylist()
    hints = {int(r["cell_id"]): (-1 if r["provisional"] is None else int(r["provisional"])) for r in table}
    development = split_masks(bench, manifest, "development")
    families = np.flatnonzero(development[bench["fam_target"]])
    rng = np.random.default_rng(35)
    families = np.sort(rng.choice(families, size=min(sample, len(families)), replace=False))
    lengths = np.diff(bench["target_offset"])[bench["fam_target"]]
    plus_offset = np.concatenate(([0], np.cumsum(lengths)))
    classes = ("hint in the partner set", "hint: another type", "hint outside the vocabulary", "no hint")
    outcomes_names = ("member", "nonmember", "unresolved")
    policies = ("direction", "ag", "recal")
    counts = {p: np.zeros((len(classes), 3)) for p in policies}
    cost = {p: 0 for p in policies}
    cost["weight"] = 0
    sessions = 0
    mean_p = {p: np.zeros((len(classes), 3)) for p in ("prior",)}
    seen = np.zeros(len(classes))
    for f in families.tolist():
        t = int(bench["fam_target"][f])
        lo, hi = int(bench["target_offset"][t]), int(bench["target_offset"][t + 1])
        cells, weights, labels = bench["pre_cell"][lo:hi], bench["pre_weight"][lo:hi], bench["pre_label"][lo:hi]
        codes = bench["fam_codes"][bench["fam_code_offset"][f] : bench["fam_code_offset"][f + 1]]
        outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
        hint = np.array([hints[c] for c in cells.tolist()])
        cls = np.where(hint < 0, 3, np.where(hint == 0, 2, np.where(np.isin(hint, codes), 0, 1)))
        plus = np.clip(bench["plus_provisional"][plus_offset[f] : plus_offset[f] + hi - lo].astype(float), 0, 1)
        unknown = np.clip(bench["unknown_provisional"][lo:hi].astype(float), 0, 1 - plus)
        prob = np.column_stack((plus, 1 - plus - unknown, unknown))
        total = int(weights.sum())
        for num, den in RHOS:
            threshold, _ = e.threshold_goals(total, num * int(bench["fam_source_weight"][f]), den * int(bench["fam_source_total"][f]))
            inst = e.threshold_instance(cells, weights, threshold)
            if e.full_information_status(inst, outcomes) != "positive":
                continue
            sessions += 1
            np.add.at(mean_p["prior"], cls, prob)
            np.add.at(seen, cls, 1)
            cost["weight"] += e.evaluate(inst, outcomes, e.weight_order(inst))[0]
            orders = dict(
                direction=e.direction_order(inst, prob),
                ag=e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, prob)),
                recal=e.run_adaptive(inst, outcomes, e.RecalibratedGreedy(inst, prob)),
            )
            for name, order in orders.items():
                c, _ = e.evaluate(inst, outcomes, order)
                paid = np.asarray(order[:c], dtype=np.intp)
                np.add.at(counts[name], (cls[paid], outcomes[paid]), 1)
                cost[name] += c
    out = dict(sampled_families=len(families), positive_sessions=sessions, cost_ratio_to_weight={p: cost[p] / cost["weight"] for p in policies})
    out["input_composition"] = {classes[i]: dict(share=float(seen[i] / seen.sum()), mean_prior=dict(zip(outcomes_names, (mean_p["prior"][i] / max(seen[i], 1)).tolist()))) for i in range(len(classes))}
    out["paid"] = {
        p: {classes[i]: dict(zip(outcomes_names, (counts[p][i] / counts[p].sum()).tolist())) for i in range(len(classes))} for p in policies
    }
    out["paid_class_share"] = {p: {classes[i]: float(counts[p][i].sum() / counts[p].sum()) for i in range(len(classes))} for p in policies}
    return out


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text())
    vocabulary = json.loads((ROOT / "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json").read_text())["vocabulary"]
    lookup = {name: i for i, name in enumerate(vocabulary) if i}
    result = dict(unresolved={view: part_a_b(view, manifest, lookup) for view in VIEWS}, provisional_payments_development=part_c(manifest))
    path = ROOT / "outputs/v35/analysis/diagnostics.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
