#!/usr/bin/env python3
"""A kit for timing expert decisions: the one measurement of the paper's cost unit that needs a person.

The paper counts one unit of cost per expert decision on the type of an input cell and reports no time per decision.
This script prepares a sheet with which an annotator can time such decisions, and summarizes the filled sheet.

  sample      draws 30 decisions from development questions (15 per connectome; a third each with the reference
              outcome member, nonmember and unresolved; cells among the twenty heaviest inputs of a retention
              question at one half), and writes timing/sheet.csv for the annotator and timing/key.csv with the
              reference outcomes, which the annotator should not open
  summarize   reads the filled sheet and writes outputs/timing/summary.json: seconds per decision (median and
              quartiles, overall and by decision) and the agreement with the reference

A decision: open the input cell next to the hemibrain types of the partner set and record whether it belongs to one
of them (member), to another hemibrain type (nonmember), or cannot be matched (unresolved), and the seconds from
opening the cell to the decision. Held-out questions are not used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

BENCH = ROOT / "outputs/v35/benchmark"
KIT = ROOT / "timing"
OUT = ROOT / "outputs/timing"
SEED = 20261114
PER_OUTCOME = 5
HEAVIEST = 20
OUTCOMES = ("member", "nonmember", "unresolved")
SITE = dict(flywire="https://codex.flywire.ai/app/cell_details?data_version=783&root_id={}",
            malecns="https://neuprint.janelia.org/view?dataset=male-cns%3Av1.0&bodyid={}")
FIELDS = ("row", "connectome", "target_cell", "target_link", "target_hemibrain_type", "partner_set", "input_cell", "input_link",
          "synapses_onto_target", "release_type_of_input_cell", "decision", "seconds", "notes")


def sample(args):
    import labelings as lab
    from connectomequest.annotation_v35 import engine as e

    if (KIT / "sheet.csv").exists() and not args.force:
        raise SystemExit("timing/sheet.csv exists (use --force to draw it again; a filled sheet would be lost)")
    manifest = json.loads((BENCH / "manifest.json").read_text())
    by_code = {t["code"]: t for t in manifest["types"] if t["split"] == "development"}
    tables = lab.Tables()
    rng = np.random.default_rng(SEED)
    sheet, key = [], []
    for view in lab.VIEWS:
        with np.load(BENCH / f"{view}.npz", allow_pickle=False) as z:
            bench = {k: z[k] for k in ("target_type", "target_cell", "target_offset", "pre_cell", "pre_weight", "pre_label", "fam_target",
                                       "fam_codes", "fam_code_offset", "fam_source_weight", "fam_source_total") if k in z.files}
        if "target_cell" not in bench:
            raise SystemExit("the benchmark does not store the identifiers of the target neurons")
        families = lab.base.split_families(bench, "development")
        pool = {name: [] for name in OUTCOMES}
        for f in families.tolist():
            t = int(bench["fam_target"][f])
            lo, hi = int(bench["target_offset"][t]), int(bench["target_offset"][t + 1])
            weights, labels = bench["pre_weight"][lo:hi], bench["pre_label"][lo:hi]
            codes = bench["fam_codes"][bench["fam_code_offset"][f]:bench["fam_code_offset"][f + 1]]
            outcome = lab.outcomes(labels, codes)
            for i in np.argsort(-weights, kind="stable")[:HEAVIEST].tolist():
                pool[("unresolved", "member", "nonmember")[int(outcome[i])]].append((f, t, lo + i))
        for name in OUTCOMES:
            picks = rng.choice(len(pool[name]), size=PER_OUTCOME, replace=False)
            for f, t, i in (pool[name][j] for j in sorted(picks.tolist())):
                kind = by_code[int(bench["target_type"][t])]
                codes = bench["fam_codes"][bench["fam_code_offset"][f]:bench["fam_code_offset"][f + 1]].tolist()
                cell, target = int(bench["pre_cell"][i]), int(bench["target_cell"][t])
                sheet.append(dict(connectome=view, target_cell=target, target_link=SITE[view].format(target), target_hemibrain_type=kind["name"],
                                  partner_set="; ".join(tables.vocabulary[c] for c in codes), input_cell=cell, input_link=SITE[view].format(cell),
                                  synapses_onto_target=int(bench["pre_weight"][i]),
                                  release_type_of_input_cell=tables.rows[view].get(cell, (None,))[0] or "", decision="", seconds="", notes=""))
                key.append(dict(connectome=view, target_cell=target, input_cell=cell, reference_outcome=name))
    order = rng.permutation(len(sheet)).tolist()
    KIT.mkdir(exist_ok=True)
    for name, rows, fields in (("sheet.csv", sheet, FIELDS), ("key.csv", key, ("row", "connectome", "target_cell", "input_cell", "reference_outcome"))):
        with (KIT / name).open("w", newline="") as stream:
            table = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            table.writeheader()
            for row, j in enumerate(order, 1):
                table.writerow(dict(rows[j], row=row))
    print(f"wrote {len(sheet)} decisions to timing/sheet.csv and the reference outcomes to timing/key.csv")


def summarize(args):
    with (KIT / "sheet.csv").open(newline="") as stream:
        sheet = list(csv.DictReader(stream))
    with (KIT / "key.csv").open(newline="") as stream:
        key = {r["row"]: r["reference_outcome"] for r in csv.DictReader(stream)}
    done = [r for r in sheet if r["decision"].strip() and r["seconds"].strip()]
    if not done:
        raise SystemExit("timing/sheet.csv has no filled row: nothing was measured")
    unknown = sorted({r["decision"].strip() for r in done} - set(OUTCOMES))
    if unknown:
        raise SystemExit(f"decisions must be one of {OUTCOMES}; found {unknown}")
    seconds = np.array([float(r["seconds"]) for r in done])
    quartiles = lambda x: dict(n=int(len(x)), q25=float(np.quantile(x, 0.25)), median=float(np.median(x)), q75=float(np.quantile(x, 0.75)))
    result = dict(decisions=len(done), of=len(sheet), seconds=quartiles(seconds), total_minutes=float(seconds.sum() / 60),
                  by_decision={name: quartiles(seconds[[r["decision"].strip() == name for r in done]])
                               for name in OUTCOMES if any(r["decision"].strip() == name for r in done)},
                  agreement_with_reference=float(np.mean([r["decision"].strip() == key[r["row"]] for r in done])),
                  note="one annotator, development questions; decisions timed from opening the cell to the recorded decision")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--force", action="store_true")
    p.set_defaults(call=sample)
    sub.add_parser("summarize").set_defaults(call=summarize)
    args = parser.parse_args()
    args.call(args)


if __name__ == "__main__":
    main()
