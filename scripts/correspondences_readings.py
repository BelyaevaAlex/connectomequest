#!/usr/bin/env python3
"""Both readings of a release-specific type (descriptive summary added after the held-out analysis; no expectations).

scripts/correspondences.py ranks release-specific types by the uncertifiable questions that are certified "no" when
the cells of a type are read as nonmembers, the reading of the lenient crosswalk. A declared correspondence can also
place the cells inside a partner set, which can certify "yes". This script reads the per-question arrays of that
analysis and counts, for every type alone, the questions certified under each reading:

  outside   the cells are nonmembers: certified "no" if U - w < T   (the count of the ranked list)
  inside    the cells are members:    certified "yes" if L + w >= T

For a type whose name extends the name of a hemibrain type by a suffix (the newer release splits the hemibrain type),
that hemibrain type is the candidate counterpart, and the script also counts the questions whose partner set contains
it: there the inside reading applies, elsewhere the outside reading.

It was written after the held-out analysis of scripts/correspondences.py had been seen, to correct the reading of
its first types; it runs no policy, fits nothing and has no recorded expected outcomes.

Usage: python scripts/correspondences_readings.py     (writes outputs/correspondences/readings.json)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/correspondences"
BENCH = ROOT / "outputs/v35/benchmark"
VOCABULARY = ROOT / "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json"
VIEWS = ("malecns", "flywire")
SPLITS = ("development", "test")
FIRST = 100


def certified_inside(lower, threshold, weight):
    """Reading the cells of the type as members certifies "yes"."""
    return lower + weight >= threshold


def certified_outside(upper, threshold, weight):
    """Reading the cells of the type as nonmembers certifies "no"."""
    return upper - weight < threshold


def split_parent(name, vocabulary):
    """The longest hemibrain type name that the name extends by a suffix that starts with a lower-case letter, _ or -."""
    for k in range(len(name) - 1, 1, -1):
        if name[:k] in vocabulary and re.fullmatch(r"[a-z_-][a-z0-9_'-]*", name[k:]):
            return name[:k]
    return None


def summarize(view, split, code):
    z = np.load(OUT / f"{view}-{split}.npz")
    names = json.loads((OUT / f"{view}-{split}.json").read_text())["type_names"]
    with (OUT / f"types-{view}-{split}.csv").open(newline="") as stream:
        listed = [r for r in csv.DictReader(stream) if r["rank"]]
    with np.load(BENCH / f"{view}.npz", allow_pickle=False) as b:
        offset, codes = b["fam_code_offset"], b["fam_codes"]
    question, kind, weight = z["pair_question"], z["pair_type"], z["pair_weight"]
    lower, upper, threshold = z["lower"][question], z["upper"][question], z["threshold"][question]
    outside = certified_outside(upper, threshold, weight)
    inside = certified_inside(lower, threshold, weight)
    n = len(names)
    count = lambda mask: np.bincount(kind[mask], minlength=n)
    per_outside, per_inside, per_both, per_questions = count(outside), count(inside), count(outside & inside), np.bincount(kind, minlength=n)
    uncertifiable = int((z["status"] == 0).sum())
    index = {name: k for k, name in enumerate(names)}
    rows, splits = [], []
    for r in listed[:FIRST]:
        k = index[r["type"]]
        if per_outside[k] != int(r["certified_alone"]):
            raise SystemExit(f"{view} {split}: {r['type']} differs from the ranked list")
        row = dict(rank=int(r["rank"]), type=r["type"], superclass=r["superclass"], questions=int(per_questions[k]),
                   outside=int(per_outside[k]), inside=int(per_inside[k]), both=int(per_both[k]))
        parent = split_parent(r["type"], code)
        if parent:
            mine = kind == k
            family = z["family"][question[mine]]
            contains = np.array([code[parent] in codes[offset[f]:offset[f + 1]] for f in family.tolist()], dtype=bool)
            row.update(parent=parent, parent_in_partner_set=int(contains.sum()), inside_where_parent_in=int((inside[mine] & contains).sum()),
                       outside_where_parent_out=int((outside[mine] & ~contains).sum()), outside_where_parent_in=int((outside[mine] & contains).sum()))
            splits.append(row)
        rows.append(row)
    return dict(
        uncertifiable=uncertifiable,
        one_type_outside=float(np.isin(np.arange(len(z["status"])), question[outside]).sum() / uncertifiable),
        one_type_inside=float(np.isin(np.arange(len(z["status"])), question[inside]).sum() / uncertifiable),
        one_type_either=float(np.isin(np.arange(len(z["status"])), question[outside | inside]).sum() / uncertifiable),
        first=rows[:10], split_types=splits,
    )


def main():
    vocabulary = json.loads(VOCABULARY.read_text())["vocabulary"]
    code = {name: i for i, name in enumerate(vocabulary) if i}
    result = {split: {view: summarize(view, split, code) for view in VIEWS} for split in SPLITS}
    (OUT / "readings.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for split in SPLITS:
        for view in VIEWS:
            r = result[split][view]
            print(split, view, {k: round(r[k], 4) for k in ("one_type_outside", "one_type_inside", "one_type_either")}, "split types among the first", FIRST, ":", len(r["split_types"]))
            for row in r["first"][:5]:
                print("   ", {k: row[k] for k in ("type", "questions", "outside", "inside", "both") }, {k: row[k] for k in row if k.startswith(("parent", "inside_where", "outside_where"))})


if __name__ == "__main__":
    main()
