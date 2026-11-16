#!/usr/bin/env python3
"""The complete priority lists that remain after both rules (scripts/labelings.py reports their first ten types).

For each connectome the script rebuilds, from the stored arrays, the list of release-specific types that neither
rule labels, ranked for the held-out questions that stay uncertifiable after both rules, and writes it to
outputs/labelings/residual-{connectome}-test.csv with, for every type: its rank, the open questions with cells of the
type, the questions it would certify alone when its cells are read as nonmembers (no) or as members (yes), and the
questions certified by the list down to this type. It stops unless the first ten rows equal the recorded analysis.

Usage: python scripts/labelings_lists.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

import labelings as lab

base = lab.base


def write(view, split="test"):
    z = np.load(lab.OUT / f"{view}-{split}.npz")
    stored = np.load(base.OUT / f"{view}-{split}.npz")
    names = json.loads((base.OUT / f"{view}-{split}.json").read_text())["type_names"]
    labelled = set(json.loads((lab.OUT / f"{view}-{split}.json").read_text())["covered_types"]["both"])
    k = lab.LABELINGS.index("both")
    lower, upper, threshold = z["lower"][:, k], z["upper"][:, k], z["threshold"]
    question, kind, weight = stored["pair_question"], stored["pair_type"], stored["pair_weight"]
    open_ = lab.answer(lower, upper, threshold) == 0
    keep = ~np.isin(kind, [j for j, name in enumerate(names) if name in labelled]) & open_[question]
    question, kind, weight = question[keep], kind[keep], weight[keep]
    deficit = upper - threshold + 1
    order, certified = base.greedy_list(question, kind, weight, deficit, len(names), names)
    no, yes = upper[question] - weight < threshold[question], lower[question] + weight >= threshold[question]
    count = lambda mask: np.bincount(kind[mask], minlength=len(names))
    with_type, alone_no, alone_yes = np.bincount(kind, minlength=len(names)), count(no), count(yes)
    rank = {int(j): i + 1 for i, j in enumerate(order)}
    listed = sorted(np.unique(kind).tolist(), key=lambda j: (rank.get(j, len(names) + 1), -with_type[j], names[j]))
    rows = [dict(rank=rank.get(j, ""), type=names[j], open_questions_with_type=int(with_type[j]), certified_alone_no=int(alone_no[j]),
                 certified_alone_yes=int(alone_yes[j]), certified_by_list_so_far=int(certified[rank[j] - 1]) if j in rank else "")
            for j in listed]
    recorded = json.loads((lab.OUT / f"analysis-{split}.json").read_text())[view]["residual"]
    first = [dict(type=r["type"], questions=r["open_questions_with_type"], no=r["certified_alone_no"], yes=r["certified_alone_yes"]) for r in rows[:10]]
    if first != recorded["top"] or len(order) != recorded["length"] or len(listed) != recorded["types"]:
        raise SystemExit(f"{view}: the list differs from the recorded analysis")
    with (lab.OUT / f"residual-{view}-{split}.csv").open("w", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        table.writeheader()
        table.writerows(rows)
    print(view, len(rows), "types,", len(order), "ranked")


if __name__ == "__main__":
    for name in lab.VIEWS:
        write(name)
