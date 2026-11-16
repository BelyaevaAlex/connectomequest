#!/usr/bin/env python3
"""Descriptive agreement of exact answers between neurons and between connectomes.

Uses only the full-information status of each held-out question (no policy):
(a) two target neurons of the same type in one connectome, (b) target neurons of the
same type in the two connectomes. A pair counts when both answers are decided.
Also lists types whose strongest partner is certified lost at rho = 1/2 in both.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RHO = ("1/4", "1/2", "3/4", "1/1")
KIND = ("top1", "coalition")


def load(view, split="test"):
    run = np.load(ROOT / f"outputs/v35/runs/{view}-{split}.npz", allow_pickle=False)
    bench = np.load(ROOT / f"outputs/v35/benchmark/{view}.npz", allow_pickle=False)
    target = bench["fam_target"][run["family"]]
    table = defaultdict(dict)
    for t, code, kind, rho, truth in zip(target.tolist(), bench["target_type"][target].tolist(), bench["fam_kind"][run["family"]].tolist(), run["rho"].tolist(), run["truth"].tolist()):
        table[(code, kind, rho)][t] = truth
    return table


def agreement(pairs):
    pairs = np.asarray(pairs)
    decided = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    if not len(decided):
        return dict(pairs=int(len(pairs)), decided=0)
    same = float((decided[:, 0] == decided[:, 1]).mean())
    p_yes = [(decided[:, i] == 1).mean() for i in (0, 1)]
    expected = float(p_yes[0] * p_yes[1] + (1 - p_yes[0]) * (1 - p_yes[1]))
    kappa = (same - expected) / (1 - expected) if expected < 1 else float("nan")
    return dict(pairs=int(len(pairs)), decided=int(len(decided)), agreement=same, expected_by_chance=expected, kappa=float(kappa),
                yes_yes=int(((decided[:, 0] == 1) & (decided[:, 1] == 1)).sum()), no_no=int(((decided[:, 0] == 2) & (decided[:, 1] == 2)).sum()),
                mixed=int((decided[:, 0] != decided[:, 1]).sum()))


def main():
    tables = {view: load(view) for view in ("malecns", "flywire")}
    manifest = json.loads((ROOT / "outputs/v35/benchmark/manifest.json").read_text())
    names = {t["code"]: t for t in manifest["types"]}
    out = dict(within={}, across={})
    for view, table in tables.items():
        for kind in (0, 1):
            for rho in range(4):
                pairs = [sorted(v.items()) for (c, k, r), v in table.items() if k == kind and r == rho and len(v) == 2]
                out["within"][f"{view}:{KIND[kind]}@rho={RHO[rho]}"] = agreement([(a[1], b[1]) for a, b in pairs])
        out["within"][f"{view}:all"] = agreement([(a[1], b[1]) for v in table.values() if len(v) == 2 for a, b in [sorted(v.items())]])
    lost = []
    every = []
    for kind in (0, 1):
        for rho in range(4):
            pairs = []
            for key, male in tables["malecns"].items():
                if key[1] != kind or key[2] != rho or key not in tables["flywire"]:
                    continue
                fly = tables["flywire"][key]
                pairs += [(a, b) for a in male.values() for b in fly.values()]
                if kind == 0 and rho == 1 and all(x == 2 for x in male.values()) and all(x == 2 for x in fly.values()):
                    t = names[key[0]]
                    lost.append(dict(type=t["name"], partner=t["families"][0]["names"][0], hemibrain_share=t["families"][0]["source_weight"] / t["source_total_weight"]))
            out["across"][f"{KIND[kind]}@rho={RHO[rho]}"] = agreement(pairs)
            every += pairs
    out["across"]["all"] = agreement(every)
    out["strongest_partner_lost_in_both_at_half"] = sorted(lost, key=lambda x: -x["hemibrain_share"])
    (ROOT / "outputs/v35/analysis/agreement-test.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for group in ("within", "across"):
        for key, value in out[group].items():
            print(group, key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in value.items()})
    print(len(lost), [(x["type"], x["partner"], round(x["hemibrain_share"], 2)) for x in out["strongest_partner_lost_in_both_at_half"][:40]])


if __name__ == "__main__":
    main()
