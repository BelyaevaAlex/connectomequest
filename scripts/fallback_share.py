"""How many checks of edge cutting and information gain are chosen by their fallback: a diagnostic on development types.

Both criteria score a cell by how much its outcome is expected to change the probabilities of the decisions yes, no
and uncertifiable, estimated from sampled realizations of the prior. While the sampled realizations agree on one
decision no cell has a positive score, and the policy checks the heaviest unchecked cell instead. This script re-runs
the four policies of ``annotation_v35/decision.py`` on the certifiable questions about the new target neurons of the
development types, with the random numbers of the replication, verifies question by question that their costs are
those the replication stored, and counts the checks chosen by the fallback.

Development types only: the held-out types of the replication were run once and are not run again.

Usage: python scripts/fallback_share.py [--workers 28]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replication as rep  # noqa: E402  (sets one numerical thread per worker before numpy loads)
import numpy as np  # noqa: E402
from connectomequest.annotation_v35 import decision, engine as e  # noqa: E402

OUT = ROOT / "outputs/fallback"
_D = {}


def run_family(family):
    d, view = _D["data"], _D["view"]
    target = int(d["fam_target"][family])
    lo, hi = int(d["target_offset"][target]), int(d["target_offset"][target + 1])
    cells, weights, labels = d["pre_cell"][lo:hi], d["pre_weight"][lo:hi], d["pre_label"][lo:hi]
    n = hi - lo
    codes = d["fam_codes"][d["fam_code_offset"][family] : d["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    start = int(d["fam_plus_offset"][family])
    priors = {p: rep.probabilities(d["plus_" + p][start : start + n], d["unknown_" + p][lo:hi]) for p in rep.PRIORS[view]}
    rows = []
    for level, (question, num, den) in enumerate(rep.LEVELS):
        threshold = rep.threshold_of(question, num, den, total, int(d["fam_source_weight"][family]), int(d["fam_source_total"][family]))
        inst = e.threshold_instance(cells, weights, threshold)
        if e.full_information_status(inst, outcomes) == "unresolved":
            continue
        for prior, p in priors.items():
            for g, name in enumerate(rep.GENERIC):
                policy = decision.policy(name, inst, p, np.random.default_rng([43, family, level, g]))
                cost, _ = e.evaluate(inst, outcomes, e.run_adaptive(inst, outcomes, policy))
                rows.append((family, level, f"{prior}:{name}", cost, policy.fallbacks))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=28)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for view in rep.PRIORS:
        d = rep.load(view)
        _D.update(data=d, view=view)
        families = rep.split_families(view, d, "development")
        with get_context("fork").Pool(args.workers) as pool:
            rows = [row for group in pool.map(run_family, families, chunksize=4) for row in group]
        meta = json.loads((rep.OUT / f"{view}-development.json").read_text())
        with np.load(rep.OUT / f"{view}-development.npz", allow_pickle=False) as z:
            stored = {(int(f), int(lv)): cost for f, lv, cost in zip(z["family"], z["level"], z["cost"])}
        out = {}
        for family, level, name, cost, fallbacks in rows:
            if stored[family, level][meta["arms"].index(name)] != cost:
                raise SystemExit(f"the cost of {name} differs from the replication")
            if fallbacks > cost:
                raise AssertionError("more fallback choices than checks")
            part = rep.LEVELS[level][0]
            prior, policy = name.split(":")
            entry = out.setdefault(part, {}).setdefault(prior, {}).setdefault(policy, dict(questions=0, checks=0, fallback=0))
            entry["questions"] += 1
            entry["checks"] += int(cost)
            entry["fallback"] += int(fallbacks)
        for part in out.values():
            for prior in part.values():
                for entry in prior.values():
                    entry["share"] = entry["fallback"] / entry["checks"]
        result[view] = dict(families=len(families), **out)
        for part in ("retention", "share"):
            for prior, policies in out[part].items():
                print(view, part, prior, {name: round(x["share"], 3) for name, x in policies.items()}, flush=True)
    (OUT / "development.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
