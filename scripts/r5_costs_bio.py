#!/usr/bin/env python3
"""Round 5, workstream ``costs``: biological readout of the frozen answers on one split.

B3 partial identification of conservation (declared and lenient crosswalk), B4 certified
cross-connectome disagreements against sexual-dimorphism and fru/dsx annotations
(permutation tests over types, Holm over the four primary tests), B5 shares, answers and
costs by hemibrain truncation of the target type and of its partner set. Descriptive: no
policy is run or tuned. The held-out split is refused unless outputs/round5/record.json
lists the sha256 of this script and of every round-5 module it imports; outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from connectomequest.round5 import costs_bio as cbio
from connectomequest.round5 import costs_common as cc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=cc.SPLITS)
    parser.add_argument("--output", default=str(cc.OUT))
    parser.add_argument("--expectations", type=Path, help="JSON list, see costs_common.check_expectations")
    args = parser.parse_args()
    cc.guard(args.split, __file__, extra=[args.expectations] if args.expectations else [])
    out = Path(args.output) / f"bio-{args.split}.json"
    if out.exists():
        raise FileExistsError(out)
    if args.split == "test":
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "costs", "bio", out)
    started = time.time()
    result = dict(split=args.split, conservation={}, truncation={})
    # analyses excluded from the paper before any held-out run (outputs/round5/scope.json) are not computed on it
    excluded = json.loads((cc.ROOT / "outputs/round5/scope.json").read_text())["excluded_from_paper"] if args.split == "test" else {}
    skip_truncation, skip_dimorphism = "costs:B5_hemibrain_truncation" in excluded, "costs:B4_sexual_dimorphism" in excluded
    if skip_truncation or skip_dimorphism:
        result["excluded_by_scope"] = sorted(excluded)
    lenient_json = cc.ROOT / f"outputs/v35/analysis/lenient-{args.split}.json"
    reference = json.loads(lenient_json.read_text()) if lenient_json.exists() else None
    for view in cc.VIEWS:
        cons = cbio.conservation(view, args.split)
        cons.pop("statuses")
        if reference is not None:
            ref = reference[view]["answers"]
            cons["reproduces_lenient_analysis"] = all(
                abs(cons["answers"][mine][k] - ref[theirs][k]) < 1e-12 for mine, theirs in (("declared", "strict"), ("lenient", "lenient")) for k in ref[theirs]
            )
        result["conservation"][view] = cons
        if not skip_truncation:
            result["truncation"][view] = cbio.truncation(view, args.split)
    if not skip_dimorphism:
        result["dimorphism"] = cbio.dimorphism(args.split)
    if args.expectations is not None:
        result["expectations"] = cc.check_expectations(result, json.loads(args.expectations.read_text()))
        result["expectations_sha256"] = cc.hash_file(args.expectations)
    result["wall_seconds"] = time.time() - started
    result["code"] = {p: cc.hash_file(cc.ROOT / p) for p in cc.imported_round5_files(__file__)}
    cc.write_json_new(out, result)
    print(json.dumps({k: result[k] for k in ("split", "wall_seconds")}))
    for view in cc.VIEWS:
        print(view, "reproduces lenient analysis:", result["conservation"][view].get("reproduces_lenient_analysis"),
              "contradictions:", result["conservation"][view]["contradictions"])
        for key, v in result["conservation"][view]["intervals"].items():
            print(f"  {key:32s} n={v['targets']:4d} [{v['lower']['value']:.3f}, {v['upper']['value']:.3f}]  CI-low [{v['lower']['low']:.3f},{v['lower']['high']:.3f}] CI-up [{v['upper']['low']:.3f},{v['upper']['high']:.3f}]")
    for name, t in result.get("dimorphism", {}).get("tests", {}).items():
        print(f"  {name:36s} flagged {t['flagged_units']:4d}/{t['units']:4d} rate {t['flagged_rate']} vs {t['other_rate']:.4f} diff {t['rate_difference']:+.4f} p {t['p_value']:.4f} holm {t.get('p_holm_primary', np.nan):.4f}")
    for item in result.get("expectations", []):
        print(item["name"], round(item["observed"], 4), item["op"], item["value"], "HOLDS" if item["holds"] else "FAILS")


if __name__ == "__main__":
    main()
