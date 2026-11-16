"""Check the expected outcomes of answer propagation with other priors, recorded before the held-out run.

The statements are in outputs/round5/prop/expectations.json (prose, written before the run); each is
translated here literally into a test on the analysis of scripts/r5_prop_run.py. Writes
outputs/round5/prop/expectations-<split>-verdicts.json and the supplement table paper/tables/r5_prop_expectations.tex.

Usage: python scripts/r5_prop_expectations.py [--split test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROP = ROOT / "outputs/round5/prop"
NAMES = {"flywire": "F", "malecns": "M", "banc": "B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    split = ap.parse_args().split
    found = sorted(PROP.glob(f"analysis-{split}-*.json"))
    if len(found) != 1:
        raise SystemExit(f"expected one {split} analysis, found {[p.name for p in found]}")
    views = json.loads(found[0].read_text())["views"]
    recorded = {e["id"]: e for e in json.loads((PROP / "expectations.json").read_text())["expectations"]}
    arm = lambda v, a: views[v]["propagation"][a]
    con = lambda v, c: views[v]["contrasts"][c]["ratio"]
    out = {}

    def verdict(eid, values, ok):
        out[eid] = dict(statement=recorded[eid]["statement"], development=recorded[eid]["development"], held_out=values, holds=bool(ok))

    fly = "flywire"
    small = arm(fly, "provisional:recal+prop")["to_static_by_size"]["<500 input cells"]
    verdict("A1", f"{small['value']:.3f} [{small['low']:.3f}, {small['high']:.3f}]", small["value"] <= 0.92 and small["high"] < 1)
    allq = arm(fly, "provisional:recal+prop")["to_static"]
    verdict("A2", f"{allq['value']:.3f} [{allq['low']:.3f}, {allq['high']:.3f}]", allq["value"] <= 1.00)
    mlp = arm(fly, "mlp_s0:recal+prop")["to_static"]["value"]
    verdict("A3", f"{allq['value']:.3f} vs {mlp:.3f}", allq["value"] - mlp >= 0.05)
    ag = arm(fly, "provisional:ag+prop")["to_static"]["value"]
    verdict("A4", f"{ag:.3f}", ag <= 0.97)
    pq = arm(fly, "provisional:recal+prop")["paired_vs_static"]
    verdict("A5", f"{100 * pq['better']['value']:.0f}% / {100 * pq['worse']['value']:.0f}%",
            pq["better"]["value"] >= 0.12 and pq["worse"]["value"] <= 0.10)

    three = ("flywire", "malecns", "banc")
    vals, ok = [], True
    for v in three:
        for a in ("const_half:ag+prop", "const_half:recal+prop", "const_share:ag+prop", "const_share:recal+prop"):
            r = arm(v, a)["to_weight"]
            vals.append(r["value"])
            ok &= r["value"] <= 0.85 and r["high"] < 0.90
    verdict("B1", f"{min(vals):.3f}-{max(vals):.3f}", ok)
    b2 = [con(v, "need the MLP? const_share:recal+prop / mlp_s0:recal+prop")["value"] for v in three]
    verdict("B2", f"{min(b2):.3f}-{max(b2):.3f}", all(0.97 <= x <= 1.05 for x in b2))
    b3 = [con(v, "need the MLP? const_share:ag+prop / mlp_s0:ag+prop")["value"] for v in three]
    verdict("B3", f"{min(b3):.3f}-{max(b3):.3f}", all(x <= 1.00 for x in b3))
    b4 = [con(v, "need the MLP? const_share:recal+prop / mlp_s0:recal") for v in three]
    verdict("B4", f"{min(x['value'] for x in b4):.3f}-{max(x['value'] for x in b4):.3f}",
            all(x["value"] <= 0.90 and x["high"] < 1 for x in b4))
    b5 = [arm(v, "const_share:recal+prop")["paired_vs_weight"] for v in three]
    verdict("B5", f"{100 * min(x['better']['value'] for x in b5):.0f}-{100 * max(x['better']['value'] for x in b5):.0f}% / "
                  f"{100 * min(x['worse']['value'] for x in b5):.0f}-{100 * max(x['worse']['value'] for x in b5):.0f}%",
            all(x["better"]["value"] >= 0.30 and x["worse"]["value"] <= 0.10 for x in b5))
    c1 = [arm(v, "mlp_s0:member+prop")["to_static"] for v in three]
    verdict("C1", f"{min(x['value'] for x in c1):.3f}-{max(x['value'] for x in c1):.3f}",
            all(x["value"] <= 0.92 and x["high"] < 1 for x in c1))
    c2 = [con(v, "active search: mlp_s0:member+prop / mlp_s0:recal+prop")["value"] for v in three]
    verdict("C2", f"{min(c2):.3f}-{max(c2):.3f}", all(x >= 1.10 for x in c2))
    c3 = [arm(v, "mlp_s0:direction+prop")["to_static"]["value"] for v in three]
    verdict("C3", f"{min(c3):.3f}-{max(c3):.3f}", all(x >= 1.20 for x in c3))
    m = con(fly, "active search: provisional:member+prop / provisional:recal+prop")["value"]
    d = con(fly, "active search: provisional:direction+prop / provisional:recal+prop")["value"]
    verdict("C4", f"{m:.3f}; {d:.3f}", m >= 1.00 and 0.97 <= d <= 1.05)

    two = ("flywire", "malecns")
    s1 = [arm(v, f"seeded{f}{c}:recal+prop")["to_static"]["value"] for v in two for f in (30, 50) for c in ("", "conn")]
    verdict("S1", f"{min(s1):.3f}-{max(s1):.3f}", all(x >= 1.00 for x in s1))
    s2 = [arm(v, f"seeded10{c}:recal+prop")["to_static"]["value"] for v in two for c in ("", "conn")]
    verdict("S2", f"{min(s2):.4f}-{max(s2):.4f}", all(0.95 <= x <= 1.05 for x in s2))
    s3 = [con(v, f"external prior: seeded{f}{c}:recal+prop / mlp_s0:recal+prop")["value"]
          for v in two for f in (10, 30, 50) for c in ("", "conn")]
    verdict("S3", f"{min(s3):.3f}-{max(s3):.3f}", all(x <= 0.90 for x in s3))

    (PROP / f"expectations-{split}-verdicts.json").write_text(json.dumps(dict(source=found[0].name, verdicts=out), indent=1))
    if split == "test":
        def tex(s):
            for a, b in (("%", r"\%"), ("<=", r"$\leq$"), (">=", r"$\geq$"), ("<", "$<$"), ("_", r"\_")):
                s = s.replace(a, b)
            return s
        rows = [" & ".join((eid, tex(r["statement"]), tex(r["development"]), tex(r["held_out"]),
                            r"$\bullet$" if r["holds"] else r"$\times$")) + r"\\" for eid, r in out.items()]
        (ROOT / "paper/tables/r5_prop_expectations.tex").write_text("\n".join(rows) + "\n")
    for eid, r in out.items():
        print(f"{eid} {'HOLDS' if r['holds'] else 'FAILS'}  held-out {r['held_out']}  (dev {r['development']})  {r['statement']}")
    print(f"{sum(r['holds'] for r in out.values())} of {len(out)} hold")


if __name__ == "__main__":
    main()
