"""Check every expected outcome recorded before the held-out run of the further exploratory analyses.

Sources of the recorded expectations (all written before the held-out run, pinned by outputs/round5/record.json):
  costs     outputs/round5/costs/expectations-{analysis,bio-scoped}.json, evaluated by the analysis scripts
  policies  outputs/round5/policies/expectations.json (path, operator, threshold), evaluated here
  banc      outputs/round5/banc/expectations.json (prose), translated literally here
  priors    outputs/round5/priors/record-proposal.json "expected_outcomes" (prose), translated literally here
  prop      outputs/round5/prop/expectations.json (prose), evaluated by scripts/r5_prop_expectations.py

Writes outputs/round5/expectations-<split>.json and paper/tables/r5_expectations{,_failed}.tex.
Usage: python scripts/r5_expectations.py [--split test]
"""
from __future__ import annotations

import argparse
import json
import operator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "outputs/round5"
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
LABEL = {"costs": "Cost models and partial identification", "policies": "Optimal policy, further comparators, sampling",
         "banc": "Third connectome (BANC)", "priors": "Seeded priors and correlated errors", "prop": "Propagation with other priors"}


def read(path):
    return json.loads(Path(path).read_text())


def costs(split):
    out = []
    for path in (R5 / f"costs/analysis-{split}.json", R5 / f"costs/bio-{split}.json"):
        for e in read(path)["expectations"]:
            out.append(dict(id=e["name"].split()[0], statement=e["name"], held_out=f"{e['observed']:.3f}", holds=bool(e["holds"])))
    return out


def policies(split):
    data = read(R5 / f"policies/analysis-{split}.json")
    out = []
    for e in read(R5 / "policies/expectations.json")["expectations"]:
        x = data
        for k in e["path"]:
            x = x[k]
        if e["field"] is not None:
            x = x[e["field"]]
        out.append(dict(id=e["id"], statement=e["claim"], held_out=f"{x:.3f}", holds=bool(OPS[e["op"]](x, e["threshold"]))))
    return out


def banc(split):
    b = read(R5 / f"banc/report-{split}.json")["banc"]
    r = b["ratios"]
    cl, rw, aw, ra = r["clairvoyant/weight"], r["mlp_s0:recal/weight"], r["mlp_s0:ag/weight"], r["mlp_s0:recal/mlp_s0:ag"]
    fly_unres = read(R5 / f"banc/report-{split}.json")["frozen"]["flywire"]["uncertifiable_share"]
    rows = [
        ("E1", f"{cl['value']:.3f} [{cl['low']:.3f}, {cl['high']:.3f}]", 0.38 <= cl["value"] <= 0.55 and cl["high"] < 0.60),
        ("E2", f"{rw['value']:.3f}", 0.97 <= rw["value"] <= 1.02),
        ("E3", f"{aw['value']:.3f}", 0.97 <= aw["value"] <= 1.04),
        ("E4", f"{ra['value']:.3f} [{ra['low']:.3f}, {ra['high']:.3f}]", ra["value"] <= 1.00 and ra["high"] <= 1.01),
        ("E5", f"{100 * b['const_half:recal==weight']:.0f}%, {100 * b['const_share:ag==weight']:.0f}%",
         b["const_half:recal==weight"] == 1.0 and b["const_share:ag==weight"] == 1.0),
        ("E6", f"{b['uncertifiable_share']:.3f}", b["uncertifiable_share"] >= 0.40 and b["uncertifiable_share"] > fly_unres),
        ("E7", f"no {b['no_share']:.3f}, yes {b['yes_share']:.3f}", b["no_share"] <= 0.10 and b["yes_share"] >= 0.40),
        ("E8", f"{b['weight_median_checks']:.0f}", 3 <= b["weight_median_checks"] <= 6),
    ]
    text = {e["id"]: e["statement"] for e in read(R5 / "banc/expectations.json")["expectations"]}
    return [dict(id=i, statement=text[i], held_out=v, holds=bool(ok)) for i, v, ok in rows]


def priors(split):
    views = read(R5 / f"priors/analysis/{split}.json")["views"]
    two = ("malecns", "flywire")
    s = {v: views[v]["seeded"] for v in two}
    rw = lambda v, a: s[v]["ratio_to_weight"][a]
    c = lambda v, k: s[v]["contrasts"][k]
    rows = []
    au = {(v, f): s[v]["quality"][f"seeded{f}"]["questions_nonseed_cells"]["member_auroc_within_question"] for v in two for f in (10, 30, 50)}
    rows.append(("E1", ", ".join(f"{au[v, f]:.3f}" for v in two for f in (10, 30, 50)),
                 all(au[v, 10] >= 0.93 and au[v, 30] >= 0.98 and au[v, 50] >= 0.99 for v in two)))
    e2 = [(rw(v, "seeded10:recal")["value"], rw(v, "seeded10:direction")["value"]) for v in two]
    rows.append(("E2", "; ".join(f"{a:.3f}/{b:.3f}" for a, b in e2), all(a <= 0.72 and b <= 0.72 for a, b in e2)))
    e3 = [[rw(v, f"seeded{f}:direction")["value"] for f in (50, 30, 10)] for v in two]
    rows.append(("E3", "; ".join("<".join(f"{x:.3f}" for x in d) for d in e3), all(d[0] < d[1] < d[2] for d in e3)))
    e4 = [views[v]["seeded_vs_controls"]["seeded10:recal/seedmlp10:recal"]["value"] for v in two]
    rows.append(("E4", "; ".join(f"{x:.3f}" for x in e4), all(x <= 0.80 for x in e4)))
    e5 = [c(v, f"seeded{f}:ag/seeded{f}:direction")["value"] for v in two for f in (30, 50)]
    rows.append(("E5", f"{min(e5):.3f}-{max(e5):.3f}", all(x >= 1.03 for x in e5)))
    e6 = [c(v, f"seeded{f}:recal/seeded{f}conn:recal")["value"] for v in two for f in (10, 30, 50)]
    rows.append(("E6", f"max {max(abs(x - 1) for x in e6):.3f}", all(abs(x - 1) <= 0.02 for x in e6)))
    e7 = c("flywire", "seeded10:recal/provisional:recal")["value"]
    rows.append(("E7", f"{e7:.3f}", e7 <= 0.95))
    e8 = [c(v, "seeded10:recal/mlp_s0:recal")["value"] for v in two]
    rows.append(("E8", "; ".join(f"{x:.3f}" for x in e8), all(x <= 0.75 for x in e8)))
    shift = lambda v, k: views[v]["correlated"]["shifts"][k]["0.95"]["median"]
    e9 = [shift(v, "base/wminus/ag") for v in two]
    rows.append(("E9", "; ".join(f"{x:+.3f}" for x in e9), all(0.01 <= x <= 0.07 for x in e9)))
    e10 = [shift(v, "base/wplus/ag") for v in two]
    rows.append(("E10", "; ".join(f"{x:+.3f}" for x in e10), all(x <= 0 for x in e10)))
    e11 = [shift(v, "base/rtype/ag") for v in two]
    rows.append(("E11", "; ".join(f"{x:+.3f}" for x in e11), all(abs(x) <= 0.03 for x in e11)))
    e12 = [shift(v, f"{form}/rtype/recal") for v in two for form in ("base", "oracle")]
    rows.append(("E12", "; ".join(f"{x:+.3f}" for x in e12), all(x <= -0.03 for x in e12)))
    e13 = [views[v]["correlated"]["curves"][f"base/{form}/ag"]["crossing"]["0.95"]["value"]
           for v in two for form in ("independent", "wplus", "wminus", "rtype")]
    rows.append(("E13", f"{min(e13):.3f}-{max(e13):.3f}", all(0.70 <= x <= 0.85 for x in e13)))
    text = {}
    for e in read(R5 / "priors/record-proposal.json")["expected_outcomes"]:
        text[e.split(" ", 1)[0].rstrip(":")] = e
    return [dict(id=i, statement=text[i], held_out=v, holds=bool(ok)) for i, v, ok in rows]


def prop(split):
    v = read(R5 / f"prop/expectations-{split}-verdicts.json")["verdicts"]
    return [dict(id=i, statement=r["statement"], held_out=r["held_out"], holds=r["holds"]) for i, r in v.items()]


READABLE = (("Bayes policy", "optimal policy"), ("on the true answers", "on the reference outcomes"), ("(mlp_s0)", "(topology prior)"),
            ("mlp_s0", "topology prior"), ("the chosen VOI policy", "the one-step value-of-information rule"), ("WoR hybrid", "hybrid test without replacement"), ("WoR", "sampling without replacement"),
            ("malecns small@3: cost-aware weight order / plain", "MaleCNS, small cells at cost 3: cost-aware weight order / weight order"),
            ("F, M: seeded10[conn]:recal+prop / seeded10[conn]:recal", "FlyWire and MaleCNS, seeded prior from 10% of the neurons (both scorers): "
             "recalibrated AG with propagation / without"))
VIEW = {"malecns": "MaleCNS", "flywire": "FlyWire", "banc": "BANC"}


def readable(e):
    """The recorded statement with internal arm names spelled out, prefixed by the connectome where the id names one."""
    s = e["statement"]
    if s.split()[0][:1] in "AB" and s.split()[0][1:2].isdigit():  # drop a leading identifier such as "A2.2"
        s = s.split(" ", 1)[1]
    for old, new in READABLE:
        s = s.replace(old, new)
    view = e["id"].rsplit("-", 1)[-1]
    return f"{VIEW[view]}: {s}" if view in VIEW and not s.startswith(VIEW[view]) else s


def tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#"), ("<=", r"$\leq$"),
                 (">=", r"$\geq$"), ("<", "$<$"), (">", "$>$"), ("|", "$|$"), ("~", r"$\sim$"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    split = ap.parse_args().split
    result = {name: fn(split) for name, fn in (("costs", costs), ("policies", policies), ("banc", banc), ("priors", priors), ("prop", prop))}
    (R5 / f"expectations-{split}.json").write_text(json.dumps(result, indent=1))
    total = sum(len(v) for v in result.values())
    held = sum(e["holds"] for v in result.values() for e in v)
    for name, v in result.items():
        print(f"{name}: {sum(e['holds'] for e in v)} of {len(v)} hold")
        for e in v:
            if not e["holds"]:
                print(f"   FAILS {e['id']}: held-out {e['held_out']} -- {e['statement']}")
    print(f"all: {held} of {total}")
    if split == "test":
        rows = [f"{LABEL[n]} & {len(v)} & {sum(e['holds'] for e in v)}\\\\" for n, v in result.items()]
        (ROOT / "paper/tables/r5_expectations.tex").write_text("\n".join(rows) + "\n")
        failed = [f"{LABEL[n]} & {tex(readable(e))} & {tex(e['held_out'])}\\\\" for n, v in result.items() for e in v if not e["holds"]]
        (ROOT / "paper/tables/r5_expectations_failed.tex").write_text("\n".join(failed) + "\n")


if __name__ == "__main__":
    main()
