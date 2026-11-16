"""Numbers and tables of the paper that come from the exploratory analyses run once on held-out questions.

Writes ``paper/r5_numbers.tex`` (macros), ``paper/tables/composition3.tex`` (composition with BANC) and the
supplement tables ``paper/tables/r5_*.tex``. Each sentence of the paper that rests on one of these numbers
states a direction (cheaper, no saving, within a bound); the direction is checked here, and the build stops
if a result contradicts it. A result that does not exist yet is written as ``??``. It also writes the few
numbers of the pre-specified analysis that the text added after the held-out run of these analyses (``answers``).

Usage: python scripts/r5_paper_assets.py [--split test|development]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
R5 = ROOT / "outputs/round5"
VIEWS = ("malecns", "flywire")
TAG = {"malecns": "Male", "flywire": "Fly"}
MISSING = r"\textbf{??}"


def read(path):
    return json.loads(Path(path).read_text())


def f2(x):
    return f"{x:.2f}"


def ci(d):
    return f"{d['value']:.2f} [{d['low']:.2f}, {d['high']:.2f}]"


def ci3(d):
    """Three decimals, for intervals narrower than 0.01."""
    return f"{d['value']:.3f} [{d['low']:.3f}, {d['high']:.3f}]"


def up(x, digits):
    """Round up: used where the text says 'at most' or 'below'."""
    q = 10 ** digits
    return f"{math.ceil(x * q - 1e-9) / q:.{digits}f}"


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


class Claims:
    def __init__(self):
        self.failed = []

    def __call__(self, text, ok):
        if not ok:
            self.failed.append(text)


def source(path, split, macros, claims, build):
    """Run ``build`` on the analysis at ``path`` or leave its macros as ?? when it does not exist yet."""
    if not path.exists():
        print(f"missing {path.relative_to(ROOT)}: {len(macros)} macros left as ??")
        return {k: MISSING for k in macros}, None
    data = read(path)
    if data.get("split", split) != split:
        raise SystemExit(f"{path} is for split {data.get('split')}, not {split}")
    out, table = build(data, claims)
    if set(out) != set(macros):
        raise SystemExit(f"{path.name}: macros {sorted(set(out) ^ set(macros))} do not match the declared list")
    return out, table


# ---------------------------------------------------------------- partial identification (costs bio)
PI_MACROS = tuple(f"PI{TAG[v]}{k}" for v in VIEWS for k in ("Low", "High", "LenHigh"))


def partial_identification(data, claims):
    out, rows = {}, []
    v35 = v35_macros()
    dev = "Dev" if data["split"] == "development" else ""
    for view in VIEWS:
        iv = data["conservation"][view]["intervals"]
        dec, len_ = iv["declared:top1@rho=1/2"], iv["lenient:top1@rho=1/2"]
        lo, hi, lhi = (100 * dec["lower"]["value"], 100 * dec["upper"]["value"], 100 * len_["upper"]["value"])
        claims(f"{view}: the lower end is the certified share of yes", abs(lo - float(v35[f"{TAG[view]}{dev}TopHalfPos"])) < 0.051)
        claims(f"{view}: the upper end is 100% minus the certified share of no",
               abs(100 - hi - float(v35[f"{TAG[view]}{dev}TopHalfNeg"])) < 0.051)
        claims(f"{view}: the lenient crosswalk lowers the upper end", lhi < hi)
        out.update({f"PI{TAG[view]}Low": f"{lo:.1f}", f"PI{TAG[view]}High": f"{hi:.1f}", f"PI{TAG[view]}LenHigh": f"{lhi:.1f}"})
        rows.append((view, dec, len_))
    table = [f"{name} & {100 * d['lower']['value']:.1f} & {100 * d['upper']['value']:.1f} & {100 * l['upper']['value']:.1f}\\\\"
             for (_, d, l), name in zip(rows, ("MaleCNS", "FlyWire"))]
    return out, {"r5_identification": table}


# ---------------------------------------------------------------- policies: optimum, uncertainty sampling, WoR
POL_MACROS = ("DPQuestionsMale", "DPQuestionsFly", "DPMlpAGOptMaxPct", "DPProvAGOptExpectedPct", "DPProvAGOptRealized",
              "USMale", "USFly", "WUSMale", "WUSFly", "WorOverWrLow", "WorOverWrHigh", "WorBigMale", "WorBigFly",
              "VoiOverRecalLow", "VoiOverRecalHigh", "SafeMaxOverCheaper", "WorErrMaxPct",
              "DPMlpAGRealLow", "DPMlpAGRealHigh")


def policies(data, claims):
    out, dp_rows, us_rows, wor_rows = {}, [], [], []
    runs, dp, wor = data["runs"], data["dp"], data["wor"]
    exact = lambda d: abs(d["value"] - 1) < 1e-12 and abs(d["low"] - 1) < 1e-12 and abs(d["high"] - 1) < 1e-12
    mlp_gap, mlp_real = [], []
    for view in VIEWS:
        for prior in ("const_half", "const_share"):
            ref = runs[f"{view}:{prior}"]["reference_to_weight"]
            claims(f"{view}/{prior}: AG and recalibrated AG check as many cells as weight order",
                   exact(ref["ag"]) and exact(ref["recal"]))
            d = dp[f"{view}:{prior}:n14"]
            claims(f"{view}/{prior}: weight order is optimal on small questions",
                   exact(d["expected_over_opt"]["weight"]) and d["realized_opt_better_tied_worse"][1] == 1.0)
        dp_rows.append((view, "none ($p^+=\\tfrac12$)", dp[f"{view}:const_half:n14"]))
        d = dp[f"{view}:mlp_s0:n14"]
        out[f"DPQuestions{TAG[view]}"] = thousands(d["certifiable"])
        claims(f"{view}: the DP subsets of the constant and topology priors coincide",
               d["certifiable"] == dp[f"{view}:const_half:n14"]["certifiable"])
        mlp_gap.append(d["expected_over_opt"]["ag"]["value"] - 1)
        mlp_real.append(d["realized_to_opt"]["ag"]["value"])
        dp_rows.append((view, "topology", d))
        arms = runs[f"{view}:mlp_s0"]["arms"]
        us, wus = arms["us"]["to_weight"], arms["wus"]["to_weight"]
        claims(f"{view}: uncertainty sampling costs more than weight order", us["low"] > 1 and wus["low"] > 1)
        out[f"US{TAG[view]}"], out[f"WUS{TAG[view]}"] = f2(us["value"]), f2(wus["value"])
        us_rows.append((view, arms))
        w = wor[view]
        big = w["strata"][">=500"]
        out[f"WorBig{TAG[view]}"] = ci(big["wor"])
        claims(f"{view}: on targets with >=500 inputs sampling without replacement is below weight order, with an interval "
               f"that includes one, and below sampling with replacement",
               big["wor"]["value"] < 1 <= big["wor"]["high"] and big["wor"]["value"] < big["wr"]["value"])
        claims(f"{view}: and costs more than weight order on targets with 10-49 inputs", w["strata"]["10-49"]["wor"]["low"] > 1)
        wor_rows.append((view, w))
    d = dp["flywire:provisional:n14"]
    dp_rows.append(("flywire", "earlier release", d))
    exp_gap, real = d["expected_over_opt"]["ag"]["value"] - 1, d["realized_to_opt"]["ag"]
    claims("topology prior: AG is within 1% of the optimum in expectation", max(mlp_gap) < 0.01)
    claims("earlier-release prior: AG is near-optimal in expectation", exp_gap < 0.03)
    claims("but far from the optimum on the actual labels", real["low"] > 1.1)
    out["DPMlpAGOptMaxPct"] = up(100 * max(mlp_gap), 1)
    out["DPMlpAGRealLow"], out["DPMlpAGRealHigh"] = f2(min(mlp_real)), f2(max(mlp_real))
    claims("topology prior: AG stays within 10% of the optimum on the actual labels", max(mlp_real) < 1.1)
    out["DPProvAGOptExpectedPct"] = up(100 * exp_gap, 1)
    out["DPProvAGOptRealized"] = f2(real["value"])
    ratios = [wor[v]["contrasts"]["wor_over_wr"] for v in VIEWS]
    claims("sampling without replacement draws fewer cells than with replacement", all(r["high"] < 1 for r in ratios))
    errors = [wor[v]["adversarial"][k]["claim_error"]["rate"] for v in VIEWS for k in wor[v]["adversarial"]]
    claims("sampling tests keep their error rate on adversarial thresholds", max(errors) <= 0.05)
    out["WorErrMaxPct"] = up(100 * max(errors), 1)
    out["WorOverWrLow"], out["WorOverWrHigh"] = f2(min(r["value"] for r in ratios)), f2(max(r["value"] for r in ratios))
    voi = read(ROOT / "outputs/round5/record.json")["policies"]["frozen_choices"]["voi"]["chosen"]  # fixed on development
    pairs = [("malecns", "mlp_s0"), ("flywire", "mlp_s0"), ("flywire", "provisional")]
    vr = [runs[f"{v}:{p}"]["arms"][voi]["to_recal"] for v, p in pairs]
    claims("one-step value of information does not beat recalibrated AG", all(r["value"] >= 1 for r in vr))
    out["VoiOverRecalLow"], out["VoiOverRecalHigh"] = f2(min(r["value"] for r in vr)), f2(max(r["value"] for r in vr))
    safe = [runs[f"{v}:{p}"]["arms"]["recal+wo"]["max_per_question_over_cheaper"] for v, p in pairs]
    claims("interleaving never costs more than twice the cheaper of its two orders", max(safe) <= 2 + 1e-9)
    out["SafeMaxOverCheaper"] = f2(max(safe))
    tables = {
        "r5_dp": [f"{'MaleCNS' if v == 'malecns' else 'FlyWire'} & {p} & {thousands(d['certifiable'])} & "
                  f"{d['expected_over_opt']['ag']['value']:.3f} & {d['expected_over_opt']['weight']['value']:.3f} & "
                  f"{ci(d['realized_to_opt']['ag'])} & {ci(d['realized_to_weight']['opt'])}\\\\" for v, p, d in dp_rows],
        "r5_us": [],
        "r5_wor": [],
    }
    labels = (("us", "uncertainty sampling (US)"), ("wus", "weighted US"), ("us-recal", "US, recalibrated"),
              ("wus-recal", "weighted US, recalibrated"), (voi, "value of information"), ("recal+wo", "AG + recal., interleaved"))
    by_view = dict(us_rows)
    for arm, label in labels:
        tables["r5_us"].append(f"{label} & {ci(by_view['malecns'][arm]['to_weight'])} & {ci(by_view['flywire'][arm]['to_weight'])}\\\\")
    by_wor = dict(wor_rows)
    for stratum in ("10-49", "50-149", "150-499", ">=500"):
        cells = []
        for v in VIEWS:
            s = by_wor[v]["strata"][stratum]
            cells += [ci(s["wr"]), ci(s["wor"])]
        name = stratum.replace(">=", "$\\geq$")
        tables["r5_wor"].append(f"{name} & {' & '.join(cells)}\\\\")
    return out, tables


# ---------------------------------------------------------------- BANC replication
BANC_MACROS = ("BancRecalCI", "BancAGCI", "BancClair", "BancWeightMedian", "BancUnresPct", "BancQuestions", "BancTypes")


def banc(data, claims):
    b = data["banc"]
    r = b["ratios"]
    claims("BANC: recalibrated AG with the topology prior saves little (within 10%)",
           0.9 <= r["mlp_s0:recal/weight"]["value"] <= 1.05)
    claims("BANC: constant priors reproduce weight order", b["const_half:recal==weight"] == 1.0 and b["const_share:ag==weight"] == 1.0)
    out = dict(BancRecalCI=ci3(r["mlp_s0:recal/weight"]), BancAGCI=ci3(r["mlp_s0:ag/weight"]), BancClair=f2(r["clairvoyant/weight"]["value"]), BancWeightMedian=f"{b['weight_median_checks']:.0f}",
               BancUnresPct=f"{100 * b['uncertifiable_share']:.1f}", BancQuestions=thousands(b["questions"]),
               BancTypes=thousands(b["types"]))
    return out, {"composition3": composition3(data, claims)}


def composition3(data, claims):
    cols = [data["frozen"]["malecns"], data["frozen"]["flywire"], data["banc"]]
    rows = [("Held-out cell types", lambda c: thousands(c["types"])),
            ("Target neurons", lambda c: thousands(c["targets"])),
            ("Questions", lambda c: thousands(c["questions"])),
            ("Median input cells per target", lambda c: f"{c['median_input_cells']:.0f}"),
            ("Unresolved input cells (\\%)", lambda c: f"{100 * c['unresolved_cells_share']:.1f}"),
            ("\\quad their input synapses (\\%)", lambda c: f"{100 * c['unresolved_synapses_share']:.1f}"),
            ("Answer yes (\\%)", lambda c: f"{100 * c['yes_share']:.1f}"),
            ("Answer no (\\%)", lambda c: f"{100 * c['no_share']:.1f}"),
            ("Cannot be certified (\\%)", lambda c: f"{100 * c['uncertifiable_share']:.1f}"),
            ("Checked by weight order$^\\ast$ (\\%)", lambda c: f"{100 * c['weight_mean_fraction_read']:.0f}")]
    lines = [f"{name} & " + " & ".join(fmt(c) for c in cols) + "\\\\" for name, fmt in rows]
    two = (PAPER / "tables/composition.tex").read_text().split("\n") if data["split"] == "test" else []
    for mine, frozen in zip(lines, two):
        claims(f"composition row agrees with the pre-specified table: {frozen}",
               mine.split(" & ")[1:3] == frozen.rstrip("\\\\").split(" & ")[1:] and mine.count(" & ") == 3)
    return lines


# ---------------------------------------------------------------- priors: seeded priors and correlated errors
PRI_MACROS = ("SeedTenRecalMale", "SeedTenRecalFly", "SeedTenVsProvCI", "SeedBeyondLow", "SeedBeyondHigh",
              "CorrShiftMax", "CorrThreshMax")


def priors(data, claims):
    out, seed_rows, corr_rows = {}, [], []
    views = data["views"]
    beyond, shifts, crossings = [], [], []
    for view in VIEWS:
        s = views[view]["seeded"]
        ten = s["ratio_to_weight"]["seeded10:recal"]
        claims(f"{view}: the seeded prior saves at least a quarter", ten["high"] < 0.75)
        out[f"SeedTenRecal{TAG[view]}"] = f2(ten["value"])
        b = views[view]["seeded_vs_controls"]["seeded10:recal/seedmlp10:recal"]
        claims(f"{view}: the scorer saves beyond the seeds themselves", b["high"] < 1)
        beyond.append(b["value"])
        for f in (10, 30, 50):
            seed_rows.append((view, f, s["ratio_to_weight"][f"seeded{f}:recal"], s["ratio_to_weight"][f"seeded{f}conn:recal"],
                              views[view]["seeded_vs_controls"][f"seeded{f}:recal/seedmlp{f}:recal"]))
        c = views[view]["correlated"]
        for form in ("wplus", "wminus", "rtype"):
            shifts.append(abs(c["shifts"][f"base/{form}/ag"]["0.95"]["median"]))
        for form in ("independent", "wplus", "wminus", "rtype"):
            x = c["curves"][f"base/{form}/ag"]["crossing"]["0.95"]
            claims(f"{view}/{form}: calibrated AG reaches a 5% saving inside the sweep", x["share_never"] == 0 and x["censored"] == 0)
            crossings.append(x["value"])
            corr_rows.append((view, form, x))
    prov = views["flywire"]["seeded"]["contrasts"]["seeded10:recal/provisional:recal"]
    claims("FlyWire: the seeded prior beats the earlier-release prior", prov["high"] < 1)
    claims("correlated errors move the threshold by little", max(shifts) <= 0.05)
    claims("the threshold stays near 0.8", max(crossings) <= 0.85)
    out.update(SeedTenVsProvCI=ci(prov), SeedBeyondLow=f2(min(beyond)), SeedBeyondHigh=f2(max(beyond)),
               CorrShiftMax=up(max(shifts), 2), CorrThreshMax=up(max(crossings), 2))
    names = {"malecns": "MaleCNS", "flywire": "FlyWire"}
    forms = {"independent": "independent", "wplus": "heavy cells predicted better", "wminus": "heavy cells predicted worse",
             "rtype": "errors shared within a type"}
    tables = {
        "r5_seeded": [f"{names[v]} & {f}\\% & {ci(a)} & {ci(c)} & {ci(b)}\\\\" for v, f, a, c, b in seed_rows],
        "r5_correlated": [f"{forms[form]} & " + " & ".join(ci(x) for _, fm, x in corr_rows if fm == form) + "\\\\"
                          for form in forms],
    }
    return out, tables


# ---------------------------------------------------------------- cost models
COST_MACROS = ("BatchWeightLow", "BatchWeightHigh", "BatchTopoLow", "BatchTopoHigh", "BatchProvRecal", "BatchProvAG",
               "HeteroLowPct", "HeteroHighPct")


def costs(data, claims):
    weight = [data[v]["batch"]["batch_over_cell_weight"]["b:weight"]["value"] for v in VIEWS]
    topo = [data[v]["batch"]["batch_model"]["b:mlp_s0:recal"]["value"] for v in VIEWS]
    fly = data["flywire"]["batch"]["batch_model"]
    claims("batch cost: the topology prior saves nothing (within 3%)", 0.97 <= min(topo) and max(topo) <= 1.03)
    claims("batch cost: recalibrated AG with earlier labels saves", fly["b:provisional:recal"]["high"] < 1)
    claims("batch cost: AG with earlier labels costs more than weight order", fly["b:provisional:ag"]["value"] > 1)
    saves, rows = [], []
    for view in VIEWS:
        for setting, h in data[view]["hetero"].items():
            for arm, r in h["aware_over_plain"].items():
                saves.append(1 - r["value"])
                rows.append((view, setting, arm, r))
    claims("heterogeneous cost: every cost-aware rule saves", all(s > 0 for s in saves))
    out = dict(BatchWeightLow=f2(min(weight)), BatchWeightHigh=f2(max(weight)), BatchTopoLow=f2(min(topo)),
               BatchTopoHigh=f2(max(topo)), BatchProvRecal=f2(fly["b:provisional:recal"]["value"]),
               BatchProvAG=f2(fly["b:provisional:ag"]["value"]),
               HeteroLowPct=f"{100 * min(saves):.0f}", HeteroHighPct=f"{100 * max(saves):.0f}")
    view_names = {"malecns": "MaleCNS", "flywire": "FlyWire"}
    batch_rows = []
    for arm, label in (("b:weight", "weight order"), ("b:mlp_s0:ag", "AG, topology"), ("b:mlp_s0:recal", "AG + recal., topology"),
                       ("b:provisional:ag", "AG, earlier release"), ("b:provisional:recal", "AG + recal., earlier release"),
                       ("b:clairvoyant", "clairvoyant bound")):
        cells = []
        for v in VIEWS:
            bm = data[v]["batch"]
            if arm == "b:weight":
                cells += ["1", ci(bm["batch_over_cell_weight"][arm])]
            elif arm in bm["batch_model"]:
                cells += [ci(bm["batch_model"][arm]), ci(bm["batch_over_cell_weight"][arm])]
            else:
                cells += ["---", "---"]
        batch_rows.append(f"{label} & {' & '.join(cells)}\\\\")
    hetero_rows = []
    arms = ("weight", "mlp_s0:ag", "mlp_s0:recal", "provisional:ag", "provisional:recal")
    for v in VIEWS:
        for setting, h in data[v]["hetero"].items():
            proxy, c = setting.split("@")
            cells = [ci(h["aware_over_plain"][a]) if a in h["aware_over_plain"] else "---" for a in arms]
            hetero_rows.append(f"{view_names[v]} & {proxy} & {c} & {' & '.join(cells)}\\\\")
    return out, {"r5_batch": batch_rows, "r5_hetero": hetero_rows}


TABLE_COLUMNS = {"r5_identification": 4, "r5_dp": 7, "r5_us": 3, "r5_wor": 5, "r5_seeded": 5, "r5_correlated": 3,
                 "r5_batch": 5, "r5_hetero": 8, "r5_expectations": 3, "r5_expectations_failed": 3}
# ---------------------------------------------------------------- answers of certifiable questions (pre-specified analysis)
ANS_MACROS = ("YesShareMale", "YesShareFly")


def answers(data, claims):
    out = {}
    for v in VIEWS:
        c = data[v]["composition"]
        share = c["positive"] / (c["positive"] + c["negative"])
        claims(f"{v}: about nine in ten certifiable questions have the answer yes", 0.85 <= share <= 0.95)
        out[f"YesShare{TAG[v]}"] = f"{100 * share:.0f}"
    return out, {}


EXP_MACROS = ("RFiveExpTotal", "RFiveExpHeld")


def expectations(data, claims):
    total = sum(len(v) for v in data.values())
    held = sum(e["holds"] for v in data.values() for e in v)
    return dict(RFiveExpTotal=str(total), RFiveExpHeld=str(held)), {}


def v35_macros():
    text = (PAPER / "v35_numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("test", "development"))
    split = ap.parse_args().split
    claims = Claims()
    macros, tables = {}, {}
    for path, names, build in (
        (R5 / f"costs/bio-{split}.json", PI_MACROS, partial_identification),
        (R5 / f"policies/analysis-{split}.json", POL_MACROS, policies),
        (R5 / f"banc/report-{split}.json", BANC_MACROS, banc),
        (R5 / f"priors/analysis/{split}.json", PRI_MACROS, priors),
        (R5 / f"costs/analysis-{split}.json", COST_MACROS, costs),
        (R5 / f"expectations-{split}.json", EXP_MACROS, expectations),
        (ROOT / f"outputs/v35/analysis/{split}.json", ANS_MACROS, answers),
    ):
        out, tab = source(path, split, names, claims, build)
        macros.update(out)
        tables.update(tab or {})
    if "composition3" not in tables:  # BANC not run yet: the two pre-specified columns and a placeholder
        rows = (PAPER / "tables/composition.tex").read_text().strip().split("\n")
        tables["composition3"] = [r[: -len("\\\\")] + f" & {MISSING}\\\\" for r in rows]
    for name, k in TABLE_COLUMNS.items():
        written_elsewhere = name.startswith("r5_expectations") and (PAPER / f"tables/{name}.tex").exists() and (R5 / f"expectations-{split}.json").exists()
        if name not in tables and not written_elsewhere:
            tables[name] = [f"\\multicolumn{{{k}}}{{@{{}}l}}{{{MISSING}}}\\\\"]
    if claims.failed:
        raise SystemExit("claims of the paper no longer hold:\n  " + "\n  ".join(claims.failed))
    clash = set(macros) & set(v35_macros())
    if clash:
        raise SystemExit(f"macros defined twice: {sorted(clash)}")
    suffix = "" if split == "test" else "-development"
    lines = [f"% Generated by scripts/r5_paper_assets.py ({split} split); do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (PAPER / f"r5_numbers{suffix}.tex").write_text("\n".join(lines) + "\n")
    for name, rows in tables.items():
        (PAPER / f"tables/{name}{suffix}.tex").write_text("\n".join(rows) + "\n")
    missing = sum(v == MISSING for v in macros.values())
    print(f"{split}: {len(macros)} macros ({missing} missing), {len(tables)} tables")


if __name__ == "__main__":
    main()
