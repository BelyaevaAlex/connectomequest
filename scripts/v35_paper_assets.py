#!/usr/bin/env python3
"""Render every number, table and figure of the manuscript from verified results.

Nothing is typed by hand: inline numbers become TeX macros, tables are written as
TeX bodies and figures are drawn from the analysis JSON files. Reads only result
files; runs no experiment.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from connectomequest.annotation_v35 import engine as e

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#d9d8d2", "#c9c8c1"
VIEW_NAME = {"malecns": "MaleCNS", "flywire": "FlyWire"}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def thousands(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


class Macros:
    def __init__(self):
        self.lines = []

    def add(self, name, value):
        if not name.isalpha():
            raise ValueError("macro names must be alphabetic: " + name)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def ratio(self, name, r, digits=2):
        self.add(name, f"{r['value']:.{digits}f}")
        self.add(name + "CI", f"{r['value']:.{digits}f} [{r['low']:.{digits}f}, {r['high']:.{digits}f}]")
        change = 100 * (r["value"] - 1)
        self.add(name + "Pct", f"{abs(change):.1f}")


def view_key(view):
    return {"malecns": "Male", "flywire": "Fly"}[view]


def numbers(test, dev, manifest, pilot):
    m = Macros()
    counts = manifest["counts"]
    m.add("TypesAll", thousands(counts["types"]))
    m.add("TypesDev", thousands(counts["development"]))
    m.add("TypesTest", thousands(counts["test"]))
    for split, data, tag in (("test", test, ""), ("development", dev, "Dev")):
        total_sessions = 0
        for view in ("malecns", "flywire"):
            v, c = view_key(view) + tag, data[view]["composition"]
            total_sessions += c["sessions"]
            m.add(v + "Neurons", thousands(c["targets"]))
            m.add(v + "Questions", thousands(c["sessions"]))
            m.add(v + "MedianInputs", f"{c['median_inputs']:.0f}")
            m.add(v + "MeanInputs", f"{c['mean_inputs']:.0f}")
            m.add(v + "UnresCells", pct(c["unresolved_cell_fraction"]))
            m.add(v + "UnresWeight", pct(c["unresolved_weight_fraction"]))
            m.add(v + "PosPct", pct(c["positive"] / c["sessions"]))
            m.add(v + "NegPct", pct(c["negative"] / c["sessions"]))
            m.add(v + "UnresPct", pct(c["unresolved"] / c["sessions"]))
            m.add(v + "Certifiable", thousands(c["positive"] + c["negative"]))
            m.add(v + "WeightObs", thousands(c["total_weight_order_observations"]))
            methods, contrasts = data[view]["methods"], data[view]["contrasts"]
            m.add(v + "WeightFraction", pct(methods["weight"]["certifiable"]["mean_fraction_read"]["value"], 0))
            m.add(v + "ClairFraction", pct(methods["clairvoyant"]["certifiable"]["mean_fraction_read"]["value"], 0))
            for key, name in (("positive", "Pos"), ("negative", "Neg")):
                m.ratio(v + "MlpRecal" + name, methods["mlp_s0:recal"][key]["ratio_to_weight"])
                m.ratio(v + "MlpAG" + name, methods["mlp_s0:ag"][key]["ratio_to_weight"])
            for key, name in (
                ("random", "Random"),
                ("clairvoyant", "Clair"),
                ("stat", "Stat"),
                ("mlp_s0:member", "MlpMember"),
                ("mlp_s0:direction", "MlpDirection"),
                ("mlp_s0:ag", "MlpAG"),
                ("mlp_s0:adg", "MlpADG"),
                ("mlp_s0:recal", "MlpRecal"),
                ("mlp_s0:ggn", "MlpGGN"),
                ("mlp_s0:adir", "MlpAdir"),
                ("mlp_s0:oracle-direction", "MlpOracle"),
                ("linear:ag", "LinAG"),
                ("linear:recal", "LinRecal"),
                ("ntac:ag", "NtacAG"),
                ("ntac:recal", "NtacRecal"),
                ("ntac:member", "NtacMember"),
                ("provisional:member", "ProvMember"),
                ("provisional:direction", "ProvDirection"),
                ("provisional:ag", "ProvAG"),
                ("provisional:recal", "ProvRecal"),
                ("provisional:oracle-direction", "ProvOracle"),
                ("provisional:adir", "ProvAdir"),
                ("provisional:adg", "ProvADG"),
                ("mlp_s0:nonmember", "MlpNonmember"),
                ("mlp_s0:confident", "MlpConfident"),
                ("linear:adg", "LinADG"),
                ("ntac:adg", "NtacADG"),
                ("ntac:direction", "NtacDirection"),
                ("linear:direction", "LinDirection"),
                ("linear:member", "LinMember"),
            ):
                if key in methods:
                    m.ratio(v + name, methods[key]["certifiable"]["ratio_to_weight"])
                    m.ratio(v + name + "All", methods[key]["all"]["ratio_to_weight"])
            for key, name in (
                ("mlp_s0:recal/ag", "MlpRecalOverAG"),
                ("mlp_s0:adg/ag", "MlpADGOverAG"),
                ("mlp_s0:adir/ag", "MlpAdirOverAG"),
                ("mlp_s0:ag/direction", "MlpAGOverDirection"),
                ("mlp_s0:ag/member", "MlpAGOverMember"),
                ("linear:ag/mlp_s0:ag", "LinOverMlp"),
                ("ntac:ag/mlp_s0:ag", "NtacOverMlp"),
                ("provisional:ag/direction", "ProvAGOverDirection"),
                ("provisional:ag/direction@positive", "ProvAGOverDirectionPos"),
                ("provisional:recal/ag", "ProvRecalOverAG"),
                ("linear:recal/ag", "LinRecalOverAG"),
                ("ntac:recal/ag", "NtacRecalOverAG"),
                ("provisional:adg/ag", "ProvADGOverAG"),
            ):
                if key in contrasts:
                    m.ratio(v + name, contrasts[key])
            seeds = methods["mlp:ag"]
            m.add(v + "SeedAGMean", f"{seeds['seed_mean']:.3f}")
            m.add(v + "SeedAGSd", f"{seeds['seed_sd']:.3f}")
            m.add(v + "SeedMemberSd", f"{methods['mlp:member']['seed_sd']:.3f}")
            st = data[view]["statistical"]
            m.add(v + "StatWrong", pct(st["wrong_rate"], 2))
            m.add(v + "StatAudits", thousands(st["audits"]))
            m.add(v + "StatRisk", pct(st["share_using_risk"]))
            m.add(v + "StatWrongUnres", pct(st["wrong_rate_among_unresolvable"], 2))
            certifiable_audits = st["audits"] - st["unresolvable_audits"]
            m.add(v + "StatWrongCert", pct((st["wrong_certificates"] - st["certified_on_unresolvable"]) / certifiable_audits, 2))
            big = data[view]["strata"]["inputs >=500"]
            m.ratio(v + "StatBig", big["methods"]["stat"])
            m.add(v + "BigQuestions", thousands(big["positive"] + big["negative"]))
            m.ratio(v + "StatSmall", data[view]["strata"]["inputs 10-49"]["methods"]["stat"])
            cal = data[view]["calibration"]
            for prior, name in (("mlp_s0", "Mlp"), ("linear", "Lin"), ("ntac", "Ntac"), ("provisional", "Prov")):
                if prior in cal:
                    m.add(v + name + "Auroc", f"{cal[prior]['member_auroc_within_question']:.2f}")
                    m.add(v + name + "Ece", f"{cal[prior]['member_ece']:.2f}")
                    m.add(v + name + "PredUnres", pct(cal[prior]["mean_predicted_unresolved"]))
            m.add(v + "ObsUnres", pct(cal["mlp_s0"]["observed_unresolved_rate"]))
            cons = data[view]["conservation"]
            for rho, tag2 in (("1/4", "Quarter"), ("1/2", "Half"), ("3/4", "ThreeQ"), ("1/1", "Full")):
                for fam, ftag in (("top1", "Top"), ("coalition", "Coal")):
                    c2 = cons[f"{fam}@rho={rho}"]
                    m.add(v + ftag + tag2 + "Pos", pct(c2["positive"]))
                    m.add(v + ftag + tag2 + "Neg", pct(c2["negative"]))
                    m.add(v + ftag + tag2 + "Unres", pct(c2["unresolved"]))
                    m.add(v + ftag + tag2 + "YesAmongCert", pct(c2["positive"] / (c2["positive"] + c2["negative"]), 0))
            mech = data[view]["mechanism"]
            for key, name in (("weight", "Weight"), ("mlp_s0:ag", "MlpAG"), ("provisional:ag", "ProvAG"), ("provisional:direction", "ProvDirection"), ("provisional:recal", "ProvRecal")):
                if key in mech:
                    m.add(v + "PaidNon" + name, pct(mech[key]["nonmember"]))
            for stratum, name in (("family=top1", "TopOnly"), ("family=coalition", "CoalOnly")):
                st2 = data[view]["strata"][stratum]["methods"]
                m.ratio(v + name + "MlpRecal", st2["mlp_s0:recal"])
                m.ratio(v + name + "Clair", st2["clairvoyant"])
                if "provisional:direction" in st2:
                    m.ratio(v + name + "ProvDirection", st2["provisional:direction"])
        m.add("Questions" + ("Test" if not tag else "Dev"), thousands(total_sessions))
    for key, value in pilot.items():
        m.add(key, value)
    (PAPER / "v35_numbers.tex").write_text("% Generated by scripts/v35_paper_assets.py; do not edit.\n" + "\n".join(m.lines) + "\n")


def ci(r, digits=2):
    return f"{r['value']:.{digits}f} [{r['low']:.{digits}f}, {r['high']:.{digits}f}]"


def table_composition(test):
    c = {view: test[view]["composition"] for view in ("malecns", "flywire")}
    w = {view: test[view]["methods"]["weight"]["certifiable"]["mean_fraction_read"]["value"] for view in c}

    def row(label, f):
        return f"{label} & {f('malecns')} & {f('flywire')}\\\\"

    rows = [
        row("Held-out cell types", lambda v: thousands(c[v]["target_types"])),
        row("Target neurons", lambda v: thousands(c[v]["targets"])),
        row("Questions", lambda v: thousands(c[v]["sessions"])),
        row("Input cells per target, median", lambda v: f"{c[v]['median_inputs']:.0f}"),
        row("Unresolved input cells, pooled (\\%)", lambda v: pct(c[v]["unresolved_cell_fraction"])),
        row("\\quad their share of input synapses (\\%)", lambda v: pct(c[v]["unresolved_weight_fraction"])),
        row("Answer yes (\\%)", lambda v: pct(c[v]["positive"] / c[v]["sessions"])),
        row("Answer no (\\%)", lambda v: pct(c[v]["negative"] / c[v]["sessions"])),
        row("Cannot be certified (\\%)", lambda v: pct(c[v]["unresolved"] / c[v]["sessions"])),
        row("Cells checked by weight order$^\\ast$ (\\%)", lambda v: pct(w[v], 0)),
    ]
    (PAPER / "tables/composition.tex").write_text("\n".join(rows) + "\n")


MAIN_ROWS = (
    (None, None, "Without a prior"),
    ("\\quad random order", "random", None),
    ("\\quad sampling test ($\\alpha{=}0.05$)", "stat", None),
    (None, None, "Topology prior (weak)"),
    ("\\quad member-first", "mlp_s0:member", None),
    ("\\quad direction-fixed", "mlp_s0:direction", None),
    ("\\quad Adaptive Greedy (AG)", "mlp_s0:ag", None),
    ("\\quad Adaptive Dual Greedy", "mlp_s0:adg", None),
    ("\\quad AG + recalibration", "mlp_s0:recal", None),
    ("\\quad \\quad + answer propagation$^\\dagger$", "coup:recal", None),
    (None, None, "Earlier-release prior (strong)"),
    ("\\quad member-first", "provisional:member", None),
    ("\\quad direction-fixed", "provisional:direction", None),
    ("\\quad Adaptive Greedy (AG)", "provisional:ag", None),
    ("\\quad AG + recalibration", "provisional:recal", None),
    (None, None, "Knows every answer"),
    ("\\quad clairvoyant bound", "clairvoyant", None),
)


def table_main(test, post):
    lines = []
    coupling = read(ROOT / "outputs/neurosym/B/heldout/analysis-test.json")["views"]
    for label, key, header in MAIN_ROWS:
        if header:
            lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{header}}}}}\\\\")
            continue
        cells = []
        for view in ("malecns", "flywire"):
            if key.startswith("coup:"):
                cells.append(ci(coupling[view]["b4"][key[5:]]["to_weight"]))
            elif key == "post:hybrid":
                cells.append(ci(post[view]["hybrid"]["ratio_to_weight"]))
            elif key.startswith("post:"):
                control = post[view]["controls"].get(key[5:])
                cells.append(ci(control["ratio_to_weight"]) if control else "---")
            else:
                method = test[view]["methods"].get(key)
                cells.append(ci(method["certifiable"]["ratio_to_weight"]) if method else "---")
        lines.append(f"{label} & {cells[0]} & {cells[1]}\\\\")
    (PAPER / "tables/main.tex").write_text("\n".join(lines) + "\n")


def table_full(data, name):
    """Every arm: certifiable ratio with interval, all-question ratio, mean fraction read."""
    lines = []
    keys = ["random", "stat", "clairvoyant"]
    for prior in ("mlp_s0", "linear", "ntac", "provisional"):
        keys += [f"{prior}:{p}" for p in ("member", "nonmember", "confident", "direction", "ggn", "ag", "adg", "recal", "adir", "adir-recal", "oracle-direction")]
    pretty = {
        "random": "Random order", "stat": "Sampling test", "clairvoyant": "Clairvoyant bound", "member": "member-first",
        "nonmember": "nonmember-first", "confident": "confident-first", "direction": "direction-fixed", "ggn": "static halfspace order",
        "ag": "Adaptive Greedy", "adg": "Adaptive Dual Greedy", "recal": "AG + recalibration", "adir": "single-goal adaptive",
        "adir-recal": "single-goal adaptive + recal.", "oracle-direction": "oracle direction (diagnostic)",
    }
    prior_name = {"mlp_s0": "MLP", "linear": "Linear", "ntac": "NTAC", "provisional": "Earlier release"}
    for key in keys:
        cells = []
        for view in ("malecns", "flywire"):
            method = data[view]["methods"].get(key)
            if method is None:
                cells += ["---", "---", "---"]
            else:
                cells += [
                    ci(method["certifiable"]["ratio_to_weight"]),
                    f"{method['all']['ratio_to_weight']['value']:.3f}",
                    f"{method['certifiable']['mean_fraction_read']['value']:.3f}",
                ]
        label = pretty[key] if ":" not in key else prior_name[key.split(":")[0]] + ", " + pretty[key.split(":")[1]]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    (PAPER / f"tables/{name}.tex").write_text("\n".join(lines) + "\n")


def table_strata(test):
    lines = []
    order = (
        ("rho=1/4", "$\\rho=1/4$"), ("rho=1/2", "$\\rho=1/2$"), ("rho=3/4", "$\\rho=3/4$"), ("rho=1/1", "$\\rho=1$"),
        ("family=top1", "strongest partner"), ("family=coalition", "majority coalition"),
        ("inputs 10-49", "10--49 inputs"), ("inputs 50-149", "50--149 inputs"), ("inputs 150-499", "150--499 inputs"), ("inputs >=500", "$\\geq$500 inputs"),
        ("positive margin margin<0.5", "margin $<0.5$"), ("positive margin 0.5-1", "margin 0.5--1"), ("positive margin 1-3", "margin 1--3"), ("positive margin >=3", "margin $\\geq 3$"),
    )
    for view in ("malecns", "flywire"):
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{VIEW_NAME[view]}}}}}\\\\")
        for key, label in order:
            s = test[view]["strata"][key]
            ms = s["methods"]

            def val(name):
                return f"{ms[name]['value']:.2f}" if name in ms else "---"

            lines.append(
                f"{label} & {thousands(s['positive'] + s['negative'])} & {s['weight_fraction_read']['value']:.2f} & {val('clairvoyant')} & "
                f"{val('mlp_s0:ag')} & {val('mlp_s0:recal')} & {val('provisional:direction')} & {val('stat')}\\\\"
            )
    (PAPER / "tables/strata.tex").write_text("\n".join(lines) + "\n")


def table_calibration(test):
    lines = []
    names = {"mlp_s0": "MLP (seed 0)", "linear": "Linear", "ntac": "NTAC", "provisional": "Earlier release"}
    for prior, label in names.items():
        cells = []
        for view in ("malecns", "flywire"):
            c = test[view]["calibration"].get(prior)
            cells += ["---"] * 4 if c is None else [
                f"{c['member_auroc_within_question']:.3f}", f"{c['member_brier']:.3f}", f"{c['member_ece']:.3f}", pct(c["mean_predicted_unresolved"])
            ]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/calibration.tex").write_text("\n".join(lines) + "\n")


def table_conservation(test):
    lines = []
    for fam, label in (("top1", "Strongest partner"), ("coalition", "Majority coalition")):
        for rho in ("1/4", "1/2", "3/4", "1/1"):
            cells = []
            for view in ("malecns", "flywire"):
                c = test[view]["conservation"][f"{fam}@rho={rho}"]
                cells.append(f"{pct(c['positive'])} / {pct(c['negative'])} / {pct(c['unresolved'])}")
            shown = "1" if rho == "1/1" else rho
            lines.append(f"{label} & ${shown}$ & {cells[0]} & {cells[1]}\\\\")
    (PAPER / "tables/conservation.tex").write_text("\n".join(lines) + "\n")


def table_hypotheses(test):
    """Full table (supplement) and a compact per-connectome verdict table (main text)."""
    views = ("malecns", "flywire")

    def r(view, key):
        return test[view]["contrasts"].get(key) or test[view]["methods"][key]["certifiable"]["ratio_to_weight"]

    def verdict(checks):
        """Confirmed when the whole interval satisfies the prediction; else point estimate only."""
        point = all(f(c["value"]) for c, f in checks)
        whole = all(f(c["low"]) and f(c["high"]) for c, f in checks)
        return "confirmed" if whole else ("point estimate only" if point else "not confirmed")

    below, above = (lambda x: x < 1), (lambda x: x > 1)
    band = lambda x: 0.97 <= x <= 1.03
    big = {v: test[v]["strata"]["inputs >=500"]["methods"]["stat"] for v in views}
    wrong = {v: test[v]["statistical"]["wrong_rate"] for v in views}
    # (label, long statement, short statement, per-view evidence text, per-view verdict)
    spec = [
        ("H1", "Topology prior: AG below weight order and member-first above it", "weak prior: AG $<$ weight order $<$ member-first",
         lambda v: f"AG {ci(r(v, 'mlp_s0:ag'), 3)}; member-first {ci(r(v, 'mlp_s0:member'), 3)}",
         lambda v: verdict([(r(v, "mlp_s0:ag"), below), (r(v, "mlp_s0:member"), above)]), views),
        ("H2", "Earlier-release direction-fixed below topology AG", "strong prior, fixed order $<$ weak prior, AG",
         lambda v: f"{ci(r(v, 'provisional:direction'), 3)} vs.\\ {ci(r(v, 'mlp_s0:ag'), 3)}",
         lambda v: "confirmed" if r(v, "provisional:direction")["high"] < r(v, "mlp_s0:ag")["low"] else "point estimate only", ("flywire",)),
        ("H3", "Earlier release: AG above direction-fixed, and recalibrated AG below AG", "strong prior: fixed order $<$ AG; recal.\\ $<$ AG",
         lambda v: f"AG/fixed {ci(r(v, 'provisional:ag/direction'), 3)}; recal./AG {ci(r(v, 'provisional:recal/ag'), 3)}",
         lambda v: verdict([(r(v, "provisional:ag/direction"), above), (r(v, "provisional:recal/ag"), below)]), ("flywire",)),
        ("H4", "Topology prior: recalibrated AG at most 1.02 of AG", "weak prior: recal.\\ $\\leq 1.02\\,$AG",
         lambda v: ci(r(v, "mlp_s0:recal/ag"), 3), lambda v: verdict([(r(v, "mlp_s0:recal/ag"), lambda x: x <= 1.02)]), views),
        ("H5", "Linear model within $\\pm3\\%$ of the MLP under AG", "linear $=$ MLP within $\\pm3\\%$",
         lambda v: ci(r(v, "linear:ag/mlp_s0:ag"), 3), lambda v: verdict([(r(v, "linear:ag/mlp_s0:ag"), band)]), views),
        ("H6", "Adaptive Dual Greedy within $\\pm3\\%$ of AG", "ADG $=$ AG within $\\pm3\\%$",
         lambda v: ci(r(v, "mlp_s0:adg/ag"), 3), lambda v: verdict([(r(v, "mlp_s0:adg/ag"), band)]), views),
        ("H7a", "Sampling test at least 0.9 of weight order", "sampling test $\\geq 0.9$ weight order",
         lambda v: ci(r(v, "stat"), 3), lambda v: verdict([(r(v, "stat"), lambda x: x >= 0.9)]), views),
        ("H7b", "Sampling test below weight order for targets with $\\geq$500 input cells", "sampling test $<$ weight order, $\\geq$500 inputs",
         lambda v: ci(big[v], 3), lambda v: verdict([(big[v], below)]), views),
        ("H7c", "Wrong sampling certificates in at most 5\\% of audits", "wrong sampling certificates $\\leq 5\\%$",
         lambda v: pct(wrong[v], 2) + "\\%", lambda v: "confirmed" if wrong[v] <= 0.05 else "not confirmed", views),
        ("H8", "Static halfspace order above weight order", "static halfspace order $>$ weight order",
         lambda v: ci(r(v, "mlp_s0:ggn"), 3), lambda v: verdict([(r(v, "mlp_s0:ggn"), above)]), views),
        ("H9", "Topology prior: single-goal adaptive rule above AG", "weak prior: single-goal $>$ AG",
         lambda v: ci(r(v, "mlp_s0:adir/ag"), 3), lambda v: verdict([(r(v, "mlp_s0:adir/ag"), above)]), views),
    ]
    mark = {"confirmed": "$\\bullet$", "point estimate only": "$\\circ$", "not confirmed": "$\\times$"}
    long_rows, short_rows, verdicts = [], [], {}
    for label, long, short, evidence, decide, scope in spec:
        outcome = {v: decide(v) for v in scope}
        verdicts[label] = outcome
        cells = [evidence(v) if v in scope else "---" for v in views]
        marks = [mark[outcome[v]] if v in scope else "" for v in views]
        text = "; ".join(f"{VIEW_NAME[v]} {outcome[v]}" for v in scope) if len(set(outcome.values())) > 1 else next(iter(outcome.values()))
        long_rows.append(f"{label[1:]} & {long} & {cells[0]} & {cells[1]} & {text}\\\\")
        short_rows.append(f"{label} & {short} & {marks[0]} & {marks[1]}\\\\")
    (PAPER / "tables/hypotheses.tex").write_text("\n".join(long_rows) + "\n")
    (PAPER / "tables/hypotheses_short.tex").write_text("\n".join(short_rows) + "\n")
    flat = [o for outcome in verdicts.values() for o in outcome.values()]
    return dict(HypChecks=str(len(flat)), HypConfirmed=str(flat.count("confirmed")), HypPoint=str(flat.count("point estimate only")), HypFailed=str(flat.count("not confirmed")))


def agreement_assets():
    """Agreement of exact answers between neurons of a type and between connectomes."""
    a = read(ROOT / "outputs/v35/analysis/agreement-test.json")
    out = {}
    for view in ("malecns", "flywire"):
        w = a["within"][f"{view}:all"]
        out[view_key(view) + "WithinAgree"] = pct(w["agreement"])
        out[view_key(view) + "WithinKappa"] = f"{w['kappa']:.2f}"
        out[view_key(view) + "WithinPairs"] = thousands(w["decided"])
    x = a["across"]["all"]
    words = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
    count = len(a["strongest_partner_lost_in_both_at_half"])
    out["LostBothWord"] = words[count] if count < len(words) else str(count)
    full = [a["across"][f"{kind}@rho=1/1"]["agreement"] for kind in ("top1", "coalition")]
    out.update(AcrossFullLow=pct(min(full), 0), AcrossFullHigh=pct(max(full), 0))
    within = [a["within"][f"{view}:{kind}@rho=1/1"] for view in ("malecns", "flywire") for kind in ("top1", "coalition")]
    across = [a["across"][f"{kind}@rho=1/1"] for kind in ("top1", "coalition")]
    out.update(
        WithinFullAgreeLow=pct(min(x["agreement"] for x in within), 0), WithinFullAgreeHigh=pct(max(x["agreement"] for x in within), 0),
        WithinFullKappaLow=f"{min(x['kappa'] for x in within):.2f}", WithinFullKappaHigh=f"{max(x['kappa'] for x in within):.2f}",
        AcrossFullKappaLow=f"{min(x['kappa'] for x in across):.2f}", AcrossFullKappaHigh=f"{max(x['kappa'] for x in across):.2f}",
    )
    out.update(AcrossAgree=pct(x["agreement"]), AcrossKappa=f"{x['kappa']:.2f}", AcrossPairs=thousands(x["decided"]), LostBoth=str(len(a["strongest_partner_lost_in_both_at_half"])))
    lines = []
    for kind, label in (("top1", "Strongest partner"), ("coalition", "Majority coalition")):
        for rho in ("1/4", "1/2", "3/4", "1/1"):
            cells = []
            for group, key in (("within", f"malecns:{kind}@rho={rho}"), ("within", f"flywire:{kind}@rho={rho}"), ("across", f"{kind}@rho={rho}")):
                g = a[group][key]
                cells += [thousands(g["decided"]), pct(g["agreement"]), f"{g['kappa']:.2f}".replace("-0.00", "0.00")]
            lines.append(f"{label} & ${'1' if rho == '1/1' else rho}$ & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/agreement.tex").write_text("\n".join(lines) + "\n")
    lost = a["strongest_partner_lost_in_both_at_half"]
    out["LostBothList"] = "; ".join(f"{x['type'].replace('_', chr(92) + '_')} (partner {x['partner'].replace('_', chr(92) + '_')}, hemibrain share {100 * x['hemibrain_share']:.0f}\\%)" for x in lost)
    return out


def diagnostic_assets():
    """Input diagnostics and the lenient-reference sensitivity analysis (development split)."""
    diag = read(ROOT / "outputs/v35/analysis/diagnostics.json")
    lenient = read(ROOT / "outputs/v35/analysis/lenient-development.json")
    out = {}
    reasons = (
        ("release-specific type without correspondence", "ReleaseSpecific", "Release-specific type, no hemibrain correspondence"),
        ("several candidate hemibrain types", "Several", "Several candidate hemibrain types"),
        ("correspondence outside the vocabulary", "Outside", "Correspondence outside the vocabulary"),
        ("no type in the release", "NoType", "No type in the release"),
    )
    lines = []
    for key, tag, label in reasons:
        cells = []
        for view in ("malecns", "flywire"):
            r = diag["unresolved"][view]["reasons"][key]
            out[view_key(view) + "Unres" + tag] = pct(r["share_of_unresolved"])
            cells += [pct(r["share_of_unresolved"]), pct(r["weight_share_of_unresolved"])]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/unresolved.tex").write_text("\n".join(lines) + "\n")
    prov = diag["unresolved"]["flywire"]["provisional"]
    out.update(
        ProvTypedShare=pct(prov["share_hint_typed"]),
        ProvOutsideShare=pct(prov["share_hint_outside_vocabulary"]),
        ProvNoHintShare=pct(prov["share_without_any_hint"]),
        ProvAgreement=pct(prov["agreement_when_both_typed"]),
        ProvAgreementRounded=f"{100 * prov['agreement_when_both_typed']:.0f}",
        ProvTypedGivenResolved=pct(prov["typed_hint_given_reference_resolved"]),
        ProvResolvedGivenUntyped=pct(prov["reference_resolved_given_no_typed_hint"]),
    )
    pay = diag["provisional_payments_development"]
    out.update(PaySessions=thousands(pay["positive_sessions"]), PayFamilies=thousands(pay["sampled_families"]))
    classes = ("hint in the partner set", "hint: another type", "hint outside the vocabulary", "no hint")
    labels = ("A type in the partner set", "Another hemibrain type", "A type outside the vocabulary", "Nothing (no earlier name)")
    tags = ("Member", "Other", "Outside", "None")
    lines = []
    for cls, label, tag in zip(classes, labels, tags):
        comp = pay["input_composition"][cls]
        cells = [pct(comp["share"])]
        for policy, ptag in (("direction", "Dir"), ("ag", "AG"), ("recal", "Recal")):
            share = pay["paid_class_share"][policy][cls]
            out["Pay" + ptag + tag] = pct(share)
            cells.append(pct(share))
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule Cells checked / weight order & & " + " & ".join(f"{pay['cost_ratio_to_weight'][p]:.2f}" for p in ("direction", "ag", "recal")) + "\\\\")
    (PAPER / "tables/payments.tex").write_text("\n".join(lines) + "\n")
    other = pay["paid"]["ag"]["hint: another type"]
    out["PayOtherNonmember"] = pct(other["nonmember"] / sum(other.values()))
    lines = []
    rows = (("clairvoyant", "Clairvoyant bound", "Clair"), ("mlp_s0:member", "Topology, member-first", "MlpMember"), ("mlp_s0:direction", "Topology, direction-fixed", "MlpDirection"),
            ("mlp_s0:ag", "Topology, AG", "MlpAG"), ("mlp_s0:recal", "Topology, AG + recal.", "MlpRecal"), ("provisional:direction", "Earlier, direction-fixed", "ProvDirection"),
            ("provisional:ag", "Earlier, AG", "ProvAG"), ("provisional:recal", "Earlier, AG + recal.", "ProvRecal"))
    for view in ("malecns", "flywire"):
        v, a = view_key(view), lenient[view]["answers"]
        for kind, ktag in (("strict", "Strict"), ("lenient", "Len")):
            for answer, atag in (("positive", "Pos"), ("negative", "Neg"), ("unresolved", "Unres")):
                out[v + ktag + atag] = pct(a[kind][answer])
        out[v + "LenMoved"] = pct(lenient[view]["moved_from_unresolved"]["to_negative"])
        for key, _, tag in rows:
            if key in lenient[view]["ratio_to_weight"]:
                x = lenient[view]["ratio_to_weight"][key]
                out[v + "Len" + tag] = f"{x['value']:.2f}"
                out[v + "Len" + tag + "CI"] = ci(x)
    for key, label, _ in rows:
        cells = [ci(lenient[view]["ratio_to_weight"][key]) if key in lenient[view]["ratio_to_weight"] else "---" for view in ("malecns", "flywire")]
        lines.append(f"{label} & {cells[0]} & {cells[1]}\\\\")
    (PAPER / "tables/lenient.tex").write_text("\n".join(lines) + "\n")
    return out


def pilot_numbers():
    original = read(glob.glob(str(ROOT / "outputs/v33/transfer/reports/*/summary.json"))[0])
    corrected_dir = Path(glob.glob(str(ROOT / "outputs/v34/v33-fly-corrected/reports/*/"))[0])
    corrected = read(corrected_dir / "summary.json")
    changes = read(corrected_dir / "changes.json")
    o = {r["arm"]: r for r in original}
    c = {r["arm"]: r for r in corrected}
    rows = (("Weight order", "weight"), ("MLP, member-first", "mlp:member-fixed"), ("MLP + Adaptive Greedy", "mlp:adaptivegoal"), ("NTAC, member-first", "ntac:member-fixed"), ("NTAC + Adaptive Greedy", "ntac:adaptivegoal"))
    lines = [f"{label} & {thousands(o[k]['paid_attempts'])} & {thousands(c[k]['paid_attempts'])}\\\\" for label, k in rows]
    lines.append(f"Certified questions & {o['weight']['certified_queries']}/16 & {c['weight']['certified_queries']}/16\\\\")
    (PAPER / "tables/pilot.tex").write_text("\n".join(lines) + "\n")
    prim_o, comp_o = o["mlp:adaptivegoal"]["mean_c_b"], o["ntac:member-fixed"]["mean_c_b"]
    prim_c, comp_c = c["mlp:adaptivegoal"]["mean_c_b"], c["ntac:member-fixed"]["mean_c_b"]
    repair = read(glob.glob(str(ROOT / "outputs/v34/taxonomy-repair/reports/*/summary.json"))[0])
    parent = read(ROOT / "outputs/v34/development/reports/7da01631ee05c99f7f69226832028b3794261d277e7fbc0b8e8cdb596e7f31ef/summary.json")
    male = read(ROOT / "outputs/v34/development/reports/e82bd47be5e2f80328bed95d91d30f0f4547ce2fb75e2c68af3831bcfa4e399c/summary.json")

    def best(summary):
        rows2 = {r["arm"]: r for r in summary["rows"]}
        low = min(rows2.values(), key=lambda r: r["mean_c_b"])
        return rows2["weight"], low

    wm, bm = best(male)
    wf, bf = best(repair)
    return dict(
        PilotChangedCells=str(len(changes)),
        PilotOrigPct=f"{100 * (prim_o / comp_o - 1):.2f}",
        PilotCorrPct=f"{100 * (1 - prim_c / comp_c):.1f}",
        PilotOrigCert=f"{o['weight']['certified_queries']}",
        PilotCorrCert=f"{c['weight']['certified_queries']}",
        CompMaleQuestions=str(wm["planned_queries"]),
        CompFlyQuestions=str(wf["planned_queries"]),
        CompMaleBestPct=f"{100 * (1 - bm['mean_c_b'] / wm['mean_c_b']):.1f}",
        CompFlyBestPct=f"{100 * (1 - bf['mean_c_b'] / wf['mean_c_b']):.1f}",
        CompMaleHeadroomPct=f"{100 * (1 - wm['oracle_penalized_lower_bound'] / wm['mean_c_b']):.1f}",
        CompFlyHeadroomPct=f"{100 * (1 - wf['oracle_penalized_lower_bound'] / wf['mean_c_b']):.1f}",
        CompFlyUnresBefore=str({r["arm"]: r for r in parent["rows"]}["weight"]["unresolved"]),
        CompFlyUnresAfter=str(wf["unresolved"]),
        CompAliasCells="914",
    )


def coverage_assets():
    """Development-split sweep of the share of neurons that keep their earlier label (FlyWire)."""
    c = read(ROOT / "outputs/v35/analysis/coverage-development.json")
    levels = [f"{x:g}" for x in c["levels"]]
    names = (("member", "Member"), ("direction", "Dir"), ("ag", "AG"), ("adg", "ADG"), ("recal", "Recal"), ("adir", "Adir"))
    tags = dict(zip(levels, ("Zero", "Quarter", "Half", "ThreeQ", "Full")))
    out = dict(CovQuestions=thousands(c["questions"]), CovClair=f"{c['clairvoyant']['value']:.2f}")
    lines = []
    for level in levels:
        row = c["ratio_to_weight"][level]
        for key, tag in names:
            out["Cov" + tag + tags[level]] = f"{row[key]['value']:.2f}"
        out["CovAGOverDir" + tags[level]] = ci(c["ag_over_direction"][level])
        best = min(row[k]["value"] for k, _ in names)
        out["CovRecalGap" + tags[level]] = f"{100 * (row['recal']['value'] / best - 1):.1f}"
        lines.append(f"{100 * float(level):.0f} & " + " & ".join(ci(row[k]) for k, _ in names) + "\\\\")
    (PAPER / "tables/coverage.tex").write_text("\n".join(lines) + "\n")
    out["CovDirQuarterPct"] = f"{100 * (c['ratio_to_weight']['0.25']['direction']['value'] - 1):.0f}"
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    x = np.array([100 * float(v) for v in levels])
    series = (("direction", "direction-fixed", ORANGE, "s"), ("ag", "AG", BLUE, "o"), ("recal", "AG + recalibration", AQUA, "D"))
    for key, label, color, marker in series:
        y = np.array([c["ratio_to_weight"][v][key]["value"] for v in levels])
        lo = np.array([c["ratio_to_weight"][v][key]["low"] for v in levels])
        hi = np.array([c["ratio_to_weight"][v][key]["high"] for v in levels])
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.3, marker=marker, markersize=3.6, markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(1.0, color=INK, linewidth=0.6)
    ax.text(101, 1.0, " weight order", ha="left", va="center", fontsize=6.3, color=MUTED)
    ends = {key: c["ratio_to_weight"]["1"][key]["value"] for key, *_ in series}
    ax.text(102, ends["ag"], "AG", ha="left", va="center", fontsize=6.5, color=INK)
    ax.text(102, ends["recal"] + 0.022, "AG + recal.", ha="left", va="center", fontsize=6.5, color=INK)
    ax.text(102, ends["direction"] - 0.03, "direction-fixed", ha="left", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(-3, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.set_ylim(0.62, 1.3)
    ax.set_xlabel("neurons that keep their earlier label")
    ax.set_ylabel("cells checked / weight order")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    clean(ax)
    fig.subplots_adjust(right=0.74)
    fig.savefig(PAPER / "figures/coverage.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    return out


def posthoc_assets(post, post_dev, lenient):
    """Controls added after the held-out run, the three-way stop, the hybrid test, per-question distributions."""
    out = {}
    views = ("malecns", "flywire")
    for view in views:
        v, r = view_key(view), post[view]
        if r["integrity"]["mismatched"] or post_dev[view]["integrity"]["mismatched"]:
            raise SystemExit("post hoc costs differ from the frozen results")
        constant = [c for name, c in r["controls"].items() if name.startswith("const_")] + [c for name, c in post_dev[view]["controls"].items() if name.startswith("const_")]
        if any(c["share_identical_to_weight"] != 1.0 for c in constant):
            raise SystemExit("a no-prediction prior no longer reproduces weight order on every question")
        for name, tag in (("mlp_s0:rr", "RR"), ("ntac:rr", "NtacRR"), ("provisional:rr", "ProvRR")):
            if name in r["controls"]:
                out[v + tag] = f"{r['controls'][name]['ratio_to_weight']['value']:.2f}"
                out[v + tag + "CI"] = ci(r["controls"][name]["ratio_to_weight"])
        for name, tag in (("provisional:rr/direction", "ProvRROverDir"), ("provisional:rr/recal", "ProvRROverRecal"), ("mlp_s0:rr/recal", "RROverRecal")):
            if name in r["controls"]:
                out[v + tag] = f"{r['controls'][name]['value']:.2f}"
                out[v + tag + "CI"] = ci(r["controls"][name])
        t = r["three_way"]
        out[v + "UndFraction"] = pct(t["weight_fraction_of_inputs"]["value"], 0)
        out[v + "UndLB"] = f"{t['lower_bound_over_weight']['value']:.2f}"
        out[v + "UndQuestions"] = thousands(t["uncertifiable_questions"])
        und, every = t["uncertifiable_ratio_to_weight"], t["all_questions_ratio_to_weight"]
        adaptive = [und[f"{prior}:{policy}"]["value"] for prior in ("mlp_s0", "ntac") for policy in ("ag", "adg", "recal")]
        out[v + "UndAdaptiveMax"] = f"{max(adaptive):.3f}"
        out[v + "UndMlpMember"] = f"{und['mlp_s0:member']['value']:.2f}"
        out[v + "UndRandom"] = f"{und['random_r0']['value']:.2f}"
        out[v + "AllThreeMlpRecal"] = f"{every['mlp_s0:recal']['value']:.2f}"
        out[v + "AllThreeMlpRecalCI"] = ci(every["mlp_s0:recal"], 3)
        if "provisional:recal" in und:
            for name, tag in (("direction", "Direction"), ("recal", "Recal"), ("ag", "AG"), ("member", "Member")):
                out[v + "UndProv" + tag] = f"{und['provisional:' + name]['value']:.2f}"
                out[v + "AllThreeProv" + tag + "CI"] = ci(every["provisional:" + name])
                out[v + "AllThreeProv" + tag] = f"{every['provisional:' + name]['value']:.2f}"
        h = r["hybrid"]
        out[v + "Hybrid"] = f"{h['ratio_to_weight']['value']:.2f}"
        out[v + "HybridCI"] = ci(h["ratio_to_weight"])
        out[v + "HybridBigCI"] = ci(h["strata"][">=500"]["hybrid"])
        out[v + "HybridBigPct"] = f"{100 * (1 - h['strata']['>=500']['hybrid']['value']):.0f}"
        out[v + "HybridWrong"] = pct(h["wrong_rate"], 2)
        out[v + "HybridIdentical"] = pct(h["share_identical_to_weight"], 0)
        out[v + "DevHybridBigCI"] = ci(post_dev[view]["hybrid"]["strata"][">=500"]["hybrid"])
        q = r["per_question"]
        out[v + "WeightMedian"] = f"{q['weight_median_checks']:.0f}"
        out[v + "WeightEqClair"] = pct(q["weight_equals_clairvoyant"], 0)
        out[v + "LargeShareQuestions"] = pct(q["large_share_of_questions"])
        out[v + "LargeShareChecks"] = pct(q["large_share_of_weight_checks"], 0)
        for name, tag in (("mlp_s0:ag", "MlpAG"), ("mlp_s0:recal", "MlpRecal"), ("provisional:direction", "ProvDirection"), ("provisional:ag", "ProvAG"), ("provisional:recal", "ProvRecal")):
            if name in q["arms"]:
                a = q["arms"][name]
                out[v + tag + "Tied"] = pct(a["tied"], 0)
                out[v + tag + "Better"] = pct(a["better"], 0)
                out[v + tag + "Worse"] = pct(a["worse"], 0)
                out[v + tag + "Double"] = pct(a["at_least_double"])
                out[v + tag + "Median"] = f"{a['median']:.2f}"
                out[v + tag + "Geo"] = f"{a['geometric_mean']:.2f}"
                out[v + tag + "MeanRatio"] = f"{a['mean']:.2f}"
                out[v + tag + "Small"] = f"{a['ratio_of_totals_below_500_inputs']['value']:.2f}"
    small = [post[v]["hybrid"]["strata"][k]["hybrid"]["value"] for v in views for k in ("10-49", "50-149")]
    out["HybridSmallLow"], out["HybridSmallHigh"] = f"{min(small):.2f}", f"{max(small):.2f}"
    out["UndFractionLow"] = min(out["MaleUndFraction"], out["FlyUndFraction"])
    out["UndLBGapMax"] = f"{100 * (1 - min(post[v]['three_way']['lower_bound_over_weight']['value'] for v in views)):.0f}"
    out["WeightMedianLow"] = min(out["MaleWeightMedian"], out["FlyWeightMedian"], key=int)
    out["WeightMedianHigh"] = max(out["MaleWeightMedian"], out["FlyWeightMedian"], key=int)
    out["HybridBigPctLow"] = min(out["MaleHybridBigPct"], out["FlyHybridBigPct"], key=int)
    out["HybridBigPctHigh"] = max(out["MaleHybridBigPct"], out["FlyHybridBigPct"], key=int)
    out["UndFractionHigh"] = max(out["MaleUndFraction"], out["FlyUndFraction"])
    strict = read(ROOT / "outputs/v35/analysis/development.json")["flywire"]["methods"]["provisional:direction"]["certifiable"]["ratio_to_weight"]["value"]
    out["DevStrictSavingDirection"] = f"{100 * (1 - strict):.0f}"
    out["LenientSavingDirection"] = f"{100 * (1 - lenient['flywire']['ratio_to_weight']['provisional:direction']['value']):.0f}"
    out["LenientSavingRecal"] = f"{100 * (1 - lenient['flywire']['ratio_to_weight']['provisional:recal']['value']):.0f}"

    # Compact per-question table for the main text.
    lines = []
    compact = (("mlp_s0:ag", "Topology, AG"), ("mlp_s0:recal", "Topology, AG + recal."), ("provisional:direction", "Earlier, direction-fixed"), ("provisional:ag", "Earlier, AG"), ("provisional:recal", "Earlier, AG + recal."))
    for view in views:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{VIEW_NAME[view]}}}}}\\\\")
        for name, label in compact:
            a = post[view]["per_question"]["arms"].get(name)
            if a is None:
                continue
            lines.append(f"\\quad {label} & {a['ratio_of_totals_below_500_inputs']['value']:.2f} & {a['geometric_mean']:.2f} & {pct(a['better'], 0)} / {pct(a['tied'], 0)} / {pct(a['worse'], 0)} & {pct(a['at_least_double'])}\\\\")
    (PAPER / "tables/perquestion.tex").write_text("\n".join(lines) + "\n")

    # Full per-question table for the supplement.
    pretty = {"clairvoyant": "Clairvoyant bound", "stat": "Sampling test", "mlp_s0:member": "MLP, member-first", "mlp_s0:direction": "MLP, direction-fixed", "mlp_s0:ag": "MLP, AG",
              "mlp_s0:adg": "MLP, Adaptive Dual Greedy", "mlp_s0:recal": "MLP, AG + recal.", "linear:member": "Linear, member-first", "linear:ag": "Linear, AG", "ntac:member": "NTAC, member-first",
              "ntac:ag": "NTAC, AG", "provisional:member": "Earlier, member-first", "provisional:direction": "Earlier, direction-fixed", "provisional:ag": "Earlier, AG",
              "provisional:recal": "Earlier, AG + recal.", "provisional:adir": "Earlier, single-goal adaptive"}
    lines = []
    for view in views:
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{VIEW_NAME[view]}}}}}\\\\")
        for name, label in pretty.items():
            a = post[view]["per_question"]["arms"].get(name)
            if a is None:
                continue
            lines.append(f"{label} & {ci(a['ratio_of_totals'])} & {ci(a['ratio_of_totals_below_500_inputs'])} & {a['median']:.2f} & {a['geometric_mean']:.2f} & {a['mean']:.2f} & "
                         f"{pct(a['better'])} / {pct(a['tied'])} / {pct(a['worse'])} & {pct(a['at_least_double'])} & {a['largest']:.1f}\\\\")
    (PAPER / "tables/perquestion_full.tex").write_text("\n".join(lines) + "\n")

    # Controls: round robin on both splits.
    lines = []
    for name, label in (("mlp_s0:rr", "Round-robin, MLP prior"), ("ntac:rr", "Round-robin, NTAC prior"), ("provisional:rr", "Round-robin, earlier-release prior"),
                        ("mlp_s0:rr/recal", "\\quad MLP: round-robin / AG + recal."), ("provisional:rr/direction", "\\quad earlier: round-robin / direction-fixed"), ("provisional:rr/recal", "\\quad earlier: round-robin / AG + recal.")):
        cells = []
        for data in (post, post_dev):
            for view in views:
                c = data[view]["controls"].get(name)
                cells.append("---" if c is None else ci(c.get("ratio_to_weight", c)))
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/controls.tex").write_text("\n".join(lines) + "\n")

    # Three-way stop.
    lines = []
    rows = (("random_r0", "Random order"), ("mlp_s0:member", "MLP, member-first"), ("mlp_s0:direction", "MLP, direction-fixed"), ("mlp_s0:rr", "MLP, round-robin"), ("mlp_s0:ag", "MLP, AG"), ("mlp_s0:adg", "MLP, Adaptive Dual Greedy"),
            ("mlp_s0:recal", "MLP, AG + recal."), ("ntac:ag", "NTAC, AG"), ("provisional:member", "Earlier, member-first"), ("provisional:direction", "Earlier, direction-fixed"), ("provisional:rr", "Earlier, round-robin"),
            ("provisional:ag", "Earlier, AG"), ("provisional:recal", "Earlier, AG + recal."), ("provisional:adir", "Earlier, single-goal adaptive"))
    for name, label in rows:
        cells = []
        for view in views:
            t = post[view]["three_way"]
            if name in t["uncertifiable_ratio_to_weight"]:
                cells += [ci(t["uncertifiable_ratio_to_weight"][name]), ci(t["all_questions_ratio_to_weight"][name]), f"{t['all_questions_ratio_exhaustive'][name]['value']:.3f}"]
            else:
                cells += ["---", "---", "---"]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/threeway.tex").write_text("\n".join(lines) + "\n")

    # Hybrid test by input count, both splits.
    lines = []
    for key, label in (("10-49", "10--49"), ("50-149", "50--149"), ("150-499", "150--499"), (">=500", "$\\geq$500")):
        cells = []
        for data in (post, post_dev):
            for view in views:
                x = data[view]["hybrid"]["strata"][key]
                cells += [f"{x['sampling']['value']:.2f}", ci(x["hybrid"])]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    cells = []
    for data in (post, post_dev):
        for view in views:
            cells += ["", ci(data[view]["hybrid"]["ratio_to_weight"])]
    lines.append("\\midrule All & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/hybrid.tex").write_text("\n".join(lines) + "\n")
    record = read(ROOT / "outputs/v35/posthoc-record.json")
    out["PosthocTime"] = record["recorded_utc"].replace(" UTC", "~UTC")
    dev = read(ROOT / "outputs/v35/analysis/development.json")
    out["DevStrictSavingRecal"] = f"{100 * (1 - dev['flywire']['methods']['provisional:recal']['certifiable']['ratio_to_weight']['value']):.0f}"
    for view in views:
        v, q = view_key(view), post_dev[view]["per_question"]
        out[v + "DevLargeShareQuestions"] = pct(q["large_share_of_questions"])
        out[v + "DevLargeShareChecks"] = pct(q["large_share_of_weight_checks"], 0)
        out[v + "UndAdaptivePct"] = f"{100 * (float(out[v + 'UndAdaptiveMax']) - 1):.1f}"
        out[v + "PosthocCommon"] = str(post[view]["integrity"]["common_arms"])
        out[v + "UncertShareExhaustive"] = pct(post[view]["three_way"]["share_of_weight_checks_on_uncertifiable_exhaustive"], 0)
        out[v + "UncertShareThree"] = pct(post[view]["three_way"]["share_of_weight_checks_on_uncertifiable_three_way"], 0)
        out[v + "HybridOverSampling"] = f"{post[view]['hybrid']['ratio_to_sampling']['value']:.2f}"
        out[v + "HybridRisk"] = pct(post[view]["hybrid"]["share_using_risk"], 0)
        out[v + "HybridWrongCert"] = pct(post[view]["hybrid"]["wrong_rate_certifiable"], 2)
        out[v + "HybridWrongUnres"] = pct(post[view]["hybrid"]["wrong_rate_uncertifiable"], 2)
    wall = sum(read(ROOT / f"outputs/v35/posthoc/{view}-{split}.json")["wall_seconds"] for view in views for split in ("development", "test"))
    out["PosthocWallMinutes"] = f"{wall / 60:.0f}"
    test = read(ROOT / "outputs/v35/analysis/test.json")
    band = test["flywire"]["contrasts"]["mlp_s0:adg/ag"]
    out["FlyMlpADGOverAGThree"], out["FlyMlpADGOverAGLowThree"] = f"{band['value']:.3f}", f"{band['low']:.3f}"

    # Statements made in words about the post hoc results.
    def need(condition, claim):
        if not condition:
            raise SystemExit(f"a post hoc claim made in words no longer holds: {claim}")

    need(0.97 <= band["value"] <= 1.03 and band["low"] < 0.97, "ADG/AG point estimate inside the band, lower end outside")
    for view in views:
        r, d = post[view], post_dev[view]
        und = r["three_way"]["uncertifiable_ratio_to_weight"]
        need(all(abs(und[f"{prior}:{policy}"]["value"] - 1) < 0.0025 for prior in ("mlp_s0", "ntac") for policy in ("ag", "adg", "recal")), "adaptive rules within 0.2% on uncertifiable questions")
        need(und["mlp_s0:member"]["low"] > 1 and und["random_r0"]["low"] > 1, "member-first and random worse on uncertifiable questions")
        need(r["three_way"]["lower_bound_over_weight"]["value"] > 0.85, "lower bound close to weight order")
        need(r["controls"]["mlp_s0:rr"]["ratio_to_weight"]["low"] > 1, "round-robin with the topology prior worse than weight order")
        need(r["controls"]["ntac:rr"]["ratio_to_weight"]["value"] > 1, "round-robin with NTAC not better than weight order")
        strata = r["hybrid"]["strata"]
        need(all(strata[k]["hybrid"]["low"] > 1 for k in ("10-49", "50-149")), "hybrid test worse below 150 input cells")
        need(all(strata[k]["hybrid"]["value"] < strata[k]["sampling"]["value"] for k in ("10-49", "50-149", "150-499")), "hybrid below pure sampling under 500 inputs")
        need(strata[">=500"]["hybrid"]["value"] < 1 and d["hybrid"]["strata"][">=500"]["hybrid"]["high"] < 1, "hybrid test cheaper for the largest targets")
        need(r["hybrid"]["wrong_rate"] < 0.05, "hybrid error rate below its level")
        need(d["per_question"]["large_share_of_weight_checks"] > 1.5 * r["per_question"]["large_share_of_weight_checks"], "development split has more large targets")
        per = r["per_question"]["arms"]
        need(per["mlp_s0:recal"]["tied"] > 0.6 and per["mlp_s0:recal"]["better"] > per["mlp_s0:recal"]["worse"], "recalibrated AG mostly ties, wins more often than it loses")
        need(per["mlp_s0:recal"]["ratio_of_totals_below_500_inputs"]["value"] < 1, "gain not carried by the largest targets")
    need(post["flywire"]["hybrid"]["strata"][">=500"]["hybrid"]["high"] < 1 <= post["malecns"]["hybrid"]["strata"][">=500"]["hybrid"]["high"], "hybrid interval excludes one in FlyWire only")
    fly = post["flywire"]
    need(fly["controls"]["provisional:rr/direction"]["low"] > 1 and fly["controls"]["provisional:rr/recal"]["low"] > 1, "round-robin worse than direction-fixed and recalibrated AG")
    und = fly["three_way"]["uncertifiable_ratio_to_weight"]
    need(und["provisional:direction"]["low"] > 1 and und["provisional:recal"]["value"] < und["provisional:ag"]["value"] < und["provisional:direction"]["value"], "earlier-release policies on uncertifiable questions")
    per = fly["per_question"]["arms"]
    need(per["provisional:ag"]["at_least_double"] > 2 * per["provisional:direction"]["at_least_double"], "AG doubles the cost more than twice as often")
    lr = lenient["flywire"]["ratio_to_weight"]
    need(lr["provisional:recal"]["high"] < lr["provisional:direction"]["value"], "recalibrated AG beats direction-fixed under the lenient crosswalk")
    return out


SPLIT2 = "test"
LEVEL_TAG = {"0.6": "Six", "0.7": "Seven", "0.8": "Eight", "0.9": "Nine", "0.95": "NineFive"}
VARIANT_TAG = {"base": "Cal", "oracle": "Conf", "none": "NoU"}
POLICY_TAG = {"member": "Member", "direction": "Dir", "rr": "RR", "ag": "AG", "recal": "Recal"}


def round2_assets(split="test"):
    """Second round of review: priors of controlled quality, held-out coverage and lenient crosswalk, robustness."""
    r2 = read(ROOT / f"outputs/v35/analysis/round2-{split}.json")
    cov = read(ROOT / f"outputs/v35/analysis/coverage-{split}.json")
    cov_dev = read(ROOT / "outputs/v35/analysis/coverage-development.json")
    len_test = read(ROOT / f"outputs/v35/analysis/lenient-{split}.json")
    len_dev = read(ROOT / "outputs/v35/analysis/lenient-development.json")
    views = ("malecns", "flywire")
    out = {}
    for view in views:
        v, sw = view_key(view), r2[view]["sweep"]
        for tag, row in sw["ratio_to_weight"].items():
            level, variant = tag.split("/")
            name = v + "Sweep" + VARIANT_TAG[variant] + LEVEL_TAG[level]
            for policy, x in row.items():
                out[name + POLICY_TAG[policy]] = f"{x['value']:.2f}"
            con = sw["contrasts"][tag]
            for key, ctag in (("ag/direction", "AGOverDir"), ("recal/ag", "RecalOverAG"), ("recal/direction", "RecalOverDir")):
                out[name + ctag] = f"{con[key]['value']:.2f}"
                out[name + ctag + "CI"] = ci(con[key])
        real = sw["real_contrasts"]
        out[v + "BaseRateAG"] = f"{real['mlp_s0']['base_ag/as_is_ag']['value']:.3f}"
        out[v + "BaseRateRecalOverAG"] = f"{real['mlp_s0']['base_recal/base_ag']['value']:.3f}"
        rb = r2[view]["robustness"]
        sp = rb["single_pass"]
        out[v + "SingleInflation"] = f"{sp['weight_inflation']:.2f}"
        out[v + "SingleFamilies"] = thousands(sp["families"])
        out[v + "SingleMlpMember"] = f"{sp['ratio_to_weight']['mlp_s0:member']['value']:.2f}"
        out[v + "SingleMlpMemberQ"] = f"{sp['per_question_ratio_to_weight']['mlp_s0:member']['value']:.2f}"
        if "provisional:member" in sp["ratio_to_weight"]:
            out[v + "SingleProvMember"] = f"{sp['ratio_to_weight']['provisional:member']['value']:.2f}"
            out[v + "SingleProvMemberQ"] = f"{sp['per_question_ratio_to_weight']['provisional:member']['value']:.2f}"
        nt = rb["nontrivial"]
        out[v + "TrivialShare"] = pct(nt["share_of_certifiable"], 0)
        out[v + "NontrivialMlpAG"] = f"{nt['ratio_to_weight']['mlp_s0:ag']['value']:.2f}"
        out[v + "NontrivialMlpRecal"] = f"{nt['ratio_to_weight']['mlp_s0:recal']['value']:.2f}"
        out[v + "NontrivialStat"] = f"{nt['ratio_to_weight']['stat']['value']:.2f}"
        if "provisional:member" in sp["ratio_to_weight"]:
            out[v + "SingleProvMember"] = f"{sp['ratio_to_weight']['provisional:member']['value']:.2f}"
            out[v + "SingleProvMemberQ"] = f"{sp['per_question_ratio_to_weight']['provisional:member']['value']:.2f}"
            out[v + "NontrivialProvDirection"] = f"{nt['ratio_to_weight']['provisional:direction']['value']:.2f}"
            out[v + "NontrivialProvRecal"] = f"{nt['ratio_to_weight']['provisional:recal']['value']:.2f}"
            pr = rb["paired"]["provisional:recal/provisional:direction"]
            out[v + "PairRecalDirTotals"] = f"{pr['ratio_of_totals']['value']:.2f}"
            out[v + "PairRecalDirBetter"], out[v + "PairRecalDirTied"], out[v + "PairRecalDirWorse"] = (pct(pr[k], 0) for k in ("better", "tied", "worse"))
            out[v + "PairRecalDirGeo"] = f"{pr['geometric_mean']:.2f}"
        # Lenient crosswalk on both splits.
        for data, stag in ((len_test[view], "Test"), (len_dev[view], "")):
            a = data["answers"]
            out[v + "Len" + stag + "StrictUnresP"] = pct(a["strict"]["unresolved"])
            out[v + "Len" + stag + "LenUnresP"] = pct(a["lenient"]["unresolved"])
        lr = len_test[view]["ratio_to_weight"]
        for key, tag in (("mlp_s0:member", "MlpMember"), ("mlp_s0:ag", "MlpAG"), ("mlp_s0:recal", "MlpRecal"), ("provisional:direction", "ProvDirection"), ("provisional:ag", "ProvAG"), ("provisional:recal", "ProvRecal")):
            if key in lr:
                out[v + "LenTest" + tag] = f"{lr[key]['value']:.2f}"
                out[v + "LenTest" + tag + "CI"] = ci(lr[key])
    fly_len = len_test["flywire"]["ratio_to_weight"]
    out["LenTestSavingDirection"] = f"{100 * (1 - fly_len['provisional:direction']['value']):.0f}"
    out["LenTestSavingRecal"] = f"{100 * (1 - fly_len['provisional:recal']['value']):.0f}"
    test = read(ROOT / "outputs/v35/analysis/test.json")["flywire"]["methods"]
    out["StrictSavingRecal"] = f"{100 * (1 - test['provisional:recal']['certifiable']['ratio_to_weight']['value']):.0f}"
    frozen = read(ROOT / "outputs/v35/analysis/test.json")
    out["TargetsTest"] = thousands(sum(frozen[w]["composition"]["targets"] for w in views))
    shift = []
    for w in views:
        nt = r2[w]["robustness"]["nontrivial"]["ratio_to_weight"]
        for name, x in nt.items():
            if name.split(":")[-1] in ("ag", "adg", "recal", "direction", "adir") or name.startswith("provisional"):
                shift.append(abs(x["value"] - frozen[w]["methods"][name]["certifiable"]["ratio_to_weight"]["value"]))
    out["NontrivialShiftMax"] = f"{max(shift):.2f}"
    out["MaleNontrivialMlpMember"] = f"{r2['malecns']['robustness']['nontrivial']['ratio_to_weight']['mlp_s0:member']['value']:.2f}"
    record2 = read(ROOT / "outputs/v35/posthoc-record-2.json")
    for w in views:
        rates = record2["development_base_rates"][w]
        out[view_key(w) + "RateTop"] = pct(rates["member"]["0"])
        out[view_key(w) + "RateCoal"] = pct(rates["member"]["1"])
        out[view_key(w) + "RateUnres"] = pct(rates["unresolved"])
    wall = sum(read(ROOT / f"outputs/v35/sweep/{w}-{sp}.json")["wall_seconds"] for w in views for sp in ("development", "test"))
    out["SweepWallMinutes"] = f"{wall / 60:.0f}"
    out["SweepTestQuestions"] = thousands(sum(r2[w]["sweep"]["questions"] for w in views))
    # Ranges used in words.
    def values(view_list, variants, levels, key):
        return [r2[w]["sweep"]["contrasts"][f"{l}/{var}"][key]["value"] for w in view_list for var in variants for l in levels]
    hedge = values(views, ("base", "oracle"), ("0.8", "0.9", "0.95"), "ag/direction")
    out["HedgeLow"], out["HedgeHigh"] = f"{min(hedge):.2f}", f"{max(hedge):.2f}"
    recal = values(views, ("base", "oracle", "none"), ("0.6", "0.7", "0.8", "0.9", "0.95"), "recal/ag")
    out["SweepRecalOverAGMax"] = f"{max(recal):.2f}"
    def ratio(w, tag, policy):
        return r2[w]["sweep"]["ratio_to_weight"][tag][policy]["value"]
    for level in ("0.6", "0.7", "0.8", "0.9", "0.95"):
        savings = [ratio(w, f"{level}/base", "ag") for w in views]
        out["SweepCalAG" + LEVEL_TAG[level] + "Low"], out["SweepCalAG" + LEVEL_TAG[level] + "High"] = f"{min(savings):.2f}", f"{max(savings):.2f}"
    poor = [ratio(w, f"{l}/oracle", p) for w in views for l in ("0.6",) for p in ("member", "direction", "ag")]
    out["ConfPoorLow"], out["ConfPoorHigh"] = f"{min(poor):.2f}", f"{max(poor):.2f}"
    out["ConfPoorLowPct"], out["ConfPoorHighPct"] = f"{100 * (min(poor) - 1):.0f}", f"{100 * (max(poor) - 1):.0f}"
    poor_recal = [ratio(w, "0.6/oracle", "recal") for w in views]
    out["ConfPoorRecalLowPct"], out["ConfPoorRecalHighPct"] = f"{100 * (min(poor_recal) - 1):.0f}", f"{100 * (max(poor_recal) - 1):.0f}"
    cal_recal = [ratio(w, "0.6/base", "recal") for w in views]
    out["SweepCalRecalSixLowPct"], out["SweepCalRecalSixHighPct"] = f"{100 * (min(cal_recal) - 1):.0f}", f"{100 * (max(cal_recal) - 1):.0f}"

    out["ConfPoorRecalLow"], out["ConfPoorRecalHigh"] = f"{min(poor_recal):.2f}", f"{max(poor_recal):.2f}"
    nou = [ratio(w, f"{l}/none", "direction") / ratio(w, f"{l}/base", "direction") for w in views for l in ("0.9", "0.95")]
    out["NoUDirLow"], out["NoUDirHigh"] = f"{min(nou):.2f}", f"{max(nou):.2f}"
    base_ag = [float(out[view_key(w) + "BaseRateAG"]) for w in views]
    out["BaseRateAGLow"], out["BaseRateAGHigh"] = f"{min(base_ag):.3f}", f"{max(base_ag):.3f}"
    out["SweepAurocDev"] = f"{max(abs(r2[w]['sweep']['realized_auroc'][l][var] - float(l)) for w in views for l in r2[w]['sweep']['realized_auroc'] for var in r2[w]['sweep']['realized_auroc'][l]):.3f}"

    # Table: synthetic priors (supplement).
    lines = []
    for variant, title in (("base", "Calibrated, base-rate unresolved mass"), ("none", "Calibrated, no unresolved mass"), ("oracle", "Confident")):
        lines.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        for level in ("0.6", "0.7", "0.8", "0.9", "0.95"):
            cells = []
            for w in views:
                row = r2[w]["sweep"]["ratio_to_weight"][f"{level}/{variant}"]
                cells += [f"{row[p]['value']:.2f}" for p in ("member", "direction", "rr", "ag", "recal")]
            lines.append(f"\\quad {level} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/sweep.tex").write_text("\n".join(lines) + "\n")
    # Table: contrasts with intervals.
    lines = []
    for variant, title in (("base", "Calibrated, base-rate unresolved mass"), ("oracle", "Confident")):
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        for level in ("0.6", "0.7", "0.8", "0.9", "0.95"):
            cells = []
            for w in views:
                con = r2[w]["sweep"]["contrasts"][f"{level}/{variant}"]
                cells += [ci(con["ag/direction"]), ci(con["recal/ag"])]
            lines.append(f"\\quad {level} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/sweep_contrasts.tex").write_text("\n".join(lines) + "\n")
    # Table: coverage sweep, held-out.
    names = ("member", "direction", "ag", "adg", "recal", "adir")
    lines = []
    for level in [f"{x:g}" for x in cov["levels"]]:
        row = cov["ratio_to_weight"][level]
        lines.append(f"{100 * float(level):.0f} & " + " & ".join(ci(row[k]) for k in names) + "\\\\")
    (PAPER / "tables/coverage_test.tex").write_text("\n".join(lines) + "\n")
    tags = dict(zip([f"{x:g}" for x in cov["levels"]], ("Zero", "Quarter", "Half", "ThreeQ", "Full")))
    for level, tag in tags.items():
        row = cov["ratio_to_weight"][level]
        for key, ktag in (("direction", "Dir"), ("ag", "AG"), ("recal", "Recal")):
            out["CovTest" + ktag + tag] = f"{row[key]['value']:.2f}"
        best = min(row[k]["value"] for k in names)
        out["CovTestRecalGap" + tag] = f"{100 * (row['recal']['value'] / best - 1):.1f}"
    out["CovTestQuestions"] = thousands(cov["questions"])
    out["CovTestDirQuarterPct"] = f"{100 * (cov['ratio_to_weight']['0.25']['direction']['value'] - 1):.0f}"
    gaps = [100 * (cov["ratio_to_weight"][l]["recal"]["value"] / min(cov["ratio_to_weight"][l][k]["value"] for k in names) - 1) for l in tags]
    out["CovTestRecalGapMax"] = f"{max(gaps):.1f}"
    # Table: lenient crosswalk on both splits.
    rows = (("clairvoyant", "Clairvoyant bound"), ("mlp_s0:member", "Topology, member-first"), ("mlp_s0:direction", "Topology, direction-fixed"), ("mlp_s0:ag", "Topology, AG"),
            ("mlp_s0:recal", "Topology, AG + recal."), ("provisional:member", "Earlier, member-first"), ("provisional:direction", "Earlier, direction-fixed"), ("provisional:ag", "Earlier, AG"), ("provisional:recal", "Earlier, AG + recal."))
    lines = []
    for data, title in ((len_test, "Held-out"), (len_dev, "Development")):
        lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\")
        lines.append("\\quad cannot be certified, declared / lenient (\\%) & " + " & ".join(f"{pct(data[w]['answers']['strict']['unresolved'])} / {pct(data[w]['answers']['lenient']['unresolved'])}" for w in views) + "\\\\")
        for key, label in rows:
            cells = [ci(data[w]["ratio_to_weight"][key]) if key in data[w]["ratio_to_weight"] else "---" for w in views]
            lines.append(f"\\quad {label} & {cells[0]} & {cells[1]}\\\\")
    (PAPER / "tables/lenient_both.tex").write_text("\n".join(lines) + "\n")
    # Table: robustness of the frozen endpoint.
    lines = []
    for w in views:
        rb = r2[w]["robustness"]
        sp, nt = rb["single_pass"], rb["nontrivial"]
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{VIEW_NAME[w]}}}}}\\\\")
        for name, label in (("mlp_s0:member", "Topology, member-first"), ("provisional:member", "Earlier, member-first")):
            if name in sp["ratio_to_weight"]:
                lines.append(f"\\quad {label} & {sp['per_question_ratio_to_weight'][name]['value']:.2f} & {ci(sp['ratio_to_weight'][name])} & {ci(nt['ratio_to_weight'][name]) if name in nt['ratio_to_weight'] else '---'}\\\\")
        for name, label in (("mlp_s0:ag", "Topology, AG"), ("mlp_s0:recal", "Topology, AG + recal."), ("provisional:direction", "Earlier, direction-fixed"), ("provisional:ag", "Earlier, AG"), ("provisional:recal", "Earlier, AG + recal."), ("stat", "Sampling test")):
            if name in nt["ratio_to_weight"]:
                lines.append(f"\\quad {label} & {read_frozen_ratio(w, name)} & --- & {ci(nt['ratio_to_weight'][name])}\\\\")
    (PAPER / "tables/robustness.tex").write_text("\n".join(lines) + "\n")

    # Statements made in words.
    def need(condition, claim):
        if not condition:
            raise SystemExit(f"a round-two claim made in words no longer holds: {claim}")

    for w in views:
        sw = r2[w]["sweep"]
        need(all(abs(sw["realized_auroc"][l][var] - float(l)) < 0.01 for l in sw["realized_auroc"] for var in sw["realized_auroc"][l]), "realized AUROC on target")
        need(ratio(w, "0.6/base", "ag") >= 0.97 and ratio(w, "0.7/base", "ag") >= 0.97 and ratio(w, "0.9/base", "ag") <= 0.85, "savings need AUROC of about 0.8")
        need(all(ratio(w, f"0.6/oracle", p) > 1.15 for p in ("member", "direction", "ag")) and ratio(w, "0.6/oracle", "recal") < min(ratio(w, "0.6/oracle", p) for p in ("member", "direction", "ag")), "confident poor priors hurt, recalibration contains it")
        need(abs(float(out[view_key(w) + "BaseRateAG"]) - 1) < 0.02, "base-rate unresolved mass barely changes AG")
    need(0.9 <= min(hedge) and max(hedge) <= 1.1, "hedging costs little on priors of uniform quality")
    need(max(recal) <= 1.02, "recalibration never much worse than AG")
    need(min(nou) >= 1.15, "direction-fixed needs unresolved mass")
    need(max(recal) < 1.0, "recalibrated AG never needed more checks than AG on synthetic priors")
    for w in views:
        need(abs(float(out[view_key(w) + "BaseRateAG"]) - 1) < 0.01, "base-rate unresolved mass changes AG by less than 1%")
        need(frozen[w]["methods"]["mlp_s0:ag"]["certifiable"]["ratio_to_weight"]["value"] < min(ratio(w, f"{l}/base", "ag") for l in ("0.6", "0.7")), "real weak priors beat synthetic ones of similar AUROC under AG")
        need(r2[w]["sweep"]["ratio_to_weight"]["0.6/oracle"]["recal"]["value"] < 1.2, "recalibration limits the damage of confident poor priors")
    need(max(gaps[1:4]) == 0.0, "recalibrated AG is the cheapest of six policies at every partial coverage")
    need(cov["ratio_to_weight"]["0.25"]["ag"]["value"] < cov["ratio_to_weight"]["0.25"]["direction"]["value"] and cov["ratio_to_weight"]["0.75"]["ag"]["value"] > cov["ratio_to_weight"]["0.75"]["direction"]["value"]
         and cov["ratio_to_weight"]["0.5"]["ag"]["value"] < cov["ratio_to_weight"]["0.5"]["direction"]["value"], "direction-fixed overtakes AG between half and three-quarter coverage")
    lt = len_test["flywire"]["ratio_to_weight"]
    need(lt["provisional:recal"]["value"] < lt["provisional:direction"]["value"], "lenient: recalibrated AG better than direction-fixed")
    return out


def record2_outcomes():
    """Check every expectation of the second post hoc record against the held-out results."""
    r2 = read(ROOT / "outputs/v35/analysis/round2-test.json")
    cov = read(ROOT / "outputs/v35/analysis/coverage-test.json")["ratio_to_weight"]
    len_test = read(ROOT / "outputs/v35/analysis/lenient-test.json")
    record = read(ROOT / "outputs/v35/posthoc-record-2.json")
    views = ("malecns", "flywire")
    levels = ("0.6", "0.7", "0.8", "0.9", "0.95")

    def r(w, tag, policy):
        return r2[w]["sweep"]["ratio_to_weight"][tag][policy]["value"]

    def c(w, tag, key):
        return r2[w]["sweep"]["contrasts"][tag][key]["value"]

    rows = []

    def row(key, text, values, holds):
        rows.append((key, text, values, holds))

    per = {}
    for w in views:
        sw = r2[w]["sweep"]
        per[w] = dict(
            S1=(max(abs(sw["realized_auroc"][l][v] - float(l)) for l in levels for v in sw["realized_auroc"][l]),),
            S2=(min(r(w, f"{l}/base", "ag") for l in ("0.6", "0.7")), max(r(w, f"{l}/base", "ag") for l in ("0.9", "0.95"))),
            S3=(min(c(w, f"{l}/{v}", "ag/direction") for l in ("0.8", "0.9", "0.95") for v in ("base", "oracle")), max(c(w, f"{l}/{v}", "ag/direction") for l in ("0.8", "0.9", "0.95") for v in ("base", "oracle"))),
            S4=(min(r(w, f"{l}/none", "direction") / r(w, f"{l}/base", "direction") for l in ("0.9", "0.95")),),
            S5=(c(w, "0.6/oracle", "recal/ag"), c(w, "0.6/oracle", "recal/direction")),
            S6=(max(c(w, f"{l}/{v}", "recal/ag") for l in levels for v in ("none", "base", "oracle")),),
            S7=(min(r(w, f"{l}/{v}", "rr") / min(r(w, f"{l}/{v}", "member"), r(w, f"{l}/{v}", "direction")) for l in ("0.8", "0.9", "0.95") for v in ("none", "base", "oracle")),),
            S8=(max(abs(sw["real_contrasts"][q]["base_ag/as_is_ag"]["value"] - 1) for q in ("mlp_s0", "ntac")) + 1, max(sw["real_contrasts"][q]["base_recal/base_ag"]["value"] for q in ("mlp_s0", "ntac"))),
            L1=(100 * (len_test[w]["answers"]["strict"]["unresolved"] - len_test[w]["answers"]["lenient"]["unresolved"]), len_test[w]["moved_from_unresolved"]["to_positive"]),
            L4=(len_test[w]["ratio_to_weight"]["mlp_s0:member"]["value"], len_test[w]["ratio_to_weight"]["mlp_s0:recal"]["value"] / len_test[w]["ratio_to_weight"]["mlp_s0:ag"]["value"]),
        )
    checks = dict(
        S1=lambda v: v[0] < 0.01, S2=lambda v: v[0] >= 0.97 and v[1] <= 0.85, S3=lambda v: 0.95 <= v[0] and v[1] <= 1.10, S4=lambda v: v[0] >= 1.15,
        S5=lambda v: v[0] <= 0.90 and v[1] <= 0.95, S6=lambda v: v[0] <= 1.02, S7=lambda v: v[0] > 1, S8=lambda v: v[0] <= 1.02 and v[1] <= 1.0,
        L1=lambda v: v[0] >= 10 and v[1] == 0, L4=lambda v: v[0] > 1 and v[1] <= 1,
    )
    texts = dict(
        S1="realized AUROC within 0.01 of the target$^\\ast$", S2="calibrated: AG $\\geq 0.97$ at AUROC $\\leq 0.7$; $\\leq 0.85$ at $\\geq 0.9$",
        S3="AG / fixed in $[0.95, 1.10]$ at AUROC $\\geq 0.8$", S4="fixed without $\\unknown$ mass $\\geq 1.15\\times$ with it, AUROC $\\geq 0.9$",
        S5="confident, AUROC 0.6: recal./AG $\\leq 0.90$, recal./fixed $\\leq 0.95$", S6="recal./AG $\\leq 1.02$ at every level",
        S7="round-robin above the cheaper fixed order, AUROC $\\geq 0.8$", S8="base-rate $\\unknown$ mass: AG within 2\\%; recal./AG $\\leq 1$",
        L1="uncertifiable share falls $\\geq 10$ points; none becomes \\emph{yes}$^\\ast$", L4="topology: member-first $> 1$, recal. $\\leq$ AG",
    )
    fmt = dict(S1="{:.3f}", L1="{:.1f}")
    out = {}
    held = total = 0
    lines = []
    order = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "C1", "C2", "C3", "C4", "L1", "L2", "L3", "L4")
    names = ("member", "direction", "ag", "adg", "recal", "adir")
    flyonly = dict(
        C1=("coverage 25\\%: fixed $\\geq 1.05$", (cov["0.25"]["direction"]["value"],), lambda v: v[0] >= 1.05),
        C2=("recal. $\\leq$ AG at every coverage (max ratio)", (max(cov[l]["recal"]["value"] / cov[l]["ag"]["value"] for l in cov),), lambda v: v[0] <= 1),
        C3=("recal. cheapest or within 2\\% at 25--75\\% (gap, \\%)", (max(100 * (cov[l]["recal"]["value"] / min(cov[l][k]["value"] for k in names) - 1) for l in ("0.25", "0.5", "0.75")),), lambda v: v[0] <= 2),
        C4=("AG / fixed $< 1$ at 25\\%, $> 1$ at 75\\%", (cov["0.25"]["ag"]["value"] / cov["0.25"]["direction"]["value"], cov["0.75"]["ag"]["value"] / cov["0.75"]["direction"]["value"]), lambda v: v[0] < 1 < v[1]),
        L2=("lenient, earlier: recal. $<$ fixed", (len_test["flywire"]["ratio_to_weight"]["provisional:recal"]["value"], len_test["flywire"]["ratio_to_weight"]["provisional:direction"]["value"]), lambda v: v[0] < v[1]),
        L3=("lenient, earlier: fixed $> 0.70$", (len_test["flywire"]["ratio_to_weight"]["provisional:direction"]["value"],), lambda v: v[0] > 0.70),
    )
    for key in order:
        if key not in record["expectations"]:
            raise SystemExit(f"{key} is not in the second record")
        if key in flyonly:
            text, values, check = flyonly[key]
            ok = check(values)
            cells = ["", " / ".join(f"{x:.2f}" for x in values) if key != "C3" else f"{values[0]:.1f}"]
            marks = ["", "$\\bullet$" if ok else "$\\times$"]
            total += 1
            held += ok
        else:
            text = texts[key]
            cells, marks = [], []
            for w in views:
                ok = checks[key](per[w][key])
                f = fmt.get(key, "{:.2f}")
                cells.append(" / ".join(f.format(x) for x in per[w][key]))
                marks.append("$\\bullet$" if ok else "$\\times$")
                total += 1
                held += ok
        lines.append(f"{order.index(key) + 1} & {text} & {cells[0]} & {marks[0]} & {cells[1]} & {marks[1]}\\\\")
    (PAPER / "tables/predictions2.tex").write_text("\n".join(lines) + "\n")
    out["RecordTwoChecks"], out["RecordTwoHeld"], out["RecordTwoPredictions"] = str(total), str(held), str(len(order))
    out["RecordTwoTime"] = record["recorded_utc"].replace(" UTC", "~UTC")
    return out


def round3_assets(split="test"):
    """Third round: other earlier-release priors, ablation of the recalibration, prior quality."""
    r3 = read(ROOT / f"outputs/v35/analysis/round3-{split}.json")
    record = read(ROOT / "outputs/v35/posthoc-record-3.json")
    views = ("malecns", "flywire")
    out = {}

    def need(condition, claim):
        if not condition:
            raise SystemExit(f"round-3 claim no longer holds: {claim}")

    fly = r3["flywire"]
    early = fly["earlier"]
    mixtures = ("a0.8", "a0.9", "a0.95", "a0.99")
    ratio = lambda name, p: early[name]["ratio_to_weight"][p]["value"]
    contrast = lambda name, k: early[name]["contrasts"][k]["value"]
    agdir = [contrast(n, "ag/direction") for n in mixtures]
    out["EarlyAGDirEight"], out["EarlyAGDirNinetyNine"] = f"{contrast('a0.8', 'ag/direction'):.2f}", f"{contrast('a0.99', 'ag/direction'):.2f}"
    out["EarlyAGLow"], out["EarlyAGHigh"] = f"{min(ratio(n, 'ag') for n in mixtures):.2f}", f"{max(ratio(n, 'ag') for n in mixtures):.2f}"
    out["EarlyDirLow"], out["EarlyDirHigh"] = f"{min(ratio(n, 'direction') for n in mixtures):.2f}", f"{max(ratio(n, 'direction') for n in mixtures):.2f}"
    out["EarlyRecalLow"], out["EarlyRecalHigh"] = f"{min(ratio(n, 'recal') for n in mixtures):.2f}", f"{max(ratio(n, 'recal') for n in mixtures):.2f}"
    for p, tag in (("direction", "Dir"), ("ag", "AG"), ("recal", "Recal"), ("member", "Member")):
        out["EarlyCal" + tag] = f"{ratio('calibrated', p):.2f}"
    out["EarlyCalAGDir"] = f"{contrast('calibrated', 'ag/direction'):.2f}"
    out["EarlyCalDirSavingPct"] = f"{100 * (1 - ratio('calibrated', 'direction')):.0f}"
    out["EarlyCalAGDirPct"] = f"{100 * (contrast('calibrated', 'ag/direction') - 1):.0f}"
    out["EarlyCalAGDirCI"] = ci(early["calibrated"]["contrasts"]["ag/direction"])
    out["EarlyCalRecalDirCI"] = ci(early["calibrated"]["contrasts"]["recal/direction"])
    recdir = [contrast(n, "recal/direction") for n in early]
    out["EarlyRecalDirLow"], out["EarlyRecalDirHigh"] = f"{min(recdir):.2f}", f"{max(recdir):.2f}"
    rates = fly["development_class_rates"]
    out["RatePartnerMember"] = pct(rates["partner"][0])
    out["RateOutsideUnres"], out["RateOutsideNon"], out["RateOutsideMember"] = pct(rates["outside"][2], 0), pct(rates["outside"][1], 0), pct(rates["outside"][0], 0)
    out["RateOtherNon"] = pct(rates["other"][1])
    need(all(x > 1.10 for x in agdir) and agdir[-1] > agdir[0], "AG loses to direction-fixed at every weight, more at higher weight")
    need(contrast("calibrated", "ag/direction") < min(agdir), "calibrated class rates shrink the loss of AG")
    need(ratio("calibrated", "direction") < ratio("a0.9", "direction") and ratio("calibrated", "ag") < ratio("a0.9", "ag"), "calibrated prior is cheaper for fixed and AG")
    need(max(recdir) - min(recdir) < 0.1, "recalibrated AG stays close to direction-fixed under every construction")
    # Ablation.
    abl = {w: r3[w]["ablation"] for w in views}
    rel = lambda w, prior, v: abl[w][prior][v]["ratio_to_weight"]["value"] / abl[w][prior]["frozen"]["ratio_to_weight"]["value"]
    real = [("malecns", "mlp_s0"), ("flywire", "mlp_s0"), ("flywire", "provisional")]
    kappa = [rel(w, p, v) for w, p in real for v in ("kappa1", "kappa16", "kappa64")]
    out["KappaMaxPct"] = f"{100 * max(abs(x - 1) for x in kappa):.0f}"
    one_topo = [rel(w, "mlp_s0", "groups1") for w in views]
    out["GroupsOneTopoLowPct"], out["GroupsOneTopoHighPct"] = f"{100 * (min(one_topo) - 1):.1f}", f"{100 * (max(one_topo) - 1):.1f}"
    conf = lambda v: [abl[w]["oracle0.6"][v]["ratio_to_weight"]["value"] for w in views]
    for v, tag in (("frozen", "Frozen"), ("kappa16", "KappaSixteen"), ("kappa64", "KappaSixtyFour"), ("uprior", "UPrior")):
        out["Conf" + tag + "Low"], out["Conf" + tag + "High"] = f"{min(conf(v)):.2f}", f"{max(conf(v)):.2f}"
    need(min(conf("kappa64")) > max(conf("frozen")) + 0.1, "the margin of recalibration with confident poor priors depends on the strength")
    out["GroupsOneEarlyPct"] = f"{100 * (rel('flywire', 'provisional', 'groups1') - 1):.0f}"
    out["FlyGroupsOneTopoPct"] = f"{100 * (rel('flywire', 'mlp_s0', 'groups1') - 1):.1f}"
    eight = [rel(w, p, "groups8") for w, p in real]
    out["GroupsEightMaxPct"] = f"{100 * max(abs(x - 1) for x in eight):.0f}"
    out["UPriorEarlyPct"] = f"{100 * (rel('flywire', 'provisional', 'uprior') - 1):.0f}"
    upo = [rel(w, "oracle0.6", "uprior") for w in views]
    out["UPriorConfLowPct"], out["UPriorConfHighPct"] = f"{100 * (min(upo) - 1):.0f}", f"{100 * (max(upo) - 1):.0f}"
    out["USessionEarlyPct"] = f"{100 * (1 - rel('flywire', 'provisional', 'usession')):.0f}"
    out["USessionEarly"] = f"{abl['flywire']['provisional']['usession']['ratio_to_weight']['value']:.2f}"
    one_conf = [rel(w, "oracle0.6", "groups1") for w in views]
    out["GroupsOneConfLowPct"], out["GroupsOneConfHighPct"] = f"{100 * (1 - max(one_conf)):.0f}", f"{100 * (1 - min(one_conf)):.0f}"
    need(max(abs(x - 1) for x in kappa) < 0.05, "strength barely matters")
    need(min(one_topo) > 1.0 and rel("flywire", "provisional", "groups1") > 1.10, "one group is worse for real priors")
    need(max(one_conf) < 1.0, "one group helps confident poor synthetic priors")
    need(max(abs(x - 1) for x in eight) < 0.03, "eight groups change little")
    need(rel("flywire", "provisional", "uprior") > 1 and min(upo) > 1, "prior-only unresolved rate is worse")
    need(rel("flywire", "provisional", "usession") < 1, "session-only unresolved rate is better for earlier labels")
    lines = []
    labels = dict(frozen="As specified (strength 4, five groups, averaged)", kappa1="Strength 1", kappa16="Strength 16", kappa64="Strength 64", groups1="One group",
                  groups3="Three groups", groups8="Eight groups", uprior="Unresolved: prior only", usession="Unresolved: session only")
    columns = [("malecns", "mlp_s0"), ("flywire", "mlp_s0"), ("flywire", "provisional"), ("malecns", "oracle0.6"), ("flywire", "oracle0.6"), ("malecns", "oracle0.8"), ("flywire", "oracle0.8")]
    for v, label in labels.items():
        lines.append(f"{label} & " + " & ".join(f"{abl[w][p][v]['ratio_to_weight']['value']:.3f}" for w, p in columns) + "\\\\")
    (PAPER / "tables/ablation.tex").write_text("\n".join(lines) + "\n")
    # Table: earlier-release prior built in other ways.
    lines = []
    names = {"a0.8": "Weight 0.8", "a0.9": "Weight 0.9 (paper)", "a0.95": "Weight 0.95", "a0.99": "Weight 0.99", "calibrated": "Development class rates"}
    for n, label in names.items():
        x = early[n]
        lines.append(f"{label} & " + " & ".join(f"{x['ratio_to_weight'][p]['value']:.2f}" for p in ("member", "direction", "ag", "recal"))
                     + f" & {ci(x['contrasts']['ag/direction'])} & {ci(x['contrasts']['recal/direction'])}\\\\")
    (PAPER / "tables/earlier_constructions.tex").write_text("\n".join(lines) + "\n")
    # Prior quality.
    q = {w: r3[w]["quality"] for w in views}
    sd = [q[w]["mlp_s0"]["auroc"]["sd"] for w in views]
    hi = [q[w]["mlp_s0"]["auroc"]["at_least_0_8"] for w in views]
    lo = [q[w]["mlp_s0"]["auroc"]["below_0_5"] for w in views]
    out["QualSdLow"], out["QualSdHigh"] = f"{min(sd):.2f}", f"{max(sd):.2f}"
    out["QualHighLow"], out["QualHighHigh"] = pct(min(hi), 0), pct(max(hi), 0)
    out["QualBelowLow"], out["QualBelowHigh"] = pct(min(lo), 0), pct(max(lo), 0)
    for w in views:
        k = view_key(w)
        x = q[w]["mlp_s0"]["ag"]
        out[k + "QualReal"] = f"{x['real']['value']:.2f}"
        out[k + "QualPerQuestion"] = f"{x['predicted_per_question']['value']:.2f}"
        out[k + "QualFromMean"] = f"{x['predicted_from_mean']:.2f}"
    between = [q[w][p][pol]["predicted_per_question"]["value"] < q[w][p][pol]["real"]["value"] < q[w][p][pol]["predicted_from_mean"]
               for w in views for p in ("mlp_s0", "ntac") for pol in ("ag", "recal")]
    need(all(between), "observed cost lies between the two predictions")
    lines = []
    for w in views:
        for p, label in (("mlp_s0", "Topology"), ("ntac", "NTAC")) + ((("provisional", "Earlier release"),) if w == "flywire" else ()):
            a = q[w][p]["auroc"]
            cells = [f"{a['mean']:.2f}", f"{a['sd']:.2f}", pct(a["at_least_0_8"], 0), pct(a["below_0_5"], 0)]
            for pol in ("ag", "recal"):
                if pol in q[w][p]:
                    x = q[w][p][pol]
                    cells.append(f"{x['real']['value']:.2f} / {x['predicted_per_question']['value']:.2f} / {x['predicted_from_mean']:.2f}")
                else:
                    cells.append("---")
            lines.append(f"{'MaleCNS' if w == 'malecns' else 'FlyWire'}, {label} & " + " & ".join(cells) + "\\\\")
    (PAPER / "tables/quality.tex").write_text("\n".join(lines) + "\n")
    # Expectations of the third record.
    rows, held, total = [], 0, 0
    def add(key, text, values, ok, view_cells=None):
        nonlocal held, total
        cells = view_cells if view_cells is not None else ["", " / ".join(f"{v:.2f}" for v in values)]
        marks = [("" if c == "" else ("$\\bullet$" if o else "$\\times$")) for c, o in zip(cells, ok if isinstance(ok, list) else ["", ok])]
        for c, o in zip(cells, ok if isinstance(ok, list) else [None, ok]):
            if c != "":
                total += 1
                held += bool(o)
        rows.append(f"{key[1:]} & {text} & {cells[0]} & {marks[0]} & {cells[1]} & {marks[1]}\\\\")
    add("E1", "earlier, weights 0.8--0.99: AG / fixed $> 1.10$ (min, at 0.8, at 0.99)", (min(agdir), agdir[0], agdir[-1]), min(agdir) > 1.10 and agdir[-1] > agdir[0])
    add("E2", "class rates: AG / fixed $\\leq 1.10$; fixed below weight 0.9", (contrast("calibrated", "ag/direction"), ratio("calibrated", "direction"), ratio("a0.9", "direction")),
        contrast("calibrated", "ag/direction") <= 1.10 and ratio("calibrated", "direction") < ratio("a0.9", "direction"))
    add("E3", "recal. / fixed in $[0.95, 1.06]$, every construction", (min(recdir), max(recdir)), 0.95 <= min(recdir) and max(recdir) <= 1.06)
    k4 = {w: [rel(w2, p, v) for w2, p in real if w2 == w for v in ("kappa1", "kappa16", "kappa64")] for w in views}
    add("E4", "strength 1--64 within 4\\% (max deviation, \\%)", None, [max(abs(x - 1) for x in k4[w]) <= 0.04 for w in views],
        [f"{100 * max(abs(x - 1) for x in k4[w]):.1f}" for w in views])
    add("E5", "one group: $\\geq 2\\%$ more (topology), $\\geq 10\\%$ (earlier)", None,
        [one_topo[0] >= 1.02, one_topo[1] >= 1.02 and rel("flywire", "provisional", "groups1") >= 1.10],
        [f"{one_topo[0]:.3f}", f"{one_topo[1]:.3f} / {rel('flywire', 'provisional', 'groups1'):.2f}"])
    e8 = {w: [rel(w2, p, "groups8") for w2, p in real if w2 == w] for w in views}
    add("E6", "eight groups within 2\\% (max deviation, \\%)", None, [max(abs(x - 1) for x in e8[w]) <= 0.02 for w in views],
        [f"{100 * max(abs(x - 1) for x in e8[w]):.1f}" for w in views])
    add("E7", "unresolved: prior only more (confident 0.6; earlier); session only not more (earlier)", None,
        [upo[0] > 1, upo[1] > 1 and rel("flywire", "provisional", "uprior") > 1 and rel("flywire", "provisional", "usession") <= 1],
        [f"{upo[0]:.2f}", f"{upo[1]:.2f} / {rel('flywire', 'provisional', 'uprior'):.2f} / {rel('flywire', 'provisional', 'usession'):.2f}"])
    e8ok = {w: all(q[w][p][pol]["predicted_per_question"]["value"] < q[w][p][pol]["real"]["value"] < q[w][p][pol]["predicted_from_mean"] for p in ("mlp_s0", "ntac") for pol in ("ag", "recal")) for w in views}
    add("E8", "per-question prediction $<$ observed $<$ prediction from mean AUROC (4 cases)", None, [e8ok[w] for w in views], ["4 of 4" if e8ok[w] else "no" for w in views])
    add("E9", "topology AUROC: s.d. $\\geq 0.15$, $\\geq 25\\%$ of questions at 0.8", None, [sd[i] >= 0.15 and hi[i] >= 0.25 for i in range(2)],
        [f"{sd[i]:.2f} / {100 * hi[i]:.0f}\\%" for i in range(2)])
    for key in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"):
        if key not in record["expectations"]:
            raise SystemExit(f"{key} is not in the third record")
    (PAPER / "tables/predictions3.tex").write_text("\n".join(rows) + "\n")
    out["RecordThreeChecks"], out["RecordThreeHeld"], out["RecordThreePredictions"] = str(total), str(held), "9"
    out["RecordThreeTime"] = record["recorded_utc"].replace(" UTC", "~UTC")
    meta = [read(ROOT / f"outputs/v35/round3/{w}-{s}.json")["wall_seconds"] for w in views for s in ("development", "test")]
    out["RoundThreeWallMinutes"] = f"{sum(meta) / 60:.0f}"
    return out


def coupling_assets():
    """Fourth round: answer coupling (B4) and the other development pilots."""
    base_dir = ROOT / "outputs/neurosym"
    test = read(base_dir / "B/heldout/analysis-test.json")
    dev_all = read(base_dir / "B/heldout/analysis-development.json")
    round2 = read(base_dir / "B/round2.json")
    controls = read(base_dir / "B/round2-controls.json")
    record = read(ROOT / "outputs/v35/posthoc-record-4.json")
    picks = read(base_dir / "B/heldout/picks.json")["picks"]
    views = ("malecns", "flywire")
    out = {}

    def need(condition, claim):
        if not condition:
            raise SystemExit(f"coupling claim no longer holds: {claim}")

    for w in views:
        need(picks[w]["base"] == "recal" and picks[w]["config"] == dict(gamma=3.0, lam=8.0, tau=0.3), "one setting (recal, lam 8, gamma 3, tau 0.3) for both views")
    v = test["views"]
    b4 = {w: v[w]["b4"]["recal"] for w in views}
    saving = {w: 1 - b4[w]["to_better"]["value"] for w in views}
    for w in views:
        k = view_key(w)
        need(v[w]["better_frozen"] == "recal", "the better frozen policy is recalibrated AG")
        out[k + "CoupCI"] = ci(b4[w]["to_better"])
        out[k + "CoupPct"] = f"{100 * saving[w]:.0f}"
        out[k + "CoupWeight"] = f"{b4[w]['to_weight']['value']:.2f}"
        out[k + "CoupWeightCI"] = ci(b4[w]["to_weight"])
        out[k + "CoupAgCI"] = ci(v[w]["b4"]["ag"]["to_same_base"])
        need(b4[w]["to_better"]["high"] < 1 and v[w]["b4"]["ag"]["to_same_base"]["high"] < 1, "coupling is cheaper with both bases")
    out["CoupPctLow"], out["CoupPctHigh"] = f"{100 * min(saving.values()):.0f}", f"{100 * max(saving.values()):.0f}"
    ctl = lambda name, key: [v[w]["controls"][name][key]["value"] for w in views]
    shuf, notarget, ipsi, oracle = ctl("shuffled", "to_better"), ctl("no_target", "retained"), ctl("ipsilateral", "retained"), ctl("oracle", "retained")
    out["CoupShufLow"], out["CoupShufHigh"] = f"{min(shuf):.2f}", f"{max(shuf):.2f}"
    out["CoupNoTargetLowPct"], out["CoupNoTargetHighPct"] = f"{100 * min(notarget):.0f}", f"{100 * max(notarget):.0f}"
    out["CoupIpsiLowPct"], out["CoupIpsiHighPct"] = f"{100 * min(ipsi):.0f}", f"{100 * max(ipsi):.0f}"
    out["CoupContraLowPct"], out["CoupContraHighPct"] = f"{100 * (1 - max(ipsi)):.0f}", f"{100 * (1 - min(ipsi)):.0f}"
    out["CoupOracleLowPct"], out["CoupOracleHighPct"] = f"{100 * min(oracle):.0f}", f"{100 * max(oracle):.0f}"
    need(min(shuf) > 1.05, "a permuted kernel costs more than no coupling")
    need(min(notarget) >= 0.95, "the target column does not carry the gain")
    need(0.7 <= min(ipsi) and max(ipsi) < 0.95, "left-right pairs carry a minor share")
    need(0.95 <= min(oracle) and max(oracle) <= 1.05, "the same-type oracle is no better than the similarity kernel")
    # Diagnostic (not recorded): the oracle links resolved cells only; split held-out questions by unresolved input cells.
    few, many = [], []
    for w in views:
        with np.load(base_dir / f"B/heldout/{w}-test.npz", allow_pickle=False) as z:
            arm = {key: z[key] for key in ("family", "rho", "truth", "b4__recal", "oracle__recal")}
        with np.load(ROOT / f"outputs/v35/runs/{w}-test.npz", allow_pickle=False) as z:
            frozen = {(int(f), int(r)): (int(u), int(s)) for f, r, u, s in zip(z["family"], z["rho"], z["unresolved_cells"], z["size"])}
        share = np.array([frozen[(int(f), int(r))][0] / frozen[(int(f), int(r))][1] for f, r in zip(arm["family"], arm["rho"])])
        cert = arm["truth"] > 0
        lo_q, hi_q = np.quantile(share[cert], [0.25, 0.75])
        for sel, bucket in ((cert & (share <= lo_q), few), (cert & (share >= hi_q), many)):
            bucket.append(arm["oracle__recal"][sel].sum() / arm["b4__recal"][sel].sum())
    span = lambda xs: f"{min(xs):.2f}" if f"{min(xs):.2f}" == f"{max(xs):.2f}" else f"{min(xs):.2f}--{max(xs):.2f}"
    out["CoupOracleFewQ"], out["CoupOracleManyQ"] = span(few), span(many)
    need(max(few) < 1 and min(many) > 1, "the oracle is cheaper where few input cells are unresolved and dearer where many are")
    paired = {w: v[w]["paired_b4_vs_better"] for w in views}
    out["CoupBetterLow"], out["CoupBetterHigh"] = pct(min(p["better"] for p in paired.values()), 0), pct(max(p["better"] for p in paired.values()), 0)
    out["CoupWorseLow"], out["CoupWorseHigh"] = pct(min(p["worse"] for p in paired.values()), 0), pct(max(p["worse"] for p in paired.values()), 0)
    # Development: cross-fit (round 2) and its controls.
    cells = [(w, d) for w in ("flywire", "malecns") for d in ("fit0->eval1", "fit1->eval0")]
    prim = [round2["results"][f"{w}:{d}"]["B4"]["primary"] for w, d in cells]
    out["CoupDevLow"], out["CoupDevHigh"] = f"{min(p['value'] for p in prim):.2f}", f"{max(p['value'] for p in prim):.2f}"
    out["CoupDevCIHigh"] = f"{max(p['high'] for p in prim):.2f}"
    need(max(p["high"] for p in prim) < 0.95, "coupling passes the cross-fit on both views and directions")
    runs = controls["runs"]
    wo = [runs[f"{w}:{d}"]["eval"]["weight_ii"]["primary"]["value"] for w, d in cells]
    out["CoupWeightOnlyLow"], out["CoupWeightOnlyHigh"] = f"{min(wo):.2f}", f"{max(wo):.2f}"
    need(min(wo) > 1.0, "a kernel of synapse-count similarity alone costs more than no coupling")
    contra = [runs[f"{w}:{d}"]["lr_diagnostic"]["share_contralateral"] for w, d in cells]
    out["CoupContraPairsLow"], out["CoupContraPairsHigh"] = f"{100 * min(contra):.1f}", f"{100 * max(contra):.1f}"
    head = [min(r["recal"] for r in runs[f"{w}:{d}"]["headroom_fit_half_only"]["rows"]) - runs[f"{w}:{d}"]["headroom_fit_half_only"]["b4_fit_ratio"] for w, d in cells]
    out["CoupHeadroom"] = f"{-min(head):.3f}"
    clair = round2["clairvoyant"]
    for w in views:
        out[view_key(w) + "CoupDevAll"] = f"{dev_all['views'][w]['b4']['recal']['to_better']['value']:.3f}"
    # Supplement tables: development cross-fit and controls, held-out arms.
    names = [("B4", "Answer propagation"), ("shuffled_s0", "Permuted kernel, seed 0"), ("no_target_ii", "Profiles without the target"),
             ("weight_ii", "Synapse-count similarity only"), ("ipsilateral_only", "Without left--right pairs"), ("oracle_ii", "Oracle kernel: same reference type")]
    lines = []
    for key, label in names:
        cells_txt = [ci(runs[f"{w}:{d}"]["eval"][key]["primary"]) for w, d in cells]
        lines.append(f"{label} & " + " & ".join(cells_txt) + "\\\\")
    (PAPER / "tables/coupling_dev.tex").write_text("\n".join(lines) + "\n")
    lines = []
    rows = [("Propagation, recal.\\ AG", lambda w: ci(v[w]["b4"]["recal"]["to_better"]), lambda w: ci(v[w]["b4"]["recal"]["to_weight"])),
            ("Propagation, AG", lambda w: ci(v[w]["b4"]["ag"]["to_better"]), lambda w: ci(v[w]["b4"]["ag"]["to_weight"]))]
    for name, label in (("no_target", "Profiles without the target"), ("shuffled", "Permuted kernel"), ("ipsilateral", "Without left--right pairs"), ("oracle", "Oracle kernel: same reference type")):
        rows.append((label, lambda w, n=name: ci(v[w]["controls"][n]["to_better"]), lambda w, n=name: ci(v[w]["controls"][n]["retained"])))
    for label, f, g in rows:
        lines.append(f"{label} & {f('malecns')} & {g('malecns')} & {f('flywire')} & {g('flywire')}\\\\")
    (PAPER / "tables/coupling_test.tex").write_text("\n".join(lines) + "\n")
    # Expectations of the fourth record.
    exp = test["expectations"]
    text = {"K1": "propagation (recal.) / recal.\\ AG $\\leq 0.92$, interval $< 1$", "K2": "propagation (AG) / AG: interval $< 1$",
            "K3": "permuted kernel / recal.\\ AG $\\geq 1$", "K4": "without the target: keeps $\\geq 2/3$ of the saving",
            "K5": "without left--right pairs: keeps 0.70--1.00 of the saving", "K6": "oracle same-type kernel keeps $\\geq 1$ of the saving"}
    lines, held, total = [], 0, 0
    for key in ("K1", "K2", "K3", "K4", "K5", "K6"):
        if key not in record["expectations"]:
            raise SystemExit(f"{key} is not in the fourth record")
        marks = []
        for w in views:
            e = exp[f"{key}:{w}"]
            total += 1
            held += bool(e["holds"])
            marks.append((f"{e['value']['value']:.2f}", "$\\bullet$" if e["holds"] else "$\\times$"))
        lines.append(f"{key[1:]} & {text[key]} & {marks[0][0]} & {marks[0][1]} & {marks[1][0]} & {marks[1][1]}\\\\")
    (PAPER / "tables/predictions4.tex").write_text("\n".join(lines) + "\n")
    need((held, total) == (exp["summary"]["held"], exp["summary"]["of"]), "record-4 tally")
    need(not exp["K6:flywire"]["holds"] and not exp["K6:malecns"]["holds"] and held == total - 2, "only K6 fails")
    out["RecordFourHeld"], out["RecordFourChecks"] = str(held), str(total)
    out["RecordFourTime"] = record["recorded_utc"].replace(" UTC", "~UTC")
    secs = [read(base_dir / f"B/heldout/{w}-{s}.json")["seconds"] for w in views for s in ("development", "test")]
    out["CoupWallMinutes"] = f"{sum(secs) / 60:.0f}"
    # The other development pilots (no held-out run).
    a = read(base_dir / "A/results.json")["settings"]
    out["PilotAMale"] = f"{a['malecns:mlp_s0']['half1']['learned_vs_best_criterion']['value']:.2f}"
    out["PilotAFly"] = f"{a['flywire:mlp_s0']['half1']['learned_vs_best_criterion']['value']:.2f}"
    out["PilotAProv"] = f"{a['flywire:provisional']['half1']['learned_vs_best_criterion']['value']:.2f}"
    out["PilotAProvFit"] = f"{a['flywire:provisional']['half0']['learned_vs_best_criterion']['value']:.2f}"
    need(a["flywire:provisional"]["half1"]["learned_vs_best_criterion"]["value"] > 1.03, "the cost-trained policy loses with earlier labels")
    b = read(base_dir / "B/eval.json")
    need(b["verdict_strict"]["B2:auroc"]["do"] and sum(r["do"] for k_, r in b["verdict_strict"].items() if k_ != "overall") == 1, "only B2 passes the strict verdict")
    out["PilotBFlyAuroc"] = f"{b['views']['flywire']['B2:auroc']['auroc_minus_mlp_s0']['value']:.3f}"
    out["PilotBMaleAuroc"] = f"{b['views']['malecns']['B2:auroc']['auroc_minus_mlp_s0']['value']:.3f}"
    out["PilotBFlyRecal"] = f"{b['views']['flywire']['B2:auroc']['recal_new_over_mlp_s0']['value']:.2f}"
    out["PilotBMaleRecal"] = f"{b['views']['malecns']['B2:auroc']['recal_new_over_mlp_s0']['value']:.2f}"
    cv = read(base_dir / "C/matching.json")["verification"]["verdicts"]
    out["PilotCDropLow"] = f"{min(cv['confident']['per_view'][w]['drop_pp'] for w in views):.0f}"
    out["PilotCDropHigh"] = f"{max(cv['confident']['per_view'][w]['drop_pp'] for w in views):.0f}"
    out["PilotCOpposite"] = str(cv["confident"]["per_view"]["malecns"]["opposite"])
    out["PilotCCleanLow"] = f"{min(cv['author_nobottom']['per_view'][w]['drop_pp'] for w in views):.1f}"
    out["PilotCCleanHigh"] = f"{max(cv['author_nobottom']['per_view'][w]['drop_pp'] for w in views):.1f}"
    need(cv["confident"]["sound_verdict"] == "FAIL" and cv["author_nobottom"]["verdict"] == "DO NOT", "matching fails soundness; its validated part is below the bar")
    return out


def read_frozen_ratio(view, name):
    test = read(ROOT / "outputs/v35/analysis/test.json")[view]["methods"]
    key = "stat" if name == "stat" else name
    return f"{test[key]['certifiable']['ratio_to_weight']['value']:.2f}"


def verbal_claims(test):
    """Macros for rounded figures quoted in prose, and checks of the claims made in words."""
    out = {}
    views = ("malecns", "flywire")
    comp = {v: test[v]["composition"] for v in views}
    for v in views:
        out[view_key(v) + "ResolvedInputs"] = pct(1 - comp[v]["unresolved_cell_fraction"], 0)
        out[view_key(v) + "UnresRounded"] = pct(comp[v]["unresolved"] / comp[v]["sessions"], 0)
        out[view_key(v) + "MemberRounded"] = f"{100 * (test[v]['methods']['mlp_s0:member']['certifiable']['ratio_to_weight']['value'] - 1):.0f}"
    saving = {
        v: max(
            1 - test[v]["methods"][f"{prior}:{policy}"]["certifiable"]["ratio_to_weight"]["value"]
            for prior in ("mlp_s0", "linear", "ntac")
            for policy in ("member", "nonmember", "confident", "direction", "ggn", "ag", "adg", "recal", "adir", "adir-recal")
        )
        for v in views
    }
    out["ConnectivitySavingMax"] = f"{100 * max(saving.values()):.0f}"
    member = [100 * (test[v]["methods"][f"{prior}:member"]["certifiable"]["ratio_to_weight"]["value"] - 1) for v in views for prior in ("mlp_s0", "linear", "ntac")]
    out["MemberFirstLow"], out["MemberFirstHigh"] = f"{min(member):.0f}", f"{max(member):.0f}"
    aurocs = [test[v]["calibration"][prior]["member_auroc_within_question"] for v in views for prior in ("mlp_s0", "linear", "ntac")]
    out["WeakAurocLow"], out["WeakAurocHigh"] = f"{min(aurocs):.2f}", f"{max(aurocs):.2f}"
    fly = test["flywire"]["methods"]
    ratio = lambda key: fly[key]["certifiable"]["ratio_to_weight"]["value"]
    strong = 1 - ratio("provisional:direction")
    out["EarlierSavingRounded"] = f"{100 * strong:.0f}"
    kept = (1 - ratio("provisional:ag")) / strong
    unresolvable = np.mean([comp[v]["unresolved"] / comp[v]["sessions"] for v in views])
    per_question = {v: test[v]["methods"]["weight"]["certifiable"]["total"] / (comp[v]["positive"] + comp[v]["negative"]) for v in views}
    checks = {
        "connectivity-only priors save at most 5%": 100 * max(saving.values()) <= 5.0,
        "the earlier release saves about a third": 0.30 <= strong <= 0.37,
        "AG keeps about a third of that saving": 0.28 <= kept <= 0.38,
        "about four in ten questions cannot be certified": 0.35 <= unresolvable <= 0.45,
        "recalibrated AG is within two points of direction-fixed": abs(ratio("provisional:recal") - ratio("provisional:direction")) <= 0.02,
        "a 5% gain is about one check per question": all(0.5 <= 0.05 * x <= 1.5 for x in per_question.values()),
        "unresolved input cells carry about a third of the synapses": all(0.30 <= comp[v]["unresolved_weight_fraction"] <= 0.37 for v in views),
        "the clairvoyant bound saves little more than half": all(0.5 <= 1 - test[v]["methods"]["clairvoyant"]["certifiable"]["ratio_to_weight"]["value"] <= 0.6 for v in views),
        "recalibrated AG never needs more checks than AG": all(c["value"] <= 1.0 for v in views for k, c in test[v]["contrasts"].items() if k.endswith(":recal/ag")),
        "sampling test above weight order below 500 inputs": all(test[v]["strata"][k]["methods"]["stat"]["low"] > 1 for v in views for k in ("inputs 10-49", "inputs 50-149", "inputs 150-499")),
    }
    coverage = read(ROOT / "outputs/v35/analysis/coverage-development.json")["ratio_to_weight"]
    checks["recalibrated AG is cheapest or within 2% at every coverage level"] = all(row["recal"]["value"] <= 1.02 * min(x["value"] for x in row.values()) for row in coverage.values())
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("a claim made in words no longer holds: " + "; ".join(failed))
    return out


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5)


def example_orders():
    """Bounds trajectories for one development question (FlyWire MBON09, strongest partner)."""
    bench = np.load(ROOT / "outputs/v35/benchmark/flywire.npz", allow_pickle=False)
    target = int(np.flatnonzero(bench["target_cell"] == 720575940630864847)[0])
    family = int(np.flatnonzero((bench["fam_target"] == target) & (bench["fam_kind"] == 0))[0])
    lo, hi = int(bench["target_offset"][target]), int(bench["target_offset"][target + 1])
    cells, weights, labels = bench["pre_cell"][lo:hi], bench["pre_weight"][lo:hi], bench["pre_label"][lo:hi]
    codes = bench["fam_codes"][bench["fam_code_offset"][family] : bench["fam_code_offset"][family + 1]]
    outcomes = np.where(labels < 0, e.UNRESOLVED, np.where(np.isin(labels, codes), e.MEMBER, e.NONMEMBER))
    total = int(weights.sum())
    threshold, _ = e.threshold_goals(total, int(bench["fam_source_weight"][family]), 2 * int(bench["fam_source_total"][family]))
    inst = e.threshold_instance(cells, weights, threshold)
    lengths = np.diff(bench["target_offset"])[bench["fam_target"]]
    offset = int(np.concatenate(([0], np.cumsum(lengths)))[family])

    def prob(name):
        plus = np.clip(bench["plus_" + name][offset : offset + hi - lo].astype(float), 0, 1)
        unknown = np.clip(bench["unknown_" + name][lo:hi].astype(float), 0, 1 - plus)
        return np.column_stack((plus, 1 - plus - unknown, unknown))

    orders = {
        "weight order": e.weight_order(inst),
        "topology prior + AG": e.run_adaptive(inst, outcomes, e.AdaptiveGreedy(inst, prob("mlp_s0"))),
        "earlier release, member-first": e.positive_order(inst, prob("provisional")),
    }
    out = {}
    for name, order in orders.items():
        cost, status = e.evaluate(inst, outcomes, order)
        order = np.asarray(order)[:cost]
        lower = np.cumsum(np.where(outcomes[order] == e.MEMBER, weights[order], 0)) / total
        upper = 1 - np.cumsum(np.where(outcomes[order] == e.NONMEMBER, weights[order], 0)) / total
        out[name] = dict(cost=cost, status=status, lower=np.concatenate(([0], lower)), upper=np.concatenate(([1], upper)))
    return out, threshold / total, inst.size, float(weights[outcomes == e.MEMBER].sum() / total)


def figure_concept():
    curves, tau, size, share = example_orders()
    fig = plt.figure(figsize=(7.0, 2.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.22)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, h = 0.42, 0.15
    segments = ((0.0, 0.28, BLUE, None), (0.28, 0.46, "white", "////"), (0.46, 0.72, "#efeee9", None), (0.72, 1.0, MUTED, None))
    for x0, x1, color, hatch in segments:
        ax.add_patch(Rectangle((x0, y), x1 - x0 - 0.004, h, facecolor=color, edgecolor=MUTED if hatch or color == "#efeee9" else "none", hatch=hatch, linewidth=0.5))
    for x, label in ((0.14, "checked:\nmember"), (0.37, "checked:\nunresolved"), (0.59, "not yet\nchecked"), (0.86, "checked:\nnonmember")):
        ax.text(x, y - 0.04, label, ha="center", va="top", fontsize=6.5, color=INK, linespacing=0.95)
    ax.plot([0.53, 0.53], [y - 0.02, y + h + 0.13], color=ORANGE, linewidth=1.2)
    ax.text(0.55, y + h + 0.10, "threshold $T$", ha="left", va="center", fontsize=7, color=INK)
    ax.annotate("", xy=(0.28, y + h + 0.05), xytext=(0.0, y + h + 0.05), arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.6))
    ax.text(0.14, y + h + 0.065, "$L$", ha="center", va="bottom", fontsize=7.5)
    ax.annotate("", xy=(0.72, y + h + 0.22), xytext=(0.0, y + h + 0.22), arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.6))
    ax.text(0.36, y + h + 0.235, "$U$", ha="center", va="bottom", fontsize=7.5)
    ax.text(0.0, 0.0, "positive certificate when $L\\geq T$;\nnegative certificate when $U<T$", ha="left", va="bottom", fontsize=6.8, color=INK)
    ax.text(0.0, 1.0, "(a) synaptic input of one target neuron", ha="left", va="top", fontsize=7.5, color=INK)
    ax = fig.add_subplot(grid[1])
    styles = {"weight order": (MUTED, "-"), "topology prior + AG": (BLUE, "-"), "earlier release, member-first": (ORANGE, "-")}
    for name, c in curves.items():
        color, style = styles[name]
        x = np.arange(len(c["lower"]))
        ax.plot(x, c["lower"], color=color, linewidth=1.3, linestyle=style, label=f"{name} ({c['cost']})")
        ax.plot([c["cost"]], [c["lower"][-1]], marker="o", markersize=3.5, color=color, markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(tau, color=INK, linewidth=0.6, linestyle=(0, (4, 3)))
    ax.text(4, tau + 0.004, "threshold $T/W$", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("cells whose type has been checked")
    ax.set_ylabel("certified lower bound $L/W$")
    ax.set_xlim(0, curves["weight order"]["cost"] * 1.03)
    ax.set_ylim(0, tau * 1.13)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.6, borderaxespad=0.3, title="order (cells checked)", title_fontsize=6.5)
    ax.set_title(f"(b) one question: {size} input cells, threshold {100 * tau:.0f}% of the input (answer: yes)", fontsize=7.5, loc="left", pad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    clean(ax)
    fig.savefig(PAPER / "figures/concept.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    return curves, tau, size, share


def figure_results(test, post):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[1.0, 0.92, 0.92, 0.92], wspace=0.62))
    # (a) forest plot of cost ratios, certifiable questions.
    ax = axes[0]
    rows = (
        ("Topology, member-first", "mlp_s0:member"),
        ("Topology, direction-fixed", "mlp_s0:direction"),
        ("Topology, AG", "mlp_s0:ag"),
        ("Topology, AG+recal.", "mlp_s0:recal"),
        ("Topology, +propagation", "coup:recal"),
        ("Earlier, direction-fixed", "provisional:direction"),
        ("Earlier, AG", "provisional:ag"),
        ("Earlier, AG+recal.", "provisional:recal"),
        ("Clairvoyant bound", "clairvoyant"),
    )
    offsets = {"malecns": 0.17, "flywire": -0.17}
    colors = {"malecns": BLUE, "flywire": ORANGE}
    coupling = read(ROOT / "outputs/neurosym/B/heldout/analysis-test.json")["views"]
    for i, (label, key) in enumerate(rows):
        y = len(rows) - 1 - i
        for view in ("malecns", "flywire"):
            if key.startswith("coup:"):
                r = coupling[view]["b4"][key[5:]]["to_weight"]
            else:
                method = test[view]["methods"].get(key)
                if method is None:
                    continue
                r = method["certifiable"]["ratio_to_weight"]
            ax.plot([r["low"], r["high"]], [y + offsets[view]] * 2, color=colors[view], linewidth=1.1)
            ax.plot([r["value"]], [y + offsets[view]], marker="o" if view == "malecns" else "s", markersize=3.6, color=colors[view], markeredgecolor="white", markeredgewidth=0.5)
    ax.axvline(1.0, color=INK, linewidth=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.2, color=INK)
    ax.set_xlabel("cells checked / weight order")
    ax.set_xlim(0.38, 1.5)
    ax.set_xticks([0.4, 0.7, 1.0, 1.3])
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[v], marker="o" if v == "malecns" else "s", markersize=3.6, linewidth=1.1, label=VIEW_NAME[v]) for v in ("malecns", "flywire")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.0, handlelength=1.2, borderaxespad=0.1)
    ax.set_title("(a) cost", fontsize=7.5, loc="left", pad=3)
    clean(ax)
    # (b) priors of controlled quality, FlyWire, held-out (post hoc).
    ax = axes[1]
    sweep = read(ROOT / f"outputs/v35/analysis/round2-{SPLIT2}.json")["flywire"]["sweep"]["ratio_to_weight"]
    levels = ("0.6", "0.7", "0.8", "0.9", "0.95")
    x = np.array([float(l) for l in levels])
    for key, color, marker in (("direction", ORANGE, "s"), ("ag", BLUE, "o"), ("recal", AQUA, "D")):
        for variant, style in (("base", "-"), ("oracle", (0, (2.2, 1.6)))):
            y = [sweep[f"{l}/{variant}"][key]["value"] for l in levels]
            ax.plot(x, y, color=color, linestyle=style, linewidth=1.1, marker=marker, markersize=2.8 if variant == "oracle" else 3.2,
                    markerfacecolor="white" if variant == "oracle" else color, markeredgecolor=color if variant == "oracle" else "white", markeredgewidth=0.6)
    ax.axhline(1.0, color=INK, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([l.lstrip("0") for l in levels], fontsize=6.2, color=INK)
    ax.set_xlim(0.58, 0.97)
    ax.set_ylim(0.55, 1.42)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_xlabel("AUROC of the prior")
    ax.set_ylabel("cells checked / weight order")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=MUTED, linewidth=1.1, label="calibrated"), Line2D([], [], color=MUTED, linewidth=1.1, linestyle=(0, (2.2, 1.6)), label="confident")]
    ax.legend(handles=handles, loc="upper right", fontsize=5.8, handlelength=1.8, borderaxespad=0.1, labelspacing=0.25)
    ax.set_title("(b) prior quality", fontsize=7.5, loc="left", pad=3)
    clean(ax)
    # (c) incomplete earlier labels, FlyWire, held-out (post hoc).
    ax = axes[2]
    c = read(ROOT / f"outputs/v35/analysis/coverage-{SPLIT2}.json")
    levels = [f"{x:g}" for x in c["levels"]]
    x = np.array([100 * float(v) for v in levels])
    for key, label, color, marker in (("direction", "direction-fixed", ORANGE, "s"), ("ag", "AG", BLUE, "o"), ("recal", "AG+recal.", AQUA, "D")):
        y = np.array([c["ratio_to_weight"][v][key]["value"] for v in levels])
        lo = np.array([c["ratio_to_weight"][v][key]["low"] for v in levels])
        hi = np.array([c["ratio_to_weight"][v][key]["high"] for v in levels])
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.2, marker=marker, markersize=3.3, markeredgecolor="white", markeredgewidth=0.5, label=label)
    ax.axhline(1.0, color=INK, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}" for v in x], fontsize=6.2, color=INK)
    ax.set_ylim(0.55, 1.42)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_xlabel("neurons with earlier label (%)")
    ax.set_ylabel("cells checked / weight order")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.3, borderaxespad=0.1, labelspacing=0.25)
    ax.set_title("(c) incomplete labels", fontsize=7.5, loc="left", pad=3)
    clean(ax)
    # (d) statistical comparators by input count.
    ax = axes[3]
    strata = ("inputs 10-49", "inputs 50-149", "inputs 150-499", "inputs >=500")
    labels = ("10–\n49", "50–\n149", "150–\n499", "≥500")
    for view in ("malecns", "flywire"):
        values = [test[view]["strata"][s]["methods"]["stat"]["value"] for s in strata]
        ax.plot(np.arange(len(strata)), values, color=colors[view], linewidth=0.9, linestyle=(0, (2, 2)), marker="o" if view == "malecns" else "s", markersize=2.6, markerfacecolor="white", markeredgewidth=0.7)
        hybrid = [post[view]["hybrid"]["strata"][s.replace("inputs ", "")]["hybrid"] for s in strata]
        x = np.arange(len(strata)) + (-0.08 if view == "malecns" else 0.08)
        ax.errorbar(x, [v["value"] for v in hybrid], yerr=[[v["value"] - v["low"] for v in hybrid], [v["high"] - v["value"] for v in hybrid]],
                    color=colors[view], marker="o" if view == "malecns" else "s", markersize=3.6, linewidth=1.1, markeredgecolor="white", markeredgewidth=0.5)
    ax.axhline(1.0, color=INK, linewidth=0.6)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(labels, fontsize=5.6, color=INK, linespacing=0.9)
    ax.set_xlabel("input cells per target")
    ax.set_ylabel("cells checked / weight order")
    ax.set_ylim(0.4, 1.9)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)), marker="o", markersize=2.6, markerfacecolor="white", markeredgewidth=0.7, label="sampling only"),
               Line2D([], [], color=MUTED, linewidth=1.1, marker="o", markersize=3.6, markeredgecolor="white", markeredgewidth=0.5, label="hybrid")]
    ax.legend(handles=handles, loc="lower left", fontsize=5.8, handlelength=1.7, borderaxespad=0.1, labelspacing=0.25)
    ax.set_title("(d) tests with 5% risk", fontsize=7.5, loc="left", pad=3)
    clean(ax)
    fig.savefig(PAPER / "figures/results.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_conservation(test):
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.65), sharey=True, gridspec_kw=dict(wspace=0.12))
    rhos = ("1/4", "1/2", "3/4", "1/1")
    for ax, view in zip(axes, ("malecns", "flywire")):
        cons = test[view]["conservation"]
        bottom = np.zeros(len(rhos))
        for field, color, hatch, label in (("positive", BLUE, None, "certified yes"), ("negative", ORANGE, None, "certified no"), ("unresolved", "white", "////", "cannot be certified")):
            values = np.array([cons[f"top1@rho={r}"][field] for r in rhos])
            ax.bar(range(len(rhos)), values, bottom=bottom, width=0.64, color=color, edgecolor="white" if hatch is None else MUTED, linewidth=0.6, hatch=hatch, label=label)
            bottom += values
        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels(["¼", "½", "¾", "1"], color=INK)
        ax.set_title(VIEW_NAME[view], fontsize=7.5, pad=2)
        ax.set_ylim(0, 1)
        clean(ax)
    axes[0].set_ylabel("strongest-partner questions")
    axes[0].set_yticks([0, 0.5, 1.0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel("required fraction $\\rho$ of the hemibrain input share", fontsize=7.5, y=-0.08)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, fontsize=6, handlelength=0.9, columnspacing=0.8, handletextpad=0.3)
    fig.savefig(PAPER / "figures/conservation.pdf", metadata={"CreationDate": None})
    plt.close(fig)


ROUND5_PROP_MACROS = ("PropNoneLow", "PropNoneHigh", "PropNoneVsTopoLow", "PropNoneVsTopoHigh", "PropProvCI", "PropProvSmallCI",
                      "PropSeedTenLow", "PropSeedTenHigh", "PropSeedMoreLowPct", "PropSeedMoreHighPct")


def round5_prop_assets(split="test"):
    """Propagation with no prior, the earlier-release prior and seeded priors (round 5, held-out).

    The text states that propagation replaces a good prior rather than adding to it; every part of that
    statement is checked here, and the build stops if the held-out analysis contradicts it."""
    found = sorted((ROOT / "outputs/round5/prop").glob(f"analysis-{split}-*.json"))
    if split == "development":  # several development analyses exist; the latest is the one on the rebuilt priors
        found = found and [max(found, key=lambda p: p.stat().st_mtime)]
    if not found:
        print("round 5 propagation: held-out analysis not run yet, macros left as ??")
        return {k: r"\textbf{??}" for k in ROUND5_PROP_MACROS}
    if len(found) > 1:
        raise SystemExit(f"several held-out propagation analyses: {[p.name for p in found]}")
    views = read(found[0])["views"]
    prop = lambda view, arm: views[view]["propagation"][arm]
    none = [prop(v, "const_half:recal+prop")["to_weight"]["value"] for v in ("malecns", "flywire", "banc")]
    vs_topo = [views[v]["contrasts"]["need the MLP? const_half:recal+prop / mlp_s0:recal+prop"]["ratio"]["value"] for v in ("malecns", "flywire", "banc")]
    prov = prop("flywire", "provisional:recal+prop")
    prov_small, prov_big = (prov["to_static_by_size"][k] for k in ("<500 input cells", ">=500 input cells"))
    ten = [prop(v, f"seeded10{c}:recal+prop")["to_static"]["value"] for v in ("malecns", "flywire") for c in ("", "conn")]
    more = [prop(v, f"seeded{f}{c}:recal+prop")["to_static"]["value"] for v in ("malecns", "flywire") for f in (30, 50) for c in ("", "conn")]
    checks = {
        "without a prior propagation saves at least a seventh of weight order": max(none) <= 0.86,
        "without a prior it needs about as many checks as with the topology prior": 0.97 <= min(vs_topo) and max(vs_topo) <= 1.05,
        "with the earlier-release prior it saves on targets with <500 inputs": prov_small["high"] < 1,
        "and not on targets with >=500 inputs": prov_big["value"] >= 1,
        "with a tenth of the neurons seeded it neither saves nor costs (within 5%)": 0.95 <= min(ten) and max(ten) <= 1.05,
        "with a third or a half seeded it costs extra": min(more) >= 1,
    }
    failed = [claim for claim, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("claims about propagation and strong priors no longer hold: " + "; ".join(failed))
    return dict(
        PropNoneLow=f"{min(none):.2f}", PropNoneHigh=f"{max(none):.2f}",
        PropNoneVsTopoLow=f"{min(vs_topo):.2f}", PropNoneVsTopoHigh=f"{max(vs_topo):.2f}",
        PropProvCI=ci(prov["to_static"]), PropProvSmallCI=ci(prov_small),
        PropSeedTenLow=f"{min(ten):.2f}", PropSeedTenHigh=f"{max(ten):.2f}",
        PropSeedMoreLowPct=f"{100 * (min(more) - 1):.0f}", PropSeedMoreHighPct=f"{100 * (max(more) - 1):.0f}",
    )


def main():
    (PAPER / "tables").mkdir(exist_ok=True)
    (PAPER / "figures").mkdir(exist_ok=True)
    test = read(ROOT / "outputs/v35/analysis/test.json")
    dev = read(ROOT / "outputs/v35/analysis/development.json")
    manifest = read(ROOT / "outputs/v35/benchmark/manifest.json")
    pilot = pilot_numbers()
    curves, tau, size, share = figure_concept()
    pilot.update(
        ExampleInputs=str(size),
        ExampleShare=f"{100 * share:.0f}",
        ExampleTau=f"{100 * tau:.0f}",
        ExampleWeight=str(curves["weight order"]["cost"]),
        ExampleAG=str(curves["topology prior + AG"]["cost"]),
        ExampleProv=str(curves["earlier release, member-first"]["cost"]),
    )
    pilot.update(diagnostic_assets())
    pilot.update(agreement_assets())
    pilot.update(coverage_assets())
    pilot.update(table_hypotheses(test))
    for view in ("malecns", "flywire"):
        c, w = test[view]["composition"], test[view]["methods"]["weight"]["certifiable"]["total"]
        pilot[view_key(view) + "WeightPerQuestion"] = f"{w / (c['positive'] + c['negative']):.0f}"
        pilot[view_key(view) + "RecalPerQuestion"] = f"{test[view]['methods']['mlp_s0:recal']['certifiable']['total'] / (c['positive'] + c['negative']):.0f}"
    fly = test["flywire"]
    pilot["FlyProvPerQuestion"] = f"{fly['methods']['provisional:direction']['certifiable']['total'] / (fly['composition']['positive'] + fly['composition']['negative']):.0f}"
    for view in manifest["views"]:
        v = view_key(view["view"])
        pilot[v + "Nodes"] = thousands(view["nodes"])
        pilot[v + "CrossRelease"] = thousands(view["crosswalk"]["release-type"])
        pilot[v + "CrossField"] = thousands(view["crosswalk"]["hemibrain-field"])
        pilot[v + "CrossUnres"] = thousands(view["crosswalk"]["unresolved"])
        pilot[v + "AllNeurons"] = thousands(view["targets"])
        pilot[v + "AllSets"] = thousands(view["families"])
        pilot[v + "ResolvedNodesPct"] = pct(1 - view["crosswalk"]["unresolved"] / view["nodes"], 0)
    mlp = read(ROOT / "outputs/v32/predictors/59cd92072106eb705ce5ae288aa891a79ee47de74bbba3cc5dce30ae4fc38ea3/report.json")
    lin = read(ROOT / "outputs/v34/predictors/506e892832babb70d1b737371958ae6a3d04580cde64913648e7cdcd8abaf7ba/report.json")
    seeds = [read(ROOT / f"outputs/v35/mlp-seeds/seed{k}/report.json")["summary"]["audit_accuracy"] for k in range(1, 5)]
    pilot.update(VocabTypes=thousands(mlp["vocabulary_size"] - 1), VocabClasses=thousands(mlp["vocabulary_size"]))
    pilot.update(
        MlpAuditAccuracy=pct(mlp["audit_accuracy"]),
        LinAuditAccuracy=pct(lin["audit_accuracy"]),
        MlpTrainRows=thousands(mlp["train_rows"]),
        MlpAuditRows=thousands(mlp["audit_rows"]),
        MlpTrainClasses=thousands(mlp["train_class_coverage"]),
        MlpSeedAccLow=pct(min(seeds)),
        MlpSeedAccHigh=pct(max(seeds)),
    )
    tested = [t for t in manifest["types"] if t["split"] == "test"]
    top = [t["families"][0]["source_weight"] / t["source_total_weight"] for t in tested]
    coalition = [(f["source_weight"] / t["source_total_weight"], len(f["codes"])) for t in tested for f in t["families"] if f["kind"] == "coalition"]
    pilot.update(
        TopShareMedian=pct(float(np.median(top)), 0),
        CoalShareMedian=pct(float(np.median([c[0] for c in coalition])), 0),
        CoalSizeMedian=f"{np.median([c[1] for c in coalition]):.0f}",
        CoalTypes=thousands(len(coalition)),
        MbonTypes=str(sum(t["name"].startswith("MBON") for t in manifest["types"])),
    )
    wall = sum(read(ROOT / f"outputs/v35/runs/{view}-{split}.json")["wall_seconds"] for view in ("malecns", "flywire") for split in ("development", "test"))
    freeze = read(ROOT / "outputs/v35/freeze.json")
    pilot.update(
        RunWallMinutes=f"{wall / 60:.0f}",
        RunCpuHours=f"{wall * 28 / 3600:.1f}",
        FreezeTime=freeze["recorded_utc"].replace(" UTC", "~UTC"),
        ArmsMale=str(len(read(ROOT / "outputs/v35/runs/malecns-test.json")["arms"])),
        ArmsFly=str(len(read(ROOT / "outputs/v35/runs/flywire-test.json")["arms"])),
    )
    post = read(ROOT / "outputs/v35/analysis/posthoc-test.json")
    post_dev = read(ROOT / "outputs/v35/analysis/posthoc-development.json")
    pilot.update(posthoc_assets(post, post_dev, read(ROOT / "outputs/v35/analysis/lenient-development.json")))
    pilot.update(round2_assets())
    pilot.update(record2_outcomes())
    pilot.update(round3_assets())
    pilot.update(coupling_assets())
    # Share of the gap between recalibrated AG and the clairvoyant bound that answer propagation closes (held-out).
    coup = read(ROOT / "outputs/neurosym/B/heldout/analysis-test.json")["views"]
    gap = []
    for view in ("malecns", "flywire"):
        ratio = lambda key: test[view]["methods"][key]["certifiable"]["ratio_to_weight"]["value"]
        recal, clair = ratio("mlp_s0:recal"), ratio("clairvoyant")
        gap.append((recal - coup[view]["b4"]["recal"]["to_weight"]["value"]) / (recal - clair))
    if not (0.35 <= min(gap) and max(gap) <= 0.45):
        raise SystemExit("claim no longer holds: propagation closes about two fifths of the gap to the clairvoyant bound")
    pilot.update(PropGapLowPct=f"{100 * min(gap):.0f}", PropGapHighPct=f"{100 * max(gap):.0f}")
    pilot.update(round5_prop_assets())
    pilot.update(verbal_claims(test))
    numbers(test, dev, manifest, pilot)
    table_composition(test)
    table_main(test, post)
    table_full(test, "full_test")
    table_full(dev, "full_development")
    table_strata(test)
    table_calibration(test)
    table_conservation(test)
    figure_results(test, post)
    figure_conservation(test)
    print("assets written")


if __name__ == "__main__":
    main()
