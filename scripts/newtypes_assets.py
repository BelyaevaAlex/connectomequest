"""Numbers and tables of the paper from the third protocol (scripts/newtypes.py): cell types that no earlier
analysis used, earlier labels in both connectomes, and earlier labels for a part of the neurons.

Writes ``paper/newtypes_numbers.tex`` (macros) and, for the appendix, ``paper/tables/newtypes_questions.tex``
(composition of the new cell types), ``newtypes_expectations.tex`` (the recorded expectations whose simultaneous interval
crosses the threshold), ``newtypes_policies.tex`` (every policy on new cell types), ``newtypes_earlier.tex``
(MaleCNS earlier-release prior by answer) and ``newtypes_coverage.tex`` (partial coverage). Every sentence of the
paper that rests on these numbers states a direction or a bound; it is checked here, and the build stops if a
result contradicts it. The rows of the table of the main text that come from this protocol are written by
scripts/replication_assets.py.

The within-question AUROC of the priors is computed here from the question arrays, with the definition of the
benchmark (members against all other input cells, averaged over partner sets with both).

Usage: python scripts/newtypes_assets.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r5_paper_assets import Claims, ci, f2  # noqa: E402

PAPER = ROOT / "paper"
OUT = ROOT / "outputs/newtypes"
VIEWS = (("Male", "malecns", "MaleCNS"), ("Fly", "flywire", "FlyWire"))
SPLITS = (("heldout", "Held-out types"), ("new", "New types"))
POLICIES = (("member", "Member-first"), ("nonmember", "Nonmember-first"), ("direction", "Direction-fixed"), ("ag", "AG"),
            ("adg", "Adaptive Dual Greedy"), ("recal", "AG + recalibration"), ("prop", "AG + recalibration + propagation"),
            ("ec2", "Edge cutting"), ("ig", "Information gain"), ("imitation", "Imitation order"))
SHORT = {"Adaptive Dual Greedy": "ADG", "AG + recalibration": "AG + recal.", "AG + recalibration + propagation": "AG + recal.\\ + prop.",
         "Information gain": "Inf.\\ gain"}
SIX = ("member", "nonmember", "direction", "ag", "adg", "recal")
AT_COVERAGE = (("direction", "Direction-fixed"), ("ag", "AG"), ("recal", "AG + recalibration"),
               ("imitation-full", "Imitation, fitted at full coverage"), ("imitation-mixed", "Imitation, fitted at varying coverage"))
LEVELS = (("0.25", "a quarter", "\\tfrac14"), ("0.5", "half", "\\tfrac12"), ("0.75", "three quarters", "\\tfrac34"))


def pct(x):
    return f"{100 * x:.0f}"


def aurocs(view, split):
    """Mean within-question AUROC of both priors over the partner sets of a set that have members and other cells."""
    import newtypes
    from connectomequest.annotation_v35 import sweep

    d = newtypes.load(view, split)
    found = {prior: [] for prior in newtypes.PRIORS}
    for family in range(len(d["fam_target"])):
        _, _, _, outcomes, priors = newtypes.family_cells(d, family)
        for prior in newtypes.PRIORS:
            value = sweep.member_auroc(priors[prior][:, 0], outcomes)
            if value is not None:
                found[prior].append(value)
    return {prior: float(np.mean(values)) for prior, values in found.items()}


def main():
    a = {split: json.loads((OUT / f"analysis-{split}.json").read_text()) for split, _ in SPLITS}
    verdicts = {split: json.loads((OUT / f"verdicts-{split}.json").read_text()) for split, _ in SPLITS}
    record = json.loads((OUT / "record.json").read_text())
    fit = json.loads((OUT / "fit.json").read_text())
    manifest = json.loads((OUT / "arrays/manifest.json").read_text())
    claims = Claims()
    m = {}

    def ratio(split, view, part, arm, answer="all"):
        return a[split][view][part]["ratio"][arm][answer]

    def contrast(split, view, part, x, y, answer="all"):
        return a[split][view][part]["contrast"][f"{x}/{y}"][answer]

    def regret(split, view, part, policy):
        """Checks of ``policy`` relative to the cheapest of the six untrained policies, on the worse of the two answers."""
        worst = 0.0
        for answer in ("yes", "no"):
            costs = {p: ratio(split, view, part, f"provisional:{p}", answer)["value"] for p in SIX}
            worst = max(worst, costs[policy] / min(costs.values()))
        return worst

    # composition and expectations
    new, held = a["new"], a["heldout"]
    for tag, view, _ in VIEWS:
        m[f"NTTypes{tag}"] = f"{new[view]['types']:,}"
        m[f"NTTargets{tag}"] = f"{new[view]['targets']:,}"
        m[f"NTQuestions{tag}"] = f"{new[view]['questions']:,}"
        m[f"NTRetUncertifiable{tag}"] = pct(new[view]["retention"]["share_uncertifiable"])
        m[f"NTRetNo{tag}"] = pct(new[view]["retention"]["share_no_of_certifiable"])
        m[f"NTWeightMedian{tag}"] = f"{new[view]['retention']['weight_median']:.0f}"
    m["NTTypes"] = f"{sum(new[view]['types'] for _, view, _ in VIEWS):,}"
    m["NTTargets"] = f"{sum(new[view]['targets'] for _, view, _ in VIEWS):,}"
    m["NTQuestions"] = f"{sum(new[view]['questions'] for _, view, _ in VIEWS):,}"
    everything = verdicts["heldout"] + verdicts["new"]
    claims("the record lists the expectations that were evaluated", len(record["expectations"]) == len(everything))
    claims("the simultaneous intervals count every recorded expectation",
           all(a[split]["simultaneous"]["expectations"] == len(everything) for split, _ in SPLITS))
    m["NTExpTotal"], m["NTExpHeld"] = str(len(everything)), str(sum(v["holds"] for v in everything))
    m["NTExpSim"] = str(sum(v["holds_simultaneous"] for v in everything))
    m["NTExpSimFail"] = str(sum(not v["holds_simultaneous"] for v in everything))
    m["NTExpNew"], m["NTExpHeldout"] = str(len(verdicts["new"])), str(len(verdicts["heldout"]))
    m["NTDraws"] = f"{a['new']['simultaneous']['draws']:,}"
    claims("every recorded expectation of the third protocol holds at its point estimate", all(v["holds"] for v in everything))
    claims("the expectations whose simultaneous interval crosses the threshold are all about the new FlyWire types",
           all(v["split"] == "new" and v["quantity"].startswith("flywire|") for v in everything if not v["holds_simultaneous"]))
    for entry in manifest["views"]:
        tag = "Male" if entry["view"] == "malecns" else "Fly"
        m[f"NTFitTypes{tag}"] = f"{entry['sets']['fit']['types']:,}"
        m[f"NTDevTypes{tag}"] = f"{entry['sets']['development']['types']:,}"

    # the priors on the sets of this protocol
    quality = {(view, split): aurocs(view, split) for _, view, _ in VIEWS for split, _ in SPLITS}
    for tag, view, _ in VIEWS:
        m[f"NTMlpAuroc{tag}"] = f2(quality[view, "new"]["mlp_s0"])
        m[f"NTProvAurocNew{tag}"] = f2(quality[view, "new"]["provisional"])
        m[f"NTProvTyped{tag}"] = pct(new[view]["earlier"]["hemibrain_type"])
    m["NTMaleProvAuroc"] = f2(quality["malecns", "heldout"]["provisional"])
    m["NTMaleProvTyped"] = pct(held["malecns"]["earlier"]["hemibrain_type"])
    m["NTMaleProvAgreement"] = f"{100 * held['malecns']['earlier']['agreement']:.1f}"
    m["NTMaleProvTypedGivenResolved"] = pct(held["malecns"]["earlier"]["earlier_type_given_resolved"])
    m["NTMaleProvOutside"], m["NTMaleProvNone"] = pct(held["malecns"]["earlier"]["other_name"]), f"{100 * held['malecns']['earlier']['none']:.1f}"
    claims("both earlier releases name over nine tenths of the input cells of held-out targets",
           all(held[view]["earlier"]["none"] < 0.1 for _, view, _ in VIEWS))
    m["NTCodeFiles"], m["NTDataFiles"] = str(len(record["code"])), str(len(record["data"]))
    for tag, view, _ in VIEWS:
        m[f"NTShareNo{tag}"] = pct(new[view]["share"]["share_no_of_certifiable"])
        m[f"NTShareUncertifiable{tag}"] = pct(new[view]["share"]["share_uncertifiable"])
        m[f"NTProvAgreementNew{tag}"] = f"{100 * new[view]['earlier']['agreement']:.1f}"
    claims("the topology prior is weak on the new cell types too", all(0.6 <= quality[view, "new"]["mlp_s0"] <= 0.75 for _, view, _ in VIEWS))
    claims("the MaleCNS earlier-release prior is strong and agrees with the reference on over 99.5% of the cells both type",
           quality["malecns", "heldout"]["provisional"] >= 0.9 and held["malecns"]["earlier"]["agreement"] >= 0.995)

    # new cell types: propagation, prior quality, earlier labels, trained and generic policies
    for tag, view, _ in VIEWS:
        m[f"NTPropRecal{tag}"] = ci(contrast("new", view, "retention", "mlp_s0:prop", "mlp_s0:recal"))
        m[f"NTNoneProp{tag}"] = ci(ratio("new", view, "retention", "none:prop"))
        m[f"NTRecal{tag}"] = ci(ratio("new", view, "retention", "mlp_s0:recal"))
        m[f"NTProvDirection{tag}"] = ci(ratio("new", view, "retention", "provisional:direction"))
        m[f"NTProvAGDir{tag}"] = ci(contrast("new", view, "retention", "provisional:ag", "provisional:direction"))
        m[f"NTProvRecalDir{tag}"] = ci(contrast("new", view, "retention", "provisional:recal", "provisional:direction"))
        m[f"NTImit{tag}"] = ci(contrast("new", view, "retention", "mlp_s0:imitation", "mlp_s0:recal"))
        m[f"NTImitProv{tag}"] = ci(contrast("new", view, "retention", "provisional:imitation", "provisional:recal"))
        m[f"NTECAG{tag}"] = ci(contrast("new", view, "retention", "mlp_s0:ec2", "mlp_s0:ag"))
        m[f"NTProvECRecal{tag}"] = ci(contrast("new", view, "retention", "provisional:ec2", "provisional:recal"))
        m[f"NTSharePropRecal{tag}"] = ci(contrast("new", view, "share", "mlp_s0:prop", "mlp_s0:recal"))
    props = [contrast(split, view, "retention", "mlp_s0:prop", "mlp_s0:recal")["value"] for split, _ in SPLITS for _, view, _ in VIEWS]
    m["NTPropSaveLowPct"], m["NTPropSaveHighPct"] = pct(1 - max(props)), pct(1 - min(props))
    claims("propagation saves about a fifth of the checks of recalibrated AG on new neurons and on new cell types (15-21%)",
           0.785 <= min(props) and max(props) <= 0.855)
    none = [ratio("new", view, "retention", "none:prop") for _, view, _ in VIEWS]
    claims("without a predictor propagation saves about a fifth of the checks of weight order on new cell types", all(0.75 <= x["value"] <= 0.85 for x in none))
    claims("without a predictor it needs at most 2% fewer or more checks than with the topology prior",
           all(abs(contrast("new", view, "retention", "none:prop", "mlp_s0:prop")["value"] - 1) <= 0.02 for _, view, _ in VIEWS))
    weak = [ratio("new", view, "retention", "mlp_s0:recal")["value"] for _, view, _ in VIEWS]
    m["NTWeakSaveMaxPct"] = pct(1 - min(weak))
    claims("on new cell types the weak prior saves at most 5% with recalibrated AG", min(weak) >= 0.95)
    eight = [ratio(split, view, "retention", "synthetic0.8:ag")["value"] for split, _ in SPLITS for _, view, _ in VIEWS]
    m["NTSynEightSaveLowPct"], m["NTSynEightSaveHighPct"] = pct(1 - max(eight)), pct(1 - min(eight))
    claims("a calibrated synthetic prior of AUROC 0.7 saves nothing with AG on new cell types", all(ratio("new", view, "retention", "synthetic0.7:ag")["value"] >= 1.0 for _, view, _ in VIEWS))
    claims("at AUROC 0.8 it saves 5-13% on new neurons and on new cell types", 0.87 <= min(eight) and max(eight) <= 0.955)
    claims("at AUROC 0.9 it saves a fifth to a third on new cell types", all(0.66 <= ratio("new", view, "retention", "synthetic0.9:ag")["value"] <= 0.82 for _, view, _ in VIEWS))

    # earlier labels for every neuron, MaleCNS (held-out cell types, the target neurons of the replication)
    male = lambda part, arm, answer="all": ratio("heldout", "malecns", part, f"provisional:{arm}", answer)  # noqa: E731
    versus = lambda part, x, y, answer="all": contrast("heldout", "malecns", part, f"provisional:{x}", f"provisional:{y}", answer)  # noqa: E731
    m["NTMaleProvDirection"], m["NTMaleProvAG"], m["NTMaleProvRecal"] = (f2(male("retention", p)["value"]) for p in ("direction", "ag", "recal"))
    m["NTMaleProvDirectionCI"], m["NTMaleProvAGCI"], m["NTMaleProvRecalCI"] = (ci(male("retention", p)) for p in ("direction", "ag", "recal"))
    m["NTMaleProvAGDir"], m["NTMaleProvRecalDir"] = ci(versus("retention", "ag", "direction")), ci(versus("retention", "recal", "direction"))
    m["NTMaleProvImit"], m["NTMaleProvImitDir"] = ci(versus("retention", "imitation", "recal")), ci(versus("retention", "imitation", "direction"))
    m["NTMaleProvImitShare"] = ci(versus("share", "imitation", "recal"))
    m["NTMaleProvECRecal"], m["NTMaleProvIGRecal"] = ci(versus("retention", "ec2", "recal")), ci(versus("retention", "ig", "recal"))
    m["NTMaleProvPropRecal"] = ci(versus("retention", "prop", "recal"))
    m["NTMaleShareRecalDir"] = ci(versus("share", "recal", "direction"))
    m["NTMaleShareRecalDirYes"], m["NTMaleShareRecalDirNo"] = (f2(versus("share", "recal", "direction", x)["value"]) for x in ("yes", "no"))
    m["NTMaleShareAGDirYes"], m["NTMaleShareAGDirNo"] = (f2(versus("share", "ag", "direction", x)["value"]) for x in ("yes", "no"))
    m["NTMaleShareRecalDirPct"] = pct(versus("share", "recal", "direction")["value"] - 1)
    m["NTMaleShareDirSavePct"] = pct(1 - 1 / versus("share", "recal", "direction")["value"])
    claims("on MaleCNS share questions direction-fixed needs 11-15% fewer checks than recalibrated AG at full coverage and as many on retention questions",
           0.11 <= 1 - 1 / versus("share", "recal", "direction")["value"] <= 0.15 and versus("retention", "recal", "direction")["low"] <= 1 <= versus("retention", "recal", "direction")["high"])
    claims("with MaleCNS earlier labels direction-fixed saves over a third of the checks of weight order", male("retention", "direction")["high"] < 0.67)
    claims("AG loses to direction-fixed and recalibration repairs the loss on retention questions",
           versus("retention", "ag", "direction")["low"] > 1.05 and versus("retention", "recal", "direction")["high"] <= 1.04)
    forfeits = [(male("retention", "ag")["value"] - male("retention", "direction")["value"]) / (1 - male("retention", "direction")["value"])]
    m["NTMaleForfeitPct"] = pct(forfeits[0])
    claims("in MaleCNS AG forfeits about a quarter of the saving of direction-fixed", 0.2 <= forfeits[0] <= 0.3)
    claims("on MaleCNS share questions recalibrated AG needs over a tenth more checks than direction-fixed, on no answers",
           versus("share", "recal", "direction")["low"] > 1.1 and versus("share", "recal", "direction", "no")["low"] > 1.1
           and versus("share", "recal", "direction", "yes")["high"] <= 1.02)
    for part, name in (("retention", "Ret"), ("share", "Share")):
        m[f"NTMaleRegret{name}Pct"] = pct(regret("heldout", "malecns", part, "recal") - 1)
        m[f"NTMaleDirRegret{name}Pct"] = pct(regret("heldout", "malecns", part, "direction") - 1)
        m[f"NTMaleAGRegret{name}Pct"] = pct(regret("heldout", "malecns", part, "ag") - 1)
    claims("with MaleCNS earlier labels recalibrated AG exceeds the cheapest of six policies by over a tenth on no answers",
           all(regret("heldout", "malecns", part, "recal") >= 1.1 for part in ("retention", "share"))
           and all(ratio("heldout", "malecns", part, "provisional:recal", "no")["value"] / min(ratio("heldout", "malecns", part, f"provisional:{p}", "no")["value"] for p in SIX) >= 1.1
                   for part in ("retention", "share")))
    claims("direction-fixed stays within 5% of the cheapest of six policies on either answer in MaleCNS",
           all(regret("heldout", "malecns", part, "direction") <= 1.0505 for part in ("retention", "share")))
    new_regret = [regret("new", view, part, "recal") for _, view, _ in VIEWS for part in ("retention", "share")]
    m["NTNewRegretLowPct"], m["NTNewRegretHighPct"] = pct(min(new_regret) - 1), pct(max(new_regret) - 1)
    m["NTNewDirRegretHighPct"] = pct(max(regret("new", view, part, "direction") for _, view, _ in VIEWS for part in ("retention", "share")) - 1)
    imitated = [versus("retention", "imitation", "recal")] + [contrast("new", view, "retention", "provisional:imitation", "provisional:recal") for _, view, _ in VIEWS]
    m["NTImitProvLow"], m["NTImitProvHigh"] = f2(min(x["value"] for x in imitated)), f2(max(x["value"] for x in imitated))
    claims("with earlier labels for every neuron the imitation order saves 7-11% of the checks of recalibrated AG, in MaleCNS and on new cell types",
           all(0.89 <= x["value"] <= 0.93 and x["high"] < 1 for x in imitated))
    weak_imitation = [contrast("new", view, "retention", "mlp_s0:imitation", "mlp_s0:recal")["value"] for _, view, _ in VIEWS]
    claims("with the topology prior the imitation order needs as many checks as recalibrated AG on new cell types (within 1%)", all(abs(x - 1) <= 0.01 for x in weak_imitation))
    second = json.loads((ROOT / "outputs/replication/analysis-test.json").read_text())["flywire"]["retention"]["contrast"]
    second_imitation = json.loads((ROOT / "outputs/imitation/analysis-test.json").read_text())["flywire"]["retention"]["provisional"]["imitation_vs_recal"]["all"]
    every_imitation = [x["value"] for x in imitated] + [second_imitation["value"]]
    m["NTImitProvAllLow"], m["NTImitProvAllHigh"] = f2(min(every_imitation)), f2(max(every_imitation))
    claims("the FlyWire result of the second protocol lies in the same range", 0.89 <= second_imitation["value"] <= 0.93)
    generic = [contrast("new", view, "retention", f"provisional:{g}", "provisional:recal") for _, view, _ in VIEWS for g in ("ec2", "ig")]
    both_protocols = [versus("retention", g, "recal")["value"] for g in ("ec2", "ig")] + [second[f"provisional:{g}/provisional:recal"]["all"]["value"] for g in ("ec2", "ig")]
    m["NTGenericProvAllLowPct"], m["NTGenericProvAllHighPct"] = pct(min(both_protocols) - 1), pct(max(both_protocols) - 1)
    claims("with earlier labels both adapted criteria need at least a sixth more checks than recalibrated AG on the new neurons of both connectomes", min(both_protocols) >= 1.16)
    generic += [versus("retention", g, "recal") for g in ("ec2", "ig")]
    m["NTGenericProvLowPct"], m["NTGenericProvHighPct"] = pct(min(g["value"] for g in generic) - 1), pct(max(g["value"] for g in generic) - 1)
    undercut = [contrast(split, view, part, f"{prior}:{g}", f"{prior}:recal")["value"] for split, _ in SPLITS for _, view, _ in VIEWS for part in ("retention", "share")
                for prior in ("mlp_s0", "provisional") for g in ("ec2", "ig") if f"{prior}:{g}/{prior}:recal" in a[split][view][part]["contrast"]]
    claims("neither adapted criterion needs fewer checks than recalibrated AG in any setting of the third protocol", len(undercut) >= 20 and min(undercut) >= 1.0)
    claims("with earlier labels both adapted criteria need over a quarter more checks than recalibrated AG, in MaleCNS and on new cell types",
           all(g["value"] >= 1.25 and g["low"] > 1 for g in generic))
    topology = [contrast("new", view, "retention", f"mlp_s0:{g}", "mlp_s0:ag") for _, view, _ in VIEWS for g in ("ec2", "ig")]
    m["NTGenericLowPct"], m["NTGenericHighPct"] = pct(min(g["value"] for g in topology) - 1), pct(max(g["value"] for g in topology) - 1)
    claims("with the topology prior both adapted criteria need 2-4% more checks than AG on new cell types (in FlyWire the interval of information gain includes one)",
           all(1.015 <= g["value"] <= 1.04 for g in topology) and sum(g["low"] > 1 for g in topology) >= 3)

    # earlier labels for a part of the neurons
    def cov(split, view, part, level, x, y=None):
        return ratio(split, view, part, f"cov{level}:{x}") if y is None else contrast(split, view, part, f"cov{level}:{x}", f"cov{level}:{y}")

    for tag, view, _ in VIEWS:
        m[f"NTCovQuarterDir{tag}"] = ci(cov("heldout", view, "retention", "0.25", "direction"))
        m[f"NTCovQuarterDirValue{tag}"] = f2(cov("heldout", view, "retention", "0.25", "direction")["value"])
        m[f"NTCovQuarterRecalDir{tag}"] = ci(cov("heldout", view, "retention", "0.25", "recal", "direction"))
        m[f"NTCovQuarterRecalDirValue{tag}"] = f2(cov("heldout", view, "retention", "0.25", "recal", "direction")["value"])
        m[f"NTCovHalfRecalDir{tag}"] = ci(cov("heldout", view, "retention", "0.5", "recal", "direction"))
        m[f"NTCovHalfFull{tag}"] = ci(cov("heldout", view, "retention", "0.5", "imitation-full", "recal"))
        m[f"NTCovHalfMixed{tag}"] = ci(cov("heldout", view, "retention", "0.5", "imitation-mixed", "recal"))
        m[f"NTCovMixedFullCov{tag}"] = ci(contrast("heldout", view, "retention", "cov1.0:imitation-mixed", "provisional:recal"))
        m[f"NTCovMixedNoCov{tag}"] = ci(contrast("heldout", view, "retention", "cov0.0:imitation-mixed", "mlp_s0:recal"))
    sets = [(split, view) for split, _ in SPLITS for _, view, _ in VIEWS]
    low = ("0.25", "0.5")
    recal_dir = [cov(s, v, "retention", level, "recal", "direction") for s, v in sets for level in low]
    full = [cov(s, v, "retention", level, "imitation-full", "recal") for s, v in sets for level in low]
    mixed = [cov(s, v, "retention", level, "imitation-mixed", "recal") for s, v in sets for level in low]
    full_dir = [cov(s, v, "retention", level, "imitation-full", "direction") for s, v in sets for level in low]
    m["NTCovRecalDirLow"], m["NTCovRecalDirHigh"] = f2(min(x["value"] for x in recal_dir)), f2(max(x["value"] for x in recal_dir))
    m["NTCovRecalSaveLowPct"], m["NTCovRecalSaveHighPct"] = pct(1 - max(x["value"] for x in recal_dir)), pct(1 - min(x["value"] for x in recal_dir))
    m["NTCovFullLow"], m["NTCovFullHigh"] = f2(min(x["value"] for x in full)), f2(max(x["value"] for x in full))
    m["NTCovMixedLow"], m["NTCovMixedHigh"] = f2(min(x["value"] for x in mixed)), f2(max(x["value"] for x in mixed))
    claims("at a quarter coverage direction-fixed needs more checks than weight order, on held-out and on new cell types",
           all(cov(s, v, "retention", "0.25", "direction")["low"] > 1 for s, v in sets))
    claims("at a quarter and at half coverage recalibrated AG needs at most 0.83 of the checks of direction-fixed", all(x["value"] <= 0.83 and x["high"] < 1 for x in recal_dir))
    claims("that is, a fifth to a third fewer checks (18-33%)", 0.665 <= min(x["value"] for x in recal_dir) and max(x["value"] for x in recal_dir) <= 0.825)
    share_low = [cov(s, v, "share", level, "imitation-mixed", "recal")["value"] for s, v in sets for level in low]
    claims("at a quarter and at half coverage the order fitted at varying coverage does not undercut recalibrated AG on share questions either (at most 0.1% fewer checks)", min(share_low) >= 0.999)
    claims("the half-coverage rows of the table of the main text show that order level with recalibrated AG in FlyWire and 4-6% above it in MaleCNS",
           abs(cov("heldout", "flywire", "retention", "0.5", "imitation-mixed", "recal")["value"] - 1) <= 0.01 and 1.04 <= cov("heldout", "malecns", "retention", "0.5", "imitation-mixed", "recal")["value"] <= 1.06)
    claims("fitted at full coverage the imitation order needs over 1.7 times the checks of recalibrated AG there", all(x["value"] >= 1.7 and x["low"] > 1.4 for x in full))
    claims("and more checks than direction-fixed", all(x["low"] > 1 for x in full_dir))
    claims("fitted at varying coverage it needs 1.00-1.06 of the checks of recalibrated AG and never fewer",
           all(0.995 <= x["value"] <= 1.06 for x in mixed) and all(x["high"] >= 1 for x in mixed))
    every = [cov(s, v, "retention", level, "recal", other) for s, v in sets for level, _, _ in LEVELS for other in ("direction", "ag")]
    claims("recalibrated AG needs fewer checks than direction-fixed and AG at every partial coverage on retention questions", all(x["value"] < 1 for x in every))
    mixed_all = [cov(s, v, part, level, "imitation-mixed", "recal")["value"] for s, v in sets for part in ("retention", "share") for level, _, _ in LEVELS]
    m["NTCovMixedAllLow"], m["NTCovMixedAllHigh"] = f2(min(mixed_all)), f2(max(mixed_all))
    claims("on both kinds of question and at every partial coverage the order fitted at varying coverage is within 6% of recalibrated AG", 0.94 <= min(mixed_all) and max(mixed_all) <= 1.06)
    share_dir = [cov(s, v, "share", level, "recal", "direction")["value"] for s, v in sets for level in low]
    m["NTCovShareRecalDirLow"], m["NTCovShareRecalDirHigh"] = f2(min(share_dir)), f2(max(share_dir))
    claims("on share questions recalibrated AG needs fewer checks than direction-fixed at a quarter and at half coverage", max(share_dir) < 1)
    three = [cov(s, "malecns", "share", "0.75", "recal", "direction")["value"] for s, _ in SPLITS]
    m["NTCovShareThreeMalePct"] = pct(max(three) - 1)
    claims("at three quarters coverage direction-fixed is cheaper on MaleCNS share questions", min(three) > 1.04)
    claims("at a quarter and at half coverage outcome propagation without a predictor needs fewer checks than recalibrated AG with the "
           "partial earlier labels, on retention questions",
           all(ratio(s, v, "retention", "none:prop")["value"] < cov(s, v, "retention", level, "recal")["value"] for s, v in sets for level in low))
    claims("at a quarter and at half coverage outcome propagation without a predictor needs fewer checks than every policy that uses "
           "the partial earlier labels, on retention questions",
           all(ratio(s, v, "retention", "none:prop")["value"] < cov(s, v, "retention", level, arm)["value"]
               for s, v in sets for level in low for arm, _ in AT_COVERAGE))
    for level, tag in (("0.25", "Quarter"), ("0.5", "Half")):
        committed = [cov(s, v, "retention", level, "direction")["value"] for s, v in sets]
        m[f"NTCov{tag}DirLow"], m[f"NTCov{tag}DirHigh"] = f2(min(committed)), f2(max(committed))
    claims("at half coverage direction-fixed still needs about as many checks as weight order or more (0.95 to 1.2 times)",
           all(0.95 <= cov(s, v, "retention", "0.5", "direction")["value"] <= 1.2 for s, v in sets))
    claims("with earlier labels for every neuron the imitation order needs up to a tenth fewer checks than recalibrated AG and "
           "than direction-fixed",
           all(0.89 <= x["value"] < 1 for x in imitated) and 0.9 <= versus("retention", "imitation", "direction")["value"] < 1)
    claims("at a quarter coverage direction-fixed costs a fifth to a third more than weight order",
           all(1.15 <= cov(s, v, "retention", "0.25", "direction")["value"] <= 1.35 for s, v in sets))
    claims("at a quarter coverage recalibrated AG still saves about a tenth of the checks of weight order (7-12%)",
           all(0.88 <= cov(s, v, "retention", "0.25", "recal")["value"] <= 0.93 for s, v in sets))
    claims("recalibrated AG saves checks at every partial coverage, with its whole interval",
           all(cov(s, v, "retention", level, "recal")["high"] < 1 for s, v in sets for level, _, _ in LEVELS))
    claims("fitted at full coverage the imitation order needs more checks than weight order at every partial coverage",
           all(cov(s, v, "retention", level, "imitation-full")["value"] > 1 for s, v in sets for level, _, _ in LEVELS))
    claims("at half coverage direction-fixed needs more checks than weight order, with its whole interval, in three of the four "
           "sets of questions",
           sum(cov(s, v, "retention", "0.5", "direction")["low"] > 1 for s, v in sets) == 3 and len(sets) == 4)
    claims("at a quarter and at half coverage recalibrated AG saves a tenth to a fifth of the checks of weight order (7-20%)",
           all(0.80 <= cov(s, v, "retention", level, "recal")["value"] <= 0.93 for s, v in sets for level in low))
    claims("outcome propagation without a predictor saves a fifth of the checks of weight order in the four sets (19-23%)",
           all(0.77 <= ratio(s, v, "retention", "none:prop")["value"] <= 0.81 for s, v in sets))
    for level, tag in (("0.25", "Quarter"), ("0.5", "Half")):
        recalibrated = [cov(s, v, "retention", level, "recal")["value"] for s, v in sets]
        m[f"NTCov{tag}RecalLow"], m[f"NTCov{tag}RecalHigh"] = f2(min(recalibrated)), f2(max(recalibrated))
    margins = {level: [cov(s, v, "retention", level, "recal")["value"] - ratio(s, v, "retention", "none:prop")["value"] for s, v in sets]
               for level in low}
    claims("outcome propagation without a predictor undercuts recalibrated AG by about a tenth of the checks of weight order at a "
           "quarter coverage and by a smaller margin at a half",
           all(0.08 <= x <= 0.13 for x in margins["0.25"]) and all(0 < x <= 0.07 for x in margins["0.5"])
           and max(margins["0.5"]) < min(margins["0.25"]))
    without = [ratio(s, v, "retention", "none:prop")["value"] for s, v in sets]
    m["NTCovNonePropLow"], m["NTCovNonePropHigh"] = f2(min(without)), f2(max(without))
    claims("fitted at full coverage the imitation order needs more checks than direction-fixed at every partial coverage",
           all(cov(s, v, "retention", level, "imitation-full", "direction")["value"] > 1 for s, v in sets for level, _, _ in LEVELS))
    claims("at three quarters coverage recalibrated AG needs fewer checks than outcome propagation without a predictor",
           all(cov(s, v, "retention", "0.75", "recal")["value"] < ratio(s, v, "retention", "none:prop")["value"] for s, v in sets))
    for key in ("malecns-mixed", "flywire-mixed", "malecns-provisional"):
        tag = "".join(part.capitalize() for part in key.replace("malecns", "male").replace("flywire", "fly").split("-"))
        m[f"NTFitQuestions{tag}"] = f"{fit['models'][key]['questions']:,}"
        m[f"NTFitAuroc{tag}"] = f2(fit["models"][key]["training_auroc"])

    # tables of the appendix
    def short(d):
        return f"{d['value']:.2f} [{d['low']:.2f}, {d['high']:.2f}]"

    questions = []
    for _, view, name in VIEWS:
        for part in ("retention", "share"):
            p = new[view][part]
            questions.append(f"{name}, {part} & {p['questions']:,} & {pct(p['share_uncertifiable'])} & {p['answers']['yes']:,} & {p['answers']['no']:,} & {p['weight_median']:.0f}\\\\")
    prior_name = {"mlp_s0": "topology prior", "provisional": "earlier labels", "cov0.25": "earlier labels for $\\tfrac14$", "cov0.5": "earlier labels for $\\tfrac12$",
                  "none": "no predictor", "synthetic0.7": "synthetic prior, AUROC 0.7", "synthetic0.9": "synthetic prior, AUROC 0.9"}
    policy_name = {"prop": "propagation", "recal": "recal.\\ AG", "ag": "AG", "direction": "direction-fixed", "ec2": "edge cutting", "imitation": "imitation",
                   "imitation-full": "imitation$^b$", "imitation-mixed": "imitation$^c$"}

    def described(quantity):
        view, part, kind, key = quantity.split("|")[:4]
        first, _, second = key.partition("/")
        prior, policy = first.split(":")
        other = policy_name[second.split(":")[1]] if kind == "contrast" else "weight order"
        return f"{'Share: ' if part == 'share' else ''}{prior_name[prior]}, {policy_name[policy]} / {other}"

    def bound(v):
        if v["min"] is not None and v["max"] is not None:
            return f"{v['min']:.2f}--{v['max']:.2f}"
        return f"$\\geq$ {v['min']:.2f}" if v["min"] is not None else f"$\\leq$ {v['max']:.2f}"

    expectations = []
    for split, label in SPLITS:
        for _, view, name in VIEWS:
            for v in verdicts[split]:
                if not v["quantity"].startswith(view + "|"):
                    continue
                measure = a[split]
                for part in v["quantity"].split("|")[:-1]:
                    measure = measure[part]
                claims(f"expectation {v['id']} is described by its text", v["text"].startswith(name + ", ") and ("share questions" in v["text"]) == ("|share|" in v["quantity"]))
                if v["holds_simultaneous"]:
                    continue  # the complete list is in outputs/newtypes/verdicts-*.json
                expectations.append(f"{described(v['quantity'])} & {bound(v)} & {short(measure)} & [{v['low_simultaneous']:.2f}, {v['high_simultaneous']:.2f}]\\\\")
    policies = []
    for key, label in POLICIES:
        cells = [short(ratio("new", view, "retention", f"{prior}:{key}")) for _, view, _ in VIEWS for prior in ("mlp_s0", "provisional")]
        policies.append(label + "".join(f" & {x}" for x in cells) + "\\\\")
    policies.append("Propagation without a predictor" + "".join(f" & {short(ratio('new', view, 'retention', 'none:prop'))} & ---" for _, view, _ in VIEWS) + "\\\\")
    for level in (0.7, 0.8, 0.9):
        policies.append(f"AG, calibrated synthetic prior of AUROC {level}" + "".join(f" & {short(ratio('new', view, 'retention', f'synthetic{level}:ag'))} & ---" for _, view, _ in VIEWS) + "\\\\")
    policies.append("\\midrule\nClairvoyant bound" + "".join(f" & \\multicolumn{{2}}{{c}}{{{short(ratio('new', view, 'retention', 'clairvoyant'))}}}" for _, view, _ in VIEWS) + "\\\\")
    earlier = []
    for key, label in POLICIES:
        cells = [f2(male(part, key, answer)["value"]) for part in ("retention", "share") for answer in ("all", "yes", "no")]
        earlier.append(SHORT.get(label, label) + "".join(f" & {x}" for x in cells) + "\\\\")
    earlier.append("\\midrule\nClairvoyant bound" + "".join(f" & {f2(ratio('heldout', 'malecns', part, 'clairvoyant', answer)['value'])}" for part in ("retention", "share") for answer in ("all", "yes", "no")) + "\\\\")
    coverage = []
    for level, _, fraction in LEVELS:
        coverage.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Earlier labels for ${fraction}$ of the neurons}}}}\\\\")
        for key, label in AT_COVERAGE:
            cells = [short(cov(s, v, "retention", level, key)) for s, v in sets] + [f2(cov(s, v, "share", level, key)["value"]) for s, v in sets]
            coverage.append("\\quad " + label + "".join(f" & {x}" for x in cells) + "\\\\")

    if claims.failed:
        raise SystemExit("claims of the paper no longer hold:\n  " + "\n  ".join(claims.failed))
    taken = set()
    for name in ("v35_numbers.tex", "r5_numbers.tex", "answers_numbers.tex", "replication_numbers.tex"):
        taken |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / name).read_text()))
    if set(m) & taken:
        raise SystemExit(f"macros defined twice: {sorted(set(m) & taken)}")
    text = ["% Generated by scripts/newtypes_assets.py; do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / "newtypes_numbers.tex").write_text("\n".join(text) + "\n")
    for name, body in (("newtypes_questions", questions), ("newtypes_expectations", expectations), ("newtypes_policies", policies),
                       ("newtypes_earlier", earlier), ("newtypes_coverage", coverage)):
        (PAPER / f"tables/{name}.tex").write_text("\n".join(body) + "\n")
    print(f"{len(m)} macros, {m['NTExpHeld']} of {m['NTExpTotal']} expectations hold, {m['NTExpSim']} with the whole simultaneous interval")


if __name__ == "__main__":
    main()
