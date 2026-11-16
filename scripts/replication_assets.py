"""Numbers and tables of the paper from the replication on new target neurons (scripts/replication.py).

Writes ``paper/replication_numbers.tex`` (macros), ``paper/tables/replication.tex`` (the table of the main text),
``paper/tables/replication_policies.tex`` and ``paper/tables/replication_earlier.tex`` (every policy, appendix),
``paper/tables/replication_questions.tex`` (composition) and ``paper/tables/replication_failed.tex`` (recorded
expectations that do not hold). Every sentence of the paper that rests on these numbers states a direction or a
bound; it is checked here, and the build stops if a result contradicts it.

Two trained policies evaluated on the same target neurons under records of their own are read from
``outputs/learned`` (the tuned policy, scripts/learned.py) and ``outputs/imitation`` (the imitation order,
scripts/imitation.py); the share of checks that edge cutting and information gain choose by their fallback on
development types is read from ``outputs/fallback`` (scripts/fallback_share.py). Also writes
``paper/tables/replication_share_levels.tex`` (answers to share questions by threshold).

Usage: python scripts/replication_assets.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r5_paper_assets import Claims, ci, f2  # noqa: E402

PAPER = ROOT / "paper"
OUT = ROOT / "outputs/replication"
LEARNED = ROOT / "outputs/learned"
IMITATION = ROOT / "outputs/imitation"
FALLBACK = ROOT / "outputs/fallback/development.json"
NEWTYPES = ROOT / "outputs/newtypes"
VIEWS = (("Male", "malecns"), ("Fly", "flywire"))
POLICIES = (("member", "Member-first"), ("nonmember", "Nonmember-first"), ("direction", "Direction-fixed"), ("ag", "AG"),
            ("adg", "Adaptive Dual Greedy"), ("recal", "AG + recalibration"), ("single-recal", "Single-goal + recalibration"),
            ("prop", "AG + recalibration + propagation"), ("ec2", "Edge cutting"), ("ig", "Information gain"),
            ("ec2-recal", "Edge cutting + recalibration"), ("ig-recal", "Information gain + recalibration"))
SHORT = {"Adaptive Dual Greedy": "ADG", "AG + recalibration": "AG + recal.", "Single-goal + recalibration": "Single-goal + recal.",
         "AG + recalibration + propagation": "AG + recal.\\ + prop.", "Edge cutting + recalibration": "Edge cutting + recal.",
         "Information gain + recalibration": "Inf.\\ gain + recal.", "Information gain": "Inf.\\ gain"}


def pct(x):
    return f"{100 * x:.0f}"


def main():
    r = json.loads((OUT / "analysis-test.json").read_text())
    verdicts = json.loads((OUT / "verdicts-test.json").read_text())
    record = json.loads((OUT / "record.json").read_text())
    male, fly = r["malecns"], r["flywire"]
    claims = Claims()
    m = {}

    def ratio(view, part, arm, answer="all"):
        return r[view][part]["ratio"][arm][answer]

    def contrast(view, part, a, b, answer="all"):
        return r[view][part]["contrast"][f"{a}/{b}"][answer]

    # composition
    for tag, view in VIEWS:
        m[f"RepTypes{tag}"] = f"{r[view]['types']:,}"
        m[f"RepTargets{tag}"] = f"{r[view]['targets']:,}"
        for part, name in (("retention", "Ret"), ("share", "Share")):
            p = r[view][part]
            m[f"Rep{name}No{tag}"] = pct(p["share_no_of_certifiable"])
            m[f"Rep{name}Uncertifiable{tag}"] = pct(p["share_uncertifiable"])
    m["RepTargets"] = f"{male['targets'] + fly['targets']:,}"
    m["RepQuestions"] = f"{male['questions'] + fly['questions']:,}"
    m["RepExpTotal"], m["RepExpHeld"] = str(len(verdicts)), str(sum(v["holds"] for v in verdicts))
    claims("the record lists the expectations that were evaluated", len(record["expectations"]) == len(verdicts))
    shares = [r[view]["share"]["share_no_of_certifiable"] for _, view in VIEWS]
    m["RepShareNoLow"], m["RepShareNoHigh"] = pct(min(shares)), pct(max(shares))
    claims("two thirds or more of the certifiable share questions are answered no", min(shares) >= 0.65)
    claims("retention questions of the new targets are mostly answered yes, as in the benchmark",
           max(r[view]["retention"]["share_no_of_certifiable"] for _, view in VIEWS) <= 0.2)

    levels = {}
    for tag, view in VIEWS:
        for level in ("1/4", "1/2", "3/4"):
            a = r[view]["share"]["by_level"][level]["answers"]
            levels[tag, level] = (a["yes"], a["no"], a["uncertifiable"])
    quarter = [levels[tag, "1/4"][0] / sum(levels[tag, "1/4"][:2]) for tag, _ in VIEWS]
    three = [levels[tag, "3/4"][1] / sum(levels[tag, "3/4"][:2]) for tag, _ in VIEWS]
    m["RepShareQuarterYesLow"], m["RepShareQuarterYesHigh"] = pct(min(quarter)), pct(max(quarter))
    m["RepShareThreeNoLow"] = pct(min(three))
    claims("at a quarter most certifiable share questions are answered yes, at three quarters nearly all no", min(quarter) >= 0.7 and min(three) >= 0.98)
    for tag, view in VIEWS:
        m[f"RepShareClairNo{tag}"] = f2(ratio(view, "share", "clairvoyant", "no")["value"])

    # outcome propagation
    for tag, view in VIEWS:
        m[f"RepPropRecal{tag}"] = ci(contrast(view, "retention", "mlp_s0:prop", "mlp_s0:recal"))
        m[f"RepNoneProp{tag}"] = ci(ratio(view, "retention", "none:prop"))
        m[f"RepSharePropRecal{tag}"] = ci(contrast(view, "share", "mlp_s0:prop", "mlp_s0:recal"))
        m[f"RepRecal{tag}"] = ci(ratio(view, "retention", "mlp_s0:recal"))
    props = [contrast(view, "retention", "mlp_s0:prop", "mlp_s0:recal") for _, view in VIEWS]
    claims("propagation saves about a fifth of the checks of recalibrated AG (15-25%)", all(0.75 <= p["value"] <= 0.85 and p["high"] < 0.9 for p in props))
    none = [contrast(view, "retention", "none:prop", "mlp_s0:prop")["value"] for _, view in VIEWS]
    claims("without a predictor propagation needs as few checks as with the topology prior (within 1%)", all(abs(x - 1) <= 0.01 for x in none))
    claims("without a predictor propagation saves about a fifth relative to weight order", all(0.75 <= ratio(view, "retention", "none:prop")["value"] <= 0.85 for _, view in VIEWS))
    m["RepProvPropRecal"] = ci(contrast("flywire", "retention", "provisional:prop", "provisional:recal"))
    claims("with the earlier-release prior propagation adds less than with the topology prior",
           contrast("flywire", "retention", "provisional:prop", "provisional:recal")["value"] > props[1]["value"])
    claims("the weak prior saves at most 5% on the new targets", all(ratio(view, "retention", "mlp_s0:recal")["value"] >= 0.95 for _, view in VIEWS))

    # quality of the prior
    for tag, view in VIEWS:
        for level, name in ((0.7, "Seven"), (0.8, "Eight"), (0.9, "Nine")):
            m[f"RepSyn{name}{tag}"] = f2(ratio(view, "retention", f"synthetic{level}:ag")["value"])
    claims("a calibrated prior of AUROC 0.7 saves nothing with AG", all(ratio(view, "retention", "synthetic0.7:ag")["value"] >= 1.0 for _, view in VIEWS))
    claims("at AUROC 0.8 it saves 5-10%", all(0.90 <= ratio(view, "retention", "synthetic0.8:ag")["value"] <= 0.955 for _, view in VIEWS))
    claims("at AUROC 0.9 it saves a fifth to a quarter", all(0.745 <= ratio(view, "retention", "synthetic0.9:ag")["value"] <= 0.82 for _, view in VIEWS))
    eight = [ratio(view, "retention", "synthetic0.8:ag")["value"] for _, view in VIEWS]
    m["RepSynEightSaveLowPct"], m["RepSynEightSaveHighPct"] = pct(1 - max(eight)), pct(1 - min(eight))

    # the hedge and its repair, earlier-release prior
    m["RepProvDirection"] = ci(ratio("flywire", "retention", "provisional:direction"))
    m["RepProvAGDir"] = ci(contrast("flywire", "retention", "provisional:ag", "provisional:direction"))
    m["RepProvRecalDir"] = ci(contrast("flywire", "retention", "provisional:recal", "provisional:direction"))
    claims("on new targets AG again loses to direction-fixed and recalibration repairs most of the loss",
           contrast("flywire", "retention", "provisional:ag", "provisional:direction")["low"] > 1.2
           and contrast("flywire", "retention", "provisional:recal", "provisional:direction")["high"] < 1.06)

    # both answers on share questions
    for tag, view in VIEWS:
        m[f"RepShareRecalNo{tag}"] = f2(ratio(view, "share", "mlp_s0:recal", "no")["value"])
    topo_no = [abs(ratio(view, "share", f"mlp_s0:{p}", "no")["value"] - 1) for _, view in VIEWS for p in ("ag", "adg", "recal")]
    m["RepShareTopoNoDevPct"] = f"{100 * max(topo_no):.0f}"
    claims("on share questions answered no the adaptive policies with the topology prior stay within 1% of weight order", max(topo_no) <= 0.0105)
    m["RepShareMemberNo"] = f2(ratio("flywire", "share", "provisional:member", "no")["value"])
    m["RepShareNonmemberYes"] = f2(ratio("flywire", "share", "provisional:nonmember", "yes")["value"])
    claims("a policy committed to one answer costs more than weight order on the other",
           ratio("flywire", "share", "provisional:member", "no")["low"] > 1.2 and ratio("flywire", "share", "provisional:nonmember", "yes")["low"] > 1.5)
    yes, no = (contrast("flywire", "share", "provisional:ag", "provisional:direction", a) for a in ("yes", "no"))
    m["RepShareAGDirYes"], m["RepShareAGDirNo"] = ci(yes), ci(no)
    claims("the hedge of AG costs on yes questions and at most 2% on no questions", yes["low"] > 1.15 and no["high"] <= 1.025)
    regrets = [fly[part]["regret"]["provisional"]["recal"]["worst"] for part in ("retention", "share")]
    m["RepRegretPct"] = pct(max(regrets) - 1)
    m["RepRegretRetPct"], m["RepRegretSharePct"] = pct(regrets[0] - 1), pct(regrets[1] - 1)
    limits = {part: next(x["max"] for x in record["expectations"] if x["quantity"] == f"flywire|{part}|regret|provisional|recal|worst") for part in ("retention", "share")}
    m["RepRegretRetLimitPct"], m["RepRegretShareLimitPct"] = pct(limits["retention"] - 1), pct(limits["share"] - 1)
    claims("the recorded limits on the excess of recalibrated AG over the cheapest policy hold", regrets[0] <= limits["retention"] and regrets[1] <= limits["share"])
    claims("recalibrated AG is within 5% of the cheapest of six policies on both answers of both kinds of question", max(regrets) <= 1.05)
    direction = [fly[part]["regret"]["provisional"]["direction"]["worst"] for part in ("retention", "share")]
    m["RepDirectionRegretPct"] = pct(max(direction) - 1)
    m["RepDirRegretRetPct"] = f"{100 * (fly['retention']['regret']['provisional']['direction']['worst'] - 1):.2f}"
    m["RepDirRegretSharePct"] = f"{100 * (fly['share']['regret']['provisional']['direction']['worst'] - 1):.1f}"
    claims("direction-fixed is as close to the cheapest policy", max(direction) <= 1.055)
    others = {p: max(fly[part]["regret"]["provisional"][p]["worst"] for part in ("retention", "share")) for p in ("member", "nonmember", "ag", "adg")}
    m["RepOthersRegretPct"] = pct(min(others.values()) - 1)
    claims("the other four policies exceed the cheapest by more than a fifth on one answer", min(others.values()) >= 1.2)
    recal = [ratio(view, part, f"{prior}:recal", a)["value"] for _, view in VIEWS for part in ("retention", "share")
             for prior in (("mlp_s0", "provisional") if view == "flywire" else ("mlp_s0",)) for a in ("yes", "no")]
    claims("recalibrated AG never needs more checks than weight order on either answer (to rounding)", max(recal) <= 1.0005)

    # generic policies
    for tag, view in VIEWS:
        m[f"RepECAG{tag}"] = ci(contrast(view, "retention", "mlp_s0:ec2", "mlp_s0:ag"))
        m[f"RepIGAG{tag}"] = ci(contrast(view, "retention", "mlp_s0:ig", "mlp_s0:ag"))
        m[f"RepSingle{tag}"] = ci(contrast(view, "retention", "mlp_s0:single-recal", "mlp_s0:recal"))
        m[f"RepStopEarly{tag}"] = pct(r[view]["retention"]["own_stop"]["mlp_s0"]["before_certificate"])
        m[f"RepStopWrong{tag}"] = pct(r[view]["retention"]["own_stop"]["mlp_s0"]["wrong_decision"])
    m["RepProvECRecal"] = ci(contrast("flywire", "retention", "provisional:ec2", "provisional:recal"))
    m["RepProvIGRecal"] = ci(contrast("flywire", "retention", "provisional:ig", "provisional:recal"))
    m["RepProvSingle"] = ci(contrast("flywire", "retention", "provisional:single-recal", "provisional:recal"))
    m["RepProvStopEarly"] = pct(fly["retention"]["own_stop"]["provisional"]["before_certificate"])
    m["RepProvStopWrong"] = pct(fly["retention"]["own_stop"]["provisional"]["wrong_decision"])
    generic = [contrast(view, "retention", f"mlp_s0:{g}", "mlp_s0:ag") for _, view in VIEWS for g in ("ec2", "ig")]
    m["RepGenericLowPct"], m["RepGenericHighPct"] = pct(min(g["value"] for g in generic) - 1), pct(max(g["value"] for g in generic) - 1)
    claims("with the topology prior both generic policies need more checks than AG", all(g["low"] > 1 for g in generic))
    strong = [contrast("flywire", "retention", f"provisional:{g}", "provisional:recal") for g in ("ec2", "ig")]
    m["RepGenericProvLowPct"], m["RepGenericProvHighPct"] = pct(min(g["value"] for g in strong) - 1), pct(max(g["value"] for g in strong) - 1)
    claims("with the earlier-release prior they need at least a sixth more checks than recalibrated AG", all(g["low"] > 1.16 for g in strong))
    worse = [contrast(view, "retention", f"mlp_s0:{g}-recal", "mlp_s0:recal")["low"] for _, view in VIEWS for g in ("ec2", "ig")]
    worse += [contrast("flywire", "retention", f"provisional:{g}-recal", "provisional:recal")["low"] for g in ("ec2", "ig")]
    claims("their recalibrated variants need more checks than recalibrated AG", min(worse) > 1.0)
    same = [contrast("flywire", "retention", f"provisional:{g}", "provisional:ag") for g in ("ec2", "ig")]
    claims("with the earlier-release prior they need as many checks as AG on retention questions (intervals include one)", all(g["low"] <= 1 <= g["high"] for g in same))
    settings = [(view, prior) for _, view in VIEWS for prior in (("mlp_s0", "provisional") if view == "flywire" else ("mlp_s0",))]
    variants = {g: [contrast(view, "retention", f"{prior}:{g}-recal", f"{prior}:recal") for view, prior in settings] for g in ("ec2", "ig")}
    for g, name in (("ec2", "EC"), ("ig", "IG")):
        m[f"Rep{name}RecalLowPct"], m[f"Rep{name}RecalHighPct"] = pct(min(x["value"] for x in variants[g]) - 1), pct(max(x["value"] for x in variants[g]) - 1)
    m["RepGenericRecalLowPct"], m["RepGenericRecalHighPct"] = m["RepIGRecalLowPct"], m["RepECRecalHighPct"]
    claims("given the recalibrated probabilities both criteria need at least 6% more checks than recalibrated AG", min(x["low"] for g in variants for x in variants[g]) > 1.06)
    claims("edge cutting needs more checks with the recalibrated probabilities than with the prior",
           all(contrast(view, "retention", f"{prior}:ec2-recal", f"{prior}:recal")["value"] > contrast(view, "retention", f"{prior}:ec2", f"{prior}:recal")["value"] for view, prior in settings))
    fallback = json.loads(FALLBACK.read_text())
    raw = [fallback[view]["retention"][prior][g]["share"] for view, prior in settings for g in ("ec2", "ig")]
    recalibrated = [fallback[view]["retention"][prior][f"{g}-recal"]["share"] for view, prior in settings for g in ("ec2", "ig")]
    shares = [fallback[view]["share"][prior][g]["share"] for view, prior in settings for g in ("ec2", "ig")]
    m["FallbackLowPct"], m["FallbackHighPct"] = pct(min(raw)), pct(max(raw))
    m["FallbackRecalLowPct"], m["FallbackRecalHighPct"] = pct(min(recalibrated)), pct(max(recalibrated))
    m["FallbackShareLowPct"], m["FallbackShareHighPct"] = pct(min(shares)), pct(max(shares))
    m["FallbackQuestions"] = f"{sum(fallback[view]['retention']['mlp_s0']['ec2']['questions'] + fallback[view]['share']['mlp_s0']['ec2']['questions'] for _, view in VIEWS):,}"
    claims("the fallback chooses about three quarters or more of the checks of both criteria on development questions", min(raw) >= 0.7)
    claims("and fewer with the recalibrated probabilities", max(recalibrated) < min(raw))
    everywhere = [contrast(view, part, f"{prior}:{g}", f"{prior}:recal")["low"] for _, view in VIEWS for part in ("retention", "share")
           for prior in (("mlp_s0", "provisional") if view == "flywire" else ("mlp_s0",)) for g in ("ec2", "ig")]
    claims("both generic policies need more checks than recalibrated AG under every prior and kind of question", min(everywhere) > 1.0)
    stops = [r[view]["retention"]["own_stop"]["mlp_s0"] for _, view in VIEWS]
    claims("their stopping rule fires before a certificate on about nine in ten certifiable questions", all(0.85 <= s["before_certificate"] <= 0.95 for s in stops))
    claims("and with a decision other than the certified answer on about four in five", all(0.75 <= s["wrong_decision"] <= 0.85 for s in stops))
    singles = [contrast(view, "retention", "mlp_s0:single-recal", "mlp_s0:recal") for _, view in VIEWS]
    m["RepSingleLowPct"], m["RepSingleHighPct"] = pct(min(s["value"] for s in singles) - 1), pct(max(s["value"] for s in singles) - 1)
    claims("the single-goal policy with recalibration needs more checks than recalibrated AG with the topology prior", all(s["low"] > 1.03 for s in singles))
    claims("and about as many with the earlier-release prior", contrast("flywire", "retention", "provisional:single-recal", "provisional:recal")["high"] <= 1.04)

    # appendix
    manifest = json.loads((OUT / "benchmark/manifest.json").read_text())
    for tag, view in VIEWS:
        entry = next(v for v in manifest["views"] if v["view"] == view)
        m[f"RepTypesDev{tag}"] = f"{entry['types_by_split']['development']:,}"
        m[f"RepReproduced{tag}"] = f"{entry['reproduced_old_targets']:,}"
        m[f"RepShareNoChecks{tag}"] = pct(r[view]["share"]["share_no_checks"])
        m[f"RepStopFirst{tag}"] = pct(r[view]["retention"]["own_stop"]["mlp_s0"]["at_first_check"])
        m[f"RepShareStopWrong{tag}"] = pct(r[view]["share"]["own_stop"]["mlp_s0"]["wrong_decision"])
        m[f"RepECRecal{tag}"] = ci(contrast(view, "retention", "mlp_s0:ec2-recal", "mlp_s0:recal"))
        m[f"RepIGRecal{tag}"] = ci(contrast(view, "retention", "mlp_s0:ig-recal", "mlp_s0:recal"))
        m[f"RepNoneVsTopo{tag}"] = f2(contrast(view, "retention", "none:prop", "mlp_s0:prop")["value"])
    m["RepProvECRecalRecal"] = ci(contrast("flywire", "retention", "provisional:ec2-recal", "provisional:recal"))
    m["RepProvIGRecalRecal"] = ci(contrast("flywire", "retention", "provisional:ig-recal", "provisional:recal"))
    m["RepProvECAG"] = ci(contrast("flywire", "retention", "provisional:ec2", "provisional:ag"))
    m["RepProvIGAG"] = ci(contrast("flywire", "retention", "provisional:ig", "provisional:ag"))
    m["RepProvShareStopWrong"] = pct(fly["share"]["own_stop"]["provisional"]["wrong_decision"])
    m["RepProvSharePropRecal"] = ci(contrast("flywire", "share", "provisional:prop", "provisional:recal"))
    m["RepProvShareECRecal"] = ci(contrast("flywire", "share", "provisional:ec2", "provisional:recal"))
    m["RepProvRegretRet"] = f2(fly["retention"]["regret"]["provisional"]["recal"]["worst"])
    m["RepProvRegretShare"] = f2(fly["share"]["regret"]["provisional"]["recal"]["worst"])
    m["RepProvDirRegretRet"] = f2(fly["retention"]["regret"]["provisional"]["direction"]["worst"])
    m["RepProvDirRegretShare"] = f2(fly["share"]["regret"]["provisional"]["direction"]["worst"])
    topo = [r[view][part]["regret"]["mlp_s0"]["recal"]["worst"] for _, view in VIEWS for part in ("retention", "share")]
    m["RepTopoRegretPct"] = f"{100 * (max(topo) - 1):.1f}"
    claims("with the topology prior recalibrated AG is within 1% of the cheapest of the six policies on both answers", max(topo) <= 1.01)
    claims("the stopping rule mostly fires before the first check with the topology prior", all(r[view]["retention"]["own_stop"]["mlp_s0"]["at_first_check"] >= 0.8 for _, view in VIEWS))
    claims("the stopping rule is wrong less often on share questions and with earlier labels",
           max(r[view]["share"]["own_stop"]["mlp_s0"]["wrong_decision"] for _, view in VIEWS) <= 0.35 and fly["retention"]["own_stop"]["provisional"]["wrong_decision"] <= 0.25)
    saving = [1 - d["all"]["value"] for _, view in VIEWS for k, d in r[view]["share"]["ratio"].items() if k.startswith("mlp_s0:") and not k.endswith(":prop")]
    claims("on share questions with the topology prior no policy but propagation saves more than 1%", max(saving) <= 0.01)
    claims("the builder reproduced the stored arrays of old targets", all(next(v for v in manifest["views"] if v["view"] == view)["reproduced_old_targets"] >= 500 for _, view in VIEWS))

    # the learned policy, evaluated once on the same target neurons under its own record
    learned = json.loads((LEARNED / "analysis-test.json").read_text())
    learned_verdicts = json.loads((LEARNED / "verdicts-test.json").read_text())
    claims("the record of the learned policy lists the expectations that were evaluated",
           len(json.loads((LEARNED / "record.json").read_text())["expectations"]) == len(learned_verdicts))
    m["LearnedExpTotal"], m["LearnedExpHeld"] = str(len(learned_verdicts)), str(sum(v["holds"] for v in learned_verdicts))

    def versus(view, part, prior, answer="all"):
        return learned[view][part][prior]["learned_vs_recal"][answer]

    for tag, view in VIEWS:
        m[f"Learned{tag}"] = ci(versus(view, "retention", "mlp_s0"))
        m[f"LearnedShare{tag}"] = ci(versus(view, "share", "mlp_s0"))
    m["LearnedProv"], m["LearnedProvShare"] = ci(versus("flywire", "retention", "provisional")), ci(versus("flywire", "share", "provisional"))
    m["LearnedProvYes"], m["LearnedProvNo"] = ci(versus("flywire", "retention", "provisional", "yes")), ci(versus("flywire", "retention", "provisional", "no"))
    weak = [versus(view, part, "mlp_s0")["value"] for _, view in VIEWS for part in ("retention", "share")]
    strong = [versus("flywire", part, "provisional") for part in ("retention", "share")]
    m["LearnedWeakLow"], m["LearnedWeakHigh"] = f2(min(weak)), f2(max(weak))
    m["LearnedSaveMaxPct"] = pct(1 - min(min(weak), min(s["value"] for s in strong)))
    claims("with the topology prior the learned policy is within 1% of recalibrated AG", all(abs(x - 1) <= 0.01 for x in weak))
    claims("with earlier labels it saves 1-5% of the checks of recalibrated AG", all(0.95 <= s["value"] <= 0.99 and s["high"] < 1 for s in strong))
    claims("nowhere does it need more than 1% more checks than recalibrated AG",
           max(versus(view, part, prior)["value"] for _, view in VIEWS for part in ("retention", "share")
               for prior in (("mlp_s0", "provisional") if view == "flywire" else ("mlp_s0",))) <= 1.01)
    m["LearnedProvWeight"] = ci(learned["flywire"]["retention"]["provisional"]["learned"]["all"])
    claims("with earlier labels the learned policy is the cheapest of the policies without propagation on retention questions",
           learned["flywire"]["retention"]["provisional"]["learned"]["all"]["value"]
           < min(ratio("flywire", "retention", f"provisional:{p}")["value"] for p in ("member", "nonmember", "direction", "ag", "adg", "recal", "single-recal", "ec2", "ig")))

    # the imitation order, evaluated once on the same target neurons under its own record
    imitation = json.loads((IMITATION / "analysis-test.json").read_text())
    imitation_verdicts = json.loads((IMITATION / "verdicts-test.json").read_text())
    claims("the record of the imitation order lists the expectations that were evaluated",
           len(json.loads((IMITATION / "record.json").read_text())["expectations"]) == len(imitation_verdicts))
    m["ImitExpTotal"], m["ImitExpHeld"] = str(len(imitation_verdicts)), str(sum(v["holds"] for v in imitation_verdicts))
    fit = json.loads((IMITATION / "fit.json").read_text())
    m["ImitFeatures"] = str(len(fit["features"]))
    m["ImitTrees"] = str(fit["model"]["max_iter"])
    m["ImitQuestionsMale"], m["ImitQuestionsFly"] = (f"{fit['models'][k]['questions']:,}" for k in ("malecns:mlp_s0", "flywire:mlp_s0"))
    aurocs = [x["training_auroc"] for x in fit["models"].values()]
    m["ImitAurocLow"], m["ImitAurocHigh"] = f2(min(aurocs)), f2(max(aurocs))

    def imitated(view, part, prior, against="recal", answer="all"):
        return imitation[view][part][prior]["imitation" if against == "weight" else f"imitation_vs_{against}"][answer]

    for tag, view in VIEWS:
        m[f"Imit{tag}"] = ci(imitated(view, "retention", "mlp_s0"))
        m[f"ImitShare{tag}"] = ci(imitated(view, "share", "mlp_s0"))
        m[f"ImitWeight{tag}"] = ci(imitated(view, "retention", "mlp_s0", "weight"))
    m["ImitProv"], m["ImitProvShare"] = ci(imitated("flywire", "retention", "provisional")), ci(imitated("flywire", "share", "provisional"))
    m["ImitProvDir"], m["ImitProvShareDir"] = (ci(imitated("flywire", part, "provisional", "direction")) for part in ("retention", "share"))
    m["ImitProvWeight"] = ci(imitated("flywire", "retention", "provisional", "weight"))
    weak_imitation = [imitated(view, part, "mlp_s0")["value"] for _, view in VIEWS for part in ("retention", "share")]
    strong_imitation = [imitated("flywire", part, "provisional") for part in ("retention", "share")]
    m["ImitWeakLow"], m["ImitWeakHigh"] = f2(min(weak_imitation)), f2(max(weak_imitation))
    m["ImitWeakSaveMaxPct"] = pct(1 - min(weak_imitation))
    m["ImitProvSaveLowPct"], m["ImitProvSaveHighPct"] = pct(1 - max(s["value"] for s in strong_imitation)), pct(1 - min(s["value"] for s in strong_imitation))
    m["TrainedWeakLow"], m["TrainedWeakHigh"] = f2(min(weak + weak_imitation)), f2(max(weak + weak_imitation))
    m["TrainedSaveMaxPct"] = pct(1 - min(min(weak + weak_imitation), min(s["value"] for s in strong + strong_imitation)))
    claims("with the topology prior the imitation order needs 0.96-1.01 of the checks of recalibrated AG", all(0.96 <= x <= 1.01 for x in weak_imitation))
    claims("with earlier labels it saves 5-10% of the checks of recalibrated AG", all(0.895 <= s["value"] <= 0.955 and s["high"] < 1 for s in strong_imitation))
    claims("with earlier labels it needs fewer checks than direction-fixed", all(imitated("flywire", part, "provisional", "direction")["high"] < 1 for part in ("retention", "share")))
    claims("with earlier labels the imitation order needs fewer checks than the tuned policy on retention questions",
           imitated("flywire", "retention", "provisional", "weight")["high"] < learned["flywire"]["retention"]["provisional"]["learned"]["all"]["value"])
    weak_weight = [imitated(view, "retention", "mlp_s0", "weight")["value"] for _, view in VIEWS]
    m["ImitWeightSaveLowPct"], m["ImitWeightSaveHighPct"] = pct(1 - max(weak_weight)), pct(1 - min(weak_weight))
    claims("with the topology prior the imitation order saves 3-7% of the checks of weight order on retention questions", all(0.93 <= x <= 0.97 for x in weak_weight))
    claims("with the topology prior no trained policy saves more than 4% of the checks of recalibrated AG", min(weak + weak_imitation) >= 0.96)

    # all recorded expected outcomes of the exploratory analyses, and the per-question view of the saving from earlier labels
    earlier_macros = {}
    for name in ("v35_numbers.tex", "r5_numbers.tex", "answers_numbers.tex"):
        earlier_macros.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", (PAPER / name).read_text()))
    recorded = [int(earlier_macros[k]) for k in ("RecordTwoChecks", "RecordThreeChecks", "RecordFourChecks", "RFiveExpTotal", "AnsExpTotal")]
    held = [int(earlier_macros[k]) for k in ("RecordTwoHeld", "RecordThreeHeld", "RecordFourHeld", "RFiveExpHeld", "AnsExpHeld")]
    third = {split: json.loads((NEWTYPES / f"analysis-{split}.json").read_text()) for split in ("heldout", "new")}
    third_verdicts = [v for split in ("heldout", "new") for v in json.loads((NEWTYPES / f"verdicts-{split}.json").read_text())]
    correspondence_verdicts = json.loads((ROOT / "outputs/correspondences/verdicts-test.json").read_text())
    labeling_verdicts = json.loads((ROOT / "outputs/labelings/verdicts-test.json").read_text())
    for group in (verdicts, learned_verdicts, imitation_verdicts, third_verdicts, correspondence_verdicts, labeling_verdicts):
        recorded.append(len(group))
        held.append(sum(v["holds"] for v in group))
    m["ExpAllTotal"], m["ExpAllHeld"], m["ExpAllFailed"] = str(sum(recorded)), str(sum(held)), str(sum(recorded) - sum(held))
    claims("most recorded expected outcomes hold and some do not", 0 < sum(recorded) - sum(held) < 0.1 * sum(recorded))
    claims("on the median question recalibrated AG with earlier labels needs as many checks as weight order; the geometric mean of the per-question ratio is below 0.8",
           float(earlier_macros["FlyProvRecalMedian"]) == 1.0 and float(earlier_macros["FlyProvRecalGeo"]) < 0.8)
    saved = ratio("flywire", "share", "provisional:recal", "no")
    m["RepShareProvRecalNoSavePct"] = pct(1 - saved["value"])
    claims("with earlier labels recalibrated AG saves a tenth or more of the checks on share questions answered no", saved["high"] < 0.9)

    # table of the main text
    def both(f):
        return f("malecns") + " & " + f("flywire")

    def short(d):
        return f"{d['value']:.2f} [{d['low']:.2f}, {d['high']:.2f}]"

    def third_ratio(split, v, arm):
        return third[split][v]["retention"]["ratio"][arm]["all"]

    def third_contrast(split, v, a, b, part="retention"):
        return third[split][v][part]["contrast"][f"{a}/{b}"]["all"]

    def earlier(a, b, part="retention"):
        """MaleCNS from the third protocol, FlyWire from the replication: the same target neurons of held-out types."""
        return short(third_contrast("heldout", "malecns", f"provisional:{a}", f"provisional:{b}", part)) + " & " + short(contrast("flywire", part, f"provisional:{a}", f"provisional:{b}"))

    half = lambda a, b: both(lambda v: short(third_contrast("heldout", v, f"cov0.5:{a}", f"cov0.5:{b}")))  # noqa: E731
    claims("the MaleCNS column of the earlier-label rows and the half-coverage rows describe the target neurons of the replication",
           all(third["heldout"][v]["targets"] == r[v]["targets"] and third["heldout"][v]["questions"] == r[v]["questions"] for _, v in VIEWS))
    full = [third_contrast("heldout", "malecns", "provisional:recal", "provisional:direction")["value"], contrast("flywire", "retention", "provisional:recal", "provisional:direction")["value"]]
    m["RepFullRecalDirMaxPct"] = pct(max(abs(x - 1) for x in full))
    claims("with earlier labels for every neuron recalibrated AG and direction-fixed differ by at most 3% on retention questions", max(abs(x - 1) for x in full) <= 0.035)
    rows = [
        "\\multicolumn{3}{@{}l}{\\emph{New neurons, retention questions}}\\\\",
        "\\quad propagation / recal.\\ AG & " + both(lambda v: short(contrast(v, "retention", "mlp_s0:prop", "mlp_s0:recal"))) + "\\\\",
        "\\quad propagation, no predictor$^a$ & " + both(lambda v: short(ratio(v, "retention", "none:prop"))) + "\\\\",
        "\\multicolumn{3}{@{}l}{\\emph{New cell types, retention questions}}\\\\",
        "\\quad propagation / recal.\\ AG & " + both(lambda v: short(third_contrast("new", v, "mlp_s0:prop", "mlp_s0:recal"))) + "\\\\",
        "\\quad propagation, no predictor$^a$ & " + both(lambda v: short(third_ratio("new", v, "none:prop"))) + "\\\\",
        "\\multicolumn{3}{@{}l}{\\emph{New neurons, earlier labels for every neuron}}\\\\",
        "\\quad AG / direction-fixed & " + earlier("ag", "direction") + "\\\\",
        "\\quad recal.\\ AG / direction-fixed & " + earlier("recal", "direction") + "\\\\",
        "\\quad same, share questions & " + earlier("recal", "direction", "share") + "\\\\",
        "\\multicolumn{3}{@{}l}{\\emph{New neurons, earlier labels for half of the neurons}}\\\\",
        "\\quad recal.\\ AG / direction-fixed & " + half("recal", "direction") + "\\\\",
    ]

    # appendix: every policy with the topology prior, and with the earlier-release prior by answer
    lines = []
    for key, label in POLICIES:
        cells = [short(ratio(view, part, f"mlp_s0:{key}")) for _, view in VIEWS for part in ("retention", "share")]
        lines.append(label + "".join(f" & {x}" for x in cells) + "\\\\")
    lines.append("Tuned policy$^\\dagger$" + "".join(f" & {short(learned[view][part]['mlp_s0']['learned']['all'])}" for _, view in VIEWS for part in ("retention", "share")) + "\\\\")
    lines.append("Imitation order$^\\dagger$" + "".join(f" & {short(imitated(view, part, 'mlp_s0', 'weight'))}" for _, view in VIEWS for part in ("retention", "share")) + "\\\\")
    lines.append("Propagation without a predictor" + "".join(f" & {short(ratio(view, part, 'none:prop'))}" for _, view in VIEWS for part in ("retention", "share")) + "\\\\")
    for level in (0.7, 0.8, 0.9):
        lines.append(f"AG, calibrated synthetic prior of AUROC {level}" + "".join(f" & {short(ratio(view, part, f'synthetic{level}:ag'))}" for _, view in VIEWS for part in ("retention", "share")) + "\\\\")
    lines.append("\\midrule\nClairvoyant bound" + "".join(f" & {short(ratio(view, part, 'clairvoyant'))}" for _, view in VIEWS for part in ("retention", "share")) + "\\\\")
    earlier = []
    for key, label in POLICIES:
        cells = [f2(ratio("flywire", part, f"provisional:{key}", a)["value"]) for part in ("retention", "share") for a in ("all", "yes", "no")]
        earlier.append(SHORT.get(label, label) + "".join(f" & {x}" for x in cells) + "\\\\")
    earlier.append("Tuned policy$^\\dagger$" + "".join(f" & {f2(learned['flywire'][part]['provisional']['learned'][a]['value'])}" for part in ("retention", "share") for a in ("all", "yes", "no")) + "\\\\")
    earlier.append("Imitation order$^\\dagger$" + "".join(f" & {f2(imitated('flywire', part, 'provisional', 'weight', a)['value'])}" for part in ("retention", "share") for a in ("all", "yes", "no")) + "\\\\")
    earlier.append("\\midrule\nClairvoyant bound" + "".join(f" & {f2(ratio('flywire', part, 'clairvoyant', a)['value'])}" for part in ("retention", "share") for a in ("all", "yes", "no")) + "\\\\")
    questions = []
    for tag, view in VIEWS:
        for part, name in (("retention", "retention"), ("share", "share")):
            p = r[view][part]
            a = p["answers"]
            questions.append(f"{'MaleCNS' if tag == 'Male' else 'FlyWire'}, {name} & {p['questions']:,} & {pct(p['share_uncertifiable'])} & {a['yes']:,} & {a['no']:,} & {p['weight_median']:.0f}\\\\")
    failed = [f"New target neurons & {v['text'].replace('%', chr(92) + '%')} & {v['value']:.4f}\\\\"
              for v in verdicts if not v["holds"]]
    failed += [f"Tuned policy & {v['text'].replace('%', chr(92) + '%')} & {v['value']:.4f}\\\\" for v in learned_verdicts if not v["holds"]]
    failed += [f"Imitation order & {v['text'].replace('%', chr(92) + '%')} & {v['value']:.4f}\\\\" for v in imitation_verdicts if not v["holds"]]
    by_level = [f"{'MaleCNS' if tag == 'Male' else 'FlyWire'}, $\\tau=\\tfrac{{{level[0]}}}{{{level[2]}}}$ & {levels[tag, level][0]:,} & {levels[tag, level][1]:,} & {levels[tag, level][2]:,}\\\\"
                for tag, _ in VIEWS for level in ("1/4", "1/2", "3/4")]

    if claims.failed:
        raise SystemExit("claims of the paper no longer hold:\n  " + "\n  ".join(claims.failed))
    taken = set()
    for name in ("v35_numbers.tex", "r5_numbers.tex", "answers_numbers.tex"):
        taken |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / name).read_text()))
    if set(m) & taken:
        raise SystemExit(f"macros defined twice: {sorted(set(m) & taken)}")
    text = ["% Generated by scripts/replication_assets.py; do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / "replication_numbers.tex").write_text("\n".join(text) + "\n")
    for name, body in (("replication", rows), ("replication_policies", lines), ("replication_earlier", earlier),
                       ("replication_questions", questions), ("replication_failed", failed), ("replication_share_levels", by_level)):
        (PAPER / f"tables/{name}.tex").write_text("\n".join(body) + "\n")
    print(f"{len(m)} macros, {len(failed)} failed expectations")


if __name__ == "__main__":
    main()
