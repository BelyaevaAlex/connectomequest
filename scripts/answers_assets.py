"""Numbers and tables of the paper from the analysis of questions of both answers (scripts/answers.py).

Writes ``paper/answers_numbers.tex`` (macros), ``paper/tables/answers.tex`` (costs by answer) and
``paper/tables/answers_failed.tex`` (recorded expectations that do not hold). Every sentence of the paper that rests
on these numbers states a direction or a bound; it is checked here, and the build stops if a result contradicts it.

Usage: python scripts/answers_assets.py
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
OUT = ROOT / "outputs/answers"
RULES = (("member", "Member-first"), ("nonmember", "Nonmember-first"), ("direction", "Direction-fixed"), ("ag", "AG"),
         ("adg", "Adaptive Dual Greedy"), ("recal", "AG + recal."))
READINGS = (("declared", "provisional"), ("lenient", "provisional"), ("lenient", "provisional_lenient"))


def pct(x):
    return f"{100 * x:.0f}"


def main():
    result = json.loads((OUT / "analysis-test.json").read_text())
    verdicts = json.loads((OUT / "verdicts-test.json").read_text())
    fly, male = result["flywire"], result["malecns"]
    claims = Claims()
    m = {}

    def contrast(part, prior, a, b, answer):
        return fly[part]["contrast"][f"{prior}:{a}/{prior}:{b}"][answer]

    # how often no is reached
    for tag, view in (("Fly", fly), ("Male", male)):
        m[f"AnsDeclNoShare{tag}"] = pct(view["declared"]["share_no_of_certifiable"])
        m[f"AnsLenNoShare{tag}"] = pct(view["lenient"]["share_no_of_certifiable"])
        m[f"AnsLenNoChecks{tag}"] = pct(view["lenient"]["share_no_checks"])
    claims("under the lenient crosswalk no is a fifth or more of certifiable questions and half or more of the checks",
           min(v["lenient"]["share_no_of_certifiable"] for v in (fly, male)) >= 0.2 and min(v["lenient"]["share_no_checks"] for v in (fly, male)) >= 0.5)

    # weight order on questions answered no (topology prior), and the yes share of the declared crosswalk
    adaptive = ("ag", "adg", "recal")

    def no_deviation(part):
        return max(abs(v[part]["ratio"][f"mlp_s0:{k}"]["no"]["value"] - 1) for v in (fly, male) for k in adaptive)

    m["AnsDeclTopoNoDevPct"] = pct(no_deviation("declared"))
    m["AnsLenTopoNoDevPct"] = f"{100 * no_deviation('lenient'):.1f}"
    claims("on no questions the adaptive policies with the topology prior stay within 2.5% of weight order, and within 0.5% under the lenient crosswalk",
           no_deviation("declared") <= 0.025 and no_deviation("lenient") <= 0.005)
    for tag, view in (("Fly", fly), ("Male", male)):
        m[f"AnsLenClairNoSave{tag}"] = pct(1 - view["lenient"]["ratio"]["clairvoyant"]["no"]["value"])
    claims("on no questions under the lenient crosswalk the clairvoyant bound saves under a fifth",
           min(v["lenient"]["ratio"]["clairvoyant"]["no"]["value"] for v in (fly, male)) >= 0.8)
    claims("under the lenient crosswalk no answers a quarter (FlyWire) and more than a third (MaleCNS) of the certifiable questions",
           0.22 <= fly["lenient"]["share_no_of_certifiable"] <= 0.27 and 0.34 <= male["lenient"]["share_no_of_certifiable"] <= 0.42)
    claims("on yes questions the clairvoyant bound saves more than half",
           max(v[part]["ratio"]["clairvoyant"]["yes"]["value"] for v in (fly, male) for part in ("declared", "lenient")) < 0.5)
    claims("nine in ten certifiable questions are answered yes under the declared crosswalk",
           max(v["declared"]["share_no_of_certifiable"] for v in (fly, male)) <= 0.1)
    lenient_uncertifiable = [v["lenient"]["answers"]["unresolved"] / sum(v["lenient"]["answers"].values()) for v in (fly, male)]
    claims("under the lenient crosswalk about a fifth of the questions cannot be certified", all(0.17 <= x <= 0.23 for x in lenient_uncertifiable))
    recal = [v[part]["ratio"][arm][answer]["value"] for v in (fly, male) for part in ("declared", "lenient")
             for arm in v[part]["ratio"] if arm.endswith(":recal") for answer in ("yes", "no")]
    claims("recalibrated AG never needs more checks than weight order on either answer", max(recal) <= 1.0)
    topo_ag = fly["declared"]["ratio"]["mlp_s0:ag"]["all"]
    claims("in FlyWire the interval of AG with the topology prior includes one", topo_ag["low"] < 1 < topo_ag["high"])

    # one-sided rules are cheap on one answer only (earlier-release prior, lenient crosswalk)
    lenient = fly["lenient"]["ratio"]
    m["AnsLenDirNo"] = f2(lenient["provisional:direction"]["no"]["value"])
    m["AnsLenMemberNo"] = f2(lenient["provisional:member"]["no"]["value"])
    claims("on no questions direction-fixed and member-first cost more than weight order",
           lenient["provisional:direction"]["no"]["low"] > 1 and lenient["provisional:member"]["no"]["low"] > 1)
    ag_yes, ag_no = contrast("lenient", "provisional", "ag", "direction", "yes"), contrast("lenient", "provisional", "ag", "direction", "no")
    m["AnsLenAGDirYes"], m["AnsLenAGDirNo"] = ci(ag_yes), ci(ag_no)
    claims("the hedge costs on yes and helps on no questions", ag_yes["low"] > 1 and ag_no["high"] < 1)
    decl_yes, decl_no = contrast("declared", "provisional", "ag", "direction", "yes"), contrast("declared", "provisional", "ag", "direction", "no")
    m["AnsDeclAGDirYes"], m["AnsDeclAGDirNo"] = ci(decl_yes), ci(decl_no)
    claims("under the declared crosswalk the hedge costs on yes questions and not on no questions", decl_yes["low"] > 1 and decl_no["value"] < 1)
    rd = contrast("lenient", "provisional", "recal", "direction", "no")
    rd_decl = contrast("declared", "provisional", "recal", "direction", "no")
    m["AnsLenRecalDirNo"], m["AnsDeclRecalDirNo"] = ci(rd), ci(rd_decl)
    claims("on no questions recalibrated AG beats direction-fixed under both crosswalks", rd["high"] < 1 and rd_decl["high"] < 1)

    # regret: recalibrated AG near the cheapest rule on both answers
    built = [fly[part]["regret"]["provisional"] for part in ("declared", "lenient")]
    recal_built = max(r["recal"]["worst"] for r in built)
    others_built = min(r["others_best_worst"] for r in built)
    m["AnsRecalRegretPct"] = pct(recal_built - 1)
    m["AnsOthersRegretPct"] = pct(others_built - 1)
    claims("with the earlier labels as built recalibrated AG is within 5% of the cheapest rule on both answers", recal_built <= 1.05)
    claims("and every other rule exceeds the cheapest by at least 8% on one answer", others_built >= 1.075)
    topo = [v[part]["regret"]["mlp_s0"]["recal"]["worst"] for v in (fly, male) for part in ("declared", "lenient")]
    m["AnsTopoRegretPct"] = f"{100 * (max(topo) - 1):.1f}"
    claims("with the topology prior recalibrated AG is within 1% of the cheapest rule on both answers", max(topo) <= 1.01)

    # earlier labels read in the lenient way: the hedge stops costing
    cons = fly["lenient"]["regret"]["provisional_lenient"]
    ra = contrast("lenient", "provisional_lenient", "recal", "ag", "all")
    agd = contrast("lenient", "provisional_lenient", "ag", "direction", "all")
    m["AnsConsAGDir"] = ci(agd)
    m["AnsConsDirYes"] = f2(fly["lenient"]["ratio"]["provisional_lenient:direction"]["yes"]["value"])
    m["AnsConsRegretPct"] = pct(max(cons["recal"]["worst"], cons["ag"]["worst"]) - 1)
    claims("read in the lenient way, AG matches recalibrated AG", 0.99 <= ra["low"] and ra["high"] <= 1.01)
    claims("and beats direction-fixed, which costs more than weight order on yes questions",
           agd["high"] < 1 and fly["lenient"]["ratio"]["provisional_lenient:direction"]["yes"]["low"] > 1)

    # expectations
    m["AnsExpTotal"], m["AnsExpHeld"] = str(len(verdicts)), str(sum(v["holds"] for v in verdicts))
    failed = [v for v in verdicts if not v["holds"]]
    rows = []
    for v in failed:
        rows.append(f"Both answers & {v['text']} & {v['value']:.3f}\\\\")

    # table: cells checked relative to weight order, by answer
    lines = []
    for key, label in RULES:
        cells = [fly[part]["ratio"][f"{prior}:{key}"][answer]["value"] for part, prior in READINGS for answer in ("yes", "no")]
        lines.append(label + "".join(f" & {x:.2f}" for x in cells) + "\\\\")
    clair = [fly[part]["ratio"]["clairvoyant"][answer]["value"] for part, _ in READINGS for answer in ("yes", "no")]
    lines.append("\\midrule\nClairvoyant bound" + "".join(f" & {x:.2f}" for x in clair) + "\\\\")
    shares = []
    for part, _ in READINGS:
        shares.append(f"\\multicolumn{{2}}{{c}}{{{pct(fly[part]['share_no_of_certifiable'])} / {pct(fly[part]['share_no_checks'])}}}")
    lines.append("\\emph{No} questions, \\% of questions / checks & " + " & ".join(shares) + "\\\\")

    if claims.failed:
        raise SystemExit("claims of the paper no longer hold:\n  " + "\n  ".join(claims.failed))
    taken = set()
    for name in ("v35_numbers.tex", "r5_numbers.tex"):
        taken |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / name).read_text()))
    if set(m) & taken:
        raise SystemExit(f"macros defined twice: {sorted(set(m) & taken)}")
    text = ["% Generated by scripts/answers_assets.py; do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / "answers_numbers.tex").write_text("\n".join(text) + "\n")
    (PAPER / "tables/answers.tex").write_text("\n".join(lines) + "\n")
    (PAPER / "tables/answers_failed.tex").write_text("\n".join(rows) + "\n")
    print(f"{len(m)} macros, {len(failed)} failed expectations")


if __name__ == "__main__":
    main()
