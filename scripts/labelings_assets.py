"""Numbers and tables of the paper from the answers under other labels (scripts/labelings.py).

Writes ``paper/labelings_numbers.tex`` (macros) and five tables: ``labelings_rules.tex`` (the two rules that complete
the declared correspondences), ``labelings_earlier.tex`` (answers under the earlier release), ``labelings_circuits.tex``
(questions by circuit of the target), ``labelings_residual.tex`` and ``labelings_main.tex`` (first types of the list
that remains after both rules, for the appendix and the main text). Every sentence of the paper that rests on
these numbers states a direction or a bound; it is checked here, and the build stops if a result contradicts it.

Usage: python scripts/labelings_assets.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r5_paper_assets import Claims  # noqa: E402

PAPER = ROOT / "paper"
OUT = ROOT / "outputs/labelings"
VIEWS = (("Male", "malecns", "MaleCNS"), ("Fly", "flywire", "FlyWire"))
CIRCUITS = (("mushroom body", "Mushroom body"), ("central complex", "Central complex"), ("other", "Other"))
WORDS = ("no", "one", "two", "three", "four", "five")


def count(n):
    return f"{n:,}".replace(",", "{,}")


def tex(name):
    return re.sub(r"([_&%#])", r"\\\1", name)


def pct(d, digits=1):
    return f"{100 * d['value']:.{digits}f} [{100 * d['low']:.{digits}f}, {100 * d['high']:.{digits}f}]"


def span(values, digits=0):
    return f"{100 * min(values):.{digits}f}", f"{100 * max(values):.{digits}f}"


def main():
    result = json.loads((OUT / "analysis-test.json").read_text())
    types = json.loads((OUT / "types.json").read_text())
    verdicts = json.loads((OUT / "verdicts-test.json").read_text())
    meta = {view: json.loads((OUT / f"{view}-test.json").read_text()) for _, view, _ in VIEWS}
    claims = Claims()
    m = {}
    views = [result[view] for _, view, _ in VIEWS]

    # ------------------------------------------------------------------ the two rules
    agreement = [types[view][rule]["agreement"] for _, view, _ in VIEWS for rule in ("suffix", "linked")]
    m["LabRuleAgreeLowPct"], m["LabRuleAgreeHighPct"] = span(agreement, 1)
    m["LabRuleAgreeFloorPct"] = str(math.floor(100 * min(agreement)))
    for tag, view, _ in VIEWS:
        m[f"LabSuffixTested{tag}"] = count(types[view]["suffix"]["tested"])
        m[f"LabLinkedTested{tag}"] = count(types[view]["linked"]["tested"])
        m[f"LabRuleTypes{tag}"] = count(meta[view]["covered"]["both"])
        m[f"LabSpecificTypes{tag}"] = count(meta[view]["release_specific_types"])
        m[f"LabBothCert{tag}"] = count(result[view]["both"]["certified_questions"])
    m["LabSuffixLabelsFly"] = count(types["flywire"]["suffix"]["labels"])
    claims("the rules never disagree on a release-specific type", all(types[view]["both"]["conflicts"] == 0 for _, view, _ in VIEWS))
    claims("in MaleCNS most of what the rules certify comes from the suffix rule",
           result["malecns"]["suffix"]["certified_questions"] > 0.5 * result["malecns"]["both"]["certified_questions"])
    claims("in FlyWire the suffix rule certifies next to nothing and labels few types",
           result["flywire"]["suffix"]["certified"]["value"] < 0.005 and types["flywire"]["suffix"]["labels"] < 0.05 * types["flywire"]["suffix"]["tested"])
    tested = sum(types[view][rule]["tested"] for _, view, _ in VIEWS for rule in ("suffix", "linked"))
    m["LabRuleTested"] = count(tested)
    both = [r["both"]["certified"]["value"] for r in views]
    m["LabBothCertLowPct"], m["LabBothCertHighPct"] = span(both, 1)
    labelled = [r["both"]["unresolved_weight_labelled"]["value"] for r in views]
    m["LabBothWeightLowPct"], m["LabBothWeightHighPct"] = span(labelled)
    reversed_ = sum(r[check]["reversed_questions"] for r in views for check in ("suffix-check", "linked-check"))
    certified = [round(r["questions"] * (1 - r["uncertifiable"]["value"])) for r in views]
    same = [r[check]["same_answer"]["value"] for r in views for check in ("suffix-check", "linked-check")]
    m["LabCheckReversedWord"], m["LabCheckCertified"] = WORDS[reversed_], count(sum(certified))
    m["LabCheckSameMinPct"] = f"{math.floor(1000 * min(same)) / 10:.1f}"
    claims("each rule agrees with the declared correspondence for at least 99% of the types on which it is tested", min(agreement) >= 0.99)
    claims("the rules are tested on several thousand declared types", tested > 4000)
    claims("used in place of the declarations, the rules reverse two certified answers", reversed_ == 2)
    claims("and leave at least 99.8% of the answers unchanged", min(same) >= 0.998)
    claims("both rules together certify 2-3% of the uncertifiable questions", all(0.02 <= x <= 0.03 for x in both))
    claims("the suffix rule never reverses an answer", all(r["suffix-check"]["reversed_questions"] == 0 for r in views))
    claims("the rules label under a tenth of the release-specific types among the input cells",
           all(meta[view]["covered"]["both"] < 0.1 * meta[view]["release_specific_types"] for _, view, _ in VIEWS))
    rules = []
    for _, view, title in VIEWS:
        for rule, name in (("suffix", "suffix"), ("linked", "linked"), ("both", "both")):
            r = result[view][rule]
            if rule == "both":
                test, check = "-- & -- & " + count(types[view]["both"]["labels"]), "-- & -- & --"
            else:
                t, c = types[view][rule], result[view][f"{rule}-check"]
                test = f"{count(t['tested'])} & {100 * t['agreement']:.1f} & {count(t['labels'])}"
                check = f"{100 * c['same_answer']['value']:.2f} & {c['reversed_questions']} & {c['lost_questions']}"
            rules.append(f"{title if rule == 'suffix' else ''} & {name} & {test} & {pct(r['certified'])} & {100 * r['yes']['value']:.1f} & "
                         f"{100 * r['no']['value']:.1f} & {check}\\\\")

    # ------------------------------------------------------------------ the earlier release
    earlier = [r["earlier"] for r in views]
    transitions = [e["transitions"] for e in earlier]
    was_certified = [sum(v for k, v in t.items() if not k.startswith("uncertifiable")) for t in transitions]
    lost = sum(e["lost_questions"] for e in earlier)
    reversed_earlier = sum(e["reversed_questions"] for e in earlier)
    m["LabEarlierCertified"], m["LabEarlierLost"], m["LabEarlierReversedWord"] = count(sum(was_certified)), count(lost), WORDS[reversed_earlier]
    m["LabKeptFloorPct"] = f"{math.floor(1000 * min(e['kept']['value'] for e in earlier)) / 10:.1f}"
    claims("the one reversed answer is a FlyWire question that goes from no to yes",
           result["flywire"]["earlier"]["transitions"]["no to yes"] == 1 and result["malecns"]["earlier"]["reversed_questions"] == 0)
    claims("both reversals of the test of the rules come from the linked rule in MaleCNS",
           result["malecns"]["linked-check"]["reversed_questions"] == 2)
    for (tag, view, _), e in zip(VIEWS, earlier):
        m[f"LabKeptPct{tag}"] = f"{100 * e['kept']['value']:.2f}".rstrip("0").rstrip(".") if e["kept"]["value"] < 0.9995 else f"{100 * e['kept']['value']:.2f}"
        m[f"LabValidPct{tag}"] = f"{100 * e['certificate_valid']['value']:.1f}"
        m[f"LabChangedPct{tag}"] = f"{100 * e['questions_with_other_outcome']['value']:.0f}"
        m[f"LabGainedPct{tag}"] = f"{100 * e['gained']['value']:.1f}" if e["gained"]["value"] < 0.01 else f"{100 * e['gained']['value']:.0f}"
        m[f"LabBeforePct{tag}"] = f"{100 * e['certified_before']['value']:.0f}"
        m[f"LabCertCells{tag}"] = WORDS[int(e["certificate_cells"])]
    fly = result["flywire"]["earlier"]
    m["LabGainedYesPctFly"] = f"{100 * fly['gained_yes']['value']:.0f}"
    m["LabGainedFly"] = count(fly["gained_questions"])
    m["LabChangedCellsPctFly"] = f"{100 * fly['cells_with_other_outcome']['value']:.0f}"
    claims("at least 99.8% of the answers certified under the earlier release are certified again with the same answer",
           all(e["kept"]["low"] >= 0.997 and e["kept"]["value"] >= 0.998 for e in earlier))
    claims("one answer is reversed", reversed_earlier == 1)
    claims("at least 99% of the smallest certificates stay valid cell by cell", all(e["certificate_valid"]["value"] >= 0.99 for e in earlier))
    claims("a smallest certificate has a median of two or three cells", sorted(e["certificate_cells"] for e in earlier) == [2.0, 3.0])
    claims("in FlyWire three quarters of the questions contain an input cell whose outcome changed", 0.72 <= fly["questions_with_other_outcome"]["value"] <= 0.82)
    claims("the later FlyWire release settles about a sixth of the questions that the earlier one left open", 0.14 <= fly["gained"]["value"] <= 0.18)
    claims("about four fifths of them with yes", 0.77 <= fly["gained_yes"]["value"] <= 0.87)
    claims("the two MaleCNS releases differ much less", result["malecns"]["earlier"]["gained"]["value"] < 0.01
           and result["malecns"]["earlier"]["questions_with_other_outcome"]["value"] < 0.25)
    row = lambda name, cells: f"{name} & {cells[0]} & {cells[1]}\\\\"
    table_earlier = [
        row("Questions certified under the earlier release (\\%)", [f"{100 * e['certified']['value']:.1f}" for e in earlier]),
        row("\\quad certified with the same answer under the later release (\\%)", [pct(e["kept"], 2) for e in earlier]),
        row("\\quad certificate lost (questions)", [count(e["lost_questions"]) for e in earlier]),
        row("\\quad answer reversed (questions)", [count(e["reversed_questions"]) for e in earlier]),
        row("\\quad smallest certificate valid cell by cell (\\%)", [pct(e["certificate_valid"], 2) for e in earlier]),
        row("\\quad cells of a smallest certificate (median)", [f"{e['certificate_cells']:.0f}" for e in earlier]),
        row("Questions left open by the earlier release (\\%)", [f"{100 * e['uncertifiable']['value']:.1f}" for e in earlier]),
        row("\\quad certified under the later release (\\%)", [pct(e["gained"]) for e in earlier]),
        row("\\quad\\quad of these with the answer \\emph{yes} (\\%)", [pct(e["gained_yes"], 0) for e in earlier]),
        row("Answers of the later release already certified under the earlier (\\%)", [pct(e["certified_before"]) for e in earlier]),
        row("Questions with an input cell of another outcome (\\%)", [pct(e["questions_with_other_outcome"]) for e in earlier]),
        row("Input cells with another outcome (\\%)", [pct(e["cells_with_other_outcome"]) for e in earlier]),
    ]

    # ------------------------------------------------------------------ the list that remains after both rules
    residual = [r["residual"] for r in views]
    hundred = [r["first"]["100"]["value"] for r in residual]
    half = [r["types_for"]["0.50"] for r in residual]
    m["LabResHundredLowPct"], m["LabResHundredHighPct"] = span(hundred)
    m["LabResHalfLow"], m["LabResHalfHigh"] = str(min(half)), str(max(half))
    m["LabResYesLowPct"], m["LabResYesHighPct"] = span([r["one_type_yes"]["value"] for r in residual])
    m["LabResNoLowPct"], m["LabResNoHighPct"] = span([r["one_type_no"]["value"] for r in residual])
    for (tag, _, _), r in zip(VIEWS, residual):
        m[f"LabResTypes{tag}"] = count(r["types"])
        m[f"LabResOpenPct{tag}"] = f"{100 * r['open']['value']:.0f}"
    claims("after both rules the first hundred types of the list certify 11-15% of the questions it can certify", all(0.11 <= x <= 0.15 for x in hundred))
    claims("and half of them need more than five hundred types", min(half) > 500)
    claims("after both rules more than 2,800 release-specific types remain in the open questions", min(r["types"] for r in residual) > 2800)
    claims("no type that a rule labels is left in the lists",
           all(not set(row["type"] for row in r["top"]) & set(meta[view]["covered_types"]["both"]) for (_, view, _), r in zip(VIEWS, residual)))
    top, short = [], []
    for i in range(10):
        cells = [str(i + 1)]
        for r in residual:
            t = r["top"][i]
            cells += [tex(t["type"]), str(t["questions"]), str(t["no"]), str(t["yes"])]
        top.append(" & ".join(cells) + "\\\\")
    for i in range(5):
        cells = []
        for r in residual:
            t = r["top"][i]
            cells += [tex(t["type"]), str(t["no"]), str(t["yes"])]
        short.append(" & ".join(cells) + "\\\\")

    # ------------------------------------------------------------------ circuits
    circuits = []
    share = lambda circuit: [r["circuit"][circuit]["uncertifiable"] for r in views]
    for key, tag in (("mushroom body", "Mb"), ("central complex", "Cx"), ("other", "Other")):
        m[f"Lab{tag}LowPct"], m[f"Lab{tag}HighPct"] = span([x["value"] for x in share(key)])
        m[f"Lab{tag}Types"] = count(views[0]["circuit"][key]["target_types"])
    claims("12-19% of the questions about mushroom-body and central-complex targets are uncertifiable",
           all(0.115 <= x["value"] <= 0.19 for key in ("mushroom body", "central complex") for x in share(key)))
    claims("against 37-47% for other targets", all(0.365 <= x["value"] <= 0.47 for x in share("other")))
    claims("the intervals of both circuits lie below the interval of the other targets in each connectome",
           all(r["circuit"][key]["uncertifiable"]["high"] < r["circuit"]["other"]["uncertifiable"]["low"]
               for r in views for key in ("mushroom body", "central complex")))
    claims("the same target types are counted in both connectomes",
           all(views[0]["circuit"][key]["target_types"] == views[1]["circuit"][key]["target_types"] for key, _ in CIRCUITS))
    ten = [r["circuit"]["mushroom body"]["list"]["first"]["10"]["value"] for r in views]
    m["LabMbTenLowPct"], m["LabMbTenHighPct"] = span(ten)
    m["LabMbOpenLow"], m["LabMbOpenHigh"] = (str(f(r["circuit"]["mushroom body"]["list"]["uncertifiable_questions"] for r in views)) for f in (min, max))
    claims("for mushroom-body targets ten types certify at least three quarters of the questions that their list can certify", min(ten) >= 0.75)
    claims("fewer than fifty mushroom-body questions are uncertifiable in each connectome",
           max(r["circuit"]["mushroom body"]["list"]["uncertifiable_questions"] for r in views) < 50)
    for key, name in CIRCUITS:
        for (_, view, title), r in zip(VIEWS, views):
            c = r["circuit"][key]
            circuits.append(f"{name if view == 'malecns' else ''} & {title} & {count(c['target_types'])} & {count(c['questions'])} & {pct(c['uncertifiable'])} & "
                            f"{count(c['list']['uncertifiable_questions'])} & {count(c['list']['reachable_questions'])} & {count(c['list']['types'])} & "
                            f"{c['list']['types_for']['0.50']} & {100 * c['list']['first']['10']['value']:.0f}\\\\")

    held = sum(v["holds"] for v in verdicts)
    m["LabExpTotal"], m["LabExpHeld"] = str(len(verdicts)), str(held)
    claims("all recorded expectations of this analysis hold", held == len(verdicts))

    if claims.failed:
        raise SystemExit("claims of the paper no longer hold:\n  " + "\n  ".join(claims.failed))
    taken = set()
    for name in ("v35_numbers.tex", "r5_numbers.tex", "answers_numbers.tex", "correspondences_numbers.tex"):
        taken |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / name).read_text()))
    if set(m) & taken:
        raise SystemExit(f"macros defined twice: {sorted(set(m) & taken)}")
    text = ["% Generated by scripts/labelings_assets.py; do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / "labelings_numbers.tex").write_text("\n".join(text) + "\n")
    for name, lines in (("rules", rules), ("earlier", table_earlier), ("circuits", circuits), ("residual", top), ("main", short)):
        (PAPER / f"tables/labelings_{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(m)} macros, {len(verdicts) - held} failed expectations")


if __name__ == "__main__":
    main()
