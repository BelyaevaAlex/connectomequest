"""Numbers and tables of the paper from the analysis of missing type correspondences (scripts/correspondences.py)
and from the summary of both readings of a type (scripts/correspondences_readings.py).

Writes ``paper/correspondences_numbers.tex`` (macros), ``paper/tables/correspondences.tex`` (summary by connectome)
and ``paper/tables/correspondences_top.tex`` and ``correspondences_main.tex`` (first types of the priority lists,
for the appendix and the main text). Every sentence of the paper that
rests on these numbers states a direction or a bound; it is checked here, and the build stops if a result
contradicts it.

Usage: python scripts/correspondences_assets.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from r5_paper_assets import Claims  # noqa: E402

PAPER = ROOT / "paper"
OUT = ROOT / "outputs/correspondences"
VIEWS = (("Male", "malecns"), ("Fly", "flywire"))
# superclasses of the two releases under common names
CLASSES = (("Central brain, intrinsic", dict(malecns="cb_intrinsic", flywire="central")),
           ("Optic lobe, intrinsic", dict(malecns="ol_intrinsic", flywire="optic")),
           ("Visual projection", dict(malecns="visual_projection", flywire="visual_projection")),
           ("Ascending", dict(malecns="ascending_neuron", flywire="ascending")))
SHORT = {"cb_intrinsic": "central", "central": "central", "ol_intrinsic": "optic", "optic": "optic", "visual_projection": "vis.\\ proj.",
         "ascending_neuron": "ascending", "ascending": "ascending", "visual_centrifugal": "vis.\\ centrif.",
         "descending_neuron": "descending", "descending": "descending"}


def pct(x):
    return f"{100 * x:.0f}"


def pct1(d):
    return f"{100 * d['value']:.1f} [{100 * d['low']:.1f}, {100 * d['high']:.1f}]"


def count(n):
    return f"{n:,}".replace(",", "{,}")


def tex(name):
    return re.sub(r"([_&%#])", r"\\\1", name)


def main():
    result = json.loads((OUT / "analysis-test.json").read_text())
    verdicts = json.loads((OUT / "verdicts-test.json").read_text())
    development = json.loads((OUT / "analysis-development.json").read_text())
    claims = Claims()
    m = {}
    views = [result[view] for _, view in VIEWS]

    for tag, view in VIEWS:
        r = result[view]
        m[f"CorrTypes{tag}"] = count(r["types"])
        m[f"CorrReach{tag}Pct"] = pct(r["reachable"]["value"])
        m[f"CorrAlone{tag}Pct"] = pct(r["certified_alone"]["value"])
        m[f"CorrNeeded{tag}"] = f"{r['types_needed']['median']:.0f}"
        m[f"CorrHalf{tag}"] = count(r["list"]["types_for"]["0.50"])
        m[f"CorrHundred{tag}Pct"] = pct(r["list"]["first"]["100"]["value"])
        m[f"CorrTransferWhole{tag}Pct"] = pct(r["transfer"]["whole"]["value"])
        m[f"CorrDevList{tag}"] = count(development[view]["list"]["length"])
        for (label, names), key in zip(CLASSES, ("Central", "Optic", "Projection", "Ascending")):
            m[f"Corr{key}{tag}Pct"] = pct(r["superclass"][names[view]]["weight"]["value"])
    alone = [r["certified_alone"]["value"] for r in views]
    m["CorrAloneLowPct"], m["CorrAloneHighPct"] = pct(min(alone)), pct(max(alone))
    half = [r["list"]["types_for"]["0.50"] for r in views]
    m["CorrHalfLow"], m["CorrHalfHigh"] = count(min(half)), count(max(half))
    hundred = [r["list"]["first"]["100"]["value"] for r in views]
    m["CorrHundredLowPct"], m["CorrHundredHighPct"] = pct(min(hundred)), pct(max(hundred))
    needed = [r["types_needed"]["median"] for r in views]
    m["CorrNeededLow"], m["CorrNeededHigh"] = f"{min(needed):.0f}", f"{max(needed):.0f}"
    types = [r["types"] for r in views]
    m["CorrTypesLow"], m["CorrTypesHigh"] = count(min(types)), count(max(types))
    central = [r["superclass"][CLASSES[0][1][view]]["weight"]["value"] for _, view in VIEWS for r in (result[view],)]
    optic = [r["superclass"][CLASSES[1][1][view]]["weight"]["value"] for _, view in VIEWS for r in (result[view],)]
    whole = [r["transfer"]["whole"]["value"] for r in views]
    m["CorrTransferWholeLowPct"], m["CorrTransferWholeHighPct"] = pct(min(whole)), pct(max(whole))
    m["CorrExpTotal"], m["CorrExpHeld"] = str(len(verdicts)), str(sum(v["holds"] for v in verdicts))

    claims("more than three thousand release-specific types occur among the input cells of uncertifiable questions", min(types) > 3000)
    claims("for 6-7% of the uncertifiable questions the declaration of one release-specific type suffices", all(0.055 <= x <= 0.075 for x in alone))
    claims("one type decides under 2% of them whatever is declared", max(r["decided_alone"]["value"] for r in views) < 0.02)
    claims("half of the questions that all types certify need five or six types or more", all(5 <= x <= 6 for x in needed))
    claims("half of them need more than 600 types of the priority list", min(half) > 600)
    claims("the first 100 types of the list certify at most 15% of them", max(hundred) <= 0.15)
    claims("about half of the blocking weight belongs to intrinsic central-brain types", all(0.45 <= x <= 0.55 for x in central))
    claims("and about a quarter to intrinsic optic-lobe types", all(0.22 <= x <= 0.28 for x in optic))
    claims("the four named superclasses carry at least 95% of the blocking weight",
           all(sum(result[view]["superclass"][names[view]]["weight"]["value"] for _, names in CLASSES) >= 0.95 for _, view in VIEWS))
    claims("the development list certifies 60-85% of the held-out questions that all types certify, its first 100 types under 8%",
           all(0.60 <= x <= 0.85 for x in whole) and max(r["transfer"]["first"]["100"]["value"] for r in views) < 0.08)
    claims("the list built from the held-out questions is never worse on them than the development list",
           all(r["list"]["first"][k]["value"] >= r["transfer"]["first"][k]["value"] for r in views for k in r["list"]["first"]))
    claims("all recorded expectations of this analysis hold", all(v["holds"] for v in verdicts))

    def both(cell):
        return " & ".join(cell(result[view]) for _, view in VIEWS)

    rows = [
        "Uncertifiable questions & " + both(lambda r: count(round(r["uncertifiable"]["value"] * r["questions"]))) + "\\\\",
        "\\quad certified if all release-specific types are nonmembers (\\%) & " + both(lambda r: pct1(r["reachable"])) + "\\\\",
        "Release-specific types among their input cells & " + both(lambda r: count(r["types"])) + "\\\\",
        "Certified by declaring one type a nonmember (\\%) & " + both(lambda r: pct1(r["certified_alone"])) + "\\\\",
        "Certified by one type whatever is declared (\\%) & " + both(lambda r: pct1(r["decided_alone"])) + "\\\\",
        "Types needed, median (quartiles)$^a$ & " + both(lambda r: f"{r['types_needed']['median']:.0f} ({r['types_needed']['q25']:.0f}--{r['types_needed']['q75']:.0f})") + "\\\\",
        "\\midrule\n\\multicolumn{3}{@{}l}{\\emph{Priority list built from these questions: questions certified$^a$ (\\%)}}\\\\",
    ]
    for k in ("100", "1000"):
        rows.append(f"\\quad first {count(int(k))} types & " + both(lambda r: pct1(r["list"]["first"][k])) + "\\\\")
    rows.append("\\quad types needed for a quarter, half, three quarters & " + both(
        lambda r: ", ".join(count(r["list"]["types_for"][s]) for s in ("0.25", "0.50", "0.75"))) + "\\\\")
    rows.append("\\midrule\n\\multicolumn{3}{@{}l}{\\emph{Priority list built from development questions: held-out questions certified$^a$ (\\%)}}\\\\")
    rows.append("\\quad first 100 types & " + both(lambda r: pct1(r["transfer"]["first"]["100"])) + "\\\\")
    rows.append("\\quad whole list & " + both(lambda r: pct1(r["transfer"]["whole"])) + "\\\\")
    rows.append("\\midrule\n\\multicolumn{3}{@{}l}{\\emph{Superclass of the release: share of the blocking weight$^b$ (\\%); types}}\\\\")
    for label, names in CLASSES:
        rows.append(f"\\quad {label} & " + " & ".join(
            f"{pct1(result[view]['superclass'][names[view]]['weight'])}; {count(result[view]['superclass'][names[view]]['types'])}" for _, view in VIEWS) + "\\\\")
    named = {view: {names[view] for _, names in CLASSES} for _, view in VIEWS}
    rows.append("\\quad other superclasses & " + " & ".join(
        f"{100 * sum(c['weight']['value'] for k, c in result[view]['superclass'].items() if k not in named[view]):.1f}; "
        f"{count(sum(c['types'] for k, c in result[view]['superclass'].items() if k not in named[view]))}" for _, view in VIEWS) + "\\\\")

    # types whose name extends the name of a hemibrain type by a suffix: the newer release splits a hemibrain type
    vocabulary = set(json.loads((ROOT / "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json").read_text())["vocabulary"][1:])

    def splits(name):
        return any(name[:k] in vocabulary and re.fullmatch(r"[a-z_-][a-z0-9_'-]*", name[k:]) for k in range(2, len(name)))

    readings = json.loads((OUT / "readings.json").read_text())["test"]
    tables = {}
    for tag, view in VIEWS:
        with (OUT / f"types-{view}-test.csv").open(newline="") as stream:
            listed = [r for r in csv.DictReader(stream) if r["rank"]]
        tables[view] = listed[:10]
        m[f"CorrSplit{tag}"] = str(sum(splits(r["type"]) for r in listed[:100]))
        for r, both in zip(tables[view], readings[view]["first"]):  # the summary of both readings describes the same list
            if (r["type"], int(r["certified_alone"])) != (both["type"], both["outside"]):
                raise SystemExit(f"{view}: the summary of both readings does not match the ranked list")
            r["inside"] = str(both["inside"])
        if m[f"CorrSplit{tag}"] != str(len(readings[view]["split_types"])):
            raise SystemExit(f"{view}: the summary of both readings finds other split types")
    inside = [readings[view]["one_type_inside"] for _, view in VIEWS]
    m["CorrInsideLowPct"], m["CorrInsideHighPct"] = pct(min(inside)), pct(max(inside))
    claims("one type read as inside the partner set would certify yes for 45-60% of the uncertifiable questions", all(0.45 <= x <= 0.60 for x in inside))
    claims("read as outside, one type certifies no for the share that the recorded analysis reports",
           all(abs(readings[view]["one_type_outside"] - result[view]["certified_alone"]["value"]) < 1e-12 for _, view in VIEWS))
    lc = [r for r in readings["malecns"]["split_types"] if r["parent"] == "LC10" and r["rank"] <= 4]
    contains = [r["parent_in_partner_set"] / r["questions"] for r in lc]
    m["CorrLCInLowPct"], m["CorrLCInHighPct"] = pct(min(contains)), pct(max(contains))
    m["CorrLCYesLow"], m["CorrLCYesHigh"] = str(min(r["inside_where_parent_in"] for r in lc)), str(max(r["inside_where_parent_in"] for r in lc))
    m["CorrLCOutside"], m["CorrLCOutsideWrong"] = str(sum(r["outside"] for r in lc)), str(sum(r["outside_where_parent_in"] for r in lc))
    claims("LC10 belongs to the partner set of two thirds or more of the questions that each of its four subtypes blocks", len(lc) == 4 and min(contains) >= 0.65)
    claims("most questions that the list credits to the LC10 subtypes have LC10 in their partner set",
           int(m["CorrLCOutsideWrong"]) > 0.8 * int(m["CorrLCOutside"]))
    claims("the first four types of the MaleCNS list are the four subtypes of LC10, a hemibrain type",
           sorted(r["type"] for r in tables["malecns"][:4]) == ["LC10a", "LC10c-1", "LC10c-2", "LC10d"] and "LC10" in vocabulary)
    claims("names that extend a hemibrain type name are a minority of the first 100 types of either list",
           max(int(m["CorrSplitMale"]), int(m["CorrSplitFly"])) <= 20)
    top = []
    for i in range(10):
        cells = [str(i + 1)]
        for _, view in VIEWS:
            r = tables[view][i]
            if int(r["rank"]) != i + 1:
                raise SystemExit("type list is not sorted by rank")
            cells += [f"{tex(r['type'])} ({SHORT.get(r['superclass'], tex(r['superclass']))})", r["uncertifiable_questions_with_type"],
                      r["certified_alone"], r["inside"], r["certified_by_list_so_far"]]
        top.append(" & ".join(cells) + "\\\\")

    short = []
    for i in range(5):
        cells = []
        for _, view in VIEWS:
            r = tables[view][i]
            cells += [tex(r["type"]), r["certified_alone"], r["inside"]]
        short.append(" & ".join(cells) + "\\\\")
    claims("the whole development list certifies less than the list built from the held-out questions, which certifies all of them",
           all(r["transfer"]["whole"]["value"] < r["list"]["whole"]["value"] == 1.0 for r in views))

    if claims.failed:
        raise SystemExit("claims of the paper no longer hold:\n  " + "\n  ".join(claims.failed))
    taken = set()
    for name in ("v35_numbers.tex", "r5_numbers.tex", "answers_numbers.tex"):
        taken |= set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / name).read_text()))
    if set(m) & taken:
        raise SystemExit(f"macros defined twice: {sorted(set(m) & taken)}")
    text = ["% Generated by scripts/correspondences_assets.py; do not edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / "correspondences_numbers.tex").write_text("\n".join(text) + "\n")
    (PAPER / "tables/correspondences.tex").write_text("\n".join(rows) + "\n")
    (PAPER / "tables/correspondences_top.tex").write_text("\n".join(top) + "\n")
    (PAPER / "tables/correspondences_main.tex").write_text("\n".join(short) + "\n")
    print(f"{len(m)} macros, {sum(not v['holds'] for v in verdicts)} failed expectations")


if __name__ == "__main__":
    main()
