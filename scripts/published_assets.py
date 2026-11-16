#!/usr/bin/env python3
"""Numbers and tables of the paper from the published statements (scripts/published.py).

Writes ``paper/published_numbers.tex`` (macros), ``paper/tables/published_main.tex`` (the table of the main text: one
row per statement with the bounds and the answers at the published share) and ``paper/tables/published.tex`` (the
table of the appendix, which continues it with the answers at half the share and the checks at the published share). Every statement of
the file is a row of both tables. The sentences of the paper that rest on these numbers state an order or a direction; they are
checked here, and the build stops if a result contradicts one.

The shares of the statements in the hemibrain without the threshold of the benchmark come from
``outputs/published/source.json`` (scripts/published_source.py).

Usage: python scripts/published_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import published as p  # noqa: E402
from r5_paper_assets import Claims  # noqa: E402

PAPER = ROOT / "paper"
WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
POLICIES = ("weight", "direction", "recal", "ag")
# row labels that fit one column; the other rows use the labels of the statements file
SHORT = {"fb4r-mbon09-mbon21": r"MBON09+21 $\to$ FB4R", "kcgs2-kc": r"all KCs $\to$ KC$\gamma$s2"}


def count(n):
    return f"{n:,}".replace(",", "{,}")


def word(n):
    return WORDS[n] if n < len(WORDS) else str(n)


def share(value):
    """A share in percent: one decimal below ten, none from ten on; a dash when no neuron was asked."""
    if value is None:
        return "---"
    value *= 100
    return "0" if value == 0 else f"{value:.1f}" if value < 9.95 else f"{value:.0f}"


def answers(level):
    return f"{level['yes']}/{level['no']}/{level['uncertifiable']}"


def tables(statements, result):
    main, full = [], []
    for entry in result["statements"]:
        s = statements[entry["id"]]
        published = f"{float(s['percent']):g}"
        hemibrain = share(entry["hemibrain"]["share"])
        label = SHORT.get(entry["id"], s["label"])
        short, long = [label, published, hemibrain], [label]
        for view in result["views"]:
            v = entry["views"][view]
            if not v["neurons"]:
                short += ["---", "---"]
                long += ["0", "---", "---"]
                continue
            at_share, at_half = v["levels"]["published"], v["levels"]["half"]
            bounds = f"{share(v['lower'])}--{share(v['upper'])}"
            short += [bounds, answers(at_share)]
            long += [str(v["neurons"]), answers(at_half), f"{at_share['weight']}/{at_share['direction']}" if "weight" in at_share else "---"]
        main.append(" & ".join(short) + "\\\\")
        full.append(" & ".join(long) + "\\\\")
    return main, full


def main():
    statements = {s["id"]: s for s in p.read_statements(ROOT / p.STATEMENTS)}
    result = json.loads((p.OUT / "results.json").read_text())
    if [s["id"] for s in result["statements"]] != list(statements):
        raise SystemExit("the results are not those of the statements file")
    if p.sha256(p.OUT / "questions.npz") != result["questions_sha256"]:
        raise SystemExit("the questions are not those of the results")
    q = np.load(p.OUT / "questions.npz")
    truth, level = q["truth"], q["level"]
    yes, no, open_ = truth == p.STATUS["positive"], truth == p.STATUS["negative"], truth == p.STATUS["unresolved"]
    at_share, at_half = level == 0, level == 1
    certifiable = ~open_
    names = POLICIES + ("clairvoyant", "size")
    total = {name: int(q[name][certifiable & at_share].sum()) for name in names}  # at the published share
    half = {name: int(q[name][certifiable & at_half].sum()) for name in names}
    asked = int(at_share.sum())
    neurons = len({(int(view), int(cell)) for view, cell in zip(q["view"][at_share], q["cell"][at_share])})
    sources = sorted({s["source"] for s in statements.values()})
    pooled = {}  # checks on the certifiable questions of both shares, per statement and connectome
    for entry in result["statements"]:
        for view, v in entry["views"].items():
            levels = [a for a in v["levels"].values() if "weight" in a]
            if levels:
                pooled[entry["id"], view] = {name: sum(a[name] for a in levels) for name in POLICIES}
    dearer = [key for key, c in pooled.items() if max(c["recal"], c["ag"]) > c["weight"]]
    weakest = max(pooled, key=lambda key: pooled[key]["weight"] / pooled[key]["direction"])
    absent = [(entry["id"], view) for entry in result["statements"] for view, v in entry["views"].items() if v["neurons"] and v["lower"] == 0]
    source = {s["id"]: s for s in json.loads((p.OUT / "source.json").read_text())["statements"]}
    gap = max(abs(100 * source[key]["share_all"] - float(s["percent"])) for key, s in statements.items())
    raised = [key for key in statements if source[key]["share_kept"] > source[key]["share_all"]]
    saving = 1 - total["direction"] / total["weight"]
    first = {name: int(q[name][0]) for name in q.files}  # the first question of the file: first statement, MaleCNS, published share
    order = list(statements)
    asked_share = certifiable & at_share
    above = asked_share & (q["recal"] > q["direction"])  # questions on which recalibrated AG needs more checks
    above_in = sorted({(order[s], result["views"][v]) for s, v in zip(q["statement"][above], q["view"][above])})
    heavy = asked_share & (q["statement"] == order.index("kcgs2-kc")) & (q["view"] == result["views"].index("malecns"))
    rest_saving = 1 - int(q["direction"][asked_share & ~heavy].sum()) / int(q["weight"][asked_share & ~heavy].sum())
    upper = (q["member"] + q["unresolved"]) / q["total"]  # the largest share that the unresolved input cells still allow
    kc_fly = at_share & (q["statement"] == order.index("kcgs2-kc")) & (q["view"] == result["views"].index("flywire"))
    nearly = sorted(key for key, s in statements.items() if "nearly" in s["quote"] and "nearly" in s["reading"])
    at = {(e["id"], view): v["levels"]["published"] for e in result["statements"] for view, v in e["views"].items()}

    claims = Claims()
    claims("the same questions are asked at both shares", int(at_half.sum()) == asked)
    claims("two target types serve two statements, so there are fewer neurons than questions", neurons < asked)
    claims("every order certifies with at least the clairvoyant number of checks",
           all(total[name] >= total["clairvoyant"] and half[name] >= half["clairvoyant"] for name in POLICIES))
    claims("at the published share direction-fixed checks the fewest cells, then recalibrated AG, then weight order, then AG",
           total["direction"] < total["recal"] < total["weight"] < total["ag"])
    claims("at the published share direction-fixed saves 25 to 35% of the checks of weight order", 0.25 <= saving <= 0.35)
    claims("recalibrated AG recovers only part of the loss of AG: it stays at least 10% above direction-fixed",
           total["recal"] >= 1.1 * total["direction"])
    claims("recalibrated AG is cheaper than direction-fixed on no certifiable question",
           not (q["recal"][certifiable] < q["direction"][certifiable]).any())
    claims("at half the share direction-fixed needs fewer checks than weight order", half["direction"] < half["weight"])
    claims("at the published share recalibrated AG exceeds direction-fixed on four questions: the KCgs2 neurons of MaleCNS asked about "
           "Kenyon cells and the SMP054 neurons of FlyWire asked about aIPg",
           int(above.sum()) == 4 and above_in == [("kcgs2-kc", "malecns"), ("smp054-aipg", "flywire")])
    claims("the two KCgs2 neurons of MaleCNS take more than half of the checks of weight order at the published share",
           int(heavy.sum()) == 2 and 2 * int(q["weight"][heavy].sum()) > total["weight"])
    claims("on the other questions direction-fixed still saves a quarter to a third of the checks", 0.25 <= rest_saving <= 0.35)
    claims("these two neurons account for about half of the checks of weight order and of direction-fixed",
           all(0.45 <= int(q[name][heavy].sum()) / total[name] <= 0.55 for name in ("weight", "direction")))
    claims("the only no answers at the published share are those of the KCgs2 neurons of FlyWire",
           int((no & at_share).sum()) == at["kcgs2-kc", "flywire"]["no"] > 0)
    claims("in the hemibrain the threshold raises the share of MBON09 and MBON21 in the input of FB4R, which over all synapses is "
           "already at the published share",
           source["fb4r-mbon09-mbon21"]["share_kept"] > source["fb4r-mbon09-mbon21"]["share_all"]
           >= float(statements["fb4r-mbon09-mbon21"]["percent"]) / 100)
    claims("over all synapses the share of the Kenyon cells is below the published half and above the share in the benchmark graph",
           source["kcgs2-kc"]["share_kept"] < source["kcgs2-kc"]["share_all"] < float(statements["kcgs2-kc"]["percent"]) / 100)
    claims("most questions are certified yes at the published share", 2 * int((yes & at_share).sum()) > asked)
    claims("some questions are certified no at the published share", int((no & at_share).sum()) > 0)
    claims("some questions are uncertifiable at the published share", int((open_ & at_share).sum()) > 0)
    claims("no question is certified no at half the share", int((no & at_half).sum()) == 0)
    claims("more questions are certified yes at half the share", int((yes & at_half).sum()) > int((yes & at_share).sum()))
    claims("the statements come from two papers", len(sources) == 2)
    claims("two statements are read from 'nearly': the input of FB4R and that of KCgs2 from Kenyon cells", nearly == ["fb4r-mbon09-mbon21", "kcgs2-kc"])
    claims("the no answers do not rest on reading 'nearly half' as a half: in both KCgs2 neurons of FlyWire the bounds leave "
           "Kenyon cells at most 35% of the input",
           int(kc_fly.sum()) == 2 and bool(no[kc_fly].all()) and float(upper[kc_fly].max()) <= 0.35)
    claims("four policies were run on the statements", len(POLICIES) == 4)
    still_open = open_ & at_share
    open_male = int((still_open & (q["view"] == result["views"].index("malecns"))).sum())
    claims("seven of the nine open questions at the published share are in MaleCNS", int(still_open.sum()) == 9 and open_male == 7)
    claims("in every open question the partner cells found fall short of the published share and the unresolved cells carry "
           "enough input to reach it",
           bool(((q["member"] < q["threshold"]) & (q["member"] + q["unresolved"] >= q["threshold"]))[still_open].all()))
    claims("in the hemibrain graph of the benchmark the share of the Kenyon cells is below the published half, the answer that "
           "the KCgs2 neurons of FlyWire receive",
           source["kcgs2-kc"]["share_kept"] < float(statements["kcgs2-kc"]["percent"]) / 100 and bool(no[kc_fly].all()))
    claims("recalibrated AG needs more checks than direction-fixed on the statements at the published share", total["recal"] > total["direction"])
    recal_saving = 1 - total["recal"] / total["weight"]
    claims("on the statements recalibrated AG saves under a tenth of the checks of weight order", 0 < recal_saving < 0.1)
    names = {row["type"]: row for row in json.loads((p.OUT / "names.json").read_text())["types"]}
    claims("no MaleCNS neuron has the type pC1d or aIPg3 and the release declares none to be either, while it has types that begin "
           "with pC1 and with aIPg",
           set(names) == {"pC1d", "aIPg3"}
           and all(row["neurons"] == 0 and row["declared_as_type"] == 0 and row["types_sharing_stem"] > 0 for row in names.values()))
    claims("the two statements that name these types are uncertifiable in every MaleCNS neuron at the published share",
           all(at[key, "malecns"]["uncertifiable"] == at[key, "malecns"]["neurons"] > 0 for key in ("smp054-pc1d", "smp054-aipg")))
    gaps = [abs(100 * source[key]["share_all"] - float(s["percent"])) for key, s in statements.items()]
    unresolved_share = (q["unresolved"] / q["total"])[still_open & (q["view"] == result["views"].index("malecns"))]
    claims("in the uncertifiable questions of MaleCNS unresolved cells carry up to nearly half of the input",
           0.4 <= float(unresolved_share.max()) < 0.5)
    searched = json.loads((ROOT / "inputs/published/searched.json").read_text())["papers"]
    claims("twenty papers were searched, the two sources among them",
           len({s["pmcid"] for s in searched}) == len(searched) == 20 and {s["doi"] for s in statements.values()} <= {s["doi"] for s in searched})
    claims("the statements are quoted from five sentences", len({s["quote"] for s in statements.values()}) == 5)
    claims("the input of FB4R from MBON09 and MBON21 is certified in every neuron of both connectomes",
           all(at["fb4r-mbon09-mbon21", view]["yes"] == at["fb4r-mbon09-mbon21", view]["neurons"] > 0 for view in result["views"]))
    claims("the input of KCgs2 from Kenyon cells reaches the published share in MaleCNS and falls short of it in FlyWire",
           at["kcgs2-kc", "malecns"]["yes"] == at["kcgs2-kc", "malecns"]["neurons"] > 0
           and at["kcgs2-kc", "flywire"]["no"] == at["kcgs2-kc", "flywire"]["neurons"] > 0)
    claims("the partner type without a neuron is pC1d in MaleCNS, uncertifiable there at both shares",
           absent == [("smp054-pc1d", "malecns")]
           and all(a["uncertifiable"] == a["neurons"] for a in
                   next(e for e in result["statements"] if e["id"] == "smp054-pc1d")["views"]["malecns"]["levels"].values()))
    claims("direction-fixed never needs more checks than weight order for a statement and connectome",
           all(c["direction"] <= c["weight"] for c in pooled.values()))
    claims("for one statement and connectome AG and recalibrated AG need more checks than weight order",
           len(dearer) == 1 and min(pooled[dearer[0]]["recal"], pooled[dearer[0]]["ag"]) > pooled[dearer[0]]["weight"])
    claims("the largest saving is on the statement with the smallest published share, certifiable only at half of it",
           weakest[0] == min(statements, key=lambda key: float(statements[key]["percent"])) and "weight" not in at[weakest])
    claims("the check in the source connectome covers the statements of the file", list(source) == list(statements))
    claims("over all synapses the shares are within five points of the published ones", gap < 5)
    claims("the five-synapse threshold raises five of the six shares and lowers that of the connections among Kenyon cells",
           len(raised) == 5 and "kcgs2-kc" not in raised)
    claims("the shares over pairs of at least five synapses are those of the hemibrain graph of the benchmark",
           all(abs(source[e["id"]]["share_kept"] - e["hemibrain"]["share"]) < 1e-12 for e in result["statements"]))
    claims("the worked example is the first FB4R question in MaleCNS at the published share, certified yes",
           (first["statement"], first["view"], first["level"], first["truth"]) == (0, 0, 0, p.STATUS["positive"])
           and list(statements)[0] == "fb4r-mbon09-mbon21" and result["views"][0] == "malecns"
           and first["direction"] == first["clairvoyant"] < first["weight"])
    if claims.failed:
        raise SystemExit("a claim made in words no longer holds:\n  " + "\n  ".join(claims.failed))

    ids = list(statements)
    kc = next(e for e in result["statements"] if e["id"] == "kcgs2-kc")
    macros = dict(
        PubStatements=word(len(statements)), PubAsked=asked, PubNeurons=neurons,
        PubShareYes=int((yes & at_share).sum()), PubShareNo=word(int((no & at_share).sum())), PubShareUncert=word(int((open_ & at_share).sum())),
        PubShareCertifiable=int((certifiable & at_share).sum()),
        PubHalfYes=int((yes & at_half).sum()), PubHalfUncert=word(int((open_ & at_half).sum())),
        PubShareWeight=total["weight"], PubShareDirection=total["direction"], PubShareRecal=total["recal"], PubShareAG=total["ag"],
        PubShareClair=total["clairvoyant"], PubShareSavingPct=f"{100 * saving:.0f}",
        PubHalfWeight=half["weight"], PubHalfDirection=half["direction"],
        PubShareWeightMedian=f"{np.median(q['weight'][certifiable & at_share]):g}",
        PubShareSizeMedian=f"{np.median(q['size'][certifiable & at_share]):g}",
        PubExCells=first["size"], PubExSynapses=first["total"], PubExThreshold=first["threshold"],
        PubExWeight=word(first["weight"]), PubExDirection=word(first["direction"]),
        PubKcHemPct=share(kc["hemibrain"]["share"]), PubKcPublishedPct=f"{float(statements['kcgs2-kc']['percent']):g}",
        PubKcAllPct=share(source["kcgs2-kc"]["share_all"]),
        PubFbAllPct=share(source["fb4r-mbon09-mbon21"]["share_all"]), PubFbHemPct=share(source["fb4r-mbon09-mbon21"]["share_kept"]),
        PubKcFlyUpperPct=share(float(upper[kc_fly].max())), PubPolicies=word(len(POLICIES)),
        PubOpenMale=word(open_male), PubSearched={20: "twenty"}[len(searched)],
        PubRecalSavingPct=f"{100 * recal_saving:.0f}", PubGapMin=f"{min(gaps):.1f}", PubGapMax=f"{max(gaps):.1f}",
        PubOpenUnresMaxPct=f"{100 * float(unresolved_share.max()):.0f}",
        PubNamesPcTypes=names["pC1d"]["types_sharing_stem"], PubNamesAipgTypes=names["aIPg3"]["types_sharing_stem"],
        PubRecalAbove=word(int(above.sum())), PubHeavyWeight=int(q["weight"][heavy].sum()), PubRestSavingPct=f"{100 * rest_saving:.0f}",
        PubDearWeight=pooled[dearer[0]]["weight"], PubDearDirection=pooled[dearer[0]]["direction"],
        PubDearRecal=pooled[dearer[0]]["recal"], PubDearAG=pooled[dearer[0]]["ag"],
        PubWeakWeight=pooled[weakest]["weight"], PubWeakDirection=word(pooled[weakest]["direction"]),
        PubSourceShares=", ".join(share(source[key]["share_all"]) for key in ids[:-1]) + " and " + share(source[ids[-1]]["share_all"]),
        PubAbsentUnresolvedPct=f"{100 * next(e for e in result['statements'] if e['id'] == absent[0][0])['views'][absent[0][1]]['upper']:.0f}")
    (PAPER / "published_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    main_rows, full_rows = tables(statements, result)
    (PAPER / "tables/published_main.tex").write_text("\n".join(main_rows) + "\n")
    (PAPER / "tables/published.tex").write_text("\n".join(full_rows) + "\n")
    print(f"{len(macros)} macros, {len(full_rows)} statements")


if __name__ == "__main__":
    main()
