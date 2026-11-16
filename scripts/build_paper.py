#!/usr/bin/env python3
"""Build the manuscript (main text, references, appendix and reproducibility checklist in one document), including the
further exploratory analyses.

scripts/v35_paper_assets.py and scripts/build_v35_paper.py are pinned by outputs/round5/record.json, the record
written before the held-out run of the further exploratory analyses, and are not edited. This script runs them
unchanged, with three additions written after that run:

1. v35_paper_assets.round5_prop_assets, the recorded check of the statement on propagation with other priors,
   is run as recorded. On the held-out split one of its six parts fails: with seeds from 10% of the neurons,
   propagation saves 4-5% of the checks, and in one of four comparisons just over the 5% the recorded statement
   allows (ratio 0.9497 against a bound of 0.95; the same statement is the
   recorded expectation S2 in outputs/round5/prop/expectations.json; the appendix reports the failure). Only
   this failure is tolerated; it is printed, and the paper states the revised statement checked by
   revised_prop_assets below.
2. The generated tables use the terms of the paper: the method is "outcome propagation" (the recorded code wrote
   "answer propagation"), and "certificate" is kept for exact certificates, so the hypothesis about the sampling
   test speaks of its wrong answers (the recorded code wrote "wrong sampling certificates"); the order by the larger
   expected progress toward either goal is the "max-progress order" (the recorded code wrote "confident-first",
   which the paper reserves for confident synthetic priors).
3. The numbers and tables of the further analyses are written by scripts/r5_paper_assets.py, those of the analysis
   of questions of both answers by scripts/answers_assets.py, those of the replication on new target neurons by
   scripts/replication_assets.py, those of the tests on new cell types by scripts/newtypes_assets.py, those of the
   analysis of missing type correspondences by scripts/correspondences_assets.py, those of the answers under other
   labels by scripts/labelings_assets.py, those of the family of Proposition 1 by scripts/hedge_family.py and those
   of the published statements by scripts/published_assets.py.
   scripts/figures_extra.py draws the explanatory figures of the main text (pipeline, hedge, releases) and the curve
   of the complete lists of missing correspondences from these results; it runs no policy on held-out questions.
4. The table of the main text keeps the policies that carry its argument (paper/tables/main_core.tex), with the
   pre-specified result of Adaptive Dual Greedy under the earlier-release prior added from the generated numbers; the
   complete table of the pre-specified policies stays in the appendix, in a one-column form written from the
   generated table (paper/tables/full_test_narrow.tex: every row, with the endpoint and its interval).

Usage: python scripts/build_paper.py [--assets-only]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_v35_paper as build  # noqa: E402  (pinned, run unchanged)
import v35_paper_assets as assets  # noqa: E402  (pinned, run unchanged)

RECORDED_FAILURE = ("claims about propagation and strong priors no longer hold: "
                    "with a tenth of the neurons seeded it neither saves nor costs (within 5%)")
RELABEL = (("answer propagation", "outcome propagation"), ("Answer propagation", "Outcome propagation"),
           ("Knows every answer", "Knows every outcome"),
           ("Wrong sampling certificates in at most 5\\% of audits", "Wrong answers of the sampling test in at most 5\\% of audits"),
           ("wrong sampling certificates $\\leq 5\\%$", "wrong sampling answers $\\leq 5\\%$"),
           (", confident-first", ", max-progress order"))
REVISED_MACROS = ("PropNoneLow", "PropNoneHigh", "PropNoneVsTopoLow", "PropNoneVsTopoHigh", "PropProvCI", "PropProvSmallCI",
                  "PropProvBigCI", "PropProvSavePct", "PropSeedTenLowPct", "PropSeedTenHighPct", "PropSeedMoreLowPct",
                  "PropSeedMoreHighPct")
recorded_prop_assets = assets.round5_prop_assets


def revised_prop_assets(split="test"):
    """The statement of the paper, revised after the held-out run: propagation saves most without a prior, less
    with the earlier-release prior, less again with a seeded prior from 10% of the neurons, and costs extra with
    seeds from 30 or 50%. The build stops if the analysis contradicts any part of it."""
    found = sorted((ROOT / "outputs/round5/prop").glob(f"analysis-{split}-*.json"))
    if split == "development":
        found = found and [max(found, key=lambda p: p.stat().st_mtime)]
    if not found:
        print("propagation with other priors: analysis not run yet, macros left as ??")
        return {k: r"\textbf{??}" for k in REVISED_MACROS}
    if len(found) > 1:
        raise SystemExit(f"several propagation analyses: {[p.name for p in found]}")
    views = assets.read(found[0])["views"]
    prop = lambda view, arm: views[view]["propagation"][arm]
    three = ("malecns", "flywire", "banc")
    none = [prop(v, "const_half:recal+prop")["to_weight"]["value"] for v in three]
    vs_topo = [views[v]["contrasts"]["need the MLP? const_half:recal+prop / mlp_s0:recal+prop"]["ratio"]["value"] for v in three]
    prov = prop("flywire", "provisional:recal+prop")
    prov_small, prov_big = (prov["to_static_by_size"][k] for k in ("<500 input cells", ">=500 input cells"))
    ten = [prop(v, f"seeded10{c}:recal+prop")["to_static"] for v in ("malecns", "flywire") for c in ("", "conn")]
    more = [prop(v, f"seeded{f}{c}:recal+prop")["to_static"] for v in ("malecns", "flywire") for f in (30, 50) for c in ("", "conn")]
    checks = {
        "without a prior propagation saves at least a seventh of weight order": max(none) <= 0.86,
        "without a prior it needs about as many checks as with the topology prior": 0.97 <= min(vs_topo) and max(vs_topo) <= 1.05,
        "with the earlier-release prior it saves": prov["to_static"]["high"] < 1,
        "on targets with <500 inputs": prov_small["high"] < 1,
        "and not on targets with >=500 inputs": prov_big["value"] >= 1,
        "with a tenth of the neurons seeded it saves, less than with the earlier-release prior":
            all(t["high"] < 1 for t in ten) and prov["to_static"]["value"] < min(t["value"] for t in ten),
        "with a third or a half seeded it costs extra": all(m["low"] > 1 for m in more),
    }
    failed = [claim for claim, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("revised claims about propagation and strong priors do not hold: " + "; ".join(failed))
    ten_v, more_v = [t["value"] for t in ten], [m["value"] for m in more]
    return dict(
        PropNoneLow=f"{min(none):.2f}", PropNoneHigh=f"{max(none):.2f}",
        PropNoneVsTopoLow=f"{min(vs_topo):.2f}", PropNoneVsTopoHigh=f"{max(vs_topo):.2f}",
        PropProvCI=assets.ci(prov["to_static"]), PropProvSmallCI=assets.ci(prov_small), PropProvBigCI=assets.ci(prov_big),
        PropProvSavePct=f"{100 * (1 - prov['to_static']['value']):.0f}",
        PropSeedTenLowPct=f"{100 * (1 - max(ten_v)):.0f}", PropSeedTenHighPct=f"{100 * (1 - min(ten_v)):.0f}",
        PropSeedMoreLowPct=f"{100 * (min(more_v) - 1):.0f}", PropSeedMoreHighPct=f"{100 * (max(more_v) - 1):.0f}",
    )


def prop_assets(split="test"):
    """Run the recorded check unchanged, tolerate only its known failure, then return the revised macros."""
    try:
        recorded_prop_assets(split)
    except SystemExit as failure:
        if str(failure) != RECORDED_FAILURE:
            raise
        print(f"recorded check fails as reported in the appendix: {failure}")
    return revised_prop_assets(split)


def relabel():
    for path in sorted((build.PAPER / "tables").glob("*.tex")):
        text = path.read_text()
        new = text
        for old, term in RELABEL:
            new = new.replace(old, term)
        if new != text:
            path.write_text(new)


CORE_DROPPED = ("\\multicolumn{3}{@{}l}{\\emph{Without a prior}}", "\\quad random order", "\\quad sampling test", "\\quad member-first")
EARLIER_AG = "\\quad Adaptive Greedy (AG) & --- &"


def macro(name):
    found = re.findall(r"\\newcommand\{\\" + name + r"\}\{(.*)\}", (build.PAPER / "v35_numbers.tex").read_text())
    if len(found) != 1:
        raise SystemExit(f"macro {name}: found {len(found)} definitions")
    return found[0]


def core_table():
    """Main-text table: the generated table of all pre-specified policies without the comparators named in CORE_DROPPED,
    and with the pre-specified result of Adaptive Dual Greedy under the earlier-release prior, which the generated
    table gives only for the topology prior. The build stops unless both standard adaptive policies forfeit about two
    thirds of the saving of direction-fixed with that prior, as the text says."""
    lines = (build.PAPER / "tables/main.tex").read_text().splitlines()
    kept = [line for line in lines if not line.startswith(CORE_DROPPED)]
    dropped = [line for line in lines if line.startswith(CORE_DROPPED)]
    if len(dropped) != 5 or len({line.split("&")[0] for line in dropped}) != len(CORE_DROPPED):
        raise SystemExit(f"main table: expected to leave out five rows of four kinds, found {len(dropped)}")
    at = [i for i, line in enumerate(kept) if line.startswith(EARLIER_AG)]
    if len(at) != 1 or sum(line.startswith("\\quad Adaptive Dual Greedy") for line in kept) != 1:
        raise SystemExit("main table: rows of the adaptive policies are not where expected")
    kept.insert(at[0] + 1, f"\\quad Adaptive Dual Greedy & --- & {macro('FlyProvADGCI')}\\\\")
    direction = float(macro("FlyProvDirection"))
    for name in ("FlyProvAG", "FlyProvADG"):
        forfeited = (float(macro(name)) - direction) / (1 - direction)
        if not 0.6 <= forfeited <= 0.75:
            raise SystemExit(f"a claim made in words no longer holds: {name} forfeits {forfeited:.2f} of the saving of direction-fixed, not two thirds")
    (build.PAPER / "tables/main_core.tex").write_text("\n".join(kept) + "\n")


PRIORS = (("MLP, ", "Topology prior"), ("Linear, ", "Linear prior"), ("NTAC, ", "NTAC prior"), ("Earlier release, ", "Earlier-release prior"))


def narrow_full_table():
    """One-column form of the generated table of all pre-specified policies: every row, grouped by prior, with the
    endpoint and its interval for both connectomes; the two further columns of the generated table stay in the archive."""
    lines = (build.PAPER / "tables/full_test.tex").read_text().splitlines()
    out, current = [], None
    for line in lines:
        cells = [c.strip() for c in line.removesuffix("\\\\").split("&")]
        if len(cells) != 7:
            raise SystemExit(f"table of all policies: unexpected row {line!r}")
        name, prior = cells[0], None
        for prefix, title in PRIORS:
            if name.startswith(prefix):
                name, prior = name[len(prefix):], title
        if prior != current and prior is not None:
            out.append(f"\\addlinespace[2pt]\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{prior}}}}}\\\\")
        if prior is None and current is not None:
            raise SystemExit("table of all policies: a row without a prior follows the rows of a prior")
        current = prior
        indent = "" if prior is None else "\\quad "
        out.append(f"{indent}{name} & {cells[1]} & {cells[4]}\\\\")
    if sum("multicolumn" in line for line in out) != len(PRIORS) or len(out) != len(lines) + len(PRIORS):
        raise SystemExit("table of all policies: the rows do not fall into the four priors")
    (build.PAPER / "tables/full_test_narrow.tex").write_text("\n".join(out) + "\n")


BODY_PAGES = 8  # the limit of the venue; references and appendix are not counted


def compile_documents():
    """The checks of build_v35_paper.main after its asset step, for one document: main text, references, appendix.

    The body may take BODY_PAGES pages: the reference list must start on one of them, or at the very top of the
    next page. The appendix follows the references.
    """
    for name in ("main", "appendix"):
        found = build.FORBIDDEN.search((build.PAPER / f"{name}.tex").read_text())
        if found:
            raise SystemExit(f"{name}: forbidden string {found.group(0)!r}")
    if "Anonymous Submission" not in (build.PAPER / "main.tex").read_text():
        raise SystemExit("main: anonymity marker missing")
    pages = build.build("main")
    texts = [subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "main.pdf", "-"], cwd=build.PAPER,
                            capture_output=True, text=True).stdout for page in range(1, pages + 1)]
    references = [page for page, body in enumerate(texts, 1) if re.search(r"^References$", body, re.MULTILINE)]
    appendix = [page for page, body in enumerate(texts, 1) if re.search(r"^A\s+Overview$", body, re.MULTILINE)]
    if len(references) != 1 or len(appendix) != 1:
        raise SystemExit(f"reference list on pages {references}, start of the appendix on pages {appendix}; one of each is expected")
    before = re.split(r"^References$", texts[references[0] - 1], flags=re.MULTILINE)[0].strip()
    body = references[0] if before else references[0] - 1
    if body > BODY_PAGES:
        raise SystemExit(f"the body runs to page {body}; the limit is {BODY_PAGES} pages without the references")
    if appendix[0] < references[0]:
        raise SystemExit("the appendix starts before the references")
    checklist = (build.PAPER / "ReproducibilityChecklist.tex").read_text().split("% The questions start here")[1]
    if re.search(r"^\s*Type your response here\s*$", checklist, re.MULTILINE):
        raise SystemExit("the reproducibility checklist has a question without a response")
    listed = [page for page, body in enumerate(texts, 1) if re.search(r"^Reproducibility Checklist$", body, re.MULTILINE)]
    if len(listed) != 1 or listed[0] < appendix[0]:
        raise SystemExit(f"reproducibility checklist on pages {listed}; it must follow the appendix")
    print(f"reproducibility checklist starts on page {listed[0]}")
    print(f"body ends on page {body} of {BODY_PAGES} allowed; references start on page {references[0]}, appendix on page {appendix[0]}")
    print({"main": pages})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-only", action="store_true", help="write numbers, tables and figures; do not compile")
    args = ap.parse_args()
    assets.round5_prop_assets = prop_assets  # looked up by assets.main at call time
    assets.main()
    relabel()
    core_table()
    narrow_full_table()
    subprocess.run([sys.executable, str(ROOT / "scripts/r5_paper_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/answers_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/correspondences_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/labelings_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/hedge_family.py")], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / "scripts/published_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/replication_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/newtypes_assets.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/figures_extra.py")], check=True)
    if not args.assets_only:
        compile_documents()


if __name__ == "__main__":
    main()
