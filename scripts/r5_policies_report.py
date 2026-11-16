#!/usr/bin/env python3
"""Round 5, workstream ``policies``: markdown tables from ``analysis-<split>.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.round5 import policies_data as pdata  # noqa: E402


def ci(r, digits=3):
    if r is None:
        return "-"
    return f"{r['value']:.{digits}f} [{r['low']:.{digits}f}, {r['high']:.{digits}f}]"


def rate(r):
    return f"{100 * r['rate']:.2f}% [{100 * r['low']:.2f}, {100 * r['high']:.2f}] (n={r['n']})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="development")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default = pdata.OUT / f"summary-{args.split}.md"
    out = pdata.default_output(args.output or default, default, args.split)
    pdata.require_record(__file__, args.split)
    pdata.fresh(out)
    if args.split == "test":  # the round-5 one-shot gate (after every import and the output check)
        from connectomequest.round5 import heldout
        heldout.claim(__file__, "policies", "report", out)
    a = json.loads((pdata.OUT / f"analysis-{args.split}.json").read_text())
    lines = [f"# Round 5 `policies` - {args.split} split", "", "Ratios of total checks on certifiable questions, 95% type-cluster bootstrap intervals.", ""]
    lines += ["## Policy arms", "", "| view:prior | arm | / weight | / recal (stored) | / AG (stored) | other |", "|---|---|---|---|---|---|"]
    for key, res in a["runs"].items():
        for name, row in res["arms"].items():
            other = ""
            if "to_cheaper" in row:
                other = f"/ cheaper {ci(row['to_cheaper'])}; max per question {row['max_per_question_over_cheaper']:.2f}; / base {ci(row['to_base'])}"
            elif "to_prop-recal" in row and name != "prop-recal":
                other = f"/ prop-recal {ci(row['to_prop-recal'])}"
            lines.append(f"| {key} | {name} | {ci(row['to_weight'])} | {ci(row.get('to_recal'))} | {ci(row.get('to_ag'))} | {other} |")
        refs = "; ".join(f"{k} {ci(v)}" for k, v in res["reference_to_weight"].items())
        lines.append(f"| {key} | *stored* | {refs} | | | certifiable {res['certifiable']} of {res['sessions']} |")
    if a.get("cross_prior"):
        lines += ["", "**Propagation with a prior / propagation without prediction** (`prop-recal`, prior vs `const_half`, same questions): "
                  + "; ".join(f"{k} {ci(v['prop_over_prop_const_half'])}" for k, v in a["cross_prior"].items())]
    if a.get("voi_selection"):
        s = a["voi_selection"]
        lines += ["", f"**VOI selection** ({s['rule']}): " + ", ".join(f"{k} {v:.4f}" for k, v in s["scores"].items()) + f" -> **{s['chosen']}**"]
    lines += ["", "## Optimality gap (exact DP, independent prior)", ""]
    for key, res in a["dp"].items():
        lines += [f"### {key}: {res['sessions']} questions ({res['certifiable']} certifiable), {res['families']} families, {res['types']} types; sizes {res['sizes']}", "",
                  f"AG optimal in expectation on {100 * res['share_ag_optimal_in_expectation']:.1f}% of questions; optimum / weight in expectation {ci(res['expected_opt_to_weight'], 4)}", "",
                  "| policy | E[cost] / E[opt] (all) | E[cost] / E[opt] (certifiable) | largest per-question | realized / weight | realized / realized opt |", "|---|---|---|---|---|---|"]
        for name in res["expected_over_opt"]:
            lines.append(f"| {name} | {ci(res['expected_over_opt'][name], 4)} | {ci(res['expected_over_opt_certifiable'][name], 4)} | {res['largest_expected_gap'][name]:.3f} | "
                         f"{ci(res['realized_to_weight'][name])} | {ci(res['realized_to_opt'][name])} |")
        b = res["realized_opt_better_tied_worse"]
        lines += ["", f"Optimum vs weight order on the true answers: fewer / equal / more checks on {100 * b[0]:.1f}% / {100 * b[1]:.1f}% / {100 * b[2]:.1f}% of certifiable questions.", ""]
    lines += ["## Sampling tests without replacement", ""]
    for view, res in a["wor"].items():
        lines += [f"### {view} ({res['certifiable']} certifiable of {res['sessions']})", "", "| input cells | n | WoR | WoR hybrid | WR (frozen) | WR hybrid (post hoc) |", "|---|---|---|---|---|---|"]
        lines.append(f"| all | {res['certifiable']} | {ci(res['overall'].get('wor'))} | {ci(res['overall'].get('worhyb'))} | {ci(res['overall'].get('wr'))} | {ci(res['overall'].get('wrhyb'))} |")
        for label, s in res["strata"].items():
            lines.append(f"| {label} | {s['questions']} | {ci(s.get('wor'))} | {ci(s.get('worhyb'))} | {ci(s.get('wr'))} | {ci(s.get('wrhyb'))} |")
        c = res["contrasts"]
        lines += ["", f"WoR / WR: {ci(c['wor_over_wr'])}; WoR hybrid / WR hybrid: {ci(c['worhyb_over_wrhyb'])}; same, >=500 cells: {ci(c['worhyb_over_wrhyb_ge500'])}", ""]
        for name, e in res["errors"].items():
            lines.append(f"* {name}: wrong certificates on certifiable audits {rate(e['certifiable'])}, on uncertifiable {rate(e['uncertifiable'])}; statistical certificates {100 * e['share_statistical']:.1f}% of audits")
        for name, e in res.get("adversarial", {}).items():
            lines.append(f"* adversarial {name}: wrong claim {rate(e['claim_error'])}, any wrong certificate {rate(e['any_error'])}, statistical {100 * e['statistical_share']:.1f}% ({e['thresholds']} thresholds)")
        lines.append("")
    print(pdata.save_text(out, "\n".join(lines) + "\n"))


if __name__ == "__main__":
    main()
