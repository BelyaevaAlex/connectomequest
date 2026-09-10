#!/usr/bin/env python3
"""Regenerate the public graph tables from hash-verified immutable outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from analyze_hop_attribution import audit_v35
from compile_graph_tables import generate as generate_graph_audit

GENERATED = ROOT / "paper/generated"
CLUSTER = ROOT / "outputs/nemo/strong-accept/v38-anchor-cluster-audit/summary.json"
GRAPH_LABELS = {"h01": "H01", "manc": "MANC", "hemibrain": "HemiBrain"}


def publicize(text: str) -> str:
    replacements = (
        (r"\shortstack{MINERVA-\\inspired}", "Recurrent"),
        ("MINERVA-inspired", "Recurrent"),
        ("ID-free", "Attribute"),
        ("Gated/Attribute", "Expert selector"),
        ("Gated", "Expert selector"),
        ("Random+Attribute", r"Random$\to$Attribute"),
        ("coarse7", "coarse"),
        ("Coarse7", "Coarse"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def render_cluster_table() -> str:
    payload = json.loads(CLUSTER.read_text())
    rows = []
    for graph in ("h01", "manc", "hemibrain"):
        information = payload["graphs"][graph]
        contrasts = information["contrasts"]
        chosen = [row for row in contrasts if row.get("familywise")]
        if not chosen:
            chosen = [
                row
                for row in contrasts
                if row["method_a"] == "gate_v2" and row["method_b"] == "weight"
            ]
        contrast = chosen[0]
        low, high = contrast["ci95"]
        rows.append(
            f"{GRAPH_LABELS[graph]} & Expert selector$-$Weight & "
            f"{contrast['estimate']:+.4f} & $[{low:+.4f},{high:+.4f}]$ & "
            f"{contrast['clusters']} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{First-anchor-cluster bootstrap for the held-out graph comparison. Intervals for the two planned fly-graph Expert-selector--Weight contrasts use a two-comparison Bonferroni correction; H01 is descriptive with a nominal 95\% interval.}",
            r"\label{tab:cluster-bootstrap}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            "Graph & Contrast & Difference & Interval & Clusters \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_hop_table() -> str:
    audit = audit_v35(ROOT, replay=False)
    rows = []
    for graph in ("h01", "manc", "hemibrain"):
        result = audit["graphs"][graph]
        expert = result["summary"]["gate_v2"]
        control = result["summary"]["weight_first_idfree_second"]
        rows.append(
            f"{GRAPH_LABELS[graph]} & {expert['success']:.4f} & {control['success']:.4f} & "
            f"{expert['success'] - control['success']:+.4f} & "
            f"{expert['charged_actions'] - control['charged_actions']:+.2f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Hop attribution on the held-out evaluation. Expert selector and Weight$\to$Attribute share the Attribute second-hop scorer; only first-hop ordering differs.}",
            r"\label{tab:hop-attribution}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            "Graph & Expert selector & Weight$\\to$Attribute & Success difference & Charged-action difference \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_all() -> dict[Path, str]:
    generated, _ = generate_graph_audit()
    fresh = publicize(generated["fresh_contract_test.tex"])
    fresh = (
        fresh.replace(
            "Sealed new-pair follow-up evaluation:",
            "Prospectively frozen evaluation on",
        )
        .replace(
            "1,000 previously unlisted query pairs per graph and frozen controller/checkpoints.",
            "1,000 previously unused query pairs per graph.",
        )
        .replace(
            "Each entry is Success/charged actions,",
            "Entries are certificate success/unconditional charged actions,",
        )
        .replace(
            "All specified policies are reported. Excluding old pairs changes the sampling population; absolute rates are not directly comparable with the original test.",
            "All policies use the same frozen controller and observations.",
        )
    )
    coarse_main = publicize(generated["paired_coarse_main.tex"])
    coarse_extra = publicize(generated["paired_coarse_extra.tex"])
    return {
        GENERATED / "graph_results.tex": fresh,
        GENERATED / "cluster_intervals.tex": render_cluster_table(),
        GENERATED / "hop_attribution.tex": render_hop_table(),
        GENERATED / "paired_coarse_main.tex": coarse_main,
        GENERATED / "semantic_coarsening_details.tex": coarse_extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in render_all().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise AssertionError(f"stale generated table: {path}")
        else:
            path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
