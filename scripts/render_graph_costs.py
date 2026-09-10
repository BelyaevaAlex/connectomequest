#!/usr/bin/env python3
"""Render outcome-conditioned costs from the immutable fresh held-out rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import pyarrow.parquet as pq

from connectomequest.graph_reporting import format_optional, summarize_policy_costs
from connectomequest.manifest import sha256_file

SOURCE = REPO / "outputs/nemo/strong-accept/fresh-contract-test-v1"
TARGET = REPO / "paper/generated/graph_costs.tex"
GRAPHS = {"h01": "H01", "manc": "MANC", "hemibrain": "HemiBrain"}
POLICIES = {
    "weight": "Weight",
    "random": "Random",
    "minerva": r"Recurrent",
    "snapshot_v2": "Attribute",
    "gate_v2": "Expert selector",
    "random_snapshot_v2": r"Random$\rightarrow$Attribute",
}


def _load_rows() -> list[dict]:
    protocol = json.loads((SOURCE / "PROTOCOL.json").read_text())
    rows: list[dict] = []
    for graph in GRAPHS:
        directory = SOURCE / graph
        aggregate = json.loads((directory / "aggregate.json").read_text())
        if aggregate["protocol_sha256"] != sha256_file(SOURCE / "PROTOCOL.json"):
            raise ValueError("protocol digest mismatch")
        if aggregate["rows_sha256"] != sha256_file(directory / "rows.parquet"):
            raise ValueError("row digest mismatch")
        graph_rows = pq.read_table(directory / "rows.parquet").to_pylist()
        for row in graph_rows:
            row["graph"] = graph
            row["budget"] = protocol["B"]
            # The frozen controller reserves its last transition for SUBMIT.
            row["exhaustion_at"] = protocol["B"] - 1
        rows.extend(graph_rows)
    return rows


def render() -> str:
    summary = summarize_policy_costs(_load_rows())
    expected = {(graph, policy) for graph in GRAPHS for policy in POLICIES}
    if set(summary) != expected:
        missing = sorted(expected - set(summary))
        extra = sorted(set(summary) - expected)
        raise ValueError(f"policy coverage mismatch; missing={missing}, extra={extra}")
    lines = [
        r"\begin{table}[t]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Charged actions in the held-out graph evaluation, separated by outcome. The unconditional mean retains failed episodes; Exhaust. is the fraction reaching the controller's 63-action exploration limit (one transition is reserved for the zero-cost submission). These costs must be read jointly with success in Table~\ref{tab:fresh-test}.}",
        r"\label{tab:graph-costs}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Graph & Policy & $n$ & All & Success & Failure & Exhaust. \\",
        r"\midrule",
    ]
    for graph, graph_label in GRAPHS.items():
        for policy, policy_label in POLICIES.items():
            d = summary[(graph, policy)]
            lines.append(
                f"{graph_label} & {policy_label} & {d['n']} & {d['actions']:.2f} & "
                f"{format_optional(d['actions_success'])} & "
                f"{format_optional(d['actions_failure'])} & "
                f"{d['budget_exhaustion']:.3f} " + r"\\"
            )
        if graph != "hemibrain":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != text:
            raise SystemExit("stale graph-cost table")
    else:
        TARGET.write_text(text)
    print(TARGET)


if __name__ == "__main__":
    main()
