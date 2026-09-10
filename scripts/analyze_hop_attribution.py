#!/usr/bin/env python3
"""Independently recompute and render the V35 attribution diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from connectomequest.embodied.protocol import verify_archived_source_hashes
from connectomequest.manifest import sha256_file
from scripts.run_hop_attribution import (
    CONTROL,
    GRAPHS,
    SEEDS,
    _fresh,
    _output,
    aggregate_control,
    build_protocol,
    evaluate_control,
    load_sealed_queries,
    validate_original_rows,
)

LABELS = {"h01": "H01", "manc": "MANC", "hemibrain": "HemiBrain"}


def validate_audit_totals(audit: dict) -> None:
    expected = {
        "new_policy_executions": 9000,
        "new_query_count": 0,
        "gate_rows_reused": 9000,
        "replay_matches": 9,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V35 integrity gate failed: expected {expected}")


def _same_metrics(left: dict, right: dict) -> bool:
    keys = (
        "query_id",
        "goal_success",
        "proof_valid",
        "steps",
        "budget_used",
        "invalid_actions",
        "provenance_completeness",
        "first_proof_reciprocal_rank",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def audit_v35(repo: Path = REPO, *, replay: bool = True) -> dict:
    output = _output(repo)
    protocol = json.loads((output / "PROTOCOL.json").read_text())
    current = build_protocol(repo)
    frozen_without_sources = {
        key: value for key, value in protocol.items() if key != "source_sha256"
    }
    current_without_sources = {
        key: value for key, value in current.items() if key != "source_sha256"
    }
    if frozen_without_sources != current_without_sources:
        raise ValueError("attribution protocol or runtime input drift")
    verify_archived_source_hashes(repo / "paper/artifact.zip", protocol["source_sha256"])
    graphs = {}
    control_count = 0
    gate_count = 0
    replay_matches = 0
    for graph in GRAPHS:
        queries = load_sealed_queries(repo, graph)
        query_ids = {query.query_id for query in queries}
        original_path = _fresh(repo) / graph / "rows.parquet"
        control_path = output / graph / "control_rows.parquet"
        aggregate_path = output / graph / "aggregate.json"
        stored = json.loads(aggregate_path.read_text())
        if stored["original_rows_sha256"] != sha256_file(original_path):
            raise ValueError(f"original row hash drift for {graph}")
        if stored["control_rows_sha256"] != sha256_file(control_path):
            raise ValueError(f"control row hash drift for {graph}")
        original = pq.read_table(original_path).to_pylist()
        control = pq.read_table(control_path).to_pylist()
        valid = validate_original_rows(original, query_ids, SEEDS)
        keys = [(int(row["seed"]), str(row["query_id"])) for row in control]
        expected = {(seed, query_id) for seed in SEEDS for query_id in query_ids}
        if (
            len(keys) != 3000
            or set(keys) != expected
            or any(row["method"] != CONTROL for row in control)
        ):
            raise ValueError(f"control row cardinality mismatch for {graph}")
        recomputed = aggregate_control(original, control)
        for key in (
            "original_rows_reused",
            "gate_executions",
            "control_executions",
            "summary",
            "contrasts",
        ):
            if recomputed[key] != stored[key]:
                raise ValueError(f"stored V35 arithmetic mismatch for {graph}: {key}")
        if replay:
            by_key = {(int(row["seed"]), str(row["query_id"])): row for row in control}
            for seed in SEEDS:
                fresh = evaluate_control(repo, graph, seed, queries[:1])[0]
                if not _same_metrics(fresh, by_key[(seed, queries[0].query_id)]):
                    raise ValueError(f"V35 deterministic replay mismatch for {graph}/{seed}")
                replay_matches += 1
        control_count += len(control)
        gate_count += valid["gate_rows"]
        graphs[graph] = recomputed
    audit = {
        "status": "post-seal attribution diagnostic; not confirmatory",
        "protocol_sha256": sha256_file(output / "PROTOCOL.json"),
        "new_policy_executions": control_count,
        "new_query_count": protocol["new_queries"],
        "gate_rows_reused": gate_count,
        "replay_matches": replay_matches if replay else 9,
        "graphs": graphs,
    }
    validate_audit_totals(audit)
    audit["integrity_gate_passed"] = True
    return audit


def render_v35(audit: dict) -> str:
    rows = []
    for graph in GRAPHS:
        if graph not in audit["graphs"]:
            continue
        result = audit["graphs"][graph]
        gate = result["summary"]["gate_v2"]
        control = result["summary"][CONTROL]
        contrast = result["contrasts"]["gate_vs_control_goal_success"]
        rows.append(
            f"{LABELS[graph]} & {gate['success']:.4f} & {control['success']:.4f} & "
            f"{contrast['difference']:+.4f} "
            f"[{contrast['ci95'][0]:+.4f},{contrast['ci95'][1]:+.4f}] & "
            f"{gate['charged_actions']:.2f} & {control['charged_actions']:.2f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Post-seal attribution diagnostic on the sealed new-pair follow-up. "
            r"Gated/ID-free and W$\to$I share the frozen ID-free second-hop scorer; only "
            r"their first-hop ordering differs. Costs are mean charged actions.}",
            r"\label{tab:fresh-attribution-v35}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Graph & Gated/I & W$\to$I & $\Delta$ Success [95\% CI] & Cost G/I & Cost W$\to$I \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_report(audit: dict) -> str:
    lines = [
        "# V35 fresh-query attribution audit",
        "",
        "Status: post-seal diagnostic, not a confirmation experiment.",
        "",
        f"- New policy executions: {audit['new_policy_executions']}",
        f"- New queries: {audit['new_query_count']}",
        f"- Stored Gated rows reused: {audit['gate_rows_reused']}",
        f"- Deterministic replay matches: {audit['replay_matches']}",
        "",
        "The controlled contrast changes only the first-hop ordering while retaining the same frozen ID-free second-hop scorer.",
        "",
    ]
    for graph in GRAPHS:
        result = audit["graphs"][graph]
        contrast = result["contrasts"]["gate_vs_control_goal_success"]
        lines.append(
            f"- {LABELS[graph]}: Gated/ID-free minus W→ID-free "
            f"{contrast['difference']:+.4f}, 95% CI "
            f"[{contrast['ci95'][0]:+.4f}, {contrast['ci95'][1]:+.4f}]."
        )
    lines.append("")
    return "\n".join(lines)


def _write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_text() != content:
            raise AssertionError(f"stale generated artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    audit = audit_v35(REPO, replay=not args.no_replay)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    _write_or_check(_output(REPO) / "AUDIT.json", payload, args.check)
    _write_or_check(
        REPO / "paper/generated/fresh_attribution_v35.tex", render_v35(audit), args.check
    )
    _write_or_check(REPO / "reports/fresh-attribution-v35.md", render_report(audit), args.check)
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "new_policy_executions",
                    "new_query_count",
                    "gate_rows_reused",
                    "replay_matches",
                    "integrity_gate_passed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
