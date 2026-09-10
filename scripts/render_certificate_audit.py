#!/usr/bin/env python3
"""Render the public verifier audit from hash-verified immutable aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "outputs/nemo/strong-accept/proof-accounting-v2/aggregate.json"
STRUCTURAL = ROOT / "outputs/nemo/strong-accept/proof-structural-mutation-core-v1/aggregate.json"
RELEASE = ROOT / "outputs/nemo/strong-accept/RELEASE-MANIFEST.json"
GENERATED = ROOT / "paper/generated"
GRAPHS = ("h01", "manc", "hemibrain")
GRAPH_LABELS = {"h01": "H01", "manc": "MANC", "hemibrain": "HemiBrain"}
PROVENANCE = (
    ("missing_required_evidence", "Missing required receipt"),
    ("stale_episode_receipt", "Stale episode binding"),
    ("wrong_graph_source", "Wrong graph binding"),
    ("wrong_query_binding", "Wrong query binding"),
    ("wrong_relation", "Wrong relation binding"),
)
STRUCTURAL_CLASSES = (
    ("inconsistent_middle", "Inconsistent middle node"),
    ("wrong_answer_binding", "Wrong answer binding"),
    ("wrong_semantic_token", "Wrong semantic receipt"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sum_policy_counts(graph_payload: dict, key: str) -> int:
    return sum(int(policy[key]) for policy in graph_payload.values())


def summarize(proof: dict, structural: dict) -> dict:
    total = proof["total"]
    replay = {
        "episodes": int(total["episodes"]),
        "nonempty": int(total["nonempty_submit_attempts"]),
        "accepted": int(total["accepted_submissions"]),
    }
    if replay["accepted"] != replay["nonempty"]:
        raise AssertionError("not every unmodified nonempty certificate was accepted")
    if float(total["accepted_proof_replay_rate"]) != 1.0:
        raise AssertionError("unmodified replay rate is not one")

    rows: list[dict] = []
    corruption = proof["deliberate_corruption_audit"]
    for key, label in PROVENANCE:
        attempted_by_graph = {
            graph: _sum_policy_counts(corruption[graph], f"{key}:attempted") for graph in GRAPHS
        }
        rejected_by_graph = {
            graph: _sum_policy_counts(corruption[graph], f"{key}:rejected") for graph in GRAPHS
        }
        rows.append(
            {
                "family": "Provenance/binding",
                "label": label,
                "attempted_by_graph": attempted_by_graph,
                "rejected_by_graph": rejected_by_graph,
                "attempted": sum(attempted_by_graph.values()),
                "rejected": sum(rejected_by_graph.values()),
            }
        )

    for key, label in STRUCTURAL_CLASSES:
        attempted_by_graph = {
            graph: int(structural["graphs"][graph]["counts"][f"{key}:attempted"])
            for graph in GRAPHS
        }
        rejected_by_graph = {
            graph: int(structural["graphs"][graph]["counts"][f"{key}:rejected"]) for graph in GRAPHS
        }
        rows.append(
            {
                "family": "Structural",
                "label": label,
                "attempted_by_graph": attempted_by_graph,
                "rejected_by_graph": rejected_by_graph,
                "attempted": sum(attempted_by_graph.values()),
                "rejected": sum(rejected_by_graph.values()),
            }
        )

    mutation_total = {
        "attempted": sum(row["attempted"] for row in rows),
        "rejected": sum(row["rejected"] for row in rows),
    }
    if mutation_total != {"attempted": 13_800, "rejected": 13_800}:
        raise AssertionError(f"unexpected mutation totals: {mutation_total}")
    if not bool(structural["all_rejected"]):
        raise AssertionError("structural audit is not all-rejected")
    return {"replay": replay, "rows": rows, "mutation_total": mutation_total}


def verify_sources(proof: dict) -> None:
    for relative, expected in proof["input_sha256"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise AssertionError(f"proof-audit input hash mismatch: {relative}")
    release = json.loads(RELEASE.read_text())
    relative = str(STRUCTURAL.relative_to(ROOT))
    expected = release["files"][relative]
    if sha256(STRUCTURAL) != expected:
        raise AssertionError("structural aggregate differs from release manifest")


def render_numbers(summary: dict) -> str:
    replay = summary["replay"]
    mutation = summary["mutation_total"]
    return "\n".join(
        [
            f"\\newcommand{{\\VerifierReplayEpisodes}}{{{replay['episodes']:,}}}",
            f"\\newcommand{{\\VerifierReplayCertificates}}{{{replay['nonempty']:,}}}",
            f"\\newcommand{{\\VerifierAcceptedCertificates}}{{{replay['accepted']:,}}}",
            f"\\newcommand{{\\VerifierMutationAttempts}}{{{mutation['attempted']:,}}}",
            f"\\newcommand{{\\VerifierMutationRejections}}{{{mutation['rejected']:,}}}",
            "",
        ]
    )


def _cell(row: dict, graph: str) -> str:
    rejected = row["rejected_by_graph"][graph]
    attempted = row["attempted_by_graph"][graph]
    return f"{rejected:,}/{attempted:,}"


def render_table(summary: dict) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\caption{Constructed certificate mutations, reported as rejected/attempted. Rows are correlated coverage checks of the stated mutation generator, not independent attack samples or estimates of an attack-success probability.}",
        r"\label{tab:mutation-breakdown}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Mutation & H01 & MANC & HemiBrain & Total \\",
        r"\midrule",
    ]
    last_family = None
    for row in summary["rows"]:
        if last_family is not None and row["family"] != last_family:
            lines.append(r"\midrule")
        family = row["family"] if row["family"] != last_family else ""
        cells = [_cell(row, graph) for graph in GRAPHS]
        lines.append(
            f"{family} & {row['label']} & "
            + " & ".join(cells)
            + f" & {row['rejected']:,}/{row['attempted']:,} \\\\"
        )
        last_family = row["family"]
    total = summary["mutation_total"]
    lines.extend(
        [
            r"\midrule",
            f"Total & -- & -- & -- & -- & {total['rejected']:,}/{total['attempted']:,} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def outputs() -> dict[Path, str]:
    proof = json.loads(PROOF.read_text())
    structural = json.loads(STRUCTURAL.read_text())
    verify_sources(proof)
    summary = summarize(proof, structural)
    return {
        GENERATED / "certificate_numbers.tex": render_numbers(summary),
        GENERATED / "certificate_mutations.tex": render_table(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in outputs().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise AssertionError(f"stale generated verifier artifact: {path}")
        else:
            path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
