#!/usr/bin/env python3
"""Verify sealed V39 shards and evaluate prospective statistical gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connectomequest.embodied.protocol import (
    OUTPUT_ROOT,
    atomic_case_write,
    verify_source_hashes,
)
from connectomequest.embodied.statistics import (
    evaluate_correctness_gate,
    evaluate_integrity_gate,
    evaluate_practical_gate,
    evaluate_safety_gate,
    summarize_confirmation,
)

_METHODS = ("full_replan", "full_scan", "receipt_index", "unchecked_reuse")
_COMPETENT = frozenset(("full_replan", "full_scan", "receipt_index"))
_PAIR_FIELDS = (
    "layout_seed",
    "update_seed",
    "condition",
    "update_count",
    "relevant_fraction",
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _load_rows(stage_path: Path, protocol_sha256: str) -> list[dict]:
    rows = []
    for path in sorted((stage_path / "cases").glob("*.json")):
        wrapped = json.loads(path.read_text())
        if wrapped["protocol_sha256"] != protocol_sha256:
            raise ValueError(f"protocol hash mismatch: {path}")
        if wrapped["payload_sha256"] != _digest(wrapped["result"]):
            raise ValueError(f"payload hash mismatch: {path}")
        rows.append(wrapped["result"])
    if not rows:
        raise ValueError("no result shards")
    return rows


def _groups(rows):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in _PAIR_FIELDS)
        grouped.setdefault(key, {})[row["method"]] = row
    return grouped


def _compatible_update_schedules(rows: list[dict]) -> bool:
    """Verify one sealed schedule while allowing a method to terminate early.

    The update digest sequence records events actually exposed to a method.
    The design permits termination before a later scheduled event, so these
    records need only be mutually compatible prefixes of the same prospective
    schedule. The declared complete schedule length must still agree exactly.
    """

    if not rows or len({int(row["update_count"]) for row in rows}) != 1:
        return False
    observed = [tuple(row["update_digests"]) for row in rows]
    reference = max(observed, key=len)
    return all(reference[: len(prefix)] == prefix for prefix in observed)


def _rate(rows, method, field):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"missing method: {method}")
    return sum(float(row[field] > 0) for row in selected) / len(selected)


def build_report(stage_path: Path) -> dict:
    protocol_path = stage_path / "protocol.json"
    payload = json.loads(protocol_path.read_text())
    verify_source_hashes(ROOT, payload["source_hashes"])
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    rows = _load_rows(stage_path, protocol_sha256)
    grouped = _groups(rows)
    complete = [group for group in grouped.values() if set(group) == set(_METHODS)]
    if len(complete) != len(grouped):
        raise ValueError("incomplete paired cells")

    access_match = all(
        len({row["access_digest"] for row in group.values()}) == 1 for group in complete
    )
    prefix_match = all(
        len({row["pre_update_action_digest"] for row in group.values()}) == 1 for group in complete
    )
    schedule_match = all(_compatible_update_schedules(list(group.values())) for group in complete)
    stable_conditions = {
        "none",
        "irrelevant_receipt_withdrawal",
        "alternative_support",
    }
    stable_groups = [
        group for group in complete if next(iter(group.values()))["condition"] in stable_conditions
    ]
    next_action_match = all(
        len({tuple(row["action_trace"]) for row in group.values() if row["method"] in _COMPETENT})
        == 1
        for group in stable_groups
    )
    statistical = summarize_confirmation(
        rows,
        draws=int(payload["bootstrap_draws"]),
        seed=int(payload["bootstrap_seed"]),
    )
    primary = statistical["primary_stratum"]
    competent_unsupported = sum(
        int(row["unsupported_authorizations"]) for row in rows if row["method"] in _COMPETENT
    )
    gate_input = {
        "source_hashes_match": True,
        "pre_update_access_match": access_match,
        "pre_update_action_prefix_match": prefix_match,
        "update_schedules_match": schedule_match,
        "trace_replay_rate": sum(bool(row["trace_replay_ok"]) for row in rows) / len(rows),
        "oracle_fields_exposed": False,
        "suffix_stable_next_action_matches": next_action_match,
        "competent_unsupported_authorizations": competent_unsupported,
        "task_success_difference_ci_low": primary["task_success_difference_ci"][0],
        "time_reduction": primary["time_reduction"],
        "time_reduction_ci_low": primary["time_reduction_ci"][0],
        "work_reduction": primary["work_reduction"],
        "total_reasoning_reduction": primary["total_reasoning_reduction"],
        "wall_clock_increase": primary["wall_clock_increase"],
        "receipt_index_unsupported_rate": _rate(
            rows, "receipt_index", "unsupported_authorizations"
        ),
        "unchecked_reuse_unsupported_rate": _rate(
            rows, "unchecked_reuse", "unsupported_authorizations"
        ),
    }
    gates = (
        evaluate_integrity_gate(gate_input),
        evaluate_correctness_gate(gate_input),
        evaluate_practical_gate(gate_input),
        evaluate_safety_gate(gate_input),
    )
    return {
        "protocol_sha256": protocol_sha256,
        "rows": len(rows),
        "paired_cells": len(grouped),
        "statistics": statistical,
        "gates": [asdict(gate) for gate in gates],
        "all_gates_passed": all(gate.passed for gate in gates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("smoke", "pilot", "confirmation"),
        required=True,
    )
    args = parser.parse_args()
    stage_path = OUTPUT_ROOT / args.stage
    report = build_report(stage_path)
    atomic_case_write(stage_path / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
