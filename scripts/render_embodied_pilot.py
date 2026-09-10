#!/usr/bin/env python3
"""Render the deterministic V39 full-episode pilot audit for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs/nemo/strong-accept/full-episode-v39-attempt-07/pilot"
OUTPUT_JSON = ROOT / "paper/generated/full_episode_v39_pilot.json"
OUTPUT_TEX = ROOT / "paper/generated/full_episode_v39_pilot.tex"
METHODS = ("full_replan", "full_scan", "receipt_index", "unchecked_reuse")
PAIR_FIELDS = (
    "layout_seed",
    "update_seed",
    "condition",
    "update_count",
    "relevant_fraction",
)


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_rows() -> tuple[list[dict], str]:
    protocol = STAGE / "protocol.json"
    protocol_sha256 = hashlib.sha256(protocol.read_bytes()).hexdigest()
    rows = []
    for path in sorted((STAGE / "cases").glob("*.json")):
        wrapped = json.loads(path.read_text())
        if wrapped["protocol_sha256"] != protocol_sha256:
            raise ValueError(f"protocol hash mismatch: {path}")
        if wrapped["payload_sha256"] != _digest(wrapped["result"]):
            raise ValueError(f"payload hash mismatch: {path}")
        rows.append(wrapped["result"])
    return rows, protocol_sha256


def _schedule_prefixes_match(group: dict[str, dict]) -> bool:
    if len({row["update_count"] for row in group.values()}) != 1:
        return False
    observed = [tuple(row["update_digests"]) for row in group.values()]
    reference = max(observed, key=len)
    return all(reference[: len(prefix)] == prefix for prefix in observed)


def build_summary() -> dict:
    rows, protocol_sha256 = _load_rows()
    groups: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in PAIR_FIELDS)
        groups.setdefault(key, {})[row["method"]] = row
    if any(set(group) != set(METHODS) for group in groups.values()):
        raise ValueError("incomplete paired cells")
    if any(
        len({row["access_digest"] for row in group.values()}) != 1
        or len({row["pre_update_action_digest"] for row in group.values()}) != 1
        or not _schedule_prefixes_match(group)
        for group in groups.values()
    ):
        raise ValueError("method access or prospective schedule mismatch")
    if not all(row["trace_replay_ok"] for row in rows):
        raise ValueError("trace replay failed")

    primary = [
        row
        for row in rows
        if row["plan_length_stratum"] in ("long", "very_long")
        and 1 <= row["update_count"] <= 2
        and row["relevant_fraction"] <= 0.20
    ]
    methods = {}
    for method in METHODS:
        selected = [row for row in primary if row["method"] == method]
        methods[method] = {
            "n": len(selected),
            "success": mean(float(row["task_success"]) for row in selected),
            "actions": mean(row["counters"]["environment_actions"] for row in selected),
            "predicates": mean(row["counters"]["predicate_evaluations"] for row in selected),
            "successors": mean(row["counters"]["successor_expansions"] for row in selected),
            "positions": mean(row["counters"]["plan_position_visits"] for row in selected),
            "lookups": mean(row["counters"]["index_lookups"] for row in selected),
        }

    safety = {}
    for method in METHODS:
        selected = [
            row
            for row in rows
            if row["method"] == method and row["condition"] == "final_support_loss"
        ]
        safety[method] = {
            "n": len(selected),
            "unsupported_episodes": sum(row["unsupported_authorizations"] > 0 for row in selected),
        }
    if len({value["n"] for value in methods.values()}) != 1:
        raise ValueError("unpaired primary rows")
    if len({value["n"] for value in safety.values()}) != 1:
        raise ValueError("unpaired safety rows")
    return {
        "protocol_sha256": protocol_sha256,
        "rows": len(rows),
        "paired_cells": len(groups),
        "exact_replay_rate": 1.0,
        "primary": methods,
        "safety": safety,
    }


def render_tex(summary: dict) -> str:
    labels = {
        "full_replan": "Full replanning",
        "full_scan": "Complete scan",
        "receipt_index": r"\textbf{Receipt-indexed}",
        "unchecked_reuse": r"Unchecked reuse$^{\dagger}$",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{Full-episode mechanism audit. Means use 381 paired long/very-long episodes with one irrelevant evidence withdrawal; the final column uses 513 paired final-support-loss episodes. Timing is excluded.}",
        r"\label{tab:full-episode-v39}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & Success & Actions & Pred. & Succ. & Positions & Lookups & Unsupported \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = summary["primary"][method]
        unsupported = summary["safety"][method]["unsupported_episodes"]
        lines.append(
            f"{labels[method]} & {row['success']:.3f} & {row['actions']:.1f} & "
            f"{row['predicates']:.1f} & {row['successors']:.1f} & "
            f"{row['positions']:.1f} & {row['lookups']:.1f} & {unsupported}/513 \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{0.97\linewidth}{\scriptsize $^{\dagger}$Safety ablation: it does not check whether the next action remains supported. Pred./Succ. are predicate evaluations/planner-successor expansions; Positions are suffix positions inspected.}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_text() != content:
            raise AssertionError(f"stale generated artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = build_summary()
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    _write_or_check(OUTPUT_JSON, json_text, args.check)
    _write_or_check(OUTPUT_TEX, render_tex(summary), args.check)
    print(json_text, end="")


if __name__ == "__main__":
    main()
