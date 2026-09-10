"""Shared ranking metrics over immutable DecisionSnapshot candidate sets."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from connectomequest.decision_benchmark import DecisionSnapshot, DecisionStage
from connectomequest.evaluation import (
    aggregate_frontier_ranks,
    filtered_ranks,
    raw_ranks,
)
from connectomequest.manifest import sha256_file
from connectomequest.rankers.base import CandidateRanker, ordered_candidates


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _aggregate_stage(rows: list[dict]) -> dict:
    total = len(rows)
    covered_rows = [row for row in rows if row["covered"]]
    query_ids = {row["query_id"] for row in rows}
    covered_query_ids = {row["query_id"] for row in covered_rows}
    if covered_rows:
        ranking = aggregate_frontier_ranks(
            [rank for row in covered_rows for rank in row["raw_ranks"]],
            [rank for row in covered_rows for rank in row["filtered_ranks"]],
            [row["first_proof_rank"] for row in covered_rows],
            candidate_counts=[row["candidate_count"] for row in covered_rows],
            relevant_counts=[row["relevant_count"] for row in covered_rows],
        )
    else:
        ranking = {
            "queries": 0.0,
            "relevant_middles": 0.0,
            "mean_candidates": 0.0,
            "mean_relevant_middles": 0.0,
            "raw": {},
            "filtered": {},
            "first_proof": {},
        }
    latencies = [row["latency_ms"] for row in rows]
    return {
        "states": total,
        "covered_states": len(covered_rows),
        "state_coverage": len(covered_rows) / total if total else 0.0,
        "queries": len(query_ids),
        "covered_queries": len(covered_query_ids),
        "query_coverage": len(covered_query_ids) / len(query_ids) if query_ids else 0.0,
        "conditional_ranking": ranking,
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
    }


def evaluate_decision_snapshots(
    snapshots: list[DecisionSnapshot],
    ranker: CandidateRanker,
    output: Path,
    *,
    stage: DecisionStage | None = None,
    warmup: int = 1,
) -> dict:
    """Rank exactly the candidates stored in each snapshot.

    States without a visible correct candidate contribute to coverage but not to
    conditional MRR/Hits. This prevents difficult branches from disappearing.
    """

    selected = [snapshot for snapshot in snapshots if stage is None or snapshot.stage == stage]
    if not selected:
        raise ValueError("no decision snapshots selected")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    cuda_device = getattr(ranker, "device", None)
    if (
        isinstance(cuda_device, torch.device)
        and cuda_device.type == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.reset_peak_memory_stats(cuda_device)
    prepare_fn = getattr(ranker, "prepare", None)
    preparation_started = time.perf_counter()
    if callable(prepare_fn):
        prepare_fn(selected)
    preparation_ms = (time.perf_counter() - preparation_started) * 1000.0
    for _ in range(warmup):
        snapshot = selected[0]
        ranker.score(snapshot.query, snapshot.observation, list(snapshot.candidates))

    rows: list[dict] = []
    for snapshot in selected:
        candidates = list(snapshot.candidates)
        started = time.perf_counter()
        scores = ranker.score(
            snapshot.query,
            snapshot.observation,
            candidates,
        )
        ordered = ordered_candidates(candidates, scores)
        latency_ms = (time.perf_counter() - started) * 1000.0
        relevant = snapshot.relevant_candidates
        raw = raw_ranks(ordered, relevant) if relevant else []
        filtered = filtered_ranks(ordered, relevant) if relevant else []
        rows.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "query_id": snapshot.query.query_id,
                "held_out": snapshot.held_out,
                "stage": str(snapshot.stage),
                "branch_middle": snapshot.branch_middle,
                "ranker": ranker.name,
                "access_regime": str(ranker.access_regime),
                "candidate_count": len(candidates),
                "relevant_count": len(relevant),
                "covered": bool(relevant),
                "first_proof_rank": min(raw) if raw else None,
                "raw_ranks": raw,
                "filtered_ranks": filtered,
                "latency_ms": latency_ms,
            }
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(output)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["stage"]].append(row)
    peak_memory = 0
    if (
        isinstance(cuda_device, torch.device)
        and cuda_device.type == "cuda"
        and torch.cuda.is_available()
    ):
        peak_memory = torch.cuda.max_memory_allocated(cuda_device)
    summary = {
        "ranker": ranker.name,
        "access_regime": str(ranker.access_regime),
        "result_path": str(output),
        "snapshot_result_sha256": sha256_file(output),
        "stages": {
            stage_name: _aggregate_stage(stage_rows)
            for stage_name, stage_rows in sorted(grouped.items())
        },
        "peak_gpu_memory_bytes": peak_memory,
        "preparation_ms": preparation_ms,
        "amortized_preparation_ms_per_state": preparation_ms / len(selected),
        "metric_protocol": {
            "candidate_space": "exact frozen visible candidates",
            "coverage": "all states; zero when no relevant visible candidate",
            "ranking_metrics": "conditional on at least one visible relevant candidate",
            "filtering": "remove other relevant visible candidates per target",
            "relevance_visibility": "evaluator only",
            "unmeasured_warmup_calls": warmup,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
