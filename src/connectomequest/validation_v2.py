"""Leakage-safe development evaluation for the NEmo v2 active policies.

This module deliberately has no confirmatory/test entry point.  It reads only
the replication ``validation`` partition and writes below
``outputs/nemo/v2/validation``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.env import ConnectomeEnv
from connectomequest.evaluation import (
    aggregate_metrics,
    episode_metrics,
    metrics_as_dict,
)
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.policies.diverse import load_diverse_policy
from connectomequest.policies.expert_gate_v2 import (
    ObservableRandomPolicy,
    ObservableWeightPolicy,
)
from connectomequest.policies.minerva import load_minerva_policy
from connectomequest.policies.snapshot_v2 import load_snapshot_policy_v2
from connectomequest.query import Query, QueryType
from connectomequest.query_io import validate_query_snapshot

DATASETS = {
    "h01": "h01-20210729-c3",
    "manc": "manc-v1.0",
    "hemibrain": "hemibrain-v1.2.1",
}
SEEDS = (17, 29, 43)
BUDGETS = (16, 32, 64)


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validation_queries(path: Path, graph: GraphStore) -> list[Query]:
    """Read only replication-validation 2pt rows using Arrow pushdown."""

    validate_query_snapshot(path, graph)
    table = pq.read_table(
        path,
        columns=[
            "query_id",
            "query_type",
            "anchors",
            "answers",
            "split",
            "semantic_relation",
        ],
        filters=[("split", "=", "validation"), ("query_type", "=", "2pt")],
        memory_map=True,
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError("replication snapshot has no validation/2pt queries")
    if any(row["split"] != "validation" for row in rows):
        raise AssertionError("non-validation row escaped the Arrow filter")
    return [
        Query(
            query_id=row["query_id"],
            query_type=QueryType(row["query_type"]),
            anchors=tuple(row["anchors"]),
            answers=frozenset(row["answers"]),
            split="validation",
            semantic_relation=row.get("semantic_relation"),
        )
        for row in rows
    ]


def _checkpoint_payload(path: Path, device: torch.device) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload, digest


def _validate_snapshot_direction(payload: dict, held_out: str, seed: int) -> None:
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("SnapshotV2 checkpoint lacks its frozen protocol")
    if protocol.get("held_out") != held_out:
        raise ValueError("SnapshotV2 checkpoint belongs to another LOCO direction")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("SnapshotV2 checkpoint seed mismatch")
    if protocol.get("target_connectome_validation") is not False:
        raise ValueError("SnapshotV2 protocol is not source-validation-only")
    if protocol.get("immutable_test_access") is not False:
        raise ValueError("SnapshotV2 protocol does not explicitly deny test access")


def _metadata_digest(record: Any, *, expert: str) -> str | None:
    if isinstance(record, str):
        return record if len(record) == 64 else None
    if not isinstance(record, dict):
        return None
    for key in ("sha256", "checkpoint_sha256"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    if expert == "weight" and record.get("observable_builtin") == "weight":
        return "builtin:weight"
    if expert == "random" and record.get("observable_builtin") == "query_seeded_random":
        seed = record.get("seed")
        if isinstance(seed, int):
            return f"builtin:query_seeded_random:{seed}"
    return None


def _validate_gate_payload(
    payload: dict,
    *,
    held_out: str,
    seed: int,
    expert_digests: dict[str, str],
) -> None:
    if payload.get("model_kind") != "contextual_expert_gate_v2":
        raise ValueError("checkpoint is not a contextual expert gate v2")
    if payload.get("held_out") != held_out:
        raise ValueError("gate checkpoint belongs to another LOCO direction")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError("gate checkpoint seed mismatch")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("gate checkpoint lacks its source-only protocol")
    source_only = protocol.get("source_only") is True
    explicit_no_target = protocol.get("target_connectome_validation") is False
    explicit_no_test = (
        protocol.get("test_access") is False or protocol.get("immutable_test_access") is False
    )
    if not (source_only and explicit_no_target and explicit_no_test):
        raise ValueError("gate protocol must explicitly be source-only and test-blind")
    recorded = payload.get("expert_checkpoint_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("gate checkpoint lacks expert SHA-256 metadata")
    if tuple(payload.get("expert_names", ())) != tuple(expert_digests):
        raise ValueError("gate expert order does not match the loaded expert order")
    for name, actual in expert_digests.items():
        expected = _metadata_digest(recorded.get(name), expert=name)
        if expected != actual:
            raise ValueError(f"gate expert SHA-256 mismatch for {name!r}")


def load_v2_policy(
    snapshot_checkpoint: Path,
    *,
    held_out: str,
    seed: int,
    device: torch.device,
    gate_checkpoint: Path | None = None,
    minerva_checkpoint: Path | None = None,
    diverse_checkpoint: Path | None = None,
) -> tuple[Any, dict[str, str | None]]:
    """Load SnapshotV2 or a source-only contextual gate with pinned experts."""

    snapshot_payload, snapshot_sha = _checkpoint_payload(snapshot_checkpoint, device)
    _validate_snapshot_direction(snapshot_payload, held_out, seed)
    snapshot = load_snapshot_policy_v2(snapshot_payload, device)
    provenance: dict[str, str | None] = {
        "snapshot_checkpoint_sha256": snapshot_sha,
        "gate_checkpoint_sha256": None,
        "minerva_checkpoint_sha256": None,
        "diverse_checkpoint_sha256": None,
    }
    if gate_checkpoint is None:
        return snapshot, provenance

    gate_payload, gate_sha = _checkpoint_payload(gate_checkpoint, device)
    names = tuple(gate_payload.get("expert_names", ()))
    experts: dict[str, Any] = {}
    digests: dict[str, str] = {}
    minerva = None
    minerva_sha = None
    diverse = None
    diverse_sha = None
    for name in names:
        if name == "weight":
            experts[name] = ObservableWeightPolicy()
            digests[name] = "builtin:weight"
        elif name == "random":
            experts[name] = ObservableRandomPolicy(seed=seed)
            digests[name] = f"builtin:query_seeded_random:{seed}"
        elif name in {"snapshot", "snapshot_v2"}:
            experts[name] = snapshot
            digests[name] = snapshot_sha
        elif name == "minerva":
            if minerva_checkpoint is None:
                raise ValueError("gate requires --minerva-checkpoint")
            if minerva is None:
                minerva_payload, minerva_sha = _checkpoint_payload(minerva_checkpoint, device)
                minerva = load_minerva_policy(minerva_payload, device)
            experts[name] = minerva
            assert minerva_sha is not None
            digests[name] = minerva_sha
        elif name == "diverse":
            if diverse_checkpoint is None:
                raise ValueError("gate requires --diverse-checkpoint")
            if diverse is None:
                diverse_payload, diverse_sha = _checkpoint_payload(diverse_checkpoint, device)
                diverse = load_diverse_policy(diverse_payload, device)
            experts[name] = diverse
            assert diverse_sha is not None
            digests[name] = diverse_sha
        else:
            raise ValueError(f"unsupported gate expert {name!r}")
    _validate_gate_payload(
        gate_payload,
        held_out=held_out,
        seed=seed,
        expert_digests=digests,
    )
    from connectomequest.policies.expert_gate_v2 import load_contextual_expert_gate_v2

    policy = load_contextual_expert_gate_v2(gate_payload, experts)
    provenance.update(
        gate_checkpoint_sha256=gate_sha,
        minerva_checkpoint_sha256=minerva_sha,
        diverse_checkpoint_sha256=diverse_sha,
    )
    return policy, provenance


def _existing_result(output: Path, run_sha256: str) -> dict | None:
    summary_path = output.with_suffix(".json")
    if not output.exists() and not summary_path.exists():
        return None
    if not output.is_file() or not summary_path.is_file():
        raise RuntimeError(f"partial v2 result exists: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol", {}).get("run_sha256") != run_sha256:
        raise RuntimeError(f"refusing to overwrite mismatched v2 result: {output}")
    if summary.get("result_sha256") != sha256_file(output):
        raise RuntimeError(f"v2 result hash mismatch: {output}")
    summary["reused"] = True
    return summary


def evaluate_v2_validation(
    graph: GraphStore,
    queries: list[Query],
    policy: Any,
    output: Path,
    *,
    held_out: str,
    method: str,
    budget: int,
    seed: int,
    input_provenance: dict[str, str | None],
    query_path: Path,
    validation_root: Path,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    """Evaluate one immutable replication-validation cell, with hash skip guards."""

    output = output.resolve()
    validation_root = validation_root.resolve()
    if not output.is_relative_to(validation_root):
        raise ValueError("v2 validation output must remain under outputs/nemo/v2/validation")
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {BUDGETS}")
    selected = queries[:limit] if limit is not None else queries
    if not selected or any(query.split != "validation" for query in selected):
        raise ValueError("v2 development evaluator accepts validation rows only")
    adaptive = True
    run_spec: dict[str, Any] = {
        "protocol_version": 1,
        "held_out": held_out,
        "method": method,
        "partition": "replication-validation",
        "query_type": "2pt",
        "visible_neighbors": 32,
        "budget": budget,
        "answer_quota": 1,
        "candidates_per_subgoal": 4,
        "subgoal_probes": 0,
        "adaptive_probing": adaptive,
        "max_subgoal_probes": 4,
        "probe_margin_threshold": 0.05,
        "probe_entropy_threshold": 0.95,
        "seed": seed,
        "limit": limit,
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "query_sha256": sha256_file(query_path),
        **input_provenance,
    }
    run_sha256 = _canonical_sha(run_spec)
    existing = _existing_result(output, run_sha256)
    if existing is not None:
        return existing

    stats = getattr(policy, "stats", None)
    if hasattr(policy, "selection_counts"):
        policy.selection_counts.clear()
    if hasattr(policy, "fallback_count"):
        policy.fallback_count = 0
    explorer = BudgetedExplorer(
        policy=policy,
        device=device,
        ranking="policy",
        seed=seed,
        candidates_per_subgoal=4,
        subgoal_probes=0,
        adaptive_probing=True,
        max_subgoal_probes=4,
        probe_margin_threshold=0.05,
        probe_entropy_threshold=0.95,
    )
    rows = []
    metrics = []
    for query in selected:
        env = ConnectomeEnv(
            graph,
            visible_neighbors=32,
            max_steps=budget,
            budget=budget,
            enforce_proof=True,
            answer_quota=1,
        )
        env.reset(query)
        episode = episode_metrics(query, explorer.run(env, query.spec))
        metrics.append(episode)
        rows.append(
            {
                **metrics_as_dict(episode),
                "query_type": "2pt",
                "split": "validation",
                "ranking": method,
                "budget": budget,
                "answer_quota": 1,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(output)
    summary = {
        "overall": aggregate_metrics(metrics),
        "result_path": str(output),
        "result_sha256": sha256_file(output),
        "protocol": {**run_spec, "run_sha256": run_sha256},
        "policy_stats": stats() if callable(stats) else None,
        "reused": False,
    }
    summary_path = output.with_suffix(".json")
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    return summary


def baseline_reuse_manifest(repo_root: Path, validation_root: Path) -> dict:
    """Index old validation artifacts instead of rerunning baseline episodes."""

    old_root = repo_root / "outputs/nemo/active-validation/main-matrix"
    records = []
    for held_out, bundle in DATASETS.items():
        graph_root = repo_root / "data/processed" / bundle
        expected_graph_sha = sha256_file(graph_root / "manifest.json")
        expected_query_sha = sha256_file(graph_root / "queries-loco-replication-seed2026.parquet")
        for budget in BUDGETS:
            choices = [("weight", 17)]
            choices.extend((method, seed) for method in ("random", "minerva") for seed in SEEDS)
            for method, seed in choices:
                stem = old_root / f"{held_out}-b{budget}-{method}-seed{seed}"
                parquet_path = stem.with_suffix(".parquet")
                summary_path = stem.with_suffix(".json")
                if not parquet_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"missing reusable baseline cell: {stem}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                protocol = summary.get("protocol", {})
                expected_protocol = {
                    "split": "validation",
                    "query_type": "2pt",
                    "ranking": method,
                    "budget": budget,
                    "seed": seed,
                    "graph_manifest_sha256": expected_graph_sha,
                    "query_sha256": expected_query_sha,
                }
                if any(protocol.get(key) != value for key, value in expected_protocol.items()):
                    raise ValueError(f"baseline is not replication-validation/2pt: {stem}")
                records.append(
                    {
                        "held_out": held_out,
                        "budget": budget,
                        "method": method,
                        "seed": seed,
                        "parquet": str(parquet_path),
                        "parquet_sha256": sha256_file(parquet_path),
                        "summary": str(summary_path),
                        "summary_sha256": sha256_file(summary_path),
                    }
                )
    payload = {
        "policy": "reference existing immutable validation outputs; do not rerun",
        "cells": len(records),
        "records": records,
    }
    validation_root.mkdir(parents=True, exist_ok=True)
    path = validation_root / "baseline-reuse.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError("baseline reuse manifest changed; refusing silent overwrite")
    else:
        temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    return payload
