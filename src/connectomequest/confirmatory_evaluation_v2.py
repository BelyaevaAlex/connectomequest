"""Immutable same-query active evaluation for the sealed NEmo v2 test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.confirmatory_v2 import (
    DATASETS,
    authorize_test_open,
    canonical_sha,
    confirmatory_root,
    load_method_selection,
    query_path,
    read_confirmatory_test_queries,
    verify_matrix,
    verify_seal,
)
from connectomequest.env import ConnectomeEnv
from connectomequest.evaluation import aggregate_metrics, episode_metrics, metrics_as_dict
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.policies.expert_gate_v2 import (
    ObservableRandomPolicy,
    ObservableWeightPolicy,
)
from connectomequest.policies.minerva import load_minerva_policy
from connectomequest.policies.oracle import PrivilegedOraclePolicy
from connectomequest.query import Query
from connectomequest.validation_v2 import load_v2_policy


def cell_stem(root: Path, cell: dict[str, Any]) -> Path:
    return (
        root
        / "active"
        / cell["held_out"]
        / f"{cell['method']}-b{cell['budget']}-seed{cell['seed']}"
    )


def _checkpoint_paths(repo: Path, held_out: str, seed: int) -> dict[str, Path]:
    bundle = DATASETS[held_out]
    return {
        "snapshot": repo / "outputs/nemo/v2/checkpoints" / f"heldout-{held_out}-seed{seed}.pt",
        "gate": repo / "outputs/nemo/v2/gates" / f"heldout-{held_out}-seed{seed}.pt",
        "minerva": (repo / "outputs/nemo/minerva/production" / f"leave-{bundle}-seed{seed}.pt"),
        "diverse": repo / "outputs/nemo/diverse" / f"{held_out}-mr-seed{seed}.pt",
    }


def load_cell_policy(
    repo: Path,
    cell: dict[str, Any],
    device: torch.device,
    graph: GraphStore | None = None,
    queries: list[Query] | None = None,
) -> tuple[Any, dict[str, str | None]]:
    held_out = cell["held_out"]
    seed = int(cell["seed"])
    method = cell["method"]
    paths = _checkpoint_paths(repo, held_out, seed)
    if method == "weight":
        return ObservableWeightPolicy(), {"builtin_policy": "weight"}
    if method == "degree":
        return None, {"builtin_policy": "affordance_degree"}
    if method == "oracle":
        if graph is None or queries is None:
            raise ValueError("PrivilegedOraclePolicy requires the graph and sealed queries")
        answers = {query.query_id: query.answers for query in queries}
        return PrivilegedOraclePolicy(graph, answers), {
            "builtin_policy": "privileged_oracle_ceiling",
        }
    if method == "random":
        return ObservableRandomPolicy(seed=seed), {"builtin_policy": f"query_seeded_random:{seed}"}
    if method == "snapshot_v2":
        return load_v2_policy(paths["snapshot"], held_out=held_out, seed=seed, device=device)
    if method == "gate_v2":
        return load_v2_policy(
            paths["snapshot"],
            held_out=held_out,
            seed=seed,
            device=device,
            gate_checkpoint=paths["gate"],
            minerva_checkpoint=paths["minerva"],
            diverse_checkpoint=paths["diverse"],
        )
    if method == "minerva":
        digest = sha256_file(paths["minerva"])
        payload = torch.load(paths["minerva"], map_location=device, weights_only=True)
        return load_minerva_policy(payload, device), {"minerva_checkpoint_sha256": digest}
    raise ValueError(f"unsupported confirmatory method: {method}")


def _existing_result(output: Path, run_sha256: str) -> dict[str, Any] | None:
    summary_path = output.with_suffix(".json")
    if not output.exists() and not summary_path.exists():
        return None
    if not output.is_file() or not summary_path.is_file():
        raise RuntimeError(f"partial confirmatory result exists: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol", {}).get("run_sha256") != run_sha256:
        raise RuntimeError(f"refusing to overwrite mismatched confirmatory result: {output}")
    if summary.get("result_sha256") != sha256_file(output):
        raise RuntimeError(f"confirmatory result hash mismatch: {output}")
    summary["reused"] = True
    return summary


def evaluate_cell(
    repo: Path,
    graph: GraphStore,
    queries: list[Query],
    cell: dict[str, Any],
    policy: Any,
    provenance: dict[str, str | None],
    *,
    device: torch.device,
    seal_sha256: str,
    method_selection_sha256: str,
) -> dict[str, Any]:
    root = confirmatory_root(repo)
    output = cell_stem(root, cell).with_suffix(".parquet").resolve()
    if not output.is_relative_to((root / "active").resolve()):
        raise ValueError("confirmatory result must remain below outputs/nemo/v2/confirmatory")
    controller = cell["controller"]
    adaptive = controller == "adaptive_zero_to_four_probes"
    run_spec: dict[str, Any] = {
        "protocol_version": 1,
        "partition": "confirmatory-test",
        "query_type": "2pt",
        "held_out": cell["held_out"],
        "method": cell["method"],
        "budget": int(cell["budget"]),
        "seed": int(cell["seed"]),
        "controller": controller,
        "access_regime": cell.get("access_regime", "observable"),
        "visible_neighbors": 32,
        "answer_quota": 1,
        "candidates_per_subgoal": 4,
        "subgoal_probes": 0 if adaptive else 1,
        "adaptive_probing": adaptive,
        "max_subgoal_probes": 4 if adaptive else 0,
        "probe_margin_threshold": 0.05 if adaptive else None,
        "probe_entropy_threshold": 0.95 if adaptive else None,
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "query_sha256": sha256_file(query_path(repo, cell["held_out"])),
        "seal_sha256": seal_sha256,
        "method_selection_sha256": method_selection_sha256,
        **provenance,
    }
    run_sha256 = canonical_sha(run_spec)
    existing = _existing_result(output, run_sha256)
    if existing is not None:
        return existing
    if hasattr(policy, "selection_counts"):
        policy.selection_counts.clear()
    if hasattr(policy, "fallback_count"):
        policy.fallback_count = 0
    explorer = BudgetedExplorer(
        policy=policy,
        device=device,
        ranking="affordance_degree" if cell["method"] == "degree" else "policy",
        seed=int(cell["seed"]),
        candidates_per_subgoal=4,
        subgoal_probes=0 if adaptive else 1,
        adaptive_probing=adaptive,
        max_subgoal_probes=4 if adaptive else 0,
        probe_margin_threshold=0.05,
        probe_entropy_threshold=0.95,
    )
    metrics = []
    rows = []
    for query in queries:
        env = ConnectomeEnv(
            graph,
            visible_neighbors=32,
            max_steps=int(cell["budget"]),
            budget=int(cell["budget"]),
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
                "split": "test",
                "ranking": cell["method"],
                "budget": int(cell["budget"]),
                "answer_quota": 1,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(output)
    stats = getattr(policy, "stats", None)
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


def shard_direction_cells(
    selection: dict[str, Any],
    held_out: str,
    *,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    """Partition a direction's sorted cells exactly once across workers."""

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    cells = sorted(
        (cell for cell in verify_matrix(selection) if cell["held_out"] == held_out),
        key=lambda cell: (cell["method"], int(cell["budget"]), int(cell["seed"])),
    )
    return [cell for position, cell in enumerate(cells) if position % num_shards == shard_index]


def run_direction(
    repo: Path,
    held_out: str,
    *,
    device_name: str = "cpu",
    dry_run: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    repo = repo.resolve()
    seal = verify_seal(repo, require_queries=not dry_run)
    selection = load_method_selection(repo)
    cells = shard_direction_cells(
        selection, held_out, shard_index=shard_index, num_shards=num_shards
    )
    planned = [str(cell_stem(confirmatory_root(repo), cell)) for cell in cells]
    if dry_run:
        return {
            "dry_run": True,
            "held_out": held_out,
            "cells": len(cells),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "outputs": planned,
            "seal_sha256": seal["seal_sha256"],
            "test_rows_opened": False,
        }
    authorize_test_open(repo, verified_seal=seal)
    graph = GraphStore(repo / "data/processed" / DATASETS[held_out])
    queries = read_confirmatory_test_queries(repo, held_out, graph, verified_seal=seal)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    cache: dict[tuple[str, int], tuple[Any, dict[str, str | None]]] = {}
    completed = []
    for cell in cells:
        key = (cell["method"], int(cell["seed"]))
        if key not in cache:
            cache[key] = load_cell_policy(repo, cell, device, graph=graph, queries=queries)
        policy, provenance = cache[key]
        completed.append(
            evaluate_cell(
                repo,
                graph,
                queries,
                cell,
                policy,
                provenance,
                device=device,
                seal_sha256=seal["seal_sha256"],
                method_selection_sha256=selection["selection_sha256"],
            )
        )
    return {
        "dry_run": False,
        "held_out": held_out,
        "cells": len(completed),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "reused_cells": sum(bool(summary["reused"]) for summary in completed),
        "test_queries": len(queries),
        "seal_sha256": seal["seal_sha256"],
    }
