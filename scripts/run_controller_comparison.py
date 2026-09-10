#!/usr/bin/env python3
"""Post-seal strong-accept controller/reference audits on frozen v2 queries.

This intentionally writes outside outputs/nemo/v2/confirmatory so the sealed
primary matrix remains immutable.  It reuses the same ConnectomeEnv,
BudgetedExplorer, query snapshots, and checkpoint hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Graph parquet access is I/O-bound; prevent one audit process from spawning a
# large Arrow pool and starving sibling post-seal cells.
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.confirmatory_evaluation_v2 import load_cell_policy
from connectomequest.confirmatory_v2 import (
    DATASETS,
    _validate_generated_query,
    canonical_sha,
    query_path,
    query_paths,
    read_confirmatory_test_queries,
    seal_path,
    test_open_path,
)
from connectomequest.env import ConnectomeEnv
from connectomequest.evaluation import aggregate_metrics, episode_metrics, metrics_as_dict
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.policies.expert_gate_v2 import ObservableRandomPolicy
from connectomequest.policies.snapshot_v2 import load_snapshot_policy_v2
from connectomequest.policies.voi import FirstHopRandomSecondHopMinervaPolicy
from connectomequest.query import Query

ROOT = REPO / "outputs/nemo/strong-accept/connectome-controller-v2"
DEFAULT_METHODS = ("weight", "random", "snapshot_v2", "gate_v2", "minerva")
DEFAULT_SEEDS = (17, 29, 43)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(name: str) -> torch.device:
    return torch.device(name if name != "cuda" or torch.cuda.is_available() else "cpu")


def _variant_stem(
    root: Path, variant: str, held_out: str, method: str, budget: int, seed: int
) -> Path:
    return root / variant / "active" / held_out / f"{method}-b{budget}-seed{seed}"


def _existing(path: Path, run_sha256: str) -> dict[str, Any] | None:
    summary = path.with_suffix(".json")
    parquet = path.with_suffix(".parquet")
    if not parquet.exists() and not summary.exists():
        return None
    if not parquet.is_file() or not summary.is_file():
        raise RuntimeError(f"partial audit result exists: {path}")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("protocol", {}).get("run_sha256") != run_sha256:
        raise RuntimeError(f"refusing to reuse mismatched audit result: {path}")
    if payload.get("result_sha256") != sha256_file(parquet):
        raise RuntimeError(f"audit result hash mismatch: {parquet}")
    payload["reused"] = True
    return payload


def _load_reference_snapshot(path: Path, device: torch.device) -> tuple[Any, dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("model_kind") != "entity_id_free_snapshot_policy_v2":
        raise ValueError(f"not a SnapshotV2 checkpoint: {path}")
    policy = load_snapshot_policy_v2(payload, device)
    protocol = payload.get("protocol", {}) if isinstance(payload, dict) else {}
    return policy, {
        "snapshot_checkpoint_sha256": digest,
        "checkpoint_protocol_sha256": payload.get("protocol_sha256"),
        "checkpoint_target_connectome_validation": protocol.get("target_connectome_validation"),
        "checkpoint_transductive_reference": protocol.get("transductive_reference"),
    }


class FirstHopRandomSecondHopSnapshotPolicy:
    """Hybrid audit policy: random frontier branch order, learned second-hop order."""

    name = "random_snapshot_v2"

    def __init__(self, random_policy: Any, snapshot_policy: Any) -> None:
        self.random_policy = random_policy
        self.snapshot_policy = snapshot_policy

    def rank(
        self,
        query: Any,
        observation: Any,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
        chunk_size: int = 65_536,
    ) -> list[int]:
        if frontier:
            return self.random_policy.rank(
                query,
                observation,
                candidates,
                frontier=frontier,
                device=device,
                chunk_size=chunk_size,
            )
        return self.snapshot_policy.rank(
            query, observation, candidates, frontier=frontier, device=device, chunk_size=chunk_size
        )


def _load_policy(
    repo: Path,
    held_out: str,
    method: str,
    seed: int,
    device: torch.device,
    graph: GraphStore,
    queries: list[Query],
    *,
    checkpoint_root: Path | None,
    gate_root: Path | None,
    checkpoint_template: str,
    allow_reference_checkpoint: bool,
    minerva_checkpoint_seed: int | None = None,
) -> tuple[Any, dict[str, str | None], str]:
    if method in {"snapshot_v2", "random_snapshot_v2"} and (
        checkpoint_root is not None or allow_reference_checkpoint
    ):
        root = checkpoint_root or (repo / "outputs/nemo/v2/checkpoints")
        path = root / checkpoint_template.format(held_out=held_out, seed=seed)
        if allow_reference_checkpoint:
            policy, provenance = _load_reference_snapshot(path, device)
            if method == "random_snapshot_v2":
                return (
                    FirstHopRandomSecondHopSnapshotPolicy(
                        ObservableRandomPolicy(seed=seed), policy
                    ),
                    {
                        **provenance,
                        "builtin_policy": f"random_first_hop_snapshot_second_hop:{seed}",
                    },
                    "policy",
                )
            return policy, provenance, "policy"
    if method == "voi_random_minerva":
        checkpoint_seed = int(
            minerva_checkpoint_seed if minerva_checkpoint_seed is not None else seed
        )
        minerva_cell = {
            "held_out": held_out,
            "method": "minerva",
            "budget": 64,
            "seed": checkpoint_seed,
        }
        minerva, provenance = load_cell_policy(
            repo, minerva_cell, device, graph=graph, queries=queries
        )
        return (
            FirstHopRandomSecondHopMinervaPolicy(ObservableRandomPolicy(seed=seed), minerva),
            {
                **provenance,
                "builtin_policy": f"random_first_hop_minerva_second_hop:{seed}",
                "action_seed": str(seed),
                "minerva_checkpoint_seed": str(checkpoint_seed),
            },
            "policy",
        )
    if method == "random_snapshot_v2":
        snapshot_cell = {"held_out": held_out, "method": "snapshot_v2", "budget": 64, "seed": seed}
        snapshot, provenance = load_cell_policy(
            repo, snapshot_cell, device, graph=graph, queries=queries
        )
        return (
            FirstHopRandomSecondHopSnapshotPolicy(ObservableRandomPolicy(seed=seed), snapshot),
            {**provenance, "builtin_policy": f"random_first_hop_snapshot_second_hop:{seed}"},
            "policy",
        )
    cell = {"held_out": held_out, "method": method, "budget": 64, "seed": seed}
    policy, provenance = load_cell_policy(repo, cell, device, graph=graph, queries=queries)
    if method == "degree":
        return policy, provenance, "affordance_degree"
    return policy, provenance, "policy"


def verify_postseal_inputs(repo: Path) -> dict[str, Any]:
    """Verify immutable seal/query identities while permitting new audit code.

    The primary seal intentionally freezes the research code before test
    generation.  Post-seal audits may add new policy modules; they therefore
    verify the seal digest, query manifests, graph validation, and one-time
    open marker, but do not pretend that the new audit code was pre-registered.
    """
    repo = repo.resolve()
    path = seal_path(repo)
    if not path.is_file():
        raise FileNotFoundError(path)
    seal = json.loads(path.read_text(encoding="utf-8"))
    recorded = seal.get("seal_sha256")
    unsigned = dict(seal)
    unsigned.pop("seal_sha256", None)
    if recorded != canonical_sha(unsigned):
        raise ValueError("v2 confirmatory seal digest is invalid")
    if seal.get("artifacts_sha256") != canonical_sha(seal.get("artifacts", {})):
        raise ValueError("v2 confirmatory artifact-map digest is invalid")
    for held_out, path in query_paths(repo).items():
        _validate_generated_query(repo, held_out, path)
    marker_path = test_open_path(repo)
    if not marker_path.is_file():
        raise PermissionError("post-seal audit requires the existing one-time test-open marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_digest = marker.get("authorization_sha256")
    marker_unsigned = dict(marker)
    marker_unsigned.pop("authorization_sha256", None)
    if marker_digest != canonical_sha(marker_unsigned):
        raise ValueError("test-open marker digest is invalid")
    if marker.get("seal_sha256") != seal.get("seal_sha256"):
        raise ValueError("test-open marker does not pin the frozen seal")
    for held_out, path in query_paths(repo).items():
        if marker.get("query_sha256", {}).get(held_out) != sha256_file(path):
            raise ValueError("test-open marker does not pin query snapshot")
    return seal


def run_cell(args: argparse.Namespace, held_out: str, method: str, seed: int) -> dict[str, Any]:
    repo = REPO.resolve()
    seal = verify_postseal_inputs(repo)
    graph = GraphStore(repo / "data/processed" / DATASETS[held_out])
    queries = read_confirmatory_test_queries(repo, held_out, graph, verified_seal=seal)
    # Graph traversal is CPU-bound; cap PyTorch thread pools explicitly.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = _device(args.device)
    policy, provenance, ranking = _load_policy(
        repo,
        held_out,
        method,
        seed,
        device,
        graph,
        queries,
        checkpoint_root=args.checkpoint_root,
        gate_root=args.gate_root,
        checkpoint_template=args.checkpoint_template,
        allow_reference_checkpoint=args.allow_reference_checkpoint,
        minerva_checkpoint_seed=args.minerva_checkpoint_seed,
    )
    run_spec: dict[str, Any] = {
        "protocol_version": 1,
        "partition": "post-seal-strong-accept-audit",
        "variant": args.variant,
        "held_out": held_out,
        "method": method,
        "seed": seed,
        "minerva_checkpoint_seed": args.minerva_checkpoint_seed
        if method == "voi_random_minerva"
        else None,
        "budget": args.budget,
        "visible_neighbors": args.visible_neighbors,
        "answer_quota": 1,
        "candidates_per_subgoal": args.candidates_per_subgoal,
        "subgoal_probes": args.subgoal_probes,
        "adaptive_probing": args.adaptive_probing,
        "max_subgoal_probes": args.max_subgoal_probes if args.adaptive_probing else 0,
        "probe_margin_threshold": args.probe_margin_threshold if args.adaptive_probing else None,
        "probe_entropy_threshold": args.probe_entropy_threshold if args.adaptive_probing else None,
        "access_regime": "privileged_ceiling" if method == "oracle" else "observable",
        "seal_sha256": seal["seal_sha256"],
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "query_sha256": sha256_file(query_path(repo, held_out)),
        "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
        "gate_root": str(args.gate_root) if args.gate_root else None,
        "checkpoint_template": args.checkpoint_template if args.checkpoint_root else None,
        "allow_reference_checkpoint": args.allow_reference_checkpoint,
        **provenance,
    }
    run_sha = canonical_sha(run_spec)
    stem = _variant_stem(ROOT, args.variant, held_out, method, args.budget, seed)
    if args.dry_run:
        return {"dry_run": True, "stem": str(stem), "protocol": {**run_spec, "run_sha256": run_sha}}
    existing = _existing(stem, run_sha)
    if existing is not None:
        return existing
    if hasattr(policy, "selection_counts"):
        policy.selection_counts.clear()
    if hasattr(policy, "fallback_count"):
        policy.fallback_count = 0
    explorer = BudgetedExplorer(
        policy=policy,
        ranking=ranking,
        device=device,
        seed=seed,
        candidates_per_subgoal=args.candidates_per_subgoal,
        subgoal_probes=args.subgoal_probes,
        adaptive_probing=args.adaptive_probing,
        max_subgoal_probes=args.max_subgoal_probes if args.adaptive_probing else 0,
        probe_margin_threshold=args.probe_margin_threshold,
        probe_entropy_threshold=args.probe_entropy_threshold,
    )
    rows = []
    metrics = []
    for query in queries:
        env = ConnectomeEnv(
            graph,
            visible_neighbors=args.visible_neighbors,
            max_steps=args.budget,
            budget=args.budget,
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
                "ranking": method,
                "budget": args.budget,
                "answer_quota": 1,
            }
        )
    stem.parent.mkdir(parents=True, exist_ok=True)
    parquet = stem.with_suffix(".parquet")
    tmp = parquet.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    tmp.replace(parquet)
    stats = getattr(policy, "stats", None)
    summary = {
        "overall": aggregate_metrics(metrics),
        "result_path": str(parquet),
        "result_sha256": sha256_file(parquet),
        "protocol": {**run_spec, "run_sha256": run_sha},
        "policy_stats": stats() if callable(stats) else None,
        "reused": False,
    }
    tmp_json = stem.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_json.replace(stem.with_suffix(".json"))
    return summary


def aggregate(root: Path, variant: str) -> dict[str, Any]:
    base = root / variant / "active"
    records = []
    hashes = {}
    for summary_path in sorted(base.glob("*/*.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        proto = payload["protocol"]
        parquet = Path(payload["result_path"])
        if payload["result_sha256"] != sha256_file(parquet):
            raise ValueError(f"hash mismatch: {parquet}")
        records.append(
            {
                "held_out": proto["held_out"],
                "method": proto["method"],
                "seed": int(proto["seed"]),
                "budget": int(proto["budget"]),
                "metrics": payload["overall"],
                "policy_stats": payload.get("policy_stats"),
            }
        )
        hashes[str(summary_path.relative_to(REPO))] = sha256_file(summary_path)
        hashes[str(parquet.relative_to(REPO))] = sha256_file(parquet)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["held_out"], record["method"], record["budget"])].append(record)
    rows = []
    for (held_out, method, budget), items in sorted(grouped.items()):
        items.sort(key=lambda r: r["seed"])
        metric_names = tuple(items[0]["metrics"])
        metrics = {}
        for metric in metric_names:
            vals = [float(item["metrics"][metric]) for item in items]
            metrics[metric] = {
                "mean": mean(vals),
                "sample_sd": stdev(vals) if len(vals) > 1 else None,
                "n_seeds": len(vals),
            }
        rows.append(
            {
                "held_out": held_out,
                "method": method,
                "budget": budget,
                "seeds": [r["seed"] for r in items],
                "metrics": metrics,
            }
        )
    payload = {
        "variant": variant,
        "schema_version": 1,
        "rows": rows,
        "input_fingerprint_sha256": canonical_sha(sorted(hashes.items())),
        "input_sha256": hashes,
    }
    out = root / variant / "aggregate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        f"# Controller audit: {variant}",
        "",
        "| Held out | Method | Seeds | Success | Budget used | First-proof MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]

        def fmt(*names: str) -> str:
            for name in names:
                if name in m:
                    v = m[name]["mean"]
                    sd = m[name]["sample_sd"]
                    return f"{v:.4f}" + (f" ({sd:.4f})" if sd is not None else "")
            available = ", ".join(sorted(m))
            raise KeyError(f"None of {names!r} found in metrics; available: {available}")

        md.append(
            f"| {row['held_out']} | {row['method']} | {','.join(map(str, row['seeds']))} | "
            f"{fmt('goal_success', 'mean_goal_success', 'success_at_budget')} | "
            f"{fmt('budget_used', 'mean_budget_used')} | "
            f"{fmt('first_proof_reciprocal_rank', 'mean_first_proof_reciprocal_rank')} |"
        )
    (root / variant / "aggregate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"aggregate": str(out), "rows": len(rows)}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="zero_probe_full_frontier_c4")
    parser.add_argument("--held-out", default="manc,hemibrain")
    parser.add_argument("--method", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--visible-neighbors", type=int, default=32)
    parser.add_argument("--candidates-per-subgoal", type=int, default=4)
    parser.add_argument("--subgoal-probes", type=int, default=0)
    parser.add_argument("--adaptive-probing", action="store_true")
    parser.add_argument("--max-subgoal-probes", type=int, default=4)
    parser.add_argument("--probe-margin-threshold", type=float, default=0.05)
    parser.add_argument("--probe-entropy-threshold", type=float, default=0.95)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--gate-root", type=Path)
    parser.add_argument("--checkpoint-template", default="heldout-{held_out}-seed{seed}.pt")
    parser.add_argument("--allow-reference-checkpoint", action="store_true")
    parser.add_argument(
        "--minerva-checkpoint-seed",
        type=int,
        help="VOI audit: use this frozen MINERVA checkpoint for every action seed",
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate(ROOT, args.variant), sort_keys=True))
        return
    held_outs = parse_csv(args.held_out)
    methods = parse_csv(args.method)
    seeds = [int(x) for x in parse_csv(args.seeds)]
    outputs = []
    for held_out in held_outs:
        if held_out not in DATASETS:
            raise ValueError(f"unknown held-out {held_out}")
        for method in methods:
            method_seeds = [17] if method in {"weight", "degree", "oracle"} else seeds
            for seed in method_seeds:
                outputs.append(run_cell(args, held_out, method, seed))
    print(
        json.dumps(
            {"cells": len(outputs), "dry_run": args.dry_run, "variant": args.variant},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
