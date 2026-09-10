#!/usr/bin/env python3
"""Post-test paired controls: visible witnesses and first/second-hop attribution."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from run_controller_comparison import verify_postseal_inputs

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.confirmatory_evaluation_v2 import load_cell_policy
from connectomequest.confirmatory_v2 import DATASETS, query_path, read_confirmatory_test_queries
from connectomequest.embodied.metrics import paired_success_difference, write_once
from connectomequest.env import ConnectomeEnv
from connectomequest.evaluation import episode_metrics, metrics_as_dict
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.policies.oracle import PrivilegedOraclePolicy
from connectomequest.policies.review_controls import VisibleWitnessOracle, WeightFirstLearnedSecond

ROOT = REPO / "outputs/nemo/strong-accept/review-resolution-v1"
PRIMARY = (
    REPO / "outputs/nemo/strong-accept/connectome-controller-v2/budget_v32_zero_probe_b64_c4/active"
)


class AuditGraph(GraphStore):
    """Cache immutable relation names instead of rereading Parquet each episode."""

    @cached_property
    def relations(self):
        return super().relations


def protocol(graph, component, limit):
    return {
        "status": "post-test diagnostic",
        "component": component,
        "graph": graph,
        "query_sha256": sha256_file(query_path(REPO, graph)),
        "seeds": [17, 29, 43] if component == "gate" else [17],
        "limit": limit,
        "V": 32,
        "B": 64,
        "candidates_per_subgoal": 4,
        "controller": "zero_probe_full_frontier",
        "implementation_sha256": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in [
                Path(__file__),
                REPO / "src/connectomequest/policies/review_controls.py",
                REPO / "src/connectomequest/agents/explorer.py",
                REPO / "src/connectomequest/env.py",
                REPO / "src/connectomequest/proof.py",
            ]
        },
    }


def evaluate(policy, graph, query):
    env = ConnectomeEnv(
        graph, visible_neighbors=32, max_steps=64, budget=64, enforce_proof=True, answer_quota=1
    )
    env.reset(query)
    explorer = BudgetedExplorer(
        policy=policy,
        ranking="policy",
        device=torch.device("cpu"),
        candidates_per_subgoal=4,
        subgoal_probes=0,
    )
    return metrics_as_dict(episode_metrics(query, explorer.run(env, query.spec)))


def run(graph_name, component, limit):
    out = ROOT / (f"smoke-{limit}" if limit else "full") / graph_name / component
    out.mkdir(parents=True, exist_ok=True)
    with (out / "RUN.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal = verify_postseal_inputs(REPO)
        cfg = protocol(graph_name, component, limit)
        write_once(out / "PROTOCOL.json", cfg)
        if (out / "aggregate.json").exists():
            prior = json.loads((out / "aggregate.json").read_text())
            if prior["rows_sha256"] != sha256_file(out / "rows.parquet"):
                raise ValueError("completed result hash mismatch")
            print(json.dumps({"status": "reused", "output": str(out)}), flush=True)
            return
        graph = AuditGraph(REPO / "data/processed" / DATASETS[graph_name])
        queries = read_confirmatory_test_queries(REPO, graph_name, graph, verified_seal=seal)
        if limit:
            queries = queries[:limit]
        rows, counts, provenance = [], {}, {}
        started = time.monotonic()
        for seed in cfg["seeds"]:
            if component == "gate":
                gate, meta = load_cell_policy(
                    REPO,
                    {"held_out": graph_name, "seed": seed, "method": "gate_v2"},
                    torch.device("cpu"),
                )
                policies = {
                    "gate": gate,
                    "weight_first_idfree_second": WeightFirstLearnedSecond(
                        gate.experts[gate.second_hop_expert]
                    ),
                }
                provenance[str(seed)] = meta
            else:
                answers = {q.query_id: q.answers for q in queries}
                policies = {
                    "historical_oracle": PrivilegedOraclePolicy(graph, answers),
                    "visible_witness": VisibleWitnessOracle(graph, answers, 32),
                }
            for index, query in enumerate(queries):
                for method, policy in policies.items():
                    rows.append({"seed": seed, "method": method, **evaluate(policy, graph, query)})
                if (index + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "graph": graph_name,
                                "component": component,
                                "seed": seed,
                                "queries_done": index + 1,
                                "queries_total": len(queries),
                                "elapsed_seconds": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            if component == "gate":
                counts[str(seed)] = gate.stats()
        pq.write_table(pa.Table.from_pylist(rows), out / "rows.parquet", compression="zstd")
        summary = {}
        for method in policies:
            selected = [r for r in rows if r["method"] == method]
            summary[method] = {
                "episodes": len(selected),
                "success": float(np.mean([r["goal_success"] for r in selected])),
                "budget_used": float(np.mean([r["budget_used"] for r in selected])),
                "max_success_budget": max(
                    (r["budget_used"] for r in selected if r["goal_success"]), default=None
                ),
            }
        paired = {}
        left, right = list(policies)
        aligned = {}
        for r in rows:
            aligned.setdefault((r["seed"], r["query_id"]), {})[r["method"]] = r
        by_query = {}
        for (_, qid), r in aligned.items():
            by_query.setdefault(qid, []).append(
                float(r[right]["goal_success"]) - float(r[left]["goal_success"])
            )
        diffs = np.array([np.mean(d) for d in by_query.values()])
        rng = np.random.default_rng(172903)
        boots = [float(np.mean(rng.choice(diffs, len(diffs)))) for _ in range(5000)]
        paired = {
            "contrast": f"{right} minus {left}",
            "difference": float(diffs.mean()),
            "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
            "query_units": len(diffs),
            "different_success_pairs": sum(
                r[left]["goal_success"] != r[right]["goal_success"] for r in aligned.values()
            ),
            "different_budget_pairs": sum(
                r[left]["budget_used"] != r[right]["budget_used"] for r in aligned.values()
            ),
        }
        primary_check = {}
        if component == "gate" and not limit:
            for seed in cfg["seeds"]:
                path = PRIMARY / graph_name / f"gate_v2-b64-seed{seed}.parquet"
                orig = {r["query_id"]: r for r in pq.read_table(path).to_pylist()}
                new = [r for r in rows if r["seed"] == seed and r["method"] == "gate"]
                primary_check[str(seed)] = {
                    "source_sha256": sha256_file(path),
                    "success_mismatches": sum(
                        r["goal_success"] != orig[r["query_id"]]["goal_success"] for r in new
                    ),
                    "budget_mismatches": sum(
                        r["budget_used"] != orig[r["query_id"]]["budget_used"] for r in new
                    ),
                }
        result = {
            "protocol": cfg,
            "rows_sha256": sha256_file(out / "rows.parquet"),
            "checkpoint_provenance": provenance,
            "summary": summary,
            "paired": paired,
            "gate_calls": counts,
            "primary_reproduction": primary_check,
            "wall_seconds": time.monotonic() - started,
        }
        write_once(out / "aggregate.json", result)
        print(
            json.dumps(
                {"status": "complete", "output": str(out), "summary": summary, "paired": paired}
            ),
            flush=True,
        )


def pose_audit():
    root = REPO / "outputs/nemo/strong-accept/embodied-pose-aware-v1"
    aggregate = json.loads((root / "aggregate.json").read_text())
    rows = []
    for name, expected in aggregate["input_sha256"].items():
        path = REPO / name
        if sha256_file(path) != expected:
            raise ValueError("pose shard changed")
        rows.extend(pq.read_table(path).to_pylist())
    result = {"source_sha256": sha256_file(root / "aggregate.json"), "paired_pose_minus_local": {}}
    for size in (8, 12):
        selected = [r for r in rows if r["size"] == size]
        result["paired_pose_minus_local"][str(size)] = paired_success_difference(
            selected,
            left="semantic5_occlusion10_odometry2_local",
            right="semantic5_occlusion10_odometry2_pose_aware",
            seed=20260905,
            bootstrap_samples=5000,
        )
    write_once(ROOT / "pose-versus-local.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", choices=DATASETS, default="h01")
    p.add_argument("--component", choices=("oracle", "gate", "pose"), required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.component == "pose":
        pose_audit()
    else:
        run(a.graph, a.component, a.limit)
