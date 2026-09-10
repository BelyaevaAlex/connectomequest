from __future__ import annotations

import torch

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.decision_benchmark import (
    DecisionStage,
    build_query_snapshots,
    read_decision_snapshots,
    write_decision_snapshots,
)
from connectomequest.decision_evaluation import evaluate_decision_snapshots
from connectomequest.env import ConnectomeEnv
from connectomequest.manifest import sha256_file
from connectomequest.query import Query, QueryType
from connectomequest.rankers import ObservableOracleRanker, WeightRanker
from connectomequest.toy import make_toy_graph


def test_operational_probe_scores_complete_frontier(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    calls: list[list[int]] = []

    class SpyPolicy:
        def rank(
            self,
            query,
            observation,
            candidates,
            *,
            frontier,
            device,
            chunk_size=65_536,
        ):
            del query, observation, frontier, device, chunk_size
            calls.append(list(candidates))
            return list(candidates)

    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=20)
    env.reset(query)
    BudgetedExplorer(
        policy=SpyPolicy(),
        ranking="policy",
        subgoal_probes=1,
        device=torch.device("cpu"),
    ).run(env, query.spec)
    assert calls[0] == [2, 3]


def test_snapshot_round_trip_and_independent_branches(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    graph_hash = sha256_file(graph.manifest_path)
    snapshots = build_query_snapshots(
        graph,
        query,
        held_out="toy",
        graph_manifest_sha256=graph_hash,
        query_sha256="query-hash",
        visible_neighbors=8,
        subgoal_probes=1,
    )
    first = [item for item in snapshots if item.stage == DecisionStage.FIRST_HOP]
    second = [item for item in snapshots if item.stage == DecisionStage.SECOND_HOP]
    assert len(first) == 1
    assert first[0].candidates == (2, 3)
    assert first[0].relevant_candidates == frozenset({2, 3})
    assert {item.branch_middle for item in second} == {2, 3}
    assert all(item.observation.current_node == item.branch_middle for item in second)
    assert all(item.candidates == (4,) for item in second)

    path = tmp_path / "snapshots.parquet"
    manifest = write_decision_snapshots(snapshots, path)
    restored = read_decision_snapshots(path)
    assert manifest["num_snapshots"] == 3
    assert restored == snapshots


def test_common_evaluator_reports_coverage_separately(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    snapshots = build_query_snapshots(
        graph,
        query,
        held_out="toy",
        graph_manifest_sha256=sha256_file(graph.manifest_path),
        query_sha256="query-hash",
        visible_neighbors=8,
    )
    output = tmp_path / "weight.parquet"
    summary = evaluate_decision_snapshots(
        snapshots,
        WeightRanker(),
        output,
    )
    assert summary["access_regime"] == "observable"
    assert summary["stages"]["first_hop"]["state_coverage"] == 1.0
    assert summary["stages"]["second_hop"]["state_coverage"] == 1.0

    oracle = ObservableOracleRanker.from_snapshots(snapshots)
    oracle_summary = evaluate_decision_snapshots(
        snapshots,
        oracle,
        tmp_path / "oracle.parquet",
    )
    assert oracle_summary["access_regime"] == "privileged_ceiling"
    assert (
        oracle_summary["stages"]["first_hop"]["conditional_ranking"]["first_proof"]["hits_at_1"]
        == 1.0
    )
