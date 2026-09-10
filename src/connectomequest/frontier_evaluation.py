"""Partial-observation ranking evaluation for active two-hop proof search."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from connectomequest.env import Action, ActionType, ConnectomeEnv
from connectomequest.evaluation import aggregate_frontier_ranks, filtered_ranks, raw_ranks
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.policies.rank_fusion import RankFusionPolicy
from connectomequest.query import QueryType
from connectomequest.query_io import read_queries, validate_query_snapshot
from connectomequest.runner import load_inductive_policy


def frontier_relevant_candidates(
    graph: GraphStore,
    query_answers: frozenset[int],
    candidates: list[int],
    *,
    answer_mask: np.ndarray | None = None,
) -> frozenset[int]:
    """Return visible middle nodes that can complete a two-hop proof."""

    if answer_mask is None:
        answer_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    if answer_mask.shape != (graph.num_nodes,) or answer_mask.dtype != np.bool_:
        raise ValueError("answer_mask must be a graph-sized boolean vector")
    answer_ids = np.fromiter(query_answers, dtype=np.int64, count=len(query_answers))
    answer_mask[answer_ids] = True
    wiring = graph.csr("presynaptic_to")
    relevant: set[int] = set()
    try:
        for candidate in candidates:
            start, end = wiring.indptr[candidate : candidate + 2]
            if answer_mask[wiring.indices[start:end]].any():
                relevant.add(candidate)
    finally:
        answer_mask[answer_ids] = False
    return frozenset(relevant)


def evaluate_frontier_ranking(
    graph: GraphStore,
    query_path: Path,
    output: Path,
    *,
    split: str = "test",
    ranking: str = "weight",
    checkpoint: Path | None = None,
    fusion_alpha: float = 0.5,
    visible_neighbors: int = 32,
    budget: int = 64,
    subgoal_probes: int = 0,
    ranking_scope: str = "operational",
    offset: int = 0,
    limit: int | None = None,
    seed: int = 17,
    device_name: str = "cpu",
) -> dict:
    """Evaluate raw/filtered ranks on the observable first-hop frontier.

    A middle is relevant iff at least one ground-truth answer is reachable via
    its second wiring edge. Ground truth labels metrics only and never enters
    the policy observation.
    """

    validate_query_snapshot(query_path, graph)
    queries = [
        query
        for query in read_queries(query_path)
        if query.split == split and query.query_type == QueryType.TWO_HOP_TYPE
    ]
    queries = queries[offset:]
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError(f"query file contains no {split}/2pt queries")
    supported = {"weight", "random", "inductive", "inductive_fusion"}
    if ranking not in supported:
        raise ValueError(f"unsupported frontier ranking: {ranking}")
    if ranking in {"inductive", "inductive_fusion"} and checkpoint is None:
        raise ValueError("inductive frontier evaluation requires checkpoint")

    if ranking_scope not in {"operational", "scorer", "legacy_probed_subset"}:
        raise ValueError("ranking_scope must be operational, scorer, or legacy_probed_subset")

    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    policy = None
    if checkpoint is not None:
        learned = load_inductive_policy(checkpoint, device)
        policy = (
            RankFusionPolicy(learned, fusion_alpha) if ranking == "inductive_fusion" else learned
        )
    rng = random.Random(seed)
    relation = "presynaptic_to"
    answer_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    raw_all: list[int] = []
    filtered_all: list[int] = []
    first_all: list[int] = []
    candidate_counts: list[int] = []
    relevant_counts: list[int] = []
    rows: list[dict] = []

    with torch.inference_mode():
        for query in tqdm(queries, desc=f"frontier ranks {ranking}"):
            env = ConnectomeEnv(
                graph,
                visible_neighbors=visible_neighbors,
                max_steps=budget,
                budget=budget,
                answer_quota=1,
            )
            env.reset(query)
            step = env.step(Action(ActionType.INSPECT, relation=relation))
            candidates = [
                evidence.dst
                for evidence in step.observation.discovered_edges
                if evidence.src == query.anchors[0] and evidence.relation == relation
            ]
            for candidate in candidates[:subgoal_probes]:
                step = env.step(
                    Action(ActionType.PROBE_AFFORDANCE, relation=relation, target=candidate)
                )
                if step.error:
                    raise RuntimeError(f"{query.query_id}: {step.error}")
            observation = env.observe()
            if ranking == "weight":
                ordered = candidates
            elif ranking == "random":
                ordered = candidates.copy()
                rng.shuffle(ordered)
            else:
                assert policy is not None
                policy_candidates = (
                    candidates[:subgoal_probes]
                    if ranking_scope == "legacy_probed_subset" and subgoal_probes
                    else candidates
                )
                ordered = policy.rank(
                    query.spec,
                    observation,
                    policy_candidates,
                    frontier=True,
                    device=device,
                )
                if len(policy_candidates) < len(candidates):
                    probed = set(policy_candidates)
                    ordered.extend(candidate for candidate in candidates if candidate not in probed)

            relevant = frontier_relevant_candidates(
                graph, query.answers, candidates, answer_mask=answer_mask
            )
            if not relevant:
                raise ValueError(f"{query.query_id} has no useful middle in visible frontier")
            raw = raw_ranks(ordered, relevant)
            filtered = filtered_ranks(ordered, relevant)
            first = min(raw)
            raw_all.extend(raw)
            filtered_all.extend(filtered)
            first_all.append(first)
            candidate_counts.append(len(candidates))
            relevant_counts.append(len(relevant))
            rows.append(
                {
                    "query_id": query.query_id,
                    "ranking": ranking,
                    "candidate_count": len(candidates),
                    "relevant_count": len(relevant),
                    "first_proof_rank": first,
                    "raw_ranks": raw,
                    "filtered_ranks": filtered,
                }
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    summary = {
        "overall": aggregate_frontier_ranks(
            raw_all,
            filtered_all,
            first_all,
            candidate_counts=candidate_counts,
            relevant_counts=relevant_counts,
        ),
        "protocol": {
            "status": "post_hoc_diagnostic_on_frozen_test_snapshot",
            "split": split,
            "query_type": "2pt",
            "ranking": ranking,
            "candidate_space": "weight-sorted observable first-hop frontier",
            "ranking_scope": ranking_scope,
            "relevance": "visible middle reaches at least one ground-truth answer",
            "filtering": "all other relevant visible middles removed per target middle",
            "policy_observation": "initial inspect plus configured paid affordance probes",
            "visible_neighbors": visible_neighbors,
            "budget": budget,
            "subgoal_probes": subgoal_probes,
            "fusion_alpha": fusion_alpha if ranking == "inductive_fusion" else None,
            "offset": offset,
            "seed": seed,
            "graph_manifest_sha256": sha256_file(graph.manifest_path),
            "query_sha256": sha256_file(Path(query_path)),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            "device": str(device),
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
