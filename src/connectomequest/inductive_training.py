"""Memory-bounded behavior cloning for the entity-ID-free exploration policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from connectomequest.env import NodeSketch, Observation
from connectomequest.graph import GraphStore
from connectomequest.models.inductive import QUERY_TO_ID, InductiveQueryPolicy, candidate_features
from connectomequest.proof import Evidence, EvidenceKind
from connectomequest.query import Query, QueryType
from connectomequest.schema import Relation


@dataclass(frozen=True, slots=True)
class PairBatch:
    positive_features: Tensor
    negative_features: Tensor
    query_type_ids: Tensor
    semantic_flags: Tensor

    def __len__(self) -> int:
        return len(self.query_type_ids)


class ObservableIndex:
    """Weight-sorted V-neighborhoods with one shared CSR allocation."""

    def __init__(self, graph: GraphStore, visible_neighbors: int = 32):
        self.wiring = graph.csr(Relation.PRESYNAPTIC_TO)
        self.visible_neighbors = visible_neighbors

    def neighbors(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.wiring.indptr[node : node + 2]
        node_ids = self.wiring.indices[start:end]
        weights = self.wiring.data[start:end]
        if not len(node_ids):
            return node_ids, weights
        order = np.lexsort((node_ids, -weights))[: self.visible_neighbors]
        return node_ids[order], weights[order]

    def sketch(self, node: int) -> NodeSketch:
        start, end = self.wiring.indptr[node : node + 2]
        weights = self.wiring.data[start:end]
        degree = len(weights)
        total = float(weights.sum()) if degree else 0.0
        maximum = float(weights.max()) if degree else 0.0
        return NodeSketch(
            node_id=node,
            outgoing_degree=degree,
            total_weight=total,
            max_weight=maximum,
            mean_weight=total / degree if degree else 0.0,
            source_record="training_affordance_probe",
        )


def _observation(
    query: Query,
    current: int,
    edges: list[tuple[int, int, float]],
    *,
    remaining_budget: float = 60.0,
    step: int = 4,
    node_sketches: tuple[NodeSketch, ...] = (),
) -> Observation:
    relation = str(Relation.PRESYNAPTIC_TO)
    evidence = tuple(
        Evidence(EvidenceKind.EDGE, src, relation, dst, weight=float(weight))
        for src, dst, weight in edges
    )
    return Observation(
        current_node=current,
        discovered_edges=evidence,
        memory=tuple(enumerate(query.anchors[:8])),
        remaining_budget=remaining_budget,
        step=step,
        node_sketches=node_sketches,
    )


def _utility_pairs(
    candidates: list[int],
    utilities: dict[int, int],
    *,
    max_pairs: int,
) -> list[tuple[int, int]]:
    if len(candidates) < 2:
        return []
    ordered = sorted(candidates, key=lambda node: (-utilities.get(node, 0), node))
    best_value = utilities.get(ordered[0], 0)
    worse = [node for node in reversed(ordered) if utilities.get(node, 0) < best_value]
    best = [node for node in ordered if utilities.get(node, 0) == best_value]
    if not best or not worse:
        return []
    return [
        (best[index % len(best)], worse[index % len(worse)])
        for index in range(min(max_pairs, max(len(best), len(worse))))
    ]


def query_pair_batch(
    index: ObservableIndex,
    queries: list[Query],
    *,
    pairs_per_query: int = 8,
    subgoal_probes: int = 8,
) -> PairBatch | None:
    """Create oracle-labeled pairs from observations legal under the active protocol."""

    positive_rows: list[Tensor] = []
    negative_rows: list[Tensor] = []
    query_ids: list[int] = []
    semantic_flags: list[float] = []
    for query in queries:
        if query.query_type not in {
            QueryType.TWO_HOP,
            QueryType.TWO_HOP_TYPE,
            QueryType.INTERSECTION_NEGATION,
        }:
            continue
        first_ids, first_weights = index.neighbors(query.anchors[0])
        candidates = [int(node) for node in first_ids]
        edges = [
            (query.anchors[0], int(dst), float(weight))
            for dst, weight in zip(first_ids, first_weights, strict=True)
        ]
        probed = tuple(index.sketch(node) for node in candidates[:subgoal_probes])
        observation = _observation(
            query,
            query.anchors[0],
            edges,
            node_sketches=probed,
        )
        feature_rows = candidate_features(observation, query.spec, candidates, frontier=True)
        feature_by_node = dict(zip(candidates, feature_rows, strict=True))
        utilities: dict[int, int] = {}
        if query.query_type == QueryType.INTERSECTION_NEGATION:
            utilities = {candidate: int(candidate in query.answers) for candidate in candidates}
        else:
            for middle in candidates:
                successors, _ = index.neighbors(middle)
                utilities[middle] = sum(int(node) in query.answers for node in successors)
        pairs = _utility_pairs(candidates, utilities, max_pairs=pairs_per_query)
        for positive, negative in pairs:
            positive_rows.append(feature_by_node[positive])
            negative_rows.append(feature_by_node[negative])
            query_ids.append(QUERY_TO_ID[query.query_type])
            semantic_flags.append(float(query.semantic_relation is not None))

        if query.query_type != QueryType.TWO_HOP_TYPE:
            continue
        useful_middles = [node for node in candidates if utilities.get(node, 0) > 0]
        for middle in useful_middles[: max(1, pairs_per_query // 2)]:
            second_ids, second_weights = index.neighbors(middle)
            second = [int(node) for node in second_ids]
            second_edges = edges + [
                (middle, int(dst), float(weight))
                for dst, weight in zip(second_ids, second_weights, strict=True)
            ]
            second_observation = _observation(
                query,
                middle,
                second_edges,
                remaining_budget=56.0,
                step=8,
            )
            second_feature_rows = candidate_features(
                second_observation, query.spec, second, frontier=False
            )
            second_features = dict(zip(second, second_feature_rows, strict=True))
            final_utilities = {node: int(node in query.answers) for node in second}
            final_pairs = _utility_pairs(
                second,
                final_utilities,
                max_pairs=max(1, pairs_per_query // 2),
            )
            for positive, negative in final_pairs:
                positive_rows.append(second_features[positive])
                negative_rows.append(second_features[negative])
                query_ids.append(QUERY_TO_ID[query.query_type])
                semantic_flags.append(1.0)

    if not positive_rows:
        return None
    return PairBatch(
        torch.stack(positive_rows),
        torch.stack(negative_rows),
        torch.tensor(query_ids, dtype=torch.long),
        torch.tensor(semantic_flags, dtype=torch.float32),
    )


def train_inductive_epoch(
    model: InductiveQueryPolicy,
    graph: GraphStore,
    queries: list[Query],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_queries: int = 256,
    pairs_per_query: int = 8,
    visible_neighbors: int = 32,
    subgoal_probes: int = 8,
    seed: int = 17,
    use_bf16: bool = True,
) -> tuple[float, int]:
    """Train one epoch while keeping only one query minibatch of features in RAM."""

    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    index = ObservableIndex(graph, visible_neighbors)
    losses: list[float] = []
    num_pairs = 0
    model.train()
    for start in range(0, len(order), batch_queries):
        selected = [queries[idx] for idx in order[start : start + batch_queries]]
        batch = query_pair_batch(
            index,
            selected,
            pairs_per_query=pairs_per_query,
            subgoal_probes=subgoal_probes,
        )
        if batch is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        positive = batch.positive_features.to(device, non_blocking=True)
        negative = batch.negative_features.to(device, non_blocking=True)
        query_type_ids = batch.query_type_ids.to(device, non_blocking=True)
        semantic_flags = batch.semantic_flags.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            positive_scores = model(positive, query_type_ids, semantic_flags)
            negative_scores = model(negative, query_type_ids, semantic_flags)
            loss = F.softplus(negative_scores - positive_scores).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        num_pairs += len(batch)
    if not losses:
        raise ValueError("no informative observable ranking pairs were generated")
    return float(np.mean(losses)), num_pairs
