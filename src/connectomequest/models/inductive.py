"""Entity-ID-free neural policy for cross-connectome transfer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from connectomequest.env import Observation
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec, QueryType

QUERY_TO_ID = {query_type: index for index, query_type in enumerate(QueryType)}
FEATURE_DIM = 26


def candidate_features(
    observation: Observation,
    query: QuerySpec,
    candidates: list[int],
    *,
    frontier: bool,
) -> Tensor:
    """Build bounded structural features using only the observed evidence graph.

    Entity IDs are used solely as equality keys. Their numeric values never enter
    the feature tensor, so the representation is invariant to node relabeling.
    """

    if not candidates:
        return torch.empty((0, FEATURE_DIM), dtype=torch.float32)
    incoming_count: dict[int, int] = {}
    outgoing_count: dict[int, int] = {}
    incoming_weight: dict[int, float] = {}
    max_weight: dict[int, float] = {}
    incoming_non_edge: dict[int, int] = {}
    from_current: set[int] = set()
    to_current: set[int] = set()
    from_anchor: set[int] = set()
    semantic_match: set[int] = set()
    anchors = set(query.anchors)
    current_edge_weight: dict[int, float] = {}
    semantic_target = query.anchors[1] if len(query.anchors) > 1 else None

    for evidence in observation.discovered_edges:
        if evidence.kind == EvidenceKind.NON_EDGE:
            incoming_non_edge[evidence.dst] = incoming_non_edge.get(evidence.dst, 0) + 1
            continue
        incoming_count[evidence.dst] = incoming_count.get(evidence.dst, 0) + 1
        outgoing_count[evidence.src] = outgoing_count.get(evidence.src, 0) + 1
        weight = max(0.0, float(evidence.weight))
        incoming_weight[evidence.dst] = incoming_weight.get(evidence.dst, 0.0) + weight
        max_weight[evidence.dst] = max(max_weight.get(evidence.dst, 0.0), weight)
        if evidence.src == observation.current_node:
            from_current.add(evidence.dst)
            current_edge_weight[evidence.dst] = max(
                current_edge_weight.get(evidence.dst, 0.0), weight
            )
        if evidence.dst == observation.current_node:
            to_current.add(evidence.src)
        if evidence.src in anchors:
            from_anchor.add(evidence.dst)
        if (
            semantic_target is not None
            and query.semantic_relation is not None
            and evidence.relation == query.semantic_relation
            and evidence.dst == semantic_target
        ):
            semantic_match.add(evidence.src)

    memory_nodes = {node for _, node in observation.memory}
    sketches = {sketch.node_id: sketch for sketch in observation.node_sketches}
    ordered = sorted(
        candidates,
        key=lambda node: (-current_edge_weight.get(node, 0.0), node),
    )
    ranks = {node: rank for rank, node in enumerate(ordered)}
    current_max = max((current_edge_weight.get(node, 0.0) for node in candidates), default=0.0)
    current_sum = sum(current_edge_weight.get(node, 0.0) for node in candidates)
    rank_denominator = max(1, len(candidates) - 1)
    horizon = max(1.0, observation.remaining_budget + observation.step)
    budget_fraction = float(observation.remaining_budget / horizon)
    step_fraction = float(observation.step / horizon)
    rows: list[list[float]] = []
    for candidate in candidates:
        count = incoming_count.get(candidate, 0)
        local_weight = current_edge_weight.get(candidate, 0.0)
        sketch = sketches.get(candidate)
        rows.append(
            [
                math.log1p(count),
                math.log1p(outgoing_count.get(candidate, 0)),
                math.log1p(incoming_weight.get(candidate, 0.0)),
                math.log1p(max_weight.get(candidate, 0.0)),
                count / max(1.0, count + incoming_non_edge.get(candidate, 0)),
                math.log1p(incoming_non_edge.get(candidate, 0)),
                float(candidate == observation.current_node),
                float(candidate in memory_nodes),
                float(bool(query.anchors) and candidate == query.anchors[0]),
                float(len(query.anchors) > 1 and candidate == query.anchors[1]),
                float(candidate in from_current),
                float(candidate in to_current),
                float(candidate in from_anchor),
                float(candidate in semantic_match),
                budget_fraction,
                step_fraction,
                float(frontier),
                local_weight / max(current_max, 1e-8),
                local_weight / max(current_sum, 1e-8),
                1.0 - ranks[candidate] / rank_denominator,
                float(ranks[candidate] < 4),
                float(sketch is not None),
                math.log1p(sketch.outgoing_degree) if sketch else 0.0,
                math.log1p(sketch.total_weight) if sketch else 0.0,
                math.log1p(sketch.max_weight) if sketch else 0.0,
                math.log1p(sketch.mean_weight) if sketch else 0.0,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


class InductiveQueryPolicy(nn.Module):
    """Small structural policy shared across datasets and entity vocabularies."""

    def __init__(
        self,
        hidden_dim: int = 128,
        query_dim: int = 32,
        dropout: float = 0.1,
        feature_dim: int = FEATURE_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.dropout = dropout
        self.feature_dim = feature_dim
        self.query_embedding = nn.Embedding(len(QueryType), query_dim)
        self.weight_scale_raw = nn.Parameter(torch.zeros(()))
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim + query_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.scorer[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        features: Tensor,
        query_type_ids: Tensor,
        semantic_flags: Tensor,
    ) -> Tensor:
        query_vector = self.query_embedding(query_type_ids)
        inputs = torch.cat((features, query_vector, semantic_flags[:, None]), dim=-1)
        residual = self.scorer(inputs).squeeze(-1)
        weight_scale = 0.1 + torch.nn.functional.softplus(self.weight_scale_raw)
        return residual + weight_scale * features[:, 3]

    @torch.inference_mode()
    def rank(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
        chunk_size: int = 65_536,
    ) -> list[int]:
        if not candidates:
            return []
        self.eval()
        scored: list[Tensor] = []
        query_id = QUERY_TO_ID[query.query_type]
        semantic_flag = float(query.semantic_relation is not None)
        for start in range(0, len(candidates), chunk_size):
            selected = candidates[start : start + chunk_size]
            features = candidate_features(
                observation,
                query,
                selected,
                frontier=frontier,
            ).to(device, non_blocking=True)
            features = features[:, : self.feature_dim]
            query_ids = torch.full(
                (len(selected),),
                query_id,
                dtype=torch.long,
                device=device,
            )
            semantic_flags = torch.full(
                (len(selected),),
                semantic_flag,
                dtype=torch.float32,
                device=device,
            )
            scored.append(self(features, query_ids, semantic_flags).float().cpu())
        scores = torch.cat(scored)
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index]),
        )
        return [candidates[index] for index in order]
