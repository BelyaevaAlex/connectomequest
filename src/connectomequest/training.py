"""Memory-bounded static pretraining for the neural search heuristic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from connectomequest.graph import GraphStore
from connectomequest.models.torus import QUERY_TO_ID, TorusQueryHeuristic
from connectomequest.query import SEMANTIC_VISIBLE_NEIGHBORS, Query, QueryType
from connectomequest.schema import Relation


@dataclass(frozen=True, slots=True)
class SemanticTrainingIndex:
    """Weight-sorted observable wiring frontier used by policy supervision."""

    visible_successors: np.ndarray

    @classmethod
    def from_graph(
        cls,
        graph: GraphStore,
        *,
        visible_neighbors: int = SEMANTIC_VISIBLE_NEIGHBORS,
    ) -> SemanticTrainingIndex:
        wiring = graph.csr(Relation.PRESYNAPTIC_TO)
        visible = np.full((graph.num_nodes, visible_neighbors), -1, dtype=np.int64)
        for node in range(graph.num_nodes):
            start, end = wiring.indptr[node : node + 2]
            node_ids = wiring.indices[start:end]
            if not len(node_ids):
                continue
            weights = wiring.data[start:end]
            order = np.lexsort((node_ids, -weights))[:visible_neighbors]
            visible[node, : len(order)] = node_ids[order]
        return cls(visible)

    def successors(self, node: int) -> np.ndarray:
        values = self.visible_successors[node]
        return values[values >= 0]

    def final_hard_negatives(self, query: Query) -> list[int]:
        if query.query_type != QueryType.TWO_HOP_TYPE:
            return []
        first_hop = self.successors(query.anchors[0])
        if not len(first_hop):
            return []
        candidates = self.visible_successors[first_hop].reshape(-1)
        return sorted({int(node) for node in candidates if node >= 0 and node not in query.answers})

    def frontier_pairs(self, query: Query, *, limit: int = 8) -> list[tuple[int, int]]:
        if query.query_type != QueryType.TWO_HOP_TYPE:
            return []
        positives: list[int] = []
        negatives: list[int] = []
        for middle in self.successors(query.anchors[0]):
            reaches_answer = any(
                int(candidate) in query.answers for candidate in self.successors(int(middle))
            )
            (positives if reaches_answer else negatives).append(int(middle))
        if not positives or not negatives:
            return []
        return [
            (positives[index % len(positives)], negative)
            for index, negative in enumerate(negatives[:limit])
        ]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    query_type_ids: Tensor
    anchors: Tensor
    candidates: Tensor
    labels: Tensor
    frontier_query_type_ids: Tensor | None = None
    frontier_anchors: Tensor | None = None
    frontier_candidates: Tensor | None = None
    frontier_labels: Tensor | None = None


def _sample_filtered_negatives(
    query: Query,
    *,
    num_entities: int,
    count: int,
    generator: torch.Generator | None,
    hard_pool: list[int] | None = None,
    hard_fraction: float = 0.75,
) -> list[int]:
    """Sample entities that are not answers to the query.

    Rejection sampling avoids allocating a full entity complement for every
    query while remaining efficient for the current connectome answer density.
    """

    if len(query.answers) >= num_entities:
        raise ValueError("cannot sample a negative: every entity is an answer")
    available_hard = list(dict.fromkeys(hard_pool or []))
    if available_hard:
        order = torch.randperm(len(available_hard), generator=generator).tolist()
        hard_count = min(len(available_hard), round(count * hard_fraction))
        negatives = [available_hard[index] for index in order[:hard_count]]
    else:
        negatives = []
    forbidden = set(query.answers)
    while len(negatives) < count:
        needed = count - len(negatives)
        draws = torch.randint(
            0,
            num_entities,
            (max(64, needed * 2),),
            generator=generator,
        ).tolist()
        for value in draws:
            if value not in forbidden:
                negatives.append(value)
                if len(negatives) >= count:
                    break
    return negatives[:count]


def _query_tensors(rows: list[Query]) -> tuple[Tensor, Tensor]:
    query_type_ids = torch.tensor([QUERY_TO_ID[row.query_type] for row in rows])
    anchors = torch.zeros((len(rows), 2), dtype=torch.long)
    for index, row in enumerate(rows):
        anchors[index, : len(row.anchors)] = torch.tensor(row.anchors)
    return query_type_ids, anchors


def collate_queries(
    queries: list[Query],
    *,
    num_entities: int,
    negatives_per_positive: int = 32,
    max_positives_per_query: int = 16,
    generator: torch.Generator | None = None,
    semantic_index: SemanticTrainingIndex | None = None,
    frontier_negatives_per_query: int = 8,
) -> TrainingBatch:
    rows: list[Query] = []
    positives: list[int] = []
    for query in queries:
        answers = torch.tensor(sorted(query.answers), dtype=torch.long)
        if len(answers) > max_positives_per_query:
            selection = torch.randperm(len(answers), generator=generator)[:max_positives_per_query]
            answers = answers[selection]
        for answer in answers.tolist():
            rows.append(query)
            positives.append(answer)
    if not rows:
        raise ValueError("batch has no positive query answers")
    query_type_ids, anchors = _query_tensors(rows)
    hard_pools = (
        {query.query_id: semantic_index.final_hard_negatives(query) for query in queries}
        if semantic_index is not None
        else {}
    )
    positive = torch.tensor(positives, dtype=torch.long)[:, None]
    negative = torch.tensor(
        [
            _sample_filtered_negatives(
                row,
                num_entities=num_entities,
                count=negatives_per_positive,
                generator=generator,
                hard_pool=hard_pools.get(row.query_id),
            )
            for row in rows
        ],
        dtype=torch.long,
    )
    candidates = torch.cat((positive, negative), dim=1)
    labels = torch.zeros_like(candidates, dtype=torch.float32)
    labels[:, 0] = 1.0
    frontier_rows: list[Query] = []
    frontier_candidates: list[tuple[int, int]] = []
    if semantic_index is not None:
        for query in queries:
            pairs = semantic_index.frontier_pairs(
                query,
                limit=frontier_negatives_per_query,
            )
            frontier_rows.extend([query] * len(pairs))
            frontier_candidates.extend(pairs)
    if frontier_rows:
        frontier_query_type_ids, frontier_anchors = _query_tensors(frontier_rows)
        frontier_candidate_tensor = torch.tensor(frontier_candidates, dtype=torch.long)
        frontier_labels = torch.tensor([[1.0, 0.0]] * len(frontier_rows))
    else:
        frontier_query_type_ids = None
        frontier_anchors = None
        frontier_candidate_tensor = None
        frontier_labels = None
    return TrainingBatch(
        query_type_ids,
        anchors,
        candidates,
        labels,
        frontier_query_type_ids,
        frontier_anchors,
        frontier_candidate_tensor,
        frontier_labels,
    )


def train_step(
    model: TorusQueryHeuristic,
    batch: TrainingBatch,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    use_bf16: bool = True,
    frontier_loss_weight: float = 1.0,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    query_type_ids = batch.query_type_ids.to(device, non_blocking=True)
    anchors = batch.anchors.to(device, non_blocking=True)
    candidates = batch.candidates.to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16 and device.type == "cuda",
    ):
        logits = model(query_type_ids, anchors, candidates)
        loss = F.softplus(logits[:, 1:] - logits[:, :1]).mean()
        if batch.frontier_candidates is not None:
            assert batch.frontier_query_type_ids is not None
            assert batch.frontier_anchors is not None
            frontier_logits = model.frontier_logits(
                batch.frontier_query_type_ids.to(device, non_blocking=True),
                batch.frontier_anchors.to(device, non_blocking=True),
                batch.frontier_candidates.to(device, non_blocking=True),
            )
            frontier_loss = F.softplus(frontier_logits[:, 1] - frontier_logits[:, 0]).mean()
            loss = loss + frontier_loss_weight * frontier_loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
