"""Torus-valued query regions used as a learned search heuristic."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from connectomequest.query import QueryType

QUERY_TO_ID = {query_type: index for index, query_type in enumerate(QueryType)}


def wrap_angle(value: Tensor) -> Tensor:
    """Map angles into [-pi, pi) without discontinuous parameter clamping."""

    return torch.remainder(value + math.pi, 2 * math.pi) - math.pi


def torus_distance(left: Tensor, right: Tensor) -> Tensor:
    delta = torch.abs(wrap_angle(left - right))
    return torch.minimum(delta, 2 * math.pi - delta)


def circular_mean(values: Tensor, dim: int) -> Tensor:
    return torch.atan2(torch.sin(values).mean(dim=dim), torch.cos(values).mean(dim=dim))


class TorusQueryHeuristic(nn.Module):
    """Query-conditioned entity scorer with closed toroidal operations.

    The model is deliberately a heuristic, not the source of logical validity.
    The symbolic controller remains responsible for legal transitions and proofs.
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int = 256,
        *,
        semantic_policy: bool = True,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.semantic_policy = semantic_policy
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.projection = nn.Parameter(torch.empty(embedding_dim))
        self.type_projection = nn.Parameter(torch.empty(embedding_dim))
        self.query_bias = nn.Embedding(len(QueryType), embedding_dim)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        hidden_dim = max(32, embedding_dim // 2)
        if semantic_policy:
            self.semantic_candidate_head = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.semantic_frontier_head = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.semantic_candidate_head = None
            self.semantic_frontier_head = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.entity.weight, -math.pi, math.pi)
        nn.init.uniform_(self.projection, -0.1, 0.1)
        nn.init.uniform_(self.type_projection, -0.1, 0.1)
        nn.init.zeros_(self.query_bias.weight)

    def query_centers(
        self,
        query_type_ids: Tensor,
        anchors: Tensor,
        anchor_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return positive centers and optional negated centers.

        anchors has shape [batch, max_anchors]. Padded values are ignored by
        anchor_mask. Current query language requires at most two anchors.
        """

        embedded = self.entity(anchors)
        projected = wrap_angle(embedded + self.projection)
        batch = anchors.shape[0]
        positive = torch.empty(
            batch,
            self.embedding_dim,
            device=anchors.device,
            dtype=embedded.dtype,
        )
        negative: Tensor | None = None
        for query_type, query_id in QUERY_TO_ID.items():
            selected = query_type_ids == query_id
            if not selected.any():
                continue
            current = projected[selected]
            if query_type == QueryType.ONE_HOP:
                center = current[:, 0]
            elif query_type == QueryType.TWO_HOP:
                center = wrap_angle(current[:, 0] + self.projection)
            elif query_type == QueryType.TWO_HOP_TYPE:
                path_center = wrap_angle(current[:, 0] + self.projection)
                semantic_center = wrap_angle(embedded[selected, 1] - self.type_projection)
                center = circular_mean(torch.stack((path_center, semantic_center), dim=1), dim=1)
            elif query_type in (QueryType.INTERSECTION, QueryType.INTERSECTION_TYPE):
                center = circular_mean(current[:, :2], dim=1)
                if query_type == QueryType.INTERSECTION_TYPE:
                    center = wrap_angle(center + self.type_projection)
            elif query_type == QueryType.INTERSECTION_NEGATION:
                center = current[:, 0]
                if negative is None:
                    negative = torch.zeros_like(positive)
                negative[selected] = current[:, 1]
            else:
                raise ValueError(query_type)
            positive[selected] = center
        bias = self.query_bias(query_type_ids)
        return wrap_angle(positive + bias), negative

    def _semantic_logits(
        self,
        head: nn.Sequential,
        candidate_embedding: Tensor,
        type_anchors: Tensor,
    ) -> Tensor:
        type_center = wrap_angle(self.entity(type_anchors) - self.type_projection)
        if candidate_embedding.ndim == 3:
            type_center = type_center[:, None, :]
        delta = wrap_angle(candidate_embedding - type_center)
        periodic_features = torch.cat((torch.sin(delta), torch.cos(delta)), dim=-1)
        return head(periodic_features).squeeze(-1)

    def forward(
        self,
        query_type_ids: Tensor,
        anchors: Tensor,
        candidates: Tensor,
        anchor_mask: Tensor | None = None,
    ) -> Tensor:
        positive, negative = self.query_centers(query_type_ids, anchors, anchor_mask)
        candidate_embedding = self.entity(candidates)
        if candidates.ndim == 2:
            positive = positive[:, None, :]
            if negative is not None:
                negative = negative[:, None, :]
        score = -torus_distance(candidate_embedding, positive).mean(dim=-1)
        negation_rows = query_type_ids == QUERY_TO_ID[QueryType.INTERSECTION_NEGATION]
        semantic_rows = query_type_ids == QUERY_TO_ID[QueryType.TWO_HOP_TYPE]
        if self.semantic_candidate_head is not None and semantic_rows.any():
            score[semantic_rows] += self._semantic_logits(
                self.semantic_candidate_head,
                candidate_embedding[semantic_rows],
                anchors[semantic_rows, 1],
            )
        if negative is not None and negation_rows.any():
            negative_distance = torus_distance(candidate_embedding, negative).mean(dim=-1)
            score[negation_rows] += negative_distance[negation_rows]
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        return score / temperature

    def frontier_logits(
        self,
        query_type_ids: Tensor,
        anchors: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """Score intermediate actions after projecting them one step forward."""

        positive, negative = self.query_centers(query_type_ids, anchors)
        candidate_embedding = self.entity(candidates).clone()
        if candidates.ndim == 2:
            positive = positive[:, None, :]
            if negative is not None:
                negative = negative[:, None, :]
        two_hop = (query_type_ids == QUERY_TO_ID[QueryType.TWO_HOP]) | (
            query_type_ids == QUERY_TO_ID[QueryType.TWO_HOP_TYPE]
        )
        if two_hop.any():
            candidate_embedding[two_hop] = wrap_angle(
                candidate_embedding[two_hop] + self.projection
            )
        intersection_type = query_type_ids == QUERY_TO_ID[QueryType.INTERSECTION_TYPE]
        if intersection_type.any():
            candidate_embedding[intersection_type] = wrap_angle(
                candidate_embedding[intersection_type] + self.type_projection
            )
        score = -torus_distance(candidate_embedding, positive).mean(dim=-1)
        negation_rows = query_type_ids == QUERY_TO_ID[QueryType.INTERSECTION_NEGATION]
        if negative is not None and negation_rows.any():
            score[negation_rows] += torus_distance(
                candidate_embedding[negation_rows], negative[negation_rows]
            ).mean(dim=-1)
        semantic_rows = query_type_ids == QUERY_TO_ID[QueryType.TWO_HOP_TYPE]
        if self.semantic_frontier_head is not None and semantic_rows.any():
            score[semantic_rows] += self._semantic_logits(
                self.semantic_frontier_head,
                candidate_embedding[semantic_rows],
                anchors[semantic_rows, 1],
            )
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        return score / temperature

    @torch.inference_mode()
    def rank_candidates(
        self,
        query_type: QueryType,
        anchors: tuple[int, ...],
        candidates: list[int],
        *,
        device: torch.device,
        chunk_size: int = 65_536,
    ) -> list[int]:
        if not candidates:
            return []
        query_ids = torch.tensor([QUERY_TO_ID[query_type]], device=device)
        anchor_tensor = torch.tensor([anchors], device=device)
        scored: list[tuple[float, int]] = []
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            candidate_tensor = torch.tensor([chunk], device=device)
            scores = self(query_ids, anchor_tensor, candidate_tensor)[0].float().cpu().tolist()
            scored.extend(zip(scores, chunk, strict=True))
        return [node for _, node in sorted(scored, reverse=True)]

    @torch.inference_mode()
    def rank_frontier(
        self,
        query_type: QueryType,
        anchors: tuple[int, ...],
        candidates: list[int],
        *,
        device: torch.device,
        chunk_size: int = 65_536,
    ) -> list[int]:
        """Rank action candidates after projecting them to expected final answers."""

        if not candidates:
            return []
        query_ids = torch.tensor([QUERY_TO_ID[query_type]], device=device)
        anchor_tensor = torch.tensor([anchors], device=device)
        scored: list[tuple[float, int]] = []
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            candidate_tensor = torch.tensor([chunk], device=device)
            scores = self.frontier_logits(query_ids, anchor_tensor, candidate_tensor)
            values = scores[0].float().cpu().tolist()
            scored.extend(zip(values, chunk, strict=True))
        return [node for _, node in sorted(scored, reverse=True)]
