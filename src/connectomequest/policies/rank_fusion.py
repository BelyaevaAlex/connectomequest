"""Validation-calibrated fusion of learned and observable weight rankings."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from connectomequest.env import Observation
from connectomequest.policies.base import FrontierPolicy
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec


@dataclass(slots=True)
class RankFusionPolicy:
    """Convex rank fusion with an exact weight-only fallback at alpha=0."""

    learned: FrontierPolicy
    alpha: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("fusion alpha must be in [0, 1]")

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
        learned_order = self.learned.rank(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
            chunk_size=chunk_size,
        )
        candidate_set = set(candidates)
        weights: dict[int, float] = {}
        for evidence in observation.discovered_edges:
            if evidence.kind != EvidenceKind.NON_EDGE and evidence.dst in candidate_set:
                weights[evidence.dst] = max(weights.get(evidence.dst, 0.0), evidence.weight)
        weight_order = sorted(
            candidates,
            key=lambda node: (weights.get(node, 0.0), -node),
            reverse=True,
        )
        learned_rank = {node: rank for rank, node in enumerate(learned_order)}
        weight_rank = {node: rank for rank, node in enumerate(weight_order)}
        return sorted(
            candidates,
            key=lambda node: (
                self.alpha * learned_rank[node] + (1.0 - self.alpha) * weight_rank[node],
                node,
            ),
        )
