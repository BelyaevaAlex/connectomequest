"""Observable per-query gate with an exact weight-order fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from connectomequest.env import Observation
from connectomequest.models.inductive_v5 import InductiveQueryPolicyV5, PolicyUncertainty
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec


@dataclass(slots=True)
class ContextualGatePolicy:
    learned: InductiveQueryPolicyV5
    alpha: float = 0.5
    min_margin: float = 0.05
    max_entropy: float = 0.95
    max_ood: float = 3.0
    enabled_count: int = field(default=0, init=False)
    fallback_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

    def uncertainty(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> PolicyUncertainty:
        return self.learned.uncertainty(
            query, observation, candidates, frontier=frontier, device=device
        )

    def _enabled(self, uncertainty: PolicyUncertainty) -> bool:
        return (
            uncertainty.margin >= self.min_margin
            and uncertainty.normalized_entropy <= self.max_entropy
            and uncertainty.ood_score <= self.max_ood
        )

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
        del chunk_size
        if not candidates:
            return []
        candidate_set = set(candidates)
        weights: dict[int, float] = {}
        for evidence in observation.discovered_edges:
            if evidence.kind != EvidenceKind.NON_EDGE and evidence.dst in candidate_set:
                weights[evidence.dst] = max(weights.get(evidence.dst, 0.0), evidence.weight)
        weight_order = sorted(
            candidates, key=lambda node: (weights.get(node, 0.0), -node), reverse=True
        )
        uncertainty = self.uncertainty(
            query, observation, candidates, frontier=frontier, device=device
        )
        if self.alpha == 0.0 or not self._enabled(uncertainty):
            self.fallback_count += 1
            return weight_order
        self.enabled_count += 1
        learned_order = self.learned.rank(
            query, observation, candidates, frontier=frontier, device=device
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

    def stats(self) -> dict[str, int | float]:
        total = self.enabled_count + self.fallback_count
        return {
            "enabled": self.enabled_count,
            "fallback": self.fallback_count,
            "enabled_fraction": self.enabled_count / total if total else 0.0,
        }
