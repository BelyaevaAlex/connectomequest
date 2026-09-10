"""Stage-aware, entity-ID-free policy trained on frozen decision snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from connectomequest.env import Observation
from connectomequest.models.inductive_v5 import InductiveQueryPolicyV5
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime, ordered_candidates


class SnapshotPolicy(nn.Module):
    """Independent first/second-hop heads over the same observable features."""

    name = "snapshot_ranker"
    access_regime = AccessRegime.OBSERVABLE

    def __init__(self, hidden_dim: int = 128, query_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.dropout = dropout
        self.first = InductiveQueryPolicyV5(hidden_dim, query_dim, dropout)
        self.second = InductiveQueryPolicyV5(hidden_dim, query_dim, dropout)

    def head(self, frontier: bool) -> InductiveQueryPolicyV5:
        return self.first if frontier else self.second

    def logits(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> Tensor:
        _, scores = self.head(frontier)._features_and_scores(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
        )
        return scores

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
        del chunk_size
        self.eval()
        return ordered_candidates(
            candidates,
            self.logits(query, observation, candidates, frontier=frontier, device=device),
        )

    @torch.inference_mode()
    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        device = next(self.parameters()).device
        frontier = observation.current_node == query.anchors[0]
        self.eval()
        return (
            self.logits(
                query,
                observation,
                candidates,
                frontier=frontier,
                device=device,
            )
            .float()
            .cpu()
        )


@dataclass(frozen=True, slots=True)
class SnapshotPolicyConfig:
    hidden_dim: int = 128
    query_dim: int = 32
    dropout: float = 0.1


def load_snapshot_policy(payload: dict, device: torch.device) -> SnapshotPolicy:
    if payload.get("model_kind") != "entity_id_free_snapshot_policy_v1":
        raise ValueError("checkpoint is not a snapshot policy")
    config = SnapshotPolicyConfig(**payload["model_config"])
    model = SnapshotPolicy(config.hidden_dim, config.query_dim, config.dropout).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()
