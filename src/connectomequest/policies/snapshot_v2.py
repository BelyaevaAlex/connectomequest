"""Resource-efficient stage-aware ranker with explicit missing-sketch handling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from connectomequest.env import Observation
from connectomequest.models.inductive_v5 import (
    BASE_FEATURE_DIM,
    InductiveQueryPolicyV5,
    PolicyUncertainty,
)
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime, ordered_candidates


class MaskedSnapshotHeadV2(InductiveQueryPolicyV5):
    """V5 scorer with a separately learned path for an absent probe sketch.

    The v1 scorer zeroed its probe tower when a sketch was absent.  V2 keeps the
    explicit presence bit and adds a small residual conditioned on the base
    observable features.  Thus missingness is modeled, never imputed as a zero
    measurement, and costs only one narrow MLP per stage.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        query_dim: int = 32,
        dropout: float = 0.1,
        *,
        use_weight_prior: bool = True,
        use_probe_features: bool = True,
        use_query_embedding: bool = True,
    ):
        super().__init__(
            hidden_dim,
            query_dim,
            dropout,
            use_weight_prior=use_weight_prior,
            use_probe_features=use_probe_features,
            use_query_embedding=use_query_embedding,
        )
        common = query_dim + 1
        missing_hidden = max(16, hidden_dim // 4)
        self.missing_scorer = nn.Sequential(
            nn.Linear(BASE_FEATURE_DIM + common, missing_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(missing_hidden, 1),
        )
        # A scalar offset is useful when comparing probed and unprobed nodes;
        # unlike zero imputation, its semantics are explicit and auditable.
        self.missing_sketch_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.missing_scorer[-1].weight)
        nn.init.zeros_(self.missing_scorer[-1].bias)

    def forward(
        self,
        features: Tensor,
        query_type_ids: Tensor,
        semantic_flags: Tensor,
    ) -> Tensor:
        scores = super().forward(features, query_type_ids, semantic_flags)
        if not self.use_probe_features:
            return scores
        present = features[:, BASE_FEATURE_DIM].clamp(0.0, 1.0)
        missing = 1.0 - present
        embedded_query = self.query_embedding(query_type_ids)
        query_vector = (
            embedded_query if self.use_query_embedding else torch.zeros_like(embedded_query)
        )
        common = torch.cat((query_vector, semantic_flags[:, None]), dim=-1)
        missing_input = torch.cat((features[:, :BASE_FEATURE_DIM], common), dim=-1)
        residual = self.missing_scorer(missing_input).squeeze(-1)
        return scores + missing * (self.missing_sketch_bias + residual)


class SnapshotPolicyV2(nn.Module):
    """Independent first/second-hop masked heads under observable access."""

    name = "snapshot_ranker_v2"
    access_regime = AccessRegime.OBSERVABLE

    def __init__(
        self,
        hidden_dim: int = 128,
        query_dim: int = 32,
        dropout: float = 0.1,
        *,
        use_weight_prior: bool = True,
        use_probe_features: bool = True,
        use_query_embedding: bool = True,
    ):
        super().__init__()
        self.config = SnapshotPolicyV2Config(
            hidden_dim=hidden_dim,
            query_dim=query_dim,
            dropout=dropout,
            use_weight_prior=use_weight_prior,
            use_probe_features=use_probe_features,
            use_query_embedding=use_query_embedding,
        )
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.dropout = dropout
        self.first = MaskedSnapshotHeadV2(
            hidden_dim,
            query_dim,
            dropout,
            use_weight_prior=use_weight_prior,
            use_probe_features=use_probe_features,
            use_query_embedding=use_query_embedding,
        )
        self.second = MaskedSnapshotHeadV2(
            hidden_dim,
            query_dim,
            dropout,
            use_weight_prior=use_weight_prior,
            use_probe_features=use_probe_features,
            use_query_embedding=use_query_embedding,
        )

    def head(self, frontier: bool) -> MaskedSnapshotHeadV2:
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
            query, observation, candidates, frontier=frontier, device=device
        )
        return scores

    @torch.inference_mode()
    def uncertainty(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> PolicyUncertainty:
        return self.head(frontier).uncertainty(
            query, observation, candidates, frontier=frontier, device=device
        )

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
            self.logits(query, observation, candidates, frontier=frontier, device=device)
            .float()
            .cpu()
        )


@dataclass(frozen=True, slots=True)
class SnapshotPolicyV2Config:
    hidden_dim: int = 128
    query_dim: int = 32
    dropout: float = 0.1
    use_weight_prior: bool = True
    use_probe_features: bool = True
    use_query_embedding: bool = True


def load_snapshot_policy_v2(payload: dict, device: torch.device) -> SnapshotPolicyV2:
    if payload.get("model_kind") != "entity_id_free_snapshot_policy_v2":
        raise ValueError("checkpoint is not a v2 snapshot policy")
    config = SnapshotPolicyV2Config(**payload["model_config"])
    model = SnapshotPolicyV2(
        config.hidden_dim,
        config.query_dim,
        config.dropout,
        use_weight_prior=config.use_weight_prior,
        use_probe_features=config.use_probe_features,
        use_query_embedding=config.use_query_embedding,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()
