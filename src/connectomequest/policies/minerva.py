"""Entity-ID-free PyTorch MINERVA-style policy for active two-hop search."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from connectomequest.env import Observation
from connectomequest.models.inductive import QUERY_TO_ID
from connectomequest.models.inductive_v5 import FEATURE_DIM_V5, candidate_features_v5
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime, ordered_candidates


class MinervaPolicy(nn.Module):
    """Recurrent path policy whose action space is the observed frontier only.

    Unlike the original transductive MINERVA implementation, numeric entity IDs
    are never embedded. They are used only as equality keys when observable
    structural action features are constructed.
    """

    name = "pytorch_minerva"
    access_regime = AccessRegime.OBSERVABLE

    def __init__(
        self,
        hidden_dim: int = 128,
        query_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.dropout = dropout
        self.query_embedding = nn.Embedding(len(QUERY_TO_ID), query_dim)
        self.initial_state = nn.Sequential(
            nn.Linear(query_dim + 1, hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM_V5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM_V5, hidden_dim),
            nn.Tanh(),
        )
        self.path_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.action_bias = nn.Linear(FEATURE_DIM_V5, 1)

    def initial_hidden(
        self,
        query: QuerySpec,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        query_id = torch.tensor([QUERY_TO_ID[query.query_type]], device=device)
        semantic = torch.tensor(
            [[float(query.semantic_relation is not None)]],
            dtype=torch.float32,
            device=device,
        )
        context = torch.cat((self.query_embedding(query_id), semantic), dim=-1)
        hidden = self.initial_state(context)
        return hidden, torch.zeros_like(hidden)

    def transition(
        self,
        hidden: tuple[Tensor, Tensor],
        query: QuerySpec,
        observation: Observation,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        history = candidate_features_v5(
            observation,
            query,
            [observation.current_node],
            frontier=True,
        ).to(device)
        return self.path_cell(self.history_encoder(history), hidden)

    def logits(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if not candidates:
            return torch.empty(0, device=device), self.initial_hidden(query, device=device)
        state = hidden or self.initial_hidden(query, device=device)
        if not frontier and hidden is None:
            state = self.transition(state, query, observation, device=device)
        features = candidate_features_v5(
            observation,
            query,
            candidates,
            frontier=frontier,
        ).to(device)
        actions = self.action_encoder(features)
        scores = (actions * state[0]).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return scores + self.action_bias(features).squeeze(-1), state

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
        scores, _ = self.logits(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
        )
        return ordered_candidates(candidates, scores)

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
        scores, _ = self.logits(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
        )
        return scores.float().cpu()


@dataclass(frozen=True, slots=True)
class MinervaConfig:
    hidden_dim: int = 128
    query_dim: int = 32
    dropout: float = 0.1


def load_minerva_policy(payload: dict, device: torch.device) -> MinervaPolicy:
    config = MinervaConfig(**payload["model_config"])
    model = MinervaPolicy(**asdict(config)).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()
