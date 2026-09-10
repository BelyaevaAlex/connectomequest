"""Learned per-state expert selection with a conservative exact fallback."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from connectomequest.env import Observation
from connectomequest.models.inductive import QUERY_TO_ID
from connectomequest.policies.base import FrontierPolicy
from connectomequest.policies.expert_gate_features_v2 import (
    EXPERT_STAT_DIM_V2,
    ExpertGateFeaturesV2,
)
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime

CONTEXT_DIM_V2 = len(QUERY_TO_ID) + 6


def context_features_v2(
    query: QuerySpec,
    observation: Observation,
    candidates: list[int],
    *,
    frontier: bool,
) -> Tensor:
    """Return bounded state features without entity or domain identifiers."""

    query_one_hot = torch.zeros(len(QUERY_TO_ID), dtype=torch.float32)
    query_one_hot[QUERY_TO_ID[query.query_type]] = 1.0
    candidate_set = set(candidates)
    visible_sketches = sum(sketch.node_id in candidate_set for sketch in observation.node_sketches)
    visible_edges = sum(
        edge.kind != EvidenceKind.NON_EDGE
        and edge.src == observation.current_node
        and edge.dst in candidate_set
        for edge in observation.discovered_edges
    )
    size = max(1, len(candidates))
    state = torch.tensor(
        [
            float(frontier),
            float(query.semantic_relation is not None),
            min(1.0, math.log1p(len(candidates)) / math.log(33.0)),
            min(1.0, math.log1p(max(0.0, observation.remaining_budget)) / math.log(65.0)),
            visible_sketches / size,
            min(1.0, visible_edges / size),
        ],
        dtype=torch.float32,
    )
    return torch.cat((query_one_hot, state))


class ExpertGateNetwork(nn.Module):
    """Tiny CPU-friendly contextual classifier over observable state summaries."""

    def __init__(self, num_experts: int, hidden_dim: int = 32):
        super().__init__()
        if num_experts < 2:
            raise ValueError("expert gate requires at least two experts")
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(CONTEXT_DIM_V2 + EXPERT_STAT_DIM_V2 * num_experts, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_experts)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(torch.nn.functional.silu(self.input(features)))


@dataclass(slots=True)
class ContextualExpertGateV2(ExpertGateFeaturesV2):
    """Select an expert per DecisionSnapshot, falling back when uncertain."""

    experts: dict[str, FrontierPolicy]
    expert_names: tuple[str, ...]
    gate: ExpertGateNetwork
    confidence_threshold: float
    fallback_expert: str = "weight"
    second_hop_expert: str = "snapshot"
    name: str = "contextual_expert_gate_v2"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE
    selection_counts: dict[str, int] = field(default_factory=dict, init=False)
    fallback_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if tuple(self.experts) != self.expert_names:
            raise ValueError("experts must preserve the declared expert_names order")
        if self.fallback_expert not in self.experts:
            raise ValueError("fallback expert is missing")
        if self.second_hop_expert not in self.experts:
            raise ValueError("second-hop expert is missing")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")

    @torch.inference_mode()
    def select_expert(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
    ) -> str:
        probabilities = self._gate_probabilities(
            query, observation, candidates, frontier=frontier, device=device
        )
        confidence, selected = torch.max(probabilities, dim=0)
        if float(confidence) < self.confidence_threshold:
            self.fallback_count += 1
            return self.fallback_expert
        name = self.expert_names[int(selected)]
        self.selection_counts[name] = self.selection_counts.get(name, 0) + 1
        return name

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
        selected = self.second_hop_expert
        if frontier:
            selected = self.select_expert(
                query, observation, candidates, frontier=True, device=device
            )
        order = self.experts[selected].rank(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
            chunk_size=chunk_size,
        )
        if len(order) != len(candidates) or set(order) != set(candidates):
            raise ValueError(f"expert {selected!r} returned an incomplete order")
        return order

    def stats(self) -> dict[str, int | float | dict[str, int]]:
        selected = sum(self.selection_counts.values())
        total = selected + self.fallback_count
        return {
            "selected": dict(self.selection_counts),
            "fallback": self.fallback_count,
            "fallback_fraction": self.fallback_count / total if total else 0.0,
        }


@dataclass(frozen=True, slots=True)
class ObservableWeightPolicy:
    """Exact observable edge-weight order used as the conservative fallback."""

    name: str = "weight"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE

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
        del query, frontier, device, chunk_size
        candidate_set = set(candidates)
        weights: dict[int, float] = {}
        for edge in observation.discovered_edges:
            if edge.kind != EvidenceKind.NON_EDGE and edge.dst in candidate_set:
                weights[edge.dst] = max(weights.get(edge.dst, 0.0), float(edge.weight))
        return sorted(candidates, key=lambda node: (-weights.get(node, 0.0), node))


def load_contextual_expert_gate_v2(
    payload: dict,
    experts: dict[str, FrontierPolicy],
) -> ContextualExpertGateV2:
    """Restore a gate while keeping expert checkpoints explicit and auditable."""

    if payload.get("model_kind") != "contextual_expert_gate_v2":
        raise ValueError("checkpoint is not a contextual expert gate v2")
    expert_names = tuple(payload["expert_names"])
    config = payload["gate_config"]
    if int(config["num_experts"]) != len(expert_names):
        raise ValueError("gate num_experts does not match expert_names")
    if tuple(experts) != expert_names:
        raise ValueError("loaded experts do not match checkpoint order")
    gate = ExpertGateNetwork(
        num_experts=int(config["num_experts"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    gate.load_state_dict(payload["gate_state"])
    gate.eval()
    return ContextualExpertGateV2(
        experts=experts,
        expert_names=expert_names,
        gate=gate,
        confidence_threshold=float(payload["confidence_threshold"]),
        fallback_expert=payload["fallback_expert"],
        second_hop_expert=payload["second_hop_expert"],
    )


@dataclass(frozen=True, slots=True)
class ObservableRandomPolicy:
    """Deterministic query-seeded random order over canonical candidate keys."""

    seed: int = 17
    name: str = "random"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE

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
        del frontier, device, chunk_size

        def key(node: int) -> tuple[int, int]:
            payload = f"{self.seed}:{query.query_id}:{observation.current_node}:{node}".encode()
            digest = hashlib.blake2b(payload, digest_size=8).digest()
            return int.from_bytes(digest, "big"), node

        return sorted(candidates, key=key)
