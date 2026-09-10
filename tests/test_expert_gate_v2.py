from __future__ import annotations

from dataclasses import dataclass

import torch

from connectomequest.env import Observation
from connectomequest.policies.expert_gate_v2 import (
    ContextualExpertGateV2,
    ExpertGateNetwork,
    context_features_v2,
)
from connectomequest.query import QuerySpec, QueryType


@dataclass
class FixedPolicy:
    order: tuple[int, ...]

    def rank(self, query, observation, candidates, **kwargs):
        del query, observation, kwargs
        return [node for node in self.order if node in candidates]


def _state() -> tuple[QuerySpec, Observation]:
    query = QuerySpec("q", QueryType.TWO_HOP_TYPE, (0, 9), "has_type")
    return query, Observation(0, (), (), 63.0, 1)


def _gate(logits: tuple[float, float], threshold: float) -> ContextualExpertGateV2:
    network = ExpertGateNetwork(num_experts=2, hidden_dim=8)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.output.bias.copy_(torch.tensor(logits))
    return ContextualExpertGateV2(
        experts={
            "weight": FixedPolicy((1, 2, 3)),
            "learned": FixedPolicy((3, 2, 1)),
        },
        expert_names=("weight", "learned"),
        gate=network,
        confidence_threshold=threshold,
        fallback_expert="weight",
        second_hop_expert="learned",
    )


def test_context_features_are_entity_id_free_and_fixed_width() -> None:
    query, observation = _state()
    first = context_features_v2(query, observation, [11, 22, 33], frontier=True)
    relabeled = context_features_v2(query, observation, [101, 202, 303], frontier=True)
    assert first.shape == relabeled.shape
    assert first.shape == (12,)
    assert torch.equal(first, relabeled)


def test_contextual_gate_uses_exact_fallback_below_confidence() -> None:
    query, observation = _state()
    gate = _gate((0.0, 0.0), threshold=0.75)
    assert gate.rank(
        query,
        observation,
        [1, 2, 3],
        frontier=True,
        device=torch.device("cpu"),
    ) == [1, 2, 3]


def test_contextual_gate_uses_selected_expert_above_confidence() -> None:
    query, observation = _state()
    gate = _gate((-5.0, 5.0), threshold=0.75)
    assert gate.rank(
        query,
        observation,
        [1, 2, 3],
        frontier=True,
        device=torch.device("cpu"),
    ) == [3, 2, 1]


def test_contextual_gate_is_first_hop_only() -> None:
    query, observation = _state()
    gate = _gate((0.0, 0.0), threshold=0.75)
    gate.second_hop_expert = "learned"
    assert gate.rank(
        query,
        observation,
        [1, 2, 3],
        frontier=False,
        device=torch.device("cpu"),
    ) == [3, 2, 1]


def test_contextual_gate_exposes_adaptive_probe_uncertainty() -> None:
    query, observation = _state()
    gate = _gate((0.0, 0.0), threshold=0.75)
    uncertainty = gate.uncertainty(
        query,
        observation,
        [1, 2, 3],
        frontier=True,
        device=torch.device("cpu"),
    )
    assert uncertainty.margin == 0.0
