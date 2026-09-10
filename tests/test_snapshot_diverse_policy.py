from __future__ import annotations

from dataclasses import dataclass

import torch

from connectomequest.env import Observation
from connectomequest.policies.diverse import DiversePolicy
from connectomequest.policies.snapshot import SnapshotPolicy
from connectomequest.query import QuerySpec, QueryType


@dataclass
class FixedPolicy:
    reverse: bool = False

    def rank(self, query, observation, candidates, **kwargs):
        del query, observation, kwargs
        return list(reversed(candidates)) if self.reverse else list(candidates)


def _state() -> tuple[QuerySpec, Observation]:
    query = QuerySpec("q", QueryType.TWO_HOP_TYPE, (0, 9), "has_type")
    observation = Observation(0, (), (), 63.0, 1)
    return query, observation


def test_snapshot_policy_returns_complete_id_free_order() -> None:
    query, observation = _state()
    policy = SnapshotPolicy(hidden_dim=16, query_dim=8, dropout=0.0)
    candidates = [4, 2, 7]
    order = policy.rank(
        query,
        observation,
        candidates,
        frontier=True,
        device=torch.device("cpu"),
    )
    assert len(order) == len(candidates)
    assert set(order) == set(candidates)


def test_diverse_policy_round_robins_component_top_choices() -> None:
    query, observation = _state()
    policy = DiversePolicy(
        snapshot=FixedPolicy(),
        minerva=FixedPolicy(reverse=True),
        components=("snapshot", "minerva"),
    )
    assert policy.rank(
        query,
        observation,
        [1, 2, 3],
        frontier=True,
        device=torch.device("cpu"),
    ) == [1, 3, 2]
