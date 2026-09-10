"""Adapter from an observable FrontierPolicy to CandidateRanker."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from connectomequest.env import Observation
from connectomequest.policies.base import FrontierPolicy
from connectomequest.query import QuerySpec
from connectomequest.rankers.base import AccessRegime


@dataclass(slots=True)
class PolicyCandidateRanker:
    policy: FrontierPolicy
    device: torch.device
    name: str = "learned"
    access_regime: AccessRegime = AccessRegime.OBSERVABLE

    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        if not candidates:
            return torch.empty(0, dtype=torch.float32)
        order = self.policy.rank(
            query,
            observation,
            candidates,
            frontier=observation.current_node == query.anchors[0],
            device=self.device,
        )
        if len(order) != len(candidates) or set(order) != set(candidates):
            raise ValueError("frontier policy returned an incomplete candidate order")
        rank = {candidate: index for index, candidate in enumerate(order)}
        return torch.tensor(
            [float(len(candidates) - rank[candidate]) for candidate in candidates],
            dtype=torch.float32,
        )
