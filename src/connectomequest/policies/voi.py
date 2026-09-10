"""Access-controlled policy compositions for value-of-information audits.

The policy composes two observable policies: a diversity policy chooses the
first-hop branch and a learned policy ranks candidates after inspection.
"""

from __future__ import annotations

from typing import Any

import torch


class FirstHopRandomSecondHopMinervaPolicy:
    """Random first-hop exploration followed by MINERVA second-hop ranking."""

    name = "voi_random_minerva"

    def __init__(self, random_policy: Any, minerva_policy: Any) -> None:
        self.random_policy = random_policy
        self.minerva_policy = minerva_policy

    def rank(
        self,
        query: Any,
        observation: Any,
        candidates: list[int],
        *,
        frontier: bool,
        device: torch.device,
        chunk_size: int = 65_536,
    ) -> list[int]:
        policy = self.random_policy if frontier else self.minerva_policy
        return policy.rank(
            query,
            observation,
            candidates,
            frontier=frontier,
            device=device,
            chunk_size=chunk_size,
        )


__all__ = ["FirstHopRandomSecondHopMinervaPolicy"]
