"""Common contract for active frontier-ranking policies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from connectomequest.env import Observation
from connectomequest.query import QuerySpec


@runtime_checkable
class FrontierPolicy(Protocol):
    """Rank observed candidates without access to the hidden full graph."""

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
        """Return every candidate exactly once, best candidate first."""
