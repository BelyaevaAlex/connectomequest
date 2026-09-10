"""Common score-only interface for fair decision-state evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from connectomequest.env import Observation
from connectomequest.query import QuerySpec


class AccessRegime(StrEnum):
    OBSERVABLE = "observable"
    OBSERVABLE_PAID_LOOKAHEAD = "observable+paid_lookahead"
    FULL_GRAPH_ENCODER_VISIBLE_OUTPUT = "full_graph_encoder_visible_output"
    FULL_GRAPH = "full_graph"
    PRIVILEGED_CEILING = "privileged_ceiling"
    EXTERNAL_PRETRAINED = "external_pretrained"


@runtime_checkable
class CandidateRanker(Protocol):
    name: str
    access_regime: AccessRegime

    def score(
        self,
        query: QuerySpec,
        observation: Observation,
        candidates: list[int],
    ) -> Tensor:
        """Return one score per legal visible candidate."""


def ordered_candidates(candidates: list[int], scores: Tensor) -> list[int]:
    scores = torch.as_tensor(scores, dtype=torch.float32).detach().cpu()
    if scores.ndim != 1 or len(scores) != len(candidates):
        raise ValueError(
            f"ranker returned shape {tuple(scores.shape)} for {len(candidates)} candidates"
        )
    if torch.isnan(scores).any():
        raise ValueError("ranker returned NaN scores")
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate list contains duplicates")
    return [
        candidates[index]
        for index in sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index]),
        )
    ]
