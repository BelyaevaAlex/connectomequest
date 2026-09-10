"""Privileged active-search ceiling; never a deployable policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from connectomequest.env import Observation
from connectomequest.graph import GraphStore
from connectomequest.proof import EvidenceKind
from connectomequest.query import QuerySpec


@dataclass(slots=True)
class PrivilegedOraclePolicy:
    graph: GraphStore
    answers_by_query: dict[str, frozenset[int]]

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
        del device, chunk_size
        answers = self.answers_by_query[query.query_id]
        weights: dict[int, float] = {}
        for edge in observation.discovered_edges:
            if edge.kind != EvidenceKind.NON_EDGE and edge.dst in candidates:
                weights[edge.dst] = max(weights.get(edge.dst, 0.0), edge.weight)
        if frontier:
            utility = {
                middle: len(
                    answers
                    & set(
                        self.graph.neighbors(
                            middle,
                            "presynaptic_to",
                            by_weight=False,
                        ).node_ids.tolist()
                    )
                )
                for middle in candidates
            }
        else:
            utility = {candidate: int(candidate in answers) for candidate in candidates}
        return sorted(
            candidates,
            key=lambda node: (-utility[node], -weights.get(node, 0.0), node),
        )
