"""Observable expert-distribution features and adaptive-probe uncertainty."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from connectomequest.models.inductive_v5 import PolicyUncertainty

EXPERT_STAT_DIM_V2 = 5


class ExpertGateFeaturesV2:
    """Mixin for contextual gates; requires experts, gate and expert_names."""

    def _expert_statistics(self, query, observation, candidates, *, frontier, device) -> Tensor:
        orders = {}
        for name in self.expert_names:
            order = self.experts[name].rank(
                query, observation, candidates, frontier=frontier, device=device
            )
            if len(order) != len(candidates) or set(order) != set(candidates):
                raise ValueError(f"expert {name!r} returned an incomplete order")
            orders[name] = order
        fallback_rank = {node: rank for rank, node in enumerate(orders[self.fallback_expert])}
        denominator = max(1, len(candidates) - 1)
        values = []
        for name in self.expert_names:
            order = orders[name]
            uncertainty_fn = getattr(self.experts[name], "uncertainty", None)
            if uncertainty_fn is None:
                margin, normalized_entropy, ood, available = 0.0, 0.0, 0.0, 0.0
            else:
                uncertainty = uncertainty_fn(
                    query, observation, candidates, frontier=frontier, device=device
                )
                margin = float(uncertainty.margin)
                normalized_entropy = float(uncertainty.normalized_entropy)
                ood = float(uncertainty.ood_score)
                available = 1.0
            disagreement = sum(
                abs(rank - fallback_rank[node]) for rank, node in enumerate(order)
            ) / (len(order) * denominator)
            values.extend((margin, normalized_entropy, ood, available, disagreement))
        return torch.tensor(values, dtype=torch.float32)

    @torch.inference_mode()
    def _gate_probabilities(self, query, observation, candidates, *, frontier, device) -> Tensor:
        # Local import avoids a module cycle while keeping the base context public.
        from connectomequest.policies.expert_gate_v2 import context_features_v2

        base = context_features_v2(query, observation, candidates, frontier=frontier)
        expert = self._expert_statistics(
            query, observation, candidates, frontier=frontier, device=device
        )
        gate_device = next(self.gate.parameters()).device
        return torch.softmax(self.gate(torch.cat((base, expert)).to(gate_device)[None])[0], dim=0)

    @torch.inference_mode()
    def uncertainty(self, query, observation, candidates, *, frontier, device) -> PolicyUncertainty:
        if len(candidates) < 2:
            return PolicyUncertainty(float("inf"), 0.0, 0.0)
        probabilities = self._gate_probabilities(
            query, observation, candidates, frontier=frontier, device=device
        )
        ordered = torch.sort(probabilities, descending=True).values
        margin = float(ordered[0] - ordered[1])
        entropy = -torch.sum(probabilities * probabilities.clamp_min(1e-12).log())
        normalized_entropy = float(entropy / math.log(len(probabilities)))
        expert = self._expert_statistics(
            query, observation, candidates, frontier=frontier, device=device
        )
        statistics = expert.reshape(-1, EXPERT_STAT_DIM_V2)
        available = statistics[:, 3]
        ood = float((statistics[:, 2] * available).sum() / available.sum().clamp_min(1.0))
        return PolicyUncertainty(margin, normalized_entropy, ood)
