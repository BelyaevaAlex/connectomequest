"""REINFORCE training for the observable PyTorch MINERVA-style policy."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from connectomequest.decision_benchmark import DecisionSnapshot, DecisionStage
from connectomequest.policies.minerva import MinervaPolicy


@dataclass(frozen=True, slots=True)
class MinervaTrajectory:
    first: DecisionSnapshot
    branches: dict[int, DecisionSnapshot]


def group_trajectories(snapshots: list[DecisionSnapshot]) -> list[MinervaTrajectory]:
    key_type = tuple[str, str, int]
    first_by_query: dict[key_type, DecisionSnapshot] = {}
    branches: dict[key_type, dict[int, DecisionSnapshot]] = defaultdict(dict)
    for snapshot in snapshots:
        key = (snapshot.held_out, snapshot.query.query_id, snapshot.probes_used)
        if snapshot.stage == DecisionStage.FIRST_HOP:
            if key in first_by_query:
                raise ValueError(f"duplicate first-hop snapshot: {key}")
            first_by_query[key] = snapshot
        elif snapshot.branch_middle is not None:
            branches[key][snapshot.branch_middle] = snapshot
    trajectories = [
        MinervaTrajectory(first, branches[key])
        for key, first in first_by_query.items()
        if first.candidates and branches[key]
    ]
    if not trajectories:
        raise ValueError("snapshots contain no complete two-hop trajectories")
    return trajectories


def _sample_rollout(
    model: MinervaPolicy,
    trajectory: MinervaTrajectory,
    *,
    device: torch.device,
    first_hop_reward: float,
) -> tuple[Tensor, Tensor, float, bool]:
    first = trajectory.first
    first_candidates = list(first.candidates)
    first_logits, hidden = model.logits(
        first.query,
        first.observation,
        first_candidates,
        frontier=True,
        device=device,
    )
    first_distribution = torch.distributions.Categorical(logits=first_logits)
    first_index = first_distribution.sample()
    middle = first_candidates[int(first_index)]
    branch = trajectory.branches.get(middle)
    first_correct = middle in first.relevant_candidates
    if branch is None or not branch.candidates:
        reward = first_hop_reward * float(first_correct)
        return (
            first_distribution.log_prob(first_index),
            first_distribution.entropy(),
            reward,
            False,
        )

    hidden = model.transition(
        hidden,
        branch.query,
        branch.observation,
        device=device,
    )
    second_candidates = list(branch.candidates)
    second_logits, _ = model.logits(
        branch.query,
        branch.observation,
        second_candidates,
        frontier=False,
        device=device,
        hidden=hidden,
    )
    second_distribution = torch.distributions.Categorical(logits=second_logits)
    second_index = second_distribution.sample()
    answer = second_candidates[int(second_index)]
    terminal_success = answer in branch.relevant_candidates
    reward = float(terminal_success) + first_hop_reward * float(first_correct)
    return (
        first_distribution.log_prob(first_index) + second_distribution.log_prob(second_index),
        first_distribution.entropy() + second_distribution.entropy(),
        reward,
        terminal_success,
    )


def train_minerva_epoch(
    model: MinervaPolicy,
    trajectories: list[MinervaTrajectory],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_size: int = 64,
    entropy_weight: float = 0.01,
    first_hop_reward: float = 0.2,
    seed: int = 17,
) -> dict[str, float | int]:
    order = list(range(len(trajectories)))
    random.Random(seed).shuffle(order)
    model.train()
    losses: list[float] = []
    rewards: list[float] = []
    successes = 0
    optimizer.zero_grad(set_to_none=True)
    pending: list[tuple[Tensor, Tensor, float]] = []

    def update() -> None:
        if not pending:
            return
        returns = torch.tensor([item[2] for item in pending], device=device)
        baseline = returns.mean()
        policy_terms = torch.stack(
            [
                -(reward - baseline).detach() * item[0]
                for item, reward in zip(pending, returns, strict=False)
            ]
        )
        entropies = torch.stack([item[1] for item in pending])
        loss = policy_terms.mean() - entropy_weight * entropies.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        pending.clear()

    for index in order:
        log_probability, entropy, reward, success = _sample_rollout(
            model,
            trajectories[index],
            device=device,
            first_hop_reward=first_hop_reward,
        )
        pending.append((log_probability, entropy, reward))
        rewards.append(reward)
        successes += int(success)
        if len(pending) >= batch_size:
            update()
    update()
    return {
        "trajectories": len(order),
        "mean_loss": sum(losses) / len(losses) if losses else 0.0,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "terminal_success": successes / len(order) if order else 0.0,
    }
