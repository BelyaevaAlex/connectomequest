from __future__ import annotations

import torch

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.env import Action, ActionType, ConnectomeEnv
from connectomequest.evaluation import episode_metrics
from connectomequest.inductive_training import ObservableIndex
from connectomequest.inductive_training_v5 import listwise_loss, query_list_batch
from connectomequest.models.inductive_v5 import FEATURE_DIM_V5, InductiveQueryPolicyV5
from connectomequest.policies.contextual_gate import ContextualGatePolicy
from connectomequest.query import Query, QueryType
from connectomequest.toy import make_toy_graph


def test_v5_listwise_batch_is_observable_and_trainable(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.INTERSECTION_NEGATION, (0, 1), frozenset({2}), "train")
    batch = query_list_batch(
        ObservableIndex(graph, visible_neighbors=2), [query], probe_choices=(0, 1), seed=3
    )
    assert batch is not None
    assert batch.features.shape[-1] == FEATURE_DIM_V5
    assert batch.mask.sum() == 2
    model = InductiveQueryPolicyV5(hidden_dim=16, query_dim=8, dropout=0.0)
    loss = listwise_loss(model, batch, device=torch.device("cpu"))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_context_gate_alpha_zero_is_exact_weight_order(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=16)
    observation = env.reset(query)
    observation = env.step(Action(ActionType.INSPECT, relation="presynaptic_to")).observation
    candidates = [edge.dst for edge in observation.discovered_edges]
    gate = ContextualGatePolicy(
        InductiveQueryPolicyV5(hidden_dim=16, query_dim=8, dropout=0.0), alpha=0.0
    )
    assert gate.rank(
        query.spec, observation, candidates, frontier=True, device=torch.device("cpu")
    ) == [2, 3]
    assert gate.stats()["fallback"] == 1


def test_adaptive_probe_controller_retains_valid_typed_proof(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=20)
    env.reset(query)
    policy = ContextualGatePolicy(
        InductiveQueryPolicyV5(hidden_dim=16, query_dim=8, dropout=0.0), alpha=0.0
    )
    result = BudgetedExplorer(
        policy=policy,
        ranking="policy",
        adaptive_probing=True,
        max_subgoal_probes=1,
    ).run(env, query.spec)
    assert episode_metrics(query, result).goal_success


def test_proof_enforcement_ablation_keeps_proof_metric_separate(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.INTERSECTION, (0, 1), frozenset({3}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8, enforce_proof=False)
    env.reset(query)
    result = env.step(Action(ActionType.SUBMIT, answers=frozenset({3})))
    assert result.validation is not None
    assert result.validation.valid
    assert not result.validation.evidence_valid
