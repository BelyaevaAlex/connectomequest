from __future__ import annotations

import torch

from connectomequest.env import Action, ActionType, ConnectomeEnv
from connectomequest.policies.oracle import PrivilegedOraclePolicy
from connectomequest.query import Query, QueryType
from connectomequest.toy import make_toy_graph


def test_privileged_oracle_returns_every_legal_candidate(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=20)
    env.reset(query)
    observation = env.step(Action(ActionType.INSPECT, relation="presynaptic_to")).observation
    candidates = [2, 3]
    policy = PrivilegedOraclePolicy(graph, {query.query_id: query.answers})
    order = policy.rank(
        query.spec,
        observation,
        candidates,
        frontier=True,
        device=torch.device("cpu"),
    )
    assert set(order) == set(candidates)
    assert len(order) == len(candidates)
