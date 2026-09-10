from __future__ import annotations

import torch

from connectomequest.inductive_training import ObservableIndex, query_pair_batch
from connectomequest.models.inductive import InductiveQueryPolicy
from connectomequest.query import Query, QueryType
from connectomequest.toy import make_toy_graph


def test_observable_pair_batch_and_loss(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    query = Query(
        "q",
        QueryType.INTERSECTION_NEGATION,
        (0, 1),
        frozenset({2}),
        "train",
    )
    batch = query_pair_batch(ObservableIndex(graph, visible_neighbors=2), [query])
    assert batch is not None
    assert len(batch) == 1

    model = InductiveQueryPolicy(hidden_dim=16, query_dim=8, dropout=0.0)
    positive = model(
        batch.positive_features,
        batch.query_type_ids,
        batch.semantic_flags,
    )
    negative = model(
        batch.negative_features,
        batch.query_type_ids,
        batch.semantic_flags,
    )
    loss = torch.nn.functional.softplus(negative - positive).mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
