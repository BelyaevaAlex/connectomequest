from __future__ import annotations

import torch

from connectomequest.decision_benchmark import build_query_snapshots
from connectomequest.manifest import sha256_file
from connectomequest.minerva_training import group_trajectories, train_minerva_epoch
from connectomequest.policies.minerva import MinervaPolicy
from connectomequest.query import Query, QueryType
from connectomequest.toy import make_toy_graph


def _snapshots(tmp_path):
    graph = make_toy_graph(tmp_path / "graph")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "train")
    return build_query_snapshots(
        graph,
        query,
        held_out="toy",
        graph_manifest_sha256=sha256_file(graph.manifest_path),
        query_sha256="query-hash",
        visible_neighbors=8,
    )


def test_minerva_masks_actions_to_snapshot_candidates(tmp_path) -> None:
    snapshots = _snapshots(tmp_path)
    first = snapshots[0]
    model = MinervaPolicy(hidden_dim=16, query_dim=8, dropout=0.0)
    scores = model.score(first.query, first.observation, list(first.candidates))
    assert scores.shape == (len(first.candidates),)
    assert torch.isfinite(scores).all()
    assert model.rank(
        first.query,
        first.observation,
        list(first.candidates),
        frontier=True,
        device=torch.device("cpu"),
    ) == sorted(
        first.candidates,
        key=lambda node: (-float(scores[list(first.candidates).index(node)]), node),
    )


def test_minerva_reinforce_epoch_backpropagates(tmp_path) -> None:
    trajectories = group_trajectories(_snapshots(tmp_path))
    model = MinervaPolicy(hidden_dim=16, query_dim=8, dropout=0.0)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    metrics = train_minerva_epoch(
        model,
        trajectories * 4,
        optimizer,
        device=torch.device("cpu"),
        batch_size=4,
        seed=3,
    )
    assert metrics["trajectories"] == 4
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
