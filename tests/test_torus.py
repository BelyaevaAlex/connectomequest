import torch

from connectomequest.models.torus import TorusQueryHeuristic
from connectomequest.query import QueryType


def test_frontier_ranking_projects_intermediate_two_hop_nodes() -> None:
    model = TorusQueryHeuristic(num_entities=3, embedding_dim=1)
    with torch.no_grad():
        model.entity.weight[:, 0] = torch.tensor([0.0, 1.0, 2.0])
        model.projection[:] = 1.0
        model.query_bias.weight.zero_()
        model.log_temperature.zero_()

    kwargs = {
        "query_type": QueryType.TWO_HOP,
        "anchors": (0,),
        "candidates": [1, 2],
        "device": torch.device("cpu"),
    }
    final_order = model.rank_candidates(**kwargs)
    frontier_order = model.rank_frontier(**kwargs)

    assert final_order == [2, 1]
    assert frontier_order == [1, 2]
