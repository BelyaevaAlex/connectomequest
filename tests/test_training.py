import numpy as np
import torch

from connectomequest.query import Query, QueryType
from connectomequest.training import SemanticTrainingIndex, collate_queries


def test_collate_queries_uses_filtered_negatives() -> None:
    query = Query(
        "dense",
        QueryType.TWO_HOP,
        (0,),
        frozenset(range(1, 9)),
        "train",
    )
    batch = collate_queries(
        [query],
        num_entities=10,
        negatives_per_positive=64,
        max_positives_per_query=4,
        generator=torch.Generator().manual_seed(7),
    )

    negatives = batch.candidates[:, 1:]
    assert torch.all((negatives == 0) | (negatives == 9))
    assert not any(value in query.answers for value in negatives.flatten().tolist())


def test_semantic_collation_uses_observable_hard_negatives() -> None:
    visible = np.full((8, 3), -1, dtype=np.int64)
    visible[0, :2] = [1, 2]
    visible[1, :2] = [4, 5]
    visible[2, :2] = [6, 7]
    index = SemanticTrainingIndex(visible)
    query = Query(
        "typed",
        QueryType.TWO_HOP_TYPE,
        (0, 7),
        frozenset({4}),
        "train",
    )

    batch = collate_queries(
        [query],
        num_entities=8,
        negatives_per_positive=4,
        max_positives_per_query=1,
        generator=torch.Generator().manual_seed(7),
        semantic_index=index,
    )

    assert {5, 6, 7} & set(batch.candidates[0, 1:].tolist())
    assert batch.frontier_candidates is not None
    assert batch.frontier_candidates.tolist() == [[1, 2]]
