from collections import Counter, defaultdict

import pyarrow as pa

import connectomequest.query as query_module
from connectomequest.query import (
    QueryType,
    entity_split,
    execute,
    grouped_split,
    query_split,
    sample_queries,
)
from connectomequest.schema import Relation
from connectomequest.toy import make_toy_graph


def test_graph_and_exact_queries(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    assert graph.num_nodes == 7
    assert graph.neighbors(0, Relation.PRESYNAPTIC_TO).node_ids.tolist() == [2, 3]
    assert execute(graph, QueryType.ONE_HOP, (0,)) == {2, 3}
    assert execute(graph, QueryType.TWO_HOP, (0,)) == {4}
    assert execute(graph, QueryType.INTERSECTION, (0, 1)) == {3}
    assert execute(graph, QueryType.TWO_HOP_TYPE, (0, 6)) == {4}
    assert execute(graph, QueryType.INTERSECTION_NEGATION, (0, 1)) == {2}
    assert execute(graph, QueryType.INTERSECTION_TYPE, (0, 1)) == {5}


def test_grouped_split_keeps_inverse_pair_together():
    assert grouped_split(42, 7, seed=3) == grouped_split(7, 42, seed=3)


def test_query_split_is_anchor_disjoint():
    for anchor in range(100):
        assert query_split((anchor,), seed=5) == entity_split(anchor, seed=5)
    left = next(x for x in range(100) if entity_split(x, 5) == "train")
    right = next(x for x in range(100) if entity_split(x, 5) == "test")
    assert query_split((left, right), seed=5) is None


def test_sampling_seed_cannot_change_cross_type_entity_split(monkeypatch):
    class FakeGraph:
        def __init__(self):
            self.table = pa.table(
                {
                    "node_id": pa.array(range(2000), type=pa.int64()),
                    "node_type": pa.array(["neuron"] * 2000),
                }
            )

        def nodes(self, columns=None):
            return self.table.select(columns) if columns else self.table

    monkeypatch.setattr(query_module, "execute", lambda *_: {1999})
    graph = FakeGraph()
    queries = [
        *sample_queries(
            graph,
            QueryType.TWO_HOP,
            count=100,
            seed=17,
            sampling_seed=18,
        ),
        *sample_queries(
            graph,
            QueryType.INTERSECTION,
            count=100,
            seed=17,
            sampling_seed=19,
        ),
    ]
    assert Counter(query.split for query in queries) == {
        "train": 160,
        "validation": 20,
        "test": 20,
    }
    owners = defaultdict(set)
    for query in queries:
        for anchor in query.anchors:
            owners[anchor].add(query.split)
    assert all(len(splits) == 1 for splits in owners.values())
