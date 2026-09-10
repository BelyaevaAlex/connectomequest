from __future__ import annotations

import pickle

from connectomequest.baselines.export import (
    STRUCTURES,
    edge_split,
    export_baseline_bundle,
    export_link_prediction_bundle,
)
from connectomequest.query import Query, QueryType
from connectomequest.query_io import write_queries
from connectomequest.schema import Relation
from connectomequest.toy import make_toy_graph


def test_export_baseline_bundle(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    queries = [
        Query("train", QueryType.ONE_HOP, (0,), frozenset({2, 3}), "train"),
        Query(
            "semantic",
            QueryType.TWO_HOP_TYPE,
            (0, 6),
            frozenset({4}),
            "test",
            str(Relation.HAS_TYPE),
        ),
    ]
    query_path = tmp_path / "queries.parquet"
    write_queries(queries, query_path, graph, seed=17)
    report = export_baseline_bundle(graph, query_path, tmp_path / "bundle")

    assert report["num_entities"] == graph.num_nodes
    assert report["num_relations"] == 2 * len(graph.relations)
    assert (tmp_path / "bundle" / "graph.txt").samefile(tmp_path / "bundle" / "train.txt")
    with (tmp_path / "bundle" / "test-queries.pkl").open("rb") as stream:
        exported = pickle.load(stream)
    assert STRUCTURES[QueryType.TWO_HOP_TYPE] in exported
    assert len(exported[STRUCTURES[QueryType.TWO_HOP_TYPE]]) == 1


def test_link_prediction_export_has_disjoint_canonical_splits(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "graph")
    output = tmp_path / "link-prediction"
    report = export_link_prediction_bundle(graph, output, seed=17)
    canonical_count = sum(report["canonical_edge_counts"].values())
    assert canonical_count == graph.edges().num_rows
    assert report["train_triples_with_inverse"] == (2 * report["canonical_edge_counts"]["train"])
    assert (output / "graph.txt").samefile(output / "train.txt")


def test_edge_split_is_deterministic() -> None:
    first = edge_split(1, "presynaptic_to", 2, seed=17)
    second = edge_split(1, "presynaptic_to", 2, seed=17)
    assert first == second
    assert first in {"train", "valid", "test"}
