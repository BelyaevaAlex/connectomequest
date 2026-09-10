import pyarrow as pa

from connectomequest.agents.oracle import oracle_proof
from connectomequest.graph import GraphStore
from connectomequest.proof import ProofValidator
from connectomequest.query import Query, QueryType, execute, graph_semantic_relation
from connectomequest.schema import EDGE_SCHEMA, NODE_SCHEMA, NodeType, Relation


def _layer_graph(tmp_path) -> GraphStore:
    nodes = pa.Table.from_pylist(
        [
            {
                "node_id": index,
                "external_id": label,
                "node_type": str(node_type),
                "dataset": "test",
                "attrs_json": "{}",
            }
            for index, (label, node_type) in enumerate(
                [
                    ("anchor", NodeType.NEURON),
                    ("middle", NodeType.NEURON),
                    ("answer", NodeType.NEURON),
                    ("L3", NodeType.CORTICAL_LAYER),
                ]
            )
        ],
        schema=NODE_SCHEMA,
    )
    edges = pa.Table.from_pylist(
        [
            {
                "src": src,
                "dst": dst,
                "relation": str(relation),
                "weight": 1.0,
                "source_record": "test",
            }
            for src, dst, relation in [
                (0, 1, Relation.PRESYNAPTIC_TO),
                (1, 2, Relation.PRESYNAPTIC_TO),
                (2, 3, Relation.IN_CORTICAL_LAYER),
            ]
        ],
        schema=EDGE_SCHEMA,
    )
    return GraphStore.create(
        tmp_path / "layer-graph",
        nodes,
        edges,
        dataset="test",
        version="1",
        license_name="synthetic",
    )


def test_two_hop_semantic_query_preserves_layer_predicate(tmp_path) -> None:
    graph = _layer_graph(tmp_path)
    relation = str(Relation.IN_CORTICAL_LAYER)
    query = Query(
        "layer-query",
        QueryType.TWO_HOP_TYPE,
        (0, 3),
        frozenset({2}),
        "test",
        relation,
    )

    assert graph_semantic_relation(graph) == relation
    assert execute(graph, query.query_type, query.anchors, semantic_relation=relation) == {2}
    proof = oracle_proof(graph, query)
    assert ProofValidator(graph).validate(query, proof).valid
    assert any(item.relation == relation for item in proof.evidence)
