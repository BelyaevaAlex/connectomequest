from connectomequest.agents.oracle import oracle_proof
from connectomequest.env import Action, ActionType, ConnectomeEnv
from connectomequest.proof import Evidence, EvidenceKind, Proof, ProofValidator
from connectomequest.query import Query, QueryType
from connectomequest.schema import Relation
from connectomequest.toy import make_toy_graph


def test_partially_observable_episode_and_valid_proof(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.INTERSECTION, (0, 1), frozenset({3}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8)
    observation = env.reset(query)
    assert observation.discovered_edges == ()
    assert observation.memory == ((0, 0), (1, 1))
    env.step(Action(ActionType.INSPECT, relation=str(Relation.PRESYNAPTIC_TO)))
    env.step(Action(ActionType.SWITCH, slot=1))
    env.step(Action(ActionType.CHECK_EDGE, relation=str(Relation.PRESYNAPTIC_TO), target=3))
    result = env.step(Action(ActionType.SUBMIT, answers=frozenset({3})))
    assert result.terminated
    assert result.validation is not None
    assert result.validation.valid


def test_affordance_probe_is_budgeted_id_free_and_not_proof_evidence(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP, (0,), frozenset({3, 4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8)
    env.reset(query)
    relation = str(Relation.PRESYNAPTIC_TO)
    inspected = env.step(Action(ActionType.INSPECT, relation=relation))
    target = inspected.observation.discovered_edges[0].dst
    evidence_before = inspected.observation.discovered_edges

    probed = env.step(Action(ActionType.PROBE_AFFORDANCE, relation=relation, target=target))

    assert probed.error is None
    assert probed.observation.remaining_budget == 6.0
    assert probed.observation.discovered_edges == evidence_before
    assert len(probed.observation.node_sketches) == 1
    sketch = probed.observation.node_sketches[0]
    assert sketch.node_id == target
    assert sketch.outgoing_degree >= 0
    edge_artifact = next(item for item in graph.manifest.artifacts if item.path == "edges.parquet")
    expected_suffix = "edges.parquet@sha256:" + edge_artifact.sha256
    assert sketch.source_record.endswith(expected_suffix)


def test_affordance_probe_rejects_an_unobserved_node(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP, (0,), frozenset({3, 4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8)
    env.reset(query)

    result = env.step(
        Action(
            ActionType.PROBE_AFFORDANCE,
            relation=str(Relation.PRESYNAPTIC_TO),
            target=999,
        )
    )

    assert result.error == "probe target is not the current or an observed frontier node"
    assert result.observation.node_sketches == ()
    assert result.observation.discovered_edges == ()


def test_negation_requires_non_edge_certificate(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.INTERSECTION_NEGATION, (0, 1), frozenset({2}), "test")
    proof = Proof(
        "q",
        {2},
        [
            Evidence(
                EvidenceKind.EDGE,
                0,
                str(Relation.PRESYNAPTIC_TO),
                2,
            ),
            Evidence(
                EvidenceKind.NON_EDGE,
                1,
                str(Relation.PRESYNAPTIC_TO),
                2,
            ),
        ],
    )
    assert ProofValidator(graph).validate(query, proof).valid


def test_typed_two_hop_oracle_produces_complete_valid_proof(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")

    proof = oracle_proof(graph, query)
    assert ProofValidator(graph).validate(query, proof).valid
