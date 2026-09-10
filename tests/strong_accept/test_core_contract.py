from connectomequest.agents.oracle import oracle_proof
from connectomequest.env import Action, ActionType, ConnectomeEnv
from connectomequest.proof import Evidence, EvidenceKind, Proof, ProofValidator
from connectomequest.query import Query, QueryType
from connectomequest.schema import Relation
from connectomequest.toy import make_toy_graph


def _two_hop_query() -> Query:
    return Query("q", QueryType.TWO_HOP, (0,), frozenset({4}), "test")


def test_core_validator_rejects_foreign_query_and_forged_token(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = _two_hop_query()
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8)
    env.reset(query)
    env.step(Action(ActionType.INSPECT, relation=str(Relation.PRESYNAPTIC_TO)))
    env.step(Action(ActionType.PROBE_AFFORDANCE, relation=str(Relation.PRESYNAPTIC_TO), target=2))
    env.step(Action(ActionType.TRAVERSE, relation=str(Relation.PRESYNAPTIC_TO), target=2))
    env.step(Action(ActionType.INSPECT, relation=str(Relation.PRESYNAPTIC_TO)))
    valid = env.step(Action(ActionType.SUBMIT, answers=frozenset({4})))
    assert valid.validation is not None and valid.validation.valid

    evidence = list(env.state.discovered.values())
    foreign = Proof("foreign", {4}, evidence)
    strict = env.validator.validate(
        query,
        foreign,
        require_complete=False,
        active_query_id=query.query_id,
        episode_nonce=env.episode_nonce,
        issued_token_ids=env.issued_token_ids,
        current_step=env.state.step,
    )
    assert not strict.valid
    forged = Evidence(
        EvidenceKind.EDGE,
        0,
        str(Relation.PRESYNAPTIC_TO),
        2,
        source_record=env.evidence_source_record,
        token_id="forged-token",
        query_id=query.query_id,
        episode_nonce=env.episode_nonce,
        issued_step=1,
    )
    forged_result = env.validator.validate(
        query,
        Proof(query.query_id, {4}, evidence[:-1] + [forged]),
        require_complete=False,
        active_query_id=query.query_id,
        episode_nonce=env.episode_nonce,
        issued_token_ids=env.issued_token_ids,
        current_step=env.state.step,
    )
    assert not forged_result.valid


def test_core_validator_rejects_stale_nonce_and_foreign_graph_record(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = _two_hop_query()
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8)
    env.reset(query)
    env.step(Action(ActionType.INSPECT, relation=str(Relation.PRESYNAPTIC_TO)))
    evidence = list(env.state.discovered.values())
    proof = Proof(query.query_id, set(), evidence)
    stale = env.validator.validate(
        query,
        proof,
        require_complete=False,
        active_query_id=query.query_id,
        episode_nonce="stale",
        issued_token_ids=env.issued_token_ids,
        current_step=env.state.step,
    )
    assert not stale.valid
    foreign = Evidence(
        evidence[0].kind,
        evidence[0].src,
        evidence[0].relation,
        evidence[0].dst,
        source_record="other:graph@sha256:deadbeef",
        token_id=evidence[0].token_id,
        query_id=query.query_id,
        episode_nonce=env.episode_nonce,
        issued_step=evidence[0].issued_step,
    )
    wrong_graph = env.validator.validate(
        query,
        Proof(query.query_id, set(), [foreign]),
        require_complete=False,
        active_query_id=query.query_id,
        episode_nonce=env.episode_nonce,
        issued_token_ids=env.issued_token_ids,
        current_step=env.state.step,
    )
    assert not wrong_graph.valid


def test_check_edge_rejects_unobserved_claim(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.INTERSECTION_NEGATION, (0, 1), frozenset({2}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=1, max_steps=8)
    env.reset(query)
    result = env.step(
        Action(
            ActionType.CHECK_EDGE,
            relation=str(Relation.PRESYNAPTIC_TO),
            target=4,
        )
    )
    assert result.error == "check_edge target is not in the observed claim set"


def test_oracle_legacy_validator_remains_compatible(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = _two_hop_query()
    proof = oracle_proof(graph, query)
    assert ProofValidator(graph).validate(query, proof).valid


def test_two_hop_semantic_check_is_native_and_claim_masked(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=16)
    env.reset(query)
    env.step(Action(ActionType.INSPECT, relation=str(Relation.PRESYNAPTIC_TO)))
    env.step(Action(ActionType.TRAVERSE, relation=str(Relation.PRESYNAPTIC_TO), target=2))
    env.step(Action(ActionType.INSPECT, relation=str(Relation.PRESYNAPTIC_TO)))
    env.step(Action(ActionType.TRAVERSE, relation=str(Relation.PRESYNAPTIC_TO), target=4))
    illegal = env.step(
        Action(ActionType.CHECK_EDGE, relation=str(Relation.PRESYNAPTIC_TO), target=6)
    )
    assert illegal.error == "check_edge relation is not the query semantic predicate"
    assert env.legal_check_claims() == ((4, str(Relation.HAS_TYPE), 6),)
