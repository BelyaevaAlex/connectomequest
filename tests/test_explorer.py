from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.env import ConnectomeEnv
from connectomequest.evaluation import episode_metrics
from connectomequest.query import Query, QueryType
from connectomequest.toy import make_toy_graph


def test_intersection_explorer_produces_valid_proof(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.INTERSECTION, (0, 1), frozenset({3}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=8)
    env.reset(query)
    result = BudgetedExplorer().run(env, query.spec)
    metrics = episode_metrics(query, result)
    assert metrics.goal_success
    assert metrics.proof_valid
    assert metrics.invalid_actions == 0
    assert metrics.provenance_completeness == 1.0


def test_two_hop_explorer_produces_valid_proof(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP, (0,), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=12)
    env.reset(query)
    result = BudgetedExplorer().run(env, query.spec)
    assert episode_metrics(query, result).goal_success


def test_typed_two_hop_explorer_produces_semantic_proof(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=16)
    env.reset(query)
    result = BudgetedExplorer().run(env, query.spec)
    assert episode_metrics(query, result).goal_success


def test_negation_explorer_produces_certificates(tmp_path):
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.INTERSECTION_NEGATION, (0, 1), frozenset({2}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=10)
    env.reset(query)
    result = BudgetedExplorer().run(env, query.spec)
    assert episode_metrics(query, result).goal_success


def test_public_query_spec_hides_answers() -> None:
    query = Query("q", QueryType.ONE_HOP, (0,), frozenset({1}), "test")
    assert not hasattr(query.spec, "answers")


def test_empty_failed_submission_is_not_a_valid_proof(tmp_path) -> None:
    graph = make_toy_graph(tmp_path / "toy")
    query = Query("q", QueryType.TWO_HOP_TYPE, (0, 6), frozenset({4}), "test")
    env = ConnectomeEnv(graph, visible_neighbors=8, max_steps=4, budget=4)
    env.reset(query)
    metrics = episode_metrics(query, BudgetedExplorer().run(env, query.spec))
    assert not metrics.goal_success
