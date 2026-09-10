import pytest

from connectomequest.evaluation import (
    aggregate_frontier_ranks,
    aggregate_ranks,
    filtered_ranks,
    raw_ranks,
)


def test_filtered_ranks_remove_other_answers() -> None:
    ranked = [4, 2, 7, 1, 6, 3, 5, 0]
    answers = frozenset({1, 2, 3})

    assert filtered_ranks(ranked, answers) == [2, 3, 4]


def test_aggregate_ranks() -> None:
    metrics = aggregate_ranks([1, 2, 10, 100], queries=2)

    assert metrics["answers"] == 4
    assert metrics["hits_at_1"] == 0.25
    assert metrics["hits_at_10"] == 0.75


def test_raw_ranks_keep_other_answers() -> None:
    ranked = [4, 2, 7, 1, 6, 3, 5, 0]
    answers = frozenset({1, 2, 3})

    assert sorted(raw_ranks(ranked, answers)) == [2, 4, 6]


def test_aggregate_frontier_ranks_separates_three_views() -> None:
    metrics = aggregate_frontier_ranks(
        [1, 3, 2],
        [1, 2, 2],
        [1, 2],
        candidate_counts=[4, 4],
        relevant_counts=[2, 1],
    )

    assert metrics["raw"]["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)
    assert metrics["filtered"]["filtered_mrr"] == pytest.approx((1 + 1 / 2 + 1 / 2) / 3)
    assert metrics["first_proof"]["hits_at_1"] == 0.5
