from __future__ import annotations

import pytest


def test_costs_are_conditioned_without_dropping_failures():
    from connectomequest.graph_reporting import summarize_policy_costs

    rows = [
        {
            "graph": "g",
            "policy": "p",
            "query_id": "q1",
            "seed": 1,
            "success": 1,
            "actions": 4,
            "budget": 8,
        },
        {
            "graph": "g",
            "policy": "p",
            "query_id": "q2",
            "seed": 1,
            "success": 0,
            "actions": 8,
            "budget": 8,
        },
    ]

    out = summarize_policy_costs(rows)[("g", "p")]

    assert out == {
        "n": 2,
        "actions": 6.0,
        "actions_success": 4.0,
        "actions_failure": 8.0,
        "budget_exhaustion": 0.5,
    }


def test_empty_conditional_subset_is_preserved_as_none():
    from connectomequest.graph_reporting import summarize_policy_costs

    rows = [
        {
            "graph": "g",
            "policy": "p",
            "query_id": "q1",
            "seed": 1,
            "success": 1,
            "actions": 4,
            "budget": 8,
        },
    ]

    out = summarize_policy_costs(rows)[("g", "p")]

    assert out["actions_failure"] is None


def test_operational_reserve_can_define_exhaustion_before_nominal_budget():
    from connectomequest.graph_reporting import summarize_policy_costs

    rows = [
        {
            "graph": "g",
            "policy": "p",
            "query_id": "q1",
            "seed": 1,
            "success": 0,
            "actions": 7,
            "budget": 8,
            "exhaustion_at": 7,
        },
    ]

    assert summarize_policy_costs(rows)[("g", "p")]["budget_exhaustion"] == 1.0


def test_duplicate_episode_policy_rows_are_rejected():
    from connectomequest.graph_reporting import summarize_policy_costs

    row = {
        "graph": "g",
        "policy": "p",
        "query_id": "q1",
        "seed": 1,
        "success": 1,
        "actions": 4,
        "budget": 8,
    }

    with pytest.raises(ValueError, match="duplicate"):
        summarize_policy_costs([row, dict(row)])
