from copy import deepcopy

import pytest

from scripts.audit_paper_integrity import validate_cells, validate_queries, validate_summary


def fixture_rows():
    return [
        dict(method=m, seed=s, query_id=q, goal_success=int(q == "a"), budget_used=4, steps=5)
        for m, seeds in [("weight", [17]), ("learned", [17, 29, 43])]
        for s in seeds
        for q in ["a", "b"]
    ]


def check(rows):
    return validate_cells(rows, ["weight", "learned"], [17, 29, 43], ["a", "b"], 64, 64)


def test_complete_matrix_accepts_one_deterministic_seed():
    cells = check(fixture_rows())
    assert len(cells) == 4
    validate_summary(cells, {"weight": 0.5, "learned": 0.5})


@pytest.mark.parametrize(
    "fault",
    [
        "missing_seed",
        "missing_method",
        "duplicate",
        "foreign_query",
        "foreign_seed",
        "nan",
        "budget",
        "steps",
    ],
)
def test_corrupt_matrix_is_rejected(fault):
    rows = fixture_rows()
    if fault == "missing_seed":
        rows = [r for r in rows if r["seed"] != 43]
    elif fault == "missing_method":
        rows = [r for r in rows if r["method"] != "learned"]
    elif fault == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif fault == "foreign_query":
        rows[-1]["query_id"] = "c"
    elif fault == "foreign_seed":
        rows[-1]["seed"] = 99
    elif fault == "nan":
        rows[-1]["goal_success"] = float("nan")
    elif fault == "budget":
        rows[-1]["budget_used"] = 65
    else:
        rows[-1]["steps"] = 65
    with pytest.raises(ValueError):
        check(rows)


def test_stale_summary_rejected_even_when_rows_are_intact():
    with pytest.raises(ValueError, match="summary"):
        validate_summary(check(fixture_rows()), {"weight": 0.5, "learned": 0.6})


def test_stale_nested_budget_or_failure_rate_rejected():
    rows = fixture_rows()
    for r in rows:
        r.update(
            primary_success=r["goal_success"],
            distinct_middles=1,
            failure_category="success" if r["goal_success"] else "b_no_useful_middle_traversed",
        )
    cells = check(rows)
    summary = {
        m: dict(
            success=0.5,
            primary_success=0.5,
            budget=4,
            distinct_middles=1,
            episodes=2 if m == "weight" else 6,
            failure_rates={"success": 0.5, "b_no_useful_middle_traversed": 0.5},
        )
        for m in ["weight", "learned"]
    }
    validate_summary(cells, summary)
    for field, value in [("budget", 5), ("failure_rates", {"success": 1.0})]:
        wrong = deepcopy(summary)
        wrong["learned"][field] = value
        with pytest.raises(ValueError, match="summary"):
            validate_summary(cells, wrong)


@pytest.mark.parametrize(
    "fault", ["none", "prior_overlap", "pair_duplicate", "id_duplicate", "wrong_split"]
)
def test_new_pair_identity_and_exclusion(fault):
    queries = [
        dict(query_id="a", anchors=[1, 9], answers=[3]),
        dict(query_id="b", anchors=[2, 9], answers=[4]),
    ]
    prior = {(7, 9)}
    if fault == "prior_overlap":
        prior.add((1, 9))
    elif fault == "pair_duplicate":
        queries[1]["anchors"] = [1, 9]
    elif fault == "id_duplicate":
        queries[1]["query_id"] = "a"
    split = lambda anchor: "train" if fault == "wrong_split" else "test"
    if fault == "none":
        assert validate_queries(queries, 2, prior, split) == ["a", "b"]
    else:
        with pytest.raises(ValueError):
            validate_queries(queries, 2, prior, split)
