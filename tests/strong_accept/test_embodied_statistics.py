from __future__ import annotations

import pytest


def _rows():
    rows = []
    for layout, delta in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        for update_seed in (17, 29, 43):
            common = {
                "layout_seed": layout,
                "update_seed": update_seed,
                "condition": "irrelevant_receipt_withdrawal",
                "update_count": 1,
                "relevant_fraction": 0.0,
                "plan_length_stratum": "long",
            }
            rows.append({**common, "method": "full_scan", "metric": 10.0})
            rows.append({**common, "method": "receipt_index", "metric": 10.0 - delta})
    return rows


def test_paired_bootstrap_pairs_cells_and_resamples_layout_clusters():
    from connectomequest.embodied.statistics import paired_layout_bootstrap

    estimate = paired_layout_bootstrap(
        _rows(),
        "receipt_index",
        "full_scan",
        "metric",
        draws=2_000,
        seed=39,
    )
    assert estimate.mean == pytest.approx(-2.5)
    assert estimate.n_layouts == 4
    assert estimate.n_pairs == 12
    assert estimate.lower <= estimate.mean <= estimate.upper


def test_pairing_rejects_missing_or_duplicate_method_cells():
    from connectomequest.embodied.statistics import paired_layout_bootstrap

    rows = _rows()
    with pytest.raises(ValueError, match="unpaired"):
        paired_layout_bootstrap(rows[:-1], "receipt_index", "full_scan", "metric")
    with pytest.raises(ValueError, match="duplicate"):
        paired_layout_bootstrap(rows + [rows[0]], "receipt_index", "full_scan", "metric")


@pytest.mark.parametrize(
    "field,bad,good",
    [
        ("time_reduction", 0.1999, 0.20),
        ("time_reduction_ci_low", 0.10, 0.1001),
        ("work_reduction", 0.2499, 0.25),
        ("total_reasoning_reduction", 0.0, 0.0001),
        ("wall_clock_increase", 0.0101, 0.01),
    ],
)
def test_practical_gate_exact_boundaries(field, bad, good):
    from connectomequest.embodied.statistics import evaluate_practical_gate

    base = {
        "time_reduction": 0.25,
        "time_reduction_ci_low": 0.15,
        "work_reduction": 0.30,
        "total_reasoning_reduction": 0.05,
        "wall_clock_increase": 0.0,
    }
    assert evaluate_practical_gate({**base, field: bad}).passed is False
    assert evaluate_practical_gate({**base, field: good}).passed is True


def test_correctness_gate_uses_strict_noninferiority_bound():
    from connectomequest.embodied.statistics import evaluate_correctness_gate

    base = {
        "suffix_stable_next_action_matches": True,
        "competent_unsupported_authorizations": 0,
    }
    assert (
        evaluate_correctness_gate({**base, "task_success_difference_ci_low": -0.01}).passed is False
    )
    assert (
        evaluate_correctness_gate({**base, "task_success_difference_ci_low": -0.0099}).passed
        is True
    )


def test_integrity_and_safety_are_distinct_gates():
    from connectomequest.embodied.statistics import (
        evaluate_integrity_gate,
        evaluate_safety_gate,
    )

    integrity = evaluate_integrity_gate(
        {
            "source_hashes_match": True,
            "pre_update_access_match": True,
            "pre_update_action_prefix_match": True,
            "update_schedules_match": True,
            "trace_replay_rate": 1.0,
            "oracle_fields_exposed": False,
        }
    )
    safety = evaluate_safety_gate(
        {
            "competent_unsupported_authorizations": 0,
            "receipt_index_unsupported_rate": 0.0,
            "unchecked_reuse_unsupported_rate": 0.2,
        }
    )
    assert integrity.passed is True
    assert safety.passed is True
    assert integrity.name != safety.name


def test_summary_keeps_failures_and_unexposed_rows_in_denominators():
    from connectomequest.embodied.statistics import summarize_confirmation

    rows = []
    for method in ("full_scan", "receipt_index"):
        rows.extend(
            [
                {
                    "layout_seed": 1,
                    "update_seed": 17,
                    "condition": "irrelevant_receipt_withdrawal",
                    "update_count": 1,
                    "relevant_fraction": 0.0,
                    "plan_length_stratum": "long",
                    "method": method,
                    "task_success": True,
                    "exposed_update_count": 1,
                    "predicate_evaluations": 10 if method == "full_scan" else 2,
                    "successor_expansions": 0,
                    "checking_planning_ns": 100 if method == "full_scan" else 50,
                    "total_reasoning_ns": 100 if method == "full_scan" else 60,
                    "total_wall_clock_ns": 1000,
                },
                {
                    "layout_seed": 2,
                    "update_seed": 17,
                    "condition": "irrelevant_receipt_withdrawal",
                    "update_count": 1,
                    "relevant_fraction": 0.0,
                    "plan_length_stratum": "long",
                    "method": method,
                    "task_success": False,
                    "exposed_update_count": 0,
                    "predicate_evaluations": 10 if method == "full_scan" else 2,
                    "successor_expansions": 0,
                    "checking_planning_ns": 100 if method == "full_scan" else 50,
                    "total_reasoning_ns": 100 if method == "full_scan" else 60,
                    "total_wall_clock_ns": 1000,
                },
            ]
        )
    summary = summarize_confirmation(rows, draws=100, seed=3)
    assert summary["row_counts"]["all"] == 4
    assert summary["row_counts"]["failed"] == 2
    assert summary["row_counts"]["unexposed"] == 2
    assert summary["primary_stratum"]["paired_cells"] == 2
