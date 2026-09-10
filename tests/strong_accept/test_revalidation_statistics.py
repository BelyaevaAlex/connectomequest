from __future__ import annotations

import pytest


def test_efficiency_gate_uses_fully_accounted_reduction():
    from connectomequest.embodied.revalidation_statistics import evaluate_v40_gates

    report = evaluate_v40_gates(
        reduction=0.25,
        lower=0.12,
        decision_agreement=1.0,
        work_reduction=0.80,
        eligible_long=160,
    )

    assert report["correctness"]["passed"] is True
    assert report["efficiency"]["passed"] is True


def test_efficiency_gate_rejects_unaccounted_rows():
    from connectomequest.embodied.revalidation_statistics import summarize_fully_accounted

    rows = [
        {
            "layout_seed": 1,
            "update_seed": 17,
            "update_count": 4,
            "plan_length": 64,
            "method": "complete_validation",
            "validation_ns": 10,
        }
    ]

    with pytest.raises(ValueError, match="fully accounted"):
        summarize_fully_accounted(rows)


def test_primary_stratum_excludes_short_or_single_update_cells():
    from connectomequest.embodied.revalidation_statistics import primary_rows

    rows = [
        {"plan_length": 63, "update_count": 8},
        {"plan_length": 64, "update_count": 1},
        {"plan_length": 64, "update_count": 4},
        {"plan_length": 80, "update_count": 8},
    ]

    assert primary_rows(rows) == rows[2:]
