from __future__ import annotations

import pytest

from connectomequest.validation_report_v2 import paired_bootstrap_ci, render_markdown


def test_paired_bootstrap_is_deterministic_and_query_aligned() -> None:
    candidate = {"q2": 0.0, "q1": 1.0, "q3": 1.0, "q4": 0.0}
    reference = {"q1": 0.0, "q2": 0.0, "q3": 1.0, "q4": 1.0}
    first = paired_bootstrap_ci(candidate, reference, seed=7, resamples=200)
    second = paired_bootstrap_ci(candidate, reference, seed=7, resamples=200)
    assert first == second
    assert first["estimate"] == pytest.approx(0.0)
    assert first["queries"] == 4


def test_paired_bootstrap_rejects_query_mismatch() -> None:
    with pytest.raises(ValueError, match="query_id mismatch"):
        paired_bootstrap_ci({"q1": 1.0}, {"q2": 0.0}, resamples=10)


def test_markdown_is_explicitly_development_only() -> None:
    report = {
        "input_fingerprint_sha256": "a" * 64,
        "baseline_aggregates": [],
        "v2_aggregates": [],
        "primary_paired_comparisons": [],
    }
    rendered = render_markdown(report)
    assert "Development only" in rendered
    assert "no confirmatory/test access" in rendered
