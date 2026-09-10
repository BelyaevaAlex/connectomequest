from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_load_rows_rejects_changed_payload(tmp_path: Path):
    from scripts.report_revalidation_benchmark import load_verified_rows

    stage = tmp_path / "confirmation"
    cases = stage / "cases"
    cases.mkdir(parents=True)
    protocol = {"protocol": {"stage": "confirmation"}}
    (stage / "protocol.json").write_bytes(_canonical(protocol) + b"\n")
    result = {"method": "complete_validation"}
    wrapper = {
        "protocol_sha256": hashlib.sha256((stage / "protocol.json").read_bytes()).hexdigest(),
        "payload_sha256": "not-the-result-digest",
        "result": result,
    }
    (cases / "one.json").write_bytes(_canonical(wrapper) + b"\n")

    with pytest.raises(ValueError, match="payload digest"):
        load_verified_rows(stage, validate_timing_rows=False)


def test_exact_binomial_interval_handles_boundaries():
    from scripts.report_revalidation_benchmark import exact_binomial_interval

    low_zero, high_zero = exact_binomial_interval(0, 513)
    low_all, high_all = exact_binomial_interval(513, 513)

    assert low_zero == 0.0
    assert 0.0 < high_zero < 0.01
    assert 0.99 < low_all < 1.0
    assert high_all == 1.0


def test_json_repetitions_are_normalized_at_loader_boundary(tmp_path: Path):
    from scripts.report_revalidation_benchmark import load_verified_rows

    stage = tmp_path / "confirmation"
    cases = stage / "cases"
    cases.mkdir(parents=True)
    protocol = {"protocol": {"stage": "confirmation"}}
    (stage / "protocol.json").write_bytes(_canonical(protocol) + b"\n")
    result = {
        "method": "complete_validation",
        "layout_seed": 1,
        "update_seed": 17,
        "plan_length": 64,
        "update_count": 4,
        "index_build_ns": 0,
        "index_maintenance_ns": 0,
        "validation_ns": 5,
        "planning_ns": 0,
        "total_reasoning_ns": 5,
        "raw_repetitions_ns": [5, 6],
        "method_input_sha256": "input",
        "decision_sha256": "decision",
        "predicate_evaluations": 1,
        "plan_position_visits": 1,
        "index_lookups": 0,
        "posting_entries": 0,
    }
    wrapper = {
        "protocol_sha256": hashlib.sha256((stage / "protocol.json").read_bytes()).hexdigest(),
        "payload_sha256": hashlib.sha256(_canonical(result)).hexdigest(),
        "result": result,
    }
    (cases / "one.json").write_bytes(_canonical(wrapper) + b"\n")

    _, rows = load_verified_rows(stage)

    assert rows == [result]


def test_rendered_caption_separates_interventions():
    from scripts.report_revalidation_benchmark import render_figure_tex

    text = render_figure_tex(
        {
            "eligible_long": 128,
            "paired_cells": 512,
            "time_reduction": 0.5,
            "time_reduction_ci": [0.4, 0.6],
            "decision_agreement": 1.0,
        }
    )

    assert "distinct interventions" in text
    assert "irrelevant receipt withdrawals" in text
    assert "final-support loss" in text
