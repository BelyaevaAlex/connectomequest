from __future__ import annotations

import pytest


def test_confirmation_population_is_fixed_and_disjoint():
    from connectomequest.embodied.revalidation_benchmark import AmortizedProtocol

    confirmation = AmortizedProtocol.confirmation()
    development = AmortizedProtocol.development()

    assert confirmation.layout_seeds == tuple(range(40_040_000, 40_040_256))
    assert confirmation.update_counts == (1, 4, 8)
    assert set(confirmation.layout_seeds).isdisjoint(development.layout_seeds)


def test_total_reasoning_includes_all_index_costs():
    from connectomequest.embodied.revalidation_benchmark import TimingRow

    row = TimingRow(
        index_build_ns=2,
        index_maintenance_ns=3,
        validation_ns=5,
        planning_ns=7,
        total_reasoning_ns=17,
        raw_repetitions_ns=(17,),
    )

    row.validate()


def test_timing_row_rejects_incomplete_accounting():
    from connectomequest.embodied.revalidation_benchmark import TimingRow

    row = TimingRow(
        index_build_ns=2,
        index_maintenance_ns=3,
        validation_ns=5,
        planning_ns=7,
        total_reasoning_ns=16,
        raw_repetitions_ns=(16,),
    )

    with pytest.raises(ValueError, match="accounting"):
        row.validate()


def test_case_stream_contains_only_irrelevant_receipts():
    from connectomequest.embodied.evidence_index import PlanDependency
    from connectomequest.embodied.revalidation_benchmark import prepare_case_stream

    dependencies = (
        PlanDependency(0, frozenset({"needed"}), frozenset()),
        PlanDependency(1, frozenset(), frozenset()),
    )

    stream = prepare_case_stream(
        dependencies=dependencies,
        active_receipts=("needed", "other-a", "other-b"),
        update_count=2,
        update_seed=17,
    )

    assert len(stream) == 2
    assert {event.withdrawn_receipt_ids[0] for event in stream} == {
        "other-a",
        "other-b",
    }


def test_case_stream_rejects_insufficient_irrelevant_receipts():
    from connectomequest.embodied.evidence_index import PlanDependency
    from connectomequest.embodied.revalidation_benchmark import prepare_case_stream

    dependencies = (PlanDependency(0, frozenset({"needed"}), frozenset()),)

    with pytest.raises(ValueError, match="insufficient"):
        prepare_case_stream(
            dependencies=dependencies,
            active_receipts=("needed", "other"),
            update_count=2,
            update_seed=17,
        )


def test_frozen_payload_declares_the_primary_analysis_and_hashes_it():
    from scripts.freeze_revalidation_protocol import payload

    frozen = payload("confirmation")

    assert (
        frozen["primary_population"]
        == "eligible plans of length at least 64 with 4 or 8 distinct irrelevant receipt withdrawals"
    )
    assert "src/connectomequest/embodied/revalidation_statistics.py" in frozen["source_hashes"]
    assert "scripts/report_revalidation_benchmark.py" in frozen["source_hashes"]


def test_serial_timing_process_is_pinned_when_supported():
    from scripts.run_revalidation_benchmark import pin_serial_timing_process

    metadata = pin_serial_timing_process()

    assert metadata["selected_cpu"] is not None
    assert metadata["affinity_after"] == [metadata["selected_cpu"]]
