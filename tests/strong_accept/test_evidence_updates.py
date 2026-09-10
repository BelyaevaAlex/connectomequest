from __future__ import annotations

import inspect

import pytest

from connectomequest.embodied.evidence_index import PlanDependency


def _api():
    from connectomequest.embodied import updates as api

    return api


def _prefix(api, *, include_irrelevant=True, alternative=True):
    bindings = [
        ("a:key", ("r-key", "r-key-alt") if alternative else ("r-key",)),
        ("a:door", ("r-door",)),
    ]
    active = ["r-key", "r-door"] + (["r-correction"] if include_irrelevant else [])
    if alternative:
        active.append("r-key-alt")
    if include_irrelevant:
        active.extend(("r-irrelevant-1", "r-irrelevant-2", "r-irrelevant-3"))
    dependencies = (
        PlanDependency(
            0,
            frozenset({"r-key"}),
            frozenset({"cell:key"}),
            frozenset({"a:key"}),
        ),
        PlanDependency(
            1,
            frozenset({"r-door"}),
            frozenset({"cell:door"}),
            frozenset({"a:door"}),
        ),
    )
    return api.SchedulePrefix(
        action_count=5,
        active_receipt_ids=tuple(active),
        assertion_to_receipt_bindings=tuple(bindings),
        remaining_dependencies=dependencies,
        correction_bindings=(
            api.CorrectionBinding(
                semantic_key="cell:door",
                old_assertion="a:door",
                new_assertion="a:door=open",
                receipt_id="r-correction",
            ),
        ),
    )


def test_schedule_does_not_accept_method_outcomes():
    api = _api()
    signature = inspect.signature(api.build_update_schedule)
    assert "method" not in signature.parameters
    assert "result" not in signature.parameters
    assert "success" not in signature.parameters


@pytest.mark.parametrize(
    "condition",
    (
        "none",
        "irrelevant_receipt_withdrawal",
        "alternative_support",
        "final_support_loss",
        "semantic_correction",
        "sparse_burst",
    ),
)
def test_every_schedule_is_deterministic(condition):
    api = _api()
    prefix = _prefix(api)
    kwargs = dict(
        layout_seed=39_000_001,
        prefix=prefix,
        update_seed=17,
        condition=condition,
        count=0 if condition == "none" else (4 if condition == "sparse_burst" else 1),
        relevant_fraction=0.5 if condition == "sparse_burst" else 0.0,
    )
    assert api.build_update_schedule(**kwargs) == api.build_update_schedule(**kwargs)


def test_irrelevant_update_intersects_no_remaining_dependency():
    api = _api()
    prefix = _prefix(api)
    schedule = api.build_update_schedule(1, prefix, 17, "irrelevant_receipt_withdrawal", 1, 0.0)
    dependent = set().union(*(row.required_receipts for row in prefix.remaining_dependencies))
    assert schedule.exclusion_reason is None
    assert not (set(schedule.events[0].withdrawn_receipt_ids) & dependent)
    assert schedule.audit[0].relevant is False


def test_alternative_support_leaves_an_active_receipt():
    api = _api()
    prefix = _prefix(api)
    schedule = api.build_update_schedule(1, prefix, 17, "alternative_support", 1, 1.0)
    event = schedule.events[0]
    bindings = dict(prefix.assertion_to_receipt_bindings)
    remaining = set(bindings["a:key"]) - set(event.withdrawn_receipt_ids)
    assert schedule.exclusion_reason is None
    assert remaining & set(prefix.active_receipt_ids)


def test_final_support_loss_withdraws_last_active_support():
    api = _api()
    prefix = _prefix(api)
    schedule = api.build_update_schedule(1, prefix, 17, "final_support_loss", 1, 1.0)
    event = schedule.events[0]
    bindings = dict(prefix.assertion_to_receipt_bindings)
    affected = schedule.audit[0].assertion
    assert affected is not None
    assert set(bindings[affected]) <= set(event.withdrawn_receipt_ids)
    assert schedule.audit[0].relevant is True


def test_semantic_correction_uses_only_prefix_acquired_receipt():
    api = _api()
    prefix = _prefix(api)
    schedule = api.build_update_schedule(1, prefix, 17, "semantic_correction", 1, 1.0)
    event = schedule.events[0]
    added = dict(event.added_assertion_bindings)
    assert schedule.exclusion_reason is None
    assert added == {"a:door=open": ("r-correction",)}
    assert set(added["a:door=open"]) <= set(prefix.active_receipt_ids)
    assert event.changed_semantic_keys == ("cell:door",)


def test_sparse_burst_has_predeclared_relevance_count_and_public_events_hide_audit():
    api = _api()
    prefix = _prefix(api)
    schedule = api.build_update_schedule(1, prefix, 29, "sparse_burst", 4, 0.5)
    assert schedule.exclusion_reason is None
    assert len(schedule.events) == 4
    assert sum(row.relevant for row in schedule.audit) == 2
    assert all(not hasattr(event, "generator_audit_label") for event in schedule.events)
    assert [event.action_index for event in schedule.events] == sorted(
        event.action_index for event in schedule.events
    )


@pytest.mark.parametrize(
    "prefix_kwargs,condition,reason",
    [
        ({"include_irrelevant": False}, "irrelevant_receipt_withdrawal", "no_irrelevant_receipt"),
        ({"alternative": False}, "alternative_support", "no_alternative_support"),
    ],
)
def test_infeasible_schedules_are_recorded_not_substituted(prefix_kwargs, condition, reason):
    api = _api()
    schedule = api.build_update_schedule(1, _prefix(api, **prefix_kwargs), 17, condition, 1, 1.0)
    assert schedule.events == ()
    assert schedule.exclusion_reason == reason


def test_property_determinism_over_one_hundred_seeds():
    api = _api()
    prefix = _prefix(api)
    for seed in range(100):
        first = api.build_update_schedule(10 + seed, prefix, seed, "sparse_burst", 4, 0.5)
        second = api.build_update_schedule(10 + seed, prefix, seed, "sparse_burst", 4, 0.5)
        assert first == second
