import numpy as np
import pytest

from connectomequest.embodied.dynamic_task import (
    CONDITIONS,
    DynamicCase,
    FrozenPerceptionCache,
    access_signature,
    build_condition,
    enroll_case,
    execute_case,
    plan_dependencies,
)
from connectomequest.embodied.indexed_revalidation import METHODS
from connectomequest.embodied.navigation import PlanStep


class CountingPerceptor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, rgb: np.ndarray):
        self.calls += 1
        return [(0, 0, "empty", float(rgb.mean()))]


def test_perception_cache_is_keyed_by_rgb_and_checkpoint_hash() -> None:
    perceptor = CountingPerceptor()
    cache = FrozenPerceptionCache(perceptor, "a" * 64)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    first = cache.predict(rgb)
    second = cache.predict(rgb.copy())

    assert first == second == ((0, 0, "empty", 0.0),)
    assert perceptor.calls == 1
    assert cache.entries == 1


def test_perception_cache_rejects_checkpoint_drift() -> None:
    cache = FrozenPerceptionCache(CountingPerceptor(), "a" * 64)

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        cache.verify_checkpoint("b" * 64)


def test_plan_dependencies_bind_assertions_to_rgb_receipts() -> None:
    plan = (
        PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),
        PlanStep("left", (1, 0), 0, ()),
        PlanStep("forward", (1, 0), 3, (((1, -1), "empty"),)),
    )

    dependencies = plan_dependencies(
        plan,
        support={(1, 0): "rgb-1", (1, -1): "rgb-2"},
    )

    assert dependencies[0].required_receipts == frozenset({"rgb-1"})
    assert dependencies[0].required_assertions == frozenset({"cell:1,0=empty"})
    assert dependencies[1].required_receipts == frozenset()
    assert dependencies[2].required_receipts == frozenset({"rgb-2"})


def test_missing_support_for_a_required_assertion_is_rejected() -> None:
    plan = (PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),)

    with pytest.raises(ValueError, match="missing receipt support"):
        plan_dependencies(plan, support={})


def test_access_signature_ignores_method_and_outcome_fields() -> None:
    common = {
        "seed": 7,
        "corruption_seed": 17,
        "rgb_receipts": ["r0", "r1"],
        "assertion_updates": [{"withdrawn": ["r0"]}],
        "active_receipts": ["r1"],
        "assertion_receipts": {"cell:0,0=empty": ["r1"]},
        "belief_cells": [[0, 0, "empty"]],
        "pose": [[0, 0], 1],
        "inventory": None,
        "plan_sha256": "c" * 64,
        "planner": "common-counted-uniform-cost",
        "actions": ["left", "right", "forward", "pickup", "drop", "toggle"],
    }
    first = {**common, "method": "full_replan", "success": False}
    second = {**common, "method": "receipt_indexed", "success": True}

    assert access_signature(first) == access_signature(second)


def test_condition_names_are_complete_and_stable() -> None:
    assert CONDITIONS == (
        "clean",
        "irrelevant_update",
        "alternative_support",
        "relevant_support_loss",
        "false_positive_correction",
        "compound_occlusion_correction",
        "sparse_update_burst",
    )


def test_public_unlockpickupdist_enrollment_is_deterministic() -> None:
    from connectomequest.embodied.perception import Perceptor
    from connectomequest.embodied.protocol import CHECKPOINT
    from connectomequest.embodied.symbolic_controller import ColorPerceptor

    perceptor = ColorPerceptor(Perceptor(CHECKPOINT))
    first = enroll_case(116000000, 17, perceptor, max_steps=32)
    second = enroll_case(116000000, 17, perceptor, max_steps=32)

    assert first.selected is True
    assert second.selected is True
    assert first.plan_sha256 == second.plan_sha256
    assert first.initial_receipt == second.initial_receipt
    assert first.rgb_receipts == second.rgb_receipts
    assert first.dependencies == second.dependencies
    assert len(first.plan) >= 4
    assert all(step.required_receipts <= first.active_receipts for step in first.dependencies)


def simple_case() -> DynamicCase:
    plan = (
        PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),
        PlanStep("forward", (1, 0), 0, (((2, 0), "empty"),)),
        PlanStep("left", (2, 0), 0, ()),
        PlanStep("right", (2, 0), 3, ()),
    )
    dependencies = plan_dependencies(
        plan,
        support={(1, 0): "rgb-1", (2, 0): "rgb-2"},
    )
    return DynamicCase(
        seed=1,
        corruption_seed=17,
        selected=True,
        exclusion_reason=None,
        initial_receipt="rgb-0",
        rgb_receipts=("rgb-0", "rgb-1", "rgb-2"),
        prefix=(),
        plan=plan,
        dependencies=dependencies,
        active_receipts=frozenset({"rgb-0", "rgb-1", "rgb-2"}),
        assertion_receipts={
            "cell:1,0=empty": frozenset({"rgb-1"}),
            "cell:2,0=empty": frozenset({"rgb-2"}),
        },
        belief_cells=(((0, 0), "empty"), ((1, 0), "empty"), ((2, 0), "empty")),
        support=(((1, 0), "rgb-1"), ((2, 0), "rgb-2")),
        pose=((0, 0), 0),
        inventory=None,
        goals=frozenset({((2, 0), 0)}),
        plan_sha256="unused-by-fixture",
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_is_deterministic_and_method_independent(condition: str) -> None:
    case = simple_case()
    first = build_condition(case, condition)
    second = build_condition(case, condition)

    assert first == second
    assert first.condition == condition
    assert first.access_payload == second.access_payload


def test_all_methods_receive_identical_predecision_access() -> None:
    case = simple_case()
    rows = [execute_case(case, method, condition="irrelevant_update") for method in METHODS]

    assert len({row.access_sha256 for row in rows}) == 1
    assert len({row.update_sha256 for row in rows}) == 1
    assert len({row.plan_sha256 for row in rows}) == 1
    assert all(row.exact_replay for row in rows)
    assert all(row.environment_receipts == () for row in rows)


def test_receipt_index_saves_complete_scan_on_irrelevant_update() -> None:
    case = simple_case()
    indexed = execute_case(case, "receipt_indexed", condition="irrelevant_update")
    complete = execute_case(case, "complete_revalidation", condition="irrelevant_update")
    full = execute_case(case, "full_replan", condition="irrelevant_update")

    assert indexed.authorized and complete.authorized and full.authorized
    assert indexed.step_validations == 0
    assert complete.step_validations == len(case.plan)
    assert indexed.primitive_evaluations < full.primitive_evaluations


def test_relevant_support_loss_is_caught_but_unchecked_cache_is_not() -> None:
    case = simple_case()
    indexed = execute_case(case, "receipt_indexed", condition="relevant_support_loss")
    unchecked = execute_case(case, "unchecked_cache", condition="relevant_support_loss")

    assert indexed.unsupported_authorizations == 0
    assert unchecked.unsupported_authorizations == 1


def test_alternative_support_rebinds_every_assertion_from_the_same_receipt() -> None:
    from dataclasses import replace

    base = simple_case()
    shared = replace(
        base,
        active_receipts=frozenset({"rgb-0", "rgb-shared"}),
        assertion_receipts={
            "cell:1,0=empty": frozenset({"rgb-shared"}),
            "cell:2,0=empty": frozenset({"rgb-shared"}),
        },
        dependencies=plan_dependencies(
            base.plan,
            support={(1, 0): "rgb-shared", (2, 0): "rgb-shared"},
        ),
    )

    row = execute_case(shared, "receipt_indexed", condition="alternative_support")

    assert row.authorized is True
    assert row.replan is False


def test_public_case_replays_prefix_and_selected_suffix_in_environment() -> None:
    from connectomequest.embodied.perception import Perceptor
    from connectomequest.embodied.protocol import CHECKPOINT
    from connectomequest.embodied.symbolic_controller import ColorPerceptor

    perceptor = ColorPerceptor(Perceptor(CHECKPOINT))
    case = enroll_case(116000000, 17, perceptor, max_steps=32)
    row = execute_case(case, "receipt_indexed", condition="irrelevant_update")

    assert row.exact_replay is True
    assert row.environment_receipts[0] == case.initial_receipt
    assert len(row.environment_receipts) == len(case.prefix) + row.environment_actions + 1
