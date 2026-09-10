from __future__ import annotations

import inspect

import pytest


def _api():
    from connectomequest.embodied import environment as api

    return api


@pytest.mark.parametrize(
    "stratum,bounds",
    [
        ("short", (8, 15)),
        ("medium", (16, 31)),
        ("long", (32, 63)),
        ("very_long", (64, None)),
    ],
)
def test_generated_layout_is_solvable_before_public_stratum_enrollment(stratum, bounds):
    api = _api()
    layout = api.generate_layout(39_000_001, stratum)
    assert layout.exclusion_reason is None or (
        layout.exclusion_reason in api.ALLOWED_GENERATOR_EXCLUSIONS
    )
    if layout.exclusion_reason is None:
        assert layout.reference_solution_length > 0
        assert layout.spec.horizon == 8192
        if stratum == "long":
            assert layout.spec.room_size >= 22
        if stratum == "very_long":
            assert layout.spec.room_size >= 34


def test_generator_signature_cannot_consult_method_outcomes():
    api = _api()
    parameters = inspect.signature(api.generate_layout).parameters
    assert "method" not in parameters
    assert "result" not in parameters
    assert "success" not in parameters


def test_same_layout_seed_produces_identical_enrollment():
    api = _api()
    first = api.generate_layout(39_000_019, "long")
    second = api.generate_layout(39_000_019, "long")
    assert first == second


def test_reset_and_action_replay_are_byte_deterministic():
    api = _api()
    generated = api.generate_layout(39_000_029, "medium")
    if generated.exclusion_reason is not None:
        pytest.skip(generated.exclusion_reason)

    def replay():
        env = api.reset_layout(generated.spec)
        rows = []
        try:
            rows.append(
                (
                    env.observation().receipt_id,
                    env.observation().position,
                    env.observation().direction,
                    env.observation().reward,
                    env.observation().terminated,
                    api.audit_environment_hash(env),
                )
            )
            for action in generated.enrollment_action_trace[:8]:
                after = env.step(action)
                rows.append(
                    (
                        after.receipt_id,
                        after.position,
                        after.direction,
                        after.reward,
                        after.terminated,
                        after.acknowledged,
                        api.audit_environment_hash(env),
                    )
                )
                if after.terminated:
                    break
            return tuple(rows)
        finally:
            env.close()

    assert replay() == replay()


def test_stratum_classifier_has_closed_nonoverlapping_boundaries():
    api = _api()
    assert api.classify_remaining_length(7) is None
    assert api.classify_remaining_length(8) == "short"
    assert api.classify_remaining_length(15) == "short"
    assert api.classify_remaining_length(16) == "medium"
    assert api.classify_remaining_length(31) == "medium"
    assert api.classify_remaining_length(32) == "long"
    assert api.classify_remaining_length(63) == "long"
    assert api.classify_remaining_length(64) == "very_long"
