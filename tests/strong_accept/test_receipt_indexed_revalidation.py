from dataclasses import replace

import pytest

from connectomequest.embodied.evidence_index import EvidenceUpdate, PlanDependency
from connectomequest.embodied.indexed_revalidation import (
    EvidenceState,
    RevalidationInput,
    plan_sha256,
    revalidate,
)
from connectomequest.embodied.navigation import PlanStep


class Belief:
    def __init__(self, cells: dict[tuple[int, int], str]):
        self.cells = dict(cells)

    def label(self, position: tuple[int, int]) -> str | None:
        return self.cells.get(tuple(position))


def request() -> RevalidationInput:
    plan = (
        PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),
        PlanStep("forward", (1, 0), 0, (((2, 0), "empty"),)),
    )
    dependencies = (
        PlanDependency(
            0,
            frozenset({"r0"}),
            frozenset({"cell:1,0"}),
            required_assertions=frozenset({"cell:1,0=empty"}),
        ),
        PlanDependency(
            1,
            frozenset({"r1"}),
            frozenset({"cell:2,0"}),
            required_assertions=frozenset({"cell:2,0=empty"}),
        ),
    )
    return RevalidationInput(
        plan=plan,
        cursor=0,
        dependencies=dependencies,
        update=EvidenceUpdate(frozenset(), frozenset(), frozenset()),
        evidence=EvidenceState(
            active_receipts=frozenset({"r0", "r1"}),
            assertion_receipts={
                "cell:1,0=empty": frozenset({"r0"}),
                "cell:2,0=empty": frozenset({"r1"}),
            },
        ),
        belief=Belief({(0, 0): "empty", (1, 0): "empty", (2, 0): "empty"}),
        pose=((0, 0), 0),
        goals=frozenset({((2, 0), 0)}),
        has_key=False,
        trusted_update=True,
        provenance_complete=True,
        sealed_plan_sha256=plan_sha256(plan),
    )


def test_irrelevant_update_avoids_indexed_scan_but_not_complete_scan() -> None:
    base = request()
    update = EvidenceUpdate(frozenset({"unrelated"}), frozenset(), frozenset())

    indexed = revalidate("receipt_indexed", replace(base, update=update))
    complete = revalidate("complete_revalidation", replace(base, update=update))

    assert indexed.authorized is True
    assert indexed.affected_steps == ()
    assert indexed.work.step_validations == 0
    assert complete.work.step_validations == 2
    assert indexed.selected_plan == complete.selected_plan == base.plan


def test_alternative_receipt_rebinds_without_replanning() -> None:
    base = request()
    update = EvidenceUpdate(frozenset({"r0"}), frozenset(), frozenset())
    evidence = EvidenceState(
        active_receipts=frozenset({"r-alt", "r1"}),
        assertion_receipts={
            "cell:1,0=empty": frozenset({"r-alt"}),
            "cell:2,0=empty": frozenset({"r1"}),
        },
    )

    result = revalidate("receipt_indexed", replace(base, update=update, evidence=evidence))

    assert result.authorized is True
    assert result.replan is False
    assert result.affected_steps == (0,)
    assert result.authorization_receipts == frozenset({"r-alt"})


def test_required_immediate_support_loss_is_never_authorized_by_checkers() -> None:
    base = request()
    update = EvidenceUpdate(frozenset({"r0"}), frozenset(), frozenset())
    evidence = EvidenceState(
        active_receipts=frozenset({"r1"}),
        assertion_receipts={"cell:2,0=empty": frozenset({"r1"})},
    )
    changed = replace(
        base,
        update=update,
        evidence=evidence,
        belief=Belief({(0, 0): "empty", (1, 0): "wall", (2, 0): "empty"}),
    )

    for method in (
        "full_replan",
        "complete_revalidation",
        "dependency_unaware",
        "receipt_indexed",
    ):
        result = revalidate(method, changed)
        assert result.authorized is False
        assert result.unsupported_next_action is False

    unchecked = revalidate("unchecked_cache", changed)
    assert unchecked.authorized is True
    assert unchecked.unsupported_next_action is True


def test_new_optimization_key_triggers_common_replanner() -> None:
    base = request()
    update = EvidenceUpdate(frozenset(), frozenset({"cell:2,0"}), frozenset())

    result = revalidate("receipt_indexed", replace(base, update=update))

    assert result.replan is True
    assert result.work.states_settled > 0
    assert result.selected_plan == base.plan


def test_receipt_stripped_input_uses_complete_scan() -> None:
    base = request()
    update = EvidenceUpdate(frozenset({"r0"}), frozenset(), frozenset())

    result = revalidate(
        "receipt_indexed",
        replace(base, update=update, provenance_complete=False),
    )

    assert result.affected_steps == (0, 1)
    assert result.work.step_validations == 2


def test_behavior_matches_complete_revalidation_for_restricted_update() -> None:
    base = request()
    update = EvidenceUpdate(frozenset({"unrelated"}), frozenset(), frozenset())

    indexed = revalidate("receipt_indexed", replace(base, update=update))
    complete = revalidate("complete_revalidation", replace(base, update=update))

    assert indexed.authorized == complete.authorized
    assert indexed.selected_plan[0] == complete.selected_plan[0]


def test_dependency_oracle_is_rejected_without_diagnostic_positions() -> None:
    with pytest.raises(ValueError, match="oracle affected steps"):
        revalidate("dependency_oracle", request())


def test_untrusted_update_and_stale_plan_hash_are_rejected() -> None:
    base = request()
    with pytest.raises(ValueError, match="untrusted evidence update"):
        revalidate("receipt_indexed", replace(base, trusted_update=False))
    with pytest.raises(ValueError, match="plan hash mismatch"):
        revalidate("receipt_indexed", replace(base, sealed_plan_sha256="0" * 64))


def test_incomplete_dependency_index_is_rejected() -> None:
    base = request()
    with pytest.raises(ValueError, match="dependency index"):
        revalidate("receipt_indexed", replace(base, dependencies=base.dependencies[:1]))
