from __future__ import annotations

from connectomequest.embodied.evidence_index import PlanDependency
from connectomequest.embodied.navigation import PlanStep
from connectomequest.embodied.search import WorkCounter
from connectomequest.embodied.types import EvidenceEvent, MethodInput


class SpyPlanner:
    tie_break_id = "lexicographic-v1"

    def __init__(self):
        self.calls = []

    def __call__(self, method_input):
        from connectomequest.embodied.revalidation import PlannedSuffix

        self.calls.append(method_input)
        plan = (PlanStep("right", method_input.pose, method_input.direction, ()),)
        dependencies = (PlanDependency(0, frozenset(), frozenset(), frozenset()),)
        return PlannedSuffix(
            plan,
            dependencies,
            WorkCounter(precondition_evaluations=5, successor_evaluations=7),
        )


def _case(*, alternative=False, irrelevant=False, changed_key=False):
    plan = (
        PlanStep("forward", (0, 0), 0, (((1, 0), "empty"),)),
        PlanStep("forward", (1, 0), 0, (((2, 0), "empty"),)),
        PlanStep("left", (2, 0), 0, ()),
    )
    dependencies = (
        PlanDependency(0, frozenset({"r1"}), frozenset({"cell:1,0"}), frozenset({"a1"})),
        PlanDependency(1, frozenset({"r2"}), frozenset({"cell:2,0"}), frozenset({"a2"})),
        PlanDependency(2, frozenset(), frozenset(), frozenset()),
    )
    bindings = (
        ("a1", ("r1-alt",) if alternative else ("r1",)),
        ("a2", ("r2",)),
    )
    active = ("r1-alt", "r2") if alternative else (("r1", "r2") if irrelevant else ("r2",))
    withdrawn = ("noise",) if irrelevant else ("r1",)
    event = EvidenceEvent(
        action_index=4,
        withdrawn_receipt_ids=withdrawn,
        superseded_receipt_ids=(),
        changed_semantic_keys=("cell:1,0",) if changed_key else (),
        added_assertion_bindings=(),
        condition="test",
    )
    method_input = MethodInput(
        pose=(0, 0),
        direction=0,
        inventory=None,
        mission="pick up the green box",
        belief_snapshot=(("1,0", "empty"), ("2,0", "empty")),
        active_receipt_ids=active,
        assertion_to_receipt_bindings=bindings,
        remaining_plan=plan,
        cursor=0,
        update=event,
        public_action_set=("left", "right", "forward", "pickup", "drop", "toggle"),
    )
    return plan, dependencies, method_input


def _strategies(planner):
    from connectomequest.embodied.revalidation import (
        FullReplan,
        FullScan,
        ReceiptIndex,
        UncheckedReuse,
    )

    return (FullReplan(planner), FullScan(planner), ReceiptIndex(planner), UncheckedReuse(planner))


def test_all_strategies_share_exact_planner_object_and_tie_rule():
    planner = SpyPlanner()
    strategies = _strategies(planner)
    assert all(strategy.planner is planner for strategy in strategies)
    assert {strategy.tie_break_id for strategy in strategies} == {"lexicographic-v1"}


def test_index_and_scan_match_after_final_support_loss():
    planner = SpyPlanner()
    plan, dependencies, method_input = _case()
    strategies = _strategies(planner)
    decisions = {}
    for strategy in strategies:
        strategy.initialize(plan, dependencies)
        decisions[type(strategy).__name__] = strategy.handle_update(method_input)
    scan = decisions["FullScan"]
    indexed = decisions["ReceiptIndex"]
    assert indexed.authorized_action == scan.authorized_action == "right"
    assert indexed.replanned is scan.replanned is True
    assert indexed.unsupported_authorization is False
    assert scan.unsupported_authorization is False


def test_irrelevant_update_is_output_sensitive_for_index():
    planner = SpyPlanner()
    plan, dependencies, method_input = _case(irrelevant=True)
    _, scan, indexed, _ = _strategies(planner)
    scan.initialize(plan, dependencies)
    indexed.initialize(plan, dependencies)
    scan_decision = scan.handle_update(method_input)
    index_decision = indexed.handle_update(method_input)
    assert scan_decision.counters.plan_position_visits == len(plan)
    assert index_decision.counters.plan_position_visits == 0
    assert index_decision.counters.predicate_evaluations == 0
    assert index_decision.counters.index_lookups == 1
    assert index_decision.counters.posting_entries == 0
    assert scan_decision.authorized_action == index_decision.authorized_action == "forward"


def test_unchecked_reuse_exposes_support_loss_without_charging_hidden_audit():
    planner = SpyPlanner()
    plan, dependencies, method_input = _case()
    unchecked = _strategies(planner)[-1]
    unchecked.initialize(plan, dependencies)
    decision = unchecked.handle_update(method_input)
    assert decision.authorized_action == "forward"
    assert decision.replanned is False
    assert decision.unsupported_authorization is True
    assert decision.counters.predicate_evaluations == 0
    assert decision.counters.plan_position_visits == 0
    assert planner.calls == []


def test_alternative_support_preserves_suffix_for_scan_and_index():
    planner = SpyPlanner()
    plan, dependencies, method_input = _case(alternative=True)
    for strategy in _strategies(planner)[1:3]:
        strategy.initialize(plan, dependencies)
        decision = strategy.handle_update(method_input)
        assert decision.authorized_action == "forward"
        assert decision.replanned is False
        assert decision.unsupported_authorization is False


def test_changed_semantic_key_triggers_reoptimization_for_affected_suffix():
    planner = SpyPlanner()
    plan, dependencies, method_input = _case(alternative=True, changed_key=True)
    indexed = _strategies(planner)[2]
    indexed.initialize(plan, dependencies)
    decision = indexed.handle_update(method_input)
    assert decision.affected_positions == (0,)
    assert decision.replanned is True
    assert decision.authorized_action == "right"


def test_planner_work_timers_and_index_construction_are_separate():
    planner = SpyPlanner()
    plan, dependencies, method_input = _case()
    indexed = _strategies(planner)[2]
    indexed.initialize(plan, dependencies)
    assert indexed.index_build_ns >= 0
    assert indexed.index_peak_bytes >= 0
    decision = indexed.handle_update(method_input)
    assert decision.counters.planner_calls == 1
    assert decision.counters.predicate_evaluations >= 1
    assert decision.counters.successor_expansions == 7
    assert decision.counters.revalidation_ns >= 0
    assert decision.counters.planning_ns >= 0
