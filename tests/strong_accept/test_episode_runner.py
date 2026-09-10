from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from connectomequest.embodied.environment import generate_layout
from connectomequest.embodied.types import EpisodeProtocol

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT / "outputs/nemo/strong-accept/dependent-plan-v1/development-attempt-02/perceptor.pt"
)


@lru_cache(maxsize=4)
def _prepared(condition):
    from connectomequest.embodied.dynamic_task import FrozenPerceptionCache
    from connectomequest.embodied.episode_runner import prepare_episode
    from connectomequest.embodied.perception import Perceptor
    from connectomequest.embodied.symbolic_controller import ColorPerceptor

    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    cache = FrozenPerceptionCache(ColorPerceptor(Perceptor(CHECKPOINT)), digest)
    exclusions = []
    for seed in range(39_000_001, 39_000_013):
        generated = generate_layout(seed, "short")
        prepared = prepare_episode(
            generated,
            cache,
            update_seed=17,
            condition=condition,
            update_count=1,
            relevant_fraction=0.0 if condition == "irrelevant_receipt_withdrawal" else 1.0,
        )
        if prepared.exclusion_reason is None:
            return prepared, cache, digest
        exclusions.append(prepared.exclusion_reason)
    raise AssertionError(f"no attainable short episode: {exclusions}")


def _protocol(digest):
    return EpisodeProtocol(
        protocol_version="v39-test",
        stage="test",
        horizon=4096,
        methods=("full_replan", "full_scan", "receipt_index", "unchecked_reuse"),
        public_action_set=("left", "right", "forward", "toggle", "pickup", "drop"),
        plan_length_strata=("short",),
        update_conditions=("irrelevant_receipt_withdrawal", "final_support_loss"),
        update_counts=(1,),
        relevant_fractions=(0.0, 1.0),
        layout_seeds=(39_000_001,),
        update_seeds=(17,),
        bootstrap_seed=39,
        bootstrap_draws=100,
        common_planner_id="full-mission-ucs-v1",
        action_model_id="unlockpickupdist-v1",
        perception_sha256=digest,
    )


def test_preparation_is_method_independent_and_plan_length_is_enrolled():
    prepared, _, _ = _prepared("irrelevant_receipt_withdrawal")
    assert prepared.exclusion_reason is None
    assert 8 <= prepared.pre_update_remaining_plan_length <= 15
    assert prepared.schedule.events[0].action_index == len(prepared.prefix_actions)
    assert not hasattr(prepared, "method")


def test_no_update_divergence_before_common_event_and_complete_task_replay():
    from connectomequest.embodied.episode_runner import replay_episode, run_episode

    prepared, cache, digest = _prepared("irrelevant_receipt_withdrawal")
    protocol = _protocol(digest)
    rows = [run_episode(protocol, prepared, method, cache) for method in protocol.methods]
    assert len({row.pre_update_action_digest for row in rows}) == 1
    assert len({row.access_digest for row in rows}) == 1
    assert all(row.task_success for row in rows)
    assert all(row.trace_replay_ok for row in rows)
    assert all(replay_episode(row).exact for row in rows)
    assert len({row.action_trace for row in rows}) == 1


def test_relevant_support_loss_is_never_authorized_by_competent_methods():
    from connectomequest.embodied.episode_runner import run_episode

    prepared, cache, digest = _prepared("final_support_loss")
    protocol = _protocol(digest)
    rows = {method: run_episode(protocol, prepared, method, cache) for method in protocol.methods}
    for method in ("full_replan", "full_scan", "receipt_index"):
        assert rows[method].unsupported_authorizations == 0
    assert rows["unchecked_reuse"].unsupported_authorizations >= 1


def test_result_accounting_identity_and_no_cross_method_observation_leakage():
    from connectomequest.embodied.episode_runner import run_episode

    prepared, cache, digest = _prepared("irrelevant_receipt_withdrawal")
    row = run_episode(_protocol(digest), prepared, "receipt_index", cache)
    assert row.total_reasoning_ns == (
        row.counters.revalidation_ns + row.counters.planning_ns + row.index_build_ns
    )
    assert row.counters.environment_actions == len(row.action_trace)
    assert row.exposed_update_count == len(row.update_digests)
    assert row.index_peak_bytes >= 0


def test_prefix_explorer_has_no_legacy_controller_or_latent_environment_access():
    import inspect

    import connectomequest.embodied.episode_runner as runner

    source = inspect.getsource(runner)
    assert "embodied_v17 import Controller" not in source
    assert "controller.act" not in source
    explorer = inspect.signature(runner._exploration_action)
    assert tuple(explorer.parameters) == ("tracker", "observation", "scanned")


def test_prefix_explorer_is_deterministic_on_the_same_public_belief():
    from connectomequest.embodied.episode_runner import _exploration_action, _Tracker

    class StubPerceptor:
        def predict(self, rgb):
            return ()

    class Observation:
        position = (0, 0)
        direction = 0

    tracker = _Tracker(StubPerceptor())
    tracker.cells = {(0, 0): "empty", (1, 0): "empty"}
    scanned = {((0, 0), 0)}
    first = _exploration_action(tracker, Observation(), scanned)
    second = _exploration_action(tracker, Observation(), scanned)
    assert first == second
    assert first == "forward"


def test_prefix_explorer_stops_when_public_frontier_is_exhausted():
    from connectomequest.embodied.episode_runner import _exploration_action, _Tracker

    class StubPerceptor:
        def predict(self, rgb):
            return ()

    class Observation:
        position = (0, 0)
        direction = 0

    tracker = _Tracker(StubPerceptor())
    tracker.cells = {
        (0, 0): "empty",
        (1, 0): "wall",
        (0, 1): "wall",
        (-1, 0): "wall",
        (0, -1): "wall",
    }
    scanned = {((0, 0), direction) for direction in range(4)}

    assert _exploration_action(tracker, Observation(), scanned) is None


def test_staging_evaluates_only_the_farthest_public_candidate(monkeypatch):
    from types import SimpleNamespace

    import connectomequest.embodied.episode_runner as runner

    tracker = SimpleNamespace(
        cells={
            (0, 0): "empty",
            (1, 0): "empty",
            (2, 0): "empty",
            (3, 0): "box:green",
        },
        support={},
        inventory=None,
    )
    observation = SimpleNamespace(
        position=(0, 0),
        direction=0,
        goals=("green",),
        stage=0,
        mission="pick up the green box",
    )
    calls = []

    def no_plan(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(plan=None, dependencies=())

    monkeypatch.setattr(runner, "plan_mission", no_plan)
    assert runner._candidate_stage(tracker, observation, "long") is None
    assert len(calls) <= 4


def test_prefix_explorer_releases_a_spent_key_after_confirmed_unlock():
    from connectomequest.embodied.episode_runner import _exploration_action, _Tracker

    class StubPerceptor:
        def predict(self, rgb):
            return ()

    class Observation:
        position = (0, 0)
        direction = 2

    tracker = _Tracker(StubPerceptor())
    tracker.cells = {
        (-1, 0): "empty",
        (0, 0): "empty",
        (1, 0): "empty",
        (4, 0): "door_open:red",
        (5, 0): "door_locked:blue",
    }
    tracker.inventory = "key:red"
    action = _exploration_action(
        tracker,
        Observation(),
        {((0, 0), 0)},
    )
    assert action == "drop"


def test_semantic_correction_candidates_come_from_conflicting_acquired_receipts():
    from types import SimpleNamespace

    from connectomequest.embodied.episode_runner import _correction_bindings_for_dependencies
    from connectomequest.embodied.evidence_index import PlanDependency

    old = "cell:1,2=door_locked:red"
    new = "cell:1,2=door_open:red"
    tracker = SimpleNamespace(
        bindings={
            old: {"rgb-old"},
            new: {"rgb-new"},
            "cell:9,9=empty": {"rgb-unrelated"},
        }
    )
    dependencies = (
        PlanDependency(
            0,
            frozenset(("rgb-old",)),
            frozenset(("cell:1,2",)),
            frozenset((old,)),
        ),
    )
    rows = _correction_bindings_for_dependencies(tracker, dependencies)
    assert len(rows) == 1
    assert rows[0].semantic_key == "cell:1,2"
    assert rows[0].old_assertion == old
    assert rows[0].new_assertion == new
    assert rows[0].receipt_id == "rgb-new"


def test_prefix_explorer_drops_an_object_receipt_disproves_as_a_key():
    from connectomequest.embodied.episode_runner import _exploration_action, _Tracker

    class StubPerceptor:
        def predict(self, rgb):
            return ()

    class Observation:
        position = (0, 0)
        direction = 2

    tracker = _Tracker(StubPerceptor())
    tracker.cells = {
        (-1, 0): "empty",
        (0, 0): "empty",
        (4, 0): "door_locked:green",
    }
    tracker.inventory = "ball:unknown"
    action = _exploration_action(
        tracker,
        Observation(),
        {((0, 0), 0)},
    )
    assert action == "drop"


def test_pickup_receipt_blacklists_a_false_key_hypothesis():
    from types import SimpleNamespace

    from connectomequest.embodied.episode_runner import _Tracker

    class StubPerceptor:
        def predict(self, rgb):
            return ()

    tracker = _Tracker(StubPerceptor())
    tracker.pending = ("pickup", "key:green", "rgb-before", (1, 0))
    observation = SimpleNamespace(
        carrying="ball",
        last_action="pickup",
        acknowledged=True,
        rgb=None,
        position=(0, 0),
        direction=0,
        receipt_id="rgb-after",
    )
    tracker.update(observation)
    assert tracker.inventory == "ball:unknown"
    assert (1, 0) in tracker.disqualified_key_positions


def test_episode_planner_returns_no_action_when_no_supported_plan_exists():
    from connectomequest.embodied.episode_runner import _EpisodePlanner
    from connectomequest.embodied.types import EvidenceEvent, MethodInput

    method_input = MethodInput(
        pose=(0, 0),
        direction=0,
        inventory=None,
        mission="pick up the green box",
        belief_snapshot=(("0,0", "empty"),),
        active_receipt_ids=("rgb-0",),
        assertion_to_receipt_bindings=(("cell:0,0=empty", ("rgb-0",)),),
        remaining_plan=(),
        cursor=0,
        update=EvidenceEvent(0, (), (), (), (), "none"),
        public_action_set=("left", "right", "forward", "toggle", "pickup", "drop"),
    )

    outcome = _EpisodePlanner()(method_input)

    assert outcome.plan == ()
    assert outcome.dependencies == ()


def test_access_digest_covers_only_the_common_pre_divergence_update():
    from connectomequest.embodied.episode_runner import _pre_divergence_access_digest

    left = _pre_divergence_access_digest(("first", "left-tail"))
    right = _pre_divergence_access_digest(("first", "right-tail"))
    other = _pre_divergence_access_digest(("other", "left-tail"))

    assert left == right
    assert left != other
