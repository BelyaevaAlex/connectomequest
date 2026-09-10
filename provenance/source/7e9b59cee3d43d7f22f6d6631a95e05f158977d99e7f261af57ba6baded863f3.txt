"""Budgeted active explorer with optional neural frontier ranking."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from connectomequest.env import Action, ActionType, ConnectomeEnv, Observation, StepResult
from connectomequest.models.torus import TorusQueryHeuristic
from connectomequest.policies.base import FrontierPolicy
from connectomequest.proof import Evidence, EvidenceKind, ValidationResult
from connectomequest.query import QuerySpec, QueryType
from connectomequest.schema import Relation


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    query_id: str
    submitted_answers: frozenset[int]
    validation: ValidationResult | None
    steps: int
    budget_used: float
    invalid_actions: int
    terminated: bool
    evidence: tuple[Evidence, ...]
    first_proof_rank: int | None


class BudgetedExplorer:
    """A symbolic controller whose frontier ordering can be learned.

    The controller never reads GraphStore directly. All evidence arrives through
    environment actions, which makes the observation budget enforceable.
    """

    def __init__(
        self,
        *,
        model: TorusQueryHeuristic | None = None,
        policy: FrontierPolicy | None = None,
        device: torch.device | None = None,
        ranking: str = "weight",
        seed: int = 17,
        candidate_chunk_size: int = 65_536,
        candidates_per_subgoal: int = 4,
        subgoal_probes: int = 0,
        adaptive_probing: bool = False,
        max_subgoal_probes: int = 4,
        probe_margin_threshold: float = 0.05,
        probe_entropy_threshold: float = 0.95,
        enable_backtracking: bool = True,
        enable_budget_mask: bool = True,
        use_native_semantics: bool = True,
        policy_local_observation: bool = False,
    ):
        if ranking not in {"weight", "affordance_degree", "random", "neural", "policy"}:
            raise ValueError(f"unknown ranking: {ranking}")
        if ranking == "neural" and model is None:
            raise ValueError("neural ranking requires a model")
        if ranking == "policy" and policy is None:
            raise ValueError("policy ranking requires a FrontierPolicy")
        self.model = model
        self.policy = policy
        self.device = device or torch.device("cpu")
        self.ranking = ranking
        self.random = random.Random(seed)
        self.candidate_chunk_size = candidate_chunk_size
        if candidates_per_subgoal < 1:
            raise ValueError("candidates_per_subgoal must be positive")
        self.candidates_per_subgoal = candidates_per_subgoal
        if subgoal_probes < 0:
            raise ValueError("subgoal_probes cannot be negative")
        if max_subgoal_probes < 0:
            raise ValueError("max_subgoal_probes cannot be negative")
        self.adaptive_probing = adaptive_probing
        self.max_subgoal_probes = max_subgoal_probes
        self.probe_margin_threshold = probe_margin_threshold
        self.probe_entropy_threshold = probe_entropy_threshold
        self.enable_backtracking = enable_backtracking
        self.enable_budget_mask = enable_budget_mask
        self.use_native_semantics = use_native_semantics
        self.policy_local_observation = policy_local_observation
        self.subgoal_probes = subgoal_probes

    def run(self, env: ConnectomeEnv, query: QuerySpec) -> EpisodeResult:
        if env.query is None or env.state is None:
            raise RuntimeError("environment must be reset with the hidden query before agent.run")
        if env.query.query_id != query.query_id:
            raise ValueError("public query spec does not match the environment episode")
        invalid = 0
        first_proof_rank: int | None = None
        relation = str(Relation.PRESYNAPTIC_TO)

        def act(action: Action) -> StepResult | None:
            nonlocal invalid
            state = env.state
            if state is None or state.done:
                return None
            # Reserve one transition for the zero-cost submit action.
            if (
                self.enable_budget_mask
                and action.kind != ActionType.SUBMIT
                and (state.remaining_budget <= 1 or state.step >= env.max_steps - 1)
            ):
                return None
            result = env.step(action)
            if result.error:
                invalid += 1
            return result

        if query.query_type == QueryType.ONE_HOP:
            act(Action(ActionType.INSPECT, relation=relation))
            answers = self._outgoing_evidence(env, query.anchors[0], relation)

        elif query.query_type == QueryType.TWO_HOP:
            act(Action(ActionType.INSPECT, relation=relation))
            middles = self._outgoing_evidence(env, query.anchors[0], relation)
            middles = self._rank(env, query, middles)
            answers: set[int] = set()
            for middle in middles:
                if act(Action(ActionType.TRAVERSE, relation=relation, target=middle)) is None:
                    break
                if act(Action(ActionType.INSPECT, relation=relation)) is None:
                    break
                answers.update(self._outgoing_evidence(env, middle, relation))
                if not self.enable_backtracking or act(Action(ActionType.BACKTRACK)) is None:
                    break

        elif query.query_type == QueryType.TWO_HOP_TYPE:
            act(Action(ActionType.INSPECT, relation=relation))
            middles = self._outgoing_evidence(env, query.anchors[0], relation)
            if self.adaptive_probing and self.ranking == "policy":
                probe_order = self._weight_rank(env, middles)
                probed: list[int] = []
                while len(probed) < min(self.max_subgoal_probes, len(probe_order)):
                    if probed and self._ranking_confident(env, query, middles):
                        break
                    probed_set = set(probed)
                    middle = next(node for node in probe_order if node not in probed_set)
                    result = act(
                        Action(
                            ActionType.PROBE_AFFORDANCE,
                            relation=relation,
                            target=middle,
                        )
                    )
                    if result is None:
                        break
                    if result.error is None:
                        probed.append(middle)
                middles = self._rank(env, query, middles)
            elif self.subgoal_probes:
                probe_order = self._weight_rank(env, middles)
                probed = []
                for middle in probe_order[: self.subgoal_probes]:
                    result = act(
                        Action(
                            ActionType.PROBE_AFFORDANCE,
                            relation=relation,
                            target=middle,
                        )
                    )
                    if result is None:
                        break
                    if result.error is None:
                        probed.append(middle)
                middles = self._rank(env, query, middles)
            else:
                middles = self._rank(env, query, middles)
            answers = set()
            type_relation = (
                query.semantic_relation
                if self.use_native_semantics and query.semantic_relation
                else str(Relation.HAS_TYPE)
            )
            target_type = query.anchors[1]
            for middle_rank, middle in enumerate(middles, start=1):
                if env.answer_quota is not None and len(answers) >= env.answer_quota:
                    break
                if act(Action(ActionType.TRAVERSE, relation=relation, target=middle)) is None:
                    break
                if act(Action(ActionType.INSPECT, relation=relation)) is None:
                    break
                candidates = self._outgoing_evidence(env, middle, relation)
                candidates = self._rank(env, query, candidates, frontier=False)
                for candidate in candidates[: self.candidates_per_subgoal]:
                    assert env.state is not None
                    # Keep enough budget to return to the anchor and submit.
                    if env.state.remaining_budget <= 4:
                        break
                    if (
                        act(
                            Action(
                                ActionType.TRAVERSE,
                                relation=relation,
                                target=candidate,
                            )
                        )
                        is None
                    ):
                        break
                    if (
                        act(
                            Action(
                                ActionType.CHECK_EDGE,
                                relation=type_relation,
                                target=target_type,
                            )
                        )
                        is None
                    ):
                        break
                    if target_type in self._outgoing_evidence(env, candidate, type_relation):
                        if first_proof_rank is None:
                            first_proof_rank = middle_rank
                        answers.add(candidate)
                    if not self.enable_backtracking or act(Action(ActionType.BACKTRACK)) is None:
                        break
                if not self.enable_backtracking or act(Action(ActionType.BACKTRACK)) is None:
                    break
        elif query.query_type == QueryType.INTERSECTION:
            page_limit = max(1, (env.max_steps - 2) // 2)
            left: set[int] = set()
            for _ in range(page_limit):
                before = len(left)
                if act(Action(ActionType.INSPECT, relation=relation)) is None:
                    break
                left = self._outgoing_evidence(env, query.anchors[0], relation)
                if len(left) == before:
                    break
            act(Action(ActionType.SWITCH, slot=1))
            right: set[int] = set()
            for _ in range(page_limit):
                before = len(right)
                if act(Action(ActionType.INSPECT, relation=relation)) is None:
                    break
                right = self._outgoing_evidence(env, query.anchors[1], relation)
                if env.answer_quota is not None and len(left & right) >= env.answer_quota:
                    break
                if len(right) == before:
                    break
            answers = left & right

        elif query.query_type == QueryType.INTERSECTION_NEGATION:
            act(Action(ActionType.INSPECT, relation=relation))
            left = self._outgoing_evidence(env, query.anchors[0], relation)
            act(Action(ActionType.SWITCH, slot=1))
            answers = set()
            for candidate in self._rank(env, query, left):
                if (
                    act(
                        Action(
                            ActionType.CHECK_EDGE,
                            relation=relation,
                            target=candidate,
                        )
                    )
                    is None
                ):
                    break
                if self._has_non_edge(env, query.anchors[1], relation, candidate):
                    answers.add(candidate)

        elif query.query_type == QueryType.INTERSECTION_TYPE:
            # Reserve roughly half the budget for three-step type proofs.
            page_limit = max(1, (env.max_steps - 4) // 6)
            left: set[int] = set()
            for _ in range(page_limit):
                before = len(left)
                if act(Action(ActionType.INSPECT, relation=relation)) is None:
                    break
                left = self._outgoing_evidence(env, query.anchors[0], relation)
                if len(left) == before:
                    break
            act(Action(ActionType.SWITCH, slot=1))
            right: set[int] = set()
            for _ in range(page_limit):
                before = len(right)
                if act(Action(ActionType.INSPECT, relation=relation)) is None:
                    break
                right = self._outgoing_evidence(env, query.anchors[1], relation)
                common_size = len(left & right)
                target = env.answer_quota or float("inf")
                if common_size >= target or len(right) == before:
                    break
            common = self._rank(env, query, left & right)
            answers = set()
            for neuron in common:
                if env.answer_quota is not None and len(answers) >= env.answer_quota:
                    break
                if act(Action(ActionType.TRAVERSE, relation=relation, target=neuron)) is None:
                    break
                if act(Action(ActionType.INSPECT, relation=str(Relation.HAS_TYPE))) is None:
                    break
                answers.update(self._outgoing_evidence(env, neuron, str(Relation.HAS_TYPE)))
                if act(Action(ActionType.BACKTRACK)) is None:
                    break
        else:
            raise ValueError(query.query_type)

        final = act(Action(ActionType.SUBMIT, answers=frozenset(answers)))
        state = env.state
        assert state is not None
        return EpisodeResult(
            query_id=query.query_id,
            submitted_answers=frozenset(answers),
            validation=final.validation if final else None,
            steps=state.step,
            budget_used=env.initial_budget - state.remaining_budget,
            invalid_actions=invalid,
            terminated=state.done,
            evidence=tuple(state.discovered.values()),
            first_proof_rank=first_proof_rank,
        )

    @staticmethod
    def _outgoing_evidence(
        env: ConnectomeEnv,
        src: int,
        relation: str,
    ) -> set[int]:
        assert env.state is not None
        return {
            evidence.dst
            for evidence in env.state.discovered.values()
            if evidence.kind != EvidenceKind.NON_EDGE
            and evidence.src == src
            and evidence.relation == relation
        }

    @staticmethod
    def _has_non_edge(
        env: ConnectomeEnv,
        src: int,
        relation: str,
        dst: int,
    ) -> bool:
        assert env.state is not None
        return any(
            evidence.kind == EvidenceKind.NON_EDGE
            and evidence.src == src
            and evidence.relation == relation
            and evidence.dst == dst
            for evidence in env.state.discovered.values()
        )

    def _rank(
        self,
        env: ConnectomeEnv,
        query: QuerySpec,
        candidates: set[int],
        *,
        frontier: bool = True,
    ) -> list[int]:
        values = sorted(candidates)
        if self.ranking == "policy":
            assert self.policy is not None
            observation = env.observe()
            if self.policy_local_observation:
                observation = Observation(
                    current_node=observation.current_node,
                    discovered_edges=tuple(
                        edge
                        for edge in observation.discovered_edges
                        if edge.src == observation.current_node
                    ),
                    memory=(),
                    remaining_budget=observation.remaining_budget,
                    step=observation.step,
                    node_sketches=observation.node_sketches,
                )
            return self.policy.rank(
                query,
                observation,
                values,
                frontier=frontier,
                device=self.device,
                chunk_size=self.candidate_chunk_size,
            )
        if self.ranking == "random":
            self.random.shuffle(values)
            return values
        if self.ranking == "affordance_degree":
            sketches = {sketch.node_id: sketch for sketch in env.observe().node_sketches}
            return sorted(
                values,
                key=lambda node: (
                    sketches[node].outgoing_degree if node in sketches else -1,
                    sketches[node].total_weight if node in sketches else -1.0,
                    -node,
                ),
                reverse=True,
            )
        if self.ranking == "neural":
            assert self.model is not None
            ranker = self.model.rank_frontier if frontier else self.model.rank_candidates
            return ranker(
                query.query_type,
                query.anchors,
                values,
                device=self.device,
                chunk_size=self.candidate_chunk_size,
            )
        assert env.state is not None
        weights: dict[int, float] = {}
        for evidence in env.state.discovered.values():
            if evidence.kind != EvidenceKind.NON_EDGE and evidence.dst in candidates:
                weights[evidence.dst] = max(weights.get(evidence.dst, 0.0), evidence.weight)
        return sorted(
            values,
            key=lambda node: (weights.get(node, 0.0), -node),
            reverse=True,
        )

    @staticmethod
    def _weight_rank(env: ConnectomeEnv, candidates: set[int]) -> list[int]:
        values = sorted(candidates)
        assert env.state is not None
        weights: dict[int, float] = {}
        for evidence in env.state.discovered.values():
            if evidence.kind != EvidenceKind.NON_EDGE and evidence.dst in candidates:
                weights[evidence.dst] = max(
                    weights.get(evidence.dst, 0.0),
                    evidence.weight,
                )
        return sorted(
            values,
            key=lambda node: (weights.get(node, 0.0), -node),
            reverse=True,
        )

    def _ranking_confident(
        self, env: ConnectomeEnv, query: QuerySpec, candidates: set[int]
    ) -> bool:
        if len(candidates) < 2 or self.policy is None:
            return False
        uncertainty_fn = getattr(self.policy, "uncertainty", None)
        if uncertainty_fn is None:
            return False
        uncertainty = uncertainty_fn(
            query,
            env.observe(),
            sorted(candidates),
            frontier=True,
            device=self.device,
        )
        return (
            uncertainty.margin >= self.probe_margin_threshold
            and uncertainty.normalized_entropy <= self.probe_entropy_threshold
        )
