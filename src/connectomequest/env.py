"""Partially observable, budgeted connectome exploration environment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from connectomequest.graph import GraphStore
from connectomequest.proof import (
    Evidence,
    EvidenceKind,
    Proof,
    ProofValidator,
    ValidationResult,
    evidence_digest,
)
from connectomequest.query import Query
from connectomequest.schema import Relation


class ActionType(StrEnum):
    INSPECT = "inspect"
    PROBE_AFFORDANCE = "probe_affordance"
    TRAVERSE = "traverse"
    STORE = "store"
    SWITCH = "switch"
    CHECK_EDGE = "check_edge"
    BACKTRACK = "backtrack"
    SUBMIT = "submit"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionType
    node: int | None = None
    relation: str | None = None
    target: int | None = None
    slot: int | None = None
    answers: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class NodeSketch:
    """ID-free local statistics returned by a budgeted perception action."""

    node_id: int
    outgoing_degree: int
    total_weight: float
    max_weight: float
    mean_weight: float
    source_record: str


@dataclass(frozen=True, slots=True)
class Observation:
    current_node: int
    discovered_edges: tuple[Evidence, ...]
    memory: tuple[tuple[int, int], ...]
    remaining_budget: float
    step: int
    node_sketches: tuple[NodeSketch, ...] = ()


@dataclass(slots=True)
class EpisodeState:
    current_node: int
    history: list[int] = field(default_factory=list)
    discovered: dict[tuple[int, str, int, EvidenceKind], Evidence] = field(default_factory=dict)
    offsets: dict[tuple[int, str], int] = field(default_factory=dict)
    memory: dict[int, int] = field(default_factory=dict)
    node_sketches: dict[int, NodeSketch] = field(default_factory=dict)
    remaining_budget: float = 0.0
    step: int = 0
    done: bool = False


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    validation: ValidationResult | None = None
    error: str | None = None


class ConnectomeEnv:
    def __init__(
        self,
        graph: GraphStore,
        *,
        visible_neighbors: int = 32,
        max_steps: int = 64,
        budget: float | None = None,
        max_memory_slots: int = 8,
        answer_quota: int | None = 10,
        enforce_proof: bool = True,
        strict_provenance: bool = True,
    ):
        self.graph = graph
        self._relation_names = frozenset(graph.relations)
        self.visible_neighbors = visible_neighbors
        self.max_steps = max_steps
        self.initial_budget = float(budget if budget is not None else max_steps)
        self.max_memory_slots = max_memory_slots
        if answer_quota is not None and answer_quota < 1:
            raise ValueError("answer_quota must be positive or None")
        self.answer_quota = answer_quota
        self.enforce_proof = enforce_proof
        self.strict_provenance = strict_provenance
        edge_artifact = next(
            artifact for artifact in graph.manifest.artifacts if artifact.path == "edges.parquet"
        )
        self.evidence_source_record = (
            f"{graph.manifest.dataset}:{graph.manifest.version}:"
            f"{edge_artifact.path}@sha256:{edge_artifact.sha256}"
        )
        self.validator = ProofValidator(
            graph,
            strict=strict_provenance,
            expected_source_record=self.evidence_source_record,
        )
        self.query: Query | None = None
        self.state: EpisodeState | None = None
        self._episode_nonce = ""
        self._reset_count = 0
        self.issued_token_ids: dict[str, str] = {}

    @property
    def episode_nonce(self) -> str:
        return self._episode_nonce

    def reset(self, query: Query) -> Observation:
        self._reset_count += 1
        nonce_payload = (
            f"{self.graph.manifest.dataset}:{self.graph.manifest.version}:"
            f"{query.query_id}:{self._reset_count}"
        )
        self._episode_nonce = hashlib.sha256(nonce_payload.encode()).hexdigest()[:24]
        self.issued_token_ids.clear()
        self.query = query
        self.state = EpisodeState(
            current_node=query.anchors[0],
            memory={
                slot: anchor for slot, anchor in enumerate(query.anchors[: self.max_memory_slots])
            },
            remaining_budget=self.initial_budget,
        )
        return self._observation()

    def observe(self) -> Observation:
        """Return the current public observation without exposing GraphStore."""
        return self._observation()

    def step(self, action: Action) -> StepResult:
        state = self._require_state()
        if state.done:
            raise RuntimeError("episode is already terminated")
        state.step += 1
        cost = self._cost(action)
        if cost > state.remaining_budget:
            state.done = True
            return StepResult(self._observation(), -1.0, True, error="budget_exceeded")
        state.remaining_budget -= cost
        validation = None
        error = None
        reward = -0.01 * cost
        try:
            if action.kind == ActionType.INSPECT:
                self._inspect(action)
            elif action.kind == ActionType.PROBE_AFFORDANCE:
                self._probe_affordance(action)
            elif action.kind == ActionType.TRAVERSE:
                self._traverse(action)
            elif action.kind == ActionType.CHECK_EDGE:
                self._check_edge(action)
            elif action.kind == ActionType.STORE:
                self._store(action)
            elif action.kind == ActionType.SWITCH:
                self._switch(action)
            elif action.kind == ActionType.BACKTRACK:
                self._backtrack()
            elif action.kind == ActionType.SUBMIT:
                proof = Proof(
                    query_id=self.query.query_id,
                    submitted_answers=set(action.answers),
                    evidence=list(state.discovered.values()),
                )
                validation = self.validator.validate(
                    self.query,
                    proof,
                    require_complete=self.answer_quota is None,
                    active_query_id=self.query.query_id if self.strict_provenance else None,
                    episode_nonce=self._episode_nonce if self.strict_provenance else None,
                    issued_token_ids=self.issued_token_ids if self.strict_provenance else None,
                    current_step=state.step if self.strict_provenance else None,
                )
                required = (
                    len(self.query.answers)
                    if self.answer_quota is None
                    else min(self.answer_quota, len(self.query.answers))
                )
                if len(action.answers) < required:
                    validation = ValidationResult(
                        valid=False,
                        answer_correct=False,
                        evidence_valid=validation.evidence_valid,
                        missing_answers=validation.missing_answers,
                        extra_answers=validation.extra_answers,
                        errors=validation.errors
                        + (f"insufficient answers: {len(action.answers)} < {required}",),
                    )
                reward = 1.0 if validation.valid else -0.25
                if not self.enforce_proof:
                    validation = ValidationResult(
                        valid=validation.answer_correct,
                        answer_correct=validation.answer_correct,
                        evidence_valid=validation.evidence_valid,
                        missing_answers=validation.missing_answers,
                        extra_answers=validation.extra_answers,
                        errors=validation.errors,
                    )
                state.done = True
            else:
                raise ValueError(f"unsupported action: {action.kind}")
        except (ValueError, KeyError) as exc:
            reward -= 0.1
            error = str(exc)
        if state.step >= self.max_steps or state.remaining_budget <= 0:
            state.done = True
        return StepResult(self._observation(), reward, state.done, validation, error)

    def _inspect(self, action: Action) -> None:
        state = self._require_state()
        node = state.current_node if action.node is None else action.node
        if node != state.current_node:
            raise ValueError("inspect is only legal at the current focus")
        if action.relation is None:
            raise ValueError("inspect requires relation")
        offset_key = (node, action.relation)
        offset = state.offsets.get(offset_key, 0)
        batch = self.graph.neighbors(
            node, action.relation, offset=offset, limit=self.visible_neighbors, by_weight=True
        )
        for target, weight in zip(
            batch.node_ids.tolist(),
            batch.weights.tolist(),
            strict=True,
        ):
            key = (node, action.relation, int(target), EvidenceKind.EDGE)
            # Re-inspecting a paginated page does not create a second token for
            # the same disclosed triple; the first receipt remains replayable.
            if key in state.discovered:
                continue
            evidence = self._issue_evidence(
                EvidenceKind.EDGE,
                node,
                action.relation,
                int(target),
                weight=float(weight),
                action_kind=str(ActionType.INSPECT),
            )
            state.discovered[key] = evidence
        state.offsets[offset_key] = offset + len(batch.node_ids)

    def _probe_affordance(self, action: Action) -> None:
        state = self._require_state()
        if action.target is None:
            raise ValueError("probe_affordance requires target")
        relation = action.relation or "presynaptic_to"
        if relation != "presynaptic_to":
            raise ValueError("only wiring affordances can be probed")
        legal = action.target == state.current_node or any(
            evidence.kind == EvidenceKind.EDGE
            and evidence.src == state.current_node
            and evidence.dst == action.target
            and evidence.relation == relation
            for evidence in state.discovered.values()
        )
        if not legal:
            raise ValueError("probe target is not the current or an observed frontier node")
        batch = self.graph.neighbors(action.target, relation, by_weight=False)
        weights = batch.weights
        degree = len(weights)
        total = float(weights.sum()) if degree else 0.0
        maximum = float(weights.max()) if degree else 0.0
        state.node_sketches[action.target] = NodeSketch(
            node_id=action.target,
            outgoing_degree=degree,
            total_weight=total,
            max_weight=maximum,
            mean_weight=total / degree if degree else 0.0,
            source_record=self.evidence_source_record,
        )

    def _traverse(self, action: Action) -> None:
        state = self._require_state()
        if action.target is None or action.relation is None:
            raise ValueError("traverse requires relation and target")
        key = (state.current_node, action.relation, action.target, EvidenceKind.EDGE)
        if key not in state.discovered:
            raise ValueError("traverse target has not been observed")
        state.history.append(state.current_node)
        state.current_node = action.target

    def legal_check_claims(self) -> tuple[tuple[int, str, int], ...]:
        """Return the currently checkable claims without exposing hidden edges."""
        state = self._require_state()
        if self.query is not None and self.query.query_type.value == "2pt":
            semantic_relation = self.query.semantic_relation or "has_type"
            first_middles = {
                evidence.dst
                for evidence in state.discovered.values()
                if evidence.kind == EvidenceKind.EDGE
                and evidence.src == self.query.anchors[0]
                and evidence.relation == "presynaptic_to"
            }
            candidates = {
                evidence.dst
                for evidence in state.discovered.values()
                if evidence.kind == EvidenceKind.EDGE
                and evidence.src in first_middles
                and evidence.relation == "presynaptic_to"
            }
            target = self.query.anchors[1]
            return tuple(
                sorted(
                    (candidate, semantic_relation, target)
                    for candidate in candidates
                    if candidate == state.current_node
                )
            )
        relation = str(Relation.PRESYNAPTIC_TO)
        targets = {
            evidence.dst
            for evidence in state.discovered.values()
            if evidence.kind == EvidenceKind.EDGE
        }
        targets.update(self.query.anchors if self.query is not None else ())
        return tuple(sorted((state.current_node, relation, target) for target in targets))

    def _check_edge(self, action: Action) -> None:
        state = self._require_state()
        src = state.current_node if action.node is None else action.node
        if src != state.current_node:
            raise ValueError("check_edge is only legal at the current focus")
        if action.target is None or action.relation is None:
            raise ValueError("check_edge requires relation and target")
        if action.relation not in self._relation_names:
            raise ValueError("check_edge relation is not in the graph schema")
        if self.query is not None and self.query.query_type.value == "2pt":
            semantic_relation = self.query.semantic_relation or "has_type"
            if action.relation != semantic_relation:
                raise ValueError("check_edge relation is not the query semantic predicate")
            first_middles = {
                evidence.dst
                for evidence in state.discovered.values()
                if evidence.kind == EvidenceKind.EDGE
                and evidence.src == self.query.anchors[0]
                and evidence.relation == "presynaptic_to"
            }
            observed_targets = {
                evidence.dst
                for evidence in state.discovered.values()
                if evidence.kind == EvidenceKind.EDGE
                and evidence.src in first_middles
                and evidence.relation == "presynaptic_to"
            }
            observed_targets.add(self.query.anchors[1])
        else:
            observed_targets = {
                evidence.dst
                for evidence in state.discovered.values()
                if evidence.kind == EvidenceKind.EDGE
            }
            # Other query motifs may check an anchor or a previously disclosed
            # node, but never an unseen target.
            observed_targets.update(self.query.anchors if self.query is not None else ())
        if action.target != state.current_node and action.target not in observed_targets:
            raise ValueError("check_edge target is not in the observed claim set")
        exists = self.graph.has_edge(src, action.target, action.relation)
        kind = EvidenceKind.EDGE if exists else EvidenceKind.NON_EDGE
        key = (src, action.relation, action.target, kind)
        if key in state.discovered:
            return
        evidence = self._issue_evidence(
            kind,
            src,
            action.relation,
            action.target,
            action_kind=str(ActionType.CHECK_EDGE),
        )
        state.discovered[key] = evidence

    def _issue_evidence(
        self,
        kind: EvidenceKind,
        src: int,
        relation: str,
        dst: int,
        *,
        weight: float = 1.0,
        action_kind: str,
    ) -> Evidence:
        """Issue a query- and episode-bound evidence token."""
        state = self._require_state()
        query = self.query
        if query is None:
            raise RuntimeError("call reset before issuing evidence")
        payload = Evidence(
            kind,
            src,
            relation,
            dst,
            source_record=self.evidence_source_record,
            weight=weight,
            query_id=query.query_id,
            episode_nonce=self._episode_nonce,
            issued_step=state.step,
            action_kind=action_kind,
        )
        digest = evidence_digest(payload)
        token_id = hashlib.sha256(f"{digest}:{len(self.issued_token_ids)}".encode()).hexdigest()[
            :24
        ]
        evidence = Evidence(
            kind,
            src,
            relation,
            dst,
            source_record=self.evidence_source_record,
            weight=weight,
            token_id=token_id,
            query_id=query.query_id,
            episode_nonce=self._episode_nonce,
            issued_step=state.step,
            action_kind=action_kind,
        )
        self.issued_token_ids[token_id] = evidence_digest(evidence)
        return evidence

    def _store(self, action: Action) -> None:
        state = self._require_state()
        if action.slot is None or not 0 <= action.slot < self.max_memory_slots:
            raise ValueError("invalid memory slot")
        if action.node is not None and action.node != state.current_node:
            raise ValueError("only the current focus can be stored")
        state.memory[action.slot] = state.current_node

    def _switch(self, action: Action) -> None:
        state = self._require_state()
        if action.slot is None or action.slot not in state.memory:
            raise KeyError("empty memory slot")
        state.history.append(state.current_node)
        state.current_node = state.memory[action.slot]

    def _backtrack(self) -> None:
        state = self._require_state()
        if not state.history:
            raise ValueError("history is empty")
        state.current_node = state.history.pop()

    @staticmethod
    def _cost(action: Action) -> float:
        return 0.0 if action.kind == ActionType.SUBMIT else 1.0

    def _require_state(self) -> EpisodeState:
        if self.state is None or self.query is None:
            raise RuntimeError("call reset before step")
        return self.state

    def _observation(self) -> Observation:
        state = self._require_state()
        return Observation(
            current_node=state.current_node,
            discovered_edges=tuple(state.discovered.values()),
            memory=tuple(sorted(state.memory.items())),
            remaining_budget=state.remaining_budget,
            step=state.step,
            node_sketches=tuple(state.node_sketches[node] for node in sorted(state.node_sketches)),
        )
