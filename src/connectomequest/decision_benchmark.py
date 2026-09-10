"""Frozen, access-controlled decision states for fair candidate ranking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from connectomequest.env import Action, ActionType, ConnectomeEnv, NodeSketch, Observation
from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.proof import Evidence, EvidenceKind
from connectomequest.query import Query, QuerySpec, QueryType
from connectomequest.query_io import read_queries, validate_query_snapshot

SNAPSHOT_SCHEMA_VERSION = 1
WIRING_RELATION = "presynaptic_to"


class DecisionStage(StrEnum):
    FIRST_HOP = "first_hop"
    SECOND_HOP = "second_hop"


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """One immutable ranking decision with evaluator-only relevance labels."""

    snapshot_id: str
    held_out: str
    split: str
    stage: DecisionStage
    query: QuerySpec
    observation: Observation
    candidates: tuple[int, ...]
    relevant_candidates: frozenset[int]
    probes_used: int
    graph_manifest_sha256: str
    query_sha256: str
    branch_middle: int | None = None
    checkpoint_sha256: str | None = None

    @property
    def covered(self) -> bool:
        return bool(self.relevant_candidates)

    def to_record(self) -> dict:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "held_out": self.held_out,
            "split": self.split,
            "stage": str(self.stage),
            "query_id": self.query.query_id,
            "query_type": str(self.query.query_type),
            "anchors": list(self.query.anchors),
            "semantic_relation": self.query.semantic_relation,
            "observation_json": json.dumps(
                _observation_to_dict(self.observation),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "candidates": list(self.candidates),
            "relevant_candidates": sorted(self.relevant_candidates),
            "covered": self.covered,
            "remaining_budget": self.observation.remaining_budget,
            "probes_used": self.probes_used,
            "branch_middle": self.branch_middle,
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "query_sha256": self.query_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_record(cls, row: dict) -> DecisionSnapshot:
        version = int(row["schema_version"])
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported decision snapshot schema: {version}")
        return cls(
            snapshot_id=row["snapshot_id"],
            held_out=row["held_out"],
            split=row["split"],
            stage=DecisionStage(row["stage"]),
            query=QuerySpec(
                row["query_id"],
                QueryType(row["query_type"]),
                tuple(int(value) for value in row["anchors"]),
                row.get("semantic_relation"),
            ),
            observation=_observation_from_dict(json.loads(row["observation_json"])),
            candidates=tuple(int(value) for value in row["candidates"]),
            relevant_candidates=frozenset(int(value) for value in row["relevant_candidates"]),
            probes_used=int(row["probes_used"]),
            branch_middle=(None if row.get("branch_middle") is None else int(row["branch_middle"])),
            graph_manifest_sha256=row["graph_manifest_sha256"],
            query_sha256=row["query_sha256"],
            checkpoint_sha256=row.get("checkpoint_sha256"),
        )


def _observation_to_dict(observation: Observation) -> dict:
    return {
        "current_node": observation.current_node,
        "discovered_edges": [
            {
                "kind": str(edge.kind),
                "src": edge.src,
                "relation": edge.relation,
                "dst": edge.dst,
                "source_record": edge.source_record,
                "weight": edge.weight,
                "token_id": edge.token_id,
                "query_id": edge.query_id,
                "episode_nonce": edge.episode_nonce,
                "issued_step": edge.issued_step,
                "action_kind": edge.action_kind,
            }
            for edge in observation.discovered_edges
        ],
        "memory": [list(item) for item in observation.memory],
        "remaining_budget": observation.remaining_budget,
        "step": observation.step,
        "node_sketches": [
            {
                "node_id": sketch.node_id,
                "outgoing_degree": sketch.outgoing_degree,
                "total_weight": sketch.total_weight,
                "max_weight": sketch.max_weight,
                "mean_weight": sketch.mean_weight,
                "source_record": sketch.source_record,
            }
            for sketch in observation.node_sketches
        ],
    }


def _observation_from_dict(row: dict) -> Observation:
    return Observation(
        current_node=int(row["current_node"]),
        discovered_edges=tuple(
            Evidence(
                EvidenceKind(item["kind"]),
                int(item["src"]),
                item["relation"],
                int(item["dst"]),
                source_record=item["source_record"],
                weight=float(item["weight"]),
                token_id=item.get("token_id", ""),
                query_id=item.get("query_id", ""),
                episode_nonce=item.get("episode_nonce", ""),
                issued_step=int(item.get("issued_step", -1)),
                action_kind=item.get("action_kind", ""),
            )
            for item in row["discovered_edges"]
        ),
        memory=tuple((int(slot), int(node)) for slot, node in row["memory"]),
        remaining_budget=float(row["remaining_budget"]),
        step=int(row["step"]),
        node_sketches=tuple(
            NodeSketch(
                node_id=int(item["node_id"]),
                outgoing_degree=int(item["outgoing_degree"]),
                total_weight=float(item["total_weight"]),
                max_weight=float(item["max_weight"]),
                mean_weight=float(item["mean_weight"]),
                source_record=item["source_record"],
            )
            for item in row["node_sketches"]
        ),
    )


def _visible_candidates(
    observation: Observation,
    src: int,
    relation: str = WIRING_RELATION,
) -> tuple[int, ...]:
    return tuple(
        edge.dst
        for edge in observation.discovered_edges
        if edge.kind != EvidenceKind.NON_EDGE and edge.src == src and edge.relation == relation
    )


def _snapshot_id(
    query_id: str,
    stage: DecisionStage,
    branch_middle: int | None,
    candidates: tuple[int, ...],
) -> str:
    payload = json.dumps(
        [query_id, str(stage), branch_middle, candidates],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _prefix_environment(
    graph: GraphStore,
    query: Query,
    *,
    visible_neighbors: int,
    budget: int,
    subgoal_probes: int,
) -> tuple[ConnectomeEnv, tuple[int, ...], int]:
    env = ConnectomeEnv(
        graph,
        visible_neighbors=visible_neighbors,
        max_steps=budget,
        budget=budget,
        answer_quota=1,
    )
    env.reset(query)
    result = env.step(Action(ActionType.INSPECT, relation=WIRING_RELATION))
    if result.error:
        raise RuntimeError(f"{query.query_id}: {result.error}")
    candidates = _visible_candidates(result.observation, query.anchors[0])
    probes_used = 0
    for candidate in candidates[:subgoal_probes]:
        result = env.step(
            Action(
                ActionType.PROBE_AFFORDANCE,
                relation=WIRING_RELATION,
                target=candidate,
            )
        )
        if result.error:
            raise RuntimeError(f"{query.query_id}: {result.error}")
        probes_used += 1
    return env, candidates, probes_used


def build_query_snapshots(
    graph: GraphStore,
    query: Query,
    *,
    held_out: str,
    graph_manifest_sha256: str,
    query_sha256: str,
    visible_neighbors: int = 32,
    budget: int = 64,
    subgoal_probes: int = 0,
    stages: frozenset[DecisionStage] = frozenset(DecisionStage),
) -> list[DecisionSnapshot]:
    """Build first-hop and independent second-hop branch decisions."""

    if query.query_type != QueryType.TWO_HOP_TYPE:
        return []
    env, first_candidates, probes_used = _prefix_environment(
        graph,
        query,
        visible_neighbors=visible_neighbors,
        budget=budget,
        subgoal_probes=subgoal_probes,
    )
    answer_set = set(query.answers)
    snapshots: list[DecisionSnapshot] = []
    if DecisionStage.FIRST_HOP in stages:
        relevant = frozenset(
            middle
            for middle in first_candidates
            if answer_set
            & set(
                graph.neighbors(
                    middle,
                    WIRING_RELATION,
                    by_weight=False,
                ).node_ids.tolist()
            )
        )
        snapshots.append(
            DecisionSnapshot(
                snapshot_id=_snapshot_id(
                    query.query_id,
                    DecisionStage.FIRST_HOP,
                    None,
                    first_candidates,
                ),
                held_out=held_out,
                split=query.split,
                stage=DecisionStage.FIRST_HOP,
                query=query.spec,
                observation=env.observe(),
                candidates=first_candidates,
                relevant_candidates=relevant,
                probes_used=probes_used,
                graph_manifest_sha256=graph_manifest_sha256,
                query_sha256=query_sha256,
            )
        )

    if DecisionStage.SECOND_HOP not in stages:
        return snapshots
    for middle in first_candidates:
        branch_env, branch_first, branch_probes = _prefix_environment(
            graph,
            query,
            visible_neighbors=visible_neighbors,
            budget=budget,
            subgoal_probes=subgoal_probes,
        )
        if middle not in branch_first:
            raise RuntimeError(f"{query.query_id}: branch middle is not visible")
        result = branch_env.step(
            Action(
                ActionType.TRAVERSE,
                relation=WIRING_RELATION,
                target=middle,
            )
        )
        if result.error:
            raise RuntimeError(f"{query.query_id}: {result.error}")
        result = branch_env.step(Action(ActionType.INSPECT, relation=WIRING_RELATION))
        if result.error:
            raise RuntimeError(f"{query.query_id}: {result.error}")
        candidates = _visible_candidates(result.observation, middle)
        relevant = frozenset(answer_set & set(candidates))
        snapshots.append(
            DecisionSnapshot(
                snapshot_id=_snapshot_id(
                    query.query_id,
                    DecisionStage.SECOND_HOP,
                    middle,
                    candidates,
                ),
                held_out=held_out,
                split=query.split,
                stage=DecisionStage.SECOND_HOP,
                query=query.spec,
                observation=branch_env.observe(),
                candidates=candidates,
                relevant_candidates=relevant,
                probes_used=branch_probes,
                branch_middle=middle,
                graph_manifest_sha256=graph_manifest_sha256,
                query_sha256=query_sha256,
            )
        )
    return snapshots


def build_decision_snapshots(
    graph: GraphStore,
    query_path: Path,
    *,
    split: str = "test",
    held_out: str | None = None,
    visible_neighbors: int = 32,
    budget: int = 64,
    subgoal_probes: int = 0,
    stages: frozenset[DecisionStage] = frozenset(DecisionStage),
    offset: int = 0,
    limit: int | None = None,
) -> list[DecisionSnapshot]:
    validate_query_snapshot(query_path, graph)
    queries = [
        query
        for query in read_queries(query_path)
        if query.split == split and query.query_type == QueryType.TWO_HOP_TYPE
    ]
    queries = queries[offset:]
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError(f"query file contains no {split}/2pt queries")
    graph_hash = sha256_file(graph.manifest_path)
    query_hash = sha256_file(Path(query_path))
    dataset = held_out or f"{graph.manifest.dataset}-{graph.manifest.version}"
    snapshots: list[DecisionSnapshot] = []
    for query in queries:
        snapshots.extend(
            build_query_snapshots(
                graph,
                query,
                held_out=dataset,
                graph_manifest_sha256=graph_hash,
                query_sha256=query_hash,
                visible_neighbors=visible_neighbors,
                budget=budget,
                subgoal_probes=subgoal_probes,
                stages=stages,
            )
        )
    return snapshots


def write_decision_snapshots(
    snapshots: Iterable[DecisionSnapshot],
    output: Path,
    *,
    protocol: dict | None = None,
) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [snapshot.to_record() for snapshot in snapshots]
    if not rows:
        raise ValueError("cannot write an empty decision snapshot")
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(output)
    stage_counts: dict[str, int] = {}
    covered_counts: dict[str, int] = {}
    for row in rows:
        stage = row["stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        covered_counts[stage] = covered_counts.get(stage, 0) + int(row["covered"])
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_sha256": sha256_file(output),
        "num_snapshots": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "covered_counts": dict(sorted(covered_counts.items())),
        "graph_manifest_sha256": rows[0]["graph_manifest_sha256"],
        "query_sha256": rows[0]["query_sha256"],
        "protocol": protocol or {},
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_decision_snapshots(path: Path) -> list[DecisionSnapshot]:
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["snapshot_sha256"] != sha256_file(path):
        raise ValueError("decision snapshot SHA-256 does not match its manifest")
    return [
        DecisionSnapshot.from_record(row)
        for row in pq.read_table(path, memory_map=True).to_pylist()
    ]
