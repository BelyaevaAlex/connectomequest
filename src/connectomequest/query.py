"""Logical query representation, exact execution, and leakage-safe generation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from connectomequest.graph import GraphStore
from connectomequest.schema import Relation

SEMANTIC_VISIBLE_NEIGHBORS = 32


class QueryType(StrEnum):
    ONE_HOP = "1p"
    TWO_HOP = "2p"
    INTERSECTION = "2i"
    INTERSECTION_TYPE = "ip"
    INTERSECTION_NEGATION = "2in"
    TWO_HOP_TYPE = "2pt"


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Public task description; ground-truth answers remain environment-private."""

    query_id: str
    query_type: QueryType
    anchors: tuple[int, ...]
    semantic_relation: str | None = None


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    query_type: QueryType
    anchors: tuple[int, ...]
    answers: frozenset[int]
    split: str
    semantic_relation: str | None = None

    @property
    def spec(self) -> QuerySpec:
        return QuerySpec(self.query_id, self.query_type, self.anchors, self.semantic_relation)


def graph_semantic_relation(graph: GraphStore) -> str:
    """Choose a supported predicate without conflating semantic node classes."""

    available = set(graph.relations)
    for relation in (Relation.HAS_TYPE, Relation.IN_CORTICAL_LAYER):
        if str(relation) in available:
            return str(relation)
    raise ValueError("graph has neither has_type nor in_cortical_layer edges")


def project(graph: GraphStore, nodes: Iterable[int]) -> set[int]:
    answer: set[int] = set()

    for node in nodes:
        answer.update(
            graph.neighbors(node, Relation.PRESYNAPTIC_TO, by_weight=False).node_ids.tolist()
        )
    return answer


def execute(
    graph: GraphStore,
    query_type: QueryType,
    anchors: tuple[int, ...],
    *,
    semantic_relation: str | None = None,
) -> set[int]:
    if query_type == QueryType.ONE_HOP:
        return project(graph, anchors[:1])
    if query_type == QueryType.TWO_HOP:
        return project(graph, project(graph, anchors[:1]))
    if query_type == QueryType.TWO_HOP_TYPE:
        if len(anchors) < 2:
            raise ValueError(f"{query_type} requires a neuron and target-type anchor")
        reachable = project(graph, project(graph, anchors[:1]))
        target_type = anchors[1]
        selected_relation = semantic_relation or graph_semantic_relation(graph)
        return {
            neuron for neuron in reachable if graph.has_edge(neuron, target_type, selected_relation)
        }
    if len(anchors) < 2:
        raise ValueError(f"{query_type} requires two anchors")
    common = project(graph, anchors[:1]) & project(graph, anchors[1:2])
    if query_type == QueryType.INTERSECTION:
        return common
    if query_type == QueryType.INTERSECTION_NEGATION:
        return project(graph, anchors[:1]) - project(graph, anchors[1:2])
    if query_type == QueryType.INTERSECTION_TYPE:
        typed: set[int] = set()
        selected_relation = semantic_relation or graph_semantic_relation(graph)
        for node in common:
            typed.update(
                graph.neighbors(node, selected_relation, by_weight=False).node_ids.tolist()
            )
        return typed
    raise ValueError(query_type)


def grouped_split(*node_ids: int, seed: int = 17) -> str:
    """Assign a neuron group to a split without Python's randomized hash.

    Sorting makes direct and inverse representations of the same neuron pair land
    in the same split. The 80/10/10 thresholds are stable across machines.
    """

    canonical = ":".join(map(str, sorted(node_ids)))
    digest = hashlib.blake2b(f"{seed}:{canonical}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def entity_split(node_id: int, seed: int = 17) -> str:
    """Stable neuron-disjoint split assignment."""

    digest = hashlib.blake2b(f"entity:{seed}:{node_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def query_split(anchors: tuple[int, ...], seed: int = 17) -> str | None:
    """Return a split only when every anchor belongs to the same partition."""

    partitions = {entity_split(anchor, seed) for anchor in anchors}
    if len(partitions) != 1:
        return None
    return partitions.pop()


def sample_queries(
    graph: GraphStore,
    query_type: QueryType,
    *,
    count: int,
    seed: int = 17,
    sampling_seed: int | None = None,
    max_attempts_factor: int = 100,
) -> list[Query]:
    """Generate exact 80/10/10 quotas from fixed neuron-disjoint pools."""

    neuron_table = graph.nodes(columns=["node_id", "node_type"])
    mask = np.asarray(neuron_table["node_type"].to_pylist()) == "neuron"
    neurons = neuron_table["node_id"].to_numpy(zero_copy_only=False)[mask]
    if not len(neurons):
        return []
    pools = {
        split: np.asarray(
            [node for node in neurons if entity_split(int(node), seed) == split],
            dtype=np.int64,
        )
        for split in ("train", "validation", "test")
    }
    quotas = {
        "train": count * 8 // 10,
        "validation": count // 10,
        "test": count - (count * 8 // 10) - (count // 10),
    }
    rng = np.random.default_rng(seed if sampling_seed is None else sampling_seed)
    queries: list[Query] = []
    seen: set[tuple] = set()
    if query_type == QueryType.TWO_HOP_TYPE:
        selected_relation = graph_semantic_relation(graph)
        wiring = graph.csr(Relation.PRESYNAPTIC_TO)
        type_edges = graph.csr(selected_relation)
        type_of = np.full(graph.num_nodes, -1, dtype=np.int64)
        typed_rows = np.flatnonzero(np.diff(type_edges.indptr))
        type_of[typed_rows] = type_edges.indices[type_edges.indptr[typed_rows]]

        def visible_successors(node: int) -> np.ndarray:
            start, end = wiring.indptr[node : node + 2]
            node_ids = wiring.indices[start:end]
            weights = wiring.data[start:end]
            if not len(node_ids):
                return node_ids
            order = np.lexsort((node_ids, -weights))
            return node_ids[order[:SEMANTIC_VISIBLE_NEIGHBORS]]

        def typed_answers(neuron: int, target_type: int) -> frozenset[int]:
            found: set[int] = set()
            first_start, first_end = wiring.indptr[neuron : neuron + 2]
            for middle in wiring.indices[first_start:first_end]:
                second_start, second_end = wiring.indptr[middle : middle + 2]
                second_hop = wiring.indices[second_start:second_end]
                matching = second_hop[type_of[second_hop] == target_type]
                found.update(matching.tolist())
            return frozenset(found)

        for split, quota in quotas.items():
            pool = pools[split]
            produced = 0
            for _ in range(quota * max_attempts_factor):
                neuron = int(rng.choice(pool))
                first_hop = visible_successors(neuron)
                if not len(first_hop):
                    continue
                middle = int(rng.choice(first_hop))
                second_hop = visible_successors(middle)
                typed_second_hop = second_hop[type_of[second_hop] >= 0]
                if not len(typed_second_hop):
                    continue
                answer_seed = int(rng.choice(typed_second_hop))
                anchors = (neuron, int(type_of[answer_seed]))
                key = (query_type, *anchors)
                if key in seen:
                    continue
                answers = typed_answers(*anchors)
                if not answers:
                    continue
                identifier = f"{query_type}:{anchors}:split={seed}:sample={sampling_seed}"
                query_id = hashlib.sha1(identifier.encode()).hexdigest()[:16]
                queries.append(
                    Query(query_id, query_type, anchors, answers, split, selected_relation)
                )
                seen.add(key)
                produced += 1
                if produced >= quota:
                    break
            if produced != quota:
                raise RuntimeError(
                    f"generated {produced}/{quota} non-empty {split} queries for {query_type}"
                )
        return queries
    num_anchors = 1 if query_type in (QueryType.ONE_HOP, QueryType.TWO_HOP) else 2
    semantic_relation = (
        graph_semantic_relation(graph) if query_type == QueryType.INTERSECTION_TYPE else None
    )
    for split, quota in quotas.items():
        pool = pools[split]
        if len(pool) < num_anchors:
            raise RuntimeError(f"{split} has too few neurons for {query_type}")
        produced = 0
        for _ in range(quota * max_attempts_factor):
            anchors = tuple(int(x) for x in rng.choice(pool, size=num_anchors, replace=False))
            key = (query_type, *anchors)
            if key in seen:
                continue
            if semantic_relation is None:
                answers = execute(graph, query_type, anchors)
            else:
                answers = execute(graph, query_type, anchors, semantic_relation=semantic_relation)
            if not answers:
                continue
            identifier = f"{query_type}:{anchors}:split={seed}:sample={sampling_seed}"
            query_id = hashlib.sha1(identifier.encode()).hexdigest()[:16]
            queries.append(
                Query(query_id, query_type, anchors, frozenset(answers), split, semantic_relation)
            )
            seen.add(key)
            produced += 1
            if produced >= quota:
                break
        if produced != quota:
            raise RuntimeError(
                f"generated {produced}/{quota} non-empty {split} queries for {query_type}"
            )
    return queries
