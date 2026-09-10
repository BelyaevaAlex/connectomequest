"""Full-graph symbolic oracle used only for upper bounds and demonstrations."""

from __future__ import annotations

from connectomequest.graph import GraphStore
from connectomequest.proof import Evidence, EvidenceKind, Proof
from connectomequest.query import Query, QueryType, project
from connectomequest.schema import Relation


def oracle_proof(graph: GraphStore, query: Query) -> Proof:
    rel = str(Relation.PRESYNAPTIC_TO)
    type_rel = query.semantic_relation or str(Relation.HAS_TYPE)
    evidence: list[Evidence] = []
    answers: set[int] = set()
    first = project(graph, query.anchors[:1])
    if query.query_type == QueryType.ONE_HOP:
        answers = first
        evidence.extend(Evidence(EvidenceKind.EDGE, query.anchors[0], rel, dst) for dst in first)
    elif query.query_type == QueryType.TWO_HOP:
        for middle in first:
            second = project(graph, (middle,))
            for answer in second:
                answers.add(answer)
                evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[0], rel, middle))
                evidence.append(Evidence(EvidenceKind.EDGE, middle, rel, answer))
    elif query.query_type == QueryType.TWO_HOP_TYPE:
        target_type = query.anchors[1]
        for middle in first:
            second = project(graph, (middle,))
            for answer in second:
                if not graph.has_edge(answer, target_type, type_rel):
                    continue
                answers.add(answer)
                evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[0], rel, middle))
                evidence.append(Evidence(EvidenceKind.EDGE, middle, rel, answer))
                evidence.append(Evidence(EvidenceKind.ATTRIBUTE, answer, type_rel, target_type))
    elif query.query_type == QueryType.INTERSECTION:
        second = project(graph, query.anchors[1:2])
        answers = first & second
        for answer in answers:
            evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[0], rel, answer))
            evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[1], rel, answer))
    elif query.query_type == QueryType.INTERSECTION_NEGATION:
        second = project(graph, query.anchors[1:2])
        answers = first - second
        for answer in answers:
            evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[0], rel, answer))
            evidence.append(Evidence(EvidenceKind.NON_EDGE, query.anchors[1], rel, answer))
    elif query.query_type == QueryType.INTERSECTION_TYPE:
        second = project(graph, query.anchors[1:2])
        for neuron in first & second:
            for answer in graph.neighbors(neuron, Relation.HAS_TYPE).node_ids.tolist():
                answers.add(int(answer))
                evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[0], rel, neuron))
                evidence.append(Evidence(EvidenceKind.EDGE, query.anchors[1], rel, neuron))
                evidence.append(Evidence(EvidenceKind.ATTRIBUTE, neuron, type_rel, int(answer)))
    else:
        raise ValueError(query.query_type)
    deduplicated = list(dict.fromkeys(evidence))
    return Proof(query.query_id, answers, deduplicated)
