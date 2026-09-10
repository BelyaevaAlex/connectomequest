"""Explicit ranker boundary and issued-positive-witness checks.

These helpers do not change frozen environment/controller implementations.
The witness checker deliberately does not receive ground-truth answer sets.
"""

from connectomequest.proof import EvidenceKind, evidence_digest


class PermutationCheckedRanker:
    """Restrict an untrusted proposal module to a permutation of supplied IDs."""

    def __init__(self, ranker):
        self.ranker = ranker

    def rank(self, query, observation, candidates, **kwargs):
        expected = tuple(candidates)
        if len(expected) != len(set(expected)):
            raise ValueError("candidate input must contain unique IDs")
        result = list(self.ranker.rank(query, observation, list(expected), **kwargs))
        if len(result) != len(expected) or set(result) != set(expected):
            raise ValueError("ranker must return a permutation of the supplied candidates")
        return result


def issued_positive_witness(
    query, answer, evidence, *, source_record, nonce, registry, current_step
):
    """Check a 2pt witness using trusted issuance, not answer membership.

    Only qualifying positive tokens may discharge obligations. Extra tokens are
    ignored here; the production validator additionally validates the whole ledger.
    A token may discharge two identical edge obligations on a degenerate path.
    """
    if query.query_type.value != "2pt":
        raise ValueError("this predicate is defined only for 2pt")
    edges = set()
    for token in evidence:
        if (
            token.kind == EvidenceKind.EDGE
            and token.source_record == source_record
            and token.query_id == query.query_id
            and token.episode_nonce == nonce
            and 0 <= token.issued_step <= current_step
            and token.token_id
            and registry.get(token.token_id) == evidence_digest(token)
        ):
            edges.add((token.src, token.relation, token.dst))
    anchor, target = query.anchors
    native = query.semantic_relation or "has_type"
    return (answer, native, target) in edges and any(
        src == anchor and rel == "presynaptic_to" and (middle, "presynaptic_to", answer) in edges
        for src, rel, middle in edges
    )
