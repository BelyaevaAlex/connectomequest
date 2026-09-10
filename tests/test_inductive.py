from __future__ import annotations

import torch

from connectomequest.env import NodeSketch, Observation
from connectomequest.models.inductive import FEATURE_DIM, InductiveQueryPolicy, candidate_features
from connectomequest.policies.rank_fusion import RankFusionPolicy
from connectomequest.proof import Evidence, EvidenceKind
from connectomequest.query import QuerySpec, QueryType
from connectomequest.schema import Relation


def _rename(value: int) -> int:
    return {3: 101, 5: 109, 7: 103, 11: 107}[value]


def test_features_are_invariant_to_entity_relabeling() -> None:
    relation = str(Relation.PRESYNAPTIC_TO)
    query = QuerySpec("q", QueryType.TWO_HOP, (3,))
    observation = Observation(
        current_node=3,
        discovered_edges=(
            Evidence(EvidenceKind.EDGE, 3, relation, 5, weight=9.0),
            Evidence(EvidenceKind.EDGE, 3, relation, 7, weight=2.0),
            Evidence(EvidenceKind.EDGE, 5, relation, 11, weight=4.0),
        ),
        memory=((0, 3),),
        remaining_budget=12.0,
        step=4,
        node_sketches=(NodeSketch(5, 3, 7.0, 4.0, 7.0 / 3.0, "source"),),
    )
    renamed_query = QuerySpec("q", QueryType.TWO_HOP, (_rename(3),))
    renamed_observation = Observation(
        current_node=_rename(3),
        discovered_edges=tuple(
            Evidence(
                evidence.kind,
                _rename(evidence.src),
                evidence.relation,
                _rename(evidence.dst),
                weight=evidence.weight,
            )
            for evidence in observation.discovered_edges
        ),
        memory=((0, _rename(3)),),
        remaining_budget=12.0,
        step=4,
        node_sketches=(NodeSketch(_rename(5), 3, 7.0, 4.0, 7.0 / 3.0, "source"),),
    )
    expected = candidate_features(observation, query, [5, 7], frontier=True)
    actual = candidate_features(
        renamed_observation,
        renamed_query,
        [_rename(5), _rename(7)],
        frontier=True,
    )
    assert expected.shape == (2, FEATURE_DIM)
    torch.testing.assert_close(actual, expected)


def test_inductive_policy_ranks_without_entity_table() -> None:
    model = InductiveQueryPolicy(hidden_dim=16, query_dim=8, dropout=0.0)
    assert all("entity" not in name for name, _ in model.named_parameters())
    observation = Observation(1, (), ((0, 1),), 8.0, 0)
    query = QuerySpec("q", QueryType.ONE_HOP, (1,))
    ranked = model.rank(
        query,
        observation,
        [9, 4, 6],
        frontier=True,
        device=torch.device("cpu"),
        chunk_size=2,
    )
    assert sorted(ranked) == [4, 6, 9]


def test_rank_fusion_has_exact_learned_and_weight_endpoints() -> None:
    class FixedPolicy:
        def rank(
            self,
            query,
            observation,
            candidates,
            *,
            frontier,
            device,
            chunk_size=65_536,
        ):
            return [5, 7]

    relation = str(Relation.PRESYNAPTIC_TO)
    observation = Observation(
        current_node=1,
        discovered_edges=(
            Evidence(EvidenceKind.EDGE, 1, relation, 5, weight=2.0),
            Evidence(EvidenceKind.EDGE, 1, relation, 7, weight=9.0),
        ),
        memory=(),
        remaining_budget=8.0,
        step=0,
    )
    query = QuerySpec("q", QueryType.TWO_HOP, (1,))
    common = {
        "query": query,
        "observation": observation,
        "candidates": [5, 7],
        "frontier": True,
        "device": torch.device("cpu"),
    }
    assert RankFusionPolicy(FixedPolicy(), 0.0).rank(**common) == [7, 5]
    assert RankFusionPolicy(FixedPolicy(), 1.0).rank(**common) == [5, 7]
