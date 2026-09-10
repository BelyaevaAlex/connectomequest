from __future__ import annotations

import torch

from connectomequest.policies.snapshot_v2 import (
    MaskedSnapshotHeadV2,
    SnapshotPolicyV2,
    SnapshotPolicyV2Config,
    load_snapshot_policy_v2,
)
from connectomequest.snapshot_training import SnapshotLists
from connectomequest.snapshot_training_v2 import (
    balanced_domain_stage_indices,
    concatenate_snapshot_lists,
    stage_aware_listwise_utility_loss,
)


def _lists() -> SnapshotLists:
    # Unequal source counts make accidental unbalanced sampling visible.
    domains = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    stages = torch.tensor([0, 0, 0, 1, 0, 1, 1])
    rows = len(domains)
    return SnapshotLists(
        features=torch.zeros(rows, 2, 31),
        relevant=torch.tensor([[1.0, 0.0]] * rows),
        mask=torch.ones(rows, 2, dtype=torch.bool),
        query_ids=torch.zeros(rows, dtype=torch.long),
        semantic_flags=torch.zeros(rows),
        stages=stages,
        domains=domains,
    )


def test_balanced_indices_draw_equal_domain_stage_strata() -> None:
    data = _lists()
    indices = balanced_domain_stage_indices(
        data,
        samples_per_stratum=4,
        generator=torch.Generator().manual_seed(17),
    )
    pairs = list(zip(data.domains[indices].tolist(), data.stages[indices].tolist()))
    assert set(pairs) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(pairs.count(pair) == 4 for pair in set(pairs))


def test_cat_pads_candidate_width_before_concatenation() -> None:
    narrow = SnapshotLists(
        features=torch.full((1, 2, 31), 1.0),
        relevant=torch.tensor([[1.0, 0.0]]),
        mask=torch.ones(1, 2, dtype=torch.bool),
        query_ids=torch.tensor([3]),
        semantic_flags=torch.tensor([1.0]),
        stages=torch.tensor([0]),
        domains=torch.tensor([4]),
    )
    wide = SnapshotLists(
        features=torch.full((1, 3, 31), 2.0),
        relevant=torch.tensor([[0.0, 1.0, 0.0]]),
        mask=torch.ones(1, 3, dtype=torch.bool),
        query_ids=torch.tensor([5]),
        semantic_flags=torch.tensor([0.0]),
        stages=torch.tensor([1]),
        domains=torch.tensor([6]),
    )

    combined = concatenate_snapshot_lists([narrow, wide])

    assert combined.features.shape == (2, 3, 31)
    assert combined.relevant.shape == (2, 3)
    assert combined.mask.shape == (2, 3)
    assert torch.equal(combined.features[0, 2], torch.zeros(31))
    assert combined.relevant[0, 2].item() == 0.0
    assert not combined.mask[0, 2]
    assert combined.query_ids.tolist() == [3, 5]
    assert combined.semantic_flags.tolist() == [1.0, 0.0]
    assert combined.stages.tolist() == [0, 1]
    assert combined.domains.tolist() == [4, 6]


def test_stage_aware_utility_loss_respects_stage_weights_and_mask() -> None:
    scores = torch.tensor([[0.0, 0.0, 99.0], [0.0, 0.0, 99.0]])
    relevant = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    stages = torch.tensor([0, 1])
    loss = stage_aware_listwise_utility_loss(
        scores,
        relevant,
        mask,
        stages,
        first_hop_weight=3.0,
        second_hop_weight=1.0,
    )
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)))


def test_masked_head_has_distinct_missing_sketch_path() -> None:
    head = MaskedSnapshotHeadV2(hidden_dim=16, query_dim=8, dropout=0.0)
    features = torch.zeros(2, 31)
    features[1, 22] = 1.0  # candidate 1 has a visible sketch
    query_ids = torch.zeros(2, dtype=torch.long)
    semantic = torch.zeros(2)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.missing_sketch_bias.fill_(2.0)
    scores = head(features, query_ids, semantic)
    assert scores.tolist() == [2.0, 0.0]


def test_no_probe_features_disables_visible_and_missing_probe_paths() -> None:
    head = MaskedSnapshotHeadV2(
        hidden_dim=16,
        query_dim=8,
        dropout=0.0,
        use_weight_prior=False,
        use_probe_features=False,
    )
    features = torch.zeros(2, 31)
    features[1, 22] = 1.0
    query_ids = torch.zeros(2, dtype=torch.long)
    semantic = torch.zeros(2)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.missing_sketch_bias.fill_(2.0)
    scores = head(features, query_ids, semantic)
    assert scores.tolist() == [0.0, 0.0]


def test_no_weight_prior_removes_normalized_prior_contribution() -> None:
    features = torch.zeros(2, 31)
    features[:, 19] = torch.tensor([0.0, 1.0])
    query_ids = torch.zeros(2, dtype=torch.long)
    semantic = torch.zeros(2)
    with_prior = MaskedSnapshotHeadV2(hidden_dim=16, query_dim=8, dropout=0.0)
    no_prior = MaskedSnapshotHeadV2(
        hidden_dim=16,
        query_dim=8,
        dropout=0.0,
        use_weight_prior=False,
    )
    with torch.no_grad():
        for model in (with_prior, no_prior):
            for parameter in model.parameters():
                parameter.zero_()
        with_prior.relative_weight_scale_raw.fill_(1.0)
        no_prior.relative_weight_scale_raw.fill_(1.0)
    assert with_prior(features, query_ids, semantic).tolist()[1] > 0.0
    assert no_prior(features, query_ids, semantic).tolist() == [0.0, 0.0]


def test_no_query_embedding_makes_query_ids_invariant() -> None:
    features = torch.zeros(2, 31)
    semantic = torch.zeros(2)
    head = MaskedSnapshotHeadV2(
        hidden_dim=16,
        query_dim=8,
        dropout=0.0,
        use_weight_prior=False,
        use_query_embedding=False,
    )
    with torch.no_grad():
        head.base_scorer[-1].weight.fill_(0.1)
        head.query_embedding.weight.normal_()
    scores = head(features, torch.tensor([0, 1]), semantic)
    assert torch.isclose(scores[0], scores[1])


def test_snapshot_v2_loads_legacy_config_without_ablation_flags() -> None:
    model = SnapshotPolicyV2(hidden_dim=16, query_dim=8, dropout=0.0)
    payload = {
        "model_kind": "entity_id_free_snapshot_policy_v2",
        "model_config": {"hidden_dim": 16, "query_dim": 8, "dropout": 0.0},
        "model_state": model.state_dict(),
    }
    loaded = load_snapshot_policy_v2(payload, torch.device("cpu"))
    assert isinstance(loaded, SnapshotPolicyV2)
    assert loaded.config == SnapshotPolicyV2Config(16, 8, 0.0)


def test_snapshot_v2_returns_complete_candidate_order() -> None:
    from connectomequest.env import Observation
    from connectomequest.query import QuerySpec, QueryType

    policy = SnapshotPolicyV2(hidden_dim=16, query_dim=8, dropout=0.0)
    query = QuerySpec("q", QueryType.TWO_HOP_TYPE, (0, 9), "has_type")
    observation = Observation(0, (), (), 63.0, 1)
    candidates = [4, 2, 7]
    order = policy.rank(
        query,
        observation,
        candidates,
        frontier=True,
        device=torch.device("cpu"),
    )
    assert len(order) == len(candidates)
    assert set(order) == set(candidates)
