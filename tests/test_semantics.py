from pathlib import Path

import pytest

from connectomequest.schema import Relation
from connectomequest.semantics import semantic_profile, validate_shared_semantics
from connectomequest.toy import make_toy_graph


def test_semantic_profile_maps_native_relation_to_shared_role(tmp_path: Path) -> None:
    graph = make_toy_graph(tmp_path / "first")
    profile = semantic_profile(graph)
    assert profile.wiring_relation == str(Relation.PRESYNAPTIC_TO)
    assert profile.semantic_relation == str(Relation.HAS_TYPE)
    assert profile.semantic_role == "neuron_attribute"
    assert profile.entity_id_free


def test_shared_semantics_rejects_duplicate_source_dataset(tmp_path: Path) -> None:
    first = make_toy_graph(tmp_path / "first")
    second = make_toy_graph(tmp_path / "second")
    with pytest.raises(ValueError, match="distinct datasets"):
        validate_shared_semantics([first, second])
