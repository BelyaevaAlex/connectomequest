"""Dataset-independent semantic contract for cross-connectome transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from connectomequest.graph import GraphStore
from connectomequest.schema import Relation


@dataclass(frozen=True, slots=True)
class SharedSemanticProfile:
    dataset: str
    version: str
    wiring_relation: str
    semantic_relation: str
    semantic_role: str = "neuron_attribute"
    entity_id_free: bool = True

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def semantic_profile(graph: GraphStore) -> SharedSemanticProfile:
    """Resolve native predicates onto roles shared by all three connectomes."""

    relations = set(graph.relations)
    wiring = str(Relation.PRESYNAPTIC_TO)
    if wiring not in relations:
        raise ValueError(f"{graph.manifest.dataset} lacks canonical {wiring}")
    semantic = next(
        (
            str(relation)
            for relation in (Relation.HAS_TYPE, Relation.IN_CORTICAL_LAYER)
            if str(relation) in relations
        ),
        None,
    )
    if semantic is None:
        raise ValueError(f"{graph.manifest.dataset} lacks a neuron-attribute semantic relation")
    return SharedSemanticProfile(
        dataset=graph.manifest.dataset,
        version=graph.manifest.version,
        wiring_relation=wiring,
        semantic_relation=semantic,
    )


def validate_shared_semantics(
    graphs: list[GraphStore],
) -> list[SharedSemanticProfile]:
    """Fail closed when a source cannot instantiate the shared query roles."""

    if not graphs:
        raise ValueError("at least one source graph is required")
    profiles = [semantic_profile(graph) for graph in graphs]
    wiring_roles = {profile.wiring_relation for profile in profiles}
    semantic_roles = {profile.semantic_role for profile in profiles}
    if len(wiring_roles) != 1 or len(semantic_roles) != 1:
        raise ValueError("source graphs do not share the same semantic contract")
    datasets = [profile.dataset for profile in profiles]
    if len(set(datasets)) != len(datasets):
        raise ValueError("leave-one-connectome-out sources must be distinct datasets")
    return profiles
