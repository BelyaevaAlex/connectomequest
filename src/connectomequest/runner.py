"""Reproducible training and active-evaluation runners."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from connectomequest.agents.explorer import BudgetedExplorer
from connectomequest.env import ConnectomeEnv
from connectomequest.evaluation import (
    aggregate_metrics,
    aggregate_ranks,
    episode_metrics,
    filtered_ranks,
    metrics_as_dict,
)
from connectomequest.graph import GraphStore
from connectomequest.inductive_training import ObservableIndex, train_inductive_epoch
from connectomequest.inductive_training_v5 import FeatureMoments, train_domain_balanced_epoch
from connectomequest.manifest import sha256_file
from connectomequest.models.inductive import InductiveQueryPolicy
from connectomequest.models.inductive_v5 import InductiveQueryPolicyV5
from connectomequest.models.torus import TorusQueryHeuristic
from connectomequest.policies.contextual_gate import ContextualGatePolicy
from connectomequest.policies.diverse import load_diverse_policy
from connectomequest.policies.minerva import load_minerva_policy
from connectomequest.policies.oracle import PrivilegedOraclePolicy
from connectomequest.policies.rank_fusion import RankFusionPolicy
from connectomequest.policies.snapshot import load_snapshot_policy
from connectomequest.policies.snapshot_v2 import load_snapshot_policy_v2
from connectomequest.query import SEMANTIC_VISIBLE_NEIGHBORS, QueryType
from connectomequest.query_io import read_queries, validate_query_snapshot
from connectomequest.semantics import validate_shared_semantics
from connectomequest.training import SemanticTrainingIndex, collate_queries, train_step


def train_heuristic(
    graph: GraphStore,
    query_path: Path,
    output: Path,
    *,
    embedding_dim: int = 256,
    batch_queries: int = 128,
    negatives_per_positive: int = 32,
    positives_per_query: int = 16,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    seed: int = 17,
    device_name: str = "cuda",
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    validate_query_snapshot(query_path, graph)
    queries = [query for query in read_queries(query_path) if query.split == "train"]
    if not queries:
        raise ValueError("query file contains no train queries")
    model = TorusQueryHeuristic(graph.num_nodes, embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    semantic_index = SemanticTrainingIndex.from_graph(graph)
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        order = torch.randperm(len(queries), generator=generator).tolist()
        losses: list[float] = []
        model.train()
        for start in tqdm(
            range(0, len(order), batch_queries),
            desc=f"train epoch {epoch + 1}/{epochs}",
        ):
            selected = [queries[index] for index in order[start : start + batch_queries]]
            batch = collate_queries(
                selected,
                num_entities=graph.num_nodes,
                negatives_per_positive=negatives_per_positive,
                max_positives_per_query=positives_per_query,
                generator=generator,
                semantic_index=semantic_index,
            )
            losses.append(
                train_step(
                    model,
                    batch,
                    optimizer,
                    device=device,
                    use_bf16=True,
                )
            )
        epoch_losses.append(mean(losses))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_version": 2,
        "semantic_policy": True,
        "num_entities": graph.num_nodes,
        "embedding_dim": embedding_dim,
        "graph_manifest": str(graph.manifest_path),
        "graph_manifest_sha256": sha256_file(graph.manifest_path),
        "query_path": str(query_path),
        "query_sha256": sha256_file(Path(query_path)),
        "seed": seed,
        "epoch_losses": epoch_losses,
        "training_config": {
            "objective": "pairwise_answer_plus_semantic_frontier",
            "batch_queries": batch_queries,
            "negatives_per_positive": negatives_per_positive,
            "hard_negative_fraction": 0.75,
            "positives_per_query": positives_per_query,
            "frontier_negatives_per_query": 8,
            "frontier_loss_weight": 1.0,
            "semantic_visible_neighbors": SEMANTIC_VISIBLE_NEIGHBORS,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "precision": "bf16" if device.type == "cuda" else "fp32",
        },
    }
    torch.save(payload, output)
    report = {key: value for key, value in payload.items() if key not in {"model_state"}}
    report["checkpoint"] = str(output)
    report["checkpoint_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_inductive_policy(
    graph: GraphStore,
    query_path: Path,
    output: Path,
    *,
    hidden_dim: int = 128,
    query_dim: int = 32,
    dropout: float = 0.1,
    batch_queries: int = 256,
    pairs_per_query: int = 8,
    visible_neighbors: int = 32,
    subgoal_probes: int = 8,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device_name: str = "cuda",
) -> dict:
    """Train a policy whose checkpoint is valid on unseen entity vocabularies."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    validate_query_snapshot(query_path, graph)
    queries = [query for query in read_queries(query_path) if query.split == "train"]
    if not queries:
        raise ValueError("query file contains no train queries")
    model = InductiveQueryPolicy(hidden_dim, query_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []
    epoch_pairs: list[int] = []
    for epoch in range(epochs):
        loss, pairs = train_inductive_epoch(
            model,
            graph,
            queries,
            optimizer,
            device=device,
            batch_queries=batch_queries,
            pairs_per_query=pairs_per_query,
            visible_neighbors=visible_neighbors,
            subgoal_probes=subgoal_probes,
            seed=seed + epoch,
            use_bf16=True,
        )
        epoch_losses.append(loss)
        epoch_pairs.append(pairs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_kind": "entity_id_free_frontier_policy",
        "model_version": 4,
        "hidden_dim": hidden_dim,
        "query_dim": query_dim,
        "feature_dim": model.feature_dim,
        "dropout": dropout,
        "source_graph_manifest_sha256": sha256_file(graph.manifest_path),
        "source_query_sha256": sha256_file(Path(query_path)),
        "seed": seed,
        "epoch_losses": epoch_losses,
        "epoch_pairs": epoch_pairs,
        "training_config": {
            "objective": "pairwise_observable_oracle_behavior_cloning",
            "batch_queries": batch_queries,
            "pairs_per_query": pairs_per_query,
            "visible_neighbors": visible_neighbors,
            "epochs": epochs,
            "subgoal_probes": subgoal_probes,
            "learning_rate": learning_rate,
            "precision": "bf16" if device.type == "cuda" else "fp32",
            "entity_embeddings": False,
        },
    }
    torch.save(payload, output)
    report = {key: value for key, value in payload.items() if key != "model_state"}
    report["checkpoint"] = str(output)
    report["checkpoint_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_inductive_policy_multi(
    graphs: list[GraphStore],
    query_paths: list[Path],
    output: Path,
    *,
    hidden_dim: int = 128,
    query_dim: int = 32,
    dropout: float = 0.1,
    batch_queries: int = 256,
    pairs_per_query: int = 8,
    visible_neighbors: int = 32,
    max_queries_per_source: int = 8_000,
    subgoal_probes: int = 8,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    seed: int = 17,
    device_name: str = "cuda",
) -> dict:
    """Train one ID-free policy on distinct source connectomes only."""

    if len(graphs) != len(query_paths):
        raise ValueError("graphs and query_paths must have equal length")
    profiles = validate_shared_semantics(graphs)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    source_queries = []
    source_snapshots = []
    for source_index, (graph, query_path, profile) in enumerate(
        zip(graphs, query_paths, profiles, strict=True)
    ):
        validate_query_snapshot(query_path, graph)
        queries = [
            query
            for query in read_queries(query_path)
            if query.split == "train" and query.query_type == QueryType.TWO_HOP_TYPE
        ]
        if not queries:
            raise ValueError(f"{profile.dataset} contains no train/2pt queries")
        sampler = random.Random(seed + source_index)
        sampler.shuffle(queries)
        queries = queries[:max_queries_per_source]
        source_queries.append(queries)
        source_snapshots.append(
            {
                **profile.to_dict(),
                "graph_manifest_sha256": sha256_file(graph.manifest_path),
                "query_sha256": sha256_file(Path(query_path)),
                "train_queries": len(queries),
            }
        )
    model = InductiveQueryPolicy(hidden_dim, query_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_reports: list[list[dict[str, float | int | str]]] = []
    for epoch in range(epochs):
        source_order = list(range(len(graphs)))
        random.Random(seed + epoch).shuffle(source_order)
        epoch_report = []
        for source_index in source_order:
            loss, pairs = train_inductive_epoch(
                model,
                graphs[source_index],
                source_queries[source_index],
                optimizer,
                device=device,
                batch_queries=batch_queries,
                pairs_per_query=pairs_per_query,
                visible_neighbors=visible_neighbors,
                seed=seed + epoch * len(graphs) + source_index,
                subgoal_probes=subgoal_probes,
                use_bf16=True,
            )
            epoch_report.append(
                {
                    "dataset": profiles[source_index].dataset,
                    "loss": loss,
                    "pairs": pairs,
                }
            )
        epoch_reports.append(epoch_report)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_kind": "entity_id_free_frontier_policy",
        "model_version": 4,
        "hidden_dim": hidden_dim,
        "query_dim": query_dim,
        "feature_dim": model.feature_dim,
        "dropout": dropout,
        "source_snapshots": source_snapshots,
        "seed": seed,
        "epoch_reports": epoch_reports,
        "training_config": {
            "protocol": "leave_one_connectome_out",
            "objective": "pairwise_observable_oracle_behavior_cloning",
            "query_type": "2pt",
            "batch_queries": batch_queries,
            "pairs_per_query": pairs_per_query,
            "visible_neighbors": visible_neighbors,
            "max_queries_per_source": max_queries_per_source,
            "epochs": epochs,
            "subgoal_probes": subgoal_probes,
            "learning_rate": learning_rate,
            "precision": "bf16" if device.type == "cuda" else "fp32",
            "entity_embeddings": False,
        },
    }
    torch.save(payload, output)
    result = {key: value for key, value in payload.items() if key != "model_state"}
    result["checkpoint"] = str(output)
    result["checkpoint_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def train_inductive_policy_v5_multi(
    graphs: list[GraphStore],
    query_paths: list[Path],
    output: Path,
    *,
    hidden_dim: int = 128,
    query_dim: int = 32,
    dropout: float = 0.1,
    batch_queries: int = 128,
    visible_neighbors: int = 32,
    max_queries_per_source: int = 8_000,
    probe_choices: tuple[int, ...] = (0, 1, 2, 4),
    epochs: int = 8,
    learning_rate: float = 3e-4,
    temperature: float = 0.5,
    domain_variance_weight: float = 0.1,
    seed: int = 17,
    device_name: str = "cuda",
) -> dict:
    """Train v5 with per-step domain balance and an observable listwise target."""

    if len(graphs) != len(query_paths) or len(graphs) < 1:
        raise ValueError("graphs and query_paths must have equal, non-zero length")
    profiles = validate_shared_semantics(graphs)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    source_queries: list[list] = []
    source_snapshots: list[dict] = []
    for source_index, (graph, query_path, profile) in enumerate(
        zip(graphs, query_paths, profiles, strict=True)
    ):
        validate_query_snapshot(query_path, graph)
        queries = [
            query
            for query in read_queries(query_path)
            if query.split == "train" and query.query_type == QueryType.TWO_HOP_TYPE
        ]
        if not queries:
            raise ValueError(f"{profile.dataset} contains no train/2pt queries")
        sampler = random.Random(seed + source_index)
        sampler.shuffle(queries)
        queries = queries[:max_queries_per_source]
        source_queries.append(queries)
        source_snapshots.append(
            {
                **profile.to_dict(),
                "graph_manifest_sha256": sha256_file(graph.manifest_path),
                "query_sha256": sha256_file(Path(query_path)),
                "train_queries": len(queries),
            }
        )
    indices = [ObservableIndex(graph, visible_neighbors) for graph in graphs]
    model = InductiveQueryPolicyV5(hidden_dim, query_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    reports: list[dict] = []
    aggregate_moments = FeatureMoments.empty()
    for epoch in range(epochs):
        report, moments = train_domain_balanced_epoch(
            model,
            indices,
            source_queries,
            optimizer,
            device=device,
            batch_queries=batch_queries,
            probe_choices=probe_choices,
            temperature=temperature,
            domain_variance_weight=domain_variance_weight,
            seed=seed + epoch * 10_000,
            use_bf16=True,
        )
        aggregate_moments.total += moments.total
        aggregate_moments.squared_total += moments.squared_total
        aggregate_moments.count += moments.count
        reports.append({"epoch": epoch + 1, **report})
    feature_mean, feature_std = aggregate_moments.mean_std()
    model.set_feature_reference(feature_mean.to(device), feature_std.to(device))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_kind": "entity_id_free_frontier_policy_v5",
        "model_version": 5,
        "hidden_dim": hidden_dim,
        "query_dim": query_dim,
        "feature_dim": model.feature_dim,
        "dropout": dropout,
        "source_snapshots": source_snapshots,
        "seed": seed,
        "epoch_reports": reports,
        "training_config": {
            "protocol": "leave_one_connectome_out",
            "objective": "domain_balanced_listwise_observable_utility",
            "query_type": "2pt",
            "batch_queries_per_source": batch_queries,
            "visible_neighbors": visible_neighbors,
            "max_queries_per_source": max_queries_per_source,
            "probe_choices": list(probe_choices),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "domain_variance_weight": domain_variance_weight,
            "precision": "bf16" if device.type == "cuda" else "fp32",
            "entity_embeddings": False,
            "domain_ids": False,
            "absolute_weight_prior": False,
        },
    }
    torch.save(payload, output)
    result = {key: value for key, value in payload.items() if key != "model_state"}
    result["checkpoint"] = str(output)
    result["checkpoint_sha256"] = sha256_file(output)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_inductive_policy_v5(checkpoint: Path, device: torch.device) -> InductiveQueryPolicyV5:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("model_kind") != "entity_id_free_frontier_policy_v5":
        raise ValueError("checkpoint is not a v5 inductive frontier policy")
    model = InductiveQueryPolicyV5(
        payload["hidden_dim"], payload["query_dim"], payload["dropout"]
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_inductive_policy(checkpoint: Path, device: torch.device) -> InductiveQueryPolicy:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("model_kind") != "entity_id_free_frontier_policy":
        raise ValueError("checkpoint is not an inductive frontier policy")
    model = InductiveQueryPolicy(
        payload["hidden_dim"],
        payload["query_dim"],
        payload["dropout"],
        payload.get("feature_dim", 17),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_heuristic(
    checkpoint: Path,
    graph: GraphStore,
    device: torch.device,
) -> TorusQueryHeuristic:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["num_entities"] != graph.num_nodes:
        raise ValueError(
            f"checkpoint has {payload['num_entities']} entities, graph has {graph.num_nodes}"
        )
    expected_graph = payload.get("graph_manifest_sha256")
    if expected_graph is not None and expected_graph != sha256_file(graph.manifest_path):
        raise ValueError("checkpoint graph manifest hash does not match the active graph")

    model = TorusQueryHeuristic(
        payload["num_entities"],
        payload["embedding_dim"],
        semantic_policy=payload.get("model_version", 1) >= 2,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def benchmark_active(
    graph: GraphStore,
    query_path: Path,
    output: Path,
    *,
    split: str = "test",
    query_type: str | None = None,
    ranking: str = "weight",
    checkpoint: Path | None = None,
    fusion_alpha: float = 0.5,
    visible_neighbors: int = 32,
    budget: int = 64,
    answer_quota: int | None = 1,
    gate_min_margin: float = 0.05,
    gate_max_entropy: float = 0.95,
    gate_max_ood: float = 3.0,
    adaptive_probing: bool = False,
    max_subgoal_probes: int = 4,
    probe_margin_threshold: float = 0.05,
    probe_entropy_threshold: float = 0.95,
    enable_backtracking: bool = True,
    enable_budget_mask: bool = True,
    use_native_semantics: bool = True,
    policy_local_observation: bool = False,
    enforce_proof: bool = True,
    candidates_per_subgoal: int = 4,
    subgoal_probes: int = 0,
    offset: int = 0,
    limit: int | None = None,
    seed: int = 17,
    device_name: str = "cuda",
) -> dict:
    validate_query_snapshot(query_path, graph)
    queries = [query for query in read_queries(query_path) if query.split == split]
    if query_type is not None:
        selected_type = QueryType(query_type)
        queries = [query for query in queries if query.query_type == selected_type]
    if not queries:
        raise ValueError(f"query file contains no {split}/{query_type or 'all'} queries")
    queries = queries[offset:]
    if limit is not None:
        queries = queries[:limit]
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = None
    policy = None
    if ranking == "neural":
        if checkpoint is None:
            raise ValueError("neural benchmark requires checkpoint")
        model = load_heuristic(checkpoint, graph, device)
    elif ranking in {"inductive", "inductive_fusion"}:
        if checkpoint is None:
            raise ValueError("inductive benchmark requires checkpoint")
        learned_policy = load_inductive_policy(checkpoint, device)
        policy = (
            RankFusionPolicy(learned_policy, fusion_alpha)
            if ranking == "inductive_fusion"
            else learned_policy
        )
    elif ranking in {"inductive_v5", "inductive_v5_context", "inductive_v5_fusion"}:
        if checkpoint is None:
            raise ValueError("v5 inductive benchmark requires checkpoint")
        learned_policy_v5 = load_inductive_policy_v5(checkpoint, device)
        if ranking == "inductive_v5_fusion":
            policy = RankFusionPolicy(learned_policy_v5, fusion_alpha)
        elif ranking == "inductive_v5_context":
            policy = ContextualGatePolicy(
                learned_policy_v5,
                alpha=fusion_alpha,
                min_margin=gate_min_margin,
                max_entropy=gate_max_entropy,
                max_ood=gate_max_ood,
            )
        else:
            policy = learned_policy_v5
    elif ranking == "minerva":
        if checkpoint is None:
            raise ValueError("MINERVA benchmark requires checkpoint")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        policy = load_minerva_policy(payload, device)
    elif ranking == "snapshot":
        if checkpoint is None:
            raise ValueError("snapshot benchmark requires checkpoint")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        policy = load_snapshot_policy(payload, device)
    elif ranking == "snapshot_v2":
        if checkpoint is None:
            raise ValueError("snapshot_v2 benchmark requires checkpoint")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        policy = load_snapshot_policy_v2(payload, device)
    elif ranking == "diverse":
        if checkpoint is None:
            raise ValueError("diverse benchmark requires checkpoint")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        policy = load_diverse_policy(payload, device)
    elif ranking == "oracle":
        policy = PrivilegedOraclePolicy(
            graph,
            {query.query_id: query.answers for query in queries},
        )
    explorer = BudgetedExplorer(
        model=model,
        policy=policy,
        device=device,
        ranking="policy"
        if ranking
        in {
            "inductive",
            "inductive_fusion",
            "inductive_v5",
            "inductive_v5_context",
            "inductive_v5_fusion",
            "minerva",
            "snapshot",
            "snapshot_v2",
            "diverse",
            "oracle",
        }
        else ranking,
        seed=seed,
        adaptive_probing=adaptive_probing,
        max_subgoal_probes=max_subgoal_probes,
        probe_margin_threshold=probe_margin_threshold,
        probe_entropy_threshold=probe_entropy_threshold,
        enable_backtracking=enable_backtracking,
        enable_budget_mask=enable_budget_mask,
        use_native_semantics=use_native_semantics,
        policy_local_observation=policy_local_observation,
        subgoal_probes=subgoal_probes,
        candidates_per_subgoal=candidates_per_subgoal,
    )
    rows: list[dict] = []
    metrics_by_type: dict[str, list] = defaultdict(list)
    for query in tqdm(queries, desc=f"benchmark {ranking}"):
        env = ConnectomeEnv(
            graph,
            visible_neighbors=visible_neighbors,
            max_steps=budget,
            budget=budget,
            enforce_proof=enforce_proof,
            answer_quota=answer_quota,
        )
        env.reset(query)
        result = explorer.run(env, query.spec)
        metrics = episode_metrics(query, result)
        metrics_by_type[str(query.query_type)].append(metrics)
        rows.append(
            {
                **metrics_as_dict(metrics),
                "query_type": str(query.query_type),
                "split": query.split,
                "ranking": ranking,
                "budget": budget,
                "answer_quota": answer_quota,
                "num_expected_answers": len(query.answers),
                "num_submitted_answers": len(result.submitted_answers),
            }
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    all_metrics = [metric for values in metrics_by_type.values() for metric in values]
    summary = {
        "overall": aggregate_metrics(all_metrics),
        "by_query_type": {
            query_type: aggregate_metrics(values)
            for query_type, values in sorted(metrics_by_type.items())
        },
        "protocol": {
            "split": split,
            "query_type": query_type,
            "ranking": ranking,
            "access_regime": ("privileged_ceiling" if ranking == "oracle" else "observable"),
            "visible_neighbors": visible_neighbors,
            "fusion_alpha": fusion_alpha
            if ranking in {"inductive_fusion", "inductive_v5_context", "inductive_v5_fusion"}
            else None,
            "gate_min_margin": gate_min_margin if ranking == "inductive_v5_context" else None,
            "gate_max_entropy": gate_max_entropy if ranking == "inductive_v5_context" else None,
            "gate_max_ood": gate_max_ood if ranking == "inductive_v5_context" else None,
            "adaptive_probing": adaptive_probing,
            "max_subgoal_probes": max_subgoal_probes,
            "probe_margin_threshold": probe_margin_threshold,
            "probe_entropy_threshold": probe_entropy_threshold,
            "enable_backtracking": enable_backtracking,
            "enable_budget_mask": enable_budget_mask,
            "use_native_semantics": use_native_semantics,
            "policy_local_observation": policy_local_observation,
            "enforce_proof": enforce_proof,
            "budget": budget,
            "answer_quota": answer_quota,
            "candidates_per_subgoal": candidates_per_subgoal,
            "offset": offset,
            "subgoal_probes": subgoal_probes,
            "seed": seed,
            "graph_manifest_sha256": sha256_file(graph.manifest_path),
            "query_sha256": sha256_file(Path(query_path)),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        },
    }
    if isinstance(policy, ContextualGatePolicy):
        summary["contextual_gate"] = policy.stats()
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def evaluate_static(
    graph: GraphStore,
    query_path: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "test",
    query_type: str | None = None,
    limit: int | None = None,
    device_name: str = "cuda",
) -> dict:
    """Evaluate a checkpoint with standard answer-filtered entity ranks."""

    validate_query_snapshot(query_path, graph)
    queries = [query for query in read_queries(query_path) if query.split == split]
    if query_type is not None:
        selected_type = QueryType(query_type)
        queries = [query for query in queries if query.query_type == selected_type]
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError(f"query file contains no {split} queries")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = load_heuristic(checkpoint, graph, device)
    entities = list(range(graph.num_nodes))
    ranks_by_type: dict[str, list[int]] = defaultdict(list)
    query_counts: dict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for query in tqdm(queries, desc="filtered static CQA"):
            ordered = model.rank_candidates(
                query.query_type,
                query.anchors,
                entities,
                device=device,
            )
            query_type = str(query.query_type)
            ranks_by_type[query_type].extend(filtered_ranks(ordered, query.answers))
            query_counts[query_type] += 1
    all_ranks = [rank for ranks in ranks_by_type.values() for rank in ranks]
    summary = {
        "overall": aggregate_ranks(all_ranks, queries=len(queries)),
        "by_query_type": {
            query_type: aggregate_ranks(ranks, queries=query_counts[query_type])
            for query_type, ranks in sorted(ranks_by_type.items())
        },
        "protocol": {
            "split": split,
            "query_type": query_type,
            "filtering": "all other true answers removed",
            "candidate_space": "all graph entities",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "graph_manifest_sha256": sha256_file(graph.manifest_path),
            "query_sha256": sha256_file(Path(query_path)),
            "device": str(device),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
