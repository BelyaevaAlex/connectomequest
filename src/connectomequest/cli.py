"""Command-line entrypoint for data, validation, and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from connectomequest.adapters.h01 import (
    annotate_h01_layers,
    build_h01_streaming,
    map_h01_somas,
)
from connectomequest.adapters.hemibrain import build_hemibrain, export_hemibrain
from connectomequest.adapters.manc import build_manc
from connectomequest.baselines.export import (
    export_baseline_bundle,
    export_link_prediction_bundle,
)
from connectomequest.doctor import report_dict
from connectomequest.download import download_h01, download_manc, h01_avro_objects, inventory_h01
from connectomequest.graph import GraphStore
from connectomequest.query import QueryType
from connectomequest.query_io import generate_query_set
from connectomequest.runner import (
    benchmark_active,
    evaluate_static,
    train_heuristic,
    train_inductive_policy,
    train_inductive_policy_multi,
    train_inductive_policy_v5_multi,
)
from connectomequest.statistics import compare_active_runs
from connectomequest.toy import make_toy_graph

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def doctor() -> None:
    """Report compute resources and acceleration support."""

    typer.echo(json.dumps(report_dict(), indent=2))


@app.command("make-toy")
def make_toy(
    output: Annotated[Path, typer.Option(help="Processed graph directory")] = Path(
        "data/processed/toy"
    ),
) -> None:
    graph = make_toy_graph(output)
    typer.echo(f"created {graph.num_nodes} nodes at {output}")


@app.command("validate-graph")
def validate_graph(path: Path) -> None:
    graph = GraphStore(path)
    summary = {
        "dataset": graph.manifest.dataset,
        "version": graph.manifest.version,
        "nodes": graph.num_nodes,
        "relations": graph.relations,
        "manifest": graph.manifest.metadata,
    }
    for relation in graph.relations:
        matrix = graph.csr(relation)
        if matrix.shape != (graph.num_nodes, graph.num_nodes):
            raise typer.BadParameter(f"bad CSR shape for {relation}")
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def download(
    dataset: Annotated[str, typer.Argument(help="manc, hemibrain, or h01")],
    raw_dir: Annotated[Path, typer.Option(help="Destination directory")],
    inventory_only: Annotated[
        bool, typer.Option("--inventory-only", help="List H01 objects without downloading")
    ] = False,
    max_gib: Annotated[float, typer.Option(help="H01 download safety gate in GiB")] = 200.0,
) -> None:
    normalized = dataset.lower()
    if normalized == "manc":
        if inventory_only:
            raise typer.BadParameter("--inventory-only is only valid for H01")
        manifest = download_manc(raw_dir)
        typer.echo(f"downloaded {len(manifest.artifacts)} MANC files")
    elif normalized == "h01":
        if inventory_only:
            items, total = inventory_h01(raw_dir)
            avro = h01_avro_objects(items)
            avro_bytes = sum(item["size"] for item in avro)
            typer.echo(
                f"{len(items)} objects, {total / 2**30:.2f} GiB total; "
                f"{len(avro)} Avro objects, {avro_bytes / 2**30:.2f} GiB selected"
            )
        else:
            manifest = download_h01(raw_dir, max_bytes=int(max_gib * 2**30))
            typer.echo(f"downloaded {len(manifest.artifacts)} H01 files")
    elif normalized == "hemibrain":
        manifest = export_hemibrain(raw_dir)
        typer.echo(f"exported {len(manifest.artifacts)} HemiBrain files")
    else:
        raise typer.BadParameter(f"unsupported dataset: {dataset}")


@app.command("build-manc")
def build_manc_command(
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw/manc"),
    output: Annotated[Path, typer.Option()] = Path("data/processed/manc-v1.0"),
    min_weight: Annotated[float, typer.Option()] = 1.0,
) -> None:
    graph = build_manc(raw_dir, output, min_weight=min_weight)
    typer.echo(f"built {graph.num_nodes} nodes with relations {graph.relations}")


@app.command("map-h01-somas")
def map_h01_somas_command(
    output: Annotated[Path, typer.Option()] = Path("data/interim/h01/soma_segments.parquet"),
    workers: Annotated[int, typer.Option(min=1, max=32)] = 8,
    max_shards: Annotated[
        int | None,
        typer.Option(help="Pilot-only cap; omit for the full official layer"),
    ] = None,
) -> None:
    table = map_h01_somas(output, workers=workers, max_shards=max_shards)
    typer.echo(f"mapped {len(table)} soma labels to C3 segments at {output}")


@app.command("annotate-h01-layers")
def annotate_h01_layers_command(
    soma_map: Annotated[Path, typer.Option()] = Path("data/interim/h01/soma_segments.parquet"),
    mip: Annotated[int, typer.Option(min=0, max=6)] = 3,
    workers: Annotated[int, typer.Option(min=1, max=32)] = 8,
) -> None:
    table = annotate_h01_layers(soma_map, mip=mip, workers=workers)
    annotated = sum(value.as_py() != 0 for value in table["cortical_layer"])
    typer.echo(f"annotated {annotated}/{len(table)} somas with official cortical layers")


@app.command("build-h01-stream")
def build_h01_stream_command(
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw/h01"),
    soma_map: Annotated[Path, typer.Option()] = Path("data/interim/h01/soma_segments.parquet"),
    output: Annotated[Path, typer.Option()] = Path("data/processed/h01-20210729-c3"),
    scratch_dir: Annotated[Path, typer.Option()] = Path("/tmp/connectomequest-h01"),
    workers: Annotated[int, typer.Option(min=1, max=24)] = 8,
    min_weight: Annotated[int, typer.Option(min=1)] = 1,
    max_shards: Annotated[
        int | None,
        typer.Option(help="Pilot-only cap; omit for all 166 Avro objects"),
    ] = None,
) -> None:
    graph = build_h01_streaming(
        raw_dir,
        soma_map,
        output,
        scratch_dir=scratch_dir,
        workers=workers,
        minimum_pair_weight=min_weight,
        max_shards=max_shards,
    )
    typer.echo(f"built {graph.num_nodes} nodes with relations {graph.relations}")


@app.command("export-hemibrain")
def export_hemibrain_command(
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw/hemibrain"),
    chunk_size: Annotated[int, typer.Option(min=16, max=2048)] = 256,
) -> None:
    manifest = export_hemibrain(raw_dir, chunk_size=chunk_size)
    typer.echo(f"exported {len(manifest.artifacts)} HemiBrain files")


@app.command("build-hemibrain")
def build_hemibrain_command(
    raw_dir: Annotated[Path, typer.Option()] = Path("data/raw/hemibrain"),
    output: Annotated[Path, typer.Option()] = Path("data/processed/hemibrain-v1.2.1"),
    min_weight: Annotated[float, typer.Option(min=1)] = 1.0,
) -> None:
    graph = build_hemibrain(raw_dir, output, min_weight=min_weight)
    typer.echo(f"built {graph.num_nodes} nodes with relations {graph.relations}")


@app.command("generate-queries")
def generate_queries_command(
    graph_path: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option()] = Path("data/processed/queries.parquet"),
    count_per_type: Annotated[int, typer.Option(min=1)] = 10_000,
    seed: Annotated[int, typer.Option()] = 17,
    query_types: Annotated[str | None, typer.Option(help="Comma-separated query families")] = None,
) -> None:
    graph = GraphStore(graph_path)
    queries = generate_query_set(
        graph,
        output,
        count_per_type=count_per_type,
        seed=seed,
        query_types=(
            tuple(QueryType(value.strip()) for value in query_types.split(","))
            if query_types
            else None
        ),
    )
    typer.echo(f"wrote {len(queries)} queries to {output}")


@app.command("export-baselines")
def export_baselines_command(
    graph_path: Annotated[Path, typer.Argument()],
    query_path: Annotated[Path, typer.Argument()],
    output: Annotated[
        Path,
        typer.Option(help="Shared triple and BetaE-format bundle directory"),
    ] = Path("data/processed/baselines"),
) -> None:
    """Export one pinned snapshot for all learned CQA and path-search baselines."""

    report = export_baseline_bundle(
        GraphStore(graph_path),
        query_path,
        output,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("export-link-prediction")
def export_link_prediction_command(
    graph_path: Annotated[Path, typer.Argument()],
    output: Annotated[
        Path, typer.Option(help="Leakage-safe 1p edge-holdout bundle directory")
    ] = Path("data/processed/link-prediction"),
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Export a deterministic edge holdout for path-search baselines."""
    report = export_link_prediction_bundle(GraphStore(graph_path), output, seed=seed)
    typer.echo(json.dumps(report, indent=2))


@app.command("train")
def train_command(
    graph_path: Annotated[Path, typer.Argument()],
    query_path: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option()] = Path("checkpoints/torus-heuristic.pt"),
    embedding_dim: Annotated[int, typer.Option(min=16)] = 256,
    batch_queries: Annotated[int, typer.Option(min=1)] = 128,
    epochs: Annotated[int, typer.Option(min=1)] = 5,
    learning_rate: Annotated[float, typer.Option(min=1e-8)] = 1e-3,
    seed: Annotated[int, typer.Option()] = 17,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    report = train_heuristic(
        GraphStore(graph_path),
        query_path,
        output,
        embedding_dim=embedding_dim,
        batch_queries=batch_queries,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("train-inductive")
def train_inductive_command(
    graph_path: Annotated[Path, typer.Argument()],
    query_path: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option()] = Path("checkpoints/inductive-policy.pt"),
    hidden_dim: Annotated[int, typer.Option(min=16)] = 128,
    query_dim: Annotated[int, typer.Option(min=8)] = 32,
    batch_queries: Annotated[int, typer.Option(min=1)] = 256,
    pairs_per_query: Annotated[int, typer.Option(min=1)] = 8,
    visible_neighbors: Annotated[int, typer.Option(min=1)] = 32,
    subgoal_probes: Annotated[int, typer.Option(min=0)] = 8,
    epochs: Annotated[int, typer.Option(min=1)] = 5,
    learning_rate: Annotated[float, typer.Option(min=1e-8)] = 3e-4,
    seed: Annotated[int, typer.Option()] = 17,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Train the entity-ID-free policy for zero-shot cross-connectome transfer."""

    report = train_inductive_policy(
        GraphStore(graph_path),
        query_path,
        output,
        hidden_dim=hidden_dim,
        query_dim=query_dim,
        batch_queries=batch_queries,
        pairs_per_query=pairs_per_query,
        visible_neighbors=visible_neighbors,
        subgoal_probes=subgoal_probes,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("train-inductive-multi")
def train_inductive_multi_command(
    graph_paths: Annotated[str, typer.Argument(help="Comma-separated source graph directories")],
    query_paths: Annotated[str, typer.Argument(help="Comma-separated source query Parquet files")],
    output: Annotated[Path, typer.Option()] = Path("checkpoints/inductive-multi.pt"),
    hidden_dim: Annotated[int, typer.Option(min=16)] = 128,
    query_dim: Annotated[int, typer.Option(min=8)] = 32,
    batch_queries: Annotated[int, typer.Option(min=1)] = 256,
    pairs_per_query: Annotated[int, typer.Option(min=1)] = 8,
    visible_neighbors: Annotated[int, typer.Option(min=1)] = 32,
    subgoal_probes: Annotated[int, typer.Option(min=0)] = 8,
    max_queries_per_source: Annotated[int, typer.Option(min=1)] = 8_000,
    epochs: Annotated[int, typer.Option(min=1)] = 5,
    learning_rate: Annotated[float, typer.Option(min=1e-8)] = 3e-4,
    seed: Annotated[int, typer.Option()] = 17,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Train on multiple sources for leave-one-connectome-out transfer."""

    graph_values = [Path(value.strip()) for value in graph_paths.split(",") if value.strip()]
    query_values = [Path(value.strip()) for value in query_paths.split(",") if value.strip()]
    if len(graph_values) != len(query_values):
        raise typer.BadParameter("graph and query path counts must match")
    report = train_inductive_policy_multi(
        [GraphStore(path) for path in graph_values],
        query_values,
        output,
        hidden_dim=hidden_dim,
        query_dim=query_dim,
        batch_queries=batch_queries,
        pairs_per_query=pairs_per_query,
        visible_neighbors=visible_neighbors,
        subgoal_probes=subgoal_probes,
        max_queries_per_source=max_queries_per_source,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device_name=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("train-inductive-v5-multi")
def train_inductive_v5_multi_command(
    graph_paths: Annotated[str, typer.Argument(help="Comma-separated source graphs")],
    query_paths: Annotated[str, typer.Argument(help="Comma-separated source queries")],
    output: Annotated[Path, typer.Option()] = Path("checkpoints/inductive-v5-multi.pt"),
    hidden_dim: Annotated[int, typer.Option(min=16)] = 128,
    query_dim: Annotated[int, typer.Option(min=8)] = 32,
    batch_queries: Annotated[int, typer.Option(min=1)] = 128,
    visible_neighbors: Annotated[int, typer.Option(min=1)] = 32,
    max_queries_per_source: Annotated[int, typer.Option(min=1)] = 8_000,
    probe_choices: Annotated[str, typer.Option()] = "0,1,2,4",
    epochs: Annotated[int, typer.Option(min=1)] = 8,
    learning_rate: Annotated[float, typer.Option(min=1e-8)] = 3e-4,
    temperature: Annotated[float, typer.Option(min=1e-4)] = 0.5,
    domain_variance_weight: Annotated[float, typer.Option(min=0.0)] = 0.1,
    seed: Annotated[int, typer.Option()] = 17,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Train the domain-balanced, listwise v5 LOCO policy."""

    graph_values = [Path(value.strip()) for value in graph_paths.split(",") if value.strip()]
    query_values = [Path(value.strip()) for value in query_paths.split(",") if value.strip()]
    if len(graph_values) != len(query_values):
        raise typer.BadParameter("graph and query path counts must match")
    probes = tuple(int(value.strip()) for value in probe_choices.split(",") if value.strip())
    if not probes or any(value < 0 for value in probes):
        raise typer.BadParameter("probe choices must be non-negative integers")
    report = train_inductive_policy_v5_multi(
        [GraphStore(path) for path in graph_values],
        query_values,
        output,
        hidden_dim=hidden_dim,
        query_dim=query_dim,
        batch_queries=batch_queries,
        visible_neighbors=visible_neighbors,
        max_queries_per_source=max_queries_per_source,
        probe_choices=probes,
        epochs=epochs,
        learning_rate=learning_rate,
        temperature=temperature,
        domain_variance_weight=domain_variance_weight,
        seed=seed,
        device_name=device,
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("benchmark")
def benchmark_command(
    graph_path: Annotated[Path, typer.Argument()],
    query_path: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option()] = Path("outputs/benchmark.parquet"),
    split: Annotated[str, typer.Option()] = "test",
    query_type: Annotated[str | None, typer.Option(help="Evaluate one query type")] = None,
    ranking: Annotated[str, typer.Option()] = "weight",
    checkpoint: Annotated[Path | None, typer.Option()] = None,
    fusion_alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    visible_neighbors: Annotated[int, typer.Option(min=1)] = 32,
    budget: Annotated[int, typer.Option(min=4)] = 64,
    answer_quota: Annotated[
        int | None,
        typer.Option(help="Find-K quota; omit with --exact"),
    ] = 1,
    exact: Annotated[bool, typer.Option(help="Require complete enumeration")] = False,
    candidates_per_subgoal: Annotated[int, typer.Option(min=1)] = 4,
    subgoal_probes: Annotated[int, typer.Option(min=0)] = 0,
    gate_min_margin: Annotated[float, typer.Option(min=0.0)] = 0.05,
    gate_max_entropy: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.95,
    gate_max_ood: Annotated[float, typer.Option(min=0.0)] = 3.0,
    adaptive_probing: Annotated[bool, typer.Option()] = False,
    max_subgoal_probes: Annotated[int, typer.Option(min=0)] = 4,
    probe_margin_threshold: Annotated[float, typer.Option(min=0.0)] = 0.05,
    probe_entropy_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.95,
    enable_backtracking: Annotated[bool, typer.Option()] = True,
    enable_budget_mask: Annotated[bool, typer.Option()] = True,
    use_native_semantics: Annotated[bool, typer.Option()] = True,
    enforce_proof: Annotated[bool, typer.Option()] = True,
    policy_local_observation: Annotated[bool, typer.Option()] = False,
    offset: Annotated[int, typer.Option(min=0)] = 0,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int, typer.Option()] = 17,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    summary = benchmark_active(
        GraphStore(graph_path),
        query_path,
        output,
        split=split,
        query_type=query_type,
        ranking=ranking,
        checkpoint=checkpoint,
        fusion_alpha=fusion_alpha,
        visible_neighbors=visible_neighbors,
        budget=budget,
        answer_quota=None if exact else answer_quota,
        candidates_per_subgoal=candidates_per_subgoal,
        subgoal_probes=subgoal_probes,
        offset=offset,
        gate_min_margin=gate_min_margin,
        gate_max_entropy=gate_max_entropy,
        gate_max_ood=gate_max_ood,
        adaptive_probing=adaptive_probing,
        max_subgoal_probes=max_subgoal_probes,
        probe_margin_threshold=probe_margin_threshold,
        probe_entropy_threshold=probe_entropy_threshold,
        enable_backtracking=enable_backtracking,
        enable_budget_mask=enable_budget_mask,
        use_native_semantics=use_native_semantics,
        enforce_proof=enforce_proof,
        policy_local_observation=policy_local_observation,
        limit=limit,
        seed=seed,
        device_name=device,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("evaluate-static")
def evaluate_static_command(
    graph_path: Annotated[Path, typer.Argument()],
    query_path: Annotated[Path, typer.Argument()],
    checkpoint: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option()] = Path("outputs/static-cqa.json"),
    split: Annotated[str, typer.Option()] = "test",
    query_type: Annotated[str | None, typer.Option(help="Evaluate one query type")] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    summary = evaluate_static(
        GraphStore(graph_path),
        query_path,
        checkpoint,
        output,
        split=split,
        limit=limit,
        query_type=query_type,
        device_name=device,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("compare-active")
def compare_active_command(
    candidate: Annotated[Path, typer.Argument(help="Candidate episodic Parquet")],
    reference: Annotated[Path, typer.Argument(help="Reference episodic Parquet")],
    output: Annotated[Path, typer.Option()] = Path("outputs/paired-comparison.json"),
    bootstrap_samples: Annotated[int, typer.Option(min=1)] = 10_000,
    seed: Annotated[int, typer.Option()] = 17,
) -> None:
    """Compute paired bootstrap intervals and an exact McNemar test."""

    report = compare_active_runs(
        candidate,
        reference,
        output,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
