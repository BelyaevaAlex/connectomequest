"""Fail-closed NEmo v2 confirmatory protocol.

The public helpers in this module enforce the order

``freeze development selection -> seal artifacts -> generate queries -> open test``.

No function reads confirmatory Parquet rows before both the protocol seal and
the one-time test-open marker exist.  Development code must not import this
module to inspect confirmatory data.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import yaml

from connectomequest.graph import GraphStore
from connectomequest.manifest import sha256_file
from connectomequest.query import Query, QueryType
from connectomequest.query_io import validate_query_snapshot

DATASETS = {
    "h01": "h01-20210729-c3",
    "manc": "manc-v1.0",
    "hemibrain": "hemibrain-v1.2.1",
}
SEEDS = (17, 29, 43)
BUDGETS = (16, 32, 64)
CONFIRMATORY_SEED = 2_718_281
TOTAL_QUERIES = 50_000
TEST_QUERIES = 5_000
PROTOCOL_VERSION = 1


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def confirmatory_root(repo: Path) -> Path:
    return repo.resolve() / "outputs/nemo/v2/confirmatory"


def selection_path(repo: Path) -> Path:
    return confirmatory_root(repo) / "FROZEN-METHOD-SELECTION.json"


def seal_path(repo: Path) -> Path:
    return confirmatory_root(repo) / "FROZEN-PROTOCOL.json"


def test_open_path(repo: Path) -> Path:
    return confirmatory_root(repo) / "TEST-OPENED.json"


def query_path(repo: Path, held_out: str) -> Path:
    bundle = DATASETS[held_out]
    return (
        confirmatory_root(repo)
        / "queries"
        / bundle
        / f"queries-nemo-v2-confirmatory-seed{CONFIRMATORY_SEED}.parquet"
    )


def query_paths(repo: Path) -> dict[str, Path]:
    return {held_out: query_path(repo, held_out) for held_out in DATASETS}


def load_protocol_config(repo: Path) -> dict[str, Any]:
    path = repo.resolve() / "configs/experiments/graph_evaluation.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config.get("v2_confirmatory", {})
    expected = {
        "generator_seed": CONFIRMATORY_SEED,
        "total_queries_per_connectome": TOTAL_QUERIES,
        "expected_test_queries_per_connectome": TEST_QUERIES,
        "open_test_once": True,
        "no_hyperparameter_changes_after_open": True,
    }
    wrong = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if wrong:
        raise ValueError(f"nemo_v2 confirmatory config mismatch: {wrong}")
    return config


def method_matrix(*, include_random: bool = True, include_minerva: bool = True) -> list[dict]:
    """Return the exact, deterministic same-query active matrix."""

    cells: list[dict[str, Any]] = []
    for held_out in DATASETS:
        for budget in BUDGETS:
            cells.append(
                {
                    "held_out": held_out,
                    "method": "weight",
                    "budget": budget,
                    "seed": 17,
                    "controller": "fixed_one_probe",
                }
            )
            for method in ("snapshot_v2", "gate_v2"):
                for seed in SEEDS:
                    cells.append(
                        {
                            "held_out": held_out,
                            "method": method,
                            "budget": budget,
                            "seed": seed,
                            "controller": "adaptive_zero_to_four_probes",
                        }
                    )
        for method in ("degree", "oracle"):
            cells.append(
                {
                    "held_out": held_out,
                    "method": method,
                    "budget": 64,
                    "seed": 17,
                    "controller": "fixed_one_probe",
                    "access_regime": "privileged_ceiling"
                    if method == "oracle"
                    else "observable_paid_lookahead",
                }
            )
        for method, enabled in (("random", include_random), ("minerva", include_minerva)):
            if enabled:
                for seed in SEEDS:
                    cells.append(
                        {
                            "held_out": held_out,
                            "method": method,
                            "budget": 64,
                            "seed": seed,
                            "controller": "fixed_one_probe",
                        }
                    )
    return cells


def build_method_selection(
    repo: Path,
    *,
    include_random: bool = True,
    include_minerva: bool = True,
) -> dict[str, Any]:
    """Freeze the final matrix from development-only evidence."""

    repo = repo.resolve()
    load_protocol_config(repo)
    if any(
        path.exists() or path.with_suffix(".manifest.json").exists()
        for path in query_paths(repo).values()
    ):
        raise RuntimeError("cannot freeze methods after confirmatory query generation")
    if test_open_path(repo).exists():
        raise RuntimeError("cannot freeze methods after confirmatory test opening")
    aggregate_path = repo / "outputs/nemo/v2/validation/aggregate-development.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("development_only") is not True:
        raise ValueError("selection input is not marked development-only")
    if aggregate.get("confirmatory_or_test_access") is not False:
        raise ValueError("selection input does not explicitly deny confirmatory/test access")
    cells = method_matrix(
        include_random=include_random,
        include_minerva=include_minerva,
    )
    payload: dict[str, Any] = {
        "campaign": "nemo-2026-v2-confirmatory",
        "status": "frozen_before_confirmatory_query_generation",
        "protocol_version": PROTOCOL_VERSION,
        "selection_source": str(aggregate_path.relative_to(repo)),
        "selection_source_sha256": sha256_file(aggregate_path),
        "selection_source_input_fingerprint_sha256": aggregate.get("input_fingerprint_sha256"),
        "selection_partition": "replication-validation",
        "confirmatory_or_test_access": False,
        "primary_metric": "success_at_budget_64",
        "methods": ["weight", "degree", "snapshot_v2", "gate_v2", "oracle"]
        + (["random"] if include_random else [])
        + (["minerva"] if include_minerva else []),
        "same_query_audit_only_at_b64": ["degree", "oracle"]
        + [
            method
            for method, enabled in (("random", include_random), ("minerva", include_minerva))
            if enabled
        ],
        "matrix": cells,
        "matrix_sha256": canonical_sha(cells),
        "fixed_controller": {
            "visible_neighbors": 32,
            "answer_quota": 1,
            "candidates_per_subgoal": 4,
            "fixed_one_probe": {"subgoal_probes": 1, "adaptive_probing": False},
            "adaptive_zero_to_four_probes": {
                "subgoal_probes": 0,
                "adaptive_probing": True,
                "max_subgoal_probes": 4,
                "probe_margin_threshold": 0.05,
                "probe_entropy_threshold": 0.95,
            },
        },
        "queries": {
            "generator_seed": CONFIRMATORY_SEED,
            "total_per_connectome": TOTAL_QUERIES,
            "test_per_connectome": TEST_QUERIES,
            "query_type": "2pt",
        },
    }
    payload["selection_sha256"] = canonical_sha(payload)
    return payload


def write_method_selection(repo: Path, payload: dict[str, Any], *, dry_run: bool = False) -> Path:
    path = selection_path(repo)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError(f"frozen method selection differs: {path}")
        return path
    if dry_run:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def load_method_selection(repo: Path) -> dict[str, Any]:
    path = selection_path(repo)
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("selection_sha256", None)
    actual = canonical_sha(payload)
    payload["selection_sha256"] = digest
    if digest != actual:
        raise ValueError("frozen method selection digest is invalid")
    cells = payload.get("matrix")
    if payload.get("matrix_sha256") != canonical_sha(cells):
        raise ValueError("frozen method matrix digest is invalid")
    if payload.get("confirmatory_or_test_access") is not False:
        raise ValueError("method selection was not test-blind")
    source = repo.resolve() / payload["selection_source"]
    if sha256_file(source) != payload.get("selection_source_sha256"):
        raise ValueError("development selection source drifted")
    return payload


def _checkpoint_paths(repo: Path) -> list[Path]:
    paths: list[Path] = []
    for held_out, bundle in DATASETS.items():
        for seed in SEEDS:
            paths.extend(
                [
                    repo / "outputs/nemo/v2/checkpoints" / f"heldout-{held_out}-seed{seed}.pt",
                    repo / "outputs/nemo/v2/checkpoints" / f"heldout-{held_out}-seed{seed}.json",
                    repo / "outputs/nemo/v2/gates" / f"heldout-{held_out}-seed{seed}.pt",
                    repo / "outputs/nemo/v2/gates" / f"heldout-{held_out}-seed{seed}.json",
                    repo / "outputs/nemo/minerva/production" / f"leave-{bundle}-seed{seed}.pt",
                    repo / "outputs/nemo/diverse" / f"{held_out}-mr-seed{seed}.pt",
                ]
            )
    return paths


def _validate_gate_dependencies(repo: Path) -> None:
    """Ensure every gate is LOCO/test-blind and pins its exact experts."""

    for held_out, bundle in DATASETS.items():
        for seed in SEEDS:
            snapshot = repo / "outputs/nemo/v2/checkpoints" / f"heldout-{held_out}-seed{seed}.pt"
            gate = repo / "outputs/nemo/v2/gates" / f"heldout-{held_out}-seed{seed}.pt"
            minerva = repo / "outputs/nemo/minerva/production" / f"leave-{bundle}-seed{seed}.pt"
            diverse = repo / "outputs/nemo/diverse" / f"{held_out}-mr-seed{seed}.pt"
            snapshot_payload = torch.load(snapshot, map_location="cpu", weights_only=True)
            protocol = snapshot_payload.get("protocol", {})
            if (
                protocol.get("held_out") != held_out
                or int(snapshot_payload.get("seed", -1)) != seed
            ):
                raise ValueError(f"SnapshotV2 direction/seed mismatch: {snapshot}")
            if protocol.get("target_connectome_validation") is not False:
                raise ValueError(f"SnapshotV2 is not strict LOCO: {snapshot}")
            if protocol.get("immutable_test_access") is not False:
                raise ValueError(f"SnapshotV2 is not test-blind: {snapshot}")
            gate_payload = torch.load(gate, map_location="cpu", weights_only=True)
            gate_protocol = gate_payload.get("protocol", {})
            if (
                gate_payload.get("held_out") != held_out
                or int(gate_payload.get("seed", -1)) != seed
            ):
                raise ValueError(f"GateV2 direction/seed mismatch: {gate}")
            if not (
                gate_protocol.get("source_only") is True
                and gate_protocol.get("target_connectome_validation") is False
                and gate_protocol.get("test_access") is False
            ):
                raise ValueError(f"GateV2 is not source-only/test-blind: {gate}")
            names = tuple(gate_payload.get("expert_names", ()))
            expected_names = ("weight", "random", "snapshot_v2", "minerva", "diverse")
            if names != expected_names:
                raise ValueError(f"unexpected frozen gate experts in {gate}: {names}")
            records = gate_payload.get("expert_checkpoint_sha256", {})
            expected = {
                "snapshot_v2": sha256_file(snapshot),
                "minerva": sha256_file(minerva),
                "diverse": sha256_file(diverse),
            }
            for name, digest in expected.items():
                record = records.get(name, {})
                if record.get("sha256", record.get("checkpoint_sha256")) != digest:
                    raise ValueError(f"{name} dependency hash mismatch in {gate}")
            if records.get("weight", {}).get("observable_builtin") != "weight":
                raise ValueError(f"weight dependency is not pinned in {gate}")
            random_record = records.get("random", {})
            if not (
                random_record.get("observable_builtin") == "query_seeded_random"
                and int(random_record.get("seed", -1)) == seed
            ):
                raise ValueError(f"random dependency is not pinned in {gate}")


def tracked_paths(repo: Path) -> list[Path]:
    """All source, selected artifacts and graph identities frozen by the seal."""

    repo = repo.resolve()
    paths = [repo / "configs/experiments/graph_evaluation.yaml", selection_path(repo)]
    paths.extend(sorted((repo / "src/connectomequest").glob("**/*.py")))
    paths.extend(sorted((repo / "scripts").glob("*nemo_v2*.py")))
    paths.extend(sorted((repo / "scripts").glob("*nemo_v2*.sh")))
    paths.append(repo / "outputs/nemo/v2/validation/aggregate-development.json")
    paths.extend(_checkpoint_paths(repo))
    paths.extend(repo / "data/processed" / bundle / "manifest.json" for bundle in DATASETS.values())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing v2 seal inputs:\n" + "\n".join(map(str, missing)))
    _validate_gate_dependencies(repo)
    return sorted(set(paths))


def artifact_hashes(repo: Path) -> dict[str, str]:
    repo = repo.resolve()
    return {str(path.relative_to(repo)): sha256_file(path) for path in tracked_paths(repo)}


def verify_artifact_hashes(expected: dict[str, str], actual: dict[str, str]) -> None:
    """Fail closed on missing, added, or modified sealed inputs."""

    if expected == actual:
        return
    changed = sorted(
        name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name)
    )
    raise RuntimeError("v2 confirmatory seal mismatch: " + ", ".join(changed))


def assert_preseal_absence(repo: Path) -> None:
    forbidden = [test_open_path(repo), confirmatory_root(repo) / "active"]
    for path in query_paths(repo).values():
        forbidden.extend([path, path.with_suffix(".manifest.json")])
    present = [path for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(
            "confirmatory queries/test/results exist before protocol seal:\n"
            + "\n".join(map(str, present))
        )


def build_seal(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    config = load_protocol_config(repo)
    selection = load_method_selection(repo)
    assert_preseal_absence(repo)
    artifacts = artifact_hashes(repo)
    payload = {
        "campaign": "nemo-2026-v2-confirmatory",
        "status": "sealed_before_confirmatory_query_generation",
        "protocol_version": PROTOCOL_VERSION,
        "confirmatory_seed": CONFIRMATORY_SEED,
        "total_queries_per_connectome": TOTAL_QUERIES,
        "expected_test_queries_per_connectome": TEST_QUERIES,
        "query_paths_absent_at_seal": [
            str(path.relative_to(repo)) for path in query_paths(repo).values()
        ],
        "method_selection_sha256": selection["selection_sha256"],
        "method_matrix_sha256": selection["matrix_sha256"],
        "config_campaign": config["campaign"],
        "artifacts": artifacts,
        "artifacts_sha256": canonical_sha(artifacts),
    }
    return payload


def write_seal(repo: Path, payload: dict[str, Any], *, dry_run: bool = False) -> Path:
    path = seal_path(repo)
    if path.exists():
        raise FileExistsError(f"seal already exists: {path}; use --verify")
    if dry_run:
        return path
    payload = {**payload, "sealed_at": datetime.now(timezone.utc).isoformat()}
    payload["seal_sha256"] = canonical_sha(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _validate_generated_query(repo: Path, held_out: str, path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"partial or missing confirmatory query snapshot: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "seed": CONFIRMATORY_SEED,
        "num_queries": TOTAL_QUERIES,
        "query_type_counts": {"2pt": TOTAL_QUERIES},
        "split_counts": {
            "train": TOTAL_QUERIES * 8 // 10,
            "validation": TOTAL_QUERIES // 10,
            "test": TEST_QUERIES,
        },
        "graph_manifest_sha256": sha256_file(
            repo / "data/processed" / DATASETS[held_out] / "manifest.json"
        ),
        "query_file_sha256": sha256_file(path),
    }
    wrong = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if wrong:
        raise ValueError(f"confirmatory query manifest mismatch for {held_out}: {wrong}")
    return manifest


def verify_seal(repo: Path, *, require_queries: bool = False) -> dict[str, Any]:
    """Reject any source/checkpoint/selection drift after sealing."""

    repo = repo.resolve()
    path = seal_path(repo)
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_seal_sha = payload.pop("seal_sha256", None)
    actual_seal_sha = canonical_sha(payload)
    payload["seal_sha256"] = recorded_seal_sha
    if recorded_seal_sha != actual_seal_sha:
        raise ValueError("v2 confirmatory seal digest is invalid")
    if payload.get("artifacts_sha256") != canonical_sha(payload.get("artifacts")):
        raise ValueError("v2 confirmatory artifact-map digest is invalid")
    actual = artifact_hashes(repo)
    expected = payload.get("artifacts", {})
    verify_artifact_hashes(expected, actual)
    selection = load_method_selection(repo)
    if selection["selection_sha256"] != payload.get("method_selection_sha256"):
        raise RuntimeError("frozen method selection drifted")
    present = []
    for held_out, path in query_paths(repo).items():
        if path.exists() or path.with_suffix(".manifest.json").exists():
            _validate_generated_query(repo, held_out, path)
            present.append(held_out)
    if require_queries and set(present) != set(DATASETS):
        raise FileNotFoundError(
            f"all confirmatory query snapshots are required; present={sorted(present)}"
        )
    return payload


def authorize_test_open(
    repo: Path,
    *,
    dry_run: bool = False,
    verified_seal: dict[str, Any] | None = None,
) -> Path:
    """Create the immutable one-time marker before any test Parquet row is read."""

    repo = repo.resolve()
    seal = verified_seal or verify_seal(repo, require_queries=True)
    marker = test_open_path(repo)
    payload = {
        "status": "confirmatory_test_open_authorized_once",
        "protocol_version": PROTOCOL_VERSION,
        "seal_sha256": seal["seal_sha256"],
        "query_sha256": {
            held_out: sha256_file(path) for held_out, path in query_paths(repo).items()
        },
    }
    payload["authorization_sha256"] = canonical_sha(payload)
    if marker.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError("test-open marker differs from current sealed artifacts")
        return marker
    if dry_run:
        return marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError("concurrent test-open marker mismatch") from None
    return marker


def read_confirmatory_test_queries(
    repo: Path,
    held_out: str,
    graph: GraphStore,
    *,
    verified_seal: dict[str, Any] | None = None,
) -> list[Query]:
    """The only supported reader for v2 confirmatory test rows."""

    repo = repo.resolve()
    seal = verified_seal or verify_seal(repo, require_queries=True)
    marker_path = test_open_path(repo)
    if not marker_path.is_file():
        raise PermissionError("confirmatory test is sealed but not authorized for one-time opening")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_digest = marker.pop("authorization_sha256", None)
    actual_marker_digest = canonical_sha(marker)
    marker["authorization_sha256"] = marker_digest
    if marker_digest != actual_marker_digest:
        raise ValueError("test-open marker digest is invalid")
    if marker.get("seal_sha256") != seal.get("seal_sha256"):
        raise ValueError("test-open marker does not pin the verified seal")
    path = query_path(repo, held_out)
    if marker.get("query_sha256", {}).get(held_out) != sha256_file(path):
        raise ValueError("test-open marker does not pin this query snapshot")
    validate_query_snapshot(path, graph)
    table = pq.read_table(
        path,
        columns=[
            "query_id",
            "query_type",
            "anchors",
            "answers",
            "split",
            "semantic_relation",
        ],
        filters=[("split", "=", "test"), ("query_type", "=", "2pt")],
        memory_map=True,
    )
    rows = table.to_pylist()
    if len(rows) != TEST_QUERIES:
        raise ValueError(f"expected exactly {TEST_QUERIES} confirmatory test rows")
    if len({row["query_id"] for row in rows}) != TEST_QUERIES:
        raise ValueError("duplicate query ids in confirmatory test partition")
    if any(row["split"] != "test" or row["query_type"] != "2pt" for row in rows):
        raise AssertionError("Arrow filter admitted a non-test/non-2pt row")
    return [
        Query(
            query_id=row["query_id"],
            query_type=QueryType(row["query_type"]),
            anchors=tuple(row["anchors"]),
            answers=frozenset(row["answers"]),
            split="test",
            semantic_relation=row.get("semantic_relation"),
        )
        for row in rows
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_matrix(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate uniqueness and closed vocabulary before scheduling any work."""

    cells = selection.get("matrix")
    if not isinstance(cells, list) or not cells:
        raise ValueError("frozen method matrix is empty")
    seen = set()
    allowed_methods = {"weight", "degree", "snapshot_v2", "gate_v2", "random", "minerva", "oracle"}
    for cell in cells:
        key = (cell.get("held_out"), cell.get("method"), cell.get("budget"), cell.get("seed"))
        if key in seen:
            raise ValueError(f"duplicate confirmatory matrix cell: {key}")
        seen.add(key)
        if cell.get("held_out") not in DATASETS or cell.get("method") not in allowed_methods:
            raise ValueError(f"unsupported confirmatory matrix cell: {cell}")
        if cell.get("budget") not in BUDGETS or cell.get("seed") not in SEEDS:
            raise ValueError(f"invalid budget/seed in matrix cell: {cell}")
        expected_controller = (
            "adaptive_zero_to_four_probes"
            if cell["method"] in {"snapshot_v2", "gate_v2"}
            else "fixed_one_probe"
        )
        if cell.get("controller") != expected_controller:
            raise ValueError(f"controller mismatch in matrix cell: {cell}")
    return cells


def files_sha256(paths: Iterable[Path], repo: Path) -> dict[str, str]:
    repo = repo.resolve()
    return {str(path.resolve().relative_to(repo)): sha256_file(path) for path in paths}
