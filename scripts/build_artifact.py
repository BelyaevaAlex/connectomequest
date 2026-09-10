#!/usr/bin/env python3
"""Build the anonymous executable artifact shipped with the paper."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/artifact.zip"
BINARY_PRIVATE = re.compile(rb"eyJhbGci|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY", re.I)
TEXT_PRIVATE = re.compile(
    rb"/(?:home|workspace-[^/]+)/[^/\s]+|"
    rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    re.I,
)
TEXT_SUFFIXES = {".cff", ".json", ".md", ".pl", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"}


def contains_private_material(name: str, data: bytes) -> bool:
    if BINARY_PRIVATE.search(data):
        return True
    return Path(name).suffix.lower() in TEXT_SUFFIXES and TEXT_PRIVATE.search(data) is not None


STAMP = (2026, 9, 10, 0, 0, 0)
RESULT_FOLDERS = (
    "outputs/nemo/strong-accept/amortized-revalidation-v40/confirmation",
    "outputs/nemo/strong-accept/v38-anchor-cluster-audit",
    "outputs/nemo/strong-accept/proof-accounting-v2",
    "outputs/nemo/strong-accept/proof-structural-mutation-core-v1",
    "outputs/nemo/strong-accept/fresh-contract-test-v1",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def publication_files() -> set[Path]:
    paths: set[Path] = set()
    for folder in ("src/connectomequest", "scripts", "tests"):
        paths.update((ROOT / folder).rglob("*.py"))
    paths.discard(ROOT / "scripts/build_artifact.py")
    paths.discard(ROOT / "scripts/build_submission.py")
    paths.update((ROOT / "configs").rglob("*.yaml"))
    for folder in RESULT_FOLDERS:
        paths.update(path for path in (ROOT / folder).rglob("*") if path.is_file())
    for dataset in ("h01-20210729-c3", "hemibrain-v1.2.1", "manc-v1.0"):
        directory = ROOT / "data/processed" / dataset
        paths.update(
            directory / name for name in ("edges.parquet", "nodes.parquet", "manifest.json")
        )

    replay_root = ROOT / "outputs/nemo/strong-accept/full-episode-v39-attempt-07/pilot"
    paths.update(path for path in replay_root.iterdir() if path.is_file())
    methods = ("full_replan", "full_scan", "receipt_index", "unchecked_reuse")
    conditions = (
        "none",
        "irrelevant_receipt_withdrawal",
        "alternative_support",
        "final_support_loss",
        "semantic_correction",
        "sparse_burst",
    )
    representatives: dict[tuple[str, str], Path] = {}
    for path in sorted((replay_root / "cases").glob("*.json")):
        for method in methods:
            for condition in conditions:
                if path.name.endswith(f"-{condition}-{method}.json"):
                    representatives.setdefault((method, condition), path)
    if len(representatives) != 24:
        raise AssertionError(
            f"expected 24 representative RGB replays, found {len(representatives)}"
        )
    paths.update(representatives.values())

    for relative in (
        "outputs/nemo/strong-accept/dependent-plan-v1/development-attempt-02/perceptor.pt",
        "outputs/nemo/strong-accept/online-repair-v1/development-attempt-02/perception_report.json",
        "outputs/nemo/strong-accept/online-repair-v1/development-attempt-02/protocol.json",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        "docs/architecture.md",
        "docs/data.md",
        "docs/protocol.md",
        "docs/reproduction.md",
    ):
        paths.add(ROOT / relative)
    missing = sorted(str(path.relative_to(ROOT)) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(missing)
    return paths


ARTIFACT_README = """# ConnectomeQuest evaluation artifact

This archive contains current executable source, compact processed graph
snapshots, frozen aggregates, complete timing-confirmation records, and 24
representative RGB episode replays. Historical result paths are retained only
as immutable provenance. Exact source bytes are available for the timing confirmation;
the RGB replay records are integrity-checked but do not claim an original source snapshot.
The archive contains no manuscript or author identity.

Run `python verify_artifact.py` to verify every member hash. After installing
`requirements-tested.txt`, run `python verify_artifact.py --tests` for focused
contract and revalidation tests.
"""

VERIFIER = r"""#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for name, expected in manifest["files"].items():
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"hash mismatch: {name}")
parser = argparse.ArgumentParser()
parser.add_argument("--tests", action="store_true")
args = parser.parse_args()
if args.tests:
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/strong_accept/test_receipt_index.py",
         "tests/strong_accept/test_revalidation.py",
         "tests/strong_accept/test_revalidation_statistics.py",
         "tests/strong_accept/test_core_contract.py"],
        check=True,
    )
print(f"verified {len(manifest['files'])} files")
"""

REQUIREMENTS = """fastavro>=1.9
numpy>=1.26
pyarrow>=16
pyyaml>=6
requests>=2.31
scipy>=1.12
torch>=2.2
tqdm>=4.66
typer>=0.12
minigrid>=3.0,<4.0
matplotlib>=3.8
pytest>=8
"""


def sealed_source_provenance() -> dict[str, bytes]:
    """Carry forward the exact source bytes preserved for the timing confirmation."""

    expected: dict[str, str] = {}
    protocol_paths = (
        ROOT / "outputs/nemo/strong-accept/amortized-revalidation-v40/confirmation/protocol.json",
    )
    for path in protocol_paths:
        payload = json.loads(path.read_text())
        expected.update(payload.get("source_hashes", {}))
    graph_protocol = json.loads(
        (ROOT / "outputs/nemo/strong-accept/fresh-contract-test-v1/PROTOCOL.json").read_text()
    )
    expected.update(
        {name: digest for name, digest in graph_protocol["inputs"].items() if name.endswith(".py")}
    )
    attribution_protocol = json.loads(
        (ROOT / "outputs/nemo/strong-accept/fresh-attribution-v35/PROTOCOL.json").read_text()
    )
    expected.update(attribution_protocol["source_sha256"])
    if not OUTPUT.is_file():
        raise FileNotFoundError("existing artifact.zip is required for sealed source migration")

    carried: dict[str, bytes] = {}
    index: dict[str, str] = {}
    with zipfile.ZipFile(OUTPUT) as archive:
        names = set(archive.namelist())
        previous_index = (
            json.loads(archive.read("provenance/SOURCE_INDEX.json"))
            if "provenance/SOURCE_INDEX.json" in names
            else {}
        )
        for historical_path, digest in sorted(expected.items()):
            repository_copy = ROOT / "provenance/source" / f"{digest}.txt"
            candidates = (
                historical_path,
                f"provenance/source/{previous_index.get(historical_path, digest)}.txt",
            )
            archived_name = next((name for name in candidates if name in names), None)
            if repository_copy.is_file():
                data = repository_copy.read_bytes()
            elif archived_name is not None:
                data = archive.read(archived_name)
            else:
                raise FileNotFoundError(f"sealed source absent from provenance: {historical_path}")
            if sha256(data) != digest:
                raise AssertionError(f"sealed source digest mismatch: {historical_path}")
            carried[f"provenance/source/{digest}.txt"] = data
            index[historical_path] = digest
    carried["provenance/SOURCE_INDEX.json"] = (
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    ).encode()
    return carried


def build() -> dict[str, object]:
    additions = {
        str(path.relative_to(ROOT)): path.read_bytes() for path in sorted(publication_files())
    }
    additions.update(sealed_source_provenance())
    additions.update(
        {
            "ARTIFACT_README.md": ARTIFACT_README.encode(),
            "requirements-tested.txt": REQUIREMENTS.encode(),
            "verify_artifact.py": VERIFIER.encode(),
        }
    )
    with tempfile.NamedTemporaryFile(
        prefix="connectomequest-artifact-", suffix=".zip", dir=OUTPUT.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    hashes: dict[str, str] = {}
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(additions.items()):
                if contains_private_material(name, data):
                    raise RuntimeError(f"private material in {name}")
                archive.writestr(archive_info(name), data)
                hashes[name] = sha256(data)
            manifest = {
                "artifact": "ConnectomeQuest anonymous evaluation artifact",
                "scope": "Executable evaluation code, compact graph snapshots, frozen aggregates, timing records, and representative RGB replays; no manuscript and no claim of complete retraining or all raw traces.",
                "files": dict(sorted(hashes.items())),
            }
            archive.writestr(
                archive_info("MANIFEST.json"),
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            )
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise AssertionError("duplicate archive names")
            if any(name.endswith((".tex", ".pdf")) for name in names):
                raise AssertionError("artifact must not duplicate the manuscript")
        temporary.replace(OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(OUTPUT.relative_to(ROOT)),
        "files": len(hashes),
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
