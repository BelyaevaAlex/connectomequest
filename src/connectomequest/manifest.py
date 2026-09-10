"""Immutable provenance manifests for downloaded and processed artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class Artifact:
    path: str
    size_bytes: int
    sha256: str
    source_url: str | None = None


@dataclass(slots=True)
class Manifest:
    dataset: str
    version: str
    license: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = "1"
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_file(self, path: Path, root: Path, source_url: str | None = None) -> None:
        self.artifacts.append(
            Artifact(
                path=str(path.relative_to(root)),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source_url=source_url,
            )
        )

    def write(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> Manifest:
        payload = json.loads(path.read_text())
        payload["artifacts"] = [Artifact(**item) for item in payload.get("artifacts", [])]
        return cls(**payload)
