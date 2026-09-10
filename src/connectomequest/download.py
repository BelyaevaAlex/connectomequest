"""Resumable, checksum-recorded downloads with explicit size gates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from connectomequest.config import dataset_config
from connectomequest.manifest import Manifest

GCS_OBJECTS_API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"


def resumable_download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    chunk_size: int = 8 << 20,
    show_progress: bool = True,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            partial.unlink()
            offset = 0
        total = expected_size
        if total is None and response.headers.get("Content-Length"):
            total = offset + int(response.headers["Content-Length"])
        mode = "ab" if offset else "wb"
        with (
            partial.open(mode) as stream,
            tqdm(
                total=total,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=destination.name,
                disable=not show_progress,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    stream.write(chunk)
                    progress.update(len(chunk))
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise OSError(
            f"size mismatch for {destination.name}: {partial.stat().st_size} != {expected_size}"
        )
    partial.replace(destination)
    return destination


def list_gcs_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params: dict[str, str | int] = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        response = requests.get(
            GCS_OBJECTS_API.format(bucket=bucket),
            params=params,
            timeout=(30, 300),
        )
        response.raise_for_status()
        payload = response.json()
        items.extend(payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            return items


def download_manc(raw_dir: Path) -> Manifest:
    config = dataset_config("manc")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest("manc", config["version"], config["license"])
    for spec in config["files"].values():
        path = resumable_download(spec["url"], raw_dir / spec["filename"])
        manifest.add_file(path, raw_dir, spec["url"])
    manifest.write(raw_dir / "manifest.json")
    return manifest


def inventory_h01(raw_dir: Path) -> tuple[list[dict[str, Any]], int]:
    config = dataset_config("h01")
    prefix = config["gcs_prefix"].removeprefix("gs://")
    bucket, object_prefix = prefix.split("/", 1)
    items = list_gcs_objects(bucket, object_prefix)
    compact = [
        {
            "name": item["name"],
            "size": int(item.get("size", 0)),
            "md5Hash": item.get("md5Hash"),
            "updated": item.get("updated"),
        }
        for item in items
        if not item["name"].endswith("/")
    ]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "inventory.json").write_text(json.dumps(compact, indent=2) + "\n")
    return compact, sum(item["size"] for item in compact)


def h01_avro_objects(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the binary Avro objects and exclude parallel JSON exports."""

    selected = [item for item in items if not item["name"].endswith(".json")]
    if not selected:
        raise ValueError("H01 inventory contains no Avro export objects")
    return sorted(selected, key=lambda item: item["name"])


def download_h01(raw_dir: Path, *, max_bytes: int = 200 << 30) -> Manifest:
    config = dataset_config("h01")
    raw_dir = Path(raw_dir)
    inventory, _ = inventory_h01(raw_dir)
    items = h01_avro_objects(inventory)
    total = sum(item["size"] for item in items)
    if total > max_bytes:
        raise RuntimeError(
            f"H01 synaptic export is {total / 2**30:.1f} GiB, above the "
            f"{max_bytes / 2**30:.1f} GiB safety gate. "
            "Increase --max-gib after reviewing inventory.json."
        )
    manifest = Manifest(
        "h01",
        config["version"],
        config["license"],
        metadata={"bytes": total},
    )
    for item in items:
        relative = item["name"].split("/exported/", 1)[-1]
        url = f"https://storage.googleapis.com/h01-release/{item['name']}"
        path = resumable_download(url, raw_dir / relative, expected_size=item["size"])
        manifest.add_file(path, raw_dir, url)
    manifest.write(raw_dir / "manifest.json")
    return manifest


def read_neuprint_token() -> str:
    value = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not value:
        raise RuntimeError("NEUPRINT_TOKEN is not set")
    candidate = Path(value)
    return candidate.read_text().strip() if candidate.is_file() else value
