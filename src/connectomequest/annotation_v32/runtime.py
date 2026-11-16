"""Content-addressed artifacts and measured compute accounting for V32."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid


def content_key(payload):
    """SHA-256 of canonical JSON; caller supplies all identity inputs."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _key(key):
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('key must be lowercase SHA-256')
    return key


class ClaimBusy(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        return self.root / _key(key)

    def claim(self, key):
        return _Claim(self, _key(key))

    def validate(self, key):
        path = self.path(key)
        try:
            manifest = json.loads((path / 'manifest.json').read_text())
            if manifest['key'] != key or not isinstance(manifest['files'], dict):
                return False
            actual = {}
            for p in path.rglob('*'):
                if p.is_symlink():
                    return False
                if p.is_file() and p != path / 'manifest.json':
                    actual[p.relative_to(path).as_posix()] = hash_file(p)
            return actual == manifest['files']
        except (OSError, ValueError, KeyError, TypeError):
            return False


class _Claim:
    def __init__(self, store, key):
        self.store, self.key = store, key
        self.lock = store.root / (key + '.lock')
        self.stage = store.root / (key + '.stage.' + uuid.uuid4().hex)
        self.completed = False

    def __enter__(self):
        try:
            self.lock.mkdir()
        except FileExistsError as e:
            raise ClaimBusy(self.key) from e
        try:
            if self.store.path(self.key).exists():
                raise FileExistsError(self.key)
            self.stage.mkdir()
        except BaseException:
            self.lock.rmdir()
            raise
        return self

    def complete(self):
        files = {}
        for p in self.stage.rglob('*'):
            if p.is_symlink():
                raise ValueError('symlinks are not supported')
            if p.is_file():
                rel = p.relative_to(self.stage).as_posix()
                if rel == 'manifest.json':
                    raise ValueError('manifest.json is reserved')
                with p.open('rb') as f:
                    os.fsync(f.fileno())
                files[rel] = hash_file(p)
        manifest = self.stage / 'manifest.json'
        with manifest.open('x') as f:
            json.dump({'key': self.key, 'files': files}, f, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.rename(self.stage, self.store.path(self.key))
        self.completed = True
        return self.store.path(self.key)

    def __exit__(self, typ, value, tb):
        if not self.completed:
            shutil.rmtree(self.stage, ignore_errors=True)
        self.lock.rmdir()


def _usage_name(run_key, attempt):
    return content_key({"run_key": run_key, "attempt": attempt})


def _publish_record(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    stage = path.with_name(path.name + ".stage." + uuid.uuid4().hex)
    try:
        with stage.open("xb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        created = True
        try:
            os.link(stage, path)
        except FileExistsError:
            created = False
            if path.read_bytes() != data:
                raise ValueError("conflicting usage record")
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return created
    finally:
        stage.unlink(missing_ok=True)


def start_usage(ledger, run_key, attempt, *, gpu_count):
    if not isinstance(gpu_count, int) or gpu_count < 0:
        raise ValueError("invalid GPU count")
    name = _usage_name(run_key, attempt)
    return _publish_record(Path(ledger) / (name + ".start.json"),
                    dict(run_key=run_key, attempt=attempt, gpu_count=gpu_count))


def record_usage(ledger, run_key, *, cpu_seconds, gpu_count, wall_seconds, attempt="0", status="completed"):
    if any(not math.isfinite(x) or x < 0 for x in (cpu_seconds, wall_seconds)) or not isinstance(gpu_count, int) or gpu_count < 0 or status not in ("completed", "error"):
        raise ValueError("invalid usage")
    start_usage(ledger, run_key, attempt, gpu_count=gpu_count)
    row = dict(run_key=run_key, attempt=attempt, cpu_seconds=cpu_seconds, gpu_count=gpu_count, wall_seconds=wall_seconds, gpu_seconds=gpu_count * wall_seconds, status=status)
    _publish_record(Path(ledger) / (_usage_name(run_key, attempt) + ".done.json"), row)


def aggregate_usage(ledger):
    root = Path(ledger)
    starts, done = {}, {}
    for path in root.glob("*.start.json"):
        row = json.loads(path.read_text()); starts[(row["run_key"], row["attempt"])] = row
    for path in root.glob("*.done.json"):
        row = json.loads(path.read_text()); done[(row["run_key"], row["attempt"])] = row
    return dict(cpu_seconds=sum(x["cpu_seconds"] for x in done.values()), gpu_seconds=sum(x["gpu_seconds"] for x in done.values()), running=sorted(starts.keys() - done.keys()))


def execute(ledger, run_key, gpu_count, command, attempt="0"):
    """Run one command and record child CPU and allocated GPU wall time."""
    import resource
    import subprocess
    if not start_usage(ledger, run_key, attempt, gpu_count=gpu_count):
        raise FileExistsError("attempt already started")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    result = None
    try:
        result = subprocess.run(command, check=False).returncode
    finally:
        wall = time.monotonic() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
        record_usage(ledger, run_key, cpu_seconds=cpu, gpu_count=gpu_count, wall_seconds=wall, attempt=attempt, status="completed" if result == 0 else "error")
    return result


def _main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--run-key', required=True)
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required')
    raise SystemExit(execute(args.ledger, args.run_key, args.gpus, command, args.attempt))


if __name__ == '__main__':
    _main()
