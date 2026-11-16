"""One-shot gate for every held-out (``test``) computation of round 5.

``claim(script, workstream, key, output)`` must be called by a runner or analysis script before it
reads a held-out answer or writes a held-out result, after its imports. It refuses, before creating
anything, unless

* ``outputs/round5/record.json`` exists and maps every file it needs to exactly one sha256 under
  ``files`` (relative paths): the calling script, every module loaded from the repository's own code
  (round-5 code, the frozen engine and propagation code, imported scripts), the workstream's expectations
  files (``record["expectations"][workstream]``, one path or a list) and its data files (``record["data"][workstream]``);
* no module is loaded from outside the repository and the Python installation, the working directory is the
  repository and ``PYTHONPATH`` is the recorded one (``record["launch"]``);
* the thread-count variables equal ``record["launch"]["threads_env"]``;
* the interpreter is the repository's ``.venv``; the Python version, the vendored NTAC tree and every installed
  package the run has imported (from ``.venv``, ``.v32-deps`` or the base environment: its ``dist-info`` name, so its
  version, and the sha256 of its ``RECORD``) equal ``record["environment"]``. Packages the run does not import may
  change (the base environment is shared with other projects);
* ``key`` is listed in ``record["allowed"][workstream]``, and the command line equals the one recorded
  for it in ``record["argv"][workstream][key]`` (the number of worker processes excepted);
* every computation the record lists in ``record["after"][workstream][key]`` has finished, and its output still has
  the sha256 its claim stored;
* every existing claim was made under this same record;
* no claim exists yet for ``(script, key)``: the claim file ``outputs/round5/heldout-claims/<script>--<key>.json``
  is created atomically (and made read-only), so each held-out computation runs once. ``key`` names what
  is computed (for example the view), and ``output`` the single place where its result goes.

While a claimed computation runs, it renews a heartbeat file every minute. A run that crashed leaves a claim
with status ``started`` (or ``failed``); it may be resumed only with ``R5_RESUME_CLAIM=1``, under the same
record, with the same output, while that output does not exist yet, the earlier process is not alive on this
host and its heartbeat is older than 15 minutes; only one resumer wins each attempt. A run killed after its
output was complete (``SIGKILL`` skips the exit handler) counts as finished for the computations ordered after
it once its heartbeat is stale; the dependent claim records this. Every repository file opened for reading by
the run or its forked workers that the record does not list, and that is older than the record (files the protocol
itself produced are newer), is logged in the claim (``unlisted_reads``). A run killed between writing its ``.npz`` and
its ``.json`` leaves half a result; ``python -m connectomequest.round5.heldout quarantine <claim>`` moves that ``.npz``
aside (renamed, never deleted, and noted in the claim) so the run can be resumed. At exit
the loaded modules are checked again (late imports) and the claim is marked ``done`` (the output, and for an
``.npz`` its ``.json`` beside it, exist), ``failed``, or ``invalid`` when a late import does not match the record.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "outputs/round5/record.json"
CLAIMS = ROOT / "outputs/round5/heldout-claims"
CACHE = ROOT / "outputs/round5/cache"
_EXCLUDE = (".venv", ".v32-deps")  # third-party packages (pip-installed, vendored NTAC): checked by version
STALE_SECONDS = 15 * 60
HEARTBEAT_SECONDS = 60
_READS = {"fd": None, "busy": False}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def canonical_argv(argv) -> list[str]:
    """The command line as recorded: the script relative to the repository, then its arguments without
    ``--workers`` (the number of processes does not change a result)."""
    script, args = relative(argv[0]), list(argv[1:])
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--workers":
            skip = True
            continue
        if a.startswith("--workers="):
            continue
        out.append(a)
    return [script] + out


def _packages(dirs) -> dict:
    """{dist-info name: sha256 of its RECORD} over the given site directories."""
    out = {}
    for d in dirs:
        for info in sorted(Path(d).glob("*.dist-info")):
            record = info / "RECORD"
            out[info.name] = sha256(record) if record.is_file() else None
    return out


def _tree(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + sha256(f).encode() + b"\n")
    return h.hexdigest()


def environment() -> dict:
    deps = ROOT / ".v32-deps"
    groups = _site_groups()
    return dict(python=platform.python_version(),
                v32_deps=_packages(groups["v32_deps"]),
                ntac_commit=(deps / "NTAC-COMMIT").read_text().strip(),
                ntac_provenance_sha256=sha256(deps / "NTAC-PROVENANCE.json"),
                ntac_tree_sha256=_tree(deps / "ntac"),
                venv_packages=_packages(groups["venv_packages"]),
                base_packages=_packages(groups["base_packages"]))  # the venv includes the base environment's packages


ENVIRONMENT_KEYS = ("python", "ntac_commit", "ntac_provenance_sha256", "ntac_tree_sha256")  # compared as a whole
PACKAGE_GROUPS = ("v32_deps", "venv_packages", "base_packages")  # compared for the imported packages only


def _site_groups() -> dict:
    """Site directories by group (resolved: .venv may be a link to a venv shared with another checkout)."""
    deps = [(ROOT / ".v32-deps").resolve()]
    venv = [p.resolve() for p in sorted((ROOT / ".venv").glob("lib/python*/site-packages"))]
    base = []
    for d in sys.path:
        r = Path(d).resolve() if d and Path(d).is_dir() else None
        if r is not None and not r.is_relative_to(ROOT) and r not in deps + venv and r not in base:
            base.append(r)
    return dict(v32_deps=deps, venv_packages=venv, base_packages=base)


def _top_levels(info: Path) -> set:
    record = info / "RECORD"
    if not record.is_file():
        return set()
    out = set()
    for line in record.read_text(errors="replace").splitlines():
        first = line.split(",", 1)[0].split("/", 1)[0]
        if first and not first.endswith((".dist-info", ".data")) and first != "..":
            out.add(first[:-3] if first.endswith(".py") else first)
    return out


def imported_packages() -> dict:
    """{group: {dist-info name: sha256 of RECORD}} of the installed packages that loaded modules come from."""
    groups = _site_groups()
    loaded = {}
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f or not os.path.isabs(f):
            continue
        p = Path(f).resolve()
        matches = [(len(d.parts), group, d) for group, dirs in groups.items() for d in dirs if p.is_relative_to(d)]
        if matches:
            _, group, d = max(matches, key=lambda m: m[0])  # the most specific directory (site-packages, not the stdlib above it)
            first = p.relative_to(d).parts[0]
            loaded.setdefault((group, d), set()).add(first[:-3] if first.endswith(".py") else first)
    out = {g: {} for g in groups}
    for (group, d), tops in loaded.items():
        for info in sorted(d.glob("*.dist-info")):
            if _top_levels(info) & tops:
                record = info / "RECORD"
                out[group][info.name] = sha256(record) if record.is_file() else None
    return out


def _package_mismatches(recorded_env: dict) -> list[str]:
    bad = []
    for group, used in imported_packages().items():
        listed = recorded_env.get(group, {})
        bad += [f"{group}: {name}" for name, sha in used.items() if listed.get(name, "missing") != sha]
    return bad


def outside_modules() -> list[str]:
    """Python modules loaded from neither the repository nor the Python installation (a shadowing path)."""
    allowed = [ROOT, Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()]
    out = set()
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f or not os.path.isabs(f):
            continue
        p = Path(f).resolve()
        if p.name == "_remote_module_non_scriptable.py" and p.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            continue  # generated by torch.distributed at import time from its own template
        if p.suffix == ".py" and p.is_file() and not any(p.is_relative_to(a) for a in allowed):
            out.add(str(p))
    return sorted(out)


def loaded_repository_files() -> list[str]:
    """Relative paths of every Python module currently loaded from the repository's own code (not from .venv or .v32-deps)."""
    out = set()
    exclude = [ROOT / x for x in _EXCLUDE]
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f or not os.path.isabs(f):  # a relative __file__ (generated modules) names no file of the repository
            continue
        p = Path(f).resolve()
        if not p.is_file() or p.suffix != ".py" or not p.is_relative_to(ROOT) or any(p.is_relative_to(x) for x in exclude):
            continue
        out.add(p.relative_to(ROOT).as_posix())
    return sorted(out)


def _mismatches(files: dict, paths) -> list[str]:
    bad = []
    for rel in paths:
        want = files.get(rel)
        if not isinstance(want, str):
            bad.append(f"{rel}: not in record (exactly one sha256 required)")
        elif sha256(ROOT / rel) != want:
            bad.append(f"{rel}: sha256 differs from record")
    return bad


def _write(path: Path, payload: dict, new: bool) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        if new:
            os.link(tmp, path)  # fails if the claim exists
        else:
            os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _claims() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(CLAIMS.glob("*.json"))} if CLAIMS.exists() else {}


def _alive(attempt: dict) -> bool:
    if attempt.get("host") != socket.gethostname():
        return False
    try:
        os.kill(int(attempt["pid"]), 0)
    except (OSError, KeyError, ValueError, TypeError):
        return False
    return True


def _audit(event, args):
    if event != "open" or not args or _READS["fd"] is None or _READS["busy"]:
        return
    path, mode = args[0], args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    if not isinstance(path, (str, os.PathLike)):
        return
    if isinstance(mode, str):
        if any(c in mode for c in "wax+"):
            return
    elif isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR):
        return
    _READS["busy"] = True
    try:
        os.write(_READS["fd"], (os.path.abspath(os.fspath(path)) + "\n").encode(errors="replace"))
    except OSError:
        pass
    finally:
        _READS["busy"] = False


def _unlisted_reads(reads_file: Path, files: dict, output: str, since: float) -> set:
    out = set()
    stem = output.rsplit(".", 1)[0]
    if not reads_file.exists():
        return out
    for p in set(reads_file.read_text(errors="replace").splitlines()):
        try:
            r = Path(p).resolve()
        except (OSError, RuntimeError):
            continue
        if not r.is_relative_to(ROOT) or r.is_relative_to(CLAIMS) or r == RECORD.resolve():
            continue
        if any(r.is_relative_to(ROOT / x) for x in _EXCLUDE) or "__pycache__" in r.parts or not r.is_file():
            continue
        if r.stat().st_mtime >= since:  # produced by the protocol after the record (results, held-out caches)
            continue
        rel = r.relative_to(ROOT).as_posix()
        if rel not in files and not rel.startswith(stem):
            out.add(rel)
    return out


def _heartbeat_age(name: str) -> float:
    beat = CLAIMS / f"{name}.heartbeat"
    return time.time() - beat.stat().st_mtime if beat.exists() else float("inf")


def _output_hashes(output: str) -> dict:
    out = ROOT / output
    pair = [out] + ([out.with_suffix(".json")] if out.suffix == ".npz" else [])
    return {relative(p): sha256(p) for p in pair if p.exists()}


def _complete(output: str) -> bool:
    out = ROOT / output
    return out.exists() and (out.suffix != ".npz" or out.with_suffix(".json").exists())


def _finished(claims: dict, workstream: str, key: str):
    """'done', or 'reconciled' for a run killed after its output was complete (stale heartbeat), else None."""
    for name, c in claims.items():
        if (c.get("workstream"), c.get("key")) != (workstream, key):
            continue
        if c.get("status") == "done":
            return "done"
        if c.get("status") in ("started", "failed") and _complete(c["output"]) and _heartbeat_age(name) > STALE_SECONDS:
            c.update(status="done", output_sha256=_output_hashes(c["output"]), reconciled=dict(
                utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                note="killed after its output was complete: marked done without its exit handler (no late-import check)"))
            _write(CLAIMS / f"{name}.json", c, new=False)  # pinned like any finished claim from now on
            return "reconciled"
    return None


def claim(script, workstream: str, key: str, output) -> Path:
    if not RECORD.exists():
        raise SystemExit("held-out refused: outputs/round5/record.json does not exist")
    record_sha = sha256(RECORD)
    record = json.loads(RECORD.read_text())
    files = record.get("files", {})
    expectations = record.get("expectations", {}).get(workstream)
    if not expectations:
        raise SystemExit(f"held-out refused: the record lists no expected outcomes for workstream {workstream!r}")
    expectations = [expectations] if isinstance(expectations, str) else list(expectations)
    allowed = record.get("allowed", {}).get(workstream)
    if allowed is None or key not in allowed:
        raise SystemExit(f"held-out refused: {key!r} is not among the held-out computations recorded for {workstream!r}")
    want_argv = record.get("argv", {}).get(workstream, {}).get(key)
    argv = canonical_argv([script] + sys.argv[1:])
    if want_argv != argv:
        raise SystemExit(f"held-out refused: the command line {argv} is not the one recorded for {workstream}:{key} ({want_argv})")
    launch = record.get("launch", {})
    if Path.cwd().resolve() != ROOT or os.environ.get("PYTHONPATH") != launch.get("PYTHONPATH"):
        raise SystemExit(f"held-out refused: run from the repository with PYTHONPATH={launch.get('PYTHONPATH')!r}")
    threads = {k: os.environ.get(k) for k in launch.get("threads_env", {})}
    if threads != launch.get("threads_env", {}):
        raise SystemExit(f"held-out refused: thread settings {threads} differ from the record {launch.get('threads_env')}")
    outside = outside_modules()
    if outside:
        raise SystemExit("held-out refused: modules loaded from outside the repository and the Python installation:\n  " + "\n  ".join(outside))
    needed = [relative(script)] + loaded_repository_files() + expectations + list(record.get("data", {}).get(workstream, []))
    bad = _mismatches(files, dict.fromkeys(needed))
    if bad:
        raise SystemExit("held-out refused:\n  " + "\n  ".join(bad))
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise SystemExit(f"held-out refused: the interpreter is not the repository's .venv (sys.prefix {sys.prefix})")
    env, recorded_env = environment(), record.get("environment", {})
    wrong = [k for k in ENVIRONMENT_KEYS if recorded_env.get(k) != env[k]] + _package_mismatches(recorded_env)
    if wrong:
        raise SystemExit(f"held-out refused: the environment differs from the record in {wrong}")
    claims = _claims()
    other = sorted(n for n, c in claims.items() if c.get("record_sha256") != record_sha)
    if other:
        raise SystemExit(f"held-out refused: claims {other} were made under another record")
    reconciled, missing = [], []
    changed = [n for n, c in claims.items() if c.get("status") == "done" and _output_hashes(c["output"]) != c.get("output_sha256")]
    if changed:
        raise SystemExit(f"held-out refused: the outputs of {changed} changed after their claims finished")
    for prerequisite in record.get("after", {}).get(workstream, {}).get(key, []):
        state = _finished(claims, *prerequisite.split(":", 1))
        if state is None:
            missing.append(prerequisite)
        elif state == "reconciled":
            reconciled.append(prerequisite)
    if missing:
        raise SystemExit(f"held-out refused: {workstream}:{key} runs after {missing}, which have not finished")

    CLAIMS.mkdir(parents=True, exist_ok=True)
    name = f"{Path(script).stem}--{key}".replace("/", "_")
    path = CLAIMS / f"{name}.json"
    output = relative(output) if Path(output).resolve().is_relative_to(ROOT) else str(output)
    attempt = dict(utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), host=socket.gethostname(), pid=os.getpid(), argv=sys.argv)
    if reconciled:
        attempt["prerequisites_finished_without_exit_handler"] = reconciled
    if path.exists():
        prior = json.loads(path.read_text())
        resumable = (prior.get("status") in ("started", "failed") and os.environ.get("R5_RESUME_CLAIM") == "1"
                     and prior.get("record_sha256") == record_sha and prior.get("output") == output
                     and not (ROOT / output).exists() and not _alive(prior["attempts"][-1])
                     and _heartbeat_age(name) > STALE_SECONDS)
        if not resumable:
            raise SystemExit(f"held-out refused: {name} was already claimed (status {prior.get('status')}, output {prior.get('output')}); "
                             "each held-out computation runs once")
        lock = CLAIMS / f".{name}.resume-{len(prior['attempts'])}"
        try:
            os.link(path, lock)  # one resumer per attempt
        except FileExistsError:
            raise SystemExit(f"held-out refused: another process is resuming {name}") from None
        prior["attempts"].append(attempt)
        prior["status"] = "started"
        _write(path, prior, new=False)
    else:
        _write(path, dict(script=relative(script), workstream=workstream, key=key, output=output, status="started",
                          record_sha256=record_sha, attempts=[attempt]), new=True)

    beat = CLAIMS / f"{name}.heartbeat"
    beat.write_text(f"{socket.gethostname()} {os.getpid()}\n")
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                os.utime(beat)
            except OSError:
                pass

    threading.Thread(target=heartbeat, daemon=True).start()
    reads_file = CLAIMS / f"{name}.reads-{len(json.loads(path.read_text())['attempts'])}"
    _READS["fd"] = os.open(reads_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    sys.addaudithook(_audit)

    def finish():
        stop.set()
        state = json.loads(path.read_text())
        late = (_mismatches(files, loaded_repository_files()) + [f"{m}: loaded from outside the repository" for m in outside_modules()]
                + [f"package {m}: not as recorded" for m in _package_mismatches(record.get("environment", {}))])
        if late:
            state["status"], state["late_mismatches"] = "invalid", late
        else:
            state["status"] = "done" if _complete(output) else "failed"
        if state["status"] == "done":
            state["output_sha256"] = _output_hashes(output)
        state["unlisted_reads"] = sorted(set(state.get("unlisted_reads", [])) | _unlisted_reads(reads_file, files, output, RECORD.stat().st_mtime))
        state["finished_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        _write(path, state, new=False)

    atexit.register(finish)
    return path


def quarantine(name: str) -> Path:
    """Move aside the ``.npz`` of a run that was killed before writing its ``.json`` (renamed, never deleted)."""
    path = CLAIMS / f"{name}.json"
    state = json.loads(path.read_text())
    out = ROOT / state["output"]
    if state.get("status") not in ("started", "failed") or out.suffix != ".npz" or not out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"{name}: not a half result (status {state.get('status')}, output {state['output']})")
    if _alive(state["attempts"][-1]) or _heartbeat_age(name) <= STALE_SECONDS:
        raise SystemExit(f"{name}: the run may still be alive")
    aside = out.with_name(f"{out.name}.orphan-{len(state['attempts'])}")
    os.rename(out, aside)
    state.setdefault("quarantined", []).append(dict(utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                                                    moved=relative(aside), sha256=sha256(aside)))
    _write(path, state, new=False)
    return aside


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "quarantine":
        print(quarantine(sys.argv[2]))
    else:
        raise SystemExit("usage: python -m connectomequest.round5.heldout quarantine <claim name>")
