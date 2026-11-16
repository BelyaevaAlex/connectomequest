"""The one-shot held-out gate: record, expectations, listed key, command line, order, environment, one claim each."""

import json
import os
import socket
import stat
import sys

import pytest

from connectomequest.round5 import heldout

ENV = dict(python="3.x", ntac_commit="c", ntac_provenance_sha256="0" * 64, ntac_tree_sha256="1" * 64)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(heldout, "ROOT", tmp_path)
    monkeypatch.setattr(heldout, "RECORD", tmp_path / "record.json")
    monkeypatch.setattr(heldout, "CLAIMS", tmp_path / "claims")
    monkeypatch.setattr(heldout, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(heldout, "loaded_repository_files", lambda: [])
    monkeypatch.setattr(heldout, "environment", lambda: dict(ENV))
    monkeypatch.setattr(heldout.sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(heldout, "outside_modules", lambda: [])
    monkeypatch.setattr(heldout, "_package_mismatches", lambda env: [])
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(heldout.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "src:.v32-deps:scripts")
    for k in THREADS:
        monkeypatch.setenv(k, "1")
    exits = []
    monkeypatch.setattr(heldout.atexit, "register", exits.append)
    script = tmp_path / "run.py"
    script.write_text("print(1)\n")
    expect = tmp_path / "expect.json"
    expect.write_text("[]\n")
    data = tmp_path / "data.npz"
    data.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", [str(script), "--split", "test", "--workers", "6"])

    def write_record(**overrides):
        argv = ["run.py", "--split", "test"]
        record = dict(files={p.name: heldout.sha256(p) for p in (script, expect, data)},
                      expectations={"ws": "expect.json"}, data={"ws": ["data.npz"]}, allowed={"ws": ["a", "b"]},
                      argv={"ws": {"a": argv, "b": argv}}, environment=dict(ENV), launch={"PYTHONPATH": "src:.v32-deps:scripts", "threads_env": dict.fromkeys(THREADS, "1")})
        record.update(overrides)
        heldout.RECORD.write_text(json.dumps(record))

    return dict(root=tmp_path, script=script, data=data, exits=exits, write_record=write_record)


def test_refuses_without_record(repo):
    with pytest.raises(SystemExit, match="does not exist"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


def test_refuses_without_expectations(repo):
    repo["write_record"](expectations={})
    with pytest.raises(SystemExit, match="no expected outcomes"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


def test_refuses_unlisted_key(repo):
    repo["write_record"]()
    with pytest.raises(SystemExit, match="not among"):
        heldout.claim(repo["script"], "ws", "c", repo["root"] / "out.json")


def test_refuses_other_command_line(repo, monkeypatch):
    repo["write_record"]()
    monkeypatch.setattr(sys, "argv", [str(repo["script"]), "--split", "test", "--max-cells", "10"])
    with pytest.raises(SystemExit, match="command line"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")
    assert not heldout.CLAIMS.exists()  # a refused claim creates nothing


def test_workers_do_not_count():
    assert heldout.canonical_argv([str(heldout.ROOT / "scripts/x.py"), "--a", "1", "--workers", "16", "--workers=3", "--b"]) == \
        ["scripts/x.py", "--a", "1", "--b"]


def test_refuses_changed_data(repo):
    repo["write_record"]()
    repo["data"].write_bytes(b"y")
    with pytest.raises(SystemExit, match="data.npz: sha256 differs"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


def test_refuses_other_environment(repo):
    repo["write_record"](environment=dict(ENV, ntac_commit="other"))
    with pytest.raises(SystemExit, match="environment differs"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


def test_order_is_enforced(repo):
    repo["write_record"](after={"ws": {"b": ["ws:a"]}})
    with pytest.raises(SystemExit, match="runs after"):
        heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")
    out = repo["root"] / "out.json"
    heldout.claim(repo["script"], "ws", "a", out)
    out.write_text("{}")
    repo["exits"].pop()()
    heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")


def test_each_key_runs_once(repo, monkeypatch):
    repo["write_record"]()
    out = repo["root"] / "out.json"
    path = heldout.claim(repo["script"], "ws", "a", out)
    assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)  # the claim is read-only
    out.write_text("{}")
    repo["exits"].pop()()
    assert json.loads(path.read_text())["status"] == "done"
    with pytest.raises(SystemExit, match="already claimed"):
        heldout.claim(repo["script"], "ws", "a", out)
    monkeypatch.setenv("R5_RESUME_CLAIM", "1")
    with pytest.raises(SystemExit, match="already claimed"):  # a finished claim is never resumed
        heldout.claim(repo["script"], "ws", "a", out)
    heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")  # another listed key is its own claim


def test_npz_needs_its_json(repo):
    repo["write_record"]()
    out = repo["root"] / "out.npz"
    path = heldout.claim(repo["script"], "ws", "a", out)
    out.write_bytes(b"npz")  # the run died before writing out.json
    repo["exits"].pop()()
    assert json.loads(path.read_text())["status"] == "failed"


def _crash_elsewhere(path, stale=True):
    """Make the last attempt of a claim look like a process on another machine, with an old heartbeat."""
    state = json.loads(path.read_text())
    state["attempts"][-1]["host"] = "elsewhere"
    os.chmod(path, 0o644)
    path.write_text(json.dumps(state))
    if stale:
        beat = path.with_suffix(".heartbeat")
        os.utime(beat, (0, 0))


def test_crashed_claim_resumes_only_when_asked(repo, monkeypatch):
    repo["write_record"]()
    out = repo["root"] / "out.json"
    path = heldout.claim(repo["script"], "ws", "a", out)  # crashed: no finish, no output
    _crash_elsewhere(path)
    with pytest.raises(SystemExit, match="already claimed"):
        heldout.claim(repo["script"], "ws", "a", out)
    monkeypatch.setenv("R5_RESUME_CLAIM", "1")
    with pytest.raises(SystemExit, match="already claimed"):  # the resumed run must write to the same place
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "elsewhere.json")
    heldout.claim(repo["script"], "ws", "a", out)
    assert len(json.loads(path.read_text())["attempts"]) == 2


def test_live_claim_is_not_resumed(repo, monkeypatch):
    repo["write_record"]()
    out = repo["root"] / "out.json"
    heldout.claim(repo["script"], "ws", "a", out)  # this process is alive on this host
    monkeypatch.setenv("R5_RESUME_CLAIM", "1")
    with pytest.raises(SystemExit, match="already claimed"):
        heldout.claim(repo["script"], "ws", "a", out)


def test_resume_needs_the_same_record(repo, monkeypatch):
    repo["write_record"]()
    out = repo["root"] / "out.json"
    path = heldout.claim(repo["script"], "ws", "a", out)
    _crash_elsewhere(path)
    repo["write_record"](notes="changed")
    monkeypatch.setenv("R5_RESUME_CLAIM", "1")
    with pytest.raises(SystemExit, match="another record"):
        heldout.claim(repo["script"], "ws", "a", out)


def test_claims_bind_to_one_record(repo):
    repo["write_record"]()
    heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")
    repo["write_record"](notes="replaced")  # the record changed after a claim
    with pytest.raises(SystemExit, match="another record"):
        heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")


def test_killed_run_with_complete_output_counts_as_finished(repo):
    repo["write_record"](after={"ws": {"b": ["ws:a"]}})
    out = repo["root"] / "out.json"
    path = heldout.claim(repo["script"], "ws", "a", out)
    out.write_text("{}")  # complete, then SIGKILL: no exit handler, the claim stays "started"
    with pytest.raises(SystemExit, match="runs after"):  # still beating: not finished yet
        heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")
    os.utime(path.with_suffix(".heartbeat"), (0, 0))
    other = heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")
    assert json.loads(other.read_text())["attempts"][-1]["prerequisites_finished_without_exit_handler"] == ["ws:a"]


def test_outside_modules_are_found(tmp_path, monkeypatch):
    shadow = tmp_path / "shadow_mod_r5.py"
    shadow.write_text("x = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import shadow_mod_r5  # noqa: F401
    assert str(shadow.resolve()) in heldout.outside_modules()
    del sys.modules["shadow_mod_r5"]


def test_wrong_launch_is_refused(repo, monkeypatch):
    repo["write_record"]()
    monkeypatch.setenv("PYTHONPATH", "/elsewhere:src:.v32-deps:scripts")
    with pytest.raises(SystemExit, match="PYTHONPATH"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


def test_thread_settings_are_checked(repo, monkeypatch):
    repo["write_record"]()
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(SystemExit, match="thread settings"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


def test_changed_output_of_a_finished_claim_is_refused(repo):
    repo["write_record"](after={"ws": {"b": ["ws:a"]}})
    out = repo["root"] / "out.json"
    heldout.claim(repo["script"], "ws", "a", out)
    out.write_text("{}")
    repo["exits"].pop()()
    out.write_text('{"edited": 1}')  # the result changed after its claim finished
    with pytest.raises(SystemExit, match="changed after their claims finished"):
        heldout.claim(repo["script"], "ws", "b", repo["root"] / "other.json")


def test_half_result_is_quarantined_then_resumed(repo, monkeypatch):
    repo["write_record"]()
    out = repo["root"] / "out.npz"
    path = heldout.claim(repo["script"], "ws", "a", out)
    out.write_bytes(b"npz")  # killed before out.json
    with pytest.raises(SystemExit, match="may still be alive"):
        heldout.quarantine(path.stem)
    _crash_elsewhere(path)
    aside = heldout.quarantine(path.stem)
    assert aside.exists() and not out.exists()
    assert json.loads(path.read_text())["quarantined"][0]["moved"] == "out.npz.orphan-1"
    monkeypatch.setenv("R5_RESUME_CLAIM", "1")
    heldout.claim(repo["script"], "ws", "a", out)  # the run can now be resumed
    assert len(json.loads(path.read_text())["attempts"]) == 2


def test_reads_older_than_the_record_are_logged(repo, tmp_path):
    repo["write_record"]()
    old = tmp_path / "old.csv"
    old.write_text("x")
    os.utime(old, (0, 0))
    new = tmp_path / "new.npz"
    new.write_text("y")
    reads = tmp_path / "reads"
    reads.write_text(f"{old}\n{new}\n{tmp_path / 'data.npz'}\n")
    files = json.loads(heldout.RECORD.read_text())["files"]
    since = heldout.RECORD.stat().st_mtime
    assert heldout._unlisted_reads(reads, files, "out.json", since) == {"old.csv"}


def test_other_interpreter_is_refused(repo, monkeypatch):
    repo["write_record"]()
    monkeypatch.setattr(heldout.sys, "prefix", "/usr")
    with pytest.raises(SystemExit, match="interpreter"):
        heldout.claim(repo["script"], "ws", "a", repo["root"] / "out.json")


@pytest.mark.skipif(not (heldout.ROOT / ".v32-deps/NTAC-COMMIT").exists(), reason="needs the pinned dependency directory")
def test_imported_packages_are_pinned():
    import scipy  # noqa: F401  (a package of the base environment)
    env = heldout.environment()
    assert not heldout._package_mismatches(env)
    name = next(n for n in heldout.imported_packages()["base_packages"] if n.startswith("scipy-"))
    env["base_packages"] = {**env["base_packages"], name: "0" * 64}  # as if scipy had been reinstalled
    assert any("scipy" in m for m in heldout._package_mismatches(env))
    env["base_packages"] = {k: v for k, v in env["base_packages"].items() if k != name}
    assert any("scipy" in m for m in heldout._package_mismatches(env))
