#!/usr/bin/env python3
"""Round 5, workstream ``policies``: proposed record fragment for the held-out run.

Writes ``outputs/round5/policies/record-proposal.json`` in the layout of the round-5 one-shot gate
(``connectomequest.round5.heldout``):

* ``files``: sha256 of every repository file a held-out command loads or names - the scripts, every
  module loaded from ``src/`` and ``scripts/`` (found by tracing each runner on a few development
  families with an audit hook, ``.venv`` and ``.v32-deps`` excluded as in the gate), the expectations
  file and the data files;
* ``expectations``: ``outputs/round5/policies/expectations.json`` (``r5_policies_expectations.py``);
* ``data``: the input files the held-out commands read (traced on development and mapped to their
  held-out counterparts, plus every prior file a command names);
* ``allowed``: the held-out computations (one claim key each), and the exact commands;

plus the development results it was derived from and the frozen choices. The coordinator merges it
into ``outputs/round5/record.json``. Nothing is run on the held-out split here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectomequest.round5 import policies_data as pdata  # noqa: E402

FILES = [
    "scripts/r5_policies_run.py",
    "scripts/r5_policies_dp.py",
    "scripts/r5_policies_wor.py",
    "scripts/r5_policies_analyze.py",
    "scripts/r5_policies_report.py",
    "scripts/r5_policies_expectations.py",
    "src/connectomequest/round5/__init__.py",
    "src/connectomequest/round5/policies_data.py",
    "src/connectomequest/round5/policies_baselines.py",
    "src/connectomequest/round5/policies_dp.py",
    "src/connectomequest/round5/policies_interleave.py",
    "src/connectomequest/round5/policies_voi.py",
    "src/connectomequest/round5/policies_wor.py",
    "tests/test_round5_policies.py",
]


TRACER = r"""
import json, os, runpy, sys
log, script, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
seen, busy = set(), [False]

def hook(event, args):
    if event != "open" or busy[0] or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.path.realpath(os.fsdecode(args[0]))
    if path in seen:
        return
    seen.add(path)
    busy[0] = True
    try:
        with open(f"{log}.open.{os.getpid()}", "a") as f:
            f.write(path + "\n")
    finally:
        busy[0] = False

sys.addaudithook(hook)
sys.argv = [script] + argv
try:
    runpy.run_path(script, run_name="__main__")
finally:
    from connectomequest.round5 import heldout
    with open(f"{log}.modules", "w") as f:
        json.dump(heldout.loaded_repository_files(), f)
"""


def traces(tmp):
    """Development smoke runs of every held-out command type: (script, arguments)."""
    out = Path(tmp)
    dev = ["--split", "development"]
    return [
        ("scripts/r5_policies_run.py", ["--view", "malecns", "--prior", "mlp_s0", "--limit", "3", "--workers", "2", "--output", f"{out}/runs1", *dev]),
        ("scripts/r5_policies_run.py", ["--view", "flywire", "--prior", "provisional", "--limit", "3", "--workers", "2", "--output", f"{out}/runs2", *dev]),
        ("scripts/r5_policies_run.py", ["--view", "flywire", "--prior", "seeded30", "--limit", "3", "--workers", "2", "--output", f"{out}/runs3", *dev]),
        ("scripts/r5_policies_run.py", ["--view", "malecns", "--prior", "const_half", "--limit", "3", "--workers", "2", "--output", f"{out}/runs4", *dev]),
        ("scripts/r5_policies_dp.py", ["--view", "malecns", "--prior", "seeded30", "--limit", "12", "--workers", "2", "--output", f"{out}/dp1", *dev]),
        ("scripts/r5_policies_dp.py", ["--view", "flywire", "--prior", "mlp_s0", "--limit", "12", "--workers", "2", "--output", f"{out}/dp2", *dev]),
        ("scripts/r5_policies_wor.py", ["--view", "malecns", "--adversarial", "--repeats", "2", "--limit", "5", "--workers", "2", "--output", f"{out}/wor1", *dev]),
        ("scripts/r5_policies_wor.py", ["--view", "flywire", "--adversarial", "--repeats", "2", "--limit", "5", "--workers", "2", "--output", f"{out}/wor2", *dev]),
        ("scripts/r5_policies_analyze.py", ["--output", f"{out}/analysis.json", *dev]),
        ("scripts/r5_policies_report.py", ["--output", f"{out}/summary.md", *dev]),
    ]


def trace(tmp):
    """Repository modules loaded and files opened by each traced command."""
    env = dict(os.environ, PYTHONPATH="src:.v32-deps:scripts", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    modules, opened = set(), set()
    for i, (script, argv) in enumerate(traces(tmp)):
        log = f"{tmp}/trace{i}"
        subprocess.run([sys.executable, "-c", TRACER, log, script, *argv], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        modules.update(json.loads(Path(f"{log}.modules").read_text()))
        for part in Path(tmp).glob(f"trace{i}.open.*"):
            opened.update(part.read_text().split())
    return sorted(modules), sorted(opened)


def data_files(opened):
    """Inputs of the held-out run: traced development inputs mapped to their held-out counterparts,
    split-independent inputs as they are, and every prior file a command names."""
    root = ROOT.resolve()
    skip = (root / ".venv", root / ".v32-deps", root / ".git", (pdata.OUT).resolve(), (root / "outputs/round5/heldout-claims"))
    out = set()
    for name in opened:
        p = Path(name)
        if (not p.is_relative_to(root) or p.suffix in (".py", ".pyc", ".nbi", ".nbc") or "__pycache__" in p.parts
                or any(p.is_relative_to(s) for s in skip) or not p.is_file()):
            continue
        if p == pdata.RECORD.resolve() or p.is_relative_to(root / "outputs/round5/banc"):
            continue  # the view banc has no held-out command here
        if "development" in p.name:
            p = p.with_name(p.name.replace("development", "test"))
            if not p.exists():
                raise SystemExit(f"no held-out counterpart of {name}")
        out.add(p.relative_to(root).as_posix())
    for view in ("malecns", "flywire"):
        out.add(pdata.benchmark_path(view).resolve().relative_to(root).as_posix())
        for prior in ("seeded10", "seeded10conn", "seeded30", "seeded30conn", "seeded50", "seeded50conn"):
            out.add((pdata.PRIORS / f"{prior}-{view}.npz").resolve().relative_to(root).as_posix())
    return sorted(out)


def keys(commands):
    """Claim key of every held-out command (the same strings the scripts pass to ``heldout.claim``)."""
    out = []
    for c in commands["core"] + commands["seeded"] + commands["final"]:
        a = c.split()
        opt = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
        script = Path(a[1]).stem
        if script == "r5_policies_run":
            out.append(f"{opt['--view']}-{opt['--prior']}")
        elif script == "r5_policies_dp":
            out.append(f"dp-{opt['--view']}-{opt['--prior']}")
        elif script == "r5_policies_wor":
            out.append(f"wor-{opt['--view']}")
        else:
            out.append({"r5_policies_analyze": "analyze", "r5_policies_report": "report"}[script])
    assert len(out) == len(set(out))
    return out


def main():
    out = pdata.fresh(pdata.OUT / "record-proposal.json")
    analysis = pdata.OUT / "analysis-development.json"
    expectations_path = pdata.OUT / "expectations.json"
    expectations = json.loads(expectations_path.read_text())
    dev = json.loads(analysis.read_text())
    with tempfile.TemporaryDirectory(dir=pdata.OUT / "jobs") as tmp:
        modules, opened = trace(tmp)
    data = data_files(opened)
    rel = lambda p: Path(p).resolve().relative_to(ROOT.resolve()).as_posix()  # noqa: E731
    code = sorted(set(FILES) | set(modules) | {"src/connectomequest/round5/heldout.py"})
    files = {name: pdata.hash_file(ROOT / name) for name in code + [rel(expectations_path)] + data}
    inputs = {}
    for sub in ("runs", "dp", "wor"):
        for path in sorted((pdata.OUT / sub).glob("*-development*.npz")):
            inputs[rel(path)] = pdata.hash_file(path)
    payload = dict(
        schema="round5-policies-record-proposal-2",
        workstream="policies",
        files=files,
        expectations={"policies": rel(expectations_path)},
        data={"policies": data},
        allowed={"policies": keys(expectations["commands"])},
        commands=expectations["commands"],
        traced_modules=modules,
        development_inputs=inputs,
        development_analysis={rel(analysis): pdata.hash_file(analysis)},
        frozen_choices=dict(voi=dev.get("voi_selection"), dp_max_cells=14, propagation=dict(lam=8.0, gamma=3.0, tau=0.3), alpha=0.05,
                            wor_repeats=5, adversarial_repeats=10, interleave="shadow run of the base, base first, weight order second"),
    )
    pdata.save_json(out, payload)
    print(out, len(files), "files,", len(data), "data files,", len(payload["allowed"]["policies"]), "keys")


if __name__ == "__main__":
    main()
