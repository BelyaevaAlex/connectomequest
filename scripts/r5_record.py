"""Write ``outputs/round5/record.json``: the design of every round-5 held-out computation, fixed before any runs.

The record lists, with sha256,
* the repository's own code (every ``.py`` under ``src/`` and ``scripts/``; third-party packages in
  ``.venv`` and ``.v32-deps`` are listed by version);
* each workstream's data files and its expected outcomes (point estimates unless an interval is named);
* the held-out computations each workstream may run (``allowed``, one claim each, see
  ``connectomequest.round5.heldout``) and the commands that run them;
* the scope decision taken before any held-out run (``outputs/round5/scope.json``).

``--check`` prints what would be written; the record itself is written once and never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import sys
import tarfile
import time
from pathlib import Path

from connectomequest.round5 import heldout

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "outputs/round5/record.json"
PY = "PYTHONPATH=src:.v32-deps:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python"

V35 = ["outputs/v35/benchmark/malecns.npz", "outputs/v35/benchmark/flywire.npz", "outputs/v35/benchmark/manifest.json"]
V35_RUNS = [f"outputs/v35/runs/{v}-test.{x}" for v in ("malecns", "flywire") for x in ("npz", "json")]
V35_SWEEP = [f"outputs/v35/sweep/{v}-test.{x}" for v in ("malecns", "flywire") for x in ("npz", "json")]
BANC = ["outputs/round5/banc/benchmark.npz", "outputs/round5/banc/manifest.json", "outputs/round5/banc/view.npz",
        "outputs/round5/banc/view-names.json", "outputs/round5/banc/features.npz",
        "inputs/round5/banc/prepared-v888/public/topology.npz"]
MLP = "outputs/v32/predictors/59cd92072106eb705ce5ae288aa891a79ee47de74bbba3cc5dce30ae4fc38ea3"
SEEDED = [f"outputs/round5/priors/seeded{f}{c}-{v}.{x}" for f in ("10", "30", "50") for c in ("", "conn")
          for v in ("malecns", "flywire") for x in ("npz", "json")]
NEUROSYM = [f"outputs/neurosym/cache/{v}{x}" for v in ("malecns", "flywire") for x in (".npz", "-names.json")]
B4 = [f"outputs/neurosym/B/heldout/{v}-test.{x}" for v in ("malecns", "flywire") for x in ("npz", "json")]
POSTHOC = [f"outputs/v35/posthoc/{v}-test.{x}" for v in ("malecns", "flywire") for x in ("npz", "json")]
HEMIBRAIN = ["inputs/v32/prepared-common-v1/hemibrain/public/manifest.json"]  # the type vocabulary every view maps to
PRIOR_CACHES = [f"outputs/round5/cache/priors-{c}-{v}.npz" for c in ("modalities", "rawknn200") for v in ("malecns", "flywire")]
LAUNCH = dict(PYTHONPATH="src:.v32-deps:scripts",
              threads_env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1"))

# Files whose sha256 changed after a workstream proposed its record entries: the version (sha256) that was reviewed
# and the reason; build() refuses any other change, and any other version of these files.
ACKNOWLEDGED_CHANGES = {
    "scripts/r5_priors_run.py": ("197a99fd6919eec1391ed2b256a45d58aa841821f4d94cbd614d680ca508c0a2",
        "the coordinator inserted the shared held-out claim (heldout.claim) after the proposal"),
    "scripts/r5_priors_corrsweep.py": ("755943a0e03ed385d615aa16dd95b3ca355a769c1da4922af0f6f439d9077f2a",
        "the coordinator inserted the shared held-out claim (heldout.claim) after the proposal"),
    "scripts/r5_priors_analyze.py": ("db461677354b42485a381ec367792c784fe1389fcd3a56dcb43701bff254dd71",
        "the coordinator inserted the shared held-out claim (heldout.claim) after the proposal"),
    "src/connectomequest/round5/heldout.py": ("c2639f5c3b2738e80a878651bd0d83b5cd04d9cb1f9a8ce00d0480fbe7ba15d2",
        "the shared gate was hardened after three pre-run reviews (command line, order, environment, launch, heartbeat, quarantine)"),
    "tests/test_round5_policies.py": ("02d73eaa916c06dcd9ffddc5486c501783e97a5225cc8cd759539d314ff0f042",
        "the coordinator adapted the gate test to the hardened gate (recorded command line and environment)"),
}


def policies_key(argv):
    opt = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
    script = Path(argv[0]).stem
    if script == "r5_policies_run":
        return [f"{opt['--view']}-{opt['--prior']}"]
    if script == "r5_policies_dp":
        return [f"dp-{opt['--view']}-{opt['--prior']}"]
    if script == "r5_policies_wor":
        return [f"wor-{opt['--view']}"]
    return [{"r5_policies_analyze": "analyze", "r5_policies_report": "report"}[script]]


def priors_key(argv):
    opt = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
    script = Path(argv[0]).stem
    if script == "r5_priors_run":
        return [f"{opt['--view']}-{Path(opt.get('--prior-dir', 'outputs/round5/priors')).name}"]
    if script == "r5_priors_corrsweep":
        return [f"corrsweep-{opt['--view']}"]
    return ["analyze"]


def argv_of(command: str) -> list[str]:
    tokens = shlex.split(command)
    i = next(k for k, a in enumerate(tokens) if a.startswith("scripts/") and a.endswith(".py"))
    return heldout.canonical_argv([str(ROOT / tokens[i])] + tokens[i + 1:])


def workers_of(command: str):
    """The --workers value of a recorded command (None: the script takes no such argument)."""
    tokens = shlex.split(command)
    return int(tokens[tokens.index("--workers") + 1]) if "--workers" in tokens else None


def runnable(command: str) -> str:
    """A recorded command as it must be typed: from the repository, with the recorded environment."""
    tokens = shlex.split(command)
    i = next(k for k, a in enumerate(tokens) if a.startswith("scripts/") and a.endswith(".py"))
    env = " ".join(f"{k}={v}" for k, v in [("PYTHONPATH", LAUNCH["PYTHONPATH"]), *LAUNCH["threads_env"].items()])
    return f"{env} .venv/bin/python " + " ".join(tokens[i:])


def snapshot(paths) -> Path:
    """A copy of the recorded code and small text files beside the record (the tree is not committed)."""
    out = ROOT / "outputs/round5/record-code.tar.gz"
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} exists")
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in sorted(paths):
            if rel.endswith((".py", ".sh", ".json", ".md", ".csv", ".tsv")) and (ROOT / rel).stat().st_size < 20_000_000:
                tar.add(ROOT / rel, arcname=rel)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        os.link(tmp, out)
    finally:
        tmp.unlink()
    return out


def launch_self_check() -> None:
    """The record pins the environment it is built in, so build it in the environment the runs use."""
    problems = []
    if Path.cwd().resolve() != ROOT:
        problems.append(f"working directory {Path.cwd()} is not the repository")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        problems.append(f"interpreter prefix {sys.prefix} is not the repository's .venv")
    if os.environ.get("PYTHONPATH") != LAUNCH["PYTHONPATH"]:
        problems.append(f"PYTHONPATH {os.environ.get('PYTHONPATH')!r} is not {LAUNCH['PYTHONPATH']!r}")
    for k, v in LAUNCH["threads_env"].items():
        if os.environ.get(k) != v:
            problems.append(f"{k}={os.environ.get(k)!r}, not {v!r}")
    problems += [f"module from outside the repository and the Python installation: {m}" for m in heldout.outside_modules()]
    if problems:
        raise SystemExit("build the record in the launch environment:\n  " + "\n  ".join(problems))


def workstreams() -> dict:
    priors_proposal = json.loads((ROOT / "outputs/round5/priors/record-proposal.json").read_text())
    policies_proposal = json.loads((ROOT / "outputs/round5/policies/record-proposal.json").read_text())
    pc = policies_proposal["commands"]
    out = dict(
        costs=dict(
            expectations=["outputs/round5/costs/expectations-analysis.json", "outputs/round5/costs/expectations-bio-scoped.json"],
            data=V35 + V35_RUNS + NEUROSYM + HEMIBRAIN + ["outputs/round5/scope.json", "outputs/neurosym/cache/hemibrain.npz", "inputs/v32/flywire/cq-flywire-v3.1.0.tsv",
                                                         "outputs/v35/analysis/lenient-test.json", "inputs/v32/ontology/Supplemental_file5_hemibrain_meta.csv",
                                   "data/raw/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather"],
            runs=[(["malecns"], f"{PY} scripts/r5_costs_run.py --view malecns --split test"),
                  (["flywire"], f"{PY} scripts/r5_costs_run.py --view flywire --split test"),
                  (["analyze"], f"{PY} scripts/r5_costs_analyze.py --split test --expectations outputs/round5/costs/expectations-analysis.json"),
                  (["bio"], f"{PY} scripts/r5_costs_bio.py --split test --expectations outputs/round5/costs/expectations-bio-scoped.json")],
            after={"analyze": ["costs:malecns", "costs:flywire"]},
            note="expectations-bio-scoped.json is expectations-bio.json without B4 (sexual dimorphism) and B5 (hemibrain "
                 "truncation), which outputs/round5/scope.json excluded from the paper before any held-out run; on the "
                 "held-out split r5_costs_bio.py does not compute them."),
        banc=dict(
            expectations="outputs/round5/banc/expectations.json",
            data=V35 + V35_RUNS + BANC + [f"{MLP}/model.npz", f"{MLP}/report.json", "inputs/v32/prepared-common-v1/hemibrain/public/manifest.json"],
            runs=[(["run"], f"{PY} scripts/r5_banc_run.py --split test --workers 6"),
                  (["report"], f"{PY} scripts/r5_banc_report.py --split test")],
            after={"report": ["banc:run"]}),
        prop=dict(
            expectations="outputs/round5/prop/expectations.json",
            data=V35 + V35_RUNS + BANC + NEUROSYM + B4 + POSTHOC + SEEDED + HEMIBRAIN + [f"{MLP}/model.npz", f"{MLP}/report.json"],
            runs=[([f"{v}-main"], f"{PY} scripts/r5_prop_run.py run --split test --view {v} --name main --workers 6") for v in ("flywire", "malecns")]
                 + [([f"{v}-seeded"], f"{PY} scripts/r5_prop_run.py run --split test --view {v} --prior external --name seeded --workers 6")
                    for v in ("flywire", "malecns")] + [
                  (["banc-main"], f"{PY} scripts/r5_prop_run.py run --split test --view banc --name main --workers 6"),
                  (["analyze"], f"{PY} scripts/r5_prop_run.py analyze --split test --view flywire malecns banc")],
            after={"banc-main": ["banc:run"],
                   "flywire-seeded": ["prop:flywire-main"], "malecns-seeded": ["prop:malecns-main"],  # seeded reuses the arms of main
                   "analyze": ["prop:flywire-main", "prop:malecns-main", "prop:flywire-seeded", "prop:malecns-seeded", "prop:banc-main"]},
            note="BANC is analysed by the propagation workstream after the banc workstream's held-out run (they share the BANC view)."),
        priors=dict(
            expectations="outputs/round5/priors/record-proposal.json",
            data=V35 + V35_RUNS + V35_SWEEP + NEUROSYM + HEMIBRAIN + PRIOR_CACHES + sorted(priors_proposal["files"]) + [p for p in SEEDED if p.endswith(".json")],
            runs=[(priors_key(argv_of(c)), c) for c in priors_proposal["commands"]],
            note="expected outcomes E1-E13 are in record-proposal.json under expected_outcomes"),
        policies=dict(
            expectations=policies_proposal["expectations"]["policies"],
            data=policies_proposal["data"]["policies"],
            runs=[(policies_key(argv_of(c)), c) for c in pc["core"] + pc["seeded"] + pc["final"]],
            environment_commands=pc["environment"],
            scoped_files=sorted(policies_proposal["files"]),
            frozen_choices=policies_proposal["frozen_choices"],
            note="the policies runners also check record['policies']['files'] (their own guard); "
                 "DP is exact for questions with at most 14 input cells only"),
    )
    runs = [k for keys, _ in out["priors"]["runs"] for k in keys]
    out["priors"]["after"] = {"analyze": [f"priors:{k}" for k in runs if k != "analyze"]}
    keys = [k for ks, _ in out["policies"]["runs"] for k in ks]
    out["policies"]["after"] = {"analyze": [f"policies:{k}" for k in keys if k not in ("analyze", "report")], "report": ["policies:analyze"]}
    for name, w in out.items():
        w["allowed"] = [k for ks, _ in w["runs"] for k in ks]
        w["argv"] = {k: argv_of(c) for ks, c in w["runs"] for k in ks}
        w["workers"] = {k: workers_of(c) for ks, c in w["runs"] for k in ks}
        w["commands"] = [runnable(ws_cmd) for _, ws_cmd in w["runs"]]
        if len(set(w["allowed"])) != len(w["allowed"]):
            raise SystemExit(f"{name}: a claim key is listed twice")
    if sorted(out["policies"]["allowed"]) != sorted(policies_proposal["allowed"]["policies"]):
        raise SystemExit("policies: the keys derived from its commands differ from its proposal")
    return out


def preconditions(ws) -> None:
    """The record is written before any held-out computation and on the code the workstreams proposed."""
    for d in (heldout.CLAIMS, ROOT / "outputs/round5/heldout-locks"):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d.relative_to(ROOT)} is not empty; the record must come first")
    base = ROOT / "outputs/round5"
    early = sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*")
                   if any("-test" in x or x.startswith("test.") or x == "test" for x in p.relative_to(base).parts))
    if early:
        raise SystemExit("held-out outputs exist already:\n  " + "\n  ".join(early))
    proposed = {}
    for proposal in ("outputs/round5/priors/record-proposal.json", "outputs/round5/policies/record-proposal.json"):
        d = json.loads((ROOT / proposal).read_text())
        for section in ("code", "files"):
            for path, sha in d.get(section, {}).items():
                proposed.setdefault(path, set()).add(sha)
    changed = sorted(p for p, shas in proposed.items() if (ROOT / p).is_file() and sha256(ROOT / p) not in shas)
    unexplained = [p for p in changed if p not in ACKNOWLEDGED_CHANGES or ACKNOWLEDGED_CHANGES[p][0] != sha256(ROOT / p)]
    if unexplained:
        raise SystemExit("files changed since a workstream proposed them, without a recorded reason for this version:\n  " + "\n  ".join(unexplained))
    for w in ws.values():
        w["acknowledged_changes"] = {p: dict(sha256=ACKNOWLEDGED_CHANGES[p][0], reason=ACKNOWLEDGED_CHANGES[p][1]) for p in changed}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_files() -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for base in ("src", "scripts") for p in (ROOT / base).rglob("*.py")
                  if "__pycache__" not in p.parts)


def build() -> dict:
    launch_self_check()
    ws = workstreams()
    preconditions(ws)
    paths = set(code_files()) | {"outputs/round5/scope.json", "scripts/r5_heldout_job.sh"}
    for w in ws.values():
        e = w["expectations"]
        paths |= set([e] if isinstance(e, str) else e) | set(w["data"])
        paths |= set(w.get("scoped_files", []))
    missing = sorted(p for p in paths if not (ROOT / p).is_file())
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))
    return dict(
        written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        purpose="design of the round-5 held-out computations, fixed before any of them runs; each runs once",
        scope="outputs/round5/scope.json",
        files={p: sha256(ROOT / p) for p in sorted(paths)},
        expectations={k: w["expectations"] for k, w in ws.items()},
        data={k: w["data"] for k, w in ws.items()},
        allowed={k: w["allowed"] for k, w in ws.items()},
        argv={k: w["argv"] for k, w in ws.items()},
        workers={k: w["workers"] for k, w in ws.items()},
        after={k: w.get("after", {}) for k, w in ws.items()},
        commands={k: w["commands"] for k, w in ws.items()},
        launch=dict(cwd="<repository>", interpreter=".venv/bin/python", **LAUNCH,
                    note="the working directory, PYTHONPATH and thread settings are checked; --workers may differ, every other argument is checked"),
        notes={k: w["note"] for k, w in ws.items() if "note" in w},
        acknowledged_changes=ws["costs"]["acknowledged_changes"],
        environment=heldout.environment(),
        policies=dict(files={p: sha256(ROOT / p) for p in ws["policies"]["scoped_files"]},
                      frozen_choices=ws["policies"]["frozen_choices"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = build()
    text = json.dumps(record, indent=1) + "\n"
    if args.check:
        print(json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v) for k, v in record.items()}, indent=1))
        return
    if RECORD.exists():
        raise SystemExit(f"{RECORD.relative_to(ROOT)} exists; the record is written once")
    snap = snapshot(record["files"])
    record["code_snapshot"] = dict(path=snap.relative_to(ROOT).as_posix(), sha256=sha256(snap),
                                   note="the recorded code and small text files, to restore a byte-exact copy if a file is edited by mistake")
    text = json.dumps(record, indent=1) + "\n"
    tmp = RECORD.with_name(f".{RECORD.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        os.link(tmp, RECORD)
    finally:
        tmp.unlink()
    print(f"wrote {RECORD.relative_to(ROOT)} sha256 {sha256(RECORD)}", file=sys.stderr)


if __name__ == "__main__":
    main()
