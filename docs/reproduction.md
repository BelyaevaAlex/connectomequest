# Reproduction

## Verify the checked-in release

```bash
python -m pip install -e '.[dev,embodied]'
python scripts/build_submission.py --check
```

This command verifies generated tables and the figure, runs the public test
suite, rebuilds both PDFs, checks the eight-page body limit, scans for identity
leakage, and validates bundle hashes.

## Rebuild the release

```bash
python scripts/build_submission.py
```

The command regenerates derived tables from frozen records and writes exactly
these public files:

```text
paper/main.pdf
paper/supplement.pdf
paper/artifact.zip
paper/sources.zip
paper/verification.json
```

It does not restart model training or overwrite raw experimental records.

## Artifact verification

```bash
mkdir /tmp/connectomequest-artifact
python -m zipfile -e paper/artifact.zip /tmp/connectomequest-artifact
python /tmp/connectomequest-artifact/verify_artifact.py
python /tmp/connectomequest-artifact/verify_artifact.py --tests
```

The artifact includes compact graph snapshots, frozen aggregates, complete
timing-confirmation records, representative RGB replays, current evaluation
code, and hash-addressed source provenance for runs whose exact source bytes
were retained. It omits raw EM data, full RGB trace collections, and retraining
claims.

## Dataset preparation

Use `cq download --help` and the dataset-specific commands documented in
`docs/data.md`. Credentials must be supplied through environment variables and
must never be committed.
