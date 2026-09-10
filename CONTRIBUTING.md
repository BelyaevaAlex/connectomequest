# Contributing

Use Python 3.10 or newer, keep public interfaces under `src/connectomequest`,
and add tests for behavioral changes. Before opening a change, run:

```bash
ruff check src scripts tests
python -m pytest -q
python scripts/build_submission.py --check
```

Do not commit credentials, raw EM data, local filesystem paths, generated cache
files, or alternative manuscript versions. Changes to frozen experimental
records require a new prospectively specified evaluation rather than overwriting
existing provenance.
