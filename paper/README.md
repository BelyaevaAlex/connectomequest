# Paper and evaluation artifact

This directory contains the single current anonymous manuscript and its
supporting files. There are no archived manuscript versions in the publication
tree.

## Files

- `main.tex`, `main.pdf`: submission manuscript (8 body pages).
- `supplement.tex`, `supplement.pdf`: supplementary material.
- `generated/`: tables and the figure regenerated from frozen records.
- `artifact.zip`: executable evaluation artifact; it contains no manuscript.
- `sources.zip`: self-contained LaTeX source archive.
- `verification.json`: hashes, test counts, and release checks.

## Build

From the repository root:

```bash
python scripts/build_submission.py
python scripts/build_submission.py --check
```

Or build only the PDFs:

```bash
make -C paper
```

The manuscript uses the official NeurIPS 2026 style with the double-blind
workshop option. Author-identifying information and local filesystem paths are
rejected by the release checker.
