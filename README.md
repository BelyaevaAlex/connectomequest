# ConnectomeQuest

Certifying typed-input statements in *Drosophila* connectomes with few checks.

## Overview

A typed-input statement says that one or several cell types supply at least a given share of a neuron's synaptic
input. The code turns such a statement into a weighted threshold question over the neuron's input cells, checks the
type of one input cell at a time against a reference annotation, and stops as soon as integer bounds certify the
answer. A type predictor only orders the checks; it never enters the certificate, so a wrong prediction costs checks
but never correctness.

## Repository layout

| Path | Contents |
|---|---|
| `src/connectomequest/annotation_v35/` | certification engine (`engine.py`), risk-limited sampling test (`statistical.py`), edge cutting and information gain (`decision.py`), controls, synthetic priors and ablations (`posthoc.py`, `sweep.py`, `ablation.py`) |
| `src/connectomequest/neurosym/` | outcome propagation, connectivity constraints, candidate types and the tuned linear policy |
| `src/connectomequest/round5/` | further comparators (optimal policy, value of information, sampling without replacement), cost models, seeded and correlated priors, BANC, and the one-run gate for held-out computations (`heldout.py`) |
| `src/connectomequest/annotation_v32/`, `annotation_v34/`, `transfer/`, `adapters/` | reading the releases, the crosswalk to hemibrain types, topology predictors (MLP, linear control) |
| `scripts/` | benchmark construction, analyses, generation of numbers, tables and figures (see below) |
| `tests/` | tests of the engine, the policies and every analysis |

File and module names carry internal version tags (`v32` to `v35`, `r5_`). They are kept because the records of the
analyses refer to these paths and because the pseudo-random split, targets and seeds are keyed by strings that
contain them.

## Scripts by stage

| Stage | Scripts |
|---|---|
| Preparation of graphs and features | `prepare_annotation_v32.py`, `run_v32_target_prepare.py`, `train_annotation_v32.py`, `fit_transfer_annotation_v32_ntac.py`, `correct_v32_historical_hints.py`, `rebind_v32_hint_predictions.py` |
| Benchmark | `v35_build.py`, `v35_train_seeds.py` |
| Pre-specified comparison | `v35_run.py`, `v35_analyze.py` |
| Descriptive analyses | `v35_diagnostics.py`, `v35_lenient.py`, `v35_coverage.py`, `v35_agreement.py` |
| Controls, synthetic priors, ablations | `v35_posthoc.py`, `v35_prior_sweep.py`, `v35_round3.py` and their `*_analyze.py` |
| Outcome propagation and other extensions | `neurosym_*.py` |
| Comparators, costs, priors, BANC | `r5_*.py` |
| Questions of both answers | `answers.py` |
| New target neurons, tuned and imitation policies | `replication.py`, `learned.py`, `imitation.py`, `imitation_models.py`, `fallback_share.py` |
| New cell types and partial coverage | `newtypes.py` |
| Missing correspondences and other labels | `correspondences.py`, `correspondences_readings.py`, `labelings.py`, `labelings_lists.py` |
| Published statements | `published.py`, `published_source.py`, `published_names.py` |
| Hedge of adaptive greedy (Proposition 1) | `hedge_family.py` |
| Numbers, tables, figures, manuscript | `*_assets.py`, `figures_extra.py`, `build_v35_paper.py`, `build_paper.py` |
| Expert time (not measured) | `timing_kit.py` |

Each analysis script lists its stages in its docstring.

## Data

The analyses read public releases that are not part of this repository. They are expected at these paths:

| Release | Path |
|---|---|
| hemibrain v1.2.1 | `data/raw/hemibrain/` (`neurons.parquet`, `edges/`) |
| MaleCNS v1.0 | `data/raw/manc/` (`body-annotations-...-v1.0-...feather`, `connectome-weights-...-v1.0-...feather`) |
| MaleCNS v0.9 annotation (earlier release) | `inputs/earlier/malecns-v0.9/` |
| FlyWire 783 with the v2.1.0 and v3.1.0 annotations | `inputs/v32/flywire/` |
| BANC v888 | `inputs/round5/banc/raw/` |

Prepared graphs and features are written to `inputs/v32/prepared/`, and results go to `outputs/`. `data/`, `inputs/`,
`outputs/` and the manuscript sources (`paper/`) are kept outside version control.

## Installation and tests

```bash
pip install -e .            # or: pip install -e ".[dev]" for pytest
PYTHONPATH=src:scripts python -m pytest tests -q
```

Tests that need the connectome graphs or the raw releases are skipped when these are absent.

## Protocol

1. Build the benchmark (`v35_build.py`) and run every policy on the development split (`v35_run.py --split development`).
2. Record the protocol (`outputs/v35/freeze.json`) before any held-out run.
3. Run the held-out split once (`v35_run.py --split test --freeze ...`) and analyse it (`v35_analyze.py`).
4. Every later analysis follows the same pattern: `run` on the development split, `record` (hashes of code and
   inputs, expected outcomes), `run --split test` once per connectome, `analyze --split test`. The runners refuse the
   held-out split when the code or inputs differ from the record, and a claim file prevents a second run.

## Further documentation

- Docstring of each analysis script: its stages and outputs.
- `paper/README.md` (with the manuscript, not in this repository): the build, the scripts behind every number,
  table and figure, and the files pinned by the records.
- `timing/README.md` (not in this repository): the sheet for measuring the time of one expert decision.

## License

Apache-2.0, see `LICENSE`.
