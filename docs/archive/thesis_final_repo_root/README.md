# Uncertainty Quantification for Machine Learning Models in Transportation Policy Analysis

Master's thesis submitted at the Technical University of Munich on May 15, 2026.

| | |
|---|---|
| Author | Mohd Zamin Quadri |
| Program | M.Sc. Mathematics in Science and Engineering |
| Department | Computer Science, School of Computation, Information and Technology |
| Examiner | Prof. Dr. Stephan Gunnemann |
| Advisors | Dominik Fuchsgruber, Elena Natterer |

The immutable submitted thesis is [`document/main.pdf`](document/main.pdf). Its SHA-256 is
`0ac5309d060cda53d82a05cc837136fe853e7f9dcbabd2f4fb4b4282a39bc97e`.
Repository corrections are post-submission corrigenda and reproducibility updates; they do not
alter or replace the submitted PDF.

## Research question

Large-scale agent-based traffic simulations are expensive to evaluate repeatedly. This work
studies whether graph-neural-network surrogates can provide useful predictions of policy-induced
**traffic-volume change** (`Delta v`, veh/h) while exposing uncertainty that supports calibration,
selective prediction, and review decisions.

The experiments use a fixed 1,000-scenario subset of a 10,000-scenario Paris MATSim corpus. The
held-out test split contains 100 scenarios with 31,635 road segments each, producing 3,163,500
node-level predictions. The evidence covers one network, one intervention family (capacity
reduction), and the PointNetTransfGAT model family; it does not establish cross-city,
cross-policy, or production performance.

## Post-submission corrigendum

A full-array audit after submission corrected a material explanatory claim:

- The test target contains **872,540 exact zeros out of 3,163,500 values: 27.58%**.
- The previously reported **88.7%** is the zero share of the raw `CAPACITY_REDUCTION` input
  feature, not the target.
- The correction changes the interpretation of target sparsity but does not rewrite the
  historical model outputs or submitted thesis.

The audit also records split-specific preprocessing risk, stochastic MC Dropout replay
variation, checkpoint-loading risk, and protocol-dependent calibration results. See
[`docs/CORRIGENDUM.md`](docs/CORRIGENDUM.md) and the safe aggregate
[`analysis_outputs/THESIS_INTELLIGENCE_REPORT.md`](analysis_outputs/THESIS_INTELLIGENCE_REPORT.md).

## Audited evidence

The audited Trial 8 archive contains 3,163,500 held-out predictions:

| Evidence | Result | Boundary |
|---|---:|---|
| Deterministic Trial 8 | R2 0.596, MAE 3.96 veh/h, RMSE 7.12 veh/h | Historical held-out result |
| Trial 8 MC Dropout replay | R2 0.586, MAE 3.95 veh/h, Spearman rho 0.482 | Stochastic cached replay |
| Five-model deep ensemble | R2 0.684, MAE 3.49 veh/h, Spearman rho 0.400 | Cached full-test predictions |
| 50% selective retention | MAE 2.32 veh/h, 41.2% below full-set MAE | Retrospective triage result |
| Split conformal | 90.02% / 95.01% empirical marginal coverage | Submission-era reported result; 50/50 scenario split |

Calibration results are intentionally kept separate:

- `graph20_80_v1`: first 20 graphs calibrate and the last 80 evaluate; temperature 2.702,
  ECE 0.269 to 0.048. This protocol is backed by tracked audit artifacts.
- `node30_70_thesis_final`: random 30%/70% node split; temperature approximately 2.887,
  ECE approximately 0.356 to 0.034. Split indices were not retained, so this protocol is
  reported rather than independently replayed.
- Primary split conformal: 50 calibration scenarios and 50 evaluation scenarios, seed 42.

Coverage is empirical and marginal over the evaluated split. Nodes within a scenario are
dependent, and these results are not per-scenario or deployment guarantees.

## Repository layout

```text
document/                 Immutable submitted PDF and submission-era LaTeX source
code/                     Submission-era preprocessing, model, training, and analysis code
analysis_outputs/         Safe aggregate post-submission audit bundle and figures
scripts/analysis/         Aggregate audit generator; requires separately held evidence assets
thesis_dashboard/         Local Streamlit viewer backed by the committed aggregate bundle
tests/                    Privacy, provenance, analytics, and dashboard checks
docs/CORRIGENDUM.md       Corrections and methodological boundaries
docs/DATASET.md           Measured training corpus, feature statistics, and model input
docs/ARTIFACT_PROVENANCE.md
```

[`docs/DATASET.md`](docs/DATASET.md) documents what the published dataloaders contain, measured
over all 1,000 scenarios rather than described from the preprocessing code. Regenerate it with
`python scripts/analyse_train_data.py` once the release assets are downloaded.

## Reproducibility levels

This repository supports three different levels of inspection:

1. **Submitted-artifact verification:** validate the immutable PDF and repository contracts with
   `python scripts/check_repository.py`.
2. **Aggregate evidence review:** inspect the committed JSON/CSV/report outputs or run the local
   dashboard with `streamlit run thesis_dashboard/app.py`.
3. **Aggregate regeneration:** after restoring the separately controlled artifacts at the
   manifest paths, run `python scripts/analysis/generate_thesis_intelligence.py
   --include-local-graphs` to reproduce the committed bundle, including graph-quality aggregates.
   Every input is size- and SHA-256-checked before use. The row-level and pickle-capable inputs are
   intentionally excluded from this canonical public repository.

Raw MATSim scenarios are unavailable here, so raw-simulation-to-graph reproduction is not
supported. Training data, graph loaders, scalers, row-level prediction arrays, and checkpoints
remain local or access-controlled because of size, provenance, redistribution, and pickle-safety
boundaries. The committed aggregate bundle contains no row-level records or absolute local paths.

### Local audit environment

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r dashboard-requirements.txt -r requirements-dev.txt
.venv/Scripts/python scripts/check_repository.py
.venv/Scripts/python -m pytest -p no:cacheprovider tests
```

On Linux or macOS, use `.venv/bin/python` instead. The broader submission-era training
environment remains in `code/environment-minimal.yml`.

## Provenance and reuse

The aggregate evidence was generated from audited source commit
[`fdb4ef0`](https://github.com/mzquadri/ml_surrogates_for_agent_based_transport_models/commit/fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a).
The canonical submitted-artifact baseline is commit `4b95a3d`. Exact hashes and migration
boundaries are documented in [`docs/ARTIFACT_PROVENANCE.md`](docs/ARTIFACT_PROVENANCE.md).

The simulation corpus and inherited research scaffold are not owned solely by this repository.

The inherited scaffold under `code/scripts/` — including `gnn/`, `data_preprocessing/`,
`training/`, and the shared evaluation helpers — originates in
[`enatterer/ml_surrogates_for_agent_based_transport_models`](https://github.com/enatterer/ml_surrogates_for_agent_based_transport_models)
and is licensed **MIT, Copyright (c) 2024 Elena Natterer**. That grant, reproduced in
[`LICENSE`](LICENSE), applies to that code and to work derived from it.

No license is granted for anything else here. The thesis text, figures, model artifacts, and
source data remain reserved; contact the relevant rights holders before reusing them.

## Tooling

AI coding assistants were used during this work for code review, refactoring, and
repository maintenance. All research design, modelling decisions, analysis, and
written content are the author's own, and every reported result was verified
against the audited evidence described above.

## Citation

```text
Quadri, Mohd Zamin (2026). Uncertainty Quantification for Machine Learning Models
in Transportation Policy Analysis. Master's thesis submitted at the Technical
University of Munich, May 15, 2026.
```
