# Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models

**Master's Thesis** | Technical University of Munich | School of Computation, Information and Technology

|               |                                      |
| ------------- | ------------------------------------ |
| **Author**    | Mohd Zamin Quadri                    |
| **Programme** | M.Sc. Mathematics in Science and Engineering |
| **Supervisor**| Prof. Dr. Stephan Günnemann           |
| **Advisor**   | Dominik Fuchsgruber, M.Sc., Elena Natterer, M.Sc. |
| **Date**      | May 15, 2026                          |

**[Read the thesis (PDF)](thesis/latex_tum_official/main.pdf)**

---

## Abstract

Agent-based transport simulations like MATSim are powerful but computationally expensive. GNN surrogates approximate them orders of magnitude faster, yet lack confidence estimates -- a critical gap for policy decisions.

This thesis develops a post-hoc uncertainty quantification framework for a GNN surrogate trained on 10,000 MATSim simulations of the Paris Ile-de-France road network (31,635 road segments), combining MC Dropout, conformal prediction, calibration diagnostics, selective prediction, and error detection. No retraining is required.

---

## Key Results

| Analysis | Trial 8 | Trial 7 |
| -------- | ------- | ------- |
| Deterministic MAE / RMSE | 3.96 / 7.12 veh/h | -- |
| R^2 | 0.5957 | -- |
| MC Dropout Spearman rho | 0.482 | 0.446 |
| Conformal 90% / 95% coverage | 90.02% / 95.01% | 89.98% / 95.03% |
| ECE (before / after temp. scaling) | 0.269 / 0.048 | -- |
| Selective prediction MAE reduction @50% | 41.2% | -- |
| Error detection AUROC (top-10%) | 0.7548 | -- |

All numbers verified against raw artifacts. See [`docs/verified/`](docs/verified/) for audit reports and JSON results, and `results/` for the canonical result set.

---

## Repository Structure

```
thesis/latex_tum_official/    Working thesis document (LaTeX source + compiled PDF + DOCX)
thesis/submission_2026-05-15/ Frozen as-submitted thesis (PDF + ZIP + LaTeX) -- do not edit
thesis/variants/              Earlier document snapshots, kept for provenance
models/                       Trained model checkpoints (16 trials, PyTorch .pth)
data/                         Dataloaders, training corpus, Paris network layer -- see DATA.md (7.8 GB, not in git)
scripts/gnn/                  GNN architectures (PointNet + Transformer + GAT, incl. heteroscedastic + CQR variants)
scripts/evaluation/           UQ analysis and plotting scripts
scripts/training/             Model training pipeline (incl. deep ensemble and CQR training)
scripts/data_preprocessing/   MATSim --> PyG graph conversion
scripts/misc/                 Figure generation and analysis helpers
scripts/archive/              Superseded one-off scripts, kept for provenance
notebooks/                    Colab notebooks (training, UQ, baselines)
docs/                         Documentation and verified results
docs/figures/                 Thesis figures, EDA plots, and hand-picked diagrams
results/                      Canonical result JSONs, pre-computed predictions (.npz), metrics, training logs
analysis_outputs/             Generated analysis figures and intelligence reports
thesis_dashboard/             Streamlit dashboard over the result set
policy-dashboard/             Policy confidence desk
web_exports/                  Exported web artifacts (JSON + WebP)
presentation/                 Defence slide decks
tests/                        Test suite
run_part{2,3,4}_*.py          Reproducibility verification scripts
environment-minimal.yml       Conda environment (cross-platform)
```

> **The thesis that was examined is `thesis/submission_2026-05-15/`, not `thesis/latex_tum_official/`.** The working LaTeX has been edited since submission and compiles to a different PDF. See that folder's `README.md` for the details.

> Included: all reported result JSONs, the trained model checkpoints, and the key pre-computed prediction archives (.npz) that back the reported numbers. Excluded from version control: the 7.8 GB `data/` tree — 39 of its files exceed GitHub's 100 MB per-file limit — along with the presentation decks and the Colab-side training outputs. See [`DATA.md`](DATA.md) for what `data/` contains, where to download it, and how to verify it against `data/MANIFEST.sha256`.

---

## Reproducing Results

```bash
git clone https://github.com/mzquadri/ml_surrogates_for_agent_based_transport_models.git
cd ml_surrogates_for_agent_based_transport_models
conda env create -f environment-minimal.yml
conda activate traffic-gnn

python scripts/evaluation/run_part2_uq_analyses.py       # Selective prediction + error detection
python scripts/evaluation/run_part3_calibration_audit.py # Calibration and conformal coverage
python scripts/evaluation/run_part4_t7_crosscheck.py     # Trial 7 cross-check
```

The scripts reproduce analyses from versioned prediction artifacts; they do not retrain the models or rerun MATSim simulations. The prediction archives are mirrored under `results/predictions/`; evaluation scripts look for them under the local `data/` tree, so place the extracted `data/` directory (or point the scripts' `data/` paths at `results/predictions/`) before running. See [`docs/verified/REPRODUCIBILITY_GAP_SUMMARY.md`](docs/verified/REPRODUCIBILITY_GAP_SUMMARY.md) for a precise account of included artifacts and known limitations.

To compile the thesis: `cd thesis/latex_tum_official && pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex`

---

## Builds On

> Natterer et al. (2025). *Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis.* Transportation Research Part C, 180, 105360.

This thesis takes the trained models from the above work as given and contributes the UQ framework, calibration analysis, and cross-replication study.

---

## License

This repository is a fork of [`enatterer/ml_surrogates_for_agent_based_transport_models`](https://github.com/enatterer/ml_surrogates_for_agent_based_transport_models), released under the **MIT License, Copyright (c) 2024 Elena Natterer**.

That license governs the upstream code kept and extended here — `scripts/data_preprocessing/`, `scripts/gnn/`, `scripts/training/`, `scripts/evaluation/help_functions.py`, `scripts/evaluation/plot_functions.py`, the upstream notebooks, and `traffic-gnn.yml`. Its terms, including the copyright and permission notice reproduced in [`LICENSE`](LICENSE), continue to apply to that code and to anything derived from it.

The material added by this fork — the thesis text and figures, the trained model checkpoints, and the result artifacts — is **not** covered by that MIT grant. Reuse of those requires prior permission from the author and, where applicable, the original data and model owners.

## Tooling

AI coding assistants were used during this work for code review, refactoring, and
repository maintenance. All research design, modelling decisions, analysis, and
written content are the author's own, and every reported result was verified
against the artifacts in this repository.

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata.
