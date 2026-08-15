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

This thesis develops a post-hoc uncertainty quantification framework for a GNN surrogate trained on 1,000 MATSim simulations of the Paris Ile-de-France road network (31,635 road segments), combining MC Dropout, conformal prediction, calibration diagnostics, selective prediction, and error detection. No retraining is required.

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

Headline numbers are backed by versioned prediction and result artifacts. A new full-array audit also corrects a thesis claim: 27.6% of test targets are exactly zero, not 88.7% (the latter is the zero share of the `CAPACITY_REDUCTION` input feature). See [`analysis_outputs/THESIS_INTELLIGENCE_REPORT.md`](analysis_outputs/THESIS_INTELLIGENCE_REPORT.md) for findings and limitations.

---

## Repository Structure

```
thesis/latex_tum_official/   Thesis document (final LaTeX source + compiled PDF + DOCX)
models/                      Trained model checkpoints (16 trials, PyTorch .pth)
scripts/gnn/                 GNN architectures (PointNet + Transformer + GAT, incl. heteroscedastic + CQR variants)
scripts/evaluation/          UQ analysis and plotting scripts
scripts/analysis/            Canonical safe aggregate regeneration pipeline
scripts/training/            Model training pipeline (incl. deep ensemble and CQR training)
scripts/data_preprocessing/  MATSim --> PyG graph conversion
scripts/misc/                Figure generation and analysis helpers
docs/                        Documentation and verified results
results/                     Canonical result JSONs, pre-computed predictions (.npz), and per-trial metrics
analysis_outputs/            Safe aggregate bundle, report, manifest, tables, and figures
thesis_dashboard/            Local Traffic Policy Confidence Lab (Streamlit)
tests/                       Analytics, aggregate-privacy, and Streamlit AppTest coverage
environment-minimal.yml      Conda environment (cross-platform)
```

> Included: all reported result JSONs, the 16 trained model checkpoints, and the key pre-computed prediction archives (.npz) that back the reported numbers. Excluded from version control: confidential raw MATSim outputs and intermediate data loaders, two prediction archives over GitHub's 100 MB per-file limit (`experiment_a_fixed_data.npz`, `feature_data.npz`), and Colab-side training outputs. Raw-to-graph reproduction is therefore not possible from a fresh clone; prediction-to-analysis reproduction is the strongest supported path.

---

## Local Evidence Lab

The Streamlit dashboard consumes a schema-validated aggregate bundle. It binds to `127.0.0.1`, disables telemetry, and offers only aggregate downloads.

```powershell
conda activate thesis-env
python scripts/analysis/generate_thesis_intelligence.py
streamlit run thesis_dashboard/app.py
```

The default regeneration uses only tracked numeric NPZ files (`allow_pickle=False`) and JSON. On the audited workstation, `--include-local-graphs` additionally summarizes the trusted ignored T8 loader into aggregate statistics; never use that option with an untrusted `.pt` file.

Run the quality suite with:

```powershell
python -m pytest -p no:cacheprovider tests
ruff check thesis_dashboard scripts/analysis scripts/check_repository.py tests
pyright
python scripts/check_repository.py
```

Calibration outputs are protocol-versioned. The tracked 20/80 graph split and final-thesis 30/70 random node split are displayed separately and must not be pooled.

To compile the thesis: `cd thesis/latex_tum_official && pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex`

---

## Builds On

> Natterer et al. (2025). *Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis.* Transportation Research Part C, 180, 105360.

This thesis takes the trained models from the above work as given and contributes the UQ framework, calibration analysis, and cross-replication study.

---

## License

This repository is published for academic review and portfolio purposes. Reuse of the thesis text, figures, trained models, and included research artifacts requires prior permission from the author and, where applicable, the original data and model owners.

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata.
