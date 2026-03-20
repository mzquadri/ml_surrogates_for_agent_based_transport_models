# Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models

**Master's Thesis** | Technical University of Munich | School of Computation, Information and Technology

|               |                                      |
| ------------- | ------------------------------------ |
| **Author**    | Mohd Zamin Quadri                    |
| **Programme** | M.Sc. Mathematics in Science and Engineering |
| **Supervisor**| Prof. Dr. Stephan Günnemann           |
| **Advisor**   | Dominik Fuchsgruber, M.Sc., Elena Natterer, M.Sc. |
| **Date**      | April 2026                            |

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
| Selective prediction MAE reduction @80% | 16% | -- |
| Error detection AUROC (top-10%) | 0.759 | -- |

All numbers verified against raw artifacts. See [`docs/verified/`](docs/verified/) for audit reports and JSON results.

---

## Repository Structure

```
thesis/latex_tum_official/   Thesis document (LaTeX source + compiled PDF)
scripts/gnn/                 GNN architectures (PointNet + Transformer + GAT)
scripts/evaluation/          UQ analysis and plotting scripts
scripts/training/            Model training pipeline
scripts/data_preprocessing/  MATSim --> PyG graph conversion
docs/                        Documentation and verified results
run_part{2,3,4}_*.py         Reproducibility verification scripts
environment-minimal.yml      Conda environment (cross-platform)
```

> Training data (~4.8 GB) and model checkpoints are excluded. All reported results reproduce from included pre-computed predictions (.npz).

---

## Reproducing Results

```bash
conda env create -f environment-minimal.yml
conda activate traffic-gnn

python run_part2_uq_analyses.py         # Selective prediction + error detection
python run_part3_calibration_audit.py   # Calibration, ECE, temperature scaling
python run_part4_t7_crosscheck.py       # Trial 7 cross-replication
```

To recompile the thesis: `cd thesis/latex_tum_official && pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex`

---

## Builds On

> Natterer et al. (2025). *Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis.* Transportation Research Part C, 180, 105360.

This thesis takes the trained models from the above work as given and contributes the UQ framework, calibration analysis, and cross-replication study.

---

## License

Submitted as a Master's thesis at the Technical University of Munich. Contact the author for reuse permissions.
