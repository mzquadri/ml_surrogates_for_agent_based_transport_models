# Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models

**Master's Thesis**

**Author:** Mohd Zamin Quadri
**Programme:** M.Sc. Mathematics in Science and Engineering
**Institution:** Technical University of Munich, School of Computation, Information and Technology
**Supervisor:** Prof. Dr. Rolf Moeckel
**Advisor:** Dr. Ana Moreno
**Date:** April 2026

---

## Overview

This repository contains the source code, trained models, data, and LaTeX source files for the Master's thesis titled *Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models*.

The thesis extends the work of Natterer et al. (2025), who developed Graph Neural Network (GNN) surrogate models to replace computationally expensive MATSim agent-based transport simulations for the Paris Ile-de-France road network (31,635 road segments). While the original work focused on model architecture and deterministic prediction accuracy, this thesis investigates a critical missing component: **how much should a decision-maker trust the surrogate's predictions?**

To answer this, the thesis develops and evaluates a comprehensive uncertainty quantification (UQ) framework consisting of:

- **MC Dropout inference** -- 30 stochastic forward passes through the trained GNN to estimate per-road prediction uncertainty, achieving a Spearman rank correlation of 0.482 between predicted uncertainty and actual prediction error.
- **Conformal prediction** -- distribution-free prediction intervals with guaranteed finite-sample coverage: 90.02% at the 90% target level and 95.01% at the 95% target level.
- **Calibration analysis** -- reliability diagrams, temperature scaling (reducing Expected Calibration Error by 82%), selective prediction (retaining 80% of roads while reducing MAE by 16%), and error detection (AUROC = 0.759 for top-10% error identification).
- **Cross-replication** -- independent validation on a second trained model (Trial 7), confirming that the UQ methods generalise beyond a single training configuration.

The work demonstrates that post-hoc UQ methods can provide actionable confidence measures for GNN surrogate predictions without requiring model retraining, making them practical for deployment in transport policy decision support.

---

## Context and Relationship to Prior Work

This thesis builds on the codebase and trained models from:

> Natterer, E. S., Rao, S. R., Tejada Lapuerta, A., Engelhardt, R., Horl, S., & Bogenberger, K. (2025). Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis. *Transportation Research Part C: Emerging Technologies*, 180, 105360.

The original paper established the GNN surrogate architecture (PointNet + Transformer + GAT hybrid) and conducted eight training trials (T1--T8) with systematic hyperparameter exploration. This thesis takes the trained models as given and contributes:

1. A rigorous UQ framework applied to the best-performing model (Trial 8, R-squared = 0.5957, MAE = 3.96 veh/h)
2. Conformal prediction intervals with formal coverage guarantees
3. Calibration diagnostics and temperature scaling correction
4. Practical decision-support analyses (selective prediction, error detection)
5. Cross-replication on Trial 7 to assess generalisability

All UQ analyses were conducted post-hoc on existing model predictions; no models were retrained.

---

## Repository Contents

```
.
├── README_SUBMISSION.md            This file
├── thesis/latex_tum_official/      LaTeX source files for the thesis
│   ├── main.tex                    Master document
│   ├── main.pdf                    Compiled thesis PDF
│   ├── settings.tex                Package and style configuration
│   ├── bibliography.bib            BibLaTeX bibliography (38 entries)
│   ├── chapters/                   7 chapter .tex files
│   ├── pages/                      Abstract, acknowledgments, Zusammenfassung
│   ├── figures/                    29 thesis figures (PDF+PNG) + 5 generator scripts
│   └── logos/                      TUM/faculty logos
│
├── scripts/                        Core model and evaluation code
│   ├── gnn/models/                 GNN architectures (PointNet+Transformer+GAT, GAT, GCN, etc.)
│   ├── evaluation/                 UQ analysis: MC Dropout, conformal prediction, calibration
│   ├── data_preprocessing/         MATSim simulation data to PyG graph conversion
│   ├── training/                   Model training pipeline
│   └── misc/                       Feature importance, utilities
│
├── data/
│   ├── TR-C_Benchmarks/            8 trained model trials (T1--T8) with results
│   │   ├── */trained_model/        Trained model weights (.pth)
│   │   ├── */test_predictions.npz  Deterministic test predictions
│   │   ├── */uq_results/          MC Dropout and conformal prediction outputs
│   │   └── */*.json               Evaluation metrics
│   └── visualisation/              Paris district GeoJSON files
│
├── docs/verified/                  Verified numerical results and documentation
│   └── phase3_results/             JSON files with all reported UQ numbers
│
├── run_part2_uq_analyses.py        Selective prediction + error detection (T8)
├── run_part3_calibration_audit.py  Calibration audit runner
├── run_part4_t7_crosscheck.py      Trial 7 cross-replication runner
│
├── environment-minimal.yml         Conda environment (cross-platform)
└── traffic-gnn.yml                 Conda environment (full, pinned, Linux)
```

---

## Environment Setup

```bash
# Cross-platform (recommended):
conda env create -f environment-minimal.yml
conda activate traffic-gnn

# Exact pinned environment (Linux only):
conda env create -f traffic-gnn.yml
conda activate traffic-gnn

# Key dependencies: Python 3.10, PyTorch 2.2, PyG 2.5, numpy, scipy, matplotlib, scikit-learn
```

The full `traffic-gnn.yml` was exported on Linux and contains platform-specific build strings. For Windows or macOS, use `environment-minimal.yml` instead. The core analysis scripts depend only on standard scientific Python packages and are platform-independent.

---

## Reproducing the Thesis PDF

Requires a LaTeX distribution with `pdflatex` and `biber` (e.g., MiKTeX or TeX Live).

```bash
cd thesis/latex_tum_official/

# Full build sequence (from scratch):
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# Output: main.pdf
```

This thesis uses `biblatex` with the `biber` backend, not `bibtex`.

---

## Reproducing Thesis Figures

All thesis figures can be regenerated from the included data files:

```bash
cd thesis/latex_tum_official/figures/

# Generate trial comparison, UQ ranking, conformal, and calibration figures
python generate_all_thesis_figures.py

# Generate workflow diagrams
python generate_new_figures.py

# Generate network introduction figure
python generate_network_intro_figure.py

# Generate Phase 3 UQ figures (reliability diagram, temperature scaling)
python generate_phase3_figures.py

# Generate PointNet data flow diagram
python generate_pointnet_dataflow_figure.py
```

Each script produces paired PDF (for LaTeX) and PNG (300 dpi) outputs.

---

## Reproducing UQ Analyses

The three top-level runner scripts reproduce the UQ analyses reported in Chapter 5:

```bash
# Selective prediction + error detection for Trial 8
python run_part2_uq_analyses.py

# Calibration audit (reliability diagram, temperature scaling, ECE)
python run_part3_calibration_audit.py

# Trial 7 cross-replication (independent validation)
python run_part4_t7_crosscheck.py
```

These scripts read from pre-computed prediction files (`.npz`) and produce verified results in `docs/verified/`.

---

## Key Results

### Deterministic Performance (Trial 8)

| Metric | Value |
|--------|-------|
| R-squared | 0.5957 |
| MAE | 3.96 veh/h |
| MAE (MC Dropout mean, 30 passes) | 3.94 veh/h |
| RMSE | 7.12 veh/h |

### Uncertainty Quantification

| Analysis | Result |
|----------|--------|
| MC Dropout Spearman rho (uncertainty vs. error) | 0.482 |
| AUROC for top-10% error detection | 0.759 |
| Conformal 90% coverage | 90.02% (q_hat = 9.92) |
| Conformal 95% coverage | 95.01% (q_hat = 14.68) |
| ECE after temperature scaling | 0.048 (82% improvement over uncalibrated) |
| Selective prediction (retain 80% of roads) | 16% MAE reduction |
| 95th-percentile conformal interval half-width (k_95) | 11.34 veh/h |

### Trial 7 Cross-Replication

| Analysis | Trial 7 | Trial 8 |
|----------|---------|---------|
| MC Dropout Spearman rho | 0.469 | 0.482 |
| Conformal 90% coverage | 89.98% | 90.02% |
| Conformal 95% coverage | 95.03% | 95.01% |

All numbers verified against raw data files in `data/TR-C_Benchmarks/` and documented in `docs/verified/phase3_results/`.

---

## Model Architecture

The primary model (`scripts/gnn/models/point_net_transf_gat.py`) is a PointNet + Transformer + GAT hybrid:

- **Input:** 7 node features + 3D positional encoding per road segment
- **Local MLP:** 7 to 256 dimensions
- **Global MLP:** 514 to 256 (concatenated with max-pooled global features)
- **Transformer Encoder:** 4 attention heads, 64-dimensional keys
- **GAT Layers:** 4 attention heads, 128 hidden to 64 output dimensions
- **Output Head:** GATConv(64 to 1) for Trials 2--8; Linear(64 to 1) for Trial 1
- **MC Dropout:** p = 0.2, applied during both training and inference (30 forward passes)

---

## Data Notes

- **Training data** (`data/train_data/`, 2.62 GB) is excluded from this repository due to size. It consists of 20 pre-processed `.pt` batch files derived from 10,000 MATSim simulations of the Paris Ile-de-France transport network.
- All results reported in the thesis can be reproduced from the included `.npz` prediction files and `.json` metrics without retraining.
- Results are based on 1,000 of the 10,000 available MATSim scenarios (10% subset).
- The MATSim simulation data was generated by Dr. Ana Moreno's research group at TUM.

---

## File Integrity

- All 29 referenced figures present and generated from verified data
- All 38 bibliography entries resolve correctly (compiled with zero citation warnings)
- 150+ numerical claims cross-verified against source data files

---

## License

This work is submitted as a Master's thesis at the Technical University of Munich. Please contact the author for reuse permissions.
