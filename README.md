# Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models

**Master's Thesis** | Technical University of Munich | School of Computation, Information and Technology

|               |                                      |
| ------------- | ------------------------------------ |
| **Author**    | Mohd Zamin Quadri                    |
| **Programme** | M.Sc. Mathematics in Science and Engineering |
| **Supervisor**| Prof. Dr. Rolf Moeckel               |
| **Advisor**   | Dr. Ana Moreno                        |
| **Date**      | April 2026                            |

---

## Abstract

Agent-based transport simulations such as MATSim are powerful tools for policy analysis but computationally expensive. Graph Neural Network (GNN) surrogates can approximate these simulations orders of magnitude faster, yet their predictions lack confidence estimates -- a critical gap for policy decision-making.

This thesis develops a **post-hoc uncertainty quantification (UQ) framework** for a GNN surrogate trained on 10,000 MATSim simulations of the Paris Ile-de-France road network (31,635 road segments). The framework combines:

- **MC Dropout** -- 30 stochastic forward passes to estimate per-road prediction uncertainty
- **Conformal Prediction** -- distribution-free prediction intervals with guaranteed finite-sample coverage
- **Calibration Diagnostics** -- reliability diagrams and temperature scaling correction
- **Selective Prediction** -- flagging uncertain roads to reduce error in retained predictions
- **Error Detection** -- ranking predictions by uncertainty to identify likely failures

All methods are applied post-hoc to existing trained models. No retraining is required.

---

## Key Results

### Deterministic Performance (Trial 8)

| Metric | Value   |
| ------ | ------- |
| R^2    | 0.5957  |
| MAE    | 3.96 veh/h |
| RMSE   | 7.12 veh/h |

### Uncertainty Quantification

| Analysis | Result |
| -------- | ------ |
| MC Dropout Spearman rho (uncertainty vs. error) | 0.482 |
| Conformal 90% / 95% coverage | 90.02% / 95.01% |
| ECE improvement after temperature scaling | 82% (0.269 --> 0.048) |
| Selective prediction MAE reduction (80% retention) | 16% |
| Error detection AUROC (top-10% errors) | 0.759 |

### Cross-Replication (Trial 7)

| Metric | Trial 7 | Trial 8 |
| ------ | ------- | ------- |
| MC Dropout Spearman rho | 0.469 | 0.482 |
| Conformal 90% coverage  | 89.98% | 90.02% |
| Conformal 95% coverage  | 95.03% | 95.01% |

All numbers verified against raw data artifacts. See [`docs/verified/phase3_results/`](docs/verified/phase3_results/) for machine-readable JSON results.

---

## Repository Structure

```
.
├── thesis/                              # Thesis document
│   └── latex_tum_official/
│       ├── main.tex                     # Master document
│       ├── main.pdf                     # Compiled thesis (81 pages)
│       ├── chapters/                    # 01_introduction .. 07_conclusion
│       ├── figures/                     # 29 figures (PDF) + generation scripts
│       ├── pages/                       # Abstract, acknowledgments, Zusammenfassung
│       ├── bibliography.bib             # 38 BibLaTeX entries
│       └── settings.tex                 # Package configuration
│
├── scripts/                             # Source code
│   ├── gnn/                             # GNN architectures
│   │   ├── models/
│   │   │   ├── point_net_transf_gat.py  # Primary model (PointNet + Transformer + GAT)
│   │   │   ├── gat.py, gcn.py, ...      # Baseline architectures
│   │   │   ├── block/                   # Custom message-passing blocks
│   │   │   └── conv/                    # Custom convolution layers
│   │   ├── gnn_io.py                    # Data I/O utilities
│   │   └── help_functions.py            # Metrics, MC Dropout inference
│   ├── evaluation/                      # UQ analysis and plotting scripts
│   ├── training/                        # Model training pipeline
│   ├── data_preprocessing/              # MATSim simulation --> PyG graph conversion
│   └── misc/                            # Feature importance, notebooks
│
├── docs/                                # Documentation
│   ├── data_preprocessing.md            # Data pipeline reference
│   ├── gnn.md                           # GNN architecture notes
│   ├── training.md                      # Training procedure
│   ├── COMPLETE_RESEARCH_SUMMARY.md     # Full research overview
│   ├── ENSEMBLE_UQ_DETAILED_EXPLANATION.md
│   └── verified/                        # Verified results and audit documents
│       ├── phase3_results/              # JSON files with all reported UQ numbers
│       ├── VERIFIED_RESULTS_MASTER.csv  # Cross-reference of all numeric claims
│       └── UQ_*.md                      # Per-method audit reports
│
├── run_part2_uq_analyses.py             # Selective prediction + error detection
├── run_part3_calibration_audit.py       # Calibration audit (reliability, ECE, temp. scaling)
├── run_part4_t7_crosscheck.py           # Trial 7 cross-replication
│
├── environment-minimal.yml              # Conda environment (cross-platform)
└── traffic-gnn.yml                      # Conda environment (Linux, fully pinned)
```

> **Note:** Training data (`data/`, ~4.8 GB) and model checkpoints are excluded from this repository due to size. All reported results can be reproduced from the included pre-computed prediction files (`.npz`).

---

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (Miniconda or Anaconda)
- LaTeX distribution with `pdflatex` and `biber` (for thesis compilation only)

### Environment Setup

```bash
# Cross-platform (recommended)
conda env create -f environment-minimal.yml
conda activate traffic-gnn

# Exact pinned environment (Linux only)
conda env create -f traffic-gnn.yml
conda activate traffic-gnn
```

Core dependencies: PyTorch 2.2, PyTorch Geometric 2.5, NumPy, SciPy, scikit-learn, Matplotlib.

### Reproducing UQ Analyses

```bash
# Selective prediction + error detection (Trial 8)
python run_part2_uq_analyses.py

# Calibration audit (reliability diagram, temperature scaling, ECE)
python run_part3_calibration_audit.py

# Trial 7 cross-replication (independent validation)
python run_part4_t7_crosscheck.py
```

These scripts read pre-computed predictions (`.npz`) and write verified results to `docs/verified/`.

### Regenerating Thesis Figures

```bash
cd thesis/latex_tum_official/figures/

python generate_all_thesis_figures.py        # Trial comparison, UQ ranking, conformal, calibration
python generate_new_figures.py               # Workflow diagrams
python generate_phase3_figures.py            # Phase 3 UQ figures
python generate_network_intro_figure.py      # Network introduction
python generate_pointnet_dataflow_figure.py  # Architecture data flow
```

### Compiling the Thesis

```bash
cd thesis/latex_tum_official/
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

> Uses `biblatex` with the `biber` backend (not `bibtex`).

---

## Model Architecture

The primary surrogate is a **PointNet + Transformer + GAT** hybrid GNN ([`scripts/gnn/models/point_net_transf_gat.py`](scripts/gnn/models/point_net_transf_gat.py)):

```
Input: 5 node features + 3D positional encoding (per road segment)
  --> Local MLP (7 --> 256)
  --> Global MLP (514 --> 256, with max-pooled global context)
  --> Transformer Encoder (4 heads, d_k = 64)
  --> 4x GAT Layers (128 --> 64, 4 heads each)
  --> Output Head: GATConv(64 --> 1)
```

**Node features:** traffic volume (baseline), road capacity (baseline), capacity reduction, free-flow speed, segment length.

**MC Dropout** (p = 0.2) is applied during both training and inference to generate epistemic uncertainty estimates via 30 stochastic forward passes.

---

## Builds On

> Natterer, E. S., Rao, S. R., Tejada Lapuerta, A., Engelhardt, R., Horl, S., & Bogenberger, K. (2025). *Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis.* Transportation Research Part C: Emerging Technologies, 180, 105360.

This thesis takes the trained models from the above work as given and contributes the UQ framework, calibration analysis, and cross-replication study.

---

## License

This work is submitted as a Master's thesis at the Technical University of Munich. Contact the author for reuse permissions.
