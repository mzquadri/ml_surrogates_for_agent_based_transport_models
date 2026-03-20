# Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models

> **Master's Thesis** -- Technical University of Munich  
> School of Computation, Information and Technology  
> M.Sc. Mathematics in Science and Engineering

| | |
|---|---|
| **Author** | Mohd Zamin Quadri |
| **Supervisor** | Prof. Dr. Rolf Moeckel |
| **Advisor** | Dr. Ana Moreno |
| **Date** | April 2026 |

---

## Abstract

Agent-based transport simulations (e.g., MATSim) are powerful but computationally expensive. Graph Neural Network (GNN) surrogates can approximate these simulations orders of magnitude faster, but their predictions lack confidence estimates -- a critical gap for policy decision-making.

This thesis develops a **post-hoc uncertainty quantification (UQ) framework** for a GNN surrogate trained on 10,000 MATSim simulations of the Paris Ile-de-France road network (31,635 road segments). The framework combines:

- **MC Dropout** (30 stochastic forward passes) to estimate per-road prediction uncertainty
- **Conformal prediction** for distribution-free coverage guarantees
- **Calibration diagnostics** with temperature scaling correction
- **Selective prediction** and **error detection** for practical deployment

All UQ methods are applied post-hoc to existing trained models -- no retraining required.

### Key Results

| Metric | Value |
|---|---|
| MC Dropout rank correlation (uncertainty vs. error) | 0.482 |
| Conformal 90% / 95% coverage | 90.02% / 95.01% |
| ECE improvement after temperature scaling | 82% |
| MAE reduction via selective prediction (80% retention) | 16% |
| Error detection AUROC (top-10% errors) | 0.759 |

Results independently validated on a second trained model (Trial 7), confirming generalisability.

---

## Repository Structure

```
.
├── thesis/latex_tum_official/       # LaTeX source and compiled PDF
│   ├── main.tex                     # Master document
│   ├── main.pdf                     # Compiled thesis (81 pages)
│   ├── chapters/                    # 7 chapter files
│   ├── figures/                     # 29 figures + generation scripts
│   └── bibliography.bib            # 38 BibLaTeX entries
│
├── scripts/
│   ├── gnn/models/                  # GNN architectures
│   │   ├── point_net_transf_gat.py  # Primary model (PointNet+Transformer+GAT)
│   │   ├── gat.py, gcn.py, ...      # Baseline architectures
│   │   └── block/, conv/            # Custom layers
│   ├── evaluation/                  # UQ analysis scripts
│   ├── training/                    # Model training pipeline
│   ├── data_preprocessing/          # MATSim → PyG graph conversion
│   └── misc/                        # Feature importance, utilities
│
├── run_part2_uq_analyses.py         # Selective prediction + error detection
├── run_part3_calibration_audit.py   # Calibration audit runner
├── run_part4_t7_crosscheck.py       # Trial 7 cross-replication
│
├── docs/                            # Documentation and verified results
├── environment-minimal.yml          # Conda environment (cross-platform)
├── traffic-gnn.yml                  # Conda environment (Linux, pinned)
└── README_SUBMISSION.md             # Detailed submission documentation
```

> **Note:** Training data (`data/`) is excluded from the repository due to size (~4.8 GB). All reported results can be reproduced from the included pre-computed prediction files.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Conda (Miniconda or Anaconda)
- LaTeX distribution with `pdflatex` and `biber` (for thesis compilation)

### Environment Setup

```bash
conda env create -f environment-minimal.yml
conda activate traffic-gnn
```

Core dependencies: PyTorch 2.2, PyTorch Geometric 2.5, NumPy, SciPy, scikit-learn, Matplotlib.

### Reproducing UQ Analyses

```bash
# Selective prediction + error detection (Trial 8)
python run_part2_uq_analyses.py

# Calibration audit (reliability diagram, temperature scaling, ECE)
python run_part3_calibration_audit.py

# Trial 7 cross-replication
python run_part4_t7_crosscheck.py
```

### Compiling the Thesis

```bash
cd thesis/latex_tum_official/
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Model Architecture

The primary surrogate is a **PointNet + Transformer + GAT** hybrid GNN (`scripts/gnn/models/point_net_transf_gat.py`):

```
Input (7 node features + 3D positional encoding)
  → Local MLP (7 → 256)
  → Global MLP (514 → 256, with max-pooled global context)
  → Transformer Encoder (4 heads, d_k = 64)
  → 4× GAT Layers (128 → 64, 4 heads each)
  → Output Head: GATConv(64 → 1)
```

MC Dropout (p = 0.2) is applied during both training and inference.

---

## Builds On

> Natterer, E. S., Rao, S. R., Tejada Lapuerta, A., Engelhardt, R., Horl, S., & Bogenberger, K. (2025). *Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis.* Transportation Research Part C: Emerging Technologies, 180, 105360.

This thesis takes the trained models from the above work as given and contributes the UQ framework, calibration analysis, and cross-replication study.

---

## License

This work is submitted as a Master's thesis at the Technical University of Munich. Contact the author for reuse permissions.
