# Code — UQ for ML Models in Transportation Policy Analysis

Code and experiments for the Master's thesis **"Uncertainty Quantification for Machine Learning Models in Transportation Policy Analysis"** (TUM, submitted May 15, 2026, author: Mohd Zamin Quadri).

The immutable submitted thesis (LaTeX + compiled PDF) lives in [`../document/`](../document/).
`UQ_SUMMARY.md` is a submission-era summary. The post-submission source of record is
[`../docs/CORRIGENDUM.md`](../docs/CORRIGENDUM.md) with its safe aggregate evidence in
[`../analysis_outputs/`](../analysis_outputs/).

---

## What this code does

1. **Preprocessing** — converts raw MATSim simulation output into PyTorch Geometric graph batches (line-graph: road segments = nodes). Corpus: 10,000 Paris-scale scenarios from Natterer et al. (2025); this thesis uses a fixed 1,000-scenario subset across all eleven trials.
2. **Training** — trains the PointNetTransfGAT surrogate family (base trials T1–T8, Deep Ensemble seeds, heteroscedastic head T9, CQR heads T10/T11).
3. **UQ evaluation** — MC Dropout, regression σ-scaling (T\* = 2.887), split/adaptive conformal prediction, proper scoring rules (CRPS/PIT/Winkler), selective prediction, error detection AUROC, stratified |Δv| analysis — mostly via the Colab notebooks below.

## Headline results (submitted thesis values)

| Analysis | Value |
| -------- | ----- |
| T8 base: R² / MAE / RMSE | 0.5957 / 3.957 / 7.118 veh/h (100 test graphs, 3,163,500 nodes) |
| MC Dropout (S=30) Spearman ρ | 0.4820 |
| Raw calibration k₉₅ | 11.66 (Gaussian ideal 1.96) |
| σ-scaling: ECE | 0.356 → 0.034 (−90.5%), T\* = 2.887 |
| Split conformal PICP₉₀ / PICP₉₅ | 90.02% / 95.01% |
| Adaptive conformal conditional coverage | [59.0%, 98.1%] → [83.7%, 96.4%] |
| Selective prediction (50% retention) | MAE −41.2% → 2.32 veh/h |
| Error detection AUROC (top-10%) | 0.7548 |
| Deep Ensemble (5 members) | R² = 0.6841, ρ = 0.3997 |
| T11 (frozen-backbone CQR) | R² = 0.5835 — passes all six gates |

See `UQ_SUMMARY.md` for the full tables incl. T9/T10 gate checks, T7 cross-replication, and the JSON source for every number.

## Repository Structure

```
scripts/
  gnn/
    models/            BaseGNN, PointNetTransfGAT, frozen heteroscedastic/CQR variants
    losses/            Heteroscedastic NLL (Kendall & Gal + Seitzer regulariser), quantile (pinball) loss
    gnn_io.py, help_functions.py, heteroscedastic_mc_dropout.py
  training/            run_models.py/.ipynb, run_deep_ensemble.py, train_heteroscedastic.py, train_cqr*.py
  data_preprocessing/  process_simulations_for_gnn.py (MATSim -> PyG batches)
  misc/                Figure generation (gen_batch*.py), consistency checks
colab_*.ipynb          UQ master notebook, σ-scaling, ensembles, RF baseline
generate_thesis_figures.py
  environment-minimal.yml / traffic-gnn.yml
docs/                  Script-level docs: data_preprocessing.md, gnn.md, training.md
UQ_SUMMARY.md          Submission-era results summary with a corrigendum banner
```

> Training data, model checkpoints, and row-level prediction arrays are **not** committed.
> A post-submission full-array audit produced the aggregate evidence in `../analysis_outputs/`
> and identified the corrections and replay limitations in `../docs/CORRIGENDUM.md`.

## Setup

```bash
conda env create -f environment-minimal.yml
conda activate traffic-gnn
```

Main entry points:
- Train base trials: `python scripts/training/run_models.py --help` (see `docs/training.md`)
- Deep Ensemble: `scripts/training/run_deep_ensemble.py`
- Heteroscedastic / CQR heads: `scripts/training/train_heteroscedastic.py`, `train_cqr.py`, `train_cqr_frozen.py`
- Post-hoc UQ analyses: `colab_uq_master.ipynb` (MC Dropout, conformal, selective prediction, AUROC)

## Builds On

> Natterer et al. (2025). *Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis.* Transportation Research Part C, 180, 105360.

The preprocessing/training scaffold is adapted from that work; this thesis contributes the UQ framework, calibration analyses, uncertainty-aware training extensions, and the audit/verification artefacts.

## License

Submitted as a Master's thesis at the Technical University of Munich. Contact the author for reuse permissions.
