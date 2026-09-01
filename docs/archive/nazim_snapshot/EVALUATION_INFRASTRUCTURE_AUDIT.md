# Evaluation Infrastructure Code Audit Report

**Thesis:** Uncertainty Quantification for GNN Surrogates of Agent-Based Transport Models  
**Author:** Mohd Zamin Quadri (M.Sc. Mathematics in Science and Engineering, TUM)  
**Audit Date:** 2026-03-26  

---

## Executive Summary

This report audits all 18 Python scripts in `scripts/evaluation/` and the 7 model definition files in `scripts/gnn/models/`. The audit covers: purpose, I/O, status classification, and code-level issues (bugs, hardcoded paths, incorrect logic).

**Key Findings:**
- **10 actionable issues** identified across the evaluation infrastructure
- **1 confirmed bug** in `eign.py` (dropout called as float, not as `nn.Dropout` layer)
- **1 conformal calibration concern** in `run_conformal_comparison.py` (simplified quantile vs standard formula)
- **Hardcoded Model 8 paths** in 6+ scripts (limits reusability)
- **2 near-duplicate scripts** (`comprehensive_uq_analysis.py` vs `_fast.py`) with maintenance risk
- **1 script uses simulated data** (`generate_thesis_charts.py`) — not real results

### Status Legend

| Status | Meaning |
|--------|---------|
| **FINAL** | Used to produce thesis figures/metrics. Critical path. |
| **SUPPLEMENTARY** | Supports analysis but not directly on critical path. |
| **OBSOLETE** | Superseded by another script or no longer needed. |
| **UTILITY** | Shared helper module imported by other scripts. |

---

## Section 1: Evaluation Scripts Audit

### 1.1 `run_mc_dropout_full.py` (328 lines)

**Status: FINAL**

**Purpose:** MC Dropout inference runner with per-graph checkpointing and resume capability. Runs N stochastic forward passes through PointNetTransfGAT for all test graphs, producing mean predictions, uncertainties (std), and targets.

**Inputs:**
- Model checkpoint: `data/TR-C_Benchmarks/<model_folder>/model.pt`
- Test data: `data/TR-C_Benchmarks/test_data/` (PyG graph objects)
- CLI args: `--model` (2-8), `--samples` (MC passes, default 30), `--cpu`

**Outputs:**
- Per-graph checkpoints: `<model_folder>/uq_results/mc_checkpoints/graph_<i>.npz`
- Final aggregated: `<model_folder>/uq_results/mc_dropout_full_<N>graphs_mc<S>.npz` (keys: `predictions`, `uncertainties`, `targets`)
- Metrics JSON: `<model_folder>/uq_results/mc_dropout_metrics.json`

**Issues:**
- None critical. Well-structured with argparse for all model numbers (2-8).
- Imports `mc_dropout_predict` from `gnn.help_functions` (correct shared implementation).

---

### 1.2 `run_deterministic_full.py` (144 lines)

**Status: FINAL**

**Purpose:** Deterministic inference runner (dropout OFF). Runs a single forward pass per test graph for baseline comparison.

**Inputs:**
- Model checkpoint: `data/TR-C_Benchmarks/<model_folder>/model.pt`
- Test data: `data/TR-C_Benchmarks/test_data/`
- CLI args: `--model` (2-8), `--cpu`

**Outputs:**
- `<model_folder>/uq_results/deterministic_full_<N>graphs.npz` (keys: `predictions`, `targets`)
- Metrics JSON: `<model_folder>/uq_results/deterministic_metrics.json`

**Issues:**
- **Duplicate print statements** on lines 140-141 (minor, cosmetic).
- `to_numpy()` and `r2_score()` are redefined locally instead of importing from helpers.

---

### 1.3 `conformal_from_mc.py` (77 lines)

**Status: FINAL**

**Purpose:** Computes global and adaptive (sigma-scaled) conformal prediction intervals from pre-computed MC Dropout outputs. Implements the standard split-conformal formula.

**Inputs:**
- `--mc_test_npz`: MC test NPZ (predictions, uncertainties, targets)
- `--mc_cal_npz` (optional): Separate calibration NPZ; falls back to 20% test split
- `--alpha`: Miscoverage level (default 0.1 = 90% coverage)

**Outputs:**
- Console output only (coverage, interval width, quantile values)

**Issues:**
- None. **Correct conformal quantile formula:** `q_level = ceil((n+1)*(1-alpha)) / n` (line 8). This is the standard split-conformal quantile.
- Fallback to test split for calibration has an appropriate warning message.

---

### 1.4 `run_conformal_comparison.py` (242 lines)

**Status: SUPPLEMENTARY**

**Purpose:** Compares conformal prediction intervals with MC Dropout intervals for Model 8 at multiple coverage levels (50%, 80%, 90%, 95%). Also compares with/without UQ metrics.

**Inputs:**
- Hardcoded: `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz`

**Outputs:**
- JSON: `<model_folder>/uq_results/conformal_uq_comparison.json`
- Console comparison table

**Issues:**
- **Conformal quantile concern (line 66):** Uses `np.quantile(cal_errors, alpha)` where `alpha` is the coverage level (0.50, 0.80, 0.90, 0.95). This computes the `alpha`-th quantile of calibration errors directly. This is a **simplified approach** — it does NOT use the standard conformal formula `ceil((n+1)*(1-alpha))/n` that `conformal_from_mc.py` correctly implements. For large calibration sets the difference is negligible, but it's technically not the finite-sample-valid conformal quantile.
- **Hardcoded to Model 8 only** — cannot process other models without code changes.
- Uses 50/50 cal/test split (vs 20/80 in `conformal_from_mc.py`).

---

### 1.5 `temperature_scaling_calibration.py` (453 lines)

**Status: FINAL**

**Purpose:** Post-hoc temperature scaling calibration for MC Dropout uncertainty. Learns a single scalar T on validation data to scale uncertainties (`calibrated_σ = raw_σ * T`), then evaluates ECE improvement on test data.

**Inputs:**
- Hardcoded: `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz`

**Outputs:**
- Figures to `thesis/latex_tum_official/figures/` (calibration before/after plots)
- JSON metrics: `<model_folder>/uq_results/temperature_scaling_results.json`

**Issues:**
- **Date in header says "February 2026"** (line 13) — likely a typo for February 2025.
- Hardcoded to Model 8 paths.
- Uses custom pastel color scheme (not TUM corporate colors like other thesis scripts).

---

### 1.6 `help_functions.py` (204 lines)

**Status: UTILITY**

**Purpose:** Shared helper module for evaluation scripts. Provides:
- `data_to_geodataframe_with_og_values()` — converts PyG data to GeoDataFrame for geo-plotting
- `create_test_data_objects()` — loads and preprocesses test graph data
- Road type index extraction for per-type analysis
- Validation metric computation

**Inputs:**
- `data/visualisation/districts_paris.geojson` (loaded at import time)
- Various data files accessed through provided paths

**Outputs:**
- Returns Python objects (no file I/O)

**Issues:**
- **Loads GeoJSON at module import time** (line 23) — will fail if geojson file doesn't exist, even if the importing script doesn't need geo data.
- Depends on `data_preprocessing.help_functions` and `data_preprocessing.process_simulations_for_gnn` — tight coupling to preprocessing modules.

---

### 1.7 `plot_functions.py` (727 lines)

**Status: UTILITY**

**Purpose:** Shared plotting library for geo-visualization, radar plots, and error scatter plots. Used by other evaluation scripts for consistent plotting style.

**Inputs:**
- GeoDataFrames and metric dictionaries passed as function arguments

**Outputs:**
- Matplotlib figures (saved by calling scripts)

**Issues:**
- Large file (727 lines) with many specialized plot functions — could benefit from being split into geo plots vs metric plots.

---

### 1.8 `plot_uq_mc_results.py` (103 lines)

**Status: OBSOLETE** (superseded by `plot_uq_standard.py` and `advanced_uq_analysis.py`)

**Purpose:** Basic MC Dropout result visualization: uncertainty histogram, uncertainty-vs-error scatter, binned uncertainty-error curve.

**Inputs:**
- `--npz`: Path to MC Dropout NPZ file

**Outputs:**
- `uq_plots/` subfolder with 4 PNG plots (uncertainty hist, scatter, binned curve, residual)

**Issues:**
- Minimal functionality. All plots are produced at higher quality by `plot_uq_standard.py` and `advanced_uq_analysis.py`.

---

### 1.9 `plot_uq_standard.py` (639 lines)

**Status: FINAL**

**Purpose:** Standardized 8-plot UQ visualization suite. Creates consistent plots across all models for thesis/publication: prediction scatter, uncertainty histogram, error histogram, uncertainty-vs-error scatter, binned analysis, coverage curves, coverage comparison bar chart, and 2x2 dashboard.

**Inputs:**
- `--model-dir`: Model directory path
- `--model-num`: Model number
- Reads `uq_results/mc_dropout_full_*.npz` and `uq_results/deterministic_full_*.npz`

**Outputs:**
- 8 PNG plots to `<model-dir>/uq_results/uq_plots_standard/`
- JSON summary: `uq_standard_metrics.json`

**Issues:**
- None critical. Well-parameterized with argparse.

---

### 1.10 `advanced_uq_analysis.py` (791 lines)

**Status: FINAL**

**Purpose:** Generates supervisor-approved key thesis visualizations: binned error vs uncertainty, risk-coverage curve, interval width comparison (absolute vs sigma-normalized), hexbin density plot. Demonstrates MC Dropout's value for ranking (Spearman ρ) vs raw calibration.

**Inputs:**
- `--all` flag to process all models, or defaults to Model 8
- Reads `uq_results/mc_dropout_full_*.npz` per model

**Outputs:**
- Thesis figures to `thesis/latex_tum_official/figures/` (binned error, risk-coverage, interval comparison, hexbin)
- JSON metrics per model

**Issues:**
- Model folder paths hardcoded in `MODEL_FOLDERS` dict (same pattern as inference scripts — acceptable).
- Uses sklearn metrics (`r2_score`, `mean_absolute_error`) — adds sklearn as a dependency.

---

### 1.11 `comprehensive_uq_analysis.py` (1013 lines)

**Status: SUPPLEMENTARY**

**Purpose:** Full UQ analysis with **live inference** — runs MC Dropout on-the-fly. Implements 4 analyses: threshold-based decision making, uncertainty heat maps, feature-wise error analysis, calibration curves.

**Inputs:**
- Model checkpoint: `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/model.pt`
- Test data loaded via `help_functions.create_test_data_objects()`

**Outputs:**
- Multiple figures to `thesis/latex_tum_official/figures/`

**Issues:**
- **Redefines `mc_dropout_predict()` locally** instead of importing from `gnn.help_functions` — maintenance risk if the canonical implementation is updated.
- Hardcoded to Model 8.
- Very large (1013 lines) — combines inference + analysis + plotting in one script.

---

### 1.12 `comprehensive_uq_analysis_fast.py` (803 lines)

**Status: SUPPLEMENTARY**

**Purpose:** Near-duplicate of `comprehensive_uq_analysis.py` but **loads pre-computed MC Dropout data** instead of running inference. Same 4 analyses.

**Inputs:**
- `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz`

**Outputs:**
- Same figures as `comprehensive_uq_analysis.py`

**Issues:**
- **Near-duplicate code** with `comprehensive_uq_analysis.py` — only difference is data loading. Should be consolidated into one script with a `--precomputed` flag.
- Hardcoded to Model 8.

---

### 1.13 `ensemble_uq_experiments.py` (842 lines)

**Status: FINAL**

**Purpose:** Two experiments:
- **Experiment A:** MC Dropout vs Ensemble Variance for Model 8 — runs multiple MC inference runs with different seeds, compares variance.
- **Experiment B:** Multi-model ensemble (Models 2, 5, 6, 7, 8) — weighted average predictions, ensemble uncertainty from cross-model variance.

**Inputs:**
- Model checkpoints for all ensemble models
- Test data via graph loading
- CLI: `--experiment A|B|both`, `--cpu`

**Outputs:**
- `<model_folder>/uq_results/ensemble_exp_a_results.json`
- `<model_folder>/uq_results/ensemble_exp_b_results.json`
- Multiple comparison plots (PNG)

**Issues:**
- **GATConv key remapping** (for PyG version compatibility): remaps `lin.weight` → `lin_src.weight` + `lin_dst.weight`. This is a workaround for PyG API changes and could break if PyG changes again.
- Loads with `strict=True` (line after remapping) — other scripts use `strict=False`.
- Defines its own `mc_dropout_predict_safe()` instead of importing from `gnn.help_functions`.

---

### 1.14 `generate_thesis_charts.py` (1026 lines)

**Status: OBSOLETE** (uses wrong output path and simulated data)

**Purpose:** Generates professional thesis charts with TUM corporate colors. Includes trial progression, model comparison, UQ scatter plots, and more.

**Inputs:**
- Hardcoded metric values in source code
- Some real JSON files for trial progression data

**Outputs:**
- Figures to `thesis/latex/figures` (NOT the correct `thesis/latex_tum_official/figures/`)

**Issues:**
- **Wrong output directory:** Uses `thesis\latex\figures` instead of `thesis\latex_tum_official\figures` used by all other scripts.
- **Uses SIMULATED data** for some scatter plots (e.g., generates random noise around simulated predictions) — not real model outputs.
- Hardcoded absolute path: `c:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models\thesis\latex\figures`
- Very large (1026 lines). Many chart functions may have been superseded by `advanced_uq_analysis.py` and `regenerate_thesis_plots.py`.

---

### 1.15 `generate_with_without_uq_plots.py` (469 lines)

**Status: FINAL**

**Purpose:** Generates deterministic vs MC Dropout comparison plots and markdown summaries. Auto-detects available NPZ files for Models 2, 5, 6, 7, 8.

**Inputs:**
- `<model_folder>/uq_results/deterministic_full_*.npz`
- `<model_folder>/uq_results/mc_dropout_full_*.npz`
- CLI: `--models` (optional list), `--base-dir`

**Outputs:**
- 7 PNG plots per model + 1 markdown summary (`WITH_WITHOUT_UQ_SUMMARY_MODEL<k>.md`)
- Output to `<model_folder>/uq_results/with_without_uq_plots/`

**Issues:**
- None critical. Well-parameterized.

---

### 1.16 `regenerate_thesis_plots.py` (386 lines)

**Status: FINAL**

**Purpose:** Regenerates ensemble experiment plots with TUM corporate colors. Publication-ready versions of the plots from `ensemble_uq_experiments.py`.

**Inputs:**
- `<model_folder>/uq_results/ensemble_exp_a_results.json`
- `<model_folder>/uq_results/ensemble_exp_b_results.json`

**Outputs:**
- Figures to `thesis/latex_tum_official/figures/` (ensemble comparison plots)

**Issues:**
- Hardcoded Model 8 folder path.

---

### 1.17 `audit_model8_uq_folder.py` (385 lines)

**Status: UTILITY**

**Purpose:** Integrity audit for Model 8's UQ results folder. Checks for: duplicate files, missing expected files, inconsistent metrics across JSON files, NPZ array validity.

**Inputs:**
- `data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/`

**Outputs:**
- Console report with pass/fail checks

**Issues:**
- Hardcoded to Model 8 only.
- Useful for pre-defense sanity check but single-use.

---

### 1.18 `__init__.py` (0 lines)

**Status: UTILITY** — Empty package init file. No issues.

---

### 1.19 Notebooks (not audited in detail)

| Notebook | Purpose |
|----------|---------|
| `in_depth_analysis.ipynb` | Interactive deep-dive analysis |
| `test_model.ipynb` | Model testing/debugging |
| `visualize_benchmarking.ipynb` | Benchmark visualization |

---

## Section 2: Model Architecture Audit

### 2.1 `PointNetTransfGAT` — `scripts/gnn/models/point_net_transf_gat.py` (256 lines)

**Status: Primary model for all thesis experiments (Trials 2-8)**

#### Architecture (with default parameters)

```
Input: x [N, 5], pos [N, 3, 2], edge_index [2, E]

1. PointNetConv_1 (uses pos[:, 0, :] = start position)
   ├─ Local MLP:  Linear(5+2=7, 256) → ReLU → [Dropout]
   └─ Global MLP: Linear(256, 512) → ReLU → [Dropout] → Linear(512, 512) → ReLU → [Dropout]

2. PointNetConv_2 (uses pos[:, 1, :] = end position)
   ├─ Local MLP:  Linear(512+2=514, 256) → ReLU → [Dropout]
   └─ Global MLP: Linear(256, 512) → ReLU → [Dropout] → Linear(512, 128) → ReLU → [Dropout]

3. TransformerConv(128, 256/4=64, heads=4) → ReLU → [Dropout]
   (output: 64*4 = 256)

4. TransformerConv(256, 512/4=128, heads=4) → ReLU → [Dropout]
   (output: 128*4 = 512)

5. GATConv(512, 64) — inside GeoSequential

6. GATConv(64, 1) — final output layer (self.gat_final)

Output: node_predictions [N, 1]
```

#### Default Hyperparameters

| Parameter | Default | Model 8 Override |
|-----------|---------|-----------------|
| `in_channels` | 5 | 5 |
| `out_channels` | 1 | 1 |
| `pnc_local` | [256] | [256] |
| `pnc_global` | [512] | [512] |
| `gat_conv` | [128, 256, 512] | [128, 256, 512] |
| `dropout` | 0.3 | **0.2** |
| `use_dropout` | False | **True** |
| `predict_mode_stats` | False | False |

#### Dropout Implementation

- **Conditional:** Only applied when `use_dropout=True` (constructor parameter).
- **Layer:** `self.dropout_layer = nn.Dropout(self.dropout)` created in `define_layers()` (line 75).
- **Placement:** After every ReLU in PointNet local/global MLPs and after each TransformerConv+ReLU pair.
- **MC Dropout:** At inference time, `model.train()` is called to keep dropout active (see `mc_dropout_predict()` in `gnn/help_functions.py:401`). BatchNorm layers (if any) are set to eval.
- **Shared dropout layer:** All dropout insertions reference the same `self.dropout_layer` object. This is fine for `nn.Dropout` since it has no learnable parameters, but it means all positions share the same dropout rate.

#### Layer Count vs Checkpoint

The default architecture with `use_dropout=True` creates the following named parameter groups:

| Component | Layers with Parameters | Param Count (approx) |
|-----------|----------------------|---------------------|
| PNC1 local MLP | Linear(7,256): weight+bias | 2 |
| PNC1 global MLP | Linear(256,512): w+b, Linear(512,512): w+b | 4 |
| PNC2 local MLP | Linear(514,256): w+b | 2 |
| PNC2 global MLP | Linear(256,512): w+b, Linear(512,128): w+b | 4 |
| TransformerConv1(128→64,h=4) | lin_src, lin_dst, lin_edge, bias, att params | ~8-10 |
| TransformerConv2(256→128,h=4) | lin_src, lin_dst, lin_edge, bias, att params | ~8-10 |
| GATConv(512,64) in GeoSequential | lin_src, lin_dst, att_src, att_dst, bias | ~5 |
| GATConv(64,1) final | lin_src, lin_dst, att_src, att_dst, bias | ~5 |

**Estimated total: ~34-36 named parameter tensors in state_dict.** This is consistent with checkpoints that show 34-36 keys, confirming the architecture matches the saved model files.

#### Weight Initialization

| Layer Type | Method |
|-----------|--------|
| `nn.Linear` | Kaiming Normal (fan_out, relu) — from `BaseGNN.initialize_weights()` |
| `PointNetConv` | Kaiming Normal for weight params, zeros for bias — custom `_initialize_pointnetconv()` |
| `GATConv` | Xavier Normal for `lin.weight`, `att_src`, `att_dst`; zeros for bias — custom `_initialize_gatconv()` |

---

### 2.2 `BaseGNN` — `scripts/gnn/models/base_gnn.py` (273 lines)

**Purpose:** Abstract base class providing training loop, validation, mixed-precision support, and weight initialization.

**Key members:**
- `self.dropout = dropout` (float) — stored as attribute, NOT as `nn.Dropout`
- `self.use_dropout` (bool) — flag for conditional dropout
- `train_model()` — full training loop with early stopping, LR scheduling, WandB logging
- `initialize_weights()` — Kaiming Normal for all `nn.Linear` modules

**Issues:**
- **Deprecated PyTorch API** (line 11): `from torch.cuda.amp import GradScaler, autocast` — in PyTorch 2.x, these should be `torch.amp.GradScaler` and `torch.amp.autocast`. Still works but triggers deprecation warnings.
- `self.dropout` is a **float attribute**, not an `nn.Module`. This is fine for `PointNetTransfGAT` (which creates its own `nn.Dropout` layer), but it causes a bug in `eign.py` (see below).

---

### 2.3 `EIGN` — `scripts/gnn/models/eign.py` (413 lines)

**Purpose:** Equivariant Interaction Graph Network. Alternative architecture (not used in main thesis experiments).

**Bug (CONFIRMED):**
- **Line 168:** `x_signed = self.dropout(x_signed)`
- **Line 171:** `x_unsigned = self.dropout(x_unsigned)`
- `self.dropout` is inherited from `BaseGNN.__init__()` as `self.dropout = dropout` which is a **float** (0.3), NOT a callable `nn.Dropout` layer.
- Calling `self.dropout(x_signed)` where `self.dropout = 0.3` will attempt to call `float.__call__()` which raises `TypeError: 'float' object is not callable`.
- **Fix:** Replace with `nn.Dropout(self.dropout)(x)` or define `self.dropout_layer = nn.Dropout(dropout)` in `__init__` (following the pattern from `PointNetTransfGAT`).
- **Impact:** Low — EIGN is not used in any thesis evaluation script, but should be fixed for code correctness.

---

### 2.4 Other Model Files

| File | Lines | Purpose | Issues |
|------|-------|---------|--------|
| `gat.py` | 103 | GAT-only model (ablation) | None |
| `pnc.py` | 166 | PointNetConv-only model (ablation) | None |
| `trans_conv.py` | 127 | TransformerConv-only model (ablation) | None |
| `fc_nn.py` | 113 | Fully-connected baseline | None |
| `__init__.py` | 0 | Empty package init | None |

---

## Section 3: Cross-Cutting Issues Summary

### Issue Priority Table

| # | Severity | File | Issue | Line(s) |
|---|----------|------|-------|---------|
| 1 | **BUG** | `eign.py` | `self.dropout` is float, called as function | 168, 171 |
| 2 | **WARN** | `run_conformal_comparison.py` | Simplified quantile formula (not finite-sample valid) | 66 |
| 3 | **WARN** | `comprehensive_uq_analysis.py` | Redefines `mc_dropout_predict()` locally | — |
| 4 | **WARN** | `ensemble_uq_experiments.py` | Redefines `mc_dropout_predict_safe()` locally | 43 |
| 5 | **INFO** | `generate_thesis_charts.py` | Wrong output dir + simulated data | 47 |
| 6 | **INFO** | `temperature_scaling_calibration.py` | Date says "February 2026" | 13 |
| 7 | **INFO** | `base_gnn.py` | Deprecated `torch.cuda.amp` imports | 11 |
| 8 | **INFO** | `run_deterministic_full.py` | Duplicate print statements | 140-141 |
| 9 | **INFO** | `comprehensive_uq_analysis{,_fast}.py` | Near-duplicate scripts | — |
| 10 | **INFO** | 6+ scripts | Hardcoded Model 8 paths | Various |

### Dependency Graph (key imports)

```
run_mc_dropout_full.py ──→ gnn.help_functions.mc_dropout_predict
                       ──→ gnn.models.point_net_transf_gat.PointNetTransfGAT

run_deterministic_full.py ──→ gnn.models.point_net_transf_gat.PointNetTransfGAT

conformal_from_mc.py ──→ (standalone, numpy only)

comprehensive_uq_analysis.py ──→ gnn.models.point_net_transf_gat.PointNetTransfGAT
                              ──→ (local mc_dropout_predict — NOT imported)

ensemble_uq_experiments.py ──→ gnn.models.point_net_transf_gat.PointNetTransfGAT
                           ──→ (local mc_dropout_predict_safe — NOT imported)

help_functions.py ──→ data_preprocessing.help_functions
                  ──→ data_preprocessing.process_simulations_for_gnn
                  ──→ gnn.help_functions

PointNetTransfGAT ──→ BaseGNN (base_gnn.py)
```

---

## Section 4: Recommendations

1. **Fix `eign.py` dropout bug** — Replace `self.dropout(x)` with `self.dropout_layer(x)` after defining `self.dropout_layer = nn.Dropout(self.dropout)`.

2. **Consolidate MC Dropout predict** — Remove local `mc_dropout_predict` / `mc_dropout_predict_safe` definitions from `comprehensive_uq_analysis.py` and `ensemble_uq_experiments.py`; import from `gnn.help_functions` instead.

3. **Merge `comprehensive_uq_analysis.py` and `_fast.py`** — Add a `--precomputed` flag to one script.

4. **Mark `generate_thesis_charts.py` as deprecated** — It outputs to the wrong directory and uses simulated data. Ensure no thesis figures come from this script.

5. **Update `base_gnn.py` imports** — Replace `from torch.cuda.amp import GradScaler, autocast` with `from torch.amp import GradScaler, autocast` for PyTorch 2.x compatibility.

6. **Parameterize Model 8 hardcoding** — In scripts like `temperature_scaling_calibration.py`, `run_conformal_comparison.py`, etc., add `--model` CLI args.

---

*End of audit report.*
