# CLEANUP_PLAN.md
# Repository Cleanup Plan — What to Keep, Archive, or Delete
# Last verified: 2026-03-14

---

## Purpose

This document tells you exactly what to do with every file/folder in the repo
before thesis submission. The goal: a clean, reproducible, thesis-ready repository.

---

## Priority Legend

- **KEEP** — essential for reproducibility or thesis defense
- **ARCHIVE** — move to `archive/` subfolder; keep but don't show by default
- **FIX** — needs correction before submission
- **DELETE** — safe to remove (redundant, wrong, or temporary)
- **WRITE** — needs to be created

---

## Section 1: docs/

| File | Action | Reason |
|------|--------|--------|
| `docs/MEETING_PREPARATION.md` | **FIX or ARCHIVE** | All hyperparameters wrong vs JSON. Fix before sharing with supervisor, or move to archive/. |
| `docs/verified/` (all 23 files) | **KEEP** | Ground-truth verified outputs — authoritative for thesis |
| `docs/verified/figures/generate_verified_figures.py` | **KEEP** | Only correct figure generation script |

**Action:** Move `docs/MEETING_PREPARATION.md` → `docs/archive/MEETING_PREPARATION_OUTDATED.md`

---

## Section 2: scripts/evaluation/

| File | Action | Reason |
|------|--------|--------|
| `generate_thesis_charts.py` | **FIX or ARCHIVE** | Hardcodes wrong test_r2 values. Do not use for thesis figures. |
| `ensemble_uq_experiments.py` | **KEEP** | Used to produce experiment_a/b results; needed for reproducibility |
| Other evaluation scripts | **KEEP** | Part of reproducible pipeline |

**Action:** Add a comment at top of `generate_thesis_charts.py`:
```python
# WARNING: This script contains hardcoded metric values that do not match
# the verified JSON outputs. Use docs/verified/figures/generate_verified_figures.py instead.
```

---

## Section 3: scripts/gnn/models/

| File | Action | Reason |
|------|--------|--------|
| `point_net_transf_gat.py` | **KEEP** | Verified architecture — core contribution |
| Any other model files | **KEEP** | Part of model comparison history |

---

## Section 4: scripts/data_preprocessing/

| File | Action | Reason |
|------|--------|--------|
| `process_simulations_for_gnn.py` | **KEEP** | Defines LineGraph transform + graph construction |
| `help_functions.py` | **KEEP** | Defines target variable (line 121-123); verified |

---

## Section 5: scripts/training/

| File | Action | Reason |
|------|--------|--------|
| `run_models.py` | **KEEP** | Training loop; verified hyperparams match JSON |

---

## Section 6: data/TR-C_Benchmarks/

### Trials to Keep vs Archive

| Trial Folder | Action | Reason |
|---|---|---|
| `pointnet_transf_gat_1st_bs32_5feat_seed42/` | **ARCHIVE** | T1 excluded — wrong architecture |
| `point_net_transf_gat_2nd_try/` | **KEEP** | T2 — part of ablation story |
| `point_net_transf_gat_3rd_trial_weighted_loss/` | **KEEP** | T3 — weighted loss ablation |
| `point_net_transf_gat_4th_trial_weighted_loss/` | **KEEP** | T4 — weighted loss ablation |
| `point_net_transf_gat_5th_try/` | **KEEP** | T5 — UQ results needed |
| `point_net_transf_gat_6th_trial_lower_lr/` | **KEEP** | T6 — LR ablation |
| `point_net_transf_gat_7th_trial_80_10_10_split/` | **KEEP** | T7 — split ablation |
| `point_net_transf_gat_8th_trial_lower_dropout/` | **KEEP** | T8 — BEST MODEL, all UQ results |

### Within T8 uq_results/ — Keep Everything

| File | Action | Reason |
|---|---|---|
| `mc_dropout_full_metrics_model8_mc30_100graphs.json` | **KEEP** | Primary UQ result ρ=0.4820 |
| `conformal_standard.json` | **KEEP** | Conformal prediction results |
| `uq_comparison_model8.json` | **KEEP** | k95, 65.8%, 39.9% selective prediction |
| `ADVANCED_UQ_SUMMARY_MODEL8.md` | **KEEP** | Human-readable UQ narrative |
| `ensemble_experiments/experiment_a_results.json` | **KEEP** | Ensemble baseline comparison |
| `ensemble_experiments/experiment_b_results.json` | **KEEP** | Multi-model ensemble comparison |

---

## Section 7: Temporary / Output Files to Clean

| Pattern | Action |
|---------|--------|
| `*.pyc` / `__pycache__/` | **DELETE** |
| `wandb/` run folders | **ARCHIVE** (keep if needed for training logs) |
| Duplicate `.json` files with identical content | **DELETE** (after verifying) |
| Large intermediate `.npz` files | **KEEP** if used by figure scripts; **DELETE** otherwise |
| Any `.ipynb_checkpoints/` | **DELETE** |

---

## Section 8: Files to CREATE Before Submission

| File | Priority | Notes |
|------|----------|-------|
| `README.md` (root) | HIGH | Describe project, how to run, verified results |
| `requirements.txt` or `environment.yml` | HIGH | Reproducibility |
| `docs/verified/figures/generate_verified_figures.py` | HIGH | Already planned — correct figure script |

---

## Step-by-Step Cleanup Checklist

```
[ ] 1. Move docs/MEETING_PREPARATION.md → docs/archive/MEETING_PREPARATION_OUTDATED.md
[ ] 2. Add WARNING comment to scripts/evaluation/generate_thesis_charts.py
[ ] 3. Move T1 folder to data/TR-C_Benchmarks/archive/ (excluded from results)
[ ] 4. Delete __pycache__, .pyc, .ipynb_checkpoints
[ ] 5. Verify requirements.txt / environment.yml exists and is accurate
[ ] 6. Write root README.md with verified metrics
[ ] 7. Confirm all 23 docs/verified/ files are present and complete
[ ] 8. Final check: do any remaining scripts reference wrong metrics?
```

---

## What NEVER to Delete

- Any `test_evaluation_complete.json` or `test_results.json` — ground truth
- Any file under `data/TR-C_Benchmarks/*/uq_results/` — UQ ground truth
- `scripts/gnn/models/point_net_transf_gat.py` — architecture definition
- `scripts/data_preprocessing/help_functions.py` — target variable definition
- Anything in `docs/verified/` — all verified outputs

---

## Estimated Time to Complete Cleanup

| Task | Time |
|------|------|
| Move/archive wrong files | 10 min |
| Add warning comments | 5 min |
| Delete temp files | 5 min |
| Write README.md | 30 min |
| Verify requirements.txt | 15 min |
| **Total** | **~65 min** |
