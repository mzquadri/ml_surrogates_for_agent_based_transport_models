# REMAINING_WORK_AND_ACTION_PLAN.md
# What Is Left to Do Before Thesis Submission
# Last verified: 2026-03-14

---

## Current Status

**What is done (verified this session):**
- All 8 training trials evaluated and verified from JSON
- UQ results (MC Dropout, conformal, ensemble) verified from JSON
- 23 documentation files written to docs/verified/
- All known wrong values in old docs identified and flagged

**What is NOT done:**
The following are the remaining tasks before the thesis is complete.

---

## Priority 1: MUST DO (Thesis Cannot Be Submitted Without These)

### 1.1 Write the Thesis Document

The actual thesis LaTeX/Word document has not been written in this session.
All the raw material exists in docs/verified/ — it now needs to be structured into:

```
Chapter 1: Introduction
  - Problem: MATSim simulation cost
  - Research question: Can a GNN surrogate + UQ replace expensive simulation?
  - Thesis structure overview

Chapter 2: Background
  - MATSim and agent-based simulation
  - Graph Neural Networks (PointNet, TransformerConv, GAT, LineGraph)
  - Uncertainty Quantification (MC Dropout, conformal prediction, ensembles)
  - Paris road network and transport policies

Chapter 3: Methodology
  - Data pipeline (preprocessing → LineGraph → PyG objects)
  - PointNetTransfGAT architecture (verified from point_net_transf_gat.py)
  - Training setup (T2–T8, verified hyperparameters from JSON)
  - UQ methods: MC Dropout, conformal, ensemble

Chapter 4: Experiments
  - Performance evaluation (R², MAE, RMSE — all 7 comparable trials)
  - Ablation analysis (weighted loss, batch size, LR, dropout, split)
  - UQ evaluation (Spearman ρ for all trials)
  - Conformal prediction (90%, 95% coverage)
  - Ensemble comparison (Experiment A and B)
  - Selective prediction (39.9% MAE reduction)

Chapter 5: Discussion
  - What the R²=0.5957 means in practice
  - Why MC Dropout outperforms ensemble here
  - Limitations: data size, no OOD detection, σ not calibrated
  - Future work: more data, heteroscedastic models, EBMs, explicit OOD

Chapter 6: Conclusion
  - Summary of contributions
  - Practical recommendation: T8 + MC Dropout + conformal for deployment
```

**Reference files for each chapter:**
- Ch 1: THESIS_STORY_FROM_ZERO.md
- Ch 2: DATA_EXPLAINED_FROM_ZERO.md, FEATURES_EXPLAINED_FROM_ZERO.md, GRAPH_EXPLANATION_GUIDE.md, UQ_WORKFLOW_EXPLAINED.md
- Ch 3: INPUT_OUTPUT_PIPELINE_EXPLAINED.md, NODE_VS_GRAPH_LEVEL_EXPLAINED.md, REPO_OVERVIEW_VERIFIED.md
- Ch 4: VERIFIED_RESULTS_MASTER.csv, CHEAT_SHEET.md
- Ch 5/6: THESIS_STORY_FROM_ZERO.md, MORNING_EVENING_NIGHT_STATUS.md

---

### 1.2 Generate Thesis Figures

Run `docs/verified/figures/generate_verified_figures.py` to produce:
- fig1_trial_comparison.pdf
- fig2_uq_ranking.pdf
- fig3_conformal_coverage.pdf
- fig4_selective_prediction.pdf
- fig5_feature_correlation.pdf
- fig7_calibration.pdf

See FIGURE_REGENERATION_PLAN.md for full spec.

**Estimated time:** 2.5 hours

---

### 1.3 Fix Wrong Files in Repo

From CLEANUP_PLAN.md:
```
[ ] Add WARNING comment to scripts/evaluation/generate_thesis_charts.py
[ ] Move docs/MEETING_PREPARATION.md → docs/archive/
[ ] Move T1 folder to archive (excluded trial)
```

**Estimated time:** 15 minutes

---

## Priority 2: SHOULD DO (Strong Recommendation)

### 2.1 Write Root README.md

The repo has no README. Before submission, write a root README that:
- States the thesis title and author
- Describes the project
- Lists the verified results (T8: R²=0.5957, MAE=3.96, ρ=0.4820)
- Explains how to reproduce: Colab runbook link
- Lists the key files

**Estimated time:** 30 minutes

### 2.2 Verify requirements.txt / environment.yml

Confirm the environment file accurately reflects the PyTorch, PyG, and other
package versions used in training. This is critical for reproducibility.

**Estimated time:** 30 minutes

### 2.3 Check for Full Selective Prediction Curve Data

Check `uq_comparison_model8.json` for a full curve of MAE vs % data retained.
If available, Figure 4 can show the complete curve instead of just the 50% point.

**Estimated time:** 15 minutes

---

## Priority 3: NICE TO HAVE

### 3.1 Baseline Comparison (MLP without graph structure)

If time permits, running a simple MLP baseline (treating each node independently)
would strengthen the argument for GNN. If not run, acknowledge this limitation
explicitly in Chapter 5.

### 3.2 More MC Dropout Samples

T=30 samples was used. Running T=100 or T=50 would give smoother uncertainty
estimates. Not required — 30 samples is standard in the literature.

### 3.3 Full Feature Ablation

Testing with 3 or 4 features (removing LENGTH or FREESPEED) could show which
features are truly necessary. Not required but would strengthen Chapter 4.

---

## Summary Timeline to Submission

| Task | Hours | Priority |
|------|-------|----------|
| Write thesis (Chapters 1-6) | 40–60 | MUST |
| Generate figures | 2.5 | MUST |
| Fix wrong files in repo | 0.25 | MUST |
| Write README.md | 0.5 | SHOULD |
| Verify requirements.txt | 0.5 | SHOULD |
| Check selective prediction curve | 0.25 | SHOULD |
| Baseline MLP experiment | 4–8 | NICE |
| **Total (must + should)** | **~45–65 hours** | |

---

## What is FULLY DONE — Do Not Redo

- All metric verification (JSON → docs/verified/)
- All 23 documentation files in docs/verified/
- All figure specifications (FIGURE_REGENERATION_PLAN.md)
- All wrong-value identification (OLD_CLAIMS_AUDIT.md)
- Meeting preparation materials (MEETING_GUIDED_NOTES_HINGLISH.md, SUPERVISOR_CONTEXT_NOTES.md)

**Do NOT re-verify metrics from scratch. Trust docs/verified/VERIFIED_RESULTS_MASTER.csv.**

---

## The Single Most Important Next Action

**Open your thesis document and write Chapter 4 (Experiments) first.**
It is the most factual chapter — just transcribe numbers from VERIFIED_RESULTS_MASTER.csv
and CHEAT_SHEET.md into proper thesis prose. Then write Chapter 3 (Methodology),
which you now understand deeply from the GRAPH_EXPLANATION_GUIDE.md and
INPUT_OUTPUT_PIPELINE_EXPLAINED.md.

Leave Chapter 1 (Introduction) and Chapter 5 (Discussion) for last — they are
easier to write once you have the factual chapters done.
