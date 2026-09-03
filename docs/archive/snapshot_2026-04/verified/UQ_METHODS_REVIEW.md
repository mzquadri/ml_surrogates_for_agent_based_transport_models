# UQ Methods Review — Phase 1
## Literature-Backed Assessment of Feasible Remaining Analyses

- **Thesis:** "Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models"
- **Author:** Mohd Zamin Quadri
- **Review date:** 2026-03-18
- **Purpose:** Phase 1 — For each feasible remaining analysis identified in the Phase 0 audit, provide: (a) the research-paper foundation, (b) what the analysis would add to the thesis, (c) which citations are already in the bibliography vs need to be added.
- **Scope:** Audit-only assessment. No execution.

---

## 1. Currently Cited UQ Literature (bibliography.bib, 53 entries)

The thesis bibliography already covers these UQ-relevant areas:

| Area | Key Citations | Sufficient? |
|------|--------------|-------------|
| MC Dropout theory | Gal & Ghahramani (2016, ICML) | YES |
| Deep ensembles | Lakshminarayanan et al. (2017, NeurIPS) | YES |
| Conformal prediction | Angelopoulos & Bates (2023, FnTML); Vovk et al. (2005, book); Romano et al. (2019, NeurIPS) | YES |
| Calibration of neural networks | Guo et al. (2017, ICML); Kuleshov et al. (2018, ICML) | YES |
| Bayesian neural networks | MacKay (1992); Neal (1996); Blundell et al. (2015, ICML); Wilson & Izmailov (2020, NeurIPS) | YES |
| Aleatoric vs epistemic | Kendall & Gal (2017, NeurIPS); Hullermeier & Waegeman (2021) | YES |
| UQ surveys | Abdar et al. (2021); Gawlikowski et al. (2023); Psaros et al. (2023) | YES |
| UQ for GNNs | Zhang et al. (2019, AAAI); Hasanzadeh et al. (2020, ICML) | YES |
| UQ for traffic | Wang et al. (2023, TR-C) | YES (only one, but thesis discusses why the setting differs) |

**Assessment:** The bibliography is already strong. New analyses would require at most 3-5 additional citations for specific methods (reliability diagrams, conditional conformal, selective prediction formalization).

---

## 2. Analysis-by-Analysis Literature Review

### 2.1 Reliability Diagram (Calibration Plot)

**What it is:** A plot of expected coverage vs observed coverage across multiple confidence levels. For a regression model producing uncertainty sigma, bin predictions by predicted confidence level, then check whether the observed coverage in each bin matches the predicted coverage.

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| Guo et al. (2017), "On Calibration of Modern Neural Networks," ICML | YES (`guo2017calibration`) | Introduced reliability diagrams and Expected Calibration Error (ECE) for classification; the regression analogue follows directly |
| Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression," ICML | YES (`kuleshov2018accurate`) | **PRIMARY REFERENCE** — Defines reliability diagrams for regression: plot predicted quantile level p vs fraction of observations falling below that quantile. Proposes recalibration via isotonic regression. |
| Naeini et al. (2015), "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles," AAAI | NO — NEEDS ADDING | Introduced the binning approach for calibration assessment |

**What it adds to thesis:**
- Visual complement to the k95 = 11.34 finding — shows miscalibration across all confidence levels, not just one point
- Standard UQ evaluation that reviewers expect to see
- Already discussed textually in Chapter 2 (Section 2.4.5) and in the calibration audit tables — a figure would complete the picture

**Implementation from existing data:**
- Source: `trial8_uq_ablation_results.csv` columns `pred_mc_mean`, `pred_mc_std`, `target`
- For each nominal level p in [0.1, 0.2, ..., 0.9, 0.95], compute fraction of test nodes where |target - pred_mc_mean| <= z_p * pred_mc_std
- Plot observed fraction vs nominal level
- No new inference needed

**New bib entry needed?** Optional (Naeini 2015). Kuleshov 2018 already covers regression reliability diagrams.

**Verdict: HIGH VALUE, LOW EFFORT. Proceed in Phase 3.**

---

### 2.2 Stratified UQ Analysis by Feature Bins

**What it is:** Break down uncertainty quality (Spearman rho, MAE, coverage) by subgroups of the input feature space. For example: do high-volume roads have better or worse uncertainty ranking than low-volume roads?

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| Ovadia et al. (2019), "Can You Trust Your Model's Uncertainty?", NeurIPS | YES (`ovadia2019can`) | Evaluates UQ quality under dataset shift — stratification by input features is the non-shift version of this |
| Kendall & Gal (2017), "What Uncertainties Do We Need in Bayesian Deep Learning?", NeurIPS | YES (`kendall2017uncertainties`) | Motivates input-dependent uncertainty analysis; discusses how aleatoric uncertainty varies with input |
| Barber et al. (2021), "The Limits of Distribution-Free Conditional Predictive Inference," Information and Inference | NO — NEEDS ADDING | **KEY REFERENCE** — Proves that marginal coverage guarantees (like conformal prediction) do NOT imply conditional coverage for subgroups. Directly motivates stratified coverage analysis. |

**What it adds to thesis:**
- Explains WHERE the model is uncertain, not just HOW uncertain — transforms abstract rho into actionable deployment guidance
- Connects to the feature correlation analysis already in Section 5.11 (VOL rho=+0.332 etc.)
- If high-volume roads have worse rho AND higher sigma, that confirms the model "knows what it doesn't know" for the hardest subgroup
- Addresses the known limitation that conformal coverage is marginal, not conditional

**Implementation from existing data:**
- Source: `trial8_uq_ablation_results.csv` (3.16M rows) + `test_dl.pt` (feature tensor, cols [0,1,2,3,5])
- Bin nodes by feature quartiles (Q1-Q4 of volume, capacity, speed, etc.)
- Compute per-bin: Spearman rho, MAE, conformal coverage, mean sigma
- No new inference needed

**New bib entries needed:**
1. Barber et al. (2021) — conditional coverage impossibility result
2. Optionally: Romano et al. (2020), "Classification with Valid and Adaptive Coverage," NeurIPS — for adaptive conformal context

**Verdict: HIGH VALUE, MEDIUM EFFORT. Proceed in Phase 3.**

---

### 2.3 Per-Graph Uncertainty Variation

**What it is:** Compute graph-level (scenario-level) statistics: per-graph Spearman rho, per-graph MAE, per-graph mean sigma, per-graph conformal coverage. Report distribution across the 100 test graphs.

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| Angelopoulos & Bates (2023), "Conformal Prediction: A Gentle Introduction" | YES (`angelopoulos2023conformal`) | Discusses that marginal coverage holds over the test distribution, but individual-instance coverage can vary |
| Psaros et al. (2023), "UQ in Scientific ML" | YES (`psaros2023uq`) | Reviews per-instance vs aggregate UQ metrics |

**What it adds to thesis:**
- Shows whether UQ quality is stable across scenarios or driven by a few outlier graphs
- If some graphs have rho > 0.6 and others rho < 0.2, that's an important finding for deployment
- Provides error bars / confidence intervals for the aggregate rho = 0.4820

**Implementation from existing data:**
- Source: `checkpoints_mc30/graph_0000.npz` ... `graph_0099.npz` (100 files, each with predictions/uncertainties/targets arrays of shape (31635,))
- For each graph: compute Spearman rho, MAE, coverage at 90%/95%
- Report: mean, std, min, max, histogram
- No new inference needed

**New bib entries needed?** None — existing citations sufficient.

**Verdict: MEDIUM VALUE, LOW EFFORT. Proceed in Phase 3.**

---

### 2.4 Conformal Conditional Coverage

**What it is:** Measure conformal prediction coverage WITHIN subgroups rather than marginally over the full test set. The global conformal coverage of 90.02% may hide significant variation: some subgroups may be at 85%, others at 95%.

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| Barber et al. (2021), "The Limits of Distribution-Free Conditional Predictive Inference," Information and Inference | NO — NEEDS ADDING | **CRITICAL** — Proves that no distribution-free method can guarantee conditional coverage without assumptions. This is the theoretical motivation for checking whether conditional coverage holds empirically. |
| Romano et al. (2019), "Conformalized Quantile Regression," NeurIPS | YES (`romano2019conformalized`) | CQR provides tighter, more adaptive intervals; relevant for comparison |
| Angelopoulos & Bates (2023) | YES (`angelopoulos2023conformal`) | Discusses conditional coverage limitations in detail |
| Vovk (2012), "Conditional Validity of Inductive Conformal Predictors," AMLC | NO — optional | Formal conditional validity framework |

**What it adds to thesis:**
- Directly addresses the "marginal vs conditional" limitation stated in Chapter 2 (line 134: "the coverage guarantee is marginal rather than conditional")
- If coverage is roughly uniform across subgroups, that strengthens the conformal result
- If coverage varies, that's an honest finding — still valuable, and the thesis already frames conformal as a marginal guarantee

**Implementation from existing data:**
- Source: `trial8_uq_ablation_results.csv` + `test_dl.pt` features
- Using the 20/80 calibration/evaluation split from the calibration audit
- Compute conformal quantile q on calibration set, then measure coverage on evaluation set stratified by sigma quartile and feature quartile
- No new inference needed

**New bib entries needed:**
1. Barber et al. (2021) — same as 2.2 above (shared dependency)

**Verdict: HIGH VALUE, MEDIUM EFFORT. Proceed in Phase 3. Can share implementation with 2.2.**

---

### 2.5 MC Sample Count Ablation (S = 5, 10, 20, 30)

**What it is:** Study how the number of MC Dropout forward passes affects uncertainty quality (Spearman rho) and prediction accuracy (MAE of the MC mean). Standard ablation in any MC Dropout paper.

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation," ICML | YES (`gal2016dropout`) | Original MC Dropout paper; discusses convergence of posterior approximation with T samples |
| Gal (2016), "Uncertainty in Deep Learning," PhD thesis, University of Cambridge | NO — NEEDS ADDING | Chapter 3 analyzes convergence properties of MC Dropout uncertainty estimates as a function of T. Shows diminishing returns beyond T ~20-50 for most architectures. **Most relevant reference for sample ablation.** |
| Kendall & Gal (2017) | YES (`kendall2017uncertainties`) | Uses T=50 samples; discusses sample count choice |
| Foong et al. (2019), "In-Between Uncertainty in Bayesian Neural Networks," ICML UDL Workshop | NO — optional | Analyzes quality of MC Dropout posterior approximation |

**What it adds to thesis:**
- Answers "is S=30 enough?" — directly relevant to deployment cost (228 min for S=30)
- If S=10 gives rho = 0.47 vs S=30 gives 0.48, the cost can be cut by 3x with minimal loss
- Standard ablation that reviewers expect

**Implementation:**
- REQUIRES code change: modify `mc_dropout_predict()` in `help_functions.py` to add `return_samples=True` parameter
- Run inference once with S=30, save all 30 raw per-pass outputs
- Subsample offline: S=5 (passes 1-5), S=10 (passes 1-10), S=20 (passes 1-20), S=30 (all)
- Compute rho, MAE, sigma statistics for each S
- Requires ~4h GPU time for one inference run

**New bib entries needed:**
1. Gal (2016) PhD thesis — convergence analysis

**Verdict: HIGH VALUE, MEDIUM EFFORT. Defer to Phase 4 (requires new inference).**

---

### 2.6 Temperature Scaling Verification

**What it is:** Post-hoc calibration of MC Dropout sigma by learning a single scalar temperature T that minimizes Expected Calibration Error (ECE) on a validation set. Calibrated sigma = raw_sigma * T.

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| Guo et al. (2017), "On Calibration of Modern Neural Networks," ICML | YES (`guo2017calibration`) | **PRIMARY** — Introduced temperature scaling for classification; the regression analogue is straightforward |
| Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression," ICML | YES (`kuleshov2018accurate`) | Proposes isotonic regression for recalibration of regression models — more flexible than single-T scaling |
| Platt (1999), "Probabilistic Outputs for Support Vector Machines," in Advances in Large Margin Classifiers | NO — optional | Original Platt scaling reference |

**What it adds to thesis:**
- Could reduce k95 from 11.34 toward a more reasonable value
- Addresses the "uncalibrated MC Dropout" limitation explicitly mentioned in Chapter 6 (line 80)
- Script already exists (`temperature_scaling_calibration.py`, 453 lines)
- However: old claim (ECE 0.356->0.033) is UNVERIFIED and must be re-run from scratch

**Implementation:**
- Re-run `temperature_scaling_calibration.py` with verified inputs
- Save JSON output with: T_optimal, ECE_before, ECE_after, coverage at multiple levels
- Validate independently against known k95 = 11.34

**New bib entries needed?** None — Guo 2017 and Kuleshov 2018 already in bibliography.

**Verdict: MEDIUM VALUE, LOW EFFORT. Proceed in Phase 3 (script re-run, no new inference).**

---

### 2.7 Error Detection AUROC for Trial 7

**What it is:** Replicate the T8 error detection analysis (AUROC/AUPRC at top-10% and top-20% thresholds) for T7, completing the cross-trial validation.

**Key references:**

| Ref | Already in bib? | Details |
|-----|-----------------|---------|
| No new references needed | — | AUROC is a standard metric; the methodology is identical to the T8 analysis already in Section 5.9 |

**What it adds to thesis:**
- Completes the T7 cross-validation: selective prediction (done), calibration audit (done), error detection (NOT done)
- Minor addition — the existing T7 cross-check already proves the point

**Implementation:**
- Source: T7 MC NPZ + det NPZ (both verified to exist)
- Same code as T8 error detection: compute abs_error_det, define top-10%/20% thresholds, compute AUROC/AUPRC
- No new inference needed

**New bib entries needed?** None.

**Verdict: LOW VALUE, LOW EFFORT. Include if time permits in Phase 3, but not a priority.**

---

## 3. New Bibliography Entries Needed

Based on the review above, the following new entries would strengthen the thesis:

### Must-Add (directly supports a planned analysis)

```bibtex
@article{barber2021limits,
  title={The Limits of Distribution-Free Conditional Predictive Inference},
  author={Barber, Rina Foygel and Candes, Emmanuel J and Ramdas, Aaditya and Tibshirani, Ryan J},
  journal={Information and Inference: A Journal of the IMA},
  volume={10},
  number={2},
  pages={455--482},
  year={2021}
}
```
- **Venue:** Information and Inference (Oxford University Press)
- **Why:** Proves conditional coverage impossibility — directly motivates stratified and conditional coverage analyses (2.2, 2.4)
- **Used in:** Discussion of conditional coverage results, Chapter 5 and 6

### Should-Add (supports MC sample ablation if executed)

```bibtex
@phdthesis{gal2016uncertainty,
  title={Uncertainty in Deep Learning},
  author={Gal, Yarin},
  school={University of Cambridge},
  year={2016}
}
```
- **Why:** Chapter 3 analyzes convergence of MC Dropout as a function of sample count T
- **Used in:** MC sample ablation analysis (2.5), if executed in Phase 4

### Nice-to-Add (strengthens calibration discussion)

```bibtex
@inproceedings{naeini2015obtaining,
  title={Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles},
  author={Naeini, Mahdi Pakdaman and Cooper, Gregory F and Hauskrecht, Milos},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2015}
}
```
- **Why:** Defines the binning methodology for reliability diagrams
- **Used in:** Reliability diagram analysis (2.1)

---

## 4. Summary: Analysis-to-Literature Mapping

| # | Analysis | Phase | Key Paper(s) Already Cited | New Paper(s) Needed | Value | Effort |
|---|----------|-------|---------------------------|---------------------|-------|--------|
| 1 | Reliability diagram | 3 | Kuleshov 2018, Guo 2017 | Naeini 2015 (optional) | HIGH | LOW |
| 2 | Stratified UQ by features | 3 | Ovadia 2019, Kendall & Gal 2017 | Barber et al. 2021 (must-add) | HIGH | MEDIUM |
| 3 | Per-graph variation | 3 | Angelopoulos 2023, Psaros 2023 | None | MEDIUM | LOW |
| 4 | Conformal conditional coverage | 3 | Angelopoulos 2023, Romano 2019 | Barber et al. 2021 (same as #2) | HIGH | MEDIUM |
| 5 | MC sample ablation | 4 | Gal & Ghahramani 2016 | Gal 2016 PhD thesis (should-add) | HIGH | MEDIUM |
| 6 | Temperature scaling | 3 | Guo 2017, Kuleshov 2018 | None | MEDIUM | LOW |
| 7 | T7 error detection | 3 | (standard AUROC) | None | LOW | LOW |

### Total new bib entries: 1 must-add, 1 should-add, 1 nice-to-add (3 total)

---

## 5. What the Thesis Already Covers Well (No Additional Literature Needed)

These areas have sufficient citations and thorough analysis. No further work recommended:

- **MC Dropout theory and application** — Gal 2016, Kendall & Gal 2017, Section 5.2
- **Deep ensemble comparison** — Lakshminarayanan 2017, Section 5.3
- **Split conformal prediction** — Angelopoulos 2023, Vovk 2005, Romano 2019, Section 5.5
- **k95 miscalibration analysis** — Guo 2017, Section 5.5.1 + 5.6
- **Selective prediction** — Sections 5.7 + 5.10.1
- **Cross-trial robustness** — Section 5.10 (T7 vs T8)
- **Feature correlation** — Section 5.11
- **GNN for traffic background** — Li 2018, Yu 2018, Zhao 2020, Jiang 2022

---

## 6. Literature Gaps NOT Worth Filling (Consistent with Phase 0 "Do Not Do" List)

| Gap | Why Not Worth Addressing |
|-----|------------------------|
| Heteroscedastic / evidential networks (Amini et al. 2020, NeurIPS) | Would require retraining; better as future work citation |
| Laplace approximation for GNNs (Daxberger et al. 2021, NeurIPS) | Very heavy computation (Hessian over ~500K params); marginal thesis value |
| SWAG (Maddox et al. 2019, NeurIPS) | Requires access to training trajectory checkpoints in specific format; not how our checkpoints are saved |
| Spectral-normalized neural Gaussian processes (Liu et al. 2020, NeurIPS) | Requires architectural changes + retraining |
| Deep kernel learning (Wilson et al. 2016, ICML) | Different paradigm; would need new model |

**These should remain as "Future Work" citations in Chapter 6, not as executed analyses.**

Candidate future-work citations (already in bibliography or easily added):
- Amini et al. (2020), "Deep Evidential Regression," NeurIPS — NOT in bib, could add
- Daxberger et al. (2021), "Laplace Redux," NeurIPS — NOT in bib, could add
- Maddox et al. (2019), "A Simple Baseline for Bayesian Uncertainty in Deep Learning," NeurIPS — NOT in bib, could add

These would be added ONLY to the Future Work section as citations, not as executed analyses.

---

*This document is a literature review only. No analyses were executed. No files were modified.*
*Generated by OpenCode assistant — 2026-03-18.*
