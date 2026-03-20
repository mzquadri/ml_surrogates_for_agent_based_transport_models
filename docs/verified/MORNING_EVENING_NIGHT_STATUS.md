# MORNING / EVENING / NIGHT STATUS
> Quick-reference status document for any time you open the laptop.
> Shows exactly what is done, what is verified, and what still needs work.
> Last updated: March 2026 (OpenCode verification session).

---

## WHAT IS DONE (verified, thesis-ready)

### Model Performance
- [x] T8 best model: R2=0.5957, MAE=3.96 veh/h, RMSE=7.12 veh/h
- [x] All 8 trials verified from JSON files
- [x] Hyperparameters verified: T5-T8 all batch=8, T8 dropout=0.2
- [x] Architecture verified: PointNetConv x2 + TransformerConv x2 + GATConv x2

### UQ Performance
- [x] MC Dropout T8: rho=0.4820 (100 graphs, 30 samples)
- [x] Conformal 90%: q=9.9196, achieved=90.02%
- [x] Conformal 95%: q=14.6766, achieved=95.01%
- [x] Sigma-normalized: 65.8% narrower intervals
- [x] Rejection at 50% threshold: MAE drops 39.9%

### Data Understanding
- [x] 31,635 road segments (nodes) after LineGraph() transform
- [x] 5 input features identified and verified
- [x] Target = delta car volume (veh/h) from help_functions.py
- [x] 1,000 of 10,000 MATSim scenarios used
- [x] T1-T6: 50 test graphs (1,581,750 nodes); T7-T8: 100 test graphs (3,163,500 nodes)

### Old Claims Audited
- [x] MEETING_PREPARATION.md hyperparameters confirmed WRONG
- [x] generate_thesis_charts.py confirmed WRONG (hardcoded wrong values)
- [x] Temperature Scaling claim: NOT VERIFIED — no source file found
- [x] T2 weighted_loss=False (corrected — was listed as True)
- [x] T3 dropout=0.0 (corrected — was listed as 0.3)

---

## WHAT IS PARTIALLY DONE

### Figures
- [ ] Figures not yet regenerated with correct values
- [ ] generate_verified_figures.py written but not executed
- [ ] Old figures in docs/visuals/ may use wrong values from generate_thesis_charts.py

### Thesis Draft
- [ ] Unknown whether thesis draft in thesis_TUM_FINAL.pdf reflects correct hyperparameters
- [ ] Thesis draft should be checked for Temperature Scaling claim
- [ ] Thesis draft should be checked for batch size claims (T5-T8 = batch 8, not 32/64)

---

## WHAT IS NOT DONE / NEEDS WORK

### High Priority
1. **Regenerate all thesis figures** using verified values
   - Use `docs/verified/figures/generate_verified_figures.py`
   - Replace any old figures generated from `generate_thesis_charts.py`

2. **Check thesis draft** (thesis_TUM_FINAL.pdf) for wrong claims
   - Batch sizes (should be 8 for T5-T8)
   - Temperature Scaling (remove if present)
   - Data splits (should be 80/15/5 and 80/10/10, not 70/20/10)
   - T2 weighted loss (should be False)

3. **Find Temperature Scaling source file**
   - Search entire repo for files containing "temperature", "ECE", "calibration"
   - If not found: remove claim from thesis
   - If found: add to FILE_MANIFEST.csv and verify

### Medium Priority
4. **Supervisor emails** — paste actual emails to complete WORK_TIMELINE doc
5. **Full thesis review** — check all figure captions use verified numbers
6. **T4 hyperparameters** — batch_size missing from JSON (saved as test_results.json not test_evaluation_complete.json)

### Low Priority
7. Repo cleanup (see CLEANUP_PLAN.md)
8. Add more visual figures (see FIGURE_REGENERATION_PLAN.md)
9. Colab runbook testing

---

## NUMBERS TO HAVE IN YOUR HEAD (for meetings)

```
Best model:    T8 (8th trial)
Architecture:  PointNetTransfGAT (PointNet + Transformer + GAT)
Data:          1,000 Paris scenarios; 31,635 road nodes per graph
Training:      80/10/10 split (800/100/100 graphs)
Best R2:       0.5957    (~60% variance explained)
Best MAE:      3.96 veh/h
Best RMSE:     7.12 veh/h
MC Dropout:    rho = 0.4820 (T=30 samples, Spearman correlation)
Conformal 90%: ±9.92 veh/h (achieved 90.02%)
Conformal 95%: ±14.68 veh/h (achieved 95.01%)
Sigma quality: k95 = 11.65 (not calibrated — use conformal for guarantees)
```

---

## COMMON QUESTIONS AND ANSWERS

**Q: Why R2 = 0.6? Is that good?**
A: Predicting city-wide traffic rerouting effects at 31,635-road resolution from 5 features in <1 second (vs hours of MATSim). Yes, 60% variance explained is meaningful. Most of the unexplained variance is in high-traffic roads near the intervention — the hardest cases.

**Q: Can I trust MC Dropout sigma as a calibrated uncertainty?**
A: No. sigma is useful for RANKING which predictions are risky (rho=0.48), but it's not calibrated as a probability. k95=11.65 means naive ±1.96*sigma gives only 55% coverage. Use conformal prediction for calibrated intervals.

**Q: Why did Experiment A (ensemble) give rho=0.16 while single-model MC Dropout gives rho=0.48?**
A: The 5 ensemble models were trained on different random data subsets. Their disagreement reflects data distribution differences, not epistemic uncertainty about the predictions. Single-model MC Dropout is more informative for this task.

**Q: What is the line graph transform?**
A: Roads (edges in standard graph) become nodes. Two "road nodes" are connected if they share a junction. After transform: 31,635 nodes = 31,635 road segments. This enables node-level prediction of per-road traffic changes.

**Q: Why does weighted loss (T3, T4) perform worse?**
A: Weighted loss (higher weight for high-volume roads) focuses training on a few major roads at the expense of the broader network. The model fails to generalize to the full distribution of delta volumes, resulting in R2 ≈ 0.22-0.24 vs 0.55-0.60 for standard MSE.

**Q: What is a surrogate model?**
A: A fast approximation of an expensive simulation. Here: the GNN predicts what MATSim would output, but in seconds instead of hours.

---

## FILE LOCATIONS (quick reference)

| What you need | Where it is |
|---|---|
| Verified results (all numbers) | docs/verified/VERIFIED_RESULTS_MASTER.csv |
| What old docs got wrong | docs/verified/OLD_CLAIMS_AUDIT.md |
| Model architecture code | scripts/gnn/models/point_net_transf_gat.py |
| T8 best model JSON | data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/test_evaluation_complete.json |
| T8 UQ JSON | data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_metrics_model8_mc30_100graphs.json |
| T8 conformal JSON | data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/conformal_standard.json |
| Figure generation script | docs/verified/figures/generate_verified_figures.py |
| Thesis story (narrative) | docs/verified/THESIS_STORY_FROM_ZERO.md |
| Meeting cheat sheet | docs/verified/CHEAT_SHEET.md |
