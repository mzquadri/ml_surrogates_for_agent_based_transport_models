# Submission Manifest
## Thesis: Uncertainty Quantification for Graph Neural Network Surrogates of Agent-Based Transport Models
## Author: Mohd Zamin Quadri
## M.Sc. Mathematics in Science and Engineering, TUM

### Compilation
- Compiler: pdflatex + biber (NOT bibtex)
- 3 passes: pdflatex -> biber -> pdflatex -> pdflatex
- MiKTeX 25.12 / biber 2.21
- 94 pages, 0 errors, 0 overfull hboxes, 0 undefined citations/references

### Claim-to-Artifact Mapping

| Claim | Thesis Location | Artifact | Script |
|-------|----------------|----------|--------|
| MC Dropout R2=0.5857, MAE=3.95, rho=0.4820 | abstract, Ch5 | mc_dropout_full_metrics_model8_mc30_100graphs.json | run_mc_dropout_full.py |
| Selective pred 50%: MAE 2.32 (-41.2%) | abstract, Ch5 Tab 5.7 | selective_prediction_s30.json | regenerate_fig58_s30.py |
| Temperature T=2.70, ECE 0.269->0.048 (82%) | abstract, Ch5, Ch6 | temperature_scaling_t8.json | compute_temperature_scaling.py |
| Conformal q95=14.68, PICP=95.01% | abstract, Ch5 | conformal_standard.json | (existing pipeline) |
| CRPS mean=3.383, CRPS/MAE=0.857 | abstract, Ch5, Ch6 | crps_t8.json | compute_crps.py |
| PIT KS=0.245 (raw), 0.104 (after TS) | abstract, Ch5, Ch6 | pit_t8.json, pit_after_tempscaling_t8.json | compute_pit.py, compute_pit_after_tempscaling.py |
| Winkler 32.3 vs 49.7 (90%) | abstract, Ch5, Ch6 | winkler_t8.json | compute_winkler.py |
| S-convergence <1% S30->S50 | abstract, Ch5 | s_convergence_results.json | run_s_convergence.py |
| Adaptive conformal [90.0%, 96.2%] | abstract, Ch5, Ch6 | conformal_conditional_coverage_t8.json | (existing pipeline) |
| Ensemble PyG bug (R2~0, lin.weight mismatch) | abstract, Ch1, Ch4, Ch5, Ch6, Ch7 | ensemble_bug_root_cause.json | diagnose_ensemble_bug_v4b.py |
| CRPS/MAE optimum = 1/sqrt(2) = 0.707 | Ch6 | crps_mae_ratio_theoretical.json | verify_crps_mae_ratio_main.py |
| Bootstrap CI rho: [0.4599, 0.4689] | Ch5 | bootstrap_ci_results.json | compute_bootstrap_ci.py |

### Double-Verification Pairs (all passing)

| Metric | Main Script | Verification Script | Main JSON | Verification JSON |
|--------|------------|--------------------|-----------|--------------------|
| CRPS | compute_crps.py | verify_crps.py | crps_t8.json | crps_t8_verification.json |
| PIT | compute_pit.py | verify_pit.py | pit_t8.json | pit_t8_verification.json |
| PIT after TS | compute_pit_after_tempscaling.py | verify_pit_after_tempscaling.py | pit_after_tempscaling_t8.json | pit_after_tempscaling_t8_verification.json |
| Winkler | compute_winkler.py | verify_winkler.py | winkler_t8.json | winkler_t8_verification.json |
| CRPS/MAE ratio | verify_crps_mae_ratio_main.py | verify_crps_mae_ratio_independent.py | crps_mae_ratio_theoretical.json | crps_mae_ratio_theoretical_verification.json |

### Final Verification
- `scripts/verify_all_numbers_final.py`: 39/39 checks PASS
- Output: `docs/verified/phase3_results/final_numeric_verification.json`
- No TODO/FIXME/PLACEHOLDER in any .tex file
- All "data distribution mismatch" references replaced with verified PyG API bug explanation
