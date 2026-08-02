# UQ Implementation Audit — Summary for Examiners

**Author:** Mohd Zamin Quadri
**Audit date:** 2026-04-25
**Scope:** Independent read-only review of every UQ method used in the thesis, comparing the implementation against its textbook reference to verify that no "weak" or "failed" result is the artefact of a code bug.

## Methods audited (10)

| # | Method | Implementation reviewed |
|---|---|---|
| 1 | Monte Carlo Dropout | `code/scripts/gnn/heteroscedastic_mc_dropout.py`, `mc_dropout_full_100graphs_mc30.npz` |
| 2 | Deep Ensemble (Trial D, 5 seeds) | `code/scripts/training/run_deep_ensemble.py`, 5 distinct `model.pth` checkpoints, `deep_ensemble_predictions.npz` |
| 3 | Split conformal prediction (T8) | `colab_uq_master.ipynb` outputs in `conformal_standard.json` |
| 4 | Adaptive (σ-scaled) conformal | `phase3_results/adaptive_conformal_decile.json` |
| 5 | Post-hoc temperature scaling | `temperature_scaling_results.json` |
| 6 | T9 heteroscedastic, frozen backbone | `code/scripts/gnn/models/point_net_transf_gat_frozen_heteroscedastic.py`, `code/scripts/gnn/losses/heteroscedastic_loss.py`, `train_heteroscedastic.py` |
| 7 | T10/T11 CQR (pinball loss + conformal correction) | `code/scripts/gnn/losses/quantile_loss.py`, `train_cqr.py`, `train_cqr_frozen.py`, `point_net_transf_gat_frozen_cqr.py` |
| 8 | Selective prediction | `phase3_results/selective_prediction_s30.json` |
| 9 | Stratified UQ by \|Δv\| quartile | `code/scripts/misc/gen_batch7.py` |
| 10 | AUROC error detection | `auroc_corrected.json`, `gen_batch2.py` |

## Result

- **Bugs found: 0**
- **Re-runs required: 0**

Every implementation was found to match its textbook reference. The audit independently re-derived the conformal quantile, the pinball loss, the heteroscedastic NLL, the deep-ensemble aggregation, and the AUROC computation, and verified each against the saved numerical artefacts. The five deep-ensemble checkpoints have **identical file sizes but distinct SHA-256 hashes**, and pairwise comparison of the five member predictions confirms they are genuinely different models (mean absolute pairwise difference 1.0–2.1 veh/h, Pearson 0.88–0.97). MC Dropout sigmas have **zero zero-valued entries out of 3,163,500**, confirming dropout was actually active at inference. The pinball loss in `quantile_loss.py` matches the `yromano/cqr` reference `AllQuantileLoss.forward()` term-for-term. The heteroscedastic NLL in `heteroscedastic_loss.py` matches Kendall & Gal (2017) with the Seitzer (2022) log-variance regulariser.

## Three interpretation nuances (incorporated into Phase 2 prose, not code changes)

1. **Stratified UQ Q1.** Q1 (smallest \|Δv\| quartile) contains 790,875 nodes whose target is exactly $|\Delta v| = 0$ — segments unaffected by the policy intervention. The high ρ = 0.725 on Q1 is therefore partly mechanical: when $y = 0$, both $|\text{error}|$ and $\sigma$ reduce to functions of the model's small output magnitude on unchanged segments, and their correlation reflects that shared dependence rather than a genuine "MC Dropout works well on easy regimes" finding. The thesis Discussion §6.4 has been updated to flag this caveat — the Q1→Q4 contrast should be read as "MC Dropout works mechanically on trivial segments and breaks on hard ones."

2. **T9 vs T11 "frozen" backbone.** Both trials freeze the backbone weights via `requires_grad=False`, but the dropout behaviour differs deliberately: T9 keeps backbone dropout active during head training and inference (so the backbone supplies epistemic σ via stochastic forward passes through the head), while T11 overrides `train()` to force the backbone into `eval()` mode (dropout off, single deterministic forward pass — appropriate because CQR does not use MC samples). Methodology §3.2 has been updated to clarify the two senses of "frozen."

3. **Conformal split.** The 50/50 calibration/evaluation split is a **random scenario-level partition with seed 42**, not a sequential first-50 / last-50 split. The random partition makes the two halves statistically symmetric, so node-level exchangeability holds at the scenario aggregate. Experimental §4.3 wording has been updated from "50/50 graph split" to "random scenario-level partition (seed 42)," and the marginal-vs-conditional coverage discussion in Discussion §6.5 explicitly refers to this.

## Best explanation for each "weak" result

- **Deep Ensemble ρ = 0.3997 vs MC Dropout 0.4820** — Real finding, not a bug. Independently-seeded MSE-trained networks disagree most on random-init effects rather than on the genuinely difficult inputs; MC Dropout samples from an approximate posterior over weights and produces a more targeted spread.
- **Trial 10 R² = 0.4057 collapse** — Real finding, not a bug. Trial 11 (same loss, same head, same data, only difference is freezing the backbone) recovers R² = 0.5835. The single isolated design knob is the backbone-trainability flag.
- **Stratified Q4 ρ = 0.100** — Real finding, not a bug. Three documented mechanisms: dynamic range of \|Δv\| (Q4 spans up to 230 veh/h), saturation/non-linear flow–density effects, and sparse training-data coverage in the large-response regime.

---

*This audit was conducted by independent read-through of all UQ scripts and recomputation of all key numerical artefacts directly from the saved \*.npz prediction arrays. No method required re-running. The three interpretation nuances above have been folded into the thesis prose in §3.2, §4.3, and §6.4.*
