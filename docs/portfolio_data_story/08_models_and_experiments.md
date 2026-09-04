# Models, trials and what each one was for

Every figure here is read from a retained artifact: architecture from the checkpoint
weights, configuration and outcome from the trial's own JSON. Where a value was never
recorded it says so; nothing is inferred.

## Checkpoint inventory — 16 retained

Architecture recovered from tensor shapes alone (`scripts/data_exploration` does not
need the model class to do this).

| Trial | Checkpoint | Params | in | Head | Test R² | MAE | Test split |
| --- | --- | ---: | :-: | --- | ---: | ---: | --- |
| T1 `pointnet_transf_gat_1st_bs32_5feat_seed42` | 5.7 MB | 1,416,833 | 5 | `read_out_node_predictions` | **0.7860** | 2.972 | 50 graphs |
| T2 `2nd_try` | 5.7 MB | 1,416,835 | 5 | `gat_final` | 0.5117 | 4.328 | 50 graphs |
| T3 `3rd_trial_weighted_loss` | 5.7 MB | 1,416,835 | 5 | `gat_final` | 0.2246 | 5.990 | 50 graphs |
| T4 `4th_trial_weighted_loss` | 5.7 MB | 1,416,835 | 5 | `gat_final` | 0.2426 | 6.080 | 50 graphs |
| T5 `5th_try` | 5.7 MB | 1,416,835 | 5 | `gat_final` | 0.5553 | 4.242 | 50 graphs |
| T6 `6th_trial_lower_lr` | 5.7 MB | 1,416,835 | 5 | `gat_final` | 0.5223 | 4.324 | 50 graphs |
| T7 `7th_trial_80_10_10_split` | 5.7 MB | 1,416,835 | 5 | `gat_final` | 0.5471 | 4.060 | **100 graphs** |
| **T8** `8th_trial_lower_dropout` | 5.7 MB | 1,416,835 | 5 | `gat_final` | **0.5957** | **3.957** | **100 graphs** |
| T9 `9th_trial_heteroscedastic` | 11.4 MB | 1,416,902 | 5 | `gat_heteroscedastic_head` | not recorded | — | 100 graphs |
| T10 `10th_trial_cqr` | 5.7 MB | 1,416,902 | 5 | 2-output quantile | 0.4057 † | 4.130 † | 100 graphs |
| T11 `11th_trial_cqr_frozen` | 5.7 MB | 1,416,902 | 5 | `gat_quantile_head` | 0.5835 † | 4.302 † | 100 graphs |
| Ensemble ×5 `deep_ensemble_seed{42,137,256,389,512}` | 5×5.7 MB | 1,416,835 | 5 | `gat_final` | **0.6841** ‡ | 3.485 ‡ | 100 graphs |

† `r2_midpoint` / `mae_midpoint` — the midpoint of the predicted quantile interval, not a
point-prediction head. Not directly comparable with the R² column above.
‡ ensemble mean of 5 members; individual members score R² 0.640–0.650.

**Five architecture variants**, all sharing the same 1.42 M-parameter backbone:

| Params | Distinguishing head | Trials |
| ---: | --- | --- |
| 1,416,833 | `read_out_node_predictions` — a Linear output layer | T1 only |
| 1,416,835 | `gat_final` — GATConv(64→1) | T2–T8 and all 5 ensemble seeds |
| 1,416,902 | 2-output quantile | T10 |
| 1,416,902 | `gat_quantile_head` | T11 |
| 1,416,902 | `gat_heteroscedastic_head` | T9 |

The +67 parameters in T9–T11 are the extra head. **T1's different head is visible in the
weights**, which independently confirms the long-standing note that T1 is not comparable
with T2–T8.

## The split changed at T7 — and it matters

| Trials | Test graphs | Nodes scored |
| --- | ---: | ---: |
| T1 – T6 | **50** | 1,581,750 |
| T7 – T11, ensembles | **100** | 3,163,500 |

T7's own name records it: `7th_trial_80_10_10_split`. **R² values either side of that
boundary are not measured on the same test set**, so ranking T1–T6 against T7–T8 is not a
like-for-like comparison. Recorded as [CORRIGENDUM C9](../CORRIGENDUM.md).

This also puts T1's headline R² = 0.786 in context: a different head, a different split,
and zero dropout — which is separately why it is excluded from all uncertainty work, since
MC Dropout is undefined when σ = 0 everywhere.

## Experiment timeline

| # | Change from previous | Outcome | Retained? |
| --- | --- | --- | --- |
| T1 | Linear output head, no dropout, 50-graph split | R² 0.786 — the best point accuracy of any trial | Checkpoint kept; excluded from UQ |
| T2 | GATConv output head, dropout on | R² 0.512 — accuracy drops sharply | Baseline for the comparable family |
| T3 | Weighted loss | R² 0.225 — worst result recorded | Abandoned |
| T4 | Weighted loss, second attempt | R² 0.243 — confirms T3 | Weighted loss dropped |
| T5 | Back to unweighted | R² 0.555 — recovers | Direction confirmed |
| T6 | Lower learning rate | R² 0.522 — no gain | Not pursued |
| T7 | 80/10/10 split, 100-graph test | R² 0.547 on the larger test set | **Split standard from here on** |
| **T8** | Dropout 0.3 → **0.2** | **R² 0.596, MAE 3.96** — best comparable trial | **Baseline for all UQ work** |
| T9 | Freeze T8, add heteroscedastic head | best val NLL 3.249 @ epoch 290, 873 min | No test metrics recorded |
| T10 | Full CQR retrain from scratch | 87 hours; **gate FAIL** — R² midpoint 0.406, PICP95 91.8% | Kept as negative evidence |
| T11 | Freeze T8, train quantile head only | 40 hours; **gate PASS** — R² midpoint 0.584, PICP95 94.9% | The working CQR result |
| Ens. | 5 seeds, dropout off at inference | R² **0.684** — best point accuracy of the comparable family | Kept; see the trade-off below |

Two lessons the artifacts support directly:

**Freezing beat retraining for CQR.** T10 retrained everything for 87 hours and failed its
own acceptance gates, with the interval midpoint losing most of the backbone's accuracy
(0.406 against T8's 0.596). T11 froze the T8 backbone, trained only a quantile head for 40
hours, passed every gate, and kept the accuracy (0.584). Same method, opposite outcome,
decided by what was allowed to move.

**The ensemble bought accuracy and lost uncertainty quality.** It is the most accurate
comparable model here (R² 0.684 against T8's 0.596), yet its σ ranks errors *worse* than
MC Dropout: Spearman ρ 0.400 against 0.482, a 17.1% drop, recorded in the trial's own
`comparison_with_mc_dropout` block. Better predictions did not mean better uncertainty —
which is the whole reason the thesis evaluates the two properties separately.

## Uncertainty methods actually implemented

| Method | Input | Output | Headline verified result | Limitation |
| --- | --- | --- | --- | --- |
| **MC Dropout** | T8 checkpoint, dropout active, S=30 | per-link σ | ρ = 0.482; S=30 on the convergence plateau | σ is a ranking signal, not a calibrated scale |
| **Temperature scaling** | σ, 20/80 graph split | one scalar T | T = 2.702, ECE 0.269 → 0.048 | Fixes average width, not per-node |
| **Split conformal** | absolute residuals | shared interval | 90.17% / 95.09% coverage | Marginal only; one width for every link |
| **Adaptive conformal** | residuals ÷ σ | per-node interval | conditional coverage [83.7%, 96.4%] vs [59.0%, 98.1%] standard | Wider intervals overall |
| **Selective prediction** | σ ranking | retain/abstain | **−41.2% MAE at 50% retained** | Needs only the ranking, not the scale |
| **Error detection** | σ as score | flag top-decile errors | **AUROC 0.7585** | Ranking quality, not precision at an operating point |
| **CQR** | quantile head + conformal | calibrated interval | T11 PICP 89.8/94.9%, width 17.8/24.8 | T10 shows a full retrain can fail outright |
| **Deep ensemble** | 5 seeds | mean + σ | R² 0.684 | ρ 0.400 — worse ranking than MC Dropout |
| **Heteroscedastic** | frozen T8 + variance head | predicted σ | val NLL 3.249 | No test metrics recorded |

Everything except the ensemble and CQR training is **post-hoc**: the T8 weights are loaded,
frozen, and never updated.

## Reproducing

```bash
python scripts/data_exploration/explore_checkpoints.py --models models --results results
```

Architecture facts come from the checkpoints themselves; metrics come from
`results/trials/<trial>/`. Trials whose metrics were never written are reported as such
rather than filled in.
