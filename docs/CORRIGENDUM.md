# Post-Submission Corrigendum

This document records corrections and reproducibility findings made after the thesis was
submitted on May 15, 2026. It does not alter the submitted PDF at `thesis/submission_2026-05-15/`.

## C1: Target Zero Mass

**Submitted claim:** The thesis attributes an 88.7% exact-zero share to the test target
`Delta v` in the methodology, results, and discussion.

**Correction:** The complete cached test target contains 872,540 exact zeros among 3,163,500
values, or 27.58147621%. The 88.7% value belongs to the raw `CAPACITY_REDUCTION` input feature.

**Evidence:** Full cached Trial 7, Trial 8, and ensemble target arrays agree on the corrected
count. The aggregate result is locked by `tests/test_analysis_bundle.py` and
`scripts/check_repository.py`. Row-level arrays are not redistributed from this repository;
their hashes are recorded in `analysis_outputs/artifact_manifest.csv`.

**Impact:** The submitted explanation materially overstates target sparsity. Selective-prediction
and error-detection results remain historical measured outputs, but they must not be justified by
an 88.7% zero-target claim. Tail performance and within-scenario dependence remain important
limitations.

## C2: Evaluation Preprocessing

The historical base pipeline fits independent feature and position scalers to training,
validation, and test partitions. This does not use test targets, but evaluation-distribution
statistics influence preprocessing and weaken deployment and cross-split comparability claims.
The historical artifacts and scores are preserved. Future experiments should fit one scaler on
training data, version it with the feature schema, and apply it unchanged to later partitions.

## C3: Calibration Protocols

The following protocols answer different questions and must not be pooled:

| Protocol | Calibration/evaluation split | Status | Principal result |
|---|---|---|---|
| `graph20_80_v1` | First 20 / last 80 graphs | Tracked and replayable with controlled source artifacts | T=2.702; ECE 0.269 to 0.048 |
| `node30_70_thesis_final` | Random 30% / 70% nodes, seed 42 | Reported; split indices unavailable | T approximately 2.887; ECE approximately 0.356 to 0.034 |
| Primary split conformal | 50 / 50 scenarios, seed 42 | Thesis protocol | 90.02% / 95.01% empirical marginal coverage |

Node-level splitting does not separate graph-level dependence. Conformal coverage is empirical
and marginal over the evaluated split, not a guarantee for each scenario, link, city, or policy.

## C4: Stochastic Replay Variation

The Trial 8 MC Dropout archive and the later verification archive are separate stochastic
replays. Their rounded MAE, Spearman rho, and empirical scale values differ slightly. Every public
number must identify the archive or protocol it uses; stochastic results are not presented as
bit-identical reruns.

## C5: Replay and Checkpoint Boundaries

- Raw MATSim scenario outputs are absent, so raw-to-graph reproduction is unavailable.
- Cached prediction arrays support the strongest numerical replay path but are not public here.
- Trial 9 has no retained prediction cache.
- Trials 10 and 11 retain validation arrays but not test arrays; test scores are reported-only.
- Several historical scripts load PyTorch/PyG checkpoints permissively. Future checkpoint replay
  must validate state-dictionary key coverage and load only trusted pickle-capable artifacts.

## C6: Scope

The evaluated evidence covers one Paris network, one capacity-reduction intervention family, a
fixed 1,000-scenario subset, a 100-scenario test split, and the PointNetTransfGAT family. It does
not establish generalization to other cities, policies, model families, or production settings.

## C7: Trial 8 Error-Detection AUROC, and Trial 7 Rank Correlation

Found during the September 2026 repository audit, by recomputing every headline number
from the artifact it cites rather than from the summaries that quote it.

### C7a: Trial 8 AUROC

**Submitted and previously published values:** AUROC 0.7548 at the top-10% error
threshold and 0.7324 at top-20%. The README, `docs/UQ_SUMMARY.md`, and
`chapters/{01,05,06}.tex` all carried 0.7548, attributing it to a file named
`auroc_corrected.json`. No such file exists in this repository or in the data release.

**Verified values:** **0.7585** at top-10% and **0.7401** at top-20%.

**Source artifact:**
`TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/trial8_uq_ablation_results.csv`
(3,163,500 rows; published on the `large-files-v1` release of the companion data
repository). This is the same file `docs/verified/UQ_ERROR_DETECTION_T8.md` names as its
source, and that document already reported 0.7585 and 0.7401 correctly. The error was in
the summaries, not in the audit.

**Exact evaluation definition.** Score is `pred_mc_std`, the per-node standard deviation
over 30 MC Dropout forward passes. Positives are the nodes whose `abs_error_det` — the
absolute error of the deterministic forward pass — is at or above the 90th percentile
(top-10%, cutoff 9.9305 veh/h, 316,350 positives) or the 80th percentile (top-20%, cutoff
6.0129 veh/h, 632,700 positives). AUROC is `sklearn.metrics.roc_auc_score` over all
3,163,500 nodes. Both the cutoffs and the AUPRC values (0.3148 and 0.4547) reproduce
exactly, which confirms the row set is identical to the one the audit used.

**Why the correction is justified.** The corrected values are reproducible from a
published artifact by a stated procedure; the previous ones are reproducible from
nothing. Three independent routes agree on 0.7585: a direct recomputation from the CSV, a
fresh run of `scripts/evaluation/run_part2_uq_analyses.py`, and the already-tracked
`docs/verified/UQ_ERROR_DETECTION_T8.md`. The correction moves both figures slightly
upward, so it does not strengthen any claim that was previously weaker — the qualitative
conclusion, that MC Dropout sigma carries useful ranking signal well above the 0.500
random baseline, is unchanged.

### C7b: Trial 7 Spearman rho

**Reported value:** 0.446, in the README, `docs/UQ_SUMMARY.md`, and
`results/t7_error_detection.json`.

**Value replayable from the retained archive:** **0.4437**.

**Source artifact:**
`results/predictions/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz`.

**Definition.** Pooled Spearman correlation between `uncertainties` and the absolute
error of the MC-mean prediction, over all 3,163,500 nodes. The same definition applied to
Trial 8 reproduces its published 0.482 to within 0.0002, so the definition is not in
question. The retained Trial 7 archive's MAE, 4.0737 veh/h, matches
`t7_error_detection.json` exactly, so it is the array the reported MAE came from — but
its rho is 0.4437, not 0.44599. The per-graph mean is 0.4242, which does not explain the
gap either.

**Status.** This is a C4-class discrepancy: the reported figure appears to come from a
stochastic replay that was not retained. It is recorded rather than resolved. Documents in
this repository now quote 0.4437 as the replayable Trial 7 value and flag 0.446 as
reported-only. The cross-trial conclusion — that Trial 7 shows the same
uncertainty-error ranking behaviour as Trial 8, slightly weaker — is unaffected.

### Scope of this correction

The submitted PDF at `thesis/submission_2026-05-15/` is unchanged and still reads 0.7548,
0.7324, and 0.446. That document is the examined record. Everything outside it now
reports the verified values, and `scripts/verify_headline_results.py` asserts them against
their source artifacts on every run.

## Provenance

- Immutable submitted artifact baseline: canonical commit `4b95a3d8aca5929bb88b84bb7f7ae86c48e2f428`.
- Audited reproducibility source: `fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a` on branch
  `analysis/e2e-thesis-intelligence` of this repository.
- Audit generation date recorded in the aggregate bundle: August 15, 2026.

Additional provenance and hashes are in `docs/ARTIFACT_PROVENANCE.md`.
