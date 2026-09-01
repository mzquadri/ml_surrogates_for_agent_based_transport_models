# Post-Submission Corrigendum

This document records corrections and reproducibility findings made after the thesis was
submitted on May 15, 2026. It does not alter the submitted PDF at `document/main.pdf`.

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

## Provenance

- Immutable submitted artifact baseline: canonical commit `4b95a3d8aca5929bb88b84bb7f7ae86c48e2f428`.
- Audited reproducibility source: `fdb4ef0c9c736576ae34d5e331d8b66a7a6d877a` on branch
  `analysis/e2e-thesis-intelligence` of the retired source repository.
- Audit generation date recorded in the aggregate bundle: August 15, 2026.

Additional provenance and hashes are in `docs/ARTIFACT_PROVENANCE.md`.
