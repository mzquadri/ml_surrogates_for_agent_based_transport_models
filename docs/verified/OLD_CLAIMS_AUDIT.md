# OLD CLAIMS AUDIT
> Source of truth: JSON result files only.
> Audited: March 2026 (OpenCode verification session).
> Purpose: Identify every incorrect claim in existing docs before thesis submission.

---

## DOCUMENT AUDITED: `docs/MEETING_PREPARATION.md`

**Verdict: DO NOT TRUST HYPERPARAMETERS. Every batch size, most dropout values, most LR values, and all data splits in this doc are wrong vs the JSON files.**

---

## HYPERPARAMETER DISCREPANCIES

| Claim in MEETING_PREPARATION.md | Verified from JSON | Verdict |
|---|---|---|
| T2 batch_size = 32 | batch_size = **16** | WRONG |
| T2 dropout = 0.20 | dropout = **0.3** | WRONG |
| T2 use_weighted_loss = True | use_weighted_loss = **False** | WRONG — T2 did NOT use weighted loss |
| T2 LR = 0.001 | LR = **0.0005** | WRONG |
| T3 dropout = 0.3 | dropout = **0.0** | WRONG — T3 had dropout disabled |
| T5 batch_size = 32 | batch_size = **8** | WRONG |
| T6 batch_size = 64 | batch_size = **8** | WRONG |
| T6 LR = 0.0005 | LR = **0.0003** | WRONG (this was the defining change for T6) |
| T7 batch_size = 64 | batch_size = **8** | WRONG |
| T7 LR = 0.0005 | LR = **0.0006** | WRONG |
| T8 batch_size = 64 | batch_size = **8** | WRONG |
| T8 dropout = 0.15 | dropout = **0.2** | WRONG |
| Data split = 70/20/10 | Actual splits: **80/15/5** (T1-T6) and **80/10/10** (T7-T8) | WRONG — no trial used 70/20/10 |

---

## METRIC DISCREPANCIES

Test metrics (R2, MAE, RMSE) in MEETING_PREPARATION.md are **mostly correct** for the models verified. However:

- T1 R2=0.786 is stated correctly but T1 used a **different architecture** (linear layers, not GATConv final). It should not be cited alongside T2-T8.
- The document does not flag this architecture difference clearly enough.

---

## FIGURE SCRIPT DISCREPANCIES

**File:** `scripts/evaluation/generate_thesis_charts.py`

This script hardcodes R2/MAE/RMSE values that **do not match** the verified JSON files.

Any figure generated from this script is **unreliable**. All thesis figures must be regenerated using values from `VERIFIED_RESULTS_MASTER.csv`.

Mark `generate_thesis_charts.py` as DEPRECATED.

---

## TEMPERATURE SCALING — SPECIAL WARNING

The following result appears in prep notes:
> "Temperature Scaling reduced ECE from 0.356 to 0.033, T=2.90"

**Status: UNVERIFIED. No JSON, NPZ, checkpoint, or log file was found anywhere in the repository that contains this result.**

This must NOT be included in the thesis unless:
1. The source output file is located and read
2. The experiment can be reproduced

Until then, treat this claim as **non-existent from a thesis standpoint**.

---

## ARCHITECTURE CLAIM DISCREPANCIES

| Claim | Status |
|---|---|
| "Final layer uses linear projection" | WRONG for T2-T8. Final two layers are GATConv. T1 may have used linear. |
| "Dropout applied at all layers" | WRONG — Dropout is NOT applied in the final GATConv layer, only in PointNet and TransformerConv layers. |

---

## CLAIMS CORRECTLY STATED

| Claim | Status |
|---|---|
| Best model is T8 (8th trial) | CORRECT |
| Target variable is delta car volume (veh/h) | CORRECT |
| Paris road network, 31,635 nodes after LineGraph transform | CORRECT |
| MC Dropout provides meaningful uncertainty ranking | CORRECT (Spearman rho=0.4820 for T8) |
| Conformal prediction achieves guaranteed coverage | CORRECT (90.02% at 90% target; 95.01% at 95%) |
| Weighted loss trials (T3, T4) perform worse | CORRECT (R2 ~0.22-0.24 vs ~0.51-0.60) |
| T8 improves over T7 by lowering dropout 0.3->0.2 | CORRECT (R2: 0.547->0.596) |
| 1,000 out of 10,000 MATSim scenarios used | CORRECT |
| LineGraph() transform used from PyTorch Geometric | CORRECT |
| T3 is the first trial with weighted loss | CORRECT |

---

## CORRECTED NARRATIVE (for meeting/thesis)

**Wrong:** "We tried batch sizes of 32, 32, 16, 64, 64, 64, 64 across trials."
**Correct:** T1=32, T2=16, T3=16, T4=unknown, T5-T8=**8**.

**Wrong:** "We used a 70/20/10 train/val/test split."
**Correct:** T1-T6 used **80/15/5**; T7-T8 used **80/10/10**.

**Wrong:** "T2 introduced weighted loss."
**Correct:** T2 used **standard loss**. T3 was the first trial with weighted loss.

**Wrong:** "T3 used dropout=0.3."
**Correct:** T3 used **dropout=0.0** (no dropout) — this was intentional to test weighted loss without regularization.

---

## ACTION ITEMS

1. Regenerate ALL figures using `VERIFIED_RESULTS_MASTER.csv` values
2. Mark `generate_thesis_charts.py` as DEPRECATED
3. Correct thesis draft: batch sizes 5th-8th trial = 8, not 32/64
4. Remove Temperature Scaling claim unless source file is found
5. Fix T2 description: standard loss, not weighted
6. Fix T3 description: dropout=0.0
7. Fix data split description: 80/15/5 and 80/10/10, not 70/20/10
8. Note that T1 uses different architecture and cannot be directly compared to T2-T8
