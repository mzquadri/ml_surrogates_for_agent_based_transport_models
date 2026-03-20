# Error Detection Analysis — Trial 8

    ## Metadata

    - **Source file:** `trial8_uq_ablation_results.csv`
    - **Trial:** T8 — `point_net_transf_gat_8th_trial_lower_dropout`
    - **Architecture:** PointNetTransfGAT, GATConv(64→1) final layer, dropout=0.2
    - **MC samples used:** 30 forward passes per node
    - **Data scope:** Results based on 1,000 of 10,000 available MATSim scenarios (10% subset), Trial 8 test set.
    - **T1 note:** Trial 1 used Linear(64→1) final layer and is NOT comparable to T2–T8. All results here are T8 only.

    ---

    ## Metric Definitions

    - **Score:** `pred_mc_std` (MC Dropout σ) — higher σ = model predicts this node as more uncertain
    - **Label (top-10%):** binary flag = 1 if `abs_error_det` ≥ 90th percentile of all absolute errors
    - **Label (top-20%):** binary flag = 1 if `abs_error_det` ≥ 80th percentile of all absolute errors
    - **AUROC:** Area Under the ROC Curve — probability that σ ranks a bad prediction above a good one
    - **AUPRC:** Area Under the Precision-Recall Curve — precision-weighted detection quality
    - **Random AUROC baseline:** 0.500 (by definition)
    - **Random AUPRC baseline:** equal to the positive class rate (≈ 0.10 for top-10%, ≈ 0.20 for top-20%)

    ---

    ## Exact Thresholds Used

    | Threshold | Percentile | Cutoff (veh/h) | n positive | % of total |
    |---|---|---|---|---|
    | Top-10% errors | 90th percentile of abs_error_det | 9.9305 | 316,350 | 10.0% |
    | Top-20% errors | 80th percentile of abs_error_det | 6.0129 | 632,700 | 20.0% |

    ---

    ## Result Table

    | Threshold | Cutoff (veh/h) | AUROC | AUPRC | Random AUROC | Random AUPRC |
    |---|---|---|---|---|---|
    | Top-10% errors | 9.9305 | **0.7585** | **0.3148** | 0.500 | 0.100 |
    | Top-20% errors | 6.0129 | **0.7401** | **0.4547** | 0.500 | 0.200 |

    ---

    ## Thesis Usage

    **Safe to include:** YES

    ### Sentence you CAN write:
    > "To assess the operational utility of MC Dropout uncertainty, we treat prediction
    > quality as a binary detection task: nodes whose absolute deterministic error
    > exceeds the 90th percentile are labelled as high-error, and σ is used as the
    > detection score. MC Dropout achieves an AUROC of 0.7585 and AUPRC of
    > 0.3148 for the top-10% error threshold, substantially above the random
    > baselines of 0.500 and 0.100 respectively, confirming that σ
    > carries operational utility for identifying unreliable predictions on the
    > Trial 8 test set."

    ### Sentence to AVOID:
    > Do NOT write "σ reliably identifies all high-error predictions" — AUROC and AUPRC
    > measure ranking quality across all thresholds, not precision at any single
    > operating point. Do NOT present this as a calibration result; it is a ranking
    > evaluation. Do NOT compare these values across trials without re-running the
    > identical analysis for each trial separately.

    ---

    ## Figure

    `docs/verified/figures/t8_error_detection_auroc.pdf`  
    `docs/verified/figures/t8_error_detection_auroc.png`
    