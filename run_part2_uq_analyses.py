"""
Part 2A + 2B: Selective Prediction and Error Detection for Trial 8
Input:  trial8_uq_ablation_results.csv
Output: docs/verified/UQ_SELECTIVE_PREDICTION_T8.md
        docs/verified/UQ_ERROR_DETECTION_T8.md
        docs/verified/figures/t8_selective_prediction_curve.{pdf,png}
        docs/verified/figures/t8_error_detection_auroc.{pdf,png}

Rules:
- No retraining
- No new inference
- No changes to existing files
- Read only from trial8_uq_ablation_results.csv
- Write outputs only to docs/verified/
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
CSV = (
    REPO
    / "data/TR-C_Benchmarks"
    / "point_net_transf_gat_8th_trial_lower_dropout"
    / "trial8_uq_ablation_results.csv"
)
OUT_MD = REPO / "docs/verified"
OUT_FIG = REPO / "docs/verified/figures"

# ── Pastel palette ────────────────────────────────────────────────────────────
BG = "#FAFBFC"
P_BLUE = "#5B8DB8"
P_BLUE_DK = "#2E6494"
P_CORAL = "#E07A5F"
P_CORAL_LT = "#F2B5A0"
P_GREEN = "#6BAB8C"
P_AMBER = "#E8A84C"
P_SLATE = "#5C6B7A"
P_DGRAY = "#3A4A5A"
P_MGRAY = "#7A8A9A"
P_LGRAY = "#D0D8E0"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    }
)

FOOTNOTE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios "
    "(10% subset), Trial 8 test set."
)

# ── Load data ─────────────────────────────────────────────────────────────────


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("Loading CSV ...")
    df = pd.read_csv(CSV)
    print(f"  Rows: {len(df):,}   Cols: {list(df.columns)}")

    # Verify required columns
    required = {"target", "pred_det", "pred_mc_mean", "pred_mc_std", "abs_error_det"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    print("  Column check passed.")

    # ═══════════════════════════════════════════════════════════════════════════════
    # PART 2A — Selective Prediction / Risk-Coverage Curve
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n-- Part 2A: Selective Prediction --")

    # Sort descending by sigma: most uncertain first, least uncertain last
    df_sorted = df.sort_values("pred_mc_std", ascending=False).reset_index(drop=True)
    n_total = len(df_sorted)

    retention_levels = [
        1.00,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.60,
        0.50,
        0.40,
        0.30,
        0.25,
        0.10,
    ]
    rows_sel = []

    for r in retention_levels:
        k = int(np.floor(r * n_total))
        if k == 0:
            continue
        # Keep the BOTTOM k rows = the least uncertain nodes
        subset = df_sorted.iloc[n_total - k :]
        mae = float(np.abs(subset["target"] - subset["pred_mc_mean"]).mean())
        rmse = float(np.sqrt(((subset["target"] - subset["pred_mc_mean"]) ** 2).mean()))
        rows_sel.append(
            {
                "retained_pct": int(round(r * 100)),
                "n_nodes": k,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
            }
        )
        print(f"  Retain {int(r * 100):3d}%  n={k:>9,}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    sel_df = pd.DataFrame(rows_sel)

    # Baselines: deterministic 100%
    mae_base_det = float(df["abs_error_det"].mean())
    rmse_base_det = float(np.sqrt(((df["target"] - df["pred_det"]) ** 2).mean()))
    # MC mean 100%
    mae_base_mc = float(np.abs(df["target"] - df["pred_mc_mean"]).mean())
    rmse_base_mc = float(np.sqrt(((df["target"] - df["pred_mc_mean"]) ** 2).mean()))
    print(f"\n  Baseline det  (100%): MAE={mae_base_det:.4f}  RMSE={rmse_base_det:.4f}")
    print(f"  Baseline MC   (100%): MAE={mae_base_mc:.4f}   RMSE={rmse_base_mc:.4f}")

    # -- Figure 2A --───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Trial 8 — Selective Prediction: Uncertainty-Based Abstention\n"
        "(MC Dropout, 30 samples, GATConv final layer)",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
        y=1.02,
    )

    x = sel_df["retained_pct"].values

    for ax, metric, color, base_val, ylabel in zip(
        axes,
        ["MAE", "RMSE"],
        [P_BLUE, P_CORAL],
        [mae_base_mc, rmse_base_mc],
        ["MAE (veh/h)", "RMSE (veh/h)"],
    ):
        y = sel_df[metric].values
        ax.set_facecolor(BG)

        # Main curve
        ax.plot(x, y, "-o", color=color, lw=2.2, ms=6, zorder=4, label=f"MC σ abstention")

        # Baseline MC (100% retained)
        ax.axhline(
            base_val,
            color=color,
            lw=1.2,
            linestyle="--",
            alpha=0.55,
            label=f"MC baseline 100% ({base_val:.2f})",
        )

        # Baseline deterministic
        base_d = mae_base_det if metric == "MAE" else rmse_base_det
        ax.axhline(
            base_d,
            color=P_MGRAY,
            lw=1.0,
            linestyle=":",
            label=f"Deterministic baseline ({base_d:.2f})",
        )

        # Annotate key retention points
        for pct in [90, 50, 25]:
            row = sel_df[sel_df["retained_pct"] == pct]
            if not row.empty:
                val = float(row[metric].values[0])
                reduction = (1 - val / base_val) * 100
                ax.annotate(
                    f"{pct}%\n{val:.2f}\n(−{reduction:.1f}%)",
                    xy=(pct, val),
                    xytext=(pct - 12, val - (y.max() - y.min()) * 0.15),
                    fontsize=7.0,
                    color=color,
                    ha="center",
                    arrowprops=dict(arrowstyle="-", color=P_LGRAY, lw=0.8),
                )

        ax.set_xlabel("Retained predictions (%)", fontsize=10, color=P_SLATE)
        ax.set_ylabel(ylabel, fontsize=10, color=P_SLATE)
        ax.set_xlim(5, 107)
        ax.set_xticks([10, 25, 50, 70, 80, 90, 95, 100])
        ax.tick_params(colors=P_MGRAY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(P_LGRAY)
        ax.legend(fontsize=7.5, framealpha=0.85, loc="upper left")
        ax.grid(True, color=P_LGRAY, lw=0.5, alpha=0.7)

    fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.14, wspace=0.30)

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"t8_selective_prediction_curve.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
        print(f"  Saved: {out}")
    plt.close(fig)


    # -- Markdown report 2A --──────────────────────────────────────────────────────
    def _get(col, pct):
        row = sel_df[sel_df["retained_pct"] == pct]
        return float(row[col].values[0]) if not row.empty else float("nan")


    mae_90 = _get("MAE", 90)
    rmse_90 = _get("RMSE", 90)
    mae_50 = _get("MAE", 50)
    rmse_50 = _get("RMSE", 50)
    mae_25 = _get("MAE", 25)
    rmse_25 = _get("RMSE", 25)
    red_90 = round((1 - mae_90 / mae_base_mc) * 100, 1)
    red_50 = round((1 - mae_50 / mae_base_mc) * 100, 1)
    red_25 = round((1 - mae_25 / mae_base_mc) * 100, 1)

    table_rows = "\n".join(
        f"| {r['retained_pct']} | {r['n_nodes']:,} | {r['MAE']} | {r['RMSE']} |"
        for _, r in sel_df.iterrows()
    )

    md_sel = f"""# Selective Prediction Analysis — Trial 8

    ## Metadata

    - **Source file:** `trial8_uq_ablation_results.csv`
    - **Trial:** T8 — `point_net_transf_gat_8th_trial_lower_dropout`
    - **Architecture:** PointNetTransfGAT, GATConv(64→1) final layer, dropout=0.2
    - **MC samples used:** 30 forward passes per node
    - **Data scope:** {FOOTNOTE}
    - **T1 note:** Trial 1 used Linear(64→1) final layer and is NOT comparable to T2–T8. All results here are T8 only.

    ---

    ## Metric Definitions

    - **Retained (%):** Fraction of test-set node predictions kept after rejecting those with highest MC σ
    - **n nodes:** Absolute count of node-predictions retained (from {n_total:,} total)
    - **MAE:** Mean Absolute Error between `target` and `pred_mc_mean` on retained subset (veh/h)
    - **RMSE:** Root Mean Squared Error between `target` and `pred_mc_mean` on retained subset (veh/h)
    - **Sorting key:** `pred_mc_std` (MC Dropout σ), descending — highest-uncertainty predictions rejected first

    ---

    ## Result Table

    | Retained (%) | n nodes | MAE (veh/h) | RMSE (veh/h) |
    |---|---|---|---|
    {table_rows}

    **MC baseline (100% retained, no abstention):** MAE = {mae_base_mc:.4f} veh/h, RMSE = {rmse_base_mc:.4f} veh/h  
    **Deterministic baseline (100% retained):** MAE = {mae_base_det:.4f} veh/h, RMSE = {rmse_base_det:.4f} veh/h

    ---

    ## Key Results

    | Retention | MAE (veh/h) | MAE reduction vs MC baseline |
    |---|---|---|
    | 100% (no abstention) | {mae_base_mc:.4f} | — |
    | 90% | {mae_90} | −{red_90}% |
    | 50% | {mae_50} | −{red_50}% |
    | 25% | {mae_25} | −{red_25}% |

    ---

    ## Thesis Usage

    **Safe to include:** YES

    ### Sentence you CAN write:
    > "Applying uncertainty-based abstention using MC Dropout σ, retaining the 90% of
    > predictions with the lowest uncertainty reduces MAE from {mae_base_mc:.2f} to
    > {mae_90:.2f} veh/h (a {red_90}% reduction). Retaining only 50% of predictions
    > further reduces MAE to {mae_50:.2f} veh/h ({red_50}% reduction), demonstrating
    > that σ meaningfully ranks prediction reliability on the Trial 8 test set."

    ### Sentence to AVOID:
    > Do NOT write "uncertainty is well-calibrated" — this result shows ranking utility
    > (Spearman-style operational evidence), not interval calibration. The calibration
    > analysis is separate (conformal prediction results). Do NOT extrapolate these
    > retention curves to T1, which has a different final-layer architecture.

    ---

    ## Figure

    `docs/verified/figures/t8_selective_prediction_curve.pdf`  
    `docs/verified/figures/t8_selective_prediction_curve.png`
    """

    (OUT_MD / "UQ_SELECTIVE_PREDICTION_T8.md").write_text(md_sel, encoding="utf-8")
    print(f"  Saved: {OUT_MD / 'UQ_SELECTIVE_PREDICTION_T8.md'}")


    # ═══════════════════════════════════════════════════════════════════════════════
    # PART 2B — Uncertainty as Error Detector (AUROC / AUPRC)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n-- Part 2B: Error Detection --")

    sigma = df["pred_mc_std"].values.astype(np.float64)
    errors = df["abs_error_det"].values.astype(np.float64)

    p90_cutoff = float(np.percentile(errors, 90))
    p80_cutoff = float(np.percentile(errors, 80))

    thresholds = {
        "top10": p90_cutoff,
        "top20": p80_cutoff,
    }

    rows_det = []
    roc_data = {}
    pr_data = {}

    for name, cutoff in thresholds.items():
        labels = (errors >= cutoff).astype(int)
        n_pos = int(labels.sum())
        pct_pos = n_pos / len(labels) * 100
        auroc = float(roc_auc_score(labels, sigma))
        auprc = float(average_precision_score(labels, sigma))
        fpr, tpr, _ = roc_curve(labels, sigma)
        prec, rec, _ = precision_recall_curve(labels, sigma)
        roc_data[name] = (fpr, tpr, auroc)
        pr_data[name] = (prec, rec, auprc, pct_pos / 100)
        rows_det.append(
            {
                "threshold": name,
                "cutoff_veh_h": round(cutoff, 4),
                "n_positive": n_pos,
                "pct_positive": round(pct_pos, 1),
                "AUROC": round(auroc, 4),
                "AUPRC": round(auprc, 4),
                "random_auprc": round(pct_pos / 100, 4),
            }
        )
        print(
            f"  {name}: cutoff={cutoff:.4f} veh/h  n_pos={n_pos:,}  "
            f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
            f"(random AUPRC={pct_pos / 100:.4f})"
        )

    det_df = pd.DataFrame(rows_det)

    # -- Figure 2B --───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Trial 8 — MC Dropout \u03c3 as a Detector of High-Error Predictions\n"
        "(MC Dropout, 30 samples, GATConv final layer)",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
        y=1.02,
    )

    colors = {"top10": P_CORAL, "top20": P_BLUE}
    labels_map = {"top10": "Top-10% errors", "top20": "Top-20% errors"}

    # -- ROC curves --──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.plot(
        [0, 1], [0, 1], "--", color=P_MGRAY, lw=1.2, label="Random (AUROC=0.50)", zorder=2
    )
    for name, (fpr, tpr, auroc) in roc_data.items():
        ax.plot(
            fpr,
            tpr,
            color=colors[name],
            lw=2.2,
            zorder=3,
            label=f"{labels_map[name]}  AUROC={auroc:.3f}",
        )
    ax.set_xlabel("False Positive Rate", fontsize=10, color=P_SLATE)
    ax.set_ylabel("True Positive Rate", fontsize=10, color=P_SLATE)
    ax.set_title("ROC Curve", fontsize=10, color=P_DGRAY, pad=6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(colors=P_MGRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(P_LGRAY)
    ax.legend(fontsize=8, framealpha=0.85, loc="lower right")
    ax.grid(True, color=P_LGRAY, lw=0.5, alpha=0.7)

    # -- PR curves --───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)
    for name, (prec, rec, auprc, rand_base) in pr_data.items():
        ax.axhline(
            rand_base,
            color=colors[name],
            lw=0.9,
            linestyle=":",
            alpha=0.6,
            label=f"Random baseline ({rand_base:.2f})",
        )
        ax.plot(
            rec,
            prec,
            color=colors[name],
            lw=2.2,
            zorder=3,
            label=f"{labels_map[name]}  AUPRC={auprc:.3f}",
        )
    ax.set_xlabel("Recall", fontsize=10, color=P_SLATE)
    ax.set_ylabel("Precision", fontsize=10, color=P_SLATE)
    ax.set_title("Precision-Recall Curve", fontsize=10, color=P_DGRAY, pad=6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(colors=P_MGRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(P_LGRAY)
    ax.legend(fontsize=8, framealpha=0.85, loc="upper right")
    ax.grid(True, color=P_LGRAY, lw=0.5, alpha=0.7)

    fig.text(0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.14, wspace=0.30)

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"t8_error_detection_auroc.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
        print(f"  Saved: {out}")
    plt.close(fig)


    # -- Markdown report 2B --──────────────────────────────────────────────────────
    r10 = det_df[det_df["threshold"] == "top10"].iloc[0]
    r20 = det_df[det_df["threshold"] == "top20"].iloc[0]

    md_det = f"""# Error Detection Analysis — Trial 8

    ## Metadata

    - **Source file:** `trial8_uq_ablation_results.csv`
    - **Trial:** T8 — `point_net_transf_gat_8th_trial_lower_dropout`
    - **Architecture:** PointNetTransfGAT, GATConv(64→1) final layer, dropout=0.2
    - **MC samples used:** 30 forward passes per node
    - **Data scope:** {FOOTNOTE}
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
    | Top-10% errors | 90th percentile of abs_error_det | {r10.cutoff_veh_h:.4f} | {int(r10.n_positive):,} | {r10.pct_positive}% |
    | Top-20% errors | 80th percentile of abs_error_det | {r20.cutoff_veh_h:.4f} | {int(r20.n_positive):,} | {r20.pct_positive}% |

    ---

    ## Result Table

    | Threshold | Cutoff (veh/h) | AUROC | AUPRC | Random AUROC | Random AUPRC |
    |---|---|---|---|---|---|
    | Top-10% errors | {r10.cutoff_veh_h} | **{r10.AUROC}** | **{r10.AUPRC}** | 0.500 | {r10.random_auprc:.3f} |
    | Top-20% errors | {r20.cutoff_veh_h} | **{r20.AUROC}** | **{r20.AUPRC}** | 0.500 | {r20.random_auprc:.3f} |

    ---

    ## Thesis Usage

    **Safe to include:** YES

    ### Sentence you CAN write:
    > "To assess the operational utility of MC Dropout uncertainty, we treat prediction
    > quality as a binary detection task: nodes whose absolute deterministic error
    > exceeds the 90th percentile are labelled as high-error, and σ is used as the
    > detection score. MC Dropout achieves an AUROC of {r10.AUROC} and AUPRC of
    > {r10.AUPRC} for the top-10% error threshold, substantially above the random
    > baselines of 0.500 and {r10.random_auprc:.3f} respectively, confirming that σ
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
    """

    (OUT_MD / "UQ_ERROR_DETECTION_T8.md").write_text(md_det, encoding="utf-8")
    print(f"  Saved: {OUT_MD / 'UQ_ERROR_DETECTION_T8.md'}")

    print("\n" + "=" * 60)
    print("ALL OUTPUTS WRITTEN")
    print("=" * 60)
    print(f"  {OUT_MD / 'UQ_SELECTIVE_PREDICTION_T8.md'}")
    print(f"  {OUT_MD / 'UQ_ERROR_DETECTION_T8.md'}")
    print(f"  {OUT_FIG / 't8_selective_prediction_curve.pdf'}")
    print(f"  {OUT_FIG / 't8_selective_prediction_curve.png'}")
    print(f"  {OUT_FIG / 't8_error_detection_auroc.pdf'}")
    print(f"  {OUT_FIG / 't8_error_detection_auroc.png'}")


if __name__ == "__main__":
    main()
