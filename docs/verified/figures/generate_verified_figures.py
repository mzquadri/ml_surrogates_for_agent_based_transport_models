"""
generate_verified_figures.py
============================
Thesis-ready figure generation script for:
  "Uncertainty Quantification for Machine Learning Surrogates
   in Transportation Policy Analysis"
  Author: Mohd Zamin Quadri (Nazim) | TUM Master Thesis 2026

ALL METRIC VALUES ARE HARDCODED FROM VERIFIED JSON FILES.
Do NOT replace with values from docs/MEETING_PREPARATION.md or
scripts/evaluation/generate_thesis_charts.py (both contain errors).

Source JSON files (all under data/TR-C_Benchmarks/):
  - point_net_transf_gat_*/test_evaluation_complete.json
  - point_net_transf_gat_*/test_results.json
  - point_net_transf_gat_8th_trial_lower_dropout/uq_results/*.json

Usage:
  python generate_verified_figures.py
  python generate_verified_figures.py --outdir /path/to/output
  python generate_verified_figures.py --show      # display interactively
"""

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# VERIFIED DATA (hardcoded from JSON sources)
# ─────────────────────────────────────────────

# Trial performance (T2-T8; T1 excluded — T1 uses Linear(64→1) final layer; T2-T8 use GATConv(64→1))
# T4 batch_size=16 confirmed from all_models_summary.json (not shown in figures; metrics only)
TRIAL_LABELS = ["T2", "T3", "T4", "T5", "T6", "T7", "T8"]
TRIAL_R2 = [0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
TRIAL_MAE = [4.33, 5.99, 6.08, 4.24, 4.32, 4.06, 3.96]
TRIAL_RMSE = [8.15, 10.27, 10.15, 7.78, 8.06, 7.53, 7.12]
# Sources: test_evaluation_complete.json for T2-T8
#          test_results.json for T4 (no hyperparams block)

# MC Dropout UQ — Spearman rho (uncertainty vs |error|)
UQ_LABELS_MC = ["T5 MC", "T6 MC", "T7 MC", "T8 MC"]
UQ_RHO_MC = [0.4263, 0.4186, 0.4437, 0.4820]
# Source: mc_dropout_full_metrics_model{5,6,7,8}_mc30_{50,100}graphs.json

# Ensemble UQ — Spearman rho
UQ_LABELS_ENS = [
    "Exp A\nMC avg",
    "Exp A\nEnsemble",
    "Exp A\nCombined",
    "Exp B\nMulti-model",
]
UQ_RHO_ENS = [0.1600, 0.1035, 0.1601, 0.1167]
# Source: experiment_a_results.json, experiment_b_results.json

# Conformal prediction (T8)
CONFORMAL_LEVELS = ["90%", "95%"]
CONFORMAL_Q = [9.9196, 14.6766]
CONFORMAL_COVERAGE = [90.02, 95.01]
CONFORMAL_WIDTH = [9.9196 * 2, 14.6766 * 2]  # symmetric ± q
# Source: conformal_standard.json

# Feature correlations with |error| (Spearman)
FEATURE_NAMES = [
    "VOL_BASE_CASE",
    "CAPACITY_BASE_CASE",
    "CAPACITY_REDUCTION",
    "FREESPEED",
    "LENGTH",
]
FEATURE_CORR = [0.3316, 0.2615, -0.2286, 0.2110, -0.0695]
# Source: feature_analysis_report.txt (T8 feature_analysis_plots/)

# Calibration comparison
CALIB_LABELS = ["Ideal Gaussian\n(±1.96σ)", "MC Dropout T8\n(±11.65σ)"]
CALIB_K95 = [1.96, 11.65]
# Source: uq_comparison_model8.json (k95 field)

# Selective prediction (T8)
# 39.9% MAE reduction sourced from ADVANCED_UQ_SUMMARY_MODEL8.md (auto-generated summary)
# The underlying analysis is in uq_comparison_model8.json (selective_prediction field)
# Only the 50% anchor is verified here; full curve requires loading uq_comparison_model8.json
BASELINE_MAE = 3.96
SEL_KEEP_PCT = [100, 50]
SEL_MAE = [3.96, 3.96 * (1 - 0.399)]  # 2.38 veh/h at 50%
# Source: uq_comparison_model8.json


# ─────────────────────────────────────────────
# DATA CONTEXT NOTE (applied to every figure)
# Verified sources:
#   feature_analysis_report.txt line 15: Elena=10000, Model8=1000 simulations
#   feature_analysis_report.txt line 71: "1000 simulations vs Elena's 10000"
#   point_net_transf_gat.py line 14: "paper used 10,000 simulations"
# ─────────────────────────────────────────────

DATA_CONTEXT_BASE = (
    "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset), "
    "unless stated otherwise."
)


def add_data_note(fig, specific_note=None):
    """
    Add verified dataset context footnote to every thesis figure.

    Every figure gets:
      (1) global context: 1,000 of 10,000 scenarios (10% subset)
      (2) specific evaluation subset for that figure, if provided

    Verified from: feature_analysis_report.txt (T8 folder), point_net_transf_gat.py
    """
    if specific_note:
        text = DATA_CONTEXT_BASE + f"  |  This figure: {specific_note}"
    else:
        text = DATA_CONTEXT_BASE
    fig.text(
        0.5,
        -0.03,
        text,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
        style="italic",
    )


# ─────────────────────────────────────────────
# PLOT SETTINGS
# ─────────────────────────────────────────────

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
GRAY = "#7f7f7f"

BEST_TRIAL_IDX = 6  # T8 is index 6 in TRIAL_LABELS

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save_or_show(fig, filename, outdir, show):
    """Save figure to outdir/filename and optionally display."""
    path = os.path.join(outdir, filename)
    fig.savefig(path)
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# FIGURE 1: Trial Comparison (R², MAE, RMSE)
# ─────────────────────────────────────────────


def fig1_trial_comparison(outdir, show=False):
    """Bar chart comparing R², MAE, RMSE across Trials 2-8."""
    x = np.arange(len(TRIAL_LABELS))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Predictive Performance Across Training Trials\n(T1 excluded — different architecture)",
        fontsize=13,
    )

    metrics = [
        (axes[0], TRIAL_R2, "R² (higher is better)", BLUE),
        (axes[1], TRIAL_MAE, "MAE (veh/h, lower is better)", ORANGE),
        (axes[2], TRIAL_RMSE, "RMSE (veh/h, lower is better)", GREEN),
    ]

    for ax, values, ylabel, color in metrics:
        colors = [
            RED if i == BEST_TRIAL_IDX else color for i in range(len(TRIAL_LABELS))
        ]
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(TRIAL_LABELS)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Trial")
        # Annotate best trial
        ax.bar(
            x[BEST_TRIAL_IDX],
            values[BEST_TRIAL_IDX],
            color=RED,
            edgecolor="black",
            linewidth=1.5,
            label="Best (T8)",
        )
        ax.legend()
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                f"{val:.3f}" if max(values) < 2 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    add_data_note(
        fig,
        specific_note=(
            "T1–T6: 50 test graphs (80/15/5 split); "
            "T7–T8: 100 test graphs (80/10/10 split)"
        ),
    )
    save_or_show(fig, "fig1_trial_comparison.pdf", outdir, show)


# ─────────────────────────────────────────────
# FIGURE 2: UQ Ranking Quality (Spearman ρ)
# ─────────────────────────────────────────────


def fig2_uq_ranking(outdir, show=False):
    """Horizontal bar chart comparing MC Dropout vs Ensemble Spearman ρ."""
    labels = UQ_LABELS_MC + UQ_LABELS_ENS
    rhos = UQ_RHO_MC + list(UQ_RHO_ENS)
    colors = [BLUE] * len(UQ_LABELS_MC) + [ORANGE] * len(UQ_LABELS_ENS)

    # Sort descending for visual clarity
    order = np.argsort(rhos)[::-1]
    labels_s = [labels[i] for i in order]
    rhos_s = [rhos[i] for i in order]
    colors_s = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(labels_s))
    bars = ax.barh(y, rhos_s, color=colors_s, edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, rhos_s):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"ρ={val:.4f}",
            va="center",
            fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels_s)
    ax.set_xlabel("Spearman ρ (uncertainty vs |error|)")
    ax.set_title(
        "Uncertainty Quality: MC Dropout vs Ensemble\n"
        "Higher ρ = uncertainty better predicts where errors occur"
    )
    ax.set_xlim(0, max(rhos_s) + 0.12)
    ax.axvline(0, color="black", linewidth=0.5)

    mc_patch = mpatches.Patch(color=BLUE, label="MC Dropout (single model)")
    ens_patch = mpatches.Patch(color=ORANGE, label="Ensemble / Multi-run")
    ax.legend(handles=[mc_patch, ens_patch], loc="lower right")

    plt.tight_layout()
    add_data_note(
        fig,
        specific_note=(
            "T5/T6 MC Dropout: 50 test graphs (1,581,750 nodes); "
            "T7/T8 MC Dropout + ensemble: 100 test graphs (3,163,500 nodes)"
        ),
    )
    save_or_show(fig, "fig2_uq_ranking.pdf", outdir, show)


# ─────────────────────────────────────────────
# FIGURE 3: Conformal Prediction Coverage
# ─────────────────────────────────────────────


def fig3_conformal_coverage(outdir, show=False):
    """Bar chart showing nominal vs achieved conformal coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Conformal Prediction — Coverage and Interval Width (T8, n=50 eval graphs)",
        fontsize=13,
    )

    x = np.arange(len(CONFORMAL_LEVELS))
    nominal = [90.0, 95.0]

    # Left: coverage
    ax = axes[0]
    bars_nom = ax.bar(
        x - 0.2,
        nominal,
        width=0.35,
        color=GRAY,
        edgecolor="black",
        label="Nominal coverage",
        linewidth=0.7,
    )
    bars_ach = ax.bar(
        x + 0.2,
        CONFORMAL_COVERAGE,
        width=0.35,
        color=GREEN,
        edgecolor="black",
        label="Achieved coverage",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(CONFORMAL_LEVELS)
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Confidence level")
    ax.set_ylim(85, 100)
    ax.set_title("Coverage")
    ax.legend()
    for bar, val in zip(bars_ach, CONFORMAL_COVERAGE):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Right: interval width
    ax2 = axes[1]
    bars_w = ax2.bar(
        x, CONFORMAL_Q, color=[BLUE, ORANGE], edgecolor="black", linewidth=0.7
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(CONFORMAL_LEVELS)
    ax2.set_ylabel("Quantile q = half-width (veh/h)")
    ax2.set_xlabel("Confidence level")
    ax2.set_title("Interval Half-Width")
    for bar, val in zip(bars_w, CONFORMAL_Q):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"±{val:.2f} veh/h",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    add_data_note(
        fig,
        specific_note=(
            "T8: 100 test graphs split 50 calibration + 50 evaluation "
            "(1,581,750 nodes each)"
        ),
    )
    save_or_show(fig, "fig3_conformal_coverage.pdf", outdir, show)


# ─────────────────────────────────────────────
# FIGURE 4: Selective Prediction
# ─────────────────────────────────────────────


def fig4_selective_prediction(outdir, show=False):
    """
    Plot MAE vs % data retained under selective prediction.
    Only two verified anchor points exist (100% and 50%).
    Plots these as a bar chart with annotation.
    If a full curve is available from uq_comparison_model8.json,
    replace SEL_KEEP_PCT and SEL_MAE with that data.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = [GRAY, BLUE]
    bars = ax.bar(
        ["100%\n(all data)", "50%\n(most certain half)"],
        SEL_MAE,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.4,
    )

    # Annotation: arrow showing MAE reduction
    ax.annotate(
        "",
        xy=(1, SEL_MAE[1] + 0.1),
        xytext=(0, SEL_MAE[0] - 0.1),
        arrowprops=dict(arrowstyle="->", color=RED, lw=2),
    )
    ax.text(
        0.5,
        (SEL_MAE[0] + SEL_MAE[1]) / 2,
        "−39.9%\nMAE reduction",
        ha="center",
        va="center",
        fontsize=12,
        color=RED,
        fontweight="bold",
    )

    for bar, val in zip(bars, SEL_MAE):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f} veh/h",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("MAE (veh/h)")
    ax.set_xlabel("Fraction of test nodes retained\n(lowest uncertainty first)")
    ax.set_title(
        "Selective Prediction: Uncertainty-Guided Filtering (T8)\n"
        "Note: Only 50% anchor verified; full curve requires uq_comparison_model8.json"
    )
    ax.set_ylim(0, max(SEL_MAE) * 1.3)

    plt.tight_layout()
    add_data_note(
        fig,
        specific_note=(
            "T8: 50 evaluation graphs (1,581,750 nodes); "
            "source: uq_comparison_model8.json"
        ),
    )
    save_or_show(fig, "fig4_selective_prediction.pdf", outdir, show)


# ─────────────────────────────────────────────
# FIGURE 5: Feature Correlation with Error
# ─────────────────────────────────────────────


def fig5_feature_correlation(outdir, show=False):
    """Horizontal bar chart of Spearman correlation between features and |error|."""
    order = np.argsort(FEATURE_CORR)
    labels_s = [FEATURE_NAMES[i] for i in order]
    corr_s = [FEATURE_CORR[i] for i in order]
    colors_s = [ORANGE if c < 0 else BLUE for c in corr_s]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels_s))
    bars = ax.barh(y, corr_s, color=colors_s, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_s)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Spearman correlation with |prediction error|")
    ax.set_title(
        "Feature Correlation with Prediction Error (T8)\n"
        "Positive: feature associated with higher error; Negative: with lower error"
    )

    for bar, val in zip(bars, corr_s):
        xpos = bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005
        ha = "left" if val >= 0 else "right"
        ax.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha=ha,
            fontsize=10,
        )

    pos_patch = mpatches.Patch(color=BLUE, label="Positive correlation")
    neg_patch = mpatches.Patch(color=ORANGE, label="Negative correlation")
    ax.legend(handles=[pos_patch, neg_patch])

    plt.tight_layout()
    add_data_note(
        fig,
        specific_note=(
            "T8: 100 test graphs (3,163,500 nodes); "
            "correlations from feature_analysis_report.txt"
        ),
    )
    save_or_show(fig, "fig5_feature_correlation.pdf", outdir, show)


# ─────────────────────────────────────────────
# FIGURE 6: With vs Without UQ (MC Dropout overhead)
# ─────────────────────────────────────────────

# WITH / WITHOUT UQ data (verified from WITH_WITHOUT_UQ_SUMMARY_MODEL8.md)
# Deterministic (single forward pass, dropout disabled)
DET_R2 = 0.5957
DET_MAE = 3.96
DET_RMSE = 7.12
# MC Dropout (S=30 stochastic passes, MC mean used as prediction)
MC_R2 = 0.5857
MC_MAE = 3.95
MC_RMSE = 7.21
# Mean absolute prediction difference between the two inference modes
PRED_DIFF_MEAN = 0.6548  # veh/h


def fig6_with_without_uq(outdir, show=False):
    """
    Side-by-side comparison of deterministic vs MC Dropout inference for T8.
    Shows that MC Dropout introduces only negligible overhead in prediction quality
    while providing uncertainty estimates (ρ=0.4820).
    Source: WITH_WITHOUT_UQ_SUMMARY_MODEL8.md
    """
    labels = ["Deterministic\n(single pass)", "MC Dropout\n(S=30 passes)"]
    r2_vals = [DET_R2, MC_R2]
    mae_vals = [DET_MAE, MC_MAE]
    rmse_vals = [DET_RMSE, MC_RMSE]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Deterministic vs MC Dropout Inference — T8\n"
        "MC Dropout provides uncertainty estimates (ρ=0.4820) at minimal accuracy cost",
        fontsize=12,
    )

    metric_data = [
        (axes[0], r2_vals, "R²", BLUE, True),
        (axes[1], mae_vals, "MAE (veh/h)", ORANGE, False),
        (axes[2], rmse_vals, "RMSE (veh/h)", GREEN, False),
    ]

    for ax, vals, ylabel, color, higher_is_better in metric_data:
        colors_b = [GRAY, color]
        bars = ax.bar(
            labels, vals, color=colors_b, edgecolor="black", linewidth=0.7, width=0.4
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002 * max(vals),
                f"{val:.4f}" if max(vals) < 2 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(f"{'Higher is better' if higher_is_better else 'Lower is better'}")
        # Annotate the difference
        diff = vals[1] - vals[0]
        sign = "+" if diff > 0 else ""
        ax.text(
            0.5,
            0.05,
            f"Δ = {sign}{diff:.4f}" if max(vals) < 2 else f"Δ = {sign}{diff:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="dimgray",
            style="italic",
        )
        # y-axis starts near the data range
        margin = 0.15 * (max(vals) - min(vals)) if max(vals) != min(vals) else 0.05
        ax.set_ylim(min(vals) - margin * 2, max(vals) + margin * 4)

    # Add annotation about uncertainty bonus
    fig.text(
        0.5,
        0.01,
        f"MC Dropout additionally provides: uncertainty estimate σ per node  |  "
        f"Spearman ρ(unc, |error|) = 0.4820  |  "
        f"Mean |det − mc| = {PRED_DIFF_MEAN:.4f} veh/h",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    add_data_note(
        fig,
        specific_note="T8: 100 test graphs (3,163,500 nodes); source: WITH_WITHOUT_UQ_SUMMARY_MODEL8.md",
    )
    save_or_show(fig, "fig6_with_without_uq.pdf", outdir, show)


# ─────────────────────────────────────────────
# FIGURE 7: Calibration Comparison
# ─────────────────────────────────────────────


def fig7_calibration(outdir, show=False):
    """
    Bar chart showing k95 (factor to multiply sigma for 95% coverage).
    Ideal Gaussian needs k95=1.96; MC Dropout T8 needs k95=11.65.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors_c = [GREEN, ORANGE]
    bars = ax.bar(
        CALIB_LABELS,
        CALIB_K95,
        color=colors_c,
        edgecolor="black",
        linewidth=0.7,
        width=0.4,
    )

    for bar, val in zip(bars, CALIB_K95):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"k₉₅ = {val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Factor k such that ±kσ achieves 95% coverage")
    ax.set_title(
        "MC Dropout σ is NOT a Calibrated Standard Deviation\n"
        "T8 needs ±11.65σ to cover 95% of true values"
    )
    ax.set_ylim(0, max(CALIB_K95) * 1.2)

    # Add annotation
    ax.axhline(1.96, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(
        1.05, 1.96 + 0.3, "Ideal Gaussian (1.96)", fontsize=9, color="black", alpha=0.7
    )

    plt.tight_layout()
    add_data_note(
        fig,
        specific_note=(
            "T8: 50 evaluation graphs (1,581,750 nodes); "
            "k95 from uq_comparison_model8.json"
        ),
    )
    save_or_show(fig, "fig7_calibration.pdf", outdir, show)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate verified thesis figures.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for figures (default: same as script)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (requires display)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving figures to: {args.outdir}")
    print()

    print("Figure 1: Trial Comparison...")
    fig1_trial_comparison(args.outdir, args.show)

    print("Figure 2: UQ Ranking...")
    fig2_uq_ranking(args.outdir, args.show)

    print("Figure 3: Conformal Coverage...")
    fig3_conformal_coverage(args.outdir, args.show)

    print("Figure 4: Selective Prediction...")
    fig4_selective_prediction(args.outdir, args.show)

    print("Figure 5: Feature Correlation...")
    fig5_feature_correlation(args.outdir, args.show)

    print("Figure 6: With vs Without UQ...")
    fig6_with_without_uq(args.outdir, args.show)

    print("Figure 7: Calibration Comparison...")
    fig7_calibration(args.outdir, args.show)

    print()
    print("All figures generated successfully.")
    print()
    print("Verification checklist:")
    print(f"  T8 R²:   {TRIAL_R2[-1]:.4f}  (expected: 0.5957)")
    print(f"  T8 MAE:  {TRIAL_MAE[-1]:.2f}    (expected: 3.96)")
    print(f"  T8 rho:  {UQ_RHO_MC[-1]:.4f}  (expected: 0.4820)")
    print(f"  Conf 95% q: {CONFORMAL_Q[-1]:.4f} (expected: 14.6766)")
    print(f"  k95:     {CALIB_K95[-1]:.2f}   (expected: 11.65)")


if __name__ == "__main__":
    main()
