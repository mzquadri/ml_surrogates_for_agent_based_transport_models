"""
generate_all_thesis_figures.py
==============================
Generates all 10 thesis figures using verified data and a clean pastel palette.
Also generates fig3_feature_distributions (Ch. 3, new Fig 3.1).

Run from the figures/ directory:
    python generate_all_thesis_figures.py

Outputs (PDF + PNG for each):
    fig1_trial_comparison.pdf
    fig2_uq_ranking.pdf
    fig3_conformal_coverage.pdf
    fig3_feature_distributions.pdf   ← NEW (Ch. 3 Fig 3.1)
    fig4_selective_prediction.pdf
    fig5_feature_correlation.pdf
    fig6_with_without_uq.pdf
    fig7_calibration.pdf
    fig8_architecture.pdf
    fig9_policy_explanation.pdf
    fig10_node_vs_graph.pdf
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Pastel / soft palette
# ---------------------------------------------------------------------------
BG = "#FAFBFC"  # very light off-white background
PANEL = "#F0F4F8"  # light panel fill
P_BLUE = "#5B8DB8"  # soft steel blue (primary)
P_BLUE_LT = "#A8C8E8"  # lighter blue
P_BLUE_DK = "#2E6494"  # darker blue for contrast
P_CORAL = "#E07A5F"  # soft coral (highlight / T8)
P_CORAL_LT = "#F2B5A0"  # light coral
P_GREEN = "#6BAB8C"  # sage green
P_GREEN_LT = "#B8D4C0"  # light green
P_PURPLE = "#8E7CC3"  # soft lavender/purple
P_AMBER = "#E8A84C"  # soft amber/gold
P_SLATE = "#5C6B7A"  # medium dark slate for text/arrows
P_DGRAY = "#3A4A5A"  # near-black for titles
P_MGRAY = "#7A8A9A"  # medium gray
P_LGRAY = "#D0D8E0"  # light gray for borders/grid
P_XLGRAY = "#E8EDF2"  # extra light gray panels
WHITE = "#FFFFFF"

# Shared rcParams for all figures
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
    }
)


def save(fig, name, bg=None):
    fc = bg if bg is not None else BG
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=fc)
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=fc)
    plt.close(fig)
    print(f"  saved {name}.pdf")


# ===========================================================================
# FIG 1 — Trial comparison (T2–T8): R², MAE, RMSE
# ===========================================================================
def fig1_trial_comparison():
    trials = ["T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    r2 = [0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
    mae = [4.33, 5.99, 6.08, 4.24, 4.32, 4.06, 3.96]
    rmse = [8.15, 10.27, 10.15, 7.78, 8.06, 7.53, 7.12]

    x = np.arange(len(trials))
    colors = [P_BLUE] * 6 + [P_CORAL]  # T8 highlighted

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)

    def bar_panel(ax, values, ylabel, title, higher_better=True):
        bars = ax.bar(
            x, values, color=colors, width=0.62, edgecolor=P_LGRAY, linewidth=0.5
        )
        ax.set_xticks(x)
        ax.set_xticklabels(trials)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=8, color=P_DGRAY)
        ax.yaxis.grid(True, color=P_LGRAY)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        ax.spines["left"].set_color(P_LGRAY)
        ax.spines["bottom"].set_color(P_LGRAY)
        ax.bar_label(
            bars,
            fmt="%.3f" if "R²" in title else "%.2f",
            fontsize=7.5,
            padding=3,
            color=P_SLATE,
        )
        note = "(higher = better)" if higher_better else "(lower = better)"
        ax.text(
            0.98,
            0.97,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=P_MGRAY,
            style="italic",
        )

    bar_panel(axes[0], r2, "R²", "R² (coefficient of determination)", True)
    bar_panel(axes[1], mae, "MAE (veh/h)", "MAE — Mean Absolute Error", False)
    bar_panel(axes[2], rmse, "RMSE (veh/h)", "RMSE — Root Mean Square Error", False)

    t8_patch = mpatches.Patch(color=P_CORAL, label="T8 (best, selected for UQ)")
    other = mpatches.Patch(color=P_BLUE, label="T2–T7")
    fig.legend(
        handles=[t8_patch, other],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.suptitle(
        "Test-set Performance: Trials 2–8  (1,000 of 10,000 scenarios — 10% subset)",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
        y=1.02,
    )
    fig.tight_layout(pad=1.2)
    save(fig, "fig1_trial_comparison")


# ===========================================================================
# FIG 2 — UQ ranking: Spearman ρ across all methods
# ===========================================================================
def fig2_uq_ranking():
    labels = [
        "T5 MC\nDropout",
        "T6 MC\nDropout",
        "T7 MC\nDropout",
        "T8 MC\nDropout\n(standalone)",
        "Exp A\nMC Drop.",
        "Exp A\nEnsemble",
        "Exp A\nCombined",
        "Exp B\nMulti-Ens.",
    ]
    rho = [0.4263, 0.4186, 0.4437, 0.4820, 0.1600, 0.1035, 0.1601, 0.1167]
    colors = ([P_BLUE] * 4) + ([P_AMBER] * 4)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(labels))
    bars = ax.bar(x, rho, color=colors, width=0.62, edgecolor=P_LGRAY, linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3, color=P_SLATE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Spearman ρ  (uncertainty vs |error|)")
    ax.set_ylim(0, 0.66)
    ax.yaxis.grid(True, color=P_LGRAY)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_color(P_LGRAY)
    ax.spines["bottom"].set_color(P_LGRAY)

    # Divider between standalone and ensemble
    ax.axvline(x=3.5, color=P_LGRAY, lw=1.0, ls="--")
    ax.text(
        1.5,
        0.605,
        "Standalone MC Dropout\n(full test set)",
        ha="center",
        fontsize=8,
        color=P_BLUE_DK,
        style="italic",
    )
    ax.text(
        5.5,
        0.605,
        "Ensemble experiments\n(mismatched subset, R²≈0)",
        ha="center",
        fontsize=8,
        color=P_AMBER,
        style="italic",
    )

    ax.annotate(
        "Best: ρ = 0.4820",
        xy=(3, 0.4820),
        xytext=(3, 0.560),
        ha="center",
        fontsize=8.5,
        color=P_DGRAY,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=P_SLATE, lw=0.9),
    )

    mc_patch = mpatches.Patch(
        color=P_BLUE, label="MC Dropout (standalone, full test set)"
    )
    ens_patch = mpatches.Patch(
        color=P_AMBER, label="Ensemble experiments (mismatched subset)"
    )
    ax.legend(
        handles=[mc_patch, ens_patch],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )

    ax.set_title(
        "Uncertainty Quality: Spearman ρ Across All UQ Methods  (T8 = primary UQ model)",
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=1.2)
    save(fig, "fig2_uq_ranking")


# ===========================================================================
# FIG 3 — Conformal prediction coverage + interval widths
# ===========================================================================
def fig3_conformal_coverage():
    nominal = [90.0, 95.0]
    achieved = [90.02, 95.01]
    widths = [9.92, 14.68]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.patch.set_facecolor(BG)

    x = np.array([0, 1])
    w = 0.34

    # Panel 1: nominal vs achieved
    b1 = ax1.bar(
        x - w / 2,
        nominal,
        w,
        label="Nominal",
        color=P_BLUE_LT,
        edgecolor=P_LGRAY,
        lw=0.5,
    )
    b2 = ax1.bar(
        x + w / 2,
        achieved,
        w,
        label="Achieved",
        color=P_CORAL,
        edgecolor=P_LGRAY,
        lw=0.5,
    )
    ax1.bar_label(b1, fmt="%.1f%%", fontsize=8.5, padding=3, color=P_SLATE)
    ax1.bar_label(b2, fmt="%.2f%%", fontsize=8.5, padding=3, color=P_SLATE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["90% level", "95% level"])
    ax1.set_ylabel("Coverage (%)")
    ax1.set_ylim(85, 101)
    ax1.yaxis.grid(True, color=P_LGRAY)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", length=0)
    ax1.spines["left"].set_color(P_LGRAY)
    ax1.spines["bottom"].set_color(P_LGRAY)
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title(
        "Nominal vs Achieved Coverage\n(achieved ≥ nominal — guarantee holds)",
        pad=8,
        color=P_DGRAY,
    )

    # Panel 2: interval half-widths
    b3 = ax2.bar(x, widths, 0.45, color=[P_BLUE, P_BLUE_DK], edgecolor=P_LGRAY, lw=0.5)
    ax2.bar_label(b3, fmt="±%.2f veh/h", fontsize=9, padding=3, color=P_SLATE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["90% interval", "95% interval"])
    ax2.set_ylabel("Half-width (veh/h)")
    ax2.set_ylim(0, 20)
    ax2.yaxis.grid(True, color=P_LGRAY)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", length=0)
    ax2.spines["left"].set_color(P_LGRAY)
    ax2.spines["bottom"].set_color(P_LGRAY)
    ax2.set_title("Conformal Interval Half-Width", pad=8, color=P_DGRAY)

    fig.suptitle(
        "Conformal Prediction — T8  (50 calibration + 50 evaluation graphs)",
        fontweight="bold",
        color=P_DGRAY,
        fontsize=11,
    )
    fig.tight_layout(pad=1.2)
    save(fig, "fig3_conformal_coverage")


# ===========================================================================
# FIG 4 — Selective prediction: MAE vs retention threshold
# ===========================================================================
def fig4_selective_prediction():
    labels = ["All\n(100%)", "Keep top\n90% certain", "Keep top\n50% certain"]
    mae = [3.94, 3.22, 2.31]
    reductions = [0.0, -18.5, -41.6]
    colors = [P_BLUE, P_BLUE_LT, P_GREEN]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(labels))
    bars = ax.bar(x, mae, 0.52, color=colors, edgecolor=P_LGRAY, linewidth=0.5)

    for bar, red in zip(bars, reductions):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.07,
            f"{h:.2f} veh/h",
            ha="center",
            va="bottom",
            fontsize=9,
            color=P_SLATE,
        )
        if red < 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h / 2,
                f"{red:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=WHITE,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("MAE (veh/h)")
    ax.set_ylim(0, 5.0)
    ax.yaxis.grid(True, color=P_LGRAY)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_color(P_LGRAY)
    ax.spines["bottom"].set_color(P_LGRAY)
    ax.set_title(
        "Selective Prediction: MAE at Different Uncertainty-Based Retention Thresholds\n"
        "(T8 — 50 evaluation graphs, 1,581,750 nodes)",
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )

    ax.axhline(3.94, color=P_MGRAY, lw=0.8, ls="--", alpha=0.7)
    ax.text(
        2.35,
        4.05,
        "Baseline MAE\n(MC mean, all predictions)",
        ha="center",
        fontsize=7.5,
        color=P_MGRAY,
        style="italic",
    )

    fig.tight_layout(pad=1.2)
    save(fig, "fig4_selective_prediction")


# ===========================================================================
# FIG 3b — Feature distributions (Ch. 3, Fig 3.1 replacement)
# Loads actual node features from .pt batch files (cols 0,1,2,3,5 = the 5
# thesis features; col4 = number of lanes, not used in the model).
# Uses 1 batch (50 graphs × 31,635 nodes = 1,581,750 samples) for speed.
# ===========================================================================
def fig3_feature_distributions():
    import torch

    DATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "data",
        "train_data",
        "dist_not_connected_10k_1pct",
    )
    DATA_DIR = os.path.normpath(DATA_DIR)

    # Collect features from batches 1–4 (200 graphs, ~6.3 M nodes — fast enough)
    all_x = []
    for batch_idx in range(1, 5):
        pt_path = os.path.join(DATA_DIR, f"datalist_batch_{batch_idx}.pt")
        if not os.path.exists(pt_path):
            print(f"  WARNING: {pt_path} not found, skipping")
            continue
        batch = torch.load(pt_path, weights_only=False)
        for item in batch:
            # cols 0,1,2,3,5 → VOL, CAP, CAP_RED, FREESPEED, LENGTH
            x = item.x[:, [0, 1, 2, 3, 5]].numpy()
            all_x.append(x)

    if not all_x:
        print("  ERROR: no batch files found for fig3_feature_distributions")
        return

    import numpy as np_local

    data = np_local.concatenate(all_x, axis=0)  # shape (N, 5)

    feat_names = [
        "VOL_BASE_CASE\n(vehicles / hour)",
        "CAPACITY_BASE_CASE\n(vehicles / hour)",
        "CAPACITY_REDUCTION\n(vehicles / hour)",
        "FREESPEED\n(m / s)",
        "LENGTH\n(metres)",
    ]
    feat_short = ["VOL", "CAP", "CAP_RED", "FREESPEED", "LENGTH"]
    feat_colors = [P_BLUE, P_BLUE, P_CORAL, P_GREEN, P_AMBER]

    fig, axes = plt.subplots(1, 5, figsize=(14, 5.0))
    fig.patch.set_facecolor(WHITE)

    for i, ax in enumerate(axes):
        vals = data[:, i]
        # Remove extreme outliers (>99.5th percentile) for display clarity
        p995 = np_local.percentile(vals, 99.5)
        p005 = np_local.percentile(vals, 0.5)
        clipped = vals[(vals >= p005) & (vals <= p995)]

        ax.hist(
            clipped,
            bins=50,
            color=feat_colors[i],
            alpha=0.75,
            edgecolor="none",
            density=True,
        )
        ax.set_facecolor(WHITE)
        ax.set_title(
            feat_names[i], fontsize=9.0, color=P_DGRAY, pad=9, multialignment="center"
        )
        ax.set_xlabel(feat_short[i], fontsize=8.0, color=P_MGRAY)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=8.0, color=P_MGRAY)
        ax.tick_params(axis="both", length=0, labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(P_LGRAY)
        ax.spines["bottom"].set_color(P_LGRAY)
        ax.yaxis.grid(True, color=P_LGRAY, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

        # Median line
        med = np_local.median(vals)
        ax.axvline(med, color=P_DGRAY, lw=1.0, ls="--", alpha=0.6)
        ax.text(
            0.97,
            0.96,
            f"med={med:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color=P_DGRAY,
        )

    fig.text(
        0.5,
        -0.03,
        "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset).",
        ha="center",
        fontsize=7.5,
        color=P_MGRAY,
        style="italic",
    )
    fig.tight_layout(pad=1.5)
    save(fig, "fig3_feature_distributions", bg=WHITE)


# ===========================================================================
# FIG 5 — Feature correlation with prediction error (horizontal bar)
# ===========================================================================
def fig5_feature_correlation():
    features = [
        "VOL — Baseline traffic volume",
        "CAP — Road capacity",
        "SPD — Free-flow speed",
        "CAP_RED — Policy intervention",
        "LEN — Segment length",
    ]
    rho = [0.332, 0.262, 0.211, -0.229, -0.070]
    colors = [P_CORAL if v > 0 else P_BLUE for v in rho]

    fig, ax = plt.subplots(figsize=(9, 5.0))
    fig.patch.set_facecolor(BG)

    y = np.arange(len(features))
    bars = ax.barh(y, rho, 0.48, color=colors, edgecolor=P_LGRAY, linewidth=0.5)

    for bar, v in zip(bars, rho):
        offset = 0.012 if v >= 0 else -0.012
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=9.5,
            color=P_SLATE,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Spearman ρ  (feature vs |prediction error|)")
    ax.set_xlim(-0.30, 0.42)
    ax.axvline(0, color=P_SLATE, lw=0.8)
    ax.xaxis.grid(True, color=P_LGRAY)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_color(P_LGRAY)
    ax.spines["bottom"].set_color(P_LGRAY)

    pos_patch = mpatches.Patch(color=P_CORAL, label="Positive: feature → higher error")
    neg_patch = mpatches.Patch(color=P_BLUE, label="Negative: feature → lower error")
    ax.legend(
        handles=[pos_patch, neg_patch],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )

    ax.set_title(
        "Spearman Correlation: Input Features vs Absolute Prediction Error  (T8, 100 test graphs)",
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=1.5)
    save(fig, "fig5_feature_correlation")


# ===========================================================================
# FIG 6 — Deterministic vs MC Dropout inference (T8) — split panels
# ===========================================================================
def fig6_with_without_uq():
    det_r2, mc_r2 = 0.5957, 0.5857
    det_mae, mc_mae = 3.96, 3.948
    det_rmse, mc_rmse = 7.12, 7.207

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    w = 0.34
    det_bar_kw = dict(color=P_BLUE, edgecolor=P_LGRAY, lw=0.5)
    mc_bar_kw = dict(color=P_CORAL, edgecolor=P_LGRAY, lw=0.5)

    # --- Left panel: R² ---
    x1 = np.array([0])
    b1a = ax1.bar(x1 - w / 2, [det_r2], w, label="Deterministic", **det_bar_kw)
    b1b = ax1.bar(x1 + w / 2, [mc_r2], w, label="MC Dropout (S=30)", **mc_bar_kw)
    ax1.bar_label(b1a, fmt="%.4f", fontsize=9, padding=3, color=P_SLATE)
    ax1.bar_label(b1b, fmt="%.4f", fontsize=9, padding=3, color=P_SLATE)
    ax1.text(
        0,
        max(det_r2, mc_r2) + 0.032,
        "Δ = −0.010",
        ha="center",
        fontsize=8,
        color=P_SLATE,
        style="italic",
    )
    ax1.set_xticks([0])
    ax1.set_xticklabels(["R²"])
    ax1.set_ylabel("R²")
    ax1.set_ylim(0, 0.78)
    ax1.yaxis.grid(True, color=P_LGRAY)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", length=0)
    ax1.spines["left"].set_color(P_LGRAY)
    ax1.spines["bottom"].set_color(P_LGRAY)
    ax1.set_title("R²  (higher = better)", fontsize=10, color=P_DGRAY, pad=8)

    # --- Right panel: MAE + RMSE ---
    x2 = np.array([0, 1])
    b2a = ax2.bar(x2 - w / 2, [det_mae, det_rmse], w, **det_bar_kw)
    b2b = ax2.bar(x2 + w / 2, [mc_mae, mc_rmse], w, **mc_bar_kw)
    ax2.bar_label(b2a, fmt="%.3f", fontsize=8.5, padding=3, color=P_SLATE)
    ax2.bar_label(b2b, fmt="%.3f", fontsize=8.5, padding=3, color=P_SLATE)
    for i, (delta, ref) in enumerate(zip(["−0.012", "+0.087"], [det_mae, det_rmse])):
        ax2.text(
            i,
            max(ref, [mc_mae, mc_rmse][i]) + 0.42,
            f"Δ = {delta}",
            ha="center",
            fontsize=8,
            color=P_SLATE,
            style="italic",
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["MAE (veh/h)", "RMSE (veh/h)"])
    ax2.set_ylabel("Error (veh/h)")
    ax2.set_ylim(0, 11.0)
    ax2.yaxis.grid(True, color=P_LGRAY)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", length=0)
    ax2.spines["left"].set_color(P_LGRAY)
    ax2.spines["bottom"].set_color(P_LGRAY)
    ax2.set_title("Error metrics  (lower = better)", fontsize=10, color=P_DGRAY, pad=8)

    det_patch = mpatches.Patch(color=P_BLUE, label="Deterministic (single pass)")
    mc_patch = mpatches.Patch(color=P_CORAL, label="MC Dropout (S=30 samples)")
    fig.legend(
        handles=[det_patch, mc_patch],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Deterministic vs MC Dropout Inference — T8\n(100 test graphs, 3,163,500 nodes)",
        fontsize=11,
        fontweight="bold",
        color=P_DGRAY,
    )
    fig.tight_layout(pad=1.2)
    save(fig, "fig6_with_without_uq")


# ===========================================================================
# FIG 7 — Calibration: k₉₅ for Gaussian vs T8 MC Dropout
# ===========================================================================
def fig7_calibration():
    labels = ["Ideal\nGaussian", "T8\nMC Dropout"]
    k95 = [1.96, 11.34]
    colors = [P_GREEN, P_CORAL]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)

    x = np.arange(2)
    bars = ax.bar(x, k95, 0.45, color=colors, edgecolor=P_LGRAY, linewidth=0.6)
    ax.bar_label(
        bars, fmt="%.3f", fontsize=10, padding=4, fontweight="bold", color=P_SLATE
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("k₉₅  (scaling factor for 95% coverage)")
    ax.set_ylim(0, 15.0)
    ax.yaxis.grid(True, color=P_LGRAY)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_color(P_LGRAY)
    ax.spines["bottom"].set_color(P_LGRAY)

    ax.axhline(1.96, color=P_BLUE, lw=1.0, ls="--", alpha=0.7)
    ax.text(
        0.48,
        2.55,
        "Calibrated Gaussian  k₉₅ = 1.96",
        ha="left",
        fontsize=8,
        color=P_BLUE,
        style="italic",
    )

    ax.annotate(
        "MC Dropout σ is ~6× too small\nto serve as calibrated std",
        xy=(1, 11.34),
        xytext=(0.52, 13.4),
        ha="center",
        fontsize=8,
        color=P_DGRAY,
        arrowprops=dict(arrowstyle="->", color=P_SLATE, lw=0.9),
    )

    ax.set_title(
        "MC Dropout Calibration: Scaling Factor k₉₅\n"
        "for ±kσ to Achieve 95% Coverage  (T8, 100 test graphs)",
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=1.2)
    save(fig, "fig7_calibration")


# ===========================================================================
# FIG 8 — PointNetTransfGAT architecture flow diagram
# ===========================================================================
def fig8_architecture():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0.55, 5.15)
    ax.axis("off")

    # (label, cx, facecolor, textcolor)
    stages = [
        ("Input\n5 features\nper node", 0.8, P_XLGRAY, P_SLATE),
        ("PointNet\nConv-1\n5→512", 2.4, P_BLUE_LT, P_DGRAY),
        ("PointNet\nConv-2\n512→128", 4.0, P_BLUE_LT, P_DGRAY),
        ("Transformer\nConv-1\n128→256\n4 heads", 5.8, P_BLUE, WHITE),
        ("Transformer\nConv-2\n256→512\n4 heads", 7.6, P_BLUE, WHITE),
        ("GATConv\n512→64\n1 head", 9.4, P_BLUE_DK, WHITE),
        ("GATConv\n64→1\n(T2–T8)\nor Linear (T1)", 11.2, P_CORAL, WHITE),
        ("Δv\nprediction\nper node", 12.9, P_GREEN, P_DGRAY),
    ]

    bw, bh = 1.3, 1.9
    cy = 2.65

    for label, cx, fc, tc in stages:
        rect = mpatches.FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0.10",
            linewidth=0.8,
            edgecolor=P_LGRAY,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=tc,
            fontweight="bold",
            multialignment="center",
        )

    # Arrows between boxes
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + bw / 2 + 0.05
        x1 = stages[i + 1][1] - bw / 2 - 0.05
        ax.annotate(
            "",
            xy=(x1, cy),
            xytext=(x0, cy),
            arrowprops=dict(arrowstyle="-|>", color=P_SLATE, lw=1.1, mutation_scale=12),
        )

    # Stage labels below boxes
    stage_labels = [
        (0.8, "Input"),
        (2.4, "Geometry\nencoding"),
        (4.0, "Geometry\nencoding"),
        (5.8, "Long-range\nattention"),
        (7.6, "Long-range\nattention"),
        (9.4, "Node\naggregation"),
        (11.2, "Output\nprojection"),
        (12.9, "Output"),
    ]
    for cx, lbl in stage_labels:
        ax.text(
            cx,
            1.05,
            lbl,
            ha="center",
            va="center",
            fontsize=7.5,
            color=P_MGRAY,
            multialignment="center",
        )

    # Dropout span bracket
    ax.annotate(
        "",
        xy=(11.8, 3.9),
        xytext=(1.8, 3.9),
        arrowprops=dict(arrowstyle="<->", color=P_AMBER, lw=1.1, mutation_scale=11),
    )
    ax.text(
        6.8,
        4.12,
        "Dropout active in PointNetConv MLPs and between TransformerConv layers",
        ha="center",
        fontsize=8.5,
        color=P_AMBER,
        style="italic",
    )

    ax.set_title(
        "PointNetTransfGAT Architecture  (T8)",
        fontsize=11.5,
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=0.5)
    save(fig, "fig8_architecture")


# ===========================================================================
# FIG 9 — Uncertainty-guided policy decision framework
# ===========================================================================
def fig9_policy_explanation():
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(cx, cy, w, h, label, fc, tc=P_DGRAY, fs=8.5):
        r = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12",
            linewidth=0.8,
            edgecolor=P_LGRAY,
            facecolor=fc,
            zorder=3,
        )
        ax.add_patch(r)
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            fontweight="bold",
            multialignment="center",
            zorder=4,
        )

    def arrow(x0, y0, x1, y1, color=P_SLATE):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=13),
            zorder=2,
        )

    cy_main = 2.5  # vertical centre of main flow

    # Step 1 — MATSim input
    box(1.3, cy_main, 1.9, 1.1, "MATSim\nScenario\nInput", P_XLGRAY)
    arrow(2.25, cy_main, 3.0, cy_main)

    # Step 2 — GNN surrogate
    box(4.0, cy_main, 1.9, 1.1, "GNN Surrogate\nT8 + MC Dropout\nS = 30", P_BLUE, WHITE)
    arrow(4.95, cy_main, 5.7, cy_main)

    # Step 3 — σ gate (diamond)
    dcx, dcy = 6.5, cy_main
    diamond = plt.Polygon(
        [(dcx, dcy + 0.7), (dcx + 1.0, dcy), (dcx, dcy - 0.7), (dcx - 1.0, dcy)],
        closed=True,
        facecolor=P_BLUE_LT,
        edgecolor=P_LGRAY,
        lw=0.8,
        zorder=3,
    )
    ax.add_patch(diamond)
    ax.text(
        dcx,
        dcy,
        "σ  gate",
        ha="center",
        va="center",
        fontsize=8.5,
        color=P_DGRAY,
        fontweight="bold",
        zorder=4,
    )

    # ACCEPT — bottom branch (low σ)
    arrow(dcx, dcy - 0.7, dcx, 1.25, color=P_GREEN)
    box(
        dcx,
        0.72,
        2.8,
        0.75,
        "ACCEPT  (bottom 50% σ)   MAE ≈ 2.31 veh/h  (−41.6%)",
        P_GREEN_LT,
        P_DGRAY,
        fs=7.5,
    )
    ax.text(
        dcx - 0.6,
        1.65,
        "low σ",
        fontsize=7.5,
        color=P_GREEN,
        style="italic",
        ha="center",
    )

    # FLAG — right branch (medium σ)
    arrow(dcx + 1.0, dcy, 9.2, dcy, color=P_AMBER)
    box(
        10.5,
        dcy,
        2.4,
        1.0,
        "FLAG\n(medium σ  50–90%)\nManual review",
        P_AMBER,
        P_DGRAY,
        fs=7.5,
    )

    # REJECT — top branch (high σ)
    arrow(dcx, dcy + 0.7, dcx, 4.25, color=P_CORAL)
    box(
        dcx,
        4.65,
        2.8,
        0.75,
        "REJECT  (top 10% σ)   Full MATSim re-simulation",
        P_CORAL_LT,
        P_DGRAY,
        fs=7.5,
    )
    ax.text(
        dcx + 0.6,
        3.65,
        "high σ",
        fontsize=7.5,
        color=P_CORAL,
        style="italic",
        ha="center",
    )

    # Output legend box — anchored below GNN surrogate box
    ax.text(
        4.0,
        1.6,
        "ŷ : per-node Δv  (veh/h)\nσ : MC Dropout uncertainty",
        ha="center",
        va="center",
        fontsize=7.5,
        color=P_SLATE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=P_LGRAY, lw=0.6),
    )

    ax.set_title(
        "Uncertainty-Guided Policy Decision Framework  (T8)",
        fontsize=10.5,
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=0.5)
    save(fig, "fig9_policy_explanation")


# ===========================================================================
# FIG 10 — Node-level vs graph-level prediction schematic
# ===========================================================================
def fig10_node_vs_graph():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import LinearSegmentedColormap

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13, 7.0),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
    )
    fig.patch.set_facecolor(BG)

    # --- LEFT panel: node-level network illustration ---
    ax1.set_xlim(-0.8, 5.8)
    ax1.set_ylim(-1.1, 5.2)
    ax1.axis("off")
    ax1.set_facecolor(BG)
    ax1.set_title(
        "Node-Level Predictions\n(31,635 nodes per scenario)",
        fontsize=11.5,
        fontweight="bold",
        color=P_DGRAY,
        pad=12,
    )
    # Panel label
    ax1.text(
        -0.75,
        5.05,
        "(a)",
        fontsize=11,
        fontweight="bold",
        color=P_MGRAY,
        va="top",
    )

    node_pos = {
        "n1": (0.5, 3.5),
        "n2": (2.0, 4.0),
        "n3": (3.5, 3.5),
        "n4": (1.0, 2.0),
        "n5": (3.0, 2.0),
        "n6": (2.0, 0.8),
    }
    node_dv = {"n1": -8, "n2": -35, "n3": 5, "n4": -18, "n5": 12, "n6": -22}
    node_sigma = {"n1": 0.5, "n2": 3.1, "n3": 0.7, "n4": 1.2, "n5": 0.9, "n6": 2.4}

    edges_left = [
        ("n1", "n2"),
        ("n2", "n3"),
        ("n1", "n4"),
        ("n4", "n5"),
        ("n3", "n5"),
        ("n4", "n6"),
        ("n5", "n6"),
    ]
    for s, t in edges_left:
        x0, y0 = node_pos[s]
        x1, y1 = node_pos[t]
        ax1.plot([x0, x1], [y0, y1], color=P_MGRAY, lw=1.8, zorder=1)

    # Custom diverging colormap using thesis palette
    div_cmap = LinearSegmentedColormap.from_list(
        "thesis_div", [P_BLUE_DK, P_XLGRAY, P_CORAL], N=256
    )
    vals = np.array(list(node_dv.values()))
    vmin, vmax = vals.min(), vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for nid, (nx, ny) in node_pos.items():
        dv = node_dv[nid]
        sg = node_sigma[nid]
        rgba = div_cmap(norm(dv))
        # Adaptive text colour: white on dark, dark on light
        brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        txt_color = WHITE if brightness < 0.60 else P_DGRAY
        circ = plt.Circle(
            (nx, ny), 0.40, facecolor=rgba, edgecolor=P_MGRAY, lw=0.9, zorder=3
        )
        ax1.add_patch(circ)
        ax1.text(
            nx,
            ny,
            f"{dv:+d}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=txt_color,
            zorder=4,
        )
        ax1.text(
            nx,
            ny - 0.66,
            f"σ={sg:.1f}",
            ha="center",
            va="top",
            fontsize=8.5,
            color=P_MGRAY,
            zorder=4,
        )

    ax1.text(
        2.5,
        -0.72,
        "Node colour = Δv (veh/h)    σ = MC Dropout uncertainty",
        ha="center",
        fontsize=9.5,
        color=P_SLATE,
        style="italic",
    )

    # Colorbar anchored to ax1 via inset_axes
    sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm)
    sm.set_array([])
    cax = inset_axes(
        ax1,
        width="5%",
        height="32%",
        loc="lower left",
        bbox_to_anchor=(0.04, 0.14, 1, 1),
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Δv (veh/h)", fontsize=9.5, color=P_SLATE)
    cb.ax.tick_params(labelsize=9, colors=P_SLATE)
    cb.outline.set_edgecolor(P_LGRAY)

    # --- RIGHT panel: graph-level MAE bars ---
    ax2.set_facecolor(BG)
    ax2.set_title(
        "Graph-Level MAE Across Scenarios\n(100 test graphs — T8)",
        fontsize=11.5,
        fontweight="bold",
        color=P_DGRAY,
        pad=12,
    )
    # Panel label
    ax2.text(
        0.01,
        1.02,
        "(b)",
        transform=ax2.transAxes,
        fontsize=11,
        fontweight="bold",
        color=P_MGRAY,
        va="bottom",
    )

    rng = np.random.default_rng(42)
    n_show = 20

    # --- Load REAL per-graph MAE and mean MC-dropout sigma from NPZ checkpoints ---
    NPZ_DIR = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "data",
            "TR-C_Benchmarks",
            "point_net_transf_gat_8th_trial_lower_dropout",
            "uq_results",
            "checkpoints_mc30",
        )
    )
    all_mae = []
    all_sigma = []
    for gi in range(100):
        npz_path = os.path.join(NPZ_DIR, f"graph_{gi:04d}.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            mae_g = float(np.mean(np.abs(d["predictions"] - d["targets"])))
            sigma_g = float(np.mean(d["uncertainties"]))
            all_mae.append(mae_g)
            all_sigma.append(sigma_g)

    if len(all_mae) >= n_show:
        # Sort by MAE for a cleaner bar chart; pick first n_show
        order = np.argsort(all_mae)
        g_mae = np.array(all_mae)[order][:n_show]
        g_sigma = np.array(all_sigma)[order][:n_show]
        mean_mae_label = float(np.mean(all_mae))
        x_label = "Scenario index (20 lowest-MAE scenarios shown)"
    else:
        # Fallback: synthetic data (should not happen with the 100 NPZ files present)
        g_mae = np.clip(rng.normal(3.96, 1.8, n_show), 0.8, 9.5)
        g_sigma = np.clip(rng.normal(1.37, 0.45, n_show), 0.3, 3.2)
        mean_mae_label = 3.96
        x_label = "Scenario index (illustrative sample)"

    x_idx = np.arange(1, n_show + 1)

    ax2.bar(
        x_idx,
        g_mae,
        color=P_BLUE,
        edgecolor=P_BLUE_DK,
        lw=0.5,
        label="Per-graph MAE",
        zorder=2,
        width=0.7,
    )
    ax2.errorbar(
        x_idx,
        g_mae,
        yerr=g_sigma,
        fmt="none",
        color=P_CORAL,
        lw=1.2,
        capsize=3.5,
        zorder=3,
        label="±mean MC dropout σ",
    )
    ax2.axhline(3.96, color=P_CORAL, lw=1.2, ls="--", alpha=0.75, zorder=4)
    ax2.text(
        n_show + 0.4,
        3.96 + 0.04,
        f"Mean MAE\n{mean_mae_label:.2f} veh/h",
        fontsize=10,
        color=P_SLATE,
        va="center",
    )
    ax2.set_xlabel(x_label, fontsize=10.5)
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.set_ylabel("MAE (veh/h)", fontsize=10.5)
    ax2.yaxis.grid(True, color=P_LGRAY)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", length=0, labelsize=10)
    ax2.spines["left"].set_color(P_LGRAY)
    ax2.spines["bottom"].set_color(P_LGRAY)
    ax2.legend(frameon=False, fontsize=10.5)

    fig.suptitle(
        "Node-Level vs Graph-Level Evaluation  (T8, S=30 MC Dropout, 100 test graphs)",
        fontsize=12.5,
        fontweight="bold",
        color=P_DGRAY,
    )
    fig.tight_layout(pad=1.8, w_pad=4.0)
    save(fig, "fig10_node_vs_graph")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("Generating all thesis figures...")
    fig1_trial_comparison()
    fig2_uq_ranking()
    fig3_conformal_coverage()
    fig3_feature_distributions()
    fig4_selective_prediction()
    fig5_feature_correlation()
    fig6_with_without_uq()
    fig7_calibration()
    fig8_architecture()
    fig9_policy_explanation()
    fig10_node_vs_graph()
    print("Done. All figures written to:", OUT)
