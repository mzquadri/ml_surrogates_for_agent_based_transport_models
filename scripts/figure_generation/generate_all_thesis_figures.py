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
import sys
import json

# Add figures directory to path for thesis_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_style import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# Repo root: code/figure_generation/ → ../../ → repo root
REPO = os.path.abspath(os.path.join(OUT, "..", ".."))
DATA = os.path.join(REPO, "data", "TR-C_Benchmarks")
RESULTS = os.path.join(REPO, "results")


def _load_json(path):
    with open(path, "r") as _f:
        return json.load(_f)


def save(fig, name, bg=None):
    fc = bg if bg is not None else BG
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=fc)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=fc)
    plt.close(fig)
    print(f"  saved {name}.pdf + .png")


# ===========================================================================
# FIG 1 — Trial comparison (T2–T8): R², MAE, RMSE
# ===========================================================================
def fig1_trial_comparison():
    # Trial directory names in DATA folder (ordered T2→T8)
    trial_dirs = [
        "point_net_transf_gat_2nd_try",
        "point_net_transf_gat_3rd_trial_weighted_loss",
        "point_net_transf_gat_4th_trial_weighted_loss",
        "point_net_transf_gat_5th_try",
        "point_net_transf_gat_6th_trial_lower_lr",
        "point_net_transf_gat_7th_trial_80_10_10_split",
        "point_net_transf_gat_8th_trial_lower_dropout",
    ]
    trials = ["T2", "T3", "T4", "T5", "T6", "T7", "T8"]

    r2, mae, rmse = [], [], []
    for td in trial_dirs:
        if "4th_trial" in td:
            # T4 uses a flat-key JSON (no test_metrics nesting)
            d = _load_json(os.path.join(DATA, td, "test_results.json"))
            r2.append(d["r2_score"])
            mae.append(d["mae"])
            rmse.append(d["rmse"])
        else:
            d = _load_json(os.path.join(DATA, td, "test_evaluation_complete.json"))
            tm = d["test_metrics"]
            r2.append(tm["r2"])
            mae.append(tm["mae"])
            rmse.append(tm["rmse"])

    x = np.arange(len(trials))
    colors = [P_BLUE] * 6 + [P_CORAL]  # T8 highlighted

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.patch.set_facecolor(BG)

    def bar_panel(ax, values, ylabel, title, higher_better=True):
        bars = ax.bar(
            x, values, color=colors, width=0.62, edgecolor="white", linewidth=0.6
        )
        ax.set_xticks(x)
        ax.set_xticklabels(trials, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, pad=8, color=P_DGRAY, fontsize=11)
        ax.yaxis.grid(True, color=P_LGRAY, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        ax.spines["left"].set_color(P_LGRAY)
        ax.spines["bottom"].set_color(P_LGRAY)
        ax.bar_label(
            bars,
            fmt="%.4f" if "R²" in title else "%.2f",
            fontsize=9,
            padding=3,
            fontweight="bold",
            color=P_DGRAY,
        )
        # Add 18 % headroom so bar labels + annotation never touch the top frame
        ax.set_ylim(0, max(values) * 1.18)
        note = "(higher = better)" if higher_better else "(lower = better)"
        ax.text(
            0.98,
            0.97,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=P_SLATE,
            style="italic",
        )

    bar_panel(axes[0], r2, "R²", "R² (coefficient of determination)", True)
    bar_panel(axes[1], mae, "MAE (veh/h)", "MAE: Mean Absolute Error", False)
    bar_panel(axes[2], rmse, "RMSE (veh/h)", "RMSE: Root Mean Square Error", False)

    # Panel labels
    for i, lbl in enumerate(["(a)", "(b)", "(c)"]):
        panel_label(axes[i], lbl)

    t8_patch = mpatches.Patch(color=P_CORAL, label="T8 (best, selected for UQ)")
    other = mpatches.Patch(color=P_BLUE, label="T2–T7")
    fig.legend(
        handles=[t8_patch, other],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.93])
    fig.suptitle(
        "Test-set Performance: Trials 2\u20138  (1,000 of 10,000 scenarios, 10% subset)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        y=0.98,
    )
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
        "Exp A\nEns. Var.",
        "Exp A\nCombined",
        "Exp B\nMulti-Ens.",
    ]

    # Load Spearman ρ from verified JSON sources
    t5_rho = _load_json(os.path.join(
        DATA, "point_net_transf_gat_5th_try",
        "uq_results", "mc_dropout_full_metrics_model5_mc30_50graphs.json"
    ))["spearman"]

    t6_rho = _load_json(os.path.join(
        DATA, "point_net_transf_gat_6th_trial_lower_lr",
        "uq_results", "mc_dropout_full_metrics_model6_mc30_50graphs.json"
    ))["spearman"]

    # T7: canonical value from results/t7_error_detection.json (key: spearman_rho)
    t7_rho = _load_json(os.path.join(RESULTS, "t7_error_detection.json"))["spearman_rho"]

    t8_rho = _load_json(os.path.join(
        DATA, "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results", "mc_dropout_full_metrics_model8_mc30_100graphs.json"
    ))["spearman"]

    _exp_a = _load_json(os.path.join(
        DATA, "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results", "ensemble_experiments", "experiment_a_fixed_results.json"
    ))
    exp_a_mc_rho  = _exp_a["mc_dropout"]["spearman_rho"]
    exp_a_var_rho = _exp_a["ensemble_variance"]["spearman_rho"]
    exp_a_comb_rho = _exp_a["combined"]["spearman_rho"]

    exp_b_rho = _load_json(os.path.join(
        DATA, "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results", "ensemble_experiments", "experiment_b_fixed_results.json"
    ))["ensemble"]["spearman_rho"]

    rho = [t5_rho, t6_rho, t7_rho, t8_rho,
           exp_a_mc_rho, exp_a_var_rho, exp_a_comb_rho, exp_b_rho]

    # Best-in-panel indices (determined dynamically)
    best_standalone = rho[:4].index(max(rho[:4]))   # should be index 3 (T8)
    best_ensemble   = 4 + rho[4:].index(max(rho[4:]))  # should be index 6 (Exp A Combined)

    colors = ([P_BLUE] * 4) + ([P_AMBER] * 4)
    colors[best_standalone] = P_CORAL
    colors[best_ensemble]   = P_CORAL

    fig, ax = plt.subplots(figsize=(11, 6.0))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(labels))
    bars = ax.bar(x, rho, color=colors, width=0.62, edgecolor="white", linewidth=0.6)
    # Add hatching to ensemble bars to distinguish them visually
    for i in range(4, 8):
        bars[i].set_hatch("//")
        bars[i].set_edgecolor(P_SLATE)
    ax.bar_label(
        bars, fmt="%.4f", fontsize=9, padding=3, fontweight="bold", color=P_DGRAY
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman \u03c1  (uncertainty vs |error|)", fontsize=11)
    ax.set_ylim(0, 0.64)
    ax.yaxis.grid(True, color=P_LGRAY, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_color(P_LGRAY)
    ax.spines["bottom"].set_color(P_LGRAY)

    # Divider between standalone and ensemble
    ax.axvline(x=3.5, color=P_LGRAY, lw=1.0, ls="--")

    # Section labels as bracket-style text at the top
    # Standalone section
    ax.annotate(
        "",
        xy=(0, 0.600),
        xytext=(3, 0.600),
        arrowprops=dict(arrowstyle="-", color=P_BLUE_DK, lw=0.8),
    )
    ax.text(
        1.5,
        0.610,
        "Standalone MC Dropout (single model, S=30)",
        ha="center",
        fontsize=8.5,
        color=P_BLUE_DK,
        style="italic",
    )
    # Ensemble section
    ax.annotate(
        "",
        xy=(4, 0.600),
        xytext=(7, 0.600),
        arrowprops=dict(arrowstyle="-", color=P_AMBER, lw=0.8),
    )
    ax.text(
        5.5,
        0.610,
        "Ensemble experiments (5 seeded runs / 5 models)",
        ha="center",
        fontsize=8.5,
        color=P_AMBER,
        style="italic",
    )

    # "Best" label above T8 bar (left panel)
    ax.text(
        best_standalone,
        rho[best_standalone] + 0.040,
        "\u25b6 Best",
        ha="center",
        fontsize=8,
        color=P_CORAL,
        fontweight="bold",
    )

    # "Best" label above Exp A Combined bar (right panel)
    ax.text(
        best_ensemble,
        rho[best_ensemble] + 0.040,
        "\u25b6 Best",
        ha="center",
        fontsize=8,
        color=P_CORAL,
        fontweight="bold",
    )

    mc_patch = mpatches.Patch(
        color=P_BLUE, label="MC Dropout (standalone, single model)"
    )
    best_patch = mpatches.Patch(color=P_CORAL, label="Best in panel")
    ens_patch = mpatches.Patch(color=P_AMBER, label="Ensemble experiments")
    ax.legend(
        handles=[mc_patch, best_patch, ens_patch],
        frameon=True,
        framealpha=0.9,
        edgecolor=P_LGRAY,
        loc="lower left",
        fontsize=8.5,
    )

    ax.set_title(
        "Uncertainty Quality: Spearman \u03c1 Across All UQ Methods  (100 graphs, 3.16M nodes)",
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=1.5)
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
    # Target dashed lines
    ax1.axhline(90, color=P_SLATE, linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax1.axhline(95, color=P_SLATE, linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax1.text(1.42, 90.15, "90% target", fontsize=7.5, color=P_MGRAY, style="italic")
    ax1.text(1.42, 95.15, "95% target", fontsize=7.5, color=P_MGRAY, style="italic")
    # Delta annotations
    for i, (n, a) in enumerate(zip(nominal, achieved)):
        delta = a - n
        ax1.text(
            i,
            a + 1.0,
            f"\u0394 = +{delta:.2f}%",
            ha="center",
            fontsize=8,
            color=P_GREEN,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["90% level", "95% level"])
    ax1.set_ylabel("Achieved Coverage (%)")
    ax1.set_ylim(87, 101)
    ax1.yaxis.grid(True, color=P_LGRAY)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", length=0)
    ax1.spines["left"].set_color(P_LGRAY)
    ax1.spines["bottom"].set_color(P_LGRAY)
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title(
        "Nominal vs Achieved Coverage",
        pad=8,
        color=P_DGRAY,
    )

    # Panel 2: interval half-widths (distinct colors for 90% vs 95%)
    b3 = ax2.bar(x, widths, 0.45, color=[P_BLUE, P_AMBER], edgecolor=P_LGRAY, lw=0.5)
    ax2.bar_label(b3, fmt="\u00b1%.2f veh/h", fontsize=9, padding=3, color=P_SLATE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["90% interval", "95% interval"])
    ax2.set_ylabel("Interval Half-Width q (veh/h)")
    ax2.set_ylim(0, 18)
    ax2.yaxis.grid(True, color=P_LGRAY)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", length=0)
    ax2.spines["left"].set_color(P_LGRAY)
    ax2.spines["bottom"].set_color(P_LGRAY)
    ax2.set_title("Conformal Interval Half-Width", pad=8, color=P_DGRAY)

    panel_label(ax1, "(a)")
    panel_label(ax2, "(b)", y=1.08)
    fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.93])
    fig.suptitle(
        "Conformal Prediction: T8  (50 calibration + 50 evaluation graphs)",
        fontweight="bold",
        color=P_DGRAY,
        fontsize=13,
        y=0.98,
    )
    save(fig, "fig3_conformal_coverage")


# ===========================================================================
# FIG 4 — Selective prediction: MAE vs retention threshold
# ===========================================================================
def fig4_selective_prediction():
    labels = ["All\n(100%)", "Keep top\n90% certain", "Keep top\n50% certain"]
    mae = [3.95, 3.23, 2.32]
    reductions = [0.0, -18.3, -41.2]
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

    ax.axhline(3.95, color=P_MGRAY, lw=0.8, ls="--", alpha=0.7)
    ax.text(
        2.35,
        4.05,
        "Baseline MAE\n(MC mean, all predictions)",
        ha="center",
        fontsize=7.5,
        color=P_MGRAY,
        style="italic",
    )

    fig.tight_layout(pad=1.5)
    save(fig, "fig4_selective_prediction")


# ===========================================================================
# FIG 3b — Feature distributions (Ch. 3, Fig 3.1 replacement)
# Loads actual node features from .pt batch files (cols 0,1,2,3,5 = the 5
# thesis features; col4 = number of lanes, not used in the model).
# Uses 1 batch (50 graphs × 31,635 nodes = 1,581,750 samples) for speed.
# ===========================================================================
def fig3_feature_distributions():
    import torch
    from scipy import stats as sp_stats

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

    # Collect features from batches 1-4 (200 graphs, ~6.3 M nodes)
    all_x = []
    for batch_idx in range(1, 5):
        pt_path = os.path.join(DATA_DIR, f"datalist_batch_{batch_idx}.pt")
        if not os.path.exists(pt_path):
            print(f"  WARNING: {pt_path} not found, skipping")
            continue
        batch = torch.load(pt_path, weights_only=False)
        for item in batch:
            x = item.x[:, [0, 1, 2, 3, 5]].numpy()
            all_x.append(x)

    if not all_x:
        print("  ERROR: no batch files found for fig3_feature_distributions")
        return

    data = np.concatenate(all_x, axis=0)  # shape (N, 5)
    N = data.shape[0]
    print(f"  Loaded {N:,} nodes from {len(all_x)} graphs")

    # --- 3 top + 2 bottom centered layout (no summary panel) ---
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.5, 8.5))
    fig.patch.set_facecolor(WHITE)
    gs = GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.60)
    # Top row: 3 panels spanning 2 columns each
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    # Bottom row: 2 panels centered (offset by 1 column)
    ax_d = fig.add_subplot(gs[1, 1:3])
    ax_e = fig.add_subplot(gs[1, 3:5])
    axes_list = [ax_a, ax_b, ax_c, ax_d, ax_e]

    def _style_ax(ax):
        ax.set_facecolor(WHITE)
        ax.yaxis.grid(True, color=P_LGRAY, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(P_LGRAY)

    # ---- (a) VOL_BASE_CASE: histogram with log y-axis ----
    ax = ax_a
    panel_label(ax, "(a)")
    vol = data[:, 0]
    vol_pos = vol[vol > 0]  # exclude exact zeros for log display
    ax.hist(vol_pos, bins=80, color=P_BLUE, alpha=0.75, edgecolor="none")
    ax.set_yscale("log")
    ax.set_title("VOL_BASE_CASE (veh/h)", fontsize=11, color=P_DGRAY, pad=8)
    ax.set_xlabel("Volume (veh/h)", fontsize=10)
    ax.set_ylabel("Count (log scale)", fontsize=10)
    med_vol = float(np.median(vol))
    skew_vol = float(sp_stats.skew(vol))
    ax.axvline(med_vol, color=P_CORAL, lw=1.2, ls="--", alpha=0.8)
    ax.text(
        0.95,
        0.92,
        f"median = {med_vol:.1f}\nskew = {skew_vol:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=P_DGRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc=WHITE, ec=P_LGRAY, alpha=0.9),
    )
    _style_ax(ax)

    # ---- (b) CAPACITY_BASE_CASE: discrete bar chart ----
    ax = ax_b
    panel_label(ax, "(b)")
    cap = data[:, 1]
    unique_cap, counts_cap = np.unique(cap, return_counts=True)
    # Sort by frequency, show top 10 + "Other"
    order = np.argsort(-counts_cap)
    top_k = 10
    top_vals = unique_cap[order[:top_k]]
    top_cnts = counts_cap[order[:top_k]]
    other_cnt = counts_cap[order[top_k:]].sum() if len(order) > top_k else 0
    bar_labels = [f"{int(v)}" for v in top_vals]
    bar_counts = list(top_cnts)
    if other_cnt > 0:
        bar_labels.append("Other")
        bar_counts.append(other_cnt)
    bar_pcts = [100.0 * c / N for c in bar_counts]
    x_pos = np.arange(len(bar_labels))
    bars = ax.bar(
        x_pos, bar_pcts, color=P_BLUE, alpha=0.75, edgecolor=P_BLUE_DK, lw=0.4
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_title("CAPACITY_BASE_CASE (veh/h)", fontsize=11, color=P_DGRAY, pad=8)
    ax.set_xlabel("Capacity value", fontsize=10)
    ax.set_ylabel("Percentage of nodes (%)", fontsize=10)
    ax.text(
        0.95,
        0.92,
        f"{len(unique_cap)} unique values",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=P_DGRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc=WHITE, ec=P_LGRAY, alpha=0.9),
    )
    _style_ax(ax)

    # ---- (c) CAPACITY_REDUCTION: non-zero histogram + zero annotation ----
    ax = ax_c
    panel_label(ax, "(c)")
    capred = data[:, 2]
    n_zero = int(np.sum(capred == 0))
    pct_zero = 100.0 * n_zero / N
    nonzero = capred[capred != 0]
    ax.hist(nonzero, bins=50, color=P_CORAL, alpha=0.75, edgecolor="none")
    ax.set_yscale("log")
    ax.set_title("CAPACITY_REDUCTION (veh/h)", fontsize=11, color=P_DGRAY, pad=8)
    ax.set_xlabel("Reduction (veh/h), non-zero only", fontsize=10)
    ax.set_ylabel("Count (log scale)", fontsize=10)
    # Prominent zero annotation — positioned top-left to avoid bar overlap
    ax.text(
        0.05,
        0.95,
        f"{pct_zero:.1f}% of nodes = 0\n(no policy applied)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=P_CORAL,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0EC", ec=P_CORAL, alpha=0.95),
    )
    if len(nonzero) > 0:
        med_nz = float(np.median(nonzero))
        ax.axvline(med_nz, color=P_DGRAY, lw=1.0, ls="--", alpha=0.7)
        ax.text(
            0.05,
            0.68,
            f"non-zero median = {med_nz:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=P_DGRAY,
            bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=P_LGRAY, alpha=0.8),
        )
    _style_ax(ax)

    # ---- (d) FREESPEED: discrete bar plot in km/h ----
    ax = ax_d
    panel_label(ax, "(d)")
    fs = data[:, 3]
    fs_kmh = fs * 3.6  # m/s -> km/h
    unique_fs, counts_fs = np.unique(fs_kmh, return_counts=True)
    # Round to nearest integer for cleaner labels
    unique_fs_r = np.round(unique_fs).astype(int)
    pcts_fs = 100.0 * counts_fs / N
    bar_colors = [P_GREEN if p < 30 else P_GREEN_LT for p in pcts_fs]
    # Highlight the dominant 30 km/h bar
    for idx_f, v in enumerate(unique_fs_r):
        if v == 30:
            bar_colors[idx_f] = P_GREEN
    ax.bar(
        range(len(unique_fs_r)),
        pcts_fs,
        color=bar_colors,
        alpha=0.8,
        edgecolor=P_DGRAY,
        lw=0.3,
    )
    ax.set_xticks(range(len(unique_fs_r)))
    ax.set_xticklabels(
        [str(v) for v in unique_fs_r], fontsize=8, rotation=45, ha="right"
    )
    ax.set_title("FREESPEED (km/h)", fontsize=11, color=P_DGRAY, pad=8)
    ax.set_xlabel("Speed class (km/h)", fontsize=10)
    ax.set_ylabel("Percentage of nodes (%)", fontsize=10)
    # Annotate dominant class
    idx_30 = np.where(unique_fs_r == 30)[0]
    if len(idx_30) > 0:
        pct_30 = pcts_fs[idx_30[0]]
        ax.annotate(
            f"{pct_30:.1f}%",
            xy=(idx_30[0], pct_30),
            xytext=(idx_30[0] + 2.5, pct_30 * 0.85),
            fontsize=9,
            fontweight="bold",
            color=P_DGRAY,
            arrowprops=dict(arrowstyle="->", color=P_DGRAY, lw=1.0),
        )
    ax.text(
        0.95,
        0.92,
        f"{len(unique_fs_r)} discrete classes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=P_DGRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc=WHITE, ec=P_LGRAY, alpha=0.9),
    )
    _style_ax(ax)

    # ---- (e) LENGTH: histogram with log y-axis ----
    ax = ax_e
    panel_label(ax, "(e)")
    length = data[:, 4]
    ax.hist(length[length > 0], bins=80, color=P_AMBER, alpha=0.75, edgecolor="none")
    ax.set_yscale("log")
    ax.set_title("LENGTH (m)", fontsize=11, color=P_DGRAY, pad=8)
    ax.set_xlabel("Link length (m)", fontsize=10)
    ax.set_ylabel("Count (log scale)", fontsize=10)
    med_len = float(np.median(length))
    skew_len = float(sp_stats.skew(length))
    ax.axvline(med_len, color=P_CORAL, lw=1.2, ls="--", alpha=0.8)
    ax.text(
        0.95,
        0.92,
        f"median = {med_len:.1f} m\nskew = {skew_len:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=P_DGRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc=WHITE, ec=P_LGRAY, alpha=0.9),
    )
    _style_ax(ax)

    fig.suptitle(
        "Input Feature Distributions (200 graphs, 6.3M nodes)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        y=0.98,
    )
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
    fig.tight_layout(pad=2.0)
    save(fig, "fig5_feature_correlation")


# ===========================================================================
# FIG 6 — Deterministic vs MC Dropout inference (T8) — split panels
# ===========================================================================
def fig6_with_without_uq():
    det_r2, mc_r2 = 0.5957, 0.5857
    det_mae, mc_mae = 3.96, 3.948
    det_rmse, mc_rmse = 7.12, 7.207

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)

    w = 0.34
    det_bar_kw = dict(color=P_BLUE, edgecolor="white", lw=0.6)
    mc_bar_kw = dict(color=P_CORAL, edgecolor="white", lw=0.6)

    # --- Left panel: R² ---
    x1 = np.array([0])
    b1a = ax1.bar(x1 - w / 2, [det_r2], w, label="Deterministic", **det_bar_kw)
    b1b = ax1.bar(x1 + w / 2, [mc_r2], w, label="MC Dropout (S=30)", **mc_bar_kw)
    ax1.bar_label(
        b1a, fmt="%.4f", fontsize=9, padding=3, fontweight="bold", color=P_DGRAY
    )
    ax1.bar_label(
        b1b, fmt="%.4f", fontsize=9, padding=3, fontweight="bold", color=P_DGRAY
    )
    ax1.text(
        0,
        max(det_r2, mc_r2) + 0.045,
        "\u0394 = \u22120.010",
        ha="center",
        fontsize=9,
        color=P_SLATE,
        fontweight="bold",
        style="italic",
    )
    ax1.set_xticks([0])
    ax1.set_xticklabels(["R\u00b2"], fontsize=10)
    ax1.set_ylabel("R\u00b2", fontsize=11)
    ax1.set_ylim(0, 0.78)
    ax1.yaxis.grid(True, color=P_LGRAY, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", length=0)
    ax1.spines["left"].set_color(P_LGRAY)
    ax1.spines["bottom"].set_color(P_LGRAY)
    ax1.set_title("R\u00b2  (higher = better)", fontsize=11, color=P_DGRAY, pad=8)

    # --- Right panel: MAE + RMSE ---
    x2 = np.array([0, 1])
    b2a = ax2.bar(x2 - w / 2, [det_mae, det_rmse], w, **det_bar_kw)
    b2b = ax2.bar(x2 + w / 2, [mc_mae, mc_rmse], w, **mc_bar_kw)
    ax2.bar_label(
        b2a, fmt="%.3f", fontsize=9, padding=3, fontweight="bold", color=P_DGRAY
    )
    ax2.bar_label(
        b2b, fmt="%.3f", fontsize=9, padding=3, fontweight="bold", color=P_DGRAY
    )
    for i, (delta, ref) in enumerate(
        zip(["\u22120.012", "+0.087"], [det_mae, det_rmse])
    ):
        ax2.text(
            i,
            max(ref, [mc_mae, mc_rmse][i]) + 0.65,
            f"\u0394 = {delta}",
            ha="center",
            fontsize=9,
            color=P_SLATE,
            fontweight="bold",
            style="italic",
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["MAE (veh/h)", "RMSE (veh/h)"], fontsize=10)
    ax2.set_ylabel("Error (veh/h)", fontsize=11)
    ax2.set_ylim(0, 11.0)
    ax2.yaxis.grid(True, color=P_LGRAY, linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", length=0)
    ax2.spines["left"].set_color(P_LGRAY)
    ax2.spines["bottom"].set_color(P_LGRAY)
    ax2.set_title("Error metrics  (lower = better)", fontsize=11, color=P_DGRAY, pad=8)

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
        "Deterministic vs MC Dropout Inference: T8\n(100 test graphs, 3,163,500 nodes)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        y=0.97,
    )
    panel_label(ax1, "(a)")
    panel_label(ax2, "(b)")
    fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 0.90])
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
    fig.tight_layout(pad=1.5)
    save(fig, "fig7_calibration")


# ===========================================================================
# FIG 8 — PointNetTransfGAT architecture flow diagram
# ===========================================================================
def fig8_architecture():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0.3, 5.4)
    ax.axis("off")

    # Light fills + dark text for every box (no white-on-dark)
    C_INPUT = (P_XLGRAY, P_SLATE)
    C_PNET = ("#D2E4F0", P_DGRAY)  # very light blue
    C_TCONV = ("#B8D4E8", P_DGRAY)  # light-medium blue
    C_GAT = ("#A5C6DB", P_DGRAY)  # medium blue
    C_OUT_P = ("#F0D5C8", P_DGRAY)  # light coral
    C_PRED = ("#C4DECE", P_DGRAY)  # light green

    stages = [
        ("Input\n5 features\nper node", 0.8, *C_INPUT),
        ("PointNet\nConv-1\n5 \u2192 512", 2.35, *C_PNET),
        ("PointNet\nConv-2\n512 \u2192 128", 3.9, *C_PNET),
        ("Transformer\nConv-1\n128 \u2192 256\n4 heads", 5.6, *C_TCONV),
        ("Transformer\nConv-2\n256 \u2192 512\n4 heads", 7.3, *C_TCONV),
        ("GATConv\n512 \u2192 64\n1 head", 9.0, *C_GAT),
        ("GATConv\n64 \u2192 1\n(T2\u2013T8)\nor Linear (T1)", 10.65, *C_OUT_P),
        ("\u0394v prediction\nper node", 12.2, *C_PRED),
    ]

    bw, bh = 1.3, 1.8
    cy = 2.75

    for label, cx, fc, tc in stages:
        rect = mpatches.FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0.10",
            linewidth=0.9,
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
            fontsize=9.5,
            color=tc,
            fontweight="bold",
            multialignment="center",
        )

    # Arrows between boxes
    for i in range(len(stages) - 1):
        x0 = stages[i][1] + bw / 2 + 0.04
        x1 = stages[i + 1][1] - bw / 2 - 0.04
        ax.annotate(
            "",
            xy=(x1, cy),
            xytext=(x0, cy),
            arrowprops=dict(arrowstyle="-|>", color=P_SLATE, lw=1.2, mutation_scale=13),
        )

    # Stage labels below boxes
    stage_labels = [
        (0.8, "Input"),
        (2.35, "Geometry\nencoding"),
        (3.9, "Geometry\nencoding"),
        (5.6, "Long-range\nattention"),
        (7.3, "Long-range\nattention"),
        (9.0, "Node\naggregation"),
        (10.65, "Output\nprojection"),
        (12.2, "Output"),
    ]
    for cx, lbl in stage_labels:
        ax.text(
            cx,
            cy - bh / 2 - 0.25,
            lbl,
            ha="center",
            va="top",
            fontsize=8,
            color=P_MGRAY,
            multialignment="center",
        )

    # Dropout span bracket
    drop_y = cy + bh / 2 + 0.40
    ax.annotate(
        "",
        xy=(10.65 + bw / 2, drop_y),
        xytext=(2.35 - bw / 2, drop_y),
        arrowprops=dict(arrowstyle="<->", color=P_AMBER, lw=1.1, mutation_scale=11),
    )
    ax.text(
        6.5,
        drop_y + 0.22,
        "Dropout active in PointNetConv MLPs and between TransformerConv layers",
        ha="center",
        fontsize=8.5,
        color=P_AMBER,
        style="italic",
    )

    ax.set_title(
        "PointNetTransfGAT Architecture  (T8)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        pad=12,
    )
    fig.tight_layout(pad=1.2)
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
        "ACCEPT  (bottom 50% σ)   MAE ≈ 2.32 veh/h  (−41.2%)",
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
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        pad=10,
    )
    fig.tight_layout(pad=1.5)
    save(fig, "fig9_policy_explanation")


# ===========================================================================
# FIG 10 — Node-level vs graph-level prediction schematic
# ===========================================================================
def fig10_node_vs_graph():
    from scipy import stats as sp_stats

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))
    fig.patch.set_facecolor(WHITE)

    # --- Load REAL per-graph MAE and mean MC-dropout sigma ---
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
    graph_ids = []
    for gi in range(100):
        npz_path = os.path.join(NPZ_DIR, f"graph_{gi:04d}.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            mae_g = float(np.mean(np.abs(d["predictions"] - d["targets"])))
            sigma_g = float(np.mean(d["uncertainties"]))
            all_mae.append(mae_g)
            all_sigma.append(sigma_g)
            graph_ids.append(gi)

    all_mae = np.array(all_mae)
    all_sigma = np.array(all_sigma)
    graph_ids = np.array(graph_ids)
    n_graphs = len(all_mae)
    mean_mae = float(np.mean(all_mae))
    print(f"  Loaded {n_graphs} graphs, mean MAE = {mean_mae:.2f}")

    # Compute Spearman correlation
    rho, p_val = sp_stats.spearmanr(all_sigma, all_mae)
    print(f"  Graph-level Spearman rho = {rho:.4f}, p = {p_val:.2e}")

    # --- Panel (a): Color-coded scatter + regression + outlier labels ---
    panel_label(ax1, "(a)")
    ax1.set_facecolor(WHITE)

    sc = ax1.scatter(
        all_sigma,
        all_mae,
        c=all_mae,
        cmap="RdYlGn_r",
        s=55,
        alpha=0.85,
        edgecolors=P_DGRAY,
        linewidths=0.4,
        zorder=3,
    )

    # Regression line
    slope, intercept, r_val, _, _ = sp_stats.linregress(all_sigma, all_mae)
    x_fit = np.linspace(all_sigma.min(), all_sigma.max(), 100)
    y_fit = slope * x_fit + intercept
    ax1.plot(
        x_fit,
        y_fit,
        color=P_CORAL,
        lw=1.8,
        ls="--",
        alpha=0.8,
        label=f"Linear fit (R = {r_val:.2f})",
        zorder=2,
    )

    # Spearman annotation box
    ax1.text(
        0.05,
        0.95,
        f"Spearman rho = {rho:.2f}\np < {max(p_val, 1e-20):.0e}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=P_DGRAY,
        bbox=dict(boxstyle="round,pad=0.4", fc=WHITE, ec=P_BLUE, alpha=0.95),
    )

    # Label outliers: highest MAE and lowest MAE
    idx_max = np.argmax(all_mae)
    idx_min = np.argmin(all_mae)
    for idx, label_text in [
        (idx_max, f"Graph {graph_ids[idx_max]}\n(highest MAE)"),
        (idx_min, f"Graph {graph_ids[idx_min]}\n(lowest MAE)"),
    ]:
        offset_x = 15 if idx == idx_max else -15
        offset_y = -20 if idx == idx_max else 15
        ha = "left" if idx == idx_max else "right"
        ax1.annotate(
            label_text,
            xy=(all_sigma[idx], all_mae[idx]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8.5,
            color=P_DGRAY,
            ha=ha,
            arrowprops=dict(arrowstyle="->", color=P_MGRAY, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=P_LGRAY, alpha=0.9),
        )

    # Colorbar
    cb = fig.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
    cb.set_label("MAE (veh/h)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_edgecolor(P_LGRAY)

    ax1.set_xlabel("Mean MC Dropout uncertainty (veh/h)", fontsize=11)
    ax1.set_ylabel("MAE (veh/h)", fontsize=11)
    ax1.set_title(
        "Uncertainty vs Error: Per-Graph Correlation", fontsize=11, color=P_DGRAY, pad=8
    )
    ax1.legend(
        loc="lower right", frameon=True, fontsize=9, framealpha=0.9, edgecolor=P_LGRAY
    )
    ax1.yaxis.grid(True, color=P_LGRAY, linewidth=0.4, alpha=0.5)
    ax1.xaxis.grid(True, color=P_LGRAY, linewidth=0.4, alpha=0.3)
    ax1.set_axisbelow(True)
    for sp in ax1.spines.values():
        sp.set_color(P_LGRAY)

    # --- Panel (b): All 100 graphs sorted bar chart ---
    panel_label(ax2, "(b)")
    ax2.set_facecolor(WHITE)

    order = np.argsort(all_mae)
    sorted_mae = all_mae[order]
    sorted_sigma = all_sigma[order]
    x_idx = np.arange(1, n_graphs + 1)

    ax2.bar(
        x_idx,
        sorted_mae,
        color=P_BLUE,
        alpha=0.75,
        edgecolor="none",
        label="Per-graph MAE",
        zorder=2,
        width=0.85,
    )
    ax2.errorbar(
        x_idx,
        sorted_mae,
        yerr=sorted_sigma,
        fmt="none",
        color=P_CORAL,
        lw=0.8,
        capsize=1.5,
        zorder=3,
        label="$\\pm$ mean MC dropout $\\sigma$",
    )
    # Mean MAE line
    ax2.axhline(mean_mae, color=P_CORAL, lw=1.5, ls="--", alpha=0.8, zorder=4)
    ax2.text(
        n_graphs + 1,
        mean_mae,
        f"Mean MAE\n{mean_mae:.2f} veh/h",
        fontsize=9.5,
        color=P_SLATE,
        va="center",
        ha="left",
    )

    # Percentile annotations
    p90_mae = float(np.percentile(all_mae, 90))
    ax2.axhline(p90_mae, color=P_AMBER, lw=1.0, ls=":", alpha=0.7, zorder=4)
    ax2.text(
        n_graphs + 1,
        p90_mae,
        f"90th pctl\n{p90_mae:.2f}",
        fontsize=8.5,
        color=P_AMBER,
        va="center",
        ha="left",
    )

    ax2.set_xlabel("Scenario rank (sorted by MAE)", fontsize=11)
    ax2.set_ylabel("MAE (veh/h)", fontsize=11)
    ax2.set_title(
        "Graph-Level MAE: All 100 Test Scenarios", fontsize=11, color=P_DGRAY, pad=8
    )
    ax2.set_xlim(0, n_graphs + 8)
    tick_positions = [1, 25, 50, 75, 100]
    ax2.set_xticks(tick_positions)
    ax2.legend(
        loc="upper left", frameon=True, fontsize=9, framealpha=0.9, edgecolor=P_LGRAY
    )
    ax2.yaxis.grid(True, color=P_LGRAY, linewidth=0.4, alpha=0.5)
    ax2.set_axisbelow(True)
    for sp in ax2.spines.values():
        sp.set_color(P_LGRAY)

    fig.suptitle(
        "Node-Level vs Graph-Level Evaluation (T8, S=30 MC Dropout, 100 test graphs)",
        fontsize=13,
        fontweight="bold",
        color=P_DGRAY,
        y=0.97,
    )
    fig.tight_layout(pad=1.8, w_pad=3.0, rect=[0, 0.0, 1, 0.93])
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
