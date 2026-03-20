"""
generate_phase3_figures.py
==========================
Phase 3 UQ analyses: new figures and JSON results for thesis integration.

Analysis 3.1: Reliability diagram (calibration curve for MC Dropout)
Analysis 3.2: Stratified UQ by feature bins
Analysis 3.3: Conformal conditional coverage
Analysis 3.4: Per-graph uncertainty variation
Analysis 3.5: Temperature scaling verification
Analysis 3.6: T7 error detection AUROC

Run from the figures/ directory:
    C:\\Users\\zamin\\miniconda3\\python.exe generate_phase3_figures.py

All outputs saved to the figures/ directory as PDF.
"""

import os
import sys
import json
import warnings

# Fix Windows console encoding for Unicode characters
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
T8_DIR = os.path.join(
    REPO_ROOT, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
ABLATION_CSV = os.path.join(T8_DIR, "trial8_uq_ablation_results.csv")
TEST_DL_PT = os.path.join(T8_DIR, "data_created_during_training", "test_dl.pt")
MC_NPZ = os.path.join(T8_DIR, "uq_results", "mc_dropout_full_100graphs_mc30.npz")
PER_GRAPH_DIR = os.path.join(T8_DIR, "uq_results", "checkpoints_mc30")

T7_DIR = os.path.join(
    REPO_ROOT,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_7th_trial_80_10_10_split",
)
T7_MC_NPZ = os.path.join(T7_DIR, "uq_results", "mc_dropout_full_100graphs_mc30.npz")
T7_PER_GRAPH_DIR = os.path.join(T7_DIR, "uq_results", "checkpoints_mc30")

RESULTS_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "..", "docs", "verified", "phase3_results"
)

# ---------------------------------------------------------------------------
# Palette (identical to generate_all_thesis_figures.py)
# ---------------------------------------------------------------------------
BG = "#FAFBFC"
PANEL = "#F0F4F8"
P_BLUE = "#5B8DB8"
P_BLUE_LT = "#A8C8E8"
P_BLUE_DK = "#2E6494"
P_CORAL = "#E07A5F"
P_CORAL_LT = "#F2B5A0"
P_GREEN = "#6BAB8C"
P_GREEN_LT = "#B8D4C0"
P_PURPLE = "#8E7CC3"
P_AMBER = "#E8A84C"
P_SLATE = "#5C6B7A"
P_DGRAY = "#3A4A5A"
P_MGRAY = "#7A8A9A"
P_LGRAY = "#D0D8E0"
P_XLGRAY = "#E8EDF2"
WHITE = "#FFFFFF"

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

FOOTNOTE = "Results based on 1,000 of 10,000 available MATSim scenarios (10% subset)."


def save(fig, name, bg=None):
    fc = bg if bg is not None else BG
    pdf = os.path.join(SCRIPT_DIR, name + ".pdf")
    png = os.path.join(SCRIPT_DIR, name + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=fc)
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=fc)
    plt.close(fig)
    print(f"  saved {name}.pdf + .png")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# ANALYSIS 3.1 — Reliability Diagram (Calibration Plot)
# ===========================================================================
def analysis_31_reliability_diagram():
    """
    Reliability diagram: expected vs observed coverage for MC Dropout sigma.

    For each nominal level p, compute:
        observed_fraction = fraction of nodes where |target - pred_mc_mean| <= z_p * pred_mc_std

    where z_p = Phi^{-1}((1+p)/2) is the standard Gaussian quantile.

    Reference: Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning
    Using Calibrated Regression", ICML.
    """
    print("\n=== Analysis 3.1: Reliability Diagram ===")

    # Load data — use chunked reading for the 200MB CSV
    import pandas as pd

    print("  Loading ablation CSV (3.16M rows)...")
    df = pd.read_csv(ABLATION_CSV, usecols=["target", "pred_mc_mean", "pred_mc_std"])

    targets = df["target"].values
    means = df["pred_mc_mean"].values
    sigmas = df["pred_mc_std"].values
    N = len(targets)
    print(f"  Loaded {N:,} nodes")

    # Absolute residuals
    abs_residuals = np.abs(targets - means)

    # Nominal coverage levels
    nominal_levels = np.array(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    )

    # For each nominal level p, the Gaussian z-score for a two-sided interval with coverage p
    # is z_p = Phi^{-1}((1+p)/2)
    z_scores = stats.norm.ppf((1 + nominal_levels) / 2)

    # Compute observed coverage at each level
    observed_coverage = np.zeros_like(nominal_levels)
    for i, (p, z) in enumerate(zip(nominal_levels, z_scores)):
        interval_half_width = z * sigmas
        covered = abs_residuals <= interval_half_width
        observed_coverage[i] = np.mean(covered)

    # Also compute the ECE (Expected Calibration Error) — mean |observed - nominal|
    ece = np.mean(np.abs(observed_coverage - nominal_levels))

    # Save results JSON
    ensure_results_dir()
    results = {
        "analysis": "reliability_diagram",
        "description": "Expected vs observed coverage for MC Dropout Gaussian intervals",
        "n_nodes": int(N),
        "n_graphs": 100,
        "mc_samples": 30,
        "nominal_levels": nominal_levels.tolist(),
        "z_scores": z_scores.tolist(),
        "observed_coverage": observed_coverage.tolist(),
        "expected_calibration_error_ECE": float(ece),
        "reference": "Kuleshov et al. (2018), ICML",
        "note": "Severe under-coverage at all levels confirms k95=11.34 finding",
    }
    json_path = os.path.join(RESULTS_DIR, "reliability_diagram_t8.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {json_path}")

    # Print results table
    print(f"\n  {'Nominal':>8s}  {'z-score':>8s}  {'Observed':>10s}  {'Gap':>8s}")
    print(f"  {'--------':>8s}  {'-------':>8s}  {'--------':>10s}  {'---':>8s}")
    for p, z, obs in zip(nominal_levels, z_scores, observed_coverage):
        gap = obs - p
        print(f"  {p:8.1%}  {z:8.3f}  {obs:10.1%}  {gap:+8.1%}")
    print(f"\n  ECE = {ece:.4f}")

    # --- Figure: Reliability Diagram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(BG)

    # Left panel: reliability diagram
    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        color=P_MGRAY,
        linewidth=1.2,
        label="Perfect calibration",
        zorder=1,
    )
    ax1.plot(
        nominal_levels,
        observed_coverage,
        "o-",
        color=P_CORAL,
        linewidth=2,
        markersize=7,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        label="T8 MC Dropout",
        zorder=3,
    )

    # Shade the gap
    ax1.fill_between(
        nominal_levels,
        nominal_levels,
        observed_coverage,
        alpha=0.15,
        color=P_CORAL,
        zorder=2,
    )

    ax1.set_xlabel("Nominal coverage level", color=P_DGRAY)
    ax1.set_ylabel("Observed coverage", color=P_DGRAY)
    ax1.set_title("Reliability Diagram", color=P_DGRAY, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors=P_SLATE)

    # Add ECE annotation
    ax1.text(
        0.60,
        0.15,
        f"ECE = {ece:.3f}",
        fontsize=11,
        fontweight="bold",
        color=P_CORAL,
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=P_LGRAY, alpha=0.9
        ),
    )

    # Right panel: gap bar chart
    gaps = observed_coverage - nominal_levels
    level_labels = [f"{int(p * 100)}%" for p in nominal_levels]
    x = np.arange(len(nominal_levels))
    colors = [P_CORAL if g < 0 else P_GREEN for g in gaps]
    bars = ax1b = ax2
    bars.bar(x, gaps * 100, color=colors, width=0.65, edgecolor=P_LGRAY, linewidth=0.5)
    bars.axhline(0, color=P_MGRAY, linewidth=0.8)
    bars.set_xticks(x)
    bars.set_xticklabels(level_labels, rotation=45)
    bars.set_xlabel("Nominal coverage level", color=P_DGRAY)
    bars.set_ylabel("Coverage gap (percentage points)", color=P_DGRAY)
    bars.set_title("Calibration Gap", color=P_DGRAY, fontweight="bold")
    bars.grid(True, axis="y", alpha=0.3)
    bars.tick_params(colors=P_SLATE)

    # Add value labels on bars
    for i, (g, xi) in enumerate(zip(gaps, x)):
        bars.text(
            xi,
            g * 100 + (-1.5 if g < 0 else 0.5),
            f"{g * 100:.1f}pp",
            ha="center",
            va="top" if g < 0 else "bottom",
            fontsize=7.5,
            color=P_DGRAY,
        )

    fig.suptitle(
        "T8 MC Dropout Calibration: Reliability Diagram (S=30, 100 graphs, 3,163,500 nodes)",
        fontsize=11,
        color=P_DGRAY,
        fontweight="bold",
        y=1.02,
    )

    # Footnote
    fig.text(
        0.5, -0.04, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )

    fig.tight_layout()
    save(fig, "t8_reliability_diagram")

    return results


# ===========================================================================
# ANALYSIS 3.2 — Stratified UQ by Feature Bins
# ===========================================================================
def analysis_32_stratified_uq():
    """
    Stratified UQ analysis: break down uncertainty quality by feature quartiles.

    Memory-efficient approach: process per-graph NPZ + per-graph batch features,
    accumulate statistics without holding entire dataset in memory.

    Reference: Ovadia et al. (2019), NeurIPS; Barber et al. (2021), Info&Inference.
    """
    print("\n=== Analysis 3.2: Stratified UQ by Feature Bins ===")

    import torch
    import gc

    # First pass: collect all features to compute quartile boundaries
    # Load test_dl.pt batch by batch, extract features only
    print("  Loading test_dl.pt for feature statistics...")
    test_dl = torch.load(TEST_DL_PT, map_location="cpu", weights_only=False)
    n_graphs = len(test_dl)
    nodes_per_graph = test_dl[0].x.shape[0]
    n_features = test_dl[0].x.shape[1]
    print(f"  {n_graphs} graphs, {nodes_per_graph} nodes/graph, {n_features} features")

    # Collect feature values per column to compute quartiles (use subsampling for memory)
    # Sample ~50K nodes per feature for quartile estimation
    SAMPLE_SIZE = 50_000
    rng = np.random.RandomState(42)
    feature_samples = [[] for _ in range(n_features)]
    for batch in test_dl:
        x = batch.x.numpy().astype(np.float32)
        n = x.shape[0]
        idx = rng.choice(n, min(SAMPLE_SIZE // n_graphs + 1, n), replace=False)
        for fi in range(n_features):
            feature_samples[fi].append(x[idx, fi])

    feature_quartile_bounds = {}
    for fi in range(n_features):
        vals = np.concatenate(feature_samples[fi])
        feature_quartile_bounds[fi] = np.quantile(vals, [0.25, 0.50, 0.75])
    del feature_samples
    gc.collect()
    print("  Computed quartile boundaries from subsampled features")

    # Feature mapping
    feature_names = {
        0: "VOL_BASE_CASE",
        1: "CAPACITY_BASE_CASE",
        2: "CAPACITY_REDUCTION",
        3: "FREESPEED",
        4: "LENGTH",
    }
    feature_short = {0: "VOL", 1: "CAP", 2: "CAP_RED", 3: "SPD", 4: "LEN"}
    feature_units = {0: "veh/h", 1: "veh/h", 2: "fraction", 3: "m/s", 4: "m"}

    # Second pass: process each graph, accumulate per-quartile statistics
    # For each feature x quartile, collect: sum_errors, sum_sigma, count,
    # sum_residuals_le_q90, sum_residuals_le_q95, sigma+error pairs (subsampled for rho)

    # Conformal quantile: use first 20 graphs as calibration
    n_cal_graphs = 20

    # First compute conformal quantile from calibration graphs
    print("  Computing conformal quantile from calibration graphs...")
    cal_residuals = []
    for g in range(n_cal_graphs):
        npz_path = os.path.join(PER_GRAPH_DIR, f"graph_{g:04d}.npz")
        data = np.load(npz_path)
        cal_residuals.append(np.abs(data["targets"] - data["predictions"]))
    cal_residuals = np.concatenate(cal_residuals)
    q90 = float(np.quantile(cal_residuals, 0.90))
    q95 = float(np.quantile(cal_residuals, 0.95))
    del cal_residuals
    gc.collect()
    print(f"  Conformal quantiles: q90={q90:.3f}, q95={q95:.3f}")

    # Initialize accumulators: for each feature, for each quartile (0-3)
    accum = {}
    for fi in range(n_features):
        accum[fi] = {}
        for qi in range(4):
            accum[fi][qi] = {
                "count": 0,
                "sum_errors": 0.0,
                "sum_sigma": 0.0,
                "cov90_count": 0,
                "cov95_count": 0,
                "feat_min": float("inf"),
                "feat_max": float("-inf"),
                "sigma_samples": [],
                "error_samples": [],
            }

    # Process each graph
    SAMPLE_PER_GRAPH_FOR_RHO = 500  # sample 500 nodes/graph for rho -> 50K total max
    print("  Processing 100 graphs for stratified statistics...")
    for g in range(n_graphs):
        # Load per-graph NPZ
        npz_path = os.path.join(PER_GRAPH_DIR, f"graph_{g:04d}.npz")
        data = np.load(npz_path)
        preds = data["predictions"]
        sigmas_g = data["uncertainties"]
        targets_g = data["targets"]
        abs_errors_g = np.abs(targets_g - preds)
        abs_residuals_g = abs_errors_g  # same thing for these

        # Load features for this graph
        feat_g = test_dl[g].x.numpy().astype(np.float32)

        for fi in range(n_features):
            feat_vals = feat_g[:, fi]
            qbounds = feature_quartile_bounds[fi]
            quartile_assign = np.digitize(feat_vals, qbounds)  # 0,1,2,3

            for qi in range(4):
                mask = quartile_assign == qi
                n_q = np.sum(mask)
                if n_q == 0:
                    continue

                acc = accum[fi][qi]
                acc["count"] += n_q
                acc["sum_errors"] += float(np.sum(abs_errors_g[mask]))
                acc["sum_sigma"] += float(np.sum(sigmas_g[mask]))
                acc["cov90_count"] += int(np.sum(abs_residuals_g[mask] <= q90))
                acc["cov95_count"] += int(np.sum(abs_residuals_g[mask] <= q95))
                acc["feat_min"] = min(acc["feat_min"], float(np.min(feat_vals[mask])))
                acc["feat_max"] = max(acc["feat_max"], float(np.max(feat_vals[mask])))

                # Subsample for rho computation
                if n_q > SAMPLE_PER_GRAPH_FOR_RHO:
                    idx = rng.choice(n_q, SAMPLE_PER_GRAPH_FOR_RHO, replace=False)
                    acc["sigma_samples"].append(sigmas_g[mask][idx])
                    acc["error_samples"].append(abs_errors_g[mask][idx])
                else:
                    acc["sigma_samples"].append(sigmas_g[mask])
                    acc["error_samples"].append(abs_errors_g[mask])

        if (g + 1) % 20 == 0:
            print(f"    Graph {g + 1}/100 done")

    del test_dl
    gc.collect()

    # Compute final statistics
    quartile_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    results_all = {}

    for fi in range(n_features):
        fname = feature_names[fi]
        short = feature_short[fi]
        feat_results = {
            "feature": fname,
            "short_name": short,
            "quartile_boundaries": feature_quartile_bounds[fi].tolist(),
            "quartiles": {},
        }

        print(f"\n  Feature: {fname}")
        print(
            f"  {'Quartile':>12s}  {'Range':>20s}  {'N':>10s}  {'rho':>7s}  {'MAE':>8s}  {'MeanSig':>8s}  {'Cov90':>7s}  {'Cov95':>7s}"
        )

        for qi in range(4):
            acc = accum[fi][qi]
            n_q = acc["count"]
            if n_q < 100:
                continue

            mae = acc["sum_errors"] / n_q
            mean_sigma = acc["sum_sigma"] / n_q
            cov90 = acc["cov90_count"] / n_q
            cov95 = acc["cov95_count"] / n_q

            # Compute Spearman rho from collected samples (cap at 50K for memory safety)
            all_sig = np.concatenate(acc["sigma_samples"])
            all_err = np.concatenate(acc["error_samples"])
            MAX_RHO_SAMPLES = 50_000
            if len(all_sig) > MAX_RHO_SAMPLES:
                idx_sub = rng.choice(len(all_sig), MAX_RHO_SAMPLES, replace=False)
                all_sig = all_sig[idx_sub]
                all_err = all_err[idx_sub]
            rho, p_val = stats.spearmanr(all_sig, all_err)
            del all_sig, all_err

            quartile_data = {
                "label": quartile_labels[qi],
                "n_nodes": int(n_q),
                "feature_range": [acc["feat_min"], acc["feat_max"]],
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "mae_veh_h": float(mae),
                "mean_sigma_veh_h": float(mean_sigma),
                "conformal_coverage_90": float(cov90),
                "conformal_coverage_95": float(cov95),
            }
            feat_results["quartiles"][quartile_labels[qi]] = quartile_data

            range_str = f"[{acc['feat_min']:.2f}, {acc['feat_max']:.2f}]"
            print(
                f"  {quartile_labels[qi]:>12s}  {range_str:>20s}  {n_q:>10,d}  {rho:7.4f}  {mae:8.3f}  {mean_sigma:8.3f}  {cov90:7.1%}  {cov95:7.1%}"
            )

        results_all[short] = feat_results

        # Free rho sample memory
        for qi in range(4):
            accum[fi][qi]["sigma_samples"] = []
            accum[fi][qi]["error_samples"] = []

    del accum
    gc.collect()

    N = n_graphs * nodes_per_graph

    # Save results JSON
    ensure_results_dir()
    results_out = {
        "analysis": "stratified_uq_by_features",
        "description": "UQ quality stratified by input feature quartiles",
        "n_nodes_total": int(N),
        "conformal_calibration": {"n_cal_graphs": n_cal_graphs, "q90": q90, "q95": q95},
        "features": results_all,
        "references": [
            "Ovadia et al. (2019), NeurIPS",
            "Barber et al. (2021), Information and Inference",
        ],
    }
    json_path = os.path.join(RESULTS_DIR, "stratified_uq_t8.json")
    with open(json_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # --- Figure: 2-row panel, one column per feature ---
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    feature_order = [0, 1, 2, 3, 4]  # VOL, CAP, CAP_RED, SPD, LEN
    all_quartile_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    all_quartile_short = ["Q1", "Q2", "Q3", "Q4"]

    for fi, col_idx in enumerate(feature_order):
        short = feature_short[col_idx]
        unit = feature_units[col_idx]
        feat_data = results_all[short]

        # Only use quartiles that actually exist for this feature
        avail_ql = [ql for ql in all_quartile_labels if ql in feat_data["quartiles"]]
        avail_short = [
            all_quartile_short[all_quartile_labels.index(ql)] for ql in avail_ql
        ]
        x = np.arange(len(avail_ql))

        # Extract per-quartile values
        rhos = [feat_data["quartiles"][ql]["spearman_rho"] for ql in avail_ql]
        maes = [feat_data["quartiles"][ql]["mae_veh_h"] for ql in avail_ql]
        mean_sigmas = [
            feat_data["quartiles"][ql]["mean_sigma_veh_h"] for ql in avail_ql
        ]
        cov90s = [
            feat_data["quartiles"][ql]["conformal_coverage_90"] * 100 for ql in avail_ql
        ]

        # Top row: Spearman rho by quartile
        ax_top = axes[0, fi]
        if len(x) > 0:
            ax_top.bar(
                x, rhos, color=P_BLUE, width=0.6, edgecolor=P_LGRAY, linewidth=0.5
            )
            ax_top.axhline(
                0.4820,
                color=P_CORAL,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Global ρ",
            )
            ax_top.set_xticks(x)
            ax_top.set_xticklabels(avail_short, fontsize=8)
            ax_top.set_ylim(0, max(rhos) * 1.3 if max(rhos) > 0 else 0.6)
            for i, v in enumerate(rhos):
                ax_top.text(
                    i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color=P_DGRAY
                )
        ax_top.set_title(
            f"{short}\n({unit})", fontsize=9, color=P_DGRAY, fontweight="bold"
        )
        ax_top.grid(True, axis="y", alpha=0.3)
        ax_top.tick_params(colors=P_SLATE)
        if fi == 0:
            ax_top.set_ylabel("Spearman ρ", color=P_DGRAY)

        # Bottom row: MAE by quartile (bars) with mean sigma (line, secondary axis)
        ax_bot = axes[1, fi]
        if len(x) > 0:
            ax_bot.bar(
                x, maes, color=P_GREEN, width=0.6, edgecolor=P_LGRAY, linewidth=0.5
            )
            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels(avail_short, fontsize=8)
            # Secondary y-axis for mean sigma
            ax_sig = ax_bot.twinx()
            ax_sig.plot(
                x,
                mean_sigmas,
                "o-",
                color=P_CORAL,
                linewidth=1.5,
                markersize=5,
                markeredgecolor=WHITE,
                markeredgewidth=0.8,
                zorder=5,
            )
            ax_sig.tick_params(axis="y", colors=P_CORAL, labelsize=8)
            if fi == 4:
                ax_sig.set_ylabel("Mean σ (veh/h)", color=P_CORAL, fontsize=9)
            for i, v in enumerate(maes):
                ax_bot.text(
                    i, v + 0.05, f"{v:.2f}", ha="center", fontsize=7, color=P_DGRAY
                )
        ax_bot.grid(True, axis="y", alpha=0.3)
        ax_bot.tick_params(colors=P_SLATE)
        if fi == 0:
            ax_bot.set_ylabel("MAE (veh/h)", color=P_DGRAY)

    fig.suptitle(
        "T8 Stratified UQ: Uncertainty Quality by Feature Quartiles (S=30, 100 graphs)",
        fontsize=12,
        color=P_DGRAY,
        fontweight="bold",
        y=1.02,
    )

    # Row labels
    fig.text(
        0.01,
        0.75,
        "Spearman ρ\n(uncertainty ranking)",
        fontsize=9,
        color=P_DGRAY,
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.30,
        "MAE & Mean σ\n(error magnitude)",
        fontsize=9,
        color=P_DGRAY,
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )

    fig.text(
        0.5, -0.04, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.98])
    save(fig, "t8_stratified_uq")

    # --- Figure 2: Conformal conditional coverage ---
    fig2, axes2 = plt.subplots(1, 5, figsize=(16, 4.5), sharey=True)
    fig2.patch.set_facecolor(BG)

    feat_colors_fig2 = [P_BLUE, P_BLUE_DK, P_AMBER, P_GREEN, P_PURPLE]

    for fi, col_idx in enumerate(feature_order):
        short = feature_short[col_idx]
        feat_data = results_all[short]
        avail_ql = [ql for ql in all_quartile_labels if ql in feat_data["quartiles"]]
        avail_short = [
            all_quartile_short[all_quartile_labels.index(ql)] for ql in avail_ql
        ]
        x = np.arange(len(avail_ql))

        cov90s = [
            feat_data["quartiles"][ql]["conformal_coverage_90"] * 100 for ql in avail_ql
        ]

        ax2 = axes2[fi]
        ax2.bar(
            x,
            cov90s,
            color=feat_colors_fig2[fi],
            width=0.6,
            edgecolor=P_LGRAY,
            linewidth=0.5,
        )
        ax2.axhline(90, color=P_CORAL, linestyle="--", linewidth=1.5, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(avail_short, fontsize=8)
        ax2.set_title(short, fontsize=10, color=P_DGRAY, fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.tick_params(colors=P_SLATE)
        for i, v in enumerate(cov90s):
            ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7, color=P_DGRAY)
        if fi == 0:
            ax2.set_ylabel("Conformal Coverage (%)", color=P_DGRAY)

    # Set common y range
    all_covs = []
    for fi, col_idx in enumerate(feature_order):
        short = feature_short[col_idx]
        feat_data = results_all[short]
        for ql in all_quartile_labels:
            if ql in feat_data["quartiles"]:
                all_covs.append(
                    feat_data["quartiles"][ql]["conformal_coverage_90"] * 100
                )
    axes2[0].set_ylim(min(all_covs) - 3, max(all_covs) + 3)

    fig2.suptitle(
        "Conditional Conformal Coverage by Feature Quartile (T8, 90% nominal)",
        fontsize=11,
        color=P_DGRAY,
        fontweight="bold",
        y=1.02,
    )
    fig2.text(
        0.5, -0.04, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig2.tight_layout()
    save(fig2, "t8_conditional_coverage")

    return results_out


# ===========================================================================
# ANALYSIS 3.3 — Conformal Conditional Coverage (sigma-stratified)
# ===========================================================================
def analysis_33_conformal_conditional():
    """
    Conformal conditional coverage: stratify by MC Dropout sigma deciles
    and measure whether conformal coverage holds within each subgroup.

    Reference: Barber et al. (2021), "The Limits of Distribution-Free
    Conditional Predictive Inference," Information and Inference.
    """
    print("\n=== Analysis 3.3: Conformal Conditional Coverage (sigma-stratified) ===")

    import pandas as pd
    import gc

    print("  Loading ablation CSV...")
    df = pd.read_csv(ABLATION_CSV, usecols=["target", "pred_mc_mean", "pred_mc_std"])
    N = len(df)

    targets = df["target"].values.copy()
    pred_mc = df["pred_mc_mean"].values.copy()
    sigmas = df["pred_mc_std"].values.copy()
    del df
    gc.collect()
    abs_residuals = np.abs(targets - pred_mc)

    # 20/80 split for calibration
    nodes_per_graph = 31635
    n_cal = 20 * nodes_per_graph  # 632,700 calibration nodes

    cal_residuals = abs_residuals[:n_cal]
    eval_residuals = abs_residuals[n_cal:]
    eval_sigmas = sigmas[n_cal:]
    eval_targets = targets[n_cal:]
    eval_preds = pred_mc[n_cal:]
    N_eval = len(eval_residuals)

    # Global conformal quantile
    q90 = np.quantile(cal_residuals, 0.90)
    q95 = np.quantile(cal_residuals, 0.95)

    # Adaptive conformal quantile (sigma-normalized)
    cal_sigmas = sigmas[:n_cal]
    eps = 1e-6
    cal_normalized = cal_residuals / (cal_sigmas + eps)
    q90_adapt = np.quantile(cal_normalized, 0.90)
    q95_adapt = np.quantile(cal_normalized, 0.95)

    print(f"  Global q90={q90:.3f}, q95={q95:.3f}")
    print(f"  Adaptive q90_adapt={q90_adapt:.3f}, q95_adapt={q95_adapt:.3f}")

    # Stratify eval nodes by sigma deciles
    sigma_decile_bounds = np.quantile(eval_sigmas, np.arange(0.1, 1.0, 0.1))
    sigma_deciles = np.digitize(eval_sigmas, sigma_decile_bounds)  # 0-9

    results_deciles = []
    print(
        f"\n  {'Decile':>8s}  {'σ range':>20s}  {'N':>10s}  {'Glob90':>8s}  {'Glob95':>8s}  {'Adpt90':>8s}  {'Adpt95':>8s}  {'MAE':>8s}"
    )

    for d in range(10):
        mask = sigma_deciles == d
        n_d = np.sum(mask)
        if n_d < 10:
            continue

        d_residuals = eval_residuals[mask]
        d_sigmas = eval_sigmas[mask]

        # Global conformal coverage
        glob_cov90 = np.mean(d_residuals <= q90)
        glob_cov95 = np.mean(d_residuals <= q95)

        # Adaptive conformal coverage
        d_normalized = d_residuals / (d_sigmas + eps)
        adapt_cov90 = np.mean(d_normalized <= q90_adapt)
        adapt_cov95 = np.mean(d_normalized <= q95_adapt)

        # MAE in this decile
        d_mae = np.mean(d_residuals)

        sig_min = np.min(d_sigmas)
        sig_max = np.max(d_sigmas)

        decile_data = {
            "decile": d + 1,
            "sigma_range": [float(sig_min), float(sig_max)],
            "n_nodes": int(n_d),
            "global_coverage_90": float(glob_cov90),
            "global_coverage_95": float(glob_cov95),
            "adaptive_coverage_90": float(adapt_cov90),
            "adaptive_coverage_95": float(adapt_cov95),
            "mae_veh_h": float(d_mae),
            "mean_sigma": float(np.mean(d_sigmas)),
        }
        results_deciles.append(decile_data)

        range_str = f"[{sig_min:.3f}, {sig_max:.3f}]"
        print(
            f"  D{d + 1:>6d}  {range_str:>20s}  {n_d:>10,d}  {glob_cov90:8.1%}  {glob_cov95:8.1%}  {adapt_cov90:8.1%}  {adapt_cov95:8.1%}  {d_mae:8.3f}"
        )

    # Save results
    ensure_results_dir()
    results_out = {
        "analysis": "conformal_conditional_coverage_by_sigma",
        "description": "Conformal coverage stratified by MC Dropout sigma deciles",
        "n_eval_nodes": int(N_eval),
        "n_cal_nodes": int(n_cal),
        "global_quantiles": {"q90": float(q90), "q95": float(q95)},
        "adaptive_quantiles": {
            "q90_adapt": float(q90_adapt),
            "q95_adapt": float(q95_adapt),
        },
        "sigma_deciles": results_deciles,
        "reference": "Barber et al. (2021), Information and Inference",
    }
    json_path = os.path.join(RESULTS_DIR, "conformal_conditional_coverage_t8.json")
    with open(json_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # --- Figure: Conditional coverage by sigma decile ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)

    decile_nums = [d["decile"] for d in results_deciles]
    x = np.arange(len(decile_nums))

    glob90 = [d["global_coverage_90"] * 100 for d in results_deciles]
    glob95 = [d["global_coverage_95"] * 100 for d in results_deciles]
    adapt90 = [d["adaptive_coverage_90"] * 100 for d in results_deciles]
    adapt95 = [d["adaptive_coverage_95"] * 100 for d in results_deciles]

    # Left: Global conformal
    width = 0.35
    ax1.bar(
        x - width / 2,
        glob90,
        width,
        color=P_BLUE,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="90% coverage",
    )
    ax1.bar(
        x + width / 2,
        glob95,
        width,
        color=P_BLUE_DK,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="95% coverage",
    )
    ax1.axhline(
        90, color=P_CORAL, linestyle="--", linewidth=1.2, alpha=0.7, label="Nominal 90%"
    )
    ax1.axhline(
        95,
        color=P_CORAL_LT,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Nominal 95%",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"D{d}" for d in decile_nums], fontsize=8)
    ax1.set_xlabel("Uncertainty decile (D1=lowest σ, D10=highest σ)", color=P_DGRAY)
    ax1.set_ylabel("Coverage (%)", color=P_DGRAY)
    ax1.set_title("Global Conformal", color=P_DGRAY, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.tick_params(colors=P_SLATE)
    ax1.set_ylim(min(min(glob90), min(glob95)) - 5, 100)

    # Right: Adaptive conformal
    ax2.bar(
        x - width / 2,
        adapt90,
        width,
        color=P_GREEN,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="90% coverage",
    )
    ax2.bar(
        x + width / 2,
        adapt95,
        width,
        color=P_GREEN_LT,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="95% coverage",
    )
    ax2.axhline(
        90, color=P_CORAL, linestyle="--", linewidth=1.2, alpha=0.7, label="Nominal 90%"
    )
    ax2.axhline(
        95,
        color=P_CORAL_LT,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Nominal 95%",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{d}" for d in decile_nums], fontsize=8)
    ax2.set_xlabel("Uncertainty decile (D1=lowest σ, D10=highest σ)", color=P_DGRAY)
    ax2.set_ylabel("Coverage (%)", color=P_DGRAY)
    ax2.set_title("Adaptive Conformal (σ-normalized)", color=P_DGRAY, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.tick_params(colors=P_SLATE)
    ax2.set_ylim(min(min(adapt90), min(adapt95)) - 5, 100)

    fig.suptitle(
        "T8 Conformal Coverage by Uncertainty Decile: Global vs Adaptive (20/80 split)",
        fontsize=11.5,
        color=P_DGRAY,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5, -0.04, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig.tight_layout()
    save(fig, "t8_conformal_conditional")

    return results_out


# ===========================================================================
# ANALYSIS 3.4 — Per-Graph Uncertainty Variation
# ===========================================================================
def analysis_34_per_graph_variation():
    """
    Per-graph UQ statistics: compute Spearman rho, MAE, coverage for each
    of the 100 test graphs individually.

    Reference: Angelopoulos & Bates (2023), Psaros et al. (2023).
    """
    print("\n=== Analysis 3.4: Per-Graph Uncertainty Variation ===")

    n_graphs = 100
    per_graph_rhos = []
    per_graph_maes = []
    per_graph_sigmas = []
    per_graph_cov90_raw = []
    per_graph_cov95_raw = []

    print("  Processing 100 per-graph NPZ files...")
    for g in range(n_graphs):
        npz_path = os.path.join(PER_GRAPH_DIR, f"graph_{g:04d}.npz")
        data = np.load(npz_path)
        preds = data["predictions"]
        sigmas = data["uncertainties"]
        targets = data["targets"]

        abs_errors = np.abs(targets - preds)

        # Spearman rho
        rho, _ = stats.spearmanr(sigmas, abs_errors)
        per_graph_rhos.append(rho)

        # MAE
        mae = np.mean(abs_errors)
        per_graph_maes.append(mae)

        # Mean sigma
        mean_sig = np.mean(sigmas)
        per_graph_sigmas.append(mean_sig)

        # Raw Gaussian coverage at 90% and 95%
        z90 = stats.norm.ppf(0.95)  # 1.645
        z95 = stats.norm.ppf(0.975)  # 1.960
        cov90 = np.mean(abs_errors <= z90 * sigmas)
        cov95 = np.mean(abs_errors <= z95 * sigmas)
        per_graph_cov90_raw.append(cov90)
        per_graph_cov95_raw.append(cov95)

    per_graph_rhos = np.array(per_graph_rhos)
    per_graph_maes = np.array(per_graph_maes)
    per_graph_sigmas = np.array(per_graph_sigmas)
    per_graph_cov90_raw = np.array(per_graph_cov90_raw)
    per_graph_cov95_raw = np.array(per_graph_cov95_raw)

    print(
        f"\n  Per-graph Spearman ρ: mean={np.mean(per_graph_rhos):.4f}, "
        f"std={np.std(per_graph_rhos):.4f}, min={np.min(per_graph_rhos):.4f}, "
        f"max={np.max(per_graph_rhos):.4f}"
    )
    print(
        f"  Per-graph MAE: mean={np.mean(per_graph_maes):.3f}, "
        f"std={np.std(per_graph_maes):.3f}, min={np.min(per_graph_maes):.3f}, "
        f"max={np.max(per_graph_maes):.3f}"
    )
    print(
        f"  Per-graph mean σ: mean={np.mean(per_graph_sigmas):.4f}, "
        f"std={np.std(per_graph_sigmas):.4f}"
    )
    print(
        f"  Per-graph raw 90% cov: mean={np.mean(per_graph_cov90_raw):.1%}, "
        f"std={np.std(per_graph_cov90_raw):.1%}"
    )
    print(
        f"  Per-graph raw 95% cov: mean={np.mean(per_graph_cov95_raw):.1%}, "
        f"std={np.std(per_graph_cov95_raw):.1%}"
    )

    # Percentiles
    rho_pcts = np.percentile(per_graph_rhos, [5, 25, 50, 75, 95])
    mae_pcts = np.percentile(per_graph_maes, [5, 25, 50, 75, 95])

    print(f"  Rho percentiles [5,25,50,75,95]: {rho_pcts}")
    print(f"  MAE percentiles [5,25,50,75,95]: {mae_pcts}")

    # Save results
    ensure_results_dir()
    results = {
        "analysis": "per_graph_uncertainty_variation",
        "description": "Per-graph UQ statistics across 100 test graphs",
        "n_graphs": n_graphs,
        "nodes_per_graph": 31635,
        "mc_samples": 30,
        "spearman_rho": {
            "mean": float(np.mean(per_graph_rhos)),
            "std": float(np.std(per_graph_rhos)),
            "min": float(np.min(per_graph_rhos)),
            "max": float(np.max(per_graph_rhos)),
            "median": float(np.median(per_graph_rhos)),
            "percentiles_5_25_50_75_95": rho_pcts.tolist(),
            "all_values": per_graph_rhos.tolist(),
        },
        "mae_veh_h": {
            "mean": float(np.mean(per_graph_maes)),
            "std": float(np.std(per_graph_maes)),
            "min": float(np.min(per_graph_maes)),
            "max": float(np.max(per_graph_maes)),
            "median": float(np.median(per_graph_maes)),
            "percentiles_5_25_50_75_95": mae_pcts.tolist(),
            "all_values": per_graph_maes.tolist(),
        },
        "mean_sigma_veh_h": {
            "mean": float(np.mean(per_graph_sigmas)),
            "std": float(np.std(per_graph_sigmas)),
            "all_values": per_graph_sigmas.tolist(),
        },
        "raw_gaussian_coverage_90": {
            "mean": float(np.mean(per_graph_cov90_raw)),
            "std": float(np.std(per_graph_cov90_raw)),
            "all_values": per_graph_cov90_raw.tolist(),
        },
        "raw_gaussian_coverage_95": {
            "mean": float(np.mean(per_graph_cov95_raw)),
            "std": float(np.std(per_graph_cov95_raw)),
            "all_values": per_graph_cov95_raw.tolist(),
        },
        "references": [
            "Angelopoulos & Bates (2023), FnTML",
            "Psaros et al. (2023), JCP",
        ],
    }
    json_path = os.path.join(RESULTS_DIR, "per_graph_variation_t8.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # --- Figure: 4-panel per-graph variation ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(BG)

    # Panel 1: Histogram of per-graph rho
    ax = axes[0, 0]
    ax.hist(
        per_graph_rhos,
        bins=20,
        color=P_BLUE,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axvline(
        np.mean(per_graph_rhos),
        color=P_CORAL,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {np.mean(per_graph_rhos):.3f}",
    )
    ax.axvline(
        0.4820, color=P_AMBER, linestyle=":", linewidth=1.5, label=f"Aggregate = 0.482"
    )
    ax.set_xlabel("Spearman ρ (per graph)", color=P_DGRAY)
    ax.set_ylabel("Number of graphs", color=P_DGRAY)
    ax.set_title("Per-Graph Spearman ρ Distribution", color=P_DGRAY, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    # Panel 2: Histogram of per-graph MAE
    ax = axes[0, 1]
    ax.hist(
        per_graph_maes,
        bins=20,
        color=P_GREEN,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axvline(
        np.mean(per_graph_maes),
        color=P_CORAL,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {np.mean(per_graph_maes):.2f}",
    )
    ax.set_xlabel("MAE (veh/h, per graph)", color=P_DGRAY)
    ax.set_ylabel("Number of graphs", color=P_DGRAY)
    ax.set_title("Per-Graph MAE Distribution", color=P_DGRAY, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    # Panel 3: Scatter rho vs MAE
    ax = axes[1, 0]
    ax.scatter(
        per_graph_maes,
        per_graph_rhos,
        c=per_graph_sigmas,
        cmap="coolwarm",
        s=30,
        alpha=0.7,
        edgecolors=P_LGRAY,
        linewidth=0.3,
    )
    ax.set_xlabel("MAE (veh/h)", color=P_DGRAY)
    ax.set_ylabel("Spearman ρ", color=P_DGRAY)
    ax.set_title("ρ vs MAE (color = mean σ)", color=P_DGRAY, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=P_SLATE)
    cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean σ (veh/h)", fontsize=8)

    # Panel 4: Sorted rho with confidence band
    ax = axes[1, 1]
    sorted_rhos = np.sort(per_graph_rhos)
    graph_idx = np.arange(1, n_graphs + 1)
    ax.fill_between(graph_idx, sorted_rhos, alpha=0.3, color=P_BLUE)
    ax.plot(graph_idx, sorted_rhos, color=P_BLUE_DK, linewidth=1.5)
    ax.axhline(
        np.mean(per_graph_rhos),
        color=P_CORAL,
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {np.mean(per_graph_rhos):.3f}",
    )
    ax.axhline(
        np.median(per_graph_rhos),
        color=P_AMBER,
        linestyle=":",
        linewidth=1.2,
        label=f"Median = {np.median(per_graph_rhos):.3f}",
    )
    ax.set_xlabel("Graph rank (sorted by ρ)", color=P_DGRAY)
    ax.set_ylabel("Spearman ρ", color=P_DGRAY)
    ax.set_title("Sorted Per-Graph ρ", color=P_DGRAY, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    fig.suptitle(
        "T8 Per-Graph Uncertainty Variation (100 test graphs, 31,635 nodes each, S=30)",
        fontsize=12,
        color=P_DGRAY,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig.tight_layout()
    save(fig, "t8_per_graph_variation")

    return results


# ===========================================================================
# ANALYSIS 3.5 — Temperature Scaling Verification
# ===========================================================================
def analysis_35_temperature_scaling():
    """
    Post-hoc temperature scaling for MC Dropout uncertainty calibration.

    Method: scale all uncertainty estimates by a single scalar T:
        sigma_scaled = sigma_raw * T

    Optimal T is found by minimizing the Kuleshov-style ECE on a calibration
    set (first 20 graphs), then evaluated on a held-out evaluation set (last
    80 graphs) to prevent overfitting.

    This uses the SAME ECE definition as Analysis 3.1: for each nominal
    coverage level p, compute observed coverage using Gaussian intervals
    ±z_p * sigma, then ECE = mean |observed - nominal|.

    Reference:
      - Guo et al. (2017), "On Calibration of Modern Neural Networks", ICML
      - Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning
        Using Calibrated Regression", ICML
      - Laves et al. (2020), "Well-Calibrated Model Uncertainty with
        Temperature Scaling for Dropout Variational Inference", 4th workshop
        on Bayesian Deep Learning (NeurIPS 2019)
    """
    print("\n=== Analysis 3.5: Temperature Scaling Verification ===")

    from scipy.optimize import minimize_scalar

    # Nominal coverage levels — same as Analysis 3.1
    nominal_levels = np.array(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    )
    z_scores = stats.norm.ppf((1 + nominal_levels) / 2)

    # ----- Helper: compute ECE for a given set of (abs_residuals, sigmas, T) -----
    def compute_ece(abs_residuals, sigmas, T):
        observed = np.zeros(len(nominal_levels))
        scaled = sigmas * T
        for i, z in enumerate(z_scores):
            observed[i] = np.mean(abs_residuals <= z * scaled)
        return float(np.mean(np.abs(observed - nominal_levels))), observed

    def ece_for_opt(T, abs_residuals, sigmas):
        """Scalar objective for scipy minimize."""
        ece_val, _ = compute_ece(abs_residuals, sigmas, T)
        return ece_val

    # ----- Load calibration set (first 20 graphs) and eval set (last 80) -----
    n_total = 100
    n_cal = 20
    n_eval = n_total - n_cal

    print(f"  Split: {n_cal} calibration graphs, {n_eval} evaluation graphs")

    # Accumulate calibration data
    cal_residuals_list = []
    cal_sigmas_list = []
    for g in range(n_cal):
        npz_path = os.path.join(PER_GRAPH_DIR, f"graph_{g:04d}.npz")
        data = np.load(npz_path)
        cal_residuals_list.append(np.abs(data["targets"] - data["predictions"]))
        cal_sigmas_list.append(data["uncertainties"])

    cal_residuals = np.concatenate(cal_residuals_list)
    cal_sigmas = np.concatenate(cal_sigmas_list)
    del cal_residuals_list, cal_sigmas_list
    print(f"  Calibration set: {len(cal_residuals):,} nodes")

    # Compute original ECE on calibration set (T=1)
    ece_cal_orig, cov_cal_orig = compute_ece(cal_residuals, cal_sigmas, T=1.0)
    print(f"  Calibration ECE before scaling (T=1): {ece_cal_orig:.4f}")

    # Optimize T on calibration set
    print("  Optimizing temperature T on calibration set...")
    result = minimize_scalar(
        ece_for_opt,
        bounds=(0.1, 20.0),
        method="bounded",
        args=(cal_residuals, cal_sigmas),
        options={"xatol": 1e-4, "maxiter": 500},
    )
    T_opt = float(result.x)
    ece_cal_scaled, cov_cal_scaled = compute_ece(cal_residuals, cal_sigmas, T=T_opt)
    print(f"  Optimal T = {T_opt:.4f}")
    print(f"  Calibration ECE after scaling: {ece_cal_scaled:.4f}")

    del cal_residuals, cal_sigmas
    import gc

    gc.collect()

    # ----- Evaluate on held-out set (last 80 graphs) -----
    eval_residuals_list = []
    eval_sigmas_list = []
    for g in range(n_cal, n_total):
        npz_path = os.path.join(PER_GRAPH_DIR, f"graph_{g:04d}.npz")
        data = np.load(npz_path)
        eval_residuals_list.append(np.abs(data["targets"] - data["predictions"]))
        eval_sigmas_list.append(data["uncertainties"])

    eval_residuals = np.concatenate(eval_residuals_list)
    eval_sigmas = np.concatenate(eval_sigmas_list)
    del eval_residuals_list, eval_sigmas_list
    gc.collect()
    print(f"  Evaluation set: {len(eval_residuals):,} nodes")

    # ECE before and after on eval set
    ece_eval_orig, cov_eval_orig = compute_ece(eval_residuals, eval_sigmas, T=1.0)
    ece_eval_scaled, cov_eval_scaled = compute_ece(eval_residuals, eval_sigmas, T=T_opt)

    print(f"\n  === Evaluation Set Results ===")
    print(f"  ECE before (T=1.0):    {ece_eval_orig:.4f}")
    print(f"  ECE after  (T={T_opt:.3f}): {ece_eval_scaled:.4f}")
    print(
        f"  Improvement: {(ece_eval_orig - ece_eval_scaled) / ece_eval_orig * 100:.1f}%"
    )

    # Also compute key coverage levels before/after
    print(f"\n  {'Nominal':>8s}  {'Before':>10s}  {'After':>10s}  {'Perfect':>8s}")
    print(f"  {'--------':>8s}  {'------':>10s}  {'-----':>10s}  {'-------':>8s}")
    for i, p in enumerate(nominal_levels):
        print(
            f"  {p:8.0%}  {cov_eval_orig[i]:10.1%}  {cov_eval_scaled[i]:10.1%}  {p:8.0%}"
        )

    # Compute k95 equivalent after scaling: q_hat / sigma_hat
    # After scaling, 95% interval width = z95 * T * sigma
    # So effective k95_scaled = z95 * T = 1.96 * T
    z95 = stats.norm.ppf(0.975)
    k95_scaled = z95 * T_opt
    print(
        f"\n  Effective k95 after scaling: z_95 * T = {z95:.3f} * {T_opt:.3f} = {k95_scaled:.2f}"
    )
    print(f"  Compare with conformal k95 = 11.34")

    # ----- Save results JSON -----
    ensure_results_dir()
    results = {
        "analysis": "temperature_scaling",
        "description": "Post-hoc temperature scaling of MC Dropout sigma for calibration",
        "method": "Minimize Kuleshov ECE via scalar T: sigma_scaled = sigma * T",
        "split": f"{n_cal}/{n_eval} graph-level (first {n_cal} cal, last {n_eval} eval)",
        "calibration_nodes": int(n_cal * 31635),
        "evaluation_nodes": int(n_eval * 31635),
        "optimal_temperature_T": T_opt,
        "calibration_set": {
            "ece_before": ece_cal_orig,
            "ece_after": ece_cal_scaled,
            "coverage_before": cov_cal_orig.tolist(),
            "coverage_after": cov_cal_scaled.tolist(),
        },
        "evaluation_set": {
            "ece_before": ece_eval_orig,
            "ece_after": ece_eval_scaled,
            "ece_improvement_pct": float(
                (ece_eval_orig - ece_eval_scaled) / ece_eval_orig * 100
            ),
            "coverage_before": cov_eval_orig.tolist(),
            "coverage_after": cov_eval_scaled.tolist(),
        },
        "nominal_levels": nominal_levels.tolist(),
        "z_scores": z_scores.tolist(),
        "effective_k95_after_scaling": k95_scaled,
        "conformal_k95_for_comparison": 11.34,
        "references": [
            "Guo et al. (2017), On Calibration of Modern Neural Networks, ICML",
            "Kuleshov et al. (2018), Accurate Uncertainties for Deep Learning, ICML",
            "Laves et al. (2020), Well-Calibrated Temperature Scaling for Dropout VI, NeurIPS BDL Workshop",
        ],
    }
    json_path = os.path.join(RESULTS_DIR, "temperature_scaling_t8.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # ----- Figure: 2x2 layout -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(BG)

    # Panel 1: Reliability diagram BEFORE (eval set)
    ax = axes[0, 0]
    ax.plot(
        [0, 1], [0, 1], "--", color=P_MGRAY, linewidth=1.2, label="Perfect calibration"
    )
    ax.plot(
        nominal_levels,
        cov_eval_orig,
        "o-",
        color=P_CORAL,
        linewidth=2,
        markersize=7,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        label="Before (T=1.0)",
    )
    ax.fill_between(
        nominal_levels, nominal_levels, cov_eval_orig, alpha=0.15, color=P_CORAL
    )
    ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
    ax.set_ylabel("Observed coverage", color=P_DGRAY)
    ax.set_title("Before Temperature Scaling", color=P_DGRAY, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=P_SLATE)
    ax.text(
        0.55,
        0.12,
        f"ECE = {ece_eval_orig:.3f}",
        fontsize=11,
        fontweight="bold",
        color=P_CORAL,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=P_LGRAY, alpha=0.9
        ),
    )

    # Panel 2: Reliability diagram AFTER (eval set)
    ax = axes[0, 1]
    ax.plot(
        [0, 1], [0, 1], "--", color=P_MGRAY, linewidth=1.2, label="Perfect calibration"
    )
    ax.plot(
        nominal_levels,
        cov_eval_scaled,
        "s-",
        color=P_GREEN,
        linewidth=2,
        markersize=7,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        label=f"After (T={T_opt:.2f})",
    )
    ax.fill_between(
        nominal_levels, nominal_levels, cov_eval_scaled, alpha=0.15, color=P_GREEN
    )
    ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
    ax.set_ylabel("Observed coverage", color=P_DGRAY)
    ax.set_title(
        f"After Temperature Scaling (T={T_opt:.2f})", color=P_DGRAY, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=P_SLATE)
    ax.text(
        0.55,
        0.12,
        f"ECE = {ece_eval_scaled:.3f}",
        fontsize=11,
        fontweight="bold",
        color=P_GREEN,
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=P_LGRAY, alpha=0.9
        ),
    )

    # Panel 3: Overlay comparison
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "--", color=P_MGRAY, linewidth=1.2, label="Perfect")
    ax.plot(
        nominal_levels,
        cov_eval_orig,
        "o--",
        color=P_CORAL,
        linewidth=1.5,
        markersize=6,
        markeredgecolor=WHITE,
        markeredgewidth=1,
        alpha=0.7,
        label=f"Before (ECE={ece_eval_orig:.3f})",
    )
    ax.plot(
        nominal_levels,
        cov_eval_scaled,
        "s-",
        color=P_GREEN,
        linewidth=2,
        markersize=7,
        markeredgecolor=WHITE,
        markeredgewidth=1.2,
        label=f"After (ECE={ece_eval_scaled:.3f})",
    )
    ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
    ax.set_ylabel("Observed coverage", color=P_DGRAY)
    ax.set_title("Before vs After Comparison", color=P_DGRAY, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    # Panel 4: Coverage gap bar chart (before vs after)
    ax = axes[1, 1]
    gaps_before = (cov_eval_orig - nominal_levels) * 100
    gaps_after = (cov_eval_scaled - nominal_levels) * 100
    x = np.arange(len(nominal_levels))
    width = 0.35
    level_labels = [f"{int(p * 100)}%" for p in nominal_levels]

    bars1 = ax.bar(
        x - width / 2,
        gaps_before,
        width,
        color=P_CORAL,
        alpha=0.7,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="Before",
    )
    bars2 = ax.bar(
        x + width / 2,
        gaps_after,
        width,
        color=P_GREEN,
        alpha=0.7,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="After",
    )
    ax.axhline(0, color=P_MGRAY, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, rotation=45)
    ax.set_xlabel("Nominal coverage level", color=P_DGRAY)
    ax.set_ylabel("Coverage gap (pp)", color=P_DGRAY)
    ax.set_title("Coverage Gap Reduction", color=P_DGRAY, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    fig.suptitle(
        f"T8 Temperature Scaling Calibration (T={T_opt:.2f}, 20/80 graph split, S=30)",
        fontsize=12,
        color=P_DGRAY,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig.tight_layout()
    save(fig, "t8_temperature_scaling")

    del eval_residuals, eval_sigmas
    gc.collect()

    return results


# ===========================================================================
# ANALYSIS 3.6 — T7 Error Detection AUROC
# ===========================================================================
def analysis_36_t7_auroc():
    """
    Error detection via uncertainty ranking for Trial 7.

    Compute AUROC and AUPRC for whether MC Dropout sigma can identify
    the highest-error nodes (top-10% and top-20%).

    Also compute Spearman rho, selective prediction, and k95 for T7,
    enabling direct comparison with T8.

    Reference:
      - Ovadia et al. (2019), "Can You Trust Your Model's Uncertainty?", NeurIPS
      - Lakshminarayanan et al. (2017), "Simple and Scalable Predictive
        Uncertainty Estimation using Deep Ensembles", NeurIPS
    """
    print("\n=== Analysis 3.6: T7 Error Detection AUROC ===")

    from sklearn.metrics import roc_auc_score, average_precision_score
    import gc

    # Load T7 MC dropout data using per-graph files for memory safety
    n_graphs = 100
    print(f"  Loading T7 MC dropout data ({n_graphs} graphs)...")

    # Check if per-graph dir exists; if not, use the full NPZ
    use_per_graph = os.path.isdir(T7_PER_GRAPH_DIR)

    if use_per_graph:
        # Process per-graph for memory safety
        all_preds = []
        all_sigmas = []
        all_targets = []
        for g in range(n_graphs):
            npz_path = os.path.join(T7_PER_GRAPH_DIR, f"graph_{g:04d}.npz")
            if not os.path.exists(npz_path):
                print(f"    WARNING: {npz_path} missing, falling back to full NPZ")
                use_per_graph = False
                break
            data = np.load(npz_path)
            all_preds.append(data["predictions"])
            all_sigmas.append(data["uncertainties"])
            all_targets.append(data["targets"])
        if use_per_graph:
            preds = np.concatenate(all_preds)
            sigmas = np.concatenate(all_sigmas)
            targets = np.concatenate(all_targets)
            del all_preds, all_sigmas, all_targets
            gc.collect()

    if not use_per_graph:
        print("  Using full NPZ file...")
        data = np.load(T7_MC_NPZ)
        preds = data["predictions"].flatten()
        sigmas = data["uncertainties"].flatten()
        targets = data["targets"].flatten()

    N = len(targets)
    print(f"  Loaded {N:,} nodes")

    abs_errors = np.abs(targets - preds)

    # Spearman correlation
    # Subsample for rank correlation to avoid memory issues
    if N > 100_000:
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(N, 100_000, replace=False)
        rho, rho_p = stats.spearmanr(sigmas[idx_sample], abs_errors[idx_sample])
    else:
        rho, rho_p = stats.spearmanr(sigmas, abs_errors)
    print(f"  Spearman ρ(σ, |error|) = {rho:.4f} (p={rho_p:.2e})")

    # MAE
    mae = float(np.mean(abs_errors))
    print(f"  MAE = {mae:.3f} veh/h")

    # AUROC for top-k% error detection
    thresholds = {"top_10pct": 0.90, "top_20pct": 0.80}
    auroc_results = {}
    auprc_results = {}

    for label, quantile_thr in thresholds.items():
        error_threshold = np.quantile(abs_errors, quantile_thr)
        binary_labels = (abs_errors >= error_threshold).astype(int)
        prevalence = np.mean(binary_labels)

        auroc = roc_auc_score(binary_labels, sigmas)
        auprc = average_precision_score(binary_labels, sigmas)

        auroc_results[label] = float(auroc)
        auprc_results[label] = float(auprc)

        print(
            f"  {label}: threshold={error_threshold:.2f}, prevalence={prevalence:.1%}, "
            f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}"
        )

    # Selective prediction at 50% and 90% retention
    sorted_idx = np.argsort(sigmas)  # ascending uncertainty
    n50 = int(0.50 * N)
    n90 = int(0.90 * N)

    mae_50 = float(np.mean(abs_errors[sorted_idx[:n50]]))
    mae_90 = float(np.mean(abs_errors[sorted_idx[:n90]]))
    mae_50_reduction = (mae - mae_50) / mae * 100
    mae_90_reduction = (mae - mae_90) / mae * 100

    print(f"  Selective 50%: MAE={mae_50:.2f} ({mae_50_reduction:+.1f}%)")
    print(f"  Selective 90%: MAE={mae_90:.2f} ({mae_90_reduction:+.1f}%)")

    # k95: ratio for 95% coverage
    z95 = stats.norm.ppf(0.975)  # 1.96
    # Compute empirical coverage of raw Gaussian
    raw_coverage_95 = float(np.mean(abs_errors <= z95 * sigmas))
    print(f"  Raw Gaussian 95% coverage: {raw_coverage_95:.1%}")

    # Compute k95: find multiplier k such that 95% of abs_errors <= k * sigma
    ratios = abs_errors / np.maximum(sigmas, 1e-10)
    k95 = float(np.quantile(ratios, 0.95))
    print(f"  k95 = {k95:.2f}")

    # ----- T8 comparison values (verified) -----
    t8_comparison = {
        "rho": 0.4820,
        "mae": 3.948,
        "auroc_top_10pct": 0.7585,
        "auroc_top_20pct": 0.7401,
        "k95": 11.34,
        "selective_50_mae": 2.31,
        "selective_90_mae": 3.22,
    }

    # ----- Save results JSON -----
    ensure_results_dir()
    results = {
        "analysis": "t7_error_detection_auroc",
        "description": "Error detection and UQ quality metrics for Trial 7",
        "trial": 7,
        "n_nodes": int(N),
        "n_graphs": n_graphs,
        "mc_samples": 30,
        "spearman_rho": rho,
        "spearman_p_value": float(rho_p),
        "mae_veh_h": mae,
        "auroc": auroc_results,
        "auprc": auprc_results,
        "selective_prediction": {
            "retain_50pct": {"mae": mae_50, "reduction_pct": mae_50_reduction},
            "retain_90pct": {"mae": mae_90, "reduction_pct": mae_90_reduction},
        },
        "k95": k95,
        "raw_gaussian_coverage_95": raw_coverage_95,
        "t8_comparison": t8_comparison,
        "references": [
            "Ovadia et al. (2019), Can You Trust Your Model's Uncertainty?, NeurIPS",
            "Lakshminarayanan et al. (2017), Simple and Scalable Predictive Uncertainty, NeurIPS",
        ],
    }
    json_path = os.path.join(RESULTS_DIR, "t7_error_detection.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # ----- Figure: T7 vs T8 comparison (2x2) -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(BG)

    # Panel 1: AUROC comparison bar chart
    ax = axes[0, 0]
    metrics = ["AUROC\ntop-10%", "AUROC\ntop-20%"]
    t7_vals = [auroc_results["top_10pct"], auroc_results["top_20pct"]]
    t8_vals = [t8_comparison["auroc_top_10pct"], t8_comparison["auroc_top_20pct"]]
    x = np.arange(len(metrics))
    width = 0.3
    ax.bar(
        x - width / 2,
        t7_vals,
        width,
        color=P_PURPLE,
        alpha=0.8,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="T7",
    )
    ax.bar(
        x + width / 2,
        t8_vals,
        width,
        color=P_BLUE,
        alpha=0.8,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="T8",
    )
    # Add value labels
    for i, (v7, v8) in enumerate(zip(t7_vals, t8_vals)):
        ax.text(
            i - width / 2,
            v7 + 0.01,
            f"{v7:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=P_DGRAY,
        )
        ax.text(
            i + width / 2,
            v8 + 0.01,
            f"{v8:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=P_DGRAY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("AUROC", color=P_DGRAY)
    ax.set_title("Error Detection: AUROC", color=P_DGRAY, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_ylim(0.5, max(max(t7_vals), max(t8_vals)) + 0.08)
    ax.axhline(0.5, color=P_MGRAY, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    # Panel 2: Key metrics comparison
    ax = axes[0, 1]
    metric_names = ["Spearman ρ", "k95", "MAE\n(veh/h)"]
    t7_met = [rho, k95, mae]
    t8_met = [t8_comparison["rho"], t8_comparison["k95"], t8_comparison["mae"]]
    x = np.arange(len(metric_names))
    ax.bar(
        x - width / 2,
        t7_met,
        width,
        color=P_PURPLE,
        alpha=0.8,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="T7",
    )
    ax.bar(
        x + width / 2,
        t8_met,
        width,
        color=P_BLUE,
        alpha=0.8,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="T8",
    )
    for i, (v7, v8) in enumerate(zip(t7_met, t8_met)):
        ax.text(
            i - width / 2,
            v7 + 0.2,
            f"{v7:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=P_DGRAY,
        )
        ax.text(
            i + width / 2,
            v8 + 0.2,
            f"{v8:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=P_DGRAY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Value", color=P_DGRAY)
    ax.set_title("Key UQ Metrics: T7 vs T8", color=P_DGRAY, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    # Panel 3: Selective prediction comparison
    ax = axes[1, 0]
    retain_labels = ["50% retain", "90% retain", "Full (100%)"]
    t7_sel = [mae_50, mae_90, mae]
    t8_sel = [
        t8_comparison["selective_50_mae"],
        t8_comparison["selective_90_mae"],
        t8_comparison["mae"],
    ]
    x = np.arange(len(retain_labels))
    ax.bar(
        x - width / 2,
        t7_sel,
        width,
        color=P_PURPLE,
        alpha=0.8,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="T7",
    )
    ax.bar(
        x + width / 2,
        t8_sel,
        width,
        color=P_BLUE,
        alpha=0.8,
        edgecolor=P_LGRAY,
        linewidth=0.5,
        label="T8",
    )
    for i, (v7, v8) in enumerate(zip(t7_sel, t8_sel)):
        ax.text(
            i - width / 2,
            v7 + 0.08,
            f"{v7:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=P_DGRAY,
        )
        ax.text(
            i + width / 2,
            v8 + 0.08,
            f"{v8:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=P_DGRAY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(retain_labels)
    ax.set_ylabel("MAE (veh/h)", color=P_DGRAY)
    ax.set_title(
        "Selective Prediction: MAE by Retention", color=P_DGRAY, fontweight="bold"
    )
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=P_SLATE)

    # Panel 4: Summary text box
    ax = axes[1, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    summary_lines = [
        f"Trial 7 vs Trial 8 Summary",
        f"{'─' * 36}",
        f"",
        f"{'Metric':<22s} {'T7':>7s} {'T8':>7s}",
        f"{'─' * 36}",
        f"{'Spearman ρ':<22s} {rho:>7.3f} {t8_comparison['rho']:>7.3f}",
        f"{'k95':<22s} {k95:>7.2f} {t8_comparison['k95']:>7.2f}",
        f"{'MAE (veh/h)':<22s} {mae:>7.2f} {t8_comparison['mae']:>7.2f}",
        f"{'AUROC top-10%':<22s} {auroc_results['top_10pct']:>7.3f} {t8_comparison['auroc_top_10pct']:>7.3f}",
        f"{'AUROC top-20%':<22s} {auroc_results['top_20pct']:>7.3f} {t8_comparison['auroc_top_20pct']:>7.3f}",
        f"{'Sel. 50% MAE':<22s} {mae_50:>7.2f} {t8_comparison['selective_50_mae']:>7.2f}",
        f"{'Sel. 90% MAE':<22s} {mae_90:>7.2f} {t8_comparison['selective_90_mae']:>7.2f}",
        f"{'Raw 95% cov.':<22s} {raw_coverage_95:>6.1%}  {'55.6%':>6s}",
    ]
    text = "\n".join(summary_lines)
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="top",
        color=P_DGRAY,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor=WHITE, edgecolor=P_LGRAY, alpha=0.9
        ),
    )

    fig.suptitle(
        "T7 vs T8 Uncertainty Quality Comparison (100 graphs each, S=30)",
        fontsize=12,
        color=P_DGRAY,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5, -0.02, FOOTNOTE, ha="center", fontsize=7.5, color=P_MGRAY, style="italic"
    )
    fig.tight_layout()
    save(fig, "t7_vs_t8_uq_comparison")

    del preds, sigmas, targets, abs_errors
    gc.collect()

    return results


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Phase 3 UQ Analyses -- Figure & Result Generation")
    print("=" * 70)

    # Parse command-line args to allow running individual analyses
    import sys as _sys

    run_all = len(_sys.argv) == 1
    requested = set(_sys.argv[1:]) if not run_all else set()

    import gc

    if run_all or "3.1" in requested:
        results_31 = analysis_31_reliability_diagram()
        gc.collect()

    if run_all or "3.2" in requested:
        results_32 = analysis_32_stratified_uq()
        gc.collect()

    if run_all or "3.3" in requested:
        results_33 = analysis_33_conformal_conditional()
        gc.collect()

    if run_all or "3.4" in requested:
        results_34 = analysis_34_per_graph_variation()
        gc.collect()

    if run_all or "3.5" in requested:
        results_35 = analysis_35_temperature_scaling()
        gc.collect()

    if run_all or "3.6" in requested:
        results_36 = analysis_36_t7_auroc()
        gc.collect()

    print("\n" + "=" * 70)
    print("All requested Phase 3 analyses complete.")
    print("=" * 70)
    print(f"\nFigures saved to: {SCRIPT_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
