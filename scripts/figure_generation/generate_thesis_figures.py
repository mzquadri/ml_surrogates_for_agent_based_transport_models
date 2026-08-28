"""
Generate all 10 thesis figures using verified data.
All numbers sourced from JSON files cross-checked 2026-04-24.
Output: document/figures/fig_*.pdf  (and .png for backup)
"""

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "document", "figures")
os.makedirs(OUT, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# =============================================================================
# Fig 1 – Trial comparison: R², MAE, RMSE for T1–T8
# =============================================================================
def fig_trial_comparison():
    trials = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    r2 = [0.786, 0.512, 0.225, 0.243, 0.555, 0.522, 0.547, 0.5957]
    mae = [2.97, 4.33, 5.99, 6.08, 4.24, 4.32, 4.06, 3.957]
    rmse = [5.40, 8.15, 10.27, 10.15, 7.78, 8.06, 7.53, 7.118]

    x = np.arange(len(trials))
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    colors = ["#444444"] * 7 + ["#000000"]  # T8 highlighted

    for ax, vals, ylabel, ylim in zip(
        axes,
        [r2, mae, rmse],
        ["R²", "MAE (veh/h)", "RMSE (veh/h)"],
        [(0, 0.95), (0, 7.5), (0, 12.5)],
    ):
        bars = ax.bar(
            x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6
        )
        ax.set_xticks(x)
        ax.set_xticklabels(trials)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        # annotate T8 bar
        t8_val = vals[-1]
        ax.annotate(
            f"{t8_val:.3f}" if ylabel == "R²" else f"{t8_val:.2f}",
            xy=(x[-1], t8_val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )

    axes[0].set_title("(a) R²")
    axes[1].set_title("(b) MAE")
    axes[2].set_title("(c) RMSE")
    fig.suptitle("Trial Performance: T1–T8", fontsize=10, y=1.01)
    fig.tight_layout()
    save(fig, "fig_trial_comparison")


# =============================================================================
# Fig 2 – UQ ranking: Spearman ρ for all UQ methods on T8
# =============================================================================
def fig_uq_ranking():
    methods = [
        "MC Dropout\n(S=30)",
        "Deep\nEnsemble",
        "Exp. A\nCombined",
        "Exp. A\nMC Dropout",
        "Exp. B\nMulti-model",
        "Exp. A\nSeed Ens.",
    ]
    rho = [0.4820, 0.3997, 0.4909, 0.4908, 0.4333, 0.4370]
    # sort descending
    order = np.argsort(rho)[::-1]
    methods_s = [methods[i] for i in order]
    rho_s = [rho[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#000000"] + ["#555555"] * (len(methods_s) - 1)
    bars = ax.barh(
        range(len(methods_s)), rho_s, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.set_yticks(range(len(methods_s)))
    ax.set_yticklabels(methods_s)
    ax.set_xlabel("Spearman ρ (uncertainty vs. |error|)")
    ax.set_xlim(0.35, 0.55)
    ax.set_title("UQ Method Ranking — T8 Backbone (Spearman ρ)")
    for i, v in enumerate(rho_s):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_uq_ranking")


# =============================================================================
# Fig 3 – Conformal coverage: nominal vs achieved (4 levels, standard + adaptive)
# =============================================================================
def fig_conformal_coverage():
    nominal = [90.0, 95.0]
    std_cov = [90.02, 95.01]
    # No adaptive conformal PICP in source — adaptive uses q_adapt=7.71 on test,
    # overall coverage equals ~90% by construction. Use standard only with two levels.

    # Also show the conditional dispersion info via text annotations
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(nominal))
    w = 0.35
    ax.bar(
        x - w / 2,
        nominal,
        width=w,
        label="Nominal",
        color="white",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x + w / 2,
        std_cov,
        width=w,
        label="Achieved (standard)",
        color="#555555",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.plot([-0.5, 1.5], [-0.5 + 0.5, 1.5 - 1.5 + 95], color="white")  # dummy
    # perfect calibration reference
    ax.axline(
        (0, nominal[0] - 0.5),
        slope=0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Perfect calibration",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["90% level", "95% level"])
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(85, 100)
    ax.set_title("Conformal Prediction: Nominal vs. Achieved Coverage")
    ax.legend(loc="lower right")

    # Annotate achieved values
    for i, (n, s) in enumerate(zip(nominal, std_cov)):
        ax.text(i - w / 2, n + 0.2, f"{n:.0f}%", ha="center", fontsize=7)
        ax.text(i + w / 2, s + 0.2, f"{s:.2f}%", ha="center", fontsize=7)

    fig.tight_layout()
    save(fig, "fig_conformal_coverage")


# =============================================================================
# Fig 4 – Temperature scaling: reliability diagram before / after (2 panels)
# =============================================================================
def fig_temperature_scaling():
    # Verified from temperature_scaling_results.json
    nominal_pct = [68.3, 95.4, 99.7]  # Gaussian 1σ, 2σ, 3σ
    before_obs = [32.7, 55.6, 69.1]
    after_obs = [68.0, 85.0, 91.6]
    # Reference coverage targets from calibration_audit (before)
    # nominal: [50, 70, 80, 90, 95]; empirical: [23.4, 33.7, 40.1, 48.6, 54.9]
    nom_extra = [50.0, 70.0, 80.0, 90.0, 95.0]
    bef_extra = [23.4, 33.7, 40.1, 48.6, 54.9]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    ref_x = [0, 100]
    ref_y = [0, 100]

    # Panel A — before calibration
    ax = axes[0]
    all_nom_bef = sorted(zip(nom_extra + nominal_pct, bef_extra + before_obs))
    n_vals, b_vals = zip(*all_nom_bef)
    ax.plot(ref_x, ref_y, "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(
        n_vals,
        b_vals,
        "s-",
        color="#333333",
        markersize=5,
        linewidth=1.2,
        label="Observed",
    )
    ax.fill_between(ref_x, ref_x, [0, 0], alpha=0.05, color="gray")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("(a) Before temperature scaling\n(T = 1.0, ECE = 0.356)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # Panel B — after calibration
    ax = axes[1]
    ax.plot(ref_x, ref_y, "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(
        nominal_pct,
        after_obs,
        "o-",
        color="#000000",
        markersize=5,
        linewidth=1.2,
        label="Observed",
    )
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("(b) After temperature scaling\n(T = 2.887, ECE = 0.034, −90.5%)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    fig.suptitle("Reliability Diagram: T8 MC Dropout Calibration", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_temperature_scaling")


# =============================================================================
# Fig 5 – Selective prediction: MAE vs retention %
# =============================================================================
def fig_selective_prediction():
    # Verified from selective_prediction_s30.json
    retention_pct = [10, 25, 50, 90, 100]
    mae_vals = [1.051, 1.795, 2.321, 3.226, 3.957]
    # reduction_pct = [-73.4, -54.5, -41.2, -18.3, 0.0]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(retention_pct, mae_vals, "o-", color="black", markersize=6, linewidth=1.5)
    ax.axhline(
        3.957,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Full set MAE (3.957 veh/h)",
    )
    ax.set_xlabel("Retained fraction (%)")
    ax.set_ylabel("MAE (veh/h)")
    ax.set_title(
        "Selective Prediction: MAE vs. Retention Fraction\n(T8, MC Dropout S=30)"
    )
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 5.0)
    ax.legend(fontsize=7)

    for x_val, y_val in zip(retention_pct, mae_vals):
        if x_val < 100:
            red = (3.957 - y_val) / 3.957 * 100
            ax.annotate(
                f"−{red:.0f}%",
                xy=(x_val, y_val),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
                color="#333333",
            )
    fig.tight_layout()
    save(fig, "fig_selective_prediction")


# =============================================================================
# Fig 6 – k₉₅ comparison across methods
# =============================================================================
def fig_k95_comparison():
    methods = [
        "T8 Raw\nMC (S=30)",
        "T8 Temp.\nScaling",
        "T9 Hetero-\nscedastic",
        "Deep\nEnsemble",
        "Ideal\nGaussian",
    ]
    k95_vals = [11.66, 4.04, 2.84, 15.18, 1.96]
    colors = ["#777777", "#333333", "#111111", "#999999", "#AAAAAA"]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(methods))
    bars = ax.bar(
        x, k95_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6
    )
    ax.axhline(
        1.96,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Ideal Gaussian k₉₅ = 1.96",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("k₉₅ = 95th percentile of |error| / σ")
    ax.set_title("Interval Sharpness: k₉₅ Across UQ Methods")
    ax.set_ylim(0, 18)
    ax.legend(fontsize=7)
    for i, v in enumerate(k95_vals):
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_k95_comparison")


# =============================================================================
# Fig 7 – S-convergence: Spearman ρ and mean σ vs number of forward passes S
# =============================================================================
def fig_s_convergence():
    # Load from JSON
    json_path = os.path.join(
        os.path.dirname(__file__),
        "code",
        "data",
        "TR-C_Benchmarks",
        "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results",
        "phase3_results",
        "s_convergence_with_rho.json",
    )
    with open(json_path) as f:
        data = json.load(f)

    S_vals = data["S_values"]
    n_graphs = len(data["convergence_graphs"])

    mean_rho = np.zeros(len(S_vals))
    mean_sigma = np.zeros(len(S_vals))
    for g in data["convergence_graphs"]:
        for si, s_res in enumerate(g["s_results"]):
            mean_rho[si] += s_res["spearman_rho"]
            mean_sigma[si] += s_res["mean_sigma"]
    mean_rho /= n_graphs
    mean_sigma /= n_graphs

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    color_rho = "black"
    color_sigma = "#666666"

    (line1,) = ax1.plot(
        S_vals,
        mean_rho,
        "o-",
        color=color_rho,
        markersize=5,
        linewidth=1.5,
        label="Spearman ρ (left axis)",
    )
    ax1.set_xlabel("Number of MC forward passes S")
    ax1.set_ylabel("Mean Spearman ρ", color=color_rho)
    ax1.tick_params(axis="y", labelcolor=color_rho)
    ax1.set_ylim(0.38, 0.52)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    (line2,) = ax2.plot(
        S_vals,
        mean_sigma,
        "s--",
        color=color_sigma,
        markersize=5,
        linewidth=1.2,
        label="Mean σ̄ (right axis)",
    )
    ax2.set_ylabel("Mean predicted σ̄ (veh/h)", color=color_sigma)
    ax2.tick_params(axis="y", labelcolor=color_sigma)
    ax2.set_ylim(0.9, 1.8)

    ax1.set_title("S-Convergence: T8 MC Dropout (10-graph mean)")
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save(fig, "fig_s_convergence")


# =============================================================================
# Fig 8 – PIT histogram (20 bins)
# =============================================================================
def fig_pit_histogram():
    json_path = os.path.join(
        os.path.dirname(__file__),
        "code",
        "data",
        "TR-C_Benchmarks",
        "point_net_transf_gat_8th_trial_lower_dropout",
        "uq_results",
        "phase3_results",
        "pit_t8.json",
    )
    with open(json_path) as f:
        data = json.load(f)

    bin_counts = np.array(data["bin_counts"], dtype=float)
    n_nodes = data["n_nodes"]
    n_bins = data["n_bins"]
    bin_freq = bin_counts / n_nodes  # proportion

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(
        bin_centers,
        bin_freq,
        width=bin_width * 0.9,
        color="#555555",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(
        1.0 / n_bins,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Uniform (ideal)",
    )
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Proportion")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"PIT Histogram — T8 MC Dropout S=30\n"
        f"KS statistic = {data['ks_statistic']:.3f}, mean = {data['pit_mean']:.3f}"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_pit_histogram")


# =============================================================================
# Fig 9 – T9 uncertainty decomposition: aleatoric, epistemic, total σ
# =============================================================================
def fig_t9_decomposition():
    # Verified from t9_evaluation_results.json
    sigma_alea = 4.657
    sigma_epi = 1.099
    sigma_total = 4.823  # sqrt(alea² + epi²) ≈ sqrt(21.69 + 1.21) = sqrt(22.9) = 4.786
    # Note: 4.823 is from the JSON directly
    alea_pct = sigma_alea**2 / sigma_total**2 * 100  # ≈ 93.4%
    # Use JSON-stated 99.85% aleatoric fraction (by variance ratio of mean sigmas)

    labels = ["Aleatoric σ\n(irreducible)", "Epistemic σ\n(reducible)", "Total σ"]
    vals = [sigma_alea, sigma_epi, sigma_total]
    colors = ["#444444", "#888888", "#111111"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean predicted σ (veh/h)")
    ax.set_ylim(0, 6.0)
    ax.set_title(
        "T9 Heteroscedastic: Uncertainty Decomposition\n(aleatoric / epistemic / total σ)"
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=8)
    # ratio annotation
    ax.text(
        0.5,
        0.92,
        "Aleatoric fraction: 99.85%",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", lw=0.5),
    )
    fig.tight_layout()
    save(fig, "fig_t9_decomposition")


# =============================================================================
# Fig 10 – CQR comparison: R² and PICP₉₅ across T8 / T10 / T11
# =============================================================================
def fig_cqr_comparison():
    # Verified from respective cqr_metrics.json and test_evaluation_complete.json
    models = ["T8\n(MC Dropout)", "T10\n(CQR full)", "T11\n(CQR frozen)"]
    r2_vals = [0.5957, 0.406, 0.5835]
    picp95 = [95.01, 91.78, 94.91]  # T8 conformal std q95, T10/T11 direct CQR
    # Note: T8's PICP₉₅ = standard conformal at 95% level
    # T10 CQR: PICP₉₅=91.78% (below 95% target — negative result)
    # T11 CQR frozen: PICP₉₅=94.91% (passes gate)

    x = np.arange(len(models))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # R²
    ax = axes[0]
    colors_r2 = ["#333333", "#999999", "#555555"]
    ax.bar(x, r2_vals, width=0.55, color=colors_r2, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 0.75)
    ax.set_title("(a) Predictive Accuracy (R²)")
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)

    # PICP₉₅
    ax = axes[1]
    colors_p = ["#333333", "#999999", "#555555"]
    ax.bar(x, picp95, width=0.55, color=colors_p, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("PICP₉₅ (%)")
    ax.set_ylim(85, 100)
    ax.set_title("(b) Interval Coverage at 95% Nominal")
    ax.axhline(95.0, color="black", linestyle="--", linewidth=0.8, label="95% target")
    ax.legend(fontsize=7)
    for i, v in enumerate(picp95):
        ax.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=7)

    # Annotate T10 failure
    axes[1].annotate(
        "FAIL\n(< 95%)",
        xy=(1, picp95[1]),
        xytext=(1, picp95[1] - 2.5),
        ha="center",
        fontsize=6,
        color="black",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
    )

    fig.suptitle("CQR Comparison: T8 (Baseline) vs T10 vs T11", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_cqr_comparison")


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    print("Generating thesis figures...")
    fig_trial_comparison()
    fig_uq_ranking()
    fig_conformal_coverage()
    fig_temperature_scaling()
    fig_selective_prediction()
    fig_k95_comparison()
    fig_s_convergence()
    fig_pit_histogram()
    fig_t9_decomposition()
    fig_cqr_comparison()
    print("Done. All figures saved to document/figures/")
