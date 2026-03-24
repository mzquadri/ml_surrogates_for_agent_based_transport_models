"""
Item 6: Compute PIT after temperature scaling (sigma_scaled = sigma_raw * T).

The raw MC Dropout sigma is underdispersed (PIT KS=0.245, excess mass in tails).
Temperature scaling with T=2.7025 should widen the predictive distribution.
Question: does PIT improve toward uniformity after scaling?

PIT_scaled_i = Phi((y_i - mu_i) / (sigma_i * T))

Inputs:
    - mc_dropout_full_100graphs_mc30.npz (predictions, uncertainties, targets)
    - T = 2.7025 (from temperature_scaling_t8.json)

Outputs:
    - docs/verified/phase3_results/pit_after_tempscaling_t8.json
    - docs/verified/figures/t8_pit_after_tempscaling.pdf
    - thesis/latex_tum_official/figures/t8_pit_after_tempscaling.pdf
"""

import json
import numpy as np
from scipy.stats import norm, kstest, chi2
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure_generation"),
)
from thesis_style import *

REPO = Path(__file__).resolve().parent.parent.parent
T8_UQ = (
    REPO
    / "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results"
)
MC_NPZ = T8_UQ / "mc_dropout_full_100graphs_mc30.npz"
RESULTS_DIR = REPO / "docs/verified/phase3_results"
FIGURES_DIR = REPO / "docs/verified/figures"
THESIS_FIG_DIR = REPO / "thesis/latex_tum_official/figures"

T_OPTIMAL = 2.7025  # From temperature_scaling_t8.json
N_BINS = 20


def compute_pit_stats(pit, label, n_bins=20):
    """Compute PIT statistics for a given array of PIT values."""
    N = len(pit)
    pit_mean = float(np.mean(pit))
    pit_std = float(np.std(pit))
    pit_median = float(np.median(pit))

    hist_counts, bin_edges = np.histogram(pit, bins=n_bins, range=(0, 1))
    hist_density = hist_counts / N
    expected_density = 1.0 / n_bins
    max_deviation = float(np.max(np.abs(hist_density - expected_density)))

    # Chi-squared
    expected_count = N / n_bins
    chi2_stat = float(np.sum((hist_counts - expected_count) ** 2 / expected_count))
    chi2_pval = float(1 - chi2.cdf(chi2_stat, df=n_bins - 1))

    # KS on subsample
    rng = np.random.default_rng(42)
    subsample_idx = rng.choice(N, size=50_000, replace=False)
    pit_sub = pit[subsample_idx]
    ks_stat, ks_pval = kstest(pit_sub, "uniform")

    # First and last bin densities (U-shape indicator)
    first_bin = float(hist_density[0])
    last_bin = float(hist_density[-1])

    print(f"\n  [{label}]")
    print(f"    PIT mean:   {pit_mean:.4f} (ideal: 0.5)")
    print(f"    PIT std:    {pit_std:.4f} (ideal: {1 / np.sqrt(12):.4f})")
    print(f"    PIT median: {pit_median:.4f}")
    print(f"    KS stat:    {ks_stat:.6f}")
    print(f"    Max bin deviation: {max_deviation:.4f}")
    print(f"    First bin density: {first_bin:.4f} (ideal: {expected_density:.4f})")
    print(f"    Last bin density:  {last_bin:.4f}")

    return {
        "pit_mean": round(pit_mean, 4),
        "pit_std": round(pit_std, 4),
        "pit_median": round(pit_median, 4),
        "histogram_density": [round(float(d), 6) for d in hist_density],
        "bin_edges": [round(float(e), 4) for e in bin_edges],
        "max_bin_deviation": round(max_deviation, 4),
        "chi2_stat": round(chi2_stat, 2),
        "chi2_pval": chi2_pval,
        "ks_stat": round(float(ks_stat), 6),
        "ks_pval": float(ks_pval),
        "first_bin_density": round(first_bin, 4),
        "last_bin_density": round(last_bin, 4),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MC Dropout S=30 NPZ ...")
    data = np.load(MC_NPZ)
    mu = data["predictions"].astype(np.float64)
    sigma_raw = data["uncertainties"].astype(np.float64)
    y = data["targets"].astype(np.float64)
    N = len(y)
    print(f"  Loaded {N:,} nodes")
    print(
        f"  sigma_raw: mean={np.mean(sigma_raw):.4f}, median={np.median(sigma_raw):.4f}"
    )

    # --- Before: PIT with raw sigma ---
    print("\n=== PIT BEFORE Temperature Scaling ===")
    z_raw = (y - mu) / sigma_raw
    pit_raw = norm.cdf(z_raw)
    stats_before = compute_pit_stats(pit_raw, "Raw sigma")

    # --- After: PIT with scaled sigma ---
    sigma_scaled = sigma_raw * T_OPTIMAL
    print(f"\n=== PIT AFTER Temperature Scaling (T={T_OPTIMAL}) ===")
    print(
        f"  sigma_scaled: mean={np.mean(sigma_scaled):.4f}, median={np.median(sigma_scaled):.4f}"
    )
    z_scaled = (y - mu) / sigma_scaled
    pit_scaled = norm.cdf(z_scaled)
    stats_after = compute_pit_stats(pit_scaled, f"Scaled sigma (T={T_OPTIMAL})")

    # --- Comparison ---
    ks_improvement = (
        (stats_before["ks_stat"] - stats_after["ks_stat"])
        / stats_before["ks_stat"]
        * 100
    )
    mean_improvement_toward_05 = abs(stats_before["pit_mean"] - 0.5) - abs(
        stats_after["pit_mean"] - 0.5
    )
    print(f"\n=== COMPARISON ===")
    print(
        f"  KS stat: {stats_before['ks_stat']:.6f} -> {stats_after['ks_stat']:.6f} ({ks_improvement:+.1f}% change)"
    )
    print(f"  PIT mean closer to 0.5 by: {mean_improvement_toward_05:.4f}")
    print(
        f"  Max bin deviation: {stats_before['max_bin_deviation']:.4f} -> {stats_after['max_bin_deviation']:.4f}"
    )

    # --- Generate comparison figure (2-panel) ---
    print("\nGenerating comparison figure ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax1, ax2 = axes[0], axes[1]
    panel_label(ax1, "(a)")
    panel_label(ax2, "(b)")

    expected_density = 1.0 / N_BINS

    for ax, hist_dens, title, ks_val, pit_m, pit_s in [
        (
            axes[0],
            stats_before["histogram_density"],
            "Before Temperature Scaling (raw $\\sigma$)",
            stats_before["ks_stat"],
            stats_before["pit_mean"],
            stats_before["pit_std"],
        ),
        (
            axes[1],
            stats_after["histogram_density"],
            f"After Temperature Scaling ($\\sigma \\times T$, $T={T_OPTIMAL}$)",
            stats_after["ks_stat"],
            stats_after["pit_mean"],
            stats_after["pit_std"],
        ),
    ]:
        bin_edges = np.linspace(0, 1, N_BINS + 1)
        ax.bar(
            bin_edges[:-1],
            hist_dens,
            width=1.0 / N_BINS,
            align="edge",
            color=P_BLUE,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            label="Observed PIT",
        )
        ax.axhline(
            y=expected_density,
            color=P_CORAL,
            linestyle="--",
            linewidth=1.5,
            label=f"Uniform (1/{N_BINS} = {expected_density:.3f})",
        )
        ax.set_xlabel("PIT value", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper center")
        ax.tick_params(labelsize=9)
        ax.text(
            0.97,
            0.72,
            f"KS = {ks_val:.4f}\nMean = {pit_m:.3f}\nStd = {pit_s:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    axes[0].set_ylabel("Relative frequency", fontsize=11)

    plt.tight_layout(pad=2.0)
    pdf_path = FIGURES_DIR / "t8_pit_after_tempscaling.pdf"
    png_path = FIGURES_DIR / "t8_pit_after_tempscaling.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf_path}")

    # Copy to thesis figures
    if THESIS_FIG_DIR.exists():
        import shutil

        shutil.copy2(pdf_path, THESIS_FIG_DIR / "t8_pit_after_tempscaling.pdf")
        print(f"  Copied to thesis figures dir")

    # --- Save JSON ---
    results = {
        "metric": "PIT after Temperature Scaling",
        "model": "T8 MC Dropout S=30",
        "temperature": T_OPTIMAL,
        "data_source": str(MC_NPZ),
        "n_nodes": N,
        "n_bins": N_BINS,
        "before_tempscaling": stats_before,
        "after_tempscaling": stats_after,
        "comparison": {
            "ks_stat_reduction_pct": round(ks_improvement, 1),
            "pit_mean_improvement_toward_05": round(mean_improvement_toward_05, 4),
            "max_bin_deviation_before": stats_before["max_bin_deviation"],
            "max_bin_deviation_after": stats_after["max_bin_deviation"],
        },
    }

    out_path = RESULTS_DIR / "pit_after_tempscaling_t8.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
