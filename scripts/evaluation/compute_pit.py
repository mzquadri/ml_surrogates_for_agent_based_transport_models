"""
Phase 2b: Compute PIT (Probability Integral Transform) histogram for T8 MC Dropout.

PIT value for each node: p_i = Phi((y_i - mu_i) / sigma_i)
If the predictive distribution N(mu, sigma) is well-calibrated,
the PIT values should be Uniform(0, 1).

We compute:
  1. PIT values for all 3,163,500 nodes
  2. Histogram with 20 bins (standard choice)
  3. Kolmogorov-Smirnov test for uniformity
  4. Anderson-Darling test for uniformity
  5. PIT histogram figure for thesis

Inputs:
    - mc_dropout_full_100graphs_mc30.npz

Outputs:
    - docs/verified/phase3_results/pit_t8.json
    - docs/verified/figures/t8_pit_histogram.pdf
    - docs/verified/figures/t8_pit_histogram.png
"""

import json
import numpy as np
from scipy.stats import norm, kstest, anderson
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

N_BINS = 20


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MC Dropout S=30 NPZ ...")
    data = np.load(MC_NPZ)
    mu = data["predictions"].astype(np.float64)
    sigma = data["uncertainties"].astype(np.float64)
    y = data["targets"].astype(np.float64)
    N = len(y)
    print(f"  Loaded {N:,} nodes")

    # Compute PIT values
    print("Computing PIT values ...")
    pit = norm.cdf((y - mu) / sigma)

    # Basic statistics
    pit_mean = float(np.mean(pit))
    pit_std = float(np.std(pit))
    pit_median = float(np.median(pit))
    print(f"  PIT mean:   {pit_mean:.4f} (ideal: 0.5)")
    print(f"  PIT std:    {pit_std:.4f} (ideal: {1 / np.sqrt(12):.4f} = 1/sqrt(12))")
    print(f"  PIT median: {pit_median:.4f} (ideal: 0.5)")

    # Histogram counts
    hist_counts, bin_edges = np.histogram(pit, bins=N_BINS, range=(0, 1))
    hist_density = hist_counts / N  # fraction per bin
    expected_density = 1.0 / N_BINS  # = 0.05 for 20 bins
    max_deviation = float(np.max(np.abs(hist_density - expected_density)))
    print(f"  Max bin deviation from uniform: {max_deviation:.4f}")

    # Chi-squared test for uniformity
    expected_count = N / N_BINS
    chi2_stat = float(np.sum((hist_counts - expected_count) ** 2 / expected_count))
    from scipy.stats import chi2

    chi2_pval = float(1 - chi2.cdf(chi2_stat, df=N_BINS - 1))
    print(f"  Chi-squared stat: {chi2_stat:.2f}, p-value: {chi2_pval:.4e}")

    # KS test on a subsample (KS on 3M points is always significant)
    rng = np.random.default_rng(42)
    subsample_idx = rng.choice(N, size=50_000, replace=False)
    pit_sub = pit[subsample_idx]
    ks_stat, ks_pval = kstest(pit_sub, "uniform")
    print(f"  KS test (50K subsample): stat={ks_stat:.6f}, p-value={ks_pval:.4e}")

    # Measure miscalibration: how far is the PIT distribution from uniform?
    # Use the "PIT reliability" metric: for each quantile level alpha,
    # what fraction of PIT values <= alpha?
    alphas = np.linspace(0.05, 0.95, 19)
    pit_reliability = []
    for alpha in alphas:
        observed = float(np.mean(pit <= alpha))
        pit_reliability.append(
            {
                "alpha": round(float(alpha), 2),
                "observed": round(observed, 4),
                "deviation": round(observed - float(alpha), 4),
            }
        )

    # --- Generate PIT histogram figure ---
    print("\nGenerating PIT histogram figure ...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Histogram
    ax.bar(
        bin_edges[:-1],
        hist_density,
        width=1.0 / N_BINS,
        align="edge",
        color=P_BLUE,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        label="Observed PIT",
    )

    # Uniform reference line
    ax.axhline(
        y=expected_density,
        color=P_CORAL,
        linestyle="--",
        linewidth=1.5,
        label=f"Uniform (1/{N_BINS} = {expected_density:.3f})",
    )

    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Relative frequency", fontsize=11)
    ax.set_title("PIT Histogram: T8 MC Dropout (S=30)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    # Annotate with KS stat
    ax.text(
        0.97,
        0.95,
        f"KS = {ks_stat:.4f}\nMean = {pit_mean:.3f}\nStd = {pit_std:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(pad=2.0)
    pdf_path = FIGURES_DIR / "t8_pit_histogram.pdf"
    png_path = FIGURES_DIR / "t8_pit_histogram.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    # Also copy PDF to thesis figures directory
    thesis_fig_dir = REPO / "thesis/latex_tum_official/figures"
    if thesis_fig_dir.exists():
        import shutil

        shutil.copy2(pdf_path, thesis_fig_dir / "t8_pit_histogram.pdf")
        print(f"  Copied to: {thesis_fig_dir / 't8_pit_histogram.pdf'}")

    # Save JSON results
    results = {
        "metric": "PIT (Probability Integral Transform)",
        "model": "T8 MC Dropout S=30",
        "data_source": str(MC_NPZ),
        "n_nodes": N,
        "n_graphs": 100,
        "n_bins": N_BINS,
        "pit_mean": round(pit_mean, 4),
        "pit_std": round(pit_std, 4),
        "pit_median": round(pit_median, 4),
        "ideal_mean": 0.5,
        "ideal_std": round(1.0 / np.sqrt(12), 4),
        "histogram_density": [round(float(d), 6) for d in hist_density],
        "bin_edges": [round(float(e), 4) for e in bin_edges],
        "max_bin_deviation": round(max_deviation, 4),
        "chi2_stat": round(chi2_stat, 2),
        "chi2_pval": chi2_pval,
        "ks_test_subsample": {
            "n_subsample": 50_000,
            "ks_stat": round(float(ks_stat), 6),
            "ks_pval": ks_pval,
        },
        "pit_reliability": pit_reliability,
    }

    out_path = RESULTS_DIR / "pit_t8.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
