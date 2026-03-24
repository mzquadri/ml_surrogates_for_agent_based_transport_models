"""
Phase 2c: Compute Winkler Interval Score for T8 MC Dropout.

The Winkler (1972) interval score for a prediction interval [L, U] at level (1-alpha) is:
    S_alpha(L, U, y) = (U - L) + (2/alpha)*(L - y)*I(y < L) + (2/alpha)*(y - U)*I(y > U)

This rewards narrow intervals but penalizes missed observations.

We compute for:
  1. Gaussian intervals: [mu - z * sigma, mu + z * sigma] at 90% and 95%
  2. Conformal absolute intervals: [mu - q, mu + q] at 90% and 95%
  3. Conformal sigma-scaled intervals: [mu - k*sigma, mu + k*sigma] at 90% and 95%

Inputs:
    - mc_dropout_full_100graphs_mc30.npz (mu, sigma, y)
    - conformal_standard.json (q_90, q_95, k_90, k_95)

Outputs:
    - docs/verified/phase3_results/winkler_t8.json

Reference: Winkler (1972), "A Decision-Theoretic Approach to Interval Estimation",
           JASA, 67(337), 187-191.
"""

import json
import numpy as np
from scipy.stats import norm
from pathlib import Path

REPO = Path(
    r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models"
)
T8_UQ = (
    REPO
    / "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results"
)
MC_NPZ = T8_UQ / "mc_dropout_full_100graphs_mc30.npz"
CONFORMAL_JSON = T8_UQ / "conformal_standard.json"
RESULTS_DIR = REPO / "docs/verified/phase3_results"


def winkler_score(lower, upper, y, alpha):
    """Compute Winkler interval score for each observation.

    Args:
        lower: array of lower bounds
        upper: array of upper bounds
        y: array of true values
        alpha: significance level (e.g. 0.10 for 90% interval)

    Returns:
        Array of per-observation Winkler scores.
    """
    width = upper - lower
    penalty_below = (2.0 / alpha) * (lower - y) * (y < lower)
    penalty_above = (2.0 / alpha) * (y - upper) * (y > upper)
    return width + penalty_below + penalty_above


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MC Dropout S=30 NPZ ...")
    data = np.load(MC_NPZ)
    mu = data["predictions"].astype(np.float64)
    sigma = data["uncertainties"].astype(np.float64)
    y = data["targets"].astype(np.float64)
    N = len(y)
    print(f"  Loaded {N:,} nodes")

    print("Loading conformal results ...")
    with open(CONFORMAL_JSON) as f:
        conf = json.load(f)
    print(f"  q_90 = {conf['absolute_q_90']:.4f}")
    print(f"  q_95 = {conf['absolute_q_95']:.4f}")
    print(f"  k_90 = {conf['sigma_k_90']:.4f}")
    print(f"  k_95 = {conf['sigma_k_95']:.4f}")

    # NOTE: Conformal was computed on a 50/50 split. For Winkler, we compute
    # on ALL 3,163,500 nodes (same as CRPS/PIT) using the conformal quantiles
    # as fixed interval widths. This gives a fair apples-to-apples comparison.

    results_all = {}

    for alpha, level in [(0.10, 90), (0.05, 95)]:
        z = norm.ppf(1 - alpha / 2)
        print(f"\n--- {level}% intervals (alpha={alpha}) ---")
        print(f"  Gaussian z = {z:.4f}")

        # 1. Gaussian intervals
        L_gauss = mu - z * sigma
        U_gauss = mu + z * sigma
        w_gauss = winkler_score(L_gauss, U_gauss, y, alpha)
        coverage_gauss = float(np.mean((y >= L_gauss) & (y <= U_gauss)) * 100)
        width_gauss = float(np.mean(U_gauss - L_gauss))

        print(
            f"  Gaussian: mean Winkler = {np.mean(w_gauss):.4f}, "
            f"coverage = {coverage_gauss:.1f}%, mean width = {width_gauss:.4f}"
        )

        # 2. Conformal absolute intervals
        q = conf[f"absolute_q_{level}"]
        L_conf_abs = mu - q
        U_conf_abs = mu + q
        w_conf_abs = winkler_score(L_conf_abs, U_conf_abs, y, alpha)
        coverage_conf_abs = float(np.mean((y >= L_conf_abs) & (y <= U_conf_abs)) * 100)
        width_conf_abs = float(np.mean(U_conf_abs - L_conf_abs))

        print(
            f"  Conformal (abs): mean Winkler = {np.mean(w_conf_abs):.4f}, "
            f"coverage = {coverage_conf_abs:.1f}%, mean width = {width_conf_abs:.4f}"
        )

        # 3. Conformal sigma-scaled intervals
        k = conf[f"sigma_k_{level}"]
        L_conf_sig = mu - k * sigma
        U_conf_sig = mu + k * sigma
        w_conf_sig = winkler_score(L_conf_sig, U_conf_sig, y, alpha)
        coverage_conf_sig = float(np.mean((y >= L_conf_sig) & (y <= U_conf_sig)) * 100)
        width_conf_sig = float(np.mean(U_conf_sig - L_conf_sig))

        print(
            f"  Conformal (sigma): mean Winkler = {np.mean(w_conf_sig):.4f}, "
            f"coverage = {coverage_conf_sig:.1f}%, mean width = {width_conf_sig:.4f}"
        )

        results_all[f"{level}pct"] = {
            "alpha": alpha,
            "gaussian": {
                "z_multiplier": round(z, 4),
                "mean_winkler": round(float(np.mean(w_gauss)), 4),
                "median_winkler": round(float(np.median(w_gauss)), 4),
                "std_winkler": round(float(np.std(w_gauss)), 4),
                "coverage_pct": round(coverage_gauss, 1),
                "mean_width": round(width_gauss, 4),
            },
            "conformal_absolute": {
                "q_hat": round(q, 4),
                "mean_winkler": round(float(np.mean(w_conf_abs)), 4),
                "median_winkler": round(float(np.median(w_conf_abs)), 4),
                "std_winkler": round(float(np.std(w_conf_abs)), 4),
                "coverage_pct": round(coverage_conf_abs, 1),
                "mean_width": round(width_conf_abs, 4),
            },
            "conformal_sigma_scaled": {
                "k_hat": round(k, 4),
                "mean_winkler": round(float(np.mean(w_conf_sig)), 4),
                "median_winkler": round(float(np.median(w_conf_sig)), 4),
                "std_winkler": round(float(np.std(w_conf_sig)), 4),
                "coverage_pct": round(coverage_conf_sig, 1),
                "mean_width": round(width_conf_sig, 4),
            },
        }

    results = {
        "metric": "Winkler Interval Score",
        "reference": "Winkler (1972), JASA 67(337), 187-191",
        "model": "T8 MC Dropout S=30",
        "data_source": str(MC_NPZ),
        "conformal_source": str(CONFORMAL_JSON),
        "n_nodes": N,
        "n_graphs": 100,
        "note": "Conformal quantiles from 50/50 split applied to full dataset for comparison",
        "intervals": results_all,
    }

    out_path = RESULTS_DIR / "winkler_t8.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
