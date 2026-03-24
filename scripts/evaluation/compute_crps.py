"""
Phase 2a: Compute CRPS (Continuous Ranked Probability Score) for T8 MC Dropout.

Uses the closed-form Gaussian CRPS formula from Gneiting & Raftery (2007):
    CRPS(N(mu, sigma), y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma

Inputs:
    - mc_dropout_full_100graphs_mc30.npz (predictions=mu, uncertainties=sigma, targets=y)

Outputs:
    - docs/verified/phase3_results/crps_t8.json
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
RESULTS_DIR = REPO / "docs/verified/phase3_results"


def gaussian_crps(mu, sigma, y):
    """Closed-form CRPS for Gaussian predictive distribution.

    Reference: Gneiting & Raftery (2007), Eq. (21)
    "Strictly Proper Scoring Rules, Prediction, and Estimation"
    Journal of the American Statistical Association, 102(477), 359-378.
    """
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return crps


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MC Dropout S=30 NPZ ...")
    data = np.load(MC_NPZ)
    mu = data["predictions"].astype(np.float64)
    sigma = data["uncertainties"].astype(np.float64)
    y = data["targets"].astype(np.float64)
    N = len(y)
    print(f"  Loaded {N:,} nodes")

    # Sanity checks
    assert sigma.min() > 0, f"Non-positive sigma found: min={sigma.min()}"
    assert len(mu) == len(sigma) == len(y)

    # Compute per-node CRPS
    print("Computing per-node CRPS ...")
    crps_values = gaussian_crps(mu, sigma, y)

    # Aggregate statistics
    crps_mean = float(np.mean(crps_values))
    crps_median = float(np.median(crps_values))
    crps_std = float(np.std(crps_values))
    crps_q25 = float(np.percentile(crps_values, 25))
    crps_q75 = float(np.percentile(crps_values, 75))
    crps_q95 = float(np.percentile(crps_values, 95))
    crps_min = float(np.min(crps_values))
    crps_max = float(np.max(crps_values))

    # Also compute MAE for comparison (CRPS reduces to MAE for point forecasts)
    mae = float(np.mean(np.abs(y - mu)))

    print(f"\n  CRPS Results (T8, S=30, {N:,} nodes):")
    print(f"    Mean CRPS:   {crps_mean:.4f}")
    print(f"    Median CRPS: {crps_median:.4f}")
    print(f"    Std CRPS:    {crps_std:.4f}")
    print(f"    Q25:         {crps_q25:.4f}")
    print(f"    Q75:         {crps_q75:.4f}")
    print(f"    Q95:         {crps_q95:.4f}")
    print(f"    Min:         {crps_min:.4f}")
    print(f"    Max:         {crps_max:.4f}")
    print(f"    MAE (ref):   {mae:.4f}")
    print(f"    CRPS/MAE:    {crps_mean / mae:.4f}")

    # Stratified by uncertainty deciles
    decile_edges = np.percentile(sigma, np.arange(0, 110, 10))
    decile_results = []
    for i in range(10):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        if i == 9:
            mask = (sigma >= lo) & (sigma <= hi)
        else:
            mask = (sigma >= lo) & (sigma < hi)
        if mask.sum() == 0:
            continue
        dec_crps = float(np.mean(crps_values[mask]))
        dec_mae = float(np.mean(np.abs(y[mask] - mu[mask])))
        dec_sigma_mean = float(np.mean(sigma[mask]))
        decile_results.append(
            {
                "decile": i + 1,
                "sigma_range": [round(float(lo), 4), round(float(hi), 4)],
                "n_nodes": int(mask.sum()),
                "mean_crps": round(dec_crps, 4),
                "mean_mae": round(dec_mae, 4),
                "mean_sigma": round(dec_sigma_mean, 4),
            }
        )

    results = {
        "metric": "CRPS (Gaussian closed-form)",
        "reference": "Gneiting & Raftery (2007), JASA 102(477), Eq. 21",
        "model": "T8 MC Dropout S=30",
        "data_source": str(MC_NPZ),
        "n_nodes": N,
        "n_graphs": 100,
        "crps_mean": round(crps_mean, 4),
        "crps_median": round(crps_median, 4),
        "crps_std": round(crps_std, 4),
        "crps_q25": round(crps_q25, 4),
        "crps_q75": round(crps_q75, 4),
        "crps_q95": round(crps_q95, 4),
        "crps_min": round(crps_min, 4),
        "crps_max": round(crps_max, 4),
        "mae_reference": round(mae, 4),
        "crps_over_mae": round(crps_mean / mae, 4),
        "decile_breakdown": decile_results,
    }

    out_path = RESULTS_DIR / "crps_t8.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
