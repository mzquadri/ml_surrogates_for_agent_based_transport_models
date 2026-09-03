"""Compute Gaussian NLL from MC Dropout NPZ artifacts.

Usage: conda run -n thesis-env python scripts/compute_nll.py
"""

import json
import numpy as np
from pathlib import Path

# --- Paths ---
REPO = Path(__file__).resolve().parent.parent
T8_NPZ = (
    REPO
    / "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz"
)
T7_NPZ = (
    REPO
    / "data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/uq_results"
)
OUTPUT = REPO / "docs/verified/phase3_results/nll_results.json"


def gaussian_nll(y_true, y_pred_mean, y_pred_std, eps=1e-6):
    """Compute mean Gaussian NLL: 0.5 * mean[log(2*pi*sigma^2) + (y-mu)^2/sigma^2]."""
    sigma = np.maximum(y_pred_std, eps)
    nll_per_node = 0.5 * (
        np.log(2 * np.pi * sigma**2) + ((y_true - y_pred_mean) / sigma) ** 2
    )
    return (
        float(np.mean(nll_per_node)),
        float(np.std(nll_per_node)),
        float(np.median(nll_per_node)),
    )


def main():
    results = {}

    # --- T8 MC Dropout ---
    print("Loading T8 MC Dropout NPZ...")
    data = np.load(T8_NPZ)
    print(f"  Keys: {list(data.keys())}")
    for k in data.keys():
        print(f"    {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    # Keys are: predictions, uncertainties, targets
    y_true = data["targets"].astype(np.float64)
    y_pred_mean = data["predictions"].astype(np.float64)
    y_pred_std = data["uncertainties"].astype(np.float64)

    n = len(y_true)
    print(f"  n_nodes: {n}")
    print(
        f"  y_pred_std: min={y_pred_std.min():.6f}, max={y_pred_std.max():.6f}, mean={y_pred_std.mean():.6f}"
    )

    nll_mean, nll_std, nll_median = gaussian_nll(y_true, y_pred_mean, y_pred_std)
    print(f"\n  T8 Gaussian NLL (mean): {nll_mean:.4f}")
    print(f"  T8 Gaussian NLL (std):  {nll_std:.4f}")
    print(f"  T8 Gaussian NLL (median): {nll_median:.4f}")

    results["t8_mc_dropout"] = {
        "nll_mean": round(nll_mean, 4),
        "nll_std": round(nll_std, 4),
        "nll_median": round(nll_median, 4),
        "n_nodes": int(n),
        "source": str(T8_NPZ.name),
    }

    # --- T8 with temperature-scaled sigma ---
    T_opt = 2.7025
    nll_mean_ts, nll_std_ts, nll_median_ts = gaussian_nll(
        y_true, y_pred_mean, y_pred_std * T_opt
    )
    print(f"\n  T8 NLL (temp-scaled, T={T_opt}): {nll_mean_ts:.4f}")
    results["t8_mc_dropout_temp_scaled"] = {
        "nll_mean": round(nll_mean_ts, 4),
        "nll_std": round(nll_std_ts, 4),
        "nll_median": round(nll_median_ts, 4),
        "temperature": T_opt,
        "n_nodes": int(n),
    }

    # --- T7 MC Dropout (if available) ---
    t7_npz_path = None
    for candidate in [
        REPO
        / "data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz",
        REPO
        / "data/TR-C_Benchmarks/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_metrics_model7_mc30_100graphs.npz",
    ]:
        if candidate.exists():
            t7_npz_path = candidate
            break

    if t7_npz_path:
        print(f"\nLoading T7 MC Dropout NPZ from {t7_npz_path.name}...")
        data7 = np.load(t7_npz_path)
        print(f"  T7 keys: {list(data7.keys())}")
        # Handle both key naming conventions
        if "targets" in data7:
            y_true7 = data7["targets"].astype(np.float64)
            y_pred_mean7 = data7["predictions"].astype(np.float64)
            y_pred_std7 = data7["uncertainties"].astype(np.float64)
        else:
            y_true7 = data7["y_true"].astype(np.float64)
            y_pred_mean7 = data7["y_pred_mean"].astype(np.float64)
            y_pred_std7 = data7["y_pred_std"].astype(np.float64)
        nll7_mean, nll7_std, nll7_median = gaussian_nll(
            y_true7, y_pred_mean7, y_pred_std7
        )
        print(f"  T7 Gaussian NLL (mean): {nll7_mean:.4f}")
        results["t7_mc_dropout"] = {
            "nll_mean": round(nll7_mean, 4),
            "nll_std": round(nll7_std, 4),
            "nll_median": round(nll7_median, 4),
            "n_nodes": int(len(y_true7)),
            "source": str(t7_npz_path.name),
        }
    else:
        print("\nT7 NPZ not found, skipping.")

    # --- Save results ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
