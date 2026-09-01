"""
Phase 2 (final): Aggregate all 5 Experiment A runs into final results.
Loads exp_a_run_0.npz through exp_a_run_4.npz and computes ensemble statistics.
Usage: conda run -n thesis-env python scripts/aggregate_exp_a.py
"""

import os, sys, time, json

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(REPO_ROOT, "data", "TR-C_Benchmarks")
OUT_DIR = os.path.join(
    DATA_ROOT,
    "point_net_transf_gat_8th_trial_lower_dropout",
    "uq_results",
    "ensemble_experiments",
)

N_RUNS = 5
S = 30


if __name__ == "__main__":
    print("=" * 60)
    print("AGGREGATE EXPERIMENT A: Combining %d MC Dropout runs" % N_RUNS)
    print("=" * 60)
    t0 = time.time()

    # Load all runs
    all_preds = []
    all_uncs = []
    targets = None

    for r in range(N_RUNS):
        path = os.path.join(OUT_DIR, "exp_a_run_%d.npz" % r)
        if not os.path.exists(path):
            print("ERROR: Missing %s -- run index %d not completed yet!" % (path, r))
            sys.exit(1)
        data = np.load(path)
        all_preds.append(data["predictions"])
        all_uncs.append(data["uncertainties"])
        if r == 0:
            targets = data["targets"]
        n_nodes = len(data["predictions"])
        print("  Run %d: %d nodes loaded" % (r, n_nodes))

    # Stack into arrays
    ensemble_preds = np.array(all_preds)  # (5, n_nodes)
    mc_uncs = np.array(all_uncs)  # (5, n_nodes)
    n_nodes = len(targets)

    print("\nTotal: %d runs x %d nodes" % (N_RUNS, n_nodes))

    # Compute three uncertainty types
    avg_mc_unc = mc_uncs.mean(axis=0)  # Average of per-run MC stds
    ens_variance = ensemble_preds.std(axis=0)  # Std of per-run MC means
    combined_unc = np.sqrt(avg_mc_unc**2 + ens_variance**2)  # Combined
    ens_mean_pred = ensemble_preds.mean(axis=0)  # Mean prediction across runs

    # Compute errors
    abs_err = np.abs(ens_mean_pred - targets)
    ss_res = np.sum((targets - ens_mean_pred) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((ens_mean_pred - targets) ** 2)))

    # Spearman correlations
    rho_mc = float(spearmanr(avg_mc_unc, abs_err)[0])
    rho_ens = float(spearmanr(ens_variance, abs_err)[0])
    rho_comb = float(spearmanr(combined_unc, abs_err)[0])

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT A FINAL RESULTS (FIXED)")
    print("=" * 60)
    print("  Model: T8, Runs: %d, S: %d" % (N_RUNS, S))
    print("  Graphs: %d, Nodes: %d" % (n_nodes // 31635, n_nodes))
    print("  R2 = %.4f, MAE = %.4f, RMSE = %.4f" % (r2, mae, rmse))
    print("  ---")
    print(
        "  MC Dropout rho     = %.4f  (mean sigma = %.4f)" % (rho_mc, avg_mc_unc.mean())
    )
    print(
        "  Ensemble Var rho   = %.4f  (mean sigma = %.4f)"
        % (rho_ens, ens_variance.mean())
    )
    print(
        "  Combined rho       = %.4f  (mean sigma = %.4f)"
        % (rho_comb, combined_unc.mean())
    )
    print()

    # Also compute per-run standalone R2 (sanity check)
    print("  Per-run R2 (sanity check -- should all be ~0.58):")
    for r in range(N_RUNS):
        ss_r = np.sum((targets - all_preds[r]) ** 2)
        r2_r = 1 - ss_r / ss_tot
        print("    Run %d (seed=%d): R2=%.4f" % (r, 42 + r * 100, r2_r))

    # Save results JSON
    results = {
        "config": {
            "model": 8,
            "n_runs": N_RUNS,
            "S": S,
            "n_graphs": n_nodes // 31635,
            "n_nodes": n_nodes,
            "seeds": [42 + r * 100 for r in range(N_RUNS)],
            "weight_remapping": True,
            "strict_loading": True,
        },
        "prediction": {
            "r2": float(r2),
            "mae": mae,
            "rmse": rmse,
        },
        "mc_dropout": {
            "spearman_rho": rho_mc,
            "unc_mean": float(avg_mc_unc.mean()),
            "unc_std": float(avg_mc_unc.std()),
            "unc_min": float(avg_mc_unc.min()),
            "unc_max": float(avg_mc_unc.max()),
        },
        "ensemble_variance": {
            "spearman_rho": rho_ens,
            "unc_mean": float(ens_variance.mean()),
            "unc_std": float(ens_variance.std()),
            "unc_min": float(ens_variance.min()),
            "unc_max": float(ens_variance.max()),
        },
        "combined": {
            "spearman_rho": rho_comb,
            "unc_mean": float(combined_unc.mean()),
            "unc_std": float(combined_unc.std()),
            "unc_min": float(combined_unc.min()),
            "unc_max": float(combined_unc.max()),
        },
    }

    json_path = os.path.join(OUT_DIR, "experiment_a_fixed_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved: %s" % json_path)

    # Save NPZ artifacts
    npz_path = os.path.join(OUT_DIR, "experiment_a_fixed_data.npz")
    np.savez_compressed(
        npz_path,
        targets=targets.astype(np.float32),
        ensemble_mean_prediction=ens_mean_pred.astype(np.float32),
        avg_mc_uncertainty=avg_mc_unc.astype(np.float32),
        ensemble_variance=ens_variance.astype(np.float32),
        combined_uncertainty=combined_unc.astype(np.float32),
        # Also save per-run data for full reproducibility
        run_predictions=ensemble_preds.astype(np.float32),  # (5, n_nodes)
        run_uncertainties=mc_uncs.astype(np.float32),  # (5, n_nodes)
    )
    print("Artifacts saved: %s" % npz_path)

    elapsed = time.time() - t0
    print("\nTotal time: %.1fs" % elapsed)
    print("DONE")
