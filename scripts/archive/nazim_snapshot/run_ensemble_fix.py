"""
Wrapper to run ensemble experiments with fixed weight loading.
Usage: conda run -n thesis-env python scripts/run_ensemble_fix.py
"""

import os
import sys

# Suppress tqdm unicode issues
os.environ["PYTHONIOENCODING"] = "utf-8"

# Disable tqdm to avoid encoding issues
os.environ["TQDM_DISABLE"] = "1"

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch
import numpy as np
import json
from scipy.stats import spearmanr

from gnn.models.point_net_transf_gat import PointNetTransfGAT

# ---- Config ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "TR-C_Benchmarks")

MODEL_FOLDERS = {
    2: "point_net_transf_gat_2nd_try",
    5: "point_net_transf_gat_5th_try",
    6: "point_net_transf_gat_6th_trial_lower_lr",
    7: "point_net_transf_gat_7th_trial_80_10_10_split",
    8: "point_net_transf_gat_8th_trial_lower_dropout",
}

DROPOUT_MAP = {2: 0.3, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.2}

MODEL_WEIGHTS_R2 = {2: 0.5117, 5: 0.5553, 6: 0.5223, 7: 0.5471, 8: 0.5957}


def load_model_fixed(model_num):
    """Load model with GATConv weight remapping fix."""
    folder = os.path.join(DATA_ROOT, MODEL_FOLDERS[model_num])
    model_path = os.path.join(folder, "trained_model", "model.pth")

    model = PointNetTransfGAT(
        in_channels=5,
        out_channels=1,
        point_net_conv_layer_structure_local_mlp=[256],
        point_net_conv_layer_structure_global_mlp=[512],
        gat_conv_layer_structure=[128, 256, 512],
        dropout=DROPOUT_MAP[model_num],
        use_dropout=True,
        predict_mode_stats=False,
    )

    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Remap old PyG GATConv keys
    remapped = {}
    for k, v in state_dict.items():
        if ".lin.weight" in k:
            remapped[k.replace(".lin.weight", ".lin_src.weight")] = v
            remapped[k.replace(".lin.weight", ".lin_dst.weight")] = v
        else:
            remapped[k] = v

    model.load_state_dict(remapped, strict=True)
    model = model.to(DEVICE)
    return model


def mc_dropout_predict(model, data, S=30):
    """MC Dropout inference: S forward passes with dropout ON."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    preds = []
    with torch.no_grad():
        for _ in range(S):
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze().cpu().numpy())

    preds = np.array(preds)  # (S, n_nodes)
    return preds.mean(axis=0), preds.std(axis=0)


def run_experiment_a(n_graphs=10, n_runs=5, S=30):
    """Experiment A: MC Dropout vs Ensemble Variance on T8."""
    print("=" * 60)
    print("EXPERIMENT A: MC Dropout vs Ensemble Variance (T8)")
    print(f"  Device: {DEVICE}, Graphs: {n_graphs}, Runs: {n_runs}, S: {S}")
    print("=" * 60)

    model = load_model_fixed(8)
    print("  T8 model loaded with weight remapping fix.")

    # Load test data
    t8_folder = os.path.join(DATA_ROOT, MODEL_FOLDERS[8])
    test_dl = torch.load(
        os.path.join(t8_folder, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    n_graphs = min(n_graphs, len(test_dl))
    print(f"  Using {n_graphs} test graphs")

    all_run_preds = []
    all_run_uncs = []
    targets = None

    for run in range(n_runs):
        torch.manual_seed(42 + run * 100)
        np.random.seed(42 + run * 100)
        run_preds, run_uncs, run_targets = [], [], []

        for gi in range(n_graphs):
            data = test_dl[gi].to(DEVICE)
            mean_pred, unc = mc_dropout_predict(model, data, S)
            run_preds.append(mean_pred)
            run_uncs.append(unc)
            if run == 0:
                run_targets.append(data.y.squeeze().cpu().numpy())

            if (gi + 1) % 5 == 0 or gi == n_graphs - 1:
                print(f"    Run {run + 1}/{n_runs}, graph {gi + 1}/{n_graphs}")

        all_run_preds.append(np.concatenate(run_preds))
        all_run_uncs.append(np.concatenate(run_uncs))
        if run == 0:
            targets = np.concatenate(run_targets)

    ensemble_preds = np.array(all_run_preds)  # (n_runs, total_nodes)
    mc_uncs = np.array(all_run_uncs)  # (n_runs, total_nodes)

    # Compute uncertainties
    avg_mc_unc = mc_uncs.mean(axis=0)
    ens_variance = ensemble_preds.std(axis=0)
    combined_unc = np.sqrt(avg_mc_unc**2 + ens_variance**2)
    ens_mean_pred = ensemble_preds.mean(axis=0)

    abs_err = np.abs(ens_mean_pred - targets)
    ss_res = np.sum((targets - ens_mean_pred) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    rho_mc = spearmanr(avg_mc_unc, abs_err)[0]
    rho_ens = spearmanr(ens_variance, abs_err)[0]
    rho_comb = spearmanr(combined_unc, abs_err)[0]

    results = {
        "config": {
            "model": 8,
            "n_runs": n_runs,
            "S": S,
            "n_graphs": n_graphs,
            "n_nodes": int(len(targets)),
            "weight_remapping": True,
        },
        "prediction": {
            "r2": float(r2),
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean((ens_mean_pred - targets) ** 2))),
        },
        "mc_dropout": {
            "spearman_rho": float(rho_mc),
            "unc_mean": float(avg_mc_unc.mean()),
            "unc_std": float(avg_mc_unc.std()),
        },
        "ensemble_variance": {
            "spearman_rho": float(rho_ens),
            "unc_mean": float(ens_variance.mean()),
            "unc_std": float(ens_variance.std()),
        },
        "combined": {
            "spearman_rho": float(rho_comb),
            "unc_mean": float(combined_unc.mean()),
            "unc_std": float(combined_unc.std()),
        },
    }

    print()
    print("  EXPERIMENT A RESULTS (FIXED):")
    print(f"    R2 = {r2:.4f}, MAE = {np.mean(abs_err):.4f}")
    print(f"    MC Dropout rho   = {rho_mc:.4f} (mean sigma = {avg_mc_unc.mean():.4f})")
    print(
        f"    Ensemble Var rho = {rho_ens:.4f} (mean sigma = {ens_variance.mean():.4f})"
    )
    print(f"    Combined rho     = {rho_comb:.4f}")
    print()

    return results


def run_experiment_b(n_graphs=10):
    """Experiment B: Multi-model ensemble (T2, T5-T8)."""
    print("=" * 60)
    print("EXPERIMENT B: Multi-Model Ensemble (T2, T5, T6, T7, T8)")
    print(f"  Device: {DEVICE}, Graphs: {n_graphs}")
    print("=" * 60)

    model_nums = [2, 5, 6, 7, 8]
    models = {}
    for m in model_nums:
        print(f"  Loading model T{m}...")
        models[m] = load_model_fixed(m)
        models[m].eval()
    print("  All models loaded with weight remapping fix.")

    # Load T8 test data
    t8_folder = os.path.join(DATA_ROOT, MODEL_FOLDERS[8])
    test_dl = torch.load(
        os.path.join(t8_folder, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    n_graphs = min(n_graphs, len(test_dl))
    print(f"  Using {n_graphs} test graphs")

    all_preds = {m: [] for m in model_nums}
    all_targets = []

    for gi in range(n_graphs):
        data = test_dl[gi].to(DEVICE)
        all_targets.append(data.y.squeeze().cpu().numpy())
        with torch.no_grad():
            for m in model_nums:
                out = models[m](data)
                if isinstance(out, tuple):
                    out = out[0]
                all_preds[m].append(out.squeeze().cpu().numpy())
        if (gi + 1) % 5 == 0 or gi == n_graphs - 1:
            print(f"    Graph {gi + 1}/{n_graphs}")

    targets = np.concatenate(all_targets)
    model_preds = {m: np.concatenate(all_preds[m]) for m in model_nums}
    pred_stack = np.stack([model_preds[m] for m in model_nums], axis=0)

    # Weighted average
    weights = np.array([MODEL_WEIGHTS_R2[m] for m in model_nums])
    weights = weights / weights.sum()
    weighted_pred = np.average(pred_stack, axis=0, weights=weights)
    weighted_var = np.average(
        (pred_stack - weighted_pred) ** 2, axis=0, weights=weights
    )
    ens_unc = np.sqrt(weighted_var)

    abs_err = np.abs(weighted_pred - targets)
    ss_res = np.sum((targets - weighted_pred) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rho = spearmanr(ens_unc, abs_err)[0]

    # Individual model metrics
    indiv = {}
    for m in model_nums:
        p = model_preds[m]
        ss_r = np.sum((targets - p) ** 2)
        indiv[m] = {
            "r2": float(1 - ss_r / ss_tot),
            "mae": float(np.mean(np.abs(p - targets))),
        }

    results = {
        "config": {
            "models": model_nums,
            "n_graphs": n_graphs,
            "n_nodes": int(len(targets)),
            "weight_remapping": True,
            "weighted": True,
        },
        "individual": {str(m): indiv[m] for m in model_nums},
        "ensemble": {
            "r2": float(r2),
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean((weighted_pred - targets) ** 2))),
            "spearman_rho": float(rho),
            "unc_mean": float(ens_unc.mean()),
            "unc_std": float(ens_unc.std()),
        },
    }

    print()
    print("  EXPERIMENT B RESULTS (FIXED):")
    for m in model_nums:
        print(f"    T{m}: R2={indiv[m]['r2']:.4f}, MAE={indiv[m]['mae']:.4f}")
    print(f"    Ensemble: R2={r2:.4f}, MAE={np.mean(abs_err):.4f}, rho={rho:.4f}")
    print()

    return results


if __name__ == "__main__":
    import time

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    t0 = time.time()

    # Run both experiments with 10 graphs first (verify fix works)
    n_graphs = 10

    res_a = run_experiment_a(n_graphs=n_graphs, n_runs=5, S=30)
    res_b = run_experiment_b(n_graphs=n_graphs)

    # Save combined results
    out_dir = os.path.join(
        DATA_ROOT, MODEL_FOLDERS[8], "uq_results", "ensemble_experiments"
    )
    os.makedirs(out_dir, exist_ok=True)

    combined = {"experiment_a": res_a, "experiment_b": res_b}
    out_path = os.path.join(out_dir, "ensemble_fixed_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")
    print("DONE")
