"""
Phase 1: Experiment B -- Multi-Model Ensemble (T2, T5, T6, T7, T8)
100 test graphs, deterministic forward pass, weighted ensemble.
Usage: conda run -n thesis-env python scripts/run_exp_b_fixed.py
"""

import os, sys, time, json

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TQDM_DISABLE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch
import numpy as np
from scipy.stats import spearmanr
from gnn.models.point_net_transf_gat import PointNetTransfGAT
import gc

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
MODEL_NUMS = [2, 5, 6, 7, 8]

OUT_DIR = os.path.join(
    DATA_ROOT,
    "point_net_transf_gat_8th_trial_lower_dropout",
    "uq_results",
    "ensemble_experiments",
)


def load_model_fixed(model_num):
    """Load model with GATConv weight remapping fix + strict=True."""
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


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT B: Multi-Model Ensemble (T2, T5, T6, T7, T8)")
    print("  100 test graphs, deterministic, weighted ensemble")
    print("  Device:", DEVICE)
    print("=" * 60)
    t0 = time.time()

    # Load test data (T8's test set -- used as reference for all models)
    print("\nLoading test data...")
    t8_folder = os.path.join(DATA_ROOT, MODEL_FOLDERS[8])
    test_dl = torch.load(
        os.path.join(t8_folder, "data_created_during_training", "test_dl.pt"),
        weights_only=False,
    )
    n_graphs = len(test_dl)
    print("  %d test graphs loaded (%.1fs)" % (n_graphs, time.time() - t0))

    # Collect targets
    print("\nCollecting targets...")
    all_targets = []
    for gi in range(n_graphs):
        all_targets.append(test_dl[gi].y.squeeze().cpu().numpy())
    targets = np.concatenate(all_targets)
    n_nodes = len(targets)
    print("  %d total nodes across %d graphs" % (n_nodes, n_graphs))

    # Process each model one at a time (memory efficient)
    all_preds = {}
    for m_num in MODEL_NUMS:
        t_m = time.time()
        print("\nLoading and running T%d..." % m_num)
        model = load_model_fixed(m_num)
        model.eval()
        preds_m = []
        with torch.no_grad():
            for gi in range(n_graphs):
                data = test_dl[gi].to(DEVICE)
                out = model(data)
                if isinstance(out, tuple):
                    out = out[0]
                preds_m.append(out.squeeze().cpu().numpy())
                if (gi + 1) % 25 == 0:
                    print("    Graph %d/%d" % (gi + 1, n_graphs))
        all_preds[m_num] = np.concatenate(preds_m)

        # Quick R2 check
        ss_res = np.sum((targets - all_preds[m_num]) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2_m = 1 - ss_res / ss_tot
        mae_m = np.mean(np.abs(all_preds[m_num] - targets))
        print(
            "  T%d: R2=%.4f, MAE=%.4f (%.1fs)" % (m_num, r2_m, mae_m, time.time() - t_m)
        )

        # Sanity check: R2 should be > 0.4 (not ~0 like buggy version)
        if r2_m < 0.3:
            print("  !! WARNING: R2 too low! Fix may not be working for T%d !!" % m_num)

        del model
        gc.collect()

    # Compute weighted ensemble
    print("\nComputing weighted ensemble...")
    pred_stack = np.stack([all_preds[m] for m in MODEL_NUMS], axis=0)  # (5, n_nodes)
    weights = np.array([MODEL_WEIGHTS_R2[m] for m in MODEL_NUMS])
    weights = weights / weights.sum()

    weighted_pred = np.average(pred_stack, axis=0, weights=weights)
    weighted_var = np.average(
        (pred_stack - weighted_pred) ** 2, axis=0, weights=weights
    )
    ens_unc = np.sqrt(weighted_var)

    # Compute metrics
    abs_err = np.abs(weighted_pred - targets)
    ss_res = np.sum((targets - weighted_pred) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2_ens = 1 - ss_res / ss_tot
    mae_ens = float(np.mean(abs_err))
    rmse_ens = float(np.sqrt(np.mean((weighted_pred - targets) ** 2)))
    rho_ens = float(spearmanr(ens_unc, abs_err)[0])

    # Individual model metrics
    indiv = {}
    for m in MODEL_NUMS:
        p = all_preds[m]
        ss_r = np.sum((targets - p) ** 2)
        indiv[m] = {
            "r2": float(1 - ss_r / ss_tot),
            "mae": float(np.mean(np.abs(p - targets))),
            "rmse": float(np.sqrt(np.mean((p - targets) ** 2))),
        }

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT B RESULTS (FIXED)")
    print("=" * 60)
    for m in MODEL_NUMS:
        print(
            "  T%d: R2=%.4f, MAE=%.4f, RMSE=%.4f"
            % (m, indiv[m]["r2"], indiv[m]["mae"], indiv[m]["rmse"])
        )
    print("  ---")
    print("  Ensemble: R2=%.4f, MAE=%.4f, RMSE=%.4f" % (r2_ens, mae_ens, rmse_ens))
    print("  Ensemble Spearman rho=%.4f" % rho_ens)
    print("  Ensemble unc mean=%.4f, std=%.4f" % (ens_unc.mean(), ens_unc.std()))
    print("  Nodes: %d, Graphs: %d" % (n_nodes, n_graphs))

    # Save results JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {
        "config": {
            "models": MODEL_NUMS,
            "n_graphs": n_graphs,
            "n_nodes": n_nodes,
            "weight_remapping": True,
            "strict_loading": True,
            "weighted": True,
            "weights_r2": MODEL_WEIGHTS_R2,
        },
        "individual": {str(m): indiv[m] for m in MODEL_NUMS},
        "ensemble": {
            "r2": float(r2_ens),
            "mae": mae_ens,
            "rmse": rmse_ens,
            "spearman_rho": rho_ens,
            "unc_mean": float(ens_unc.mean()),
            "unc_std": float(ens_unc.std()),
            "unc_min": float(ens_unc.min()),
            "unc_max": float(ens_unc.max()),
        },
    }

    json_path = os.path.join(OUT_DIR, "experiment_b_fixed_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved: %s" % json_path)

    # Save NPZ artifacts
    npz_path = os.path.join(OUT_DIR, "experiment_b_fixed_data.npz")
    save_dict = {
        "targets": targets.astype(np.float32),
        "ensemble_prediction": weighted_pred.astype(np.float32),
        "ensemble_uncertainty": ens_unc.astype(np.float32),
    }
    for m in MODEL_NUMS:
        save_dict["model_%d_predictions" % m] = all_preds[m].astype(np.float32)
    np.savez_compressed(npz_path, **save_dict)
    print("Artifacts saved: %s" % npz_path)

    elapsed = time.time() - t0
    print("\nTotal time: %.1fs (%.1f min)" % (elapsed, elapsed / 60))
    print("DONE")
