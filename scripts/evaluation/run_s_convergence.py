"""
Phase 4: S-Convergence Analysis for MC Dropout.
Re-runs T8 model on test graphs with S=50 forward passes,
saves raw (S, N) predictions per graph, then subsamples at
S=5,10,15,20,25,30,35,40,45,50 to compute Spearman rho and mean sigma.

Uses the weight-remapped model to fix the PyG GATConv API mismatch.
"""

import sys, os, time, json
import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from gnn.models.point_net_transf_gat import PointNetTransfGAT

# Configuration
NUM_GRAPHS = 10  # 10 graphs ~ 316K nodes, ~30-60 min on CPU
MAX_S = 50  # Run 50 forward passes per node
S_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SEED = 42

device = torch.device("cpu")
T8_FOLDER = os.path.join(
    REPO_ROOT, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "verified", "phase3_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 4: S-CONVERGENCE ANALYSIS")
print(f"Graphs: {NUM_GRAPHS}, Max S: {MAX_S}")
print(f"S values to evaluate: {S_VALUES}")
print("=" * 70)

# --- 1. Load model with weight remapping ---
print("\n[1] Loading T8 model with weight remapping...")
model_path = os.path.join(T8_FOLDER, "trained_model", "model.pth")
state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

remapped = {}
for k, v in state_dict.items():
    if ".lin.weight" in k:
        remapped[k.replace(".lin.weight", ".lin_src.weight")] = v
        remapped[k.replace(".lin.weight", ".lin_dst.weight")] = v
    else:
        remapped[k] = v

model = PointNetTransfGAT(
    in_channels=5,
    out_channels=1,
    point_net_conv_layer_structure_local_mlp=[256],
    point_net_conv_layer_structure_global_mlp=[512],
    gat_conv_layer_structure=[128, 256, 512],
    dropout=0.3,
    use_dropout=True,
    predict_mode_stats=False,
)
model.load_state_dict(remapped, strict=True)
model = model.to(device)
print("  Model loaded and weights remapped.")

# --- 2. Load test data ---
print("\n[2] Loading test data...")
test_dl_path = os.path.join(T8_FOLDER, "data_created_during_training", "test_dl.pt")
test_set_dl = torch.load(test_dl_path, weights_only=False)
n_graphs = min(len(test_set_dl), NUM_GRAPHS)
print(f"  Using {n_graphs} of {len(test_set_dl)} test graphs")

# --- 3. Run MC Dropout with S=MAX_S, saving all raw passes ---
print(f"\n[3] Running MC Dropout inference (S={MAX_S}) on {n_graphs} graphs...")

# Set model to train mode (dropout ON), freeze BatchNorm
model.train()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()

all_raw_preds = []  # list of arrays, each shape (MAX_S, n_nodes_i)
all_targets = []  # list of arrays, each shape (n_nodes_i,)

start_time = time.time()

for g_idx in range(n_graphs):
    td = test_set_dl[g_idx].to(device)
    n_nodes = td.y.shape[0]

    # Set seed per graph for reproducibility
    torch.manual_seed(SEED + g_idx)

    raw_preds_graph = np.zeros((MAX_S, n_nodes), dtype=np.float32)

    with torch.no_grad():
        for s in range(MAX_S):
            pred = model(td)
            if isinstance(pred, tuple):
                pred = pred[0]
            raw_preds_graph[s] = pred.squeeze().cpu().numpy()

    all_raw_preds.append(raw_preds_graph)
    all_targets.append(td.y.squeeze().cpu().numpy())

    elapsed = time.time() - start_time
    avg_per_graph = elapsed / (g_idx + 1)
    eta = avg_per_graph * (n_graphs - g_idx - 1)
    print(
        f"  Graph {g_idx + 1}/{n_graphs} done ({n_nodes} nodes, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
    )

total_time = time.time() - start_time
print(f"\n  Total inference time: {total_time / 60:.1f} min")

# Concatenate all graphs
targets_all = np.concatenate(all_targets)  # (total_nodes,)
total_nodes = targets_all.shape[0]
print(f"  Total nodes: {total_nodes:,}")

# Build concatenated raw predictions: (MAX_S, total_nodes)
raw_preds_all = np.concatenate(all_raw_preds, axis=1)  # (MAX_S, total_nodes)
print(f"  Raw predictions shape: {raw_preds_all.shape}")

# --- 4. Compute S-convergence metrics ---
print(f"\n[4] Computing S-convergence metrics for S = {S_VALUES}")

convergence_results = []

for s_val in S_VALUES:
    # Use first s_val passes
    preds_subset = raw_preds_all[:s_val, :]  # (s_val, total_nodes)

    mean_pred = preds_subset.mean(axis=0)  # (total_nodes,)
    std_pred = preds_subset.std(axis=0, ddof=0)  # (total_nodes,) — unbiased=False

    abs_errors = np.abs(mean_pred - targets_all)
    rho, pval = spearmanr(std_pred, abs_errors)

    mae = np.mean(abs_errors)
    mean_sigma = np.mean(std_pred)

    # R2
    ss_res = np.sum((targets_all - mean_pred) ** 2)
    ss_tot = np.sum((targets_all - targets_all.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    result = {
        "S": s_val,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "mean_sigma": float(mean_sigma),
        "mae": float(mae),
        "r2": float(r2),
    }
    convergence_results.append(result)

    print(
        f"  S={s_val:3d}: rho={rho:.4f}, mean_sigma={mean_sigma:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
    )

# --- 5. Also compute per-graph convergence ---
print(f"\n[5] Computing per-graph convergence...")

per_graph_results = []

offset = 0
for g_idx in range(n_graphs):
    n = all_targets[g_idx].shape[0]
    graph_raw = all_raw_preds[g_idx]  # (MAX_S, n)
    graph_targets = all_targets[g_idx]  # (n,)

    graph_conv = []
    for s_val in S_VALUES:
        subset = graph_raw[:s_val, :]
        mp = subset.mean(axis=0)
        sp = subset.std(axis=0, ddof=0)
        ae = np.abs(mp - graph_targets)
        r, _ = spearmanr(sp, ae)
        graph_conv.append({"S": s_val, "rho": float(r), "mean_sigma": float(sp.mean())})

    per_graph_results.append(graph_conv)
    offset += n

# Compute mean and std of rho across graphs for each S
for s_idx, s_val in enumerate(S_VALUES):
    rhos = [pg[s_idx]["rho"] for pg in per_graph_results]
    print(
        f"  S={s_val:3d}: mean per-graph rho={np.mean(rhos):.4f} +/- {np.std(rhos):.4f}"
    )

# --- 6. Save results ---
print(f"\n[6] Saving results...")

output = {
    "config": {
        "model": "T8 (weight-remapped)",
        "n_graphs": n_graphs,
        "total_nodes": int(total_nodes),
        "max_S": MAX_S,
        "S_values": S_VALUES,
        "seed": SEED,
        "inference_time_min": round(total_time / 60, 1),
    },
    "aggregate_convergence": convergence_results,
    "per_graph_mean_rho": {
        str(s_val): {
            "mean": float(np.mean([pg[s_idx]["rho"] for pg in per_graph_results])),
            "std": float(np.std([pg[s_idx]["rho"] for pg in per_graph_results])),
        }
        for s_idx, s_val in enumerate(S_VALUES)
    },
}

out_path = os.path.join(OUTPUT_DIR, "s_convergence_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  Results saved to: {out_path}")

# Also save raw data for figure generation
raw_path = os.path.join(OUTPUT_DIR, "s_convergence_raw.npz")
np.savez_compressed(
    raw_path,
    raw_preds=raw_preds_all,
    targets=targets_all,
    s_values=np.array(S_VALUES),
)
print(f"  Raw data saved to: {raw_path}")

print("\n" + "=" * 70)
print("S-CONVERGENCE ANALYSIS COMPLETE")
print("=" * 70)
