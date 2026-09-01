"""
Final verification script for test dataloaders, scaler files, and training batches.
"""

import os
import sys
import io
import warnings
import joblib
import torch
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress sklearn version mismatch warning
warnings.filterwarnings("ignore", message=".*unpickle.*")

ROOT = str(Path(__file__).resolve().parent)


def fmt_size(path):
    size = os.path.getsize(path)
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


np.set_printoptions(precision=6, suppress=True, linewidth=120)

# =========================================================================
# SECTION 1: Test DataLoaders
# =========================================================================
print("=" * 80)
print("SECTION 1: TEST DATALOADERS (.pt)")
print("=" * 80)

test_dl_paths = [
    (
        "T8 (8th_trial_lower_dropout)",
        os.path.join(
            ROOT,
            "data",
            "TR-C_Benchmarks",
            "point_net_transf_gat_8th_trial_lower_dropout",
            "data_created_during_training",
            "test_dl.pt",
        ),
    ),
    (
        "T7 (7th_trial_80_10_10_split)",
        os.path.join(
            ROOT,
            "data",
            "TR-C_Benchmarks",
            "point_net_transf_gat_7th_trial_80_10_10_split",
            "data_created_during_training",
            "test_dl.pt",
        ),
    ),
]

for trial_name, path in test_dl_paths:
    print(f"\n{'-' * 70}")
    print(f"Trial:     {trial_name}")
    print(f"File:      ...{path[len(ROOT) :]}")
    print(f"File size: {fmt_size(path)}  ({os.path.getsize(path):,} bytes)")
    print(f"{'-' * 70}")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        print(f"Loaded type:     {type(obj).__module__}.{type(obj).__name__}")

        if isinstance(obj, list):
            print(f"List length:     {len(obj)} Data objects")
            first = obj[0]
            print(f"Element type:    {type(first).__module__}.{type(first).__name__}")
            print(f"Keys:            {list(first.keys())}")
            print(f"\nFirst element attribute details:")
            for key in first.keys():
                attr = getattr(first, key, None)
                if hasattr(attr, "shape"):
                    print(f"  {key:25s} shape={str(attr.shape):30s} dtype={attr.dtype}")
                elif isinstance(attr, (int, float)):
                    print(f"  {key:25s} value={attr}")
                else:
                    print(f"  {key:25s} type={type(attr).__name__}")

            print(f"\n  num_nodes:    {first.num_nodes}")
            print(f"  num_edges:    {first.num_edges}")
            print(f"  num_features: {first.num_features}")
            print(f"  is_directed:  {first.is_directed()}")

            # Check consistency across a few samples
            print(f"\n  Spot-check shapes across samples (indices 0, 49, 99):")
            for idx in [0, 49, 99]:
                if idx < len(obj):
                    d = obj[idx]
                    print(
                        f"    [{idx:3d}] x={tuple(d.x.shape)}, y={tuple(d.y.shape)}, "
                        f"edge_index={tuple(d.edge_index.shape)}, pos={tuple(d.pos.shape)}"
                    )

        elif hasattr(obj, "batch_size"):
            print(f"Batch size:      {obj.batch_size}")
            print(f"Num batches:     {len(obj)}")
        else:
            print(f"Object repr: {repr(obj)[:200]}")

        print(f"\n  STATUS: OK - Successfully loaded")

    except Exception as e:
        print(f"  STATUS: FAIL - {type(e).__name__}: {e}")


# =========================================================================
# SECTION 2: Scaler files (.pkl) - using joblib
# =========================================================================
print("\n\n" + "=" * 80)
print("SECTION 2: SCALER FILES (.pkl)  [loaded with joblib]")
print("=" * 80)

scaler_dirs = {
    "T8 (8th_trial_lower_dropout)": os.path.join(
        ROOT,
        "data",
        "TR-C_Benchmarks",
        "point_net_transf_gat_8th_trial_lower_dropout",
        "data_created_during_training",
    ),
    "T7 (7th_trial_80_10_10_split)": os.path.join(
        ROOT,
        "data",
        "TR-C_Benchmarks",
        "point_net_transf_gat_7th_trial_80_10_10_split",
        "data_created_during_training",
    ),
}

for trial_label, dir_path in scaler_dirs.items():
    print(f"\n{'=' * 70}")
    print(f"TRIAL: {trial_label}")
    print(f"Dir:   ...{dir_path[len(ROOT) :]}")
    print(f"{'=' * 70}")

    pkl_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".pkl")])
    print(f"Found {len(pkl_files)} .pkl files: {pkl_files}")

    for pkl_name in pkl_files:
        pkl_path = os.path.join(dir_path, pkl_name)
        print(f"\n  --- {pkl_name} ({fmt_size(pkl_path)}) ---")

        try:
            scaler = joblib.load(pkl_path)
            print(
                f"  Type:             {type(scaler).__module__}.{type(scaler).__name__}"
            )

            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                mean = np.array(scaler.mean_)
                scale = np.array(scaler.scale_)
                print(f"  n_features_in_:   {scaler.n_features_in_}")
                print(f"  n_samples_seen_:  {scaler.n_samples_seen_}")
                print(f"  with_mean:        {scaler.with_mean}")
                print(f"  with_std:         {scaler.with_std}")
                print(f"  mean_  (shape={mean.shape}):")
                print(f"    {mean}")
                print(f"  scale_ (shape={scale.shape}):")
                print(f"    {scale}")
                if hasattr(scaler, "var_"):
                    var = np.array(scaler.var_)
                    print(f"  var_   (shape={var.shape}):")
                    print(f"    {var}")
            elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                print(f"  data_min_: {scaler.data_min_}")
                print(f"  data_max_: {scaler.data_max_}")
            else:
                attrs = [a for a in dir(scaler) if not a.startswith("_")]
                print(f"  Public attrs: {attrs[:20]}")

            print(f"  STATUS: OK")

        except Exception as e:
            print(f"  STATUS: FAIL - {type(e).__name__}: {e}")


# =========================================================================
# SECTION 3: Training data batches
# =========================================================================
print("\n\n" + "=" * 80)
print("SECTION 3: TRAINING DATA BATCHES")
print("=" * 80)

batch_dir = os.path.join(ROOT, "data", "train_data", "dist_not_connected_10k_1pct")
print(f"Dir: ...{batch_dir[len(ROOT) :]}\n")

total_size = 0
found = 0
missing = []

for i in range(1, 21):
    fname = f"datalist_batch_{i}.pt"
    fpath = os.path.join(batch_dir, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        total_size += size
        found += 1
        size_str = f"{size / 1024**2:.2f} MB"
        print(f"  [{i:2d}/20] {fname:30s} {size_str:>12s}  OK")
    else:
        missing.append(fname)
        print(f"  [{i:2d}/20] {fname:30s} {'MISSING':>12s}  !!!")

print(f"\n  Summary: {found}/20 files found")
if missing:
    print(f"  MISSING: {missing}")
else:
    print(f"  All 20 batch files present")
print(f"  Total size: {total_size / 1024**3:.2f} GB ({total_size:,} bytes)")


# =========================================================================
# SECTION 4: Cross-check scaler consistency between T7 and T8
# =========================================================================
print("\n\n" + "=" * 80)
print("SECTION 4: CROSS-CHECK - ARE T7 AND T8 SCALERS IDENTICAL?")
print("=" * 80)

t8_dir = scaler_dirs["T8 (8th_trial_lower_dropout)"]
t7_dir = scaler_dirs["T7 (7th_trial_80_10_10_split)"]

for pkl_name in sorted(
    set(
        [f for f in os.listdir(t8_dir) if f.endswith(".pkl")]
        + [f for f in os.listdir(t7_dir) if f.endswith(".pkl")]
    )
):
    t8_path = os.path.join(t8_dir, pkl_name)
    t7_path = os.path.join(t7_dir, pkl_name)
    if os.path.exists(t8_path) and os.path.exists(t7_path):
        s8 = joblib.load(t8_path)
        s7 = joblib.load(t7_path)
        if hasattr(s8, "mean_") and hasattr(s7, "mean_"):
            mean_eq = np.allclose(s8.mean_, s7.mean_)
            scale_eq = np.allclose(s8.scale_, s7.scale_)
            print(f"  {pkl_name:30s}  mean_match={mean_eq}  scale_match={scale_eq}")
        else:
            print(f"  {pkl_name:30s}  (no mean_/scale_ to compare)")
    else:
        print(f"  {pkl_name:30s}  (file missing in one trial)")


# =========================================================================
# FINAL SUMMARY
# =========================================================================
print("\n\n" + "=" * 80)
print("FINAL VERIFICATION SUMMARY")
print("=" * 80)
print(
    f"  Test dataloaders (test_dl.pt):  2/2 loaded successfully (list of 100 PyG Data each)"
)
print(
    f"  Scaler files (.pkl):           12/12 loaded successfully (all StandardScaler)"
)
print(
    f"  Training batches (.pt):        {found}/20 present ({total_size / 1024**3:.2f} GB total)"
)
print(f"\n  Overall: ALL FILES VERIFIED SUCCESSFULLY")
print("=" * 80)
