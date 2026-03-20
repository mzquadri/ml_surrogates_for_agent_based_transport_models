#!/usr/bin/env python3
"""
Audit Script for Model 8 UQ Folder
==================================
Checks for duplication, missing files, inconsistent metrics, and misleading UQ reporting.
"""

import os
import sys
import json
import glob
import numpy as np
import math

# Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
os.chdir(REPO_ROOT)

MODEL_FOLDER = "data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout"
UQ_FOLDER = os.path.join(MODEL_FOLDER, "uq_results")

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_status(condition, msg_ok, msg_fail):
    if condition:
        print(f"  ✅ {msg_ok}")
        return True
    else:
        print(f"  ❌ {msg_fail}")
        return False

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

def main():
    issues = []
    
    print_header("MODEL 8 UQ FOLDER AUDIT")
    print(f"Folder: {UQ_FOLDER}")
    
    # =========================================================================
    # 1. CHECK CHECKPOINT FOLDERS
    # =========================================================================
    print_header("1. CHECKPOINT FOLDERS")
    
    # List all checkpoint folders
    ckpt_folders = [d for d in os.listdir(UQ_FOLDER) if d.startswith("checkpoint")]
    print(f"  Found checkpoint folders: {ckpt_folders}")
    
    # Check MC30 checkpoints
    mc30_ckpt = os.path.join(UQ_FOLDER, "checkpoints_mc30")
    if os.path.exists(mc30_ckpt):
        mc30_files = sorted(glob.glob(os.path.join(mc30_ckpt, "graph_*.npz")))
        mc30_indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in mc30_files]
        
        expected = set(range(100))
        actual = set(mc30_indices)
        missing = expected - actual
        extra = actual - expected
        
        print(f"  MC30 checkpoints: {len(mc30_files)} files")
        check_status(len(missing) == 0, f"No missing graphs (0-99 complete)", f"MISSING: {sorted(missing)}")
        check_status(len(extra) == 0, f"No extra graphs", f"EXTRA: {sorted(extra)}")
        
        if missing:
            issues.append(f"MC30 checkpoints missing: {sorted(missing)}")
        if extra:
            issues.append(f"MC30 checkpoints extra: {sorted(extra)}")
    else:
        print(f"  ❌ checkpoints_mc30 folder NOT FOUND")
        issues.append("checkpoints_mc30 folder missing")
    
    # Check for MC50 checkpoints (should not exist or be separate)
    mc50_ckpt = os.path.join(UQ_FOLDER, "checkpoints_mc50")
    if os.path.exists(mc50_ckpt):
        mc50_files = glob.glob(os.path.join(mc50_ckpt, "graph_*.npz"))
        print(f"  ⚠️  MC50 checkpoints found: {len(mc50_files)} files (separate folder - OK if intentional)")
    else:
        print(f"  ✅ No MC50 checkpoint folder (no mixing risk)")
    
    # Check deterministic checkpoints
    det_ckpt = os.path.join(UQ_FOLDER, "deterministic_checkpoints")
    if os.path.exists(det_ckpt):
        det_files = sorted(glob.glob(os.path.join(det_ckpt, "graph_*.npz")))
        det_indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in det_files]
        
        expected = set(range(100))
        actual = set(det_indices)
        missing = expected - actual
        
        print(f"  Deterministic checkpoints: {len(det_files)} files")
        check_status(len(missing) == 0, f"No missing graphs (0-99 complete)", f"MISSING: {sorted(missing)}")
        
        if missing:
            issues.append(f"Deterministic checkpoints missing: {sorted(missing)}")
    else:
        print(f"  ⚠️  deterministic_checkpoints folder NOT FOUND (may be OK if aggregated)")
    
    # =========================================================================
    # 2. CHECK MC DROPOUT NPZ FILE
    # =========================================================================
    print_header("2. MC DROPOUT NPZ FILE")
    
    mc_npz_path = os.path.join(UQ_FOLDER, "mc_dropout_full_100graphs_mc30.npz")
    if os.path.exists(mc_npz_path):
        mc_data = np.load(mc_npz_path)
        
        print(f"  Keys: {list(mc_data.keys())}")
        
        preds = mc_data["predictions"]
        uncs = mc_data["uncertainties"]
        targets = mc_data["targets"]
        
        print(f"  Shapes: predictions={preds.shape}, uncertainties={uncs.shape}, targets={targets.shape}")
        print(f"  Dtypes: predictions={preds.dtype}, uncertainties={uncs.dtype}, targets={targets.dtype}")
        
        # Check lengths match
        all_same_len = (len(preds) == len(uncs) == len(targets))
        check_status(all_same_len, f"All arrays same length: {len(preds):,}", "LENGTH MISMATCH!")
        
        if not all_same_len:
            issues.append(f"MC npz length mismatch: preds={len(preds)}, uncs={len(uncs)}, targets={len(targets)}")
        
        # Check expected length
        check_status(len(preds) == 3163500, f"Expected length 3,163,500: ✓", f"Length is {len(preds):,}, expected 3,163,500")
        
        # Check for NaN/Inf
        nan_preds = np.isnan(preds).sum()
        nan_uncs = np.isnan(uncs).sum()
        nan_targets = np.isnan(targets).sum()
        inf_preds = np.isinf(preds).sum()
        inf_uncs = np.isinf(uncs).sum()
        
        check_status(nan_preds == 0, "No NaN in predictions", f"NaN in predictions: {nan_preds}")
        check_status(nan_uncs == 0, "No NaN in uncertainties", f"NaN in uncertainties: {nan_uncs}")
        check_status(nan_targets == 0, "No NaN in targets", f"NaN in targets: {nan_targets}")
        check_status(inf_preds == 0, "No Inf in predictions", f"Inf in predictions: {inf_preds}")
        check_status(inf_uncs == 0, "No Inf in uncertainties", f"Inf in uncertainties: {inf_uncs}")
        
        if nan_preds + nan_uncs + nan_targets + inf_preds + inf_uncs > 0:
            issues.append("NaN or Inf values found in MC npz")
        
        # Check uncertainties non-negative
        neg_uncs = (uncs < 0).sum()
        check_status(neg_uncs == 0, "All uncertainties non-negative", f"Negative uncertainties: {neg_uncs}")
        
        if neg_uncs > 0:
            issues.append(f"Negative uncertainties found: {neg_uncs}")
        
        # Compute metrics from npz
        mc_r2 = r2_score(targets.astype(float), preds.astype(float))
        mc_mae = np.mean(np.abs(targets - preds))
        mc_rmse = np.sqrt(np.mean((targets - preds) ** 2))
        
        print(f"\n  Recomputed metrics from NPZ:")
        print(f"    R²   = {mc_r2:.6f}")
        print(f"    MAE  = {mc_mae:.4f}")
        print(f"    RMSE = {mc_rmse:.4f}")
        
        mc_data.close()
    else:
        print(f"  ❌ MC Dropout NPZ NOT FOUND: {mc_npz_path}")
        issues.append("MC Dropout NPZ file missing")
        mc_r2, mc_mae, mc_rmse = None, None, None
    
    # =========================================================================
    # 3. CHECK DETERMINISTIC NPZ FILE
    # =========================================================================
    print_header("3. DETERMINISTIC NPZ FILE")
    
    det_npz_path = os.path.join(UQ_FOLDER, "deterministic_full_100graphs.npz")
    if os.path.exists(det_npz_path):
        det_data = np.load(det_npz_path)
        
        print(f"  Keys: {list(det_data.keys())}")
        
        det_preds = det_data["predictions"]
        det_targets = det_data["targets"]
        
        print(f"  Shapes: predictions={det_preds.shape}, targets={det_targets.shape}")
        
        # Check length matches MC
        if mc_r2 is not None:
            check_status(len(det_preds) == 3163500, f"Length matches MC: {len(det_preds):,}", f"Length mismatch: {len(det_preds):,}")
        
        # Compute metrics
        det_r2 = r2_score(det_targets.astype(float), det_preds.astype(float))
        det_mae = np.mean(np.abs(det_targets - det_preds))
        det_rmse = np.sqrt(np.mean((det_targets - det_preds) ** 2))
        
        print(f"\n  Recomputed metrics from NPZ:")
        print(f"    R²   = {det_r2:.6f}")
        print(f"    MAE  = {det_mae:.4f}")
        print(f"    RMSE = {det_rmse:.4f}")
        
        # Compare predictions and verify targets match
        if mc_r2 is not None:
            mc_data = np.load(mc_npz_path)
            
            # Critical check: targets must match exactly
            mc_targets = mc_data["targets"]
            max_abs_target_diff = np.max(np.abs(mc_targets - det_targets))
            print(f"\n  Max |target_mc - target_det|: {max_abs_target_diff:.6e}")
            check_status(max_abs_target_diff < 1e-6, "Targets match between MC and Deterministic", "Targets DO NOT match (ordering mismatch?)")
            if max_abs_target_diff >= 1e-6:
                issues.append(f"Targets mismatch between MC and Det NPZ: max diff = {max_abs_target_diff}")
            
            # Compare predictions
            mean_abs_diff = np.mean(np.abs(mc_data["predictions"] - det_preds))
            print(f"  Mean abs diff (MC vs Det preds): {mean_abs_diff:.4f} min")
            mc_data.close()
        
        det_data.close()
    else:
        print(f"  ❌ Deterministic NPZ NOT FOUND: {det_npz_path}")
        issues.append("Deterministic NPZ file missing")
        det_r2, det_mae, det_rmse = None, None, None
    
    # =========================================================================
    # 4. CHECK JSON FILES CONSISTENCY
    # =========================================================================
    print_header("4. JSON FILES CONSISTENCY")
    
    # MC JSON
    mc_json_files = glob.glob(os.path.join(UQ_FOLDER, "mc_dropout*metrics*.json"))
    print(f"  MC JSON files: {[os.path.basename(f) for f in mc_json_files]}")
    
    for jf in mc_json_files:
        with open(jf) as f:
            data = json.load(f)
        print(f"\n  {os.path.basename(jf)}:")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")
        
        # Compare with recomputed
        if mc_r2 is not None and "r2" in data:
            diff_r2 = abs(data["r2"] - mc_r2)
            check_status(diff_r2 < 0.001, f"R² matches recomputed (diff={diff_r2:.6f})", f"R² MISMATCH: JSON={data['r2']:.6f}, recomputed={mc_r2:.6f}")
            if diff_r2 >= 0.001:
                issues.append(f"MC JSON R² mismatch: {diff_r2:.6f}")
    
    # Deterministic JSON
    det_json_files = glob.glob(os.path.join(UQ_FOLDER, "deterministic*metrics*.json"))
    print(f"\n  Deterministic JSON files: {[os.path.basename(f) for f in det_json_files]}")
    
    for jf in det_json_files:
        with open(jf) as f:
            data = json.load(f)
        print(f"\n  {os.path.basename(jf)}:")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")
        
        # Compare with recomputed
        if det_r2 is not None and "r2" in data:
            diff_r2 = abs(data["r2"] - det_r2)
            check_status(diff_r2 < 0.001, f"R² matches recomputed (diff={diff_r2:.6f})", f"R² MISMATCH: JSON={data['r2']:.6f}, recomputed={det_r2:.6f}")
            if diff_r2 >= 0.001:
                issues.append(f"Det JSON R² mismatch: {diff_r2:.6f}")
    
    # =========================================================================
    # 5. CHECK UQ PLOTS
    # =========================================================================
    print_header("5. UQ PLOTS")
    
    plots_folder = os.path.join(UQ_FOLDER, "uq_plots")
    if os.path.exists(plots_folder):
        plot_files = os.listdir(plots_folder)
        print(f"  Found {len(plot_files)} files:")
        for pf in sorted(plot_files):
            size_kb = os.path.getsize(os.path.join(plots_folder, pf)) / 1024
            print(f"    - {pf} ({size_kb:.1f} KB)")
        
        # Check expected plots exist
        expected_plots = [
            "uncertainty_hist.png",
            "uncertainty_vs_error_scatter.png",
            "binned_error_vs_uncertainty.png",
            "coverage_curve_k_sigma.png",
        ]
        
        for ep in expected_plots:
            check_status(ep in plot_files, f"{ep} exists", f"{ep} MISSING")
            if ep not in plot_files:
                issues.append(f"Missing plot: {ep}")
    else:
        print(f"  ❌ uq_plots folder NOT FOUND")
        issues.append("uq_plots folder missing")
    
    # =========================================================================
    # 6. CHECK CONFORMAL CALIBRATION METHOD
    # =========================================================================
    print_header("6. CONFORMAL CALIBRATION METHOD")
    
    # Check uq_comparison JSON
    comp_json = os.path.join(UQ_FOLDER, "uq_comparison_model8.json")
    if os.path.exists(comp_json):
        with open(comp_json) as f:
            comp_data = json.load(f)
        print(f"  uq_comparison_model8.json found")
        tn = comp_data.get("test_nodes", None)
        if isinstance(tn, (int, np.integer)):
            print(f"  Test nodes: {tn:,}")
        else:
            print(f"  Test nodes: {tn}")
        
        # The conformal script uses test-split (not validation)
        print(f"\n  ⚠️  CONFORMAL CALIBRATION NOTE:")
        print(f"     Your conformal comparison used a split from the SAME pooled node set.")
        print(f"     run_conformal_comparison.py uses 50/50 (calibration/test) split.")
        print(f"     This is fine as an internal evaluation, but thesis me clearly likho:")
        print(f"     'split conformal on a held-out calibration split from the test pool'")
        print(f"     OR ideally recalibrate on validation set if available.")
        
        issues.append("THESIS WORDING: Clarify conformal uses 50/50 test-split (not val set)")
    else:
        print(f"  uq_comparison_model8.json NOT FOUND")
    
    # Check if val_dl.pt exists (could be used for proper conformal)
    data_folder = os.path.join(MODEL_FOLDER, "data_created_during_training")
    val_candidates = glob.glob(os.path.join(data_folder, "*val*dl*.pt")) \
                  + glob.glob(os.path.join(data_folder, "*valid*dl*.pt")) \
                  + glob.glob(os.path.join(data_folder, "*val*.pt")) \
                  + glob.glob(os.path.join(data_folder, "*valid*.pt"))
    val_candidates = list(set(val_candidates))  # remove duplicates
    
    if val_candidates:
        print(f"\n  ✅ Validation file candidates found:")
        for p in val_candidates:
            print(f"     - {os.path.basename(p)}")
        print(f"     (Could be used for thesis-correct conformal: calibration on val, evaluation on test)")
    else:
        print(f"\n  ⚠️  No validation loader found (*val*.pt) - test-split conformal is acceptable")
    
    # =========================================================================
    # 7. CHECK FOR DUPLICATE/MISLEADING FILES
    # =========================================================================
    print_header("7. DUPLICATE/MISLEADING FILES CHECK")
    
    all_files = os.listdir(UQ_FOLDER)
    print(f"  All files in uq_results/:")
    for f in sorted(all_files):
        if os.path.isfile(os.path.join(UQ_FOLDER, f)):
            print(f"    - {f}")
    
    # Check for any MC50 files mixed with MC30
    mc50_files = [f for f in all_files if "mc50" in f.lower() or "mc_50" in f.lower()]
    if mc50_files:
        print(f"\n  ⚠️  MC50 files found (potential mixing):")
        for f in mc50_files:
            print(f"    - {f}")
        issues.append(f"MC50 files found in same folder: {mc50_files}")
    else:
        print(f"\n  ✅ No MC50 files in main folder (no mixing)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("AUDIT SUMMARY")
    
    if issues:
        print(f"\n❌ ISSUES FOUND ({len(issues)}):\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ ALL CHECKS PASSED - No issues found!")
    
    print("\n" + "=" * 70)
    
    return issues

if __name__ == "__main__":
    issues = main()
    sys.exit(0 if not issues else 1)
