"""
Cross-verification: NPZ raw arrays vs stored JSON metrics.
Thesis integrity check - all 8 checks.
"""

import numpy as np
import json
import os
import sys
import io
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
T8 = os.path.join(
    ROOT, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
T7 = os.path.join(
    ROOT, "data", "TR-C_Benchmarks", "point_net_transf_gat_7th_trial_80_10_10_split"
)

passes = 0
fails = 0


def check(name, computed, expected, tol=0.01):
    global passes, fails
    diff = abs(computed - expected)
    ok = diff <= tol
    status = "PASS" if ok else "FAIL"
    if ok:
        passes += 1
    else:
        fails += 1
    print(
        f"  [{status}] {name}: computed={computed:.6f}  expected={expected:.6f}  diff={diff:.6f}  tol={tol}"
    )
    return ok


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────
section("CHECK 1: T8 Deterministic Metrics (NPZ vs test_evaluation_complete.json)")
# ─────────────────────────────────────────────────────────────────────
det8 = np.load(os.path.join(T8, "uq_results", "deterministic_full_100graphs.npz"))
preds = det8["predictions"]
targs = det8["targets"]
with open(os.path.join(T8, "test_evaluation_complete.json")) as f:
    json_t8 = json.load(f)

r2_comp = r2_score(targs, preds)
mae_comp = mean_absolute_error(targs, preds)
rmse_comp = np.sqrt(mean_squared_error(targs, preds))
pearson_comp = np.corrcoef(preds, targs)[0, 1]
spearman_comp = stats.spearmanr(preds, targs).correlation

jt = json_t8["test_metrics"]
check("R²", r2_comp, jt["r2"], 0.01)
check("MAE", mae_comp, jt["mae"], 0.01)
check("RMSE", rmse_comp, jt["rmse"], 0.01)
check("Pearson", pearson_comp, jt["pearson"], 0.01)
check("Spearman", spearman_comp, jt["spearman"], 0.01)

# Also cross-check against the hardcoded thesis values
check("R²  vs thesis 0.5957", r2_comp, 0.5957, 0.01)
check("MAE vs thesis 3.96", mae_comp, 3.96, 0.01)
check("RMSE vs thesis 7.12", rmse_comp, 7.12, 0.01)
check("Pearson vs thesis 0.773", pearson_comp, 0.773, 0.01)

# ─────────────────────────────────────────────────────────────────────
section("CHECK 2: T8 MC Dropout Metrics (NPZ vs mc_dropout JSON)")
# ─────────────────────────────────────────────────────────────────────
mc8 = np.load(os.path.join(T8, "uq_results", "mc_dropout_full_100graphs_mc30.npz"))
mc_preds = mc8["predictions"]
mc_unc = mc8["uncertainties"]
mc_targs = mc8["targets"]

with open(
    os.path.join(T8, "uq_results", "mc_dropout_full_metrics_model8_mc30_100graphs.json")
) as f:
    json_mc8 = json.load(f)

abs_err = np.abs(mc_preds - mc_targs)
spearman_mc = stats.spearmanr(mc_unc, abs_err).correlation
mae_mc = mean_absolute_error(mc_targs, mc_preds)
unc_mean = float(np.mean(mc_unc))

check("Spearman ρ (unc vs |err|)", spearman_mc, json_mc8["spearman"], 0.005)
check("MAE", mae_mc, json_mc8["mae"], 0.01)
check("Mean uncertainty (σ̄)", unc_mean, json_mc8["unc_mean"], 0.01)

# Also vs thesis values
check("ρ vs thesis 0.4820", spearman_mc, 0.4820, 0.005)
check("MAE vs thesis 3.948", mae_mc, 3.948, 0.01)
check("σ̄ vs thesis 1.369", unc_mean, 1.369, 0.01)

# ─────────────────────────────────────────────────────────────────────
section("CHECK 3: T8 test_predictions.npz vs deterministic_full_100graphs.npz")
# ─────────────────────────────────────────────────────────────────────
tp = np.load(os.path.join(T8, "test_predictions.npz"))
tp_preds = tp["predictions"]
tp_targs = tp["targets"]

pred_match = np.allclose(tp_preds, preds, atol=1e-6)
targ_match = np.allclose(tp_targs, targs, atol=1e-6)

if pred_match:
    passes += 1
    print(f"  [PASS] Predictions arrays identical (allclose atol=1e-6)")
else:
    fails += 1
    maxdiff = float(np.max(np.abs(tp_preds - preds)))
    print(f"  [FAIL] Predictions differ — max diff = {maxdiff:.8f}")

if targ_match:
    passes += 1
    print(f"  [PASS] Targets arrays identical (allclose atol=1e-6)")
else:
    fails += 1
    maxdiff = float(np.max(np.abs(tp_targs - targs)))
    print(f"  [FAIL] Targets differ — max diff = {maxdiff:.8f}")

# ─────────────────────────────────────────────────────────────────────
section("CHECK 4: T8 Ensemble Experiment A (NPZ vs JSON)")
# ─────────────────────────────────────────────────────────────────────
ea = np.load(
    os.path.join(
        T8, "uq_results", "ensemble_experiments", "experiment_a_fixed_data.npz"
    )
)
with open(
    os.path.join(
        T8, "uq_results", "ensemble_experiments", "experiment_a_fixed_results.json"
    )
) as f:
    json_ea = json.load(f)

ea_targs = ea["targets"]
ea_run_preds = ea["run_predictions"]  # (5, N)
ea_run_unc = ea["run_uncertainties"]  # (5, N)

# Recompute ensemble mean/std from run_predictions
ens_mean_comp = np.mean(ea_run_preds, axis=0)
ens_std_comp = np.std(ea_run_preds, axis=0)

# Verify stored ensemble_mean_prediction matches recomputed
ens_mean_stored = ea["ensemble_mean_prediction"]
ens_mean_match = np.allclose(ens_mean_comp, ens_mean_stored, atol=1e-4)
if ens_mean_match:
    passes += 1
    print(f"  [PASS] ensemble_mean_prediction matches mean(run_predictions)")
else:
    fails += 1
    maxdiff = float(np.max(np.abs(ens_mean_comp - ens_mean_stored)))
    print(f"  [FAIL] ensemble_mean mismatch — max diff = {maxdiff:.6f}")

# Verify stored ensemble_variance matches recomputed
ens_var_stored = ea["ensemble_variance"]
ens_var_comp = np.std(ea_run_preds, axis=0)  # stored as std, check
# It might be stored as std not variance — let's check both
if np.allclose(ens_var_comp, ens_var_stored, atol=1e-4):
    passes += 1
    print(f"  [PASS] ensemble_variance matches std(run_predictions)")
elif np.allclose(np.var(ea_run_preds, axis=0), ens_var_stored, atol=1e-4):
    passes += 1
    print(f"  [PASS] ensemble_variance matches var(run_predictions)")
else:
    fails += 1
    print(f"  [FAIL] ensemble_variance doesn't match std or var of run_predictions")
    print(
        f"         stored mean={float(np.mean(ens_var_stored)):.6f}, "
        f"recomp std mean={float(np.mean(ens_var_comp)):.6f}, "
        f"recomp var mean={float(np.mean(np.var(ea_run_preds, axis=0))):.6f}"
    )

# MC dropout ρ: avg_mc_uncertainty vs |ensemble_mean - target|
avg_mc_unc = ea["avg_mc_uncertainty"]
abs_err_ea = np.abs(ens_mean_stored - ea_targs)
sp_mc_ea = stats.spearmanr(avg_mc_unc, abs_err_ea).correlation
check("MC ρ", sp_mc_ea, json_ea["mc_dropout"]["spearman_rho"], 0.005)
check("MC ρ vs thesis 0.4908", sp_mc_ea, 0.4908, 0.005)

# Ensemble variance ρ
sp_ens_ea = stats.spearmanr(ens_var_stored, abs_err_ea).correlation
check("Ens ρ", sp_ens_ea, json_ea["ensemble_variance"]["spearman_rho"], 0.005)
check("Ens ρ vs thesis 0.4370", sp_ens_ea, 0.4370, 0.005)

# Combined ρ
comb_unc = ea["combined_uncertainty"]
sp_comb_ea = stats.spearmanr(comb_unc, abs_err_ea).correlation
check("Combined ρ", sp_comb_ea, json_ea["combined"]["spearman_rho"], 0.005)
check("Combined ρ vs thesis 0.4909", sp_comb_ea, 0.4909, 0.005)

# ─────────────────────────────────────────────────────────────────────
section("CHECK 5: T8 Ensemble Experiment B (NPZ vs JSON)")
# ─────────────────────────────────────────────────────────────────────
eb = np.load(
    os.path.join(
        T8, "uq_results", "ensemble_experiments", "experiment_b_fixed_data.npz"
    )
)
with open(
    os.path.join(
        T8, "uq_results", "ensemble_experiments", "experiment_b_fixed_results.json"
    )
) as f:
    json_eb = json.load(f)

eb_targs = eb["targets"]
eb_ens_pred = eb["ensemble_prediction"]
eb_ens_unc = eb["ensemble_uncertainty"]

abs_err_eb = np.abs(eb_ens_pred - eb_targs)
sp_ens_eb = stats.spearmanr(eb_ens_unc, abs_err_eb).correlation
r2_eb = r2_score(eb_targs, eb_ens_pred)
mae_eb = mean_absolute_error(eb_targs, eb_ens_pred)

check("Ens ρ", sp_ens_eb, json_eb["ensemble"]["spearman_rho"], 0.005)
check("Ens ρ vs thesis 0.4333", sp_ens_eb, 0.4333, 0.005)
check("R²", r2_eb, json_eb["ensemble"]["r2"], 0.01)
check("R² vs thesis 0.5656", r2_eb, 0.5656, 0.01)
check("MAE", mae_eb, json_eb["ensemble"]["mae"], 0.01)
check("MAE vs thesis 3.99", mae_eb, 3.99, 0.01)

# ─────────────────────────────────────────────────────────────────────
section("CHECK 6: T7 Deterministic Metrics (NPZ recomputed)")
# ─────────────────────────────────────────────────────────────────────
det7 = np.load(os.path.join(T7, "uq_results", "deterministic_full_100graphs.npz"))
p7 = det7["predictions"]
t7 = det7["targets"]

r2_7 = r2_score(t7, p7)
mae_7 = mean_absolute_error(t7, p7)
rmse_7 = np.sqrt(mean_squared_error(t7, p7))
pearson_7 = np.corrcoef(p7, t7)[0, 1]

check("R²", r2_7, 0.5471, 0.01)
check("MAE", mae_7, 4.06, 0.01)
check("RMSE", rmse_7, 7.53, 0.01)
check("Pearson", pearson_7, 0.741, 0.01)

# Also vs the stored JSON
with open(
    os.path.join(T7, "uq_results", "deterministic_metrics_model7_100graphs.json")
) as f:
    json_det7 = json.load(f)
check("R² vs JSON", r2_7, json_det7["r2"], 0.01)
check("MAE vs JSON", mae_7, json_det7["mae"], 0.01)
check("RMSE vs JSON", rmse_7, json_det7["rmse"], 0.01)

# ─────────────────────────────────────────────────────────────────────
section("CHECK 7: T7 MC Dropout Metrics (NPZ recomputed)")
# ─────────────────────────────────────────────────────────────────────
mc7 = np.load(os.path.join(T7, "uq_results", "mc_dropout_full_100graphs_mc30.npz"))
mc7_preds = mc7["predictions"]
mc7_unc = mc7["uncertainties"]
mc7_targs = mc7["targets"]

abs_err_7 = np.abs(mc7_preds - mc7_targs)
sp_mc7 = stats.spearmanr(mc7_unc, abs_err_7).correlation

with open(
    os.path.join(T7, "uq_results", "mc_dropout_full_metrics_model7_mc30_100graphs.json")
) as f:
    json_mc7 = json.load(f)

check("Spearman ρ vs JSON 0.4437", sp_mc7, json_mc7["spearman"], 0.005)
check("Spearman ρ vs thesis 0.4437", sp_mc7, 0.4437, 0.005)
check("Spearman ρ vs thesis 0.4460 (100K subsample)", sp_mc7, 0.4460, 0.01)

# ─────────────────────────────────────────────────────────────────────
section("CHECK 8: T8 CSV consistency with NPZ")
# ─────────────────────────────────────────────────────────────────────
import csv

csv_path = os.path.join(T8, "trial8_uq_ablation_results.csv")

# Read first 1000 rows for spot-check
csv_targets = []
csv_pred_det = []
csv_pred_mc_mean = []
csv_pred_mc_std = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 1000:
            break
        csv_targets.append(float(row["target"]))
        csv_pred_det.append(float(row["pred_det"]))
        csv_pred_mc_mean.append(float(row["pred_mc_mean"]))
        csv_pred_mc_std.append(float(row["pred_mc_std"]))

csv_targets = np.array(csv_targets, dtype=np.float32)
csv_pred_det = np.array(csv_pred_det, dtype=np.float32)
csv_pred_mc_mean = np.array(csv_pred_mc_mean, dtype=np.float32)
csv_pred_mc_std = np.array(csv_pred_mc_std, dtype=np.float32)

# Compare with test_predictions.npz (deterministic)
tp_preds_sub = tp_preds[:1000]
tp_targs_sub = tp_targs[:1000]

# Compare with mc_dropout
mc_preds_sub = mc_preds[:1000]
mc_unc_sub = mc_unc[:1000]
mc_targs_sub = mc_targs[:1000]

# --- 8a: Deterministic columns (should be exact match) ---
targ_csv_det = np.allclose(csv_targets, tp_targs_sub, atol=1e-4)
pred_csv_det = np.allclose(csv_pred_det, tp_preds_sub, atol=1e-4)
targ_csv_mc = np.allclose(csv_targets, mc_targs_sub, atol=1e-4)

for label, result, arr1, arr2 in [
    (
        "CSV target vs test_predictions.npz target",
        targ_csv_det,
        csv_targets,
        tp_targs_sub,
    ),
    (
        "CSV pred_det vs test_predictions.npz predictions",
        pred_csv_det,
        csv_pred_det,
        tp_preds_sub,
    ),
    ("CSV target vs mc_dropout target", targ_csv_mc, csv_targets, mc_targs_sub),
]:
    if result:
        passes += 1
        print(f"  [PASS] {label}  (first 1000 rows, atol=1e-4)")
    else:
        fails += 1
        maxdiff = float(np.max(np.abs(arr1 - arr2)))
        print(f"  [FAIL] {label}  max diff={maxdiff:.8f}")

# --- 8b: MC columns --- CSV used S=50, NPZ used S=30, so exact match not expected.
#     Instead verify high Pearson correlation (same model, different S).
print()
print("  NOTE: CSV ablation used S=50 MC samples; authoritative NPZ used S=30.")
print("        Exact match not expected. Checking rank-order consistency instead.")

r_mc_pred = np.corrcoef(csv_pred_mc_mean, mc_preds_sub)[0, 1]
r_mc_unc = np.corrcoef(csv_pred_mc_std, mc_unc_sub)[0, 1]

if r_mc_pred > 0.95:
    passes += 1
    print(
        f"  [PASS] CSV pred_mc_mean vs NPZ mc predictions: Pearson r={r_mc_pred:.4f} > 0.95"
    )
else:
    fails += 1
    print(
        f"  [FAIL] CSV pred_mc_mean vs NPZ mc predictions: Pearson r={r_mc_pred:.4f} <= 0.95"
    )

if r_mc_unc > 0.90:
    passes += 1
    print(
        f"  [PASS] CSV pred_mc_std vs NPZ mc uncertainties: Pearson r={r_mc_unc:.4f} > 0.90"
    )
else:
    fails += 1
    print(
        f"  [FAIL] CSV pred_mc_std vs NPZ mc uncertainties: Pearson r={r_mc_unc:.4f} <= 0.90"
    )

# ─────────────────────────────────────────────────────────────────────
section("SUMMARY")
# ─────────────────────────────────────────────────────────────────────
total = passes + fails
print(f"\n  Total checks: {total}")
print(f"  PASSED: {passes}")
print(f"  FAILED: {fails}")
if fails == 0:
    print(f"\n  *** ALL CHECKS PASSED — NPZ data is consistent with JSON metrics ***")
else:
    print(f"\n  *** {fails} CHECK(S) FAILED — REVIEW ABOVE ***")

sys.exit(0 if fails == 0 else 1)
