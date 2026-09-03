"""
Verification script: loads artifacts (NPZ + JSON),
recomputes key metrics from raw arrays, and cross-checks against
verified JSON summaries and thesis figure expectations.

Author: Verification script for Nazim's thesis
"""

import json
import os
import sys
import pathlib
import traceback
from collections import OrderedDict

import numpy as np
from scipy import stats as sp_stats

# ============================================================
# PATHS
# ============================================================
BASE = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = (
    BASE
    / "data"
    / "TR-C_Benchmarks"
    / "point_net_transf_gat_8th_trial_lower_dropout"
    / "uq_results"
)
ARTIFACT_DIR = BASE / "docs" / "verified" / "phase3_results"
FIGURE_DIR = BASE / "thesis" / "latex_tum_official" / "figures"

MC_NPZ = DATA_DIR / "mc_dropout_full_100graphs_mc30.npz"
DET_NPZ = DATA_DIR / "deterministic_full_100graphs.npz"
MC_JSON = DATA_DIR / "mc_dropout_full_metrics_model8_mc30_100graphs.json"
CONFORMAL_JSON = DATA_DIR / "conformal_standard.json"

# Collect all results
all_results = OrderedDict()
n_pass = 0
n_fail = 0
n_warn = 0


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def check(name, computed, expected, tol=1e-3, rel=False):
    global n_pass, n_fail
    if rel and expected != 0:
        ok = abs(computed - expected) / abs(expected) < tol
    else:
        ok = abs(computed - expected) < tol
    status = "PASS" if ok else "FAIL"
    marker = "  [PASS]" if ok else "  [FAIL] ***"
    print(
        f"  {name:<55} computed={computed:<14.6f}  expected={expected:<14.6f} {marker}"
    )
    all_results[name] = {
        "computed": float(computed),
        "expected": float(expected),
        "status": status,
    }
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    return ok


def warn(msg):
    global n_warn
    n_warn += 1
    print(f"  [WARN] {msg}")


# ============================================================
# SECTION 1: Load and verify T8 MC Dropout NPZ raw data
# ============================================================
section("1. T8 MC Dropout — Raw NPZ Recomputation")

if not MC_NPZ.exists():
    print(f"  [FATAL] MC Dropout NPZ not found at: {MC_NPZ}")
    sys.exit(1)

print(f"  Loading: {MC_NPZ}")
mc_data = np.load(str(MC_NPZ))
print(f"  Keys in NPZ: {list(mc_data.keys())}")
print()

# Discover array names
y_true_key = None
y_pred_mean_key = None
y_pred_std_key = None

for k in mc_data.keys():
    kl = k.lower()
    if "true" in kl or kl == "y_true":
        y_true_key = k
    elif "mean" in kl or kl == "y_pred_mean" or kl == "y_pred":
        y_pred_mean_key = k
    elif (
        "std" in kl
        or kl == "y_pred_std"
        or kl == "sigma"
        or kl == "uncertainty"
        or kl == "uncertainties"
    ):
        y_pred_std_key = k

# If auto-detect fails, try common patterns
if y_true_key is None:
    for k in mc_data.keys():
        if k in ("y_true", "targets", "target", "y"):
            y_true_key = k
            break
if y_pred_mean_key is None:
    for k in mc_data.keys():
        if k in ("y_pred_mean", "predictions", "pred", "y_pred", "mean"):
            y_pred_mean_key = k
            break
if y_pred_std_key is None:
    for k in mc_data.keys():
        if k in ("y_pred_std", "std", "sigma", "uncertainty", "unc", "uncertainties"):
            y_pred_std_key = k
            break

print(
    f"  Detected keys -> y_true: {y_true_key}, y_pred_mean: {y_pred_mean_key}, y_pred_std: {y_pred_std_key}"
)

# Print all arrays info
for k in mc_data.keys():
    arr = mc_data[k]
    print(
        f"    '{k}': shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
    )
print()

y_true = mc_data[y_true_key].astype(np.float64)
y_pred_mean = mc_data[y_pred_mean_key].astype(np.float64)
y_pred_std = mc_data[y_pred_std_key].astype(np.float64)

n_nodes = len(y_true)
print(f"  Total nodes: {n_nodes:,}")

# --- Compute metrics from raw arrays ---
errors = y_true - y_pred_mean
abs_errors = np.abs(errors)

# R²
ss_res = np.sum(errors**2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1.0 - ss_res / ss_tot

# MAE
mae = np.mean(abs_errors)

# RMSE
rmse = np.sqrt(np.mean(errors**2))

# Spearman correlation between std and |error|
spearman_rho, spearman_p = sp_stats.spearmanr(y_pred_std, abs_errors)

# Mean uncertainty
unc_mean = np.mean(y_pred_std)
unc_std = np.std(y_pred_std)
unc_min = np.min(y_pred_std)
unc_max = np.max(y_pred_std)

print(f"\n  --- Recomputed Metrics from Raw Arrays ---")
print(f"  R²:                {r2:.10f}")
print(f"  MAE:               {mae:.10f}")
print(f"  RMSE:              {rmse:.10f}")
print(f"  Spearman rho:      {spearman_rho:.10f}")
print(f"  Spearman p-value:  {spearman_p:.2e}")
print(f"  Mean uncertainty:  {unc_mean:.10f}")
print(f"  Std uncertainty:   {unc_std:.10f}")
print(f"  Min uncertainty:   {unc_min:.10f}")
print(f"  Max uncertainty:   {unc_max:.10f}")
print()

# Load the pre-computed JSON
with open(str(MC_JSON), "r") as f:
    mc_ref = json.load(f)

print("  --- Cross-check: Recomputed vs JSON artifact ---")
check("R² (recomputed vs JSON)", r2, mc_ref["r2"], tol=1e-4)
check("MAE (recomputed vs JSON)", mae, mc_ref["mae"], tol=1e-3)
check("RMSE (recomputed vs JSON)", rmse, mc_ref["rmse"], tol=1e-3)
check("Spearman rho (recomputed vs JSON)", spearman_rho, mc_ref["spearman"], tol=1e-3)
check("Mean uncertainty (recomputed vs JSON)", unc_mean, mc_ref["unc_mean"], tol=1e-3)

# Check thesis-cited values
print("\n  --- Cross-check: Recomputed vs Thesis-cited values ---")
check("R² (recomputed vs thesis 0.5857)", r2, 0.5857, tol=5e-4)
check("MAE (recomputed vs thesis 3.95)", mae, 3.95, tol=0.01)
check("RMSE (recomputed vs thesis 7.21)", rmse, 7.21, tol=0.01)
check("Spearman rho (recomputed vs thesis 0.4820)", spearman_rho, 0.4820, tol=1e-3)
check("Mean unc (recomputed vs thesis 1.369)", unc_mean, 1.369, tol=0.001)

# ============================================================
# SECTION 2: Selective Prediction — recompute from raw arrays
# ============================================================
section("2. Selective Prediction — Recomputed from NPZ")

# Sort by uncertainty (ascending), compute MAE at various retention levels
sort_idx = np.argsort(y_pred_std)
sorted_abs_errors = abs_errors[sort_idx]

retention_levels = [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 25, 10]

with open(str(ARTIFACT_DIR / "selective_prediction_s30.json"), "r") as f:
    sel_ref = json.load(f)
sel_tab = {r["retained_pct"]: r for r in sel_ref["retention_table"]}

print(f"  {'Retain%':<10} {'Recomputed MAE':<18} {'JSON MAE':<18} {'Match?'}")
print(f"  {'-' * 60}")

for pct in retention_levels:
    n_keep = int(n_nodes * pct / 100)
    recomp_mae = np.mean(sorted_abs_errors[:n_keep])
    ref_mae = sel_tab[pct]["MAE"]
    check(f"Selective MAE @ {pct}%", recomp_mae, ref_mae, tol=0.01)

# Key reductions
baseline_mae = np.mean(abs_errors)
for tag, pct, expected_red in [
    ("retain_90pct", 90, 18.3),
    ("retain_50pct", 50, 41.2),
    ("retain_25pct", 25, 54.6),
]:
    n_keep = int(n_nodes * pct / 100)
    sel_mae = np.mean(sorted_abs_errors[:n_keep])
    reduction_pct = (1 - sel_mae / baseline_mae) * 100
    check(f"Selective reduction % @ {pct}%", reduction_pct, expected_red, tol=0.2)


# ============================================================
# SECTION 3: CRPS — recompute from raw arrays (Gaussian closed-form)
# ============================================================
section("3. CRPS — Recomputed from NPZ (Gaussian closed-form)")

# CRPS for Gaussian: CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
# where z = (y - mu) / sigma, Phi=CDF, phi=PDF
z = (y_true - y_pred_mean) / y_pred_std
phi_z = sp_stats.norm.pdf(z)
Phi_z = sp_stats.norm.cdf(z)
crps_per_node = y_pred_std * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))

crps_mean = np.mean(crps_per_node)
crps_median = np.median(crps_per_node)
crps_over_mae = crps_mean / mae

with open(str(ARTIFACT_DIR / "crps_t8.json"), "r") as f:
    crps_ref = json.load(f)

print(f"  Recomputed CRPS mean:   {crps_mean:.6f}")
print(f"  Recomputed CRPS median: {crps_median:.6f}")
print(f"  Recomputed CRPS/MAE:    {crps_over_mae:.6f}")
print()

check("CRPS mean (recomputed vs JSON)", crps_mean, crps_ref["crps_mean"], tol=0.002)
check(
    "CRPS median (recomputed vs JSON)", crps_median, crps_ref["crps_median"], tol=0.002
)
check(
    "CRPS/MAE ratio (recomputed vs JSON)",
    crps_over_mae,
    crps_ref["crps_over_mae"],
    tol=0.002,
)

# Thesis-cited values
check("CRPS mean (recomputed vs thesis 3.383)", crps_mean, 3.383, tol=0.002)
check("CRPS/MAE (recomputed vs thesis 0.857)", crps_over_mae, 0.857, tol=0.002)


# ============================================================
# SECTION 4: Temperature Scaling Verification
# ============================================================
section("4. Temperature Scaling — JSON Artifact Verification")

with open(str(ARTIFACT_DIR / "temperature_scaling_t8.json"), "r") as f:
    temp = json.load(f)

T_opt = temp["optimal_temperature_T"]
ece_before = temp["evaluation_set"]["ece_before"]
ece_after = temp["evaluation_set"]["ece_after"]
ece_improvement = temp["evaluation_set"]["ece_improvement_pct"]

print(f"  Optimal T:          {T_opt:.6f}")
print(f"  ECE before (eval):  {ece_before:.6f}")
print(f"  ECE after (eval):   {ece_after:.6f}")
print(f"  ECE improvement %:  {ece_improvement:.2f}")
print()

check("Temperature T (JSON vs thesis 2.70)", T_opt, 2.70, tol=0.01)
check("ECE before (JSON vs thesis 0.269)", ece_before, 0.269, tol=0.001)
check("ECE after (JSON vs thesis 0.048)", ece_after, 0.048, tol=0.001)
check("ECE improvement % (JSON vs thesis 82%)", ece_improvement, 82.0, tol=1.0)

# Verify ECE improvement is consistent: (before - after) / before * 100
computed_improvement = (ece_before - ece_after) / ece_before * 100
check("ECE improvement self-consistent", computed_improvement, ece_improvement, tol=0.5)


# ============================================================
# SECTION 5: Conformal Prediction — JSON Artifact Verification
# ============================================================
section("5. Conformal Prediction — JSON Artifact Verification")

with open(str(CONFORMAL_JSON), "r") as f:
    conf = json.load(f)

print(f"  Conformal q90:           {conf['absolute_q_90']:.6f}")
print(f"  Conformal PICP 90:       {conf['absolute_picp_90']:.6f}%")
print(f"  Conformal q95:           {conf['absolute_q_95']:.6f}")
print(f"  Conformal PICP 95:       {conf['absolute_picp_95']:.6f}%")
print(f"  Sigma-scaled k90:        {conf['sigma_k_90']:.6f}")
print(f"  Sigma-scaled PICP 90:    {conf['sigma_picp_90']:.6f}%")
print()

check("Conformal q90 (JSON vs thesis 9.92)", conf["absolute_q_90"], 9.92, tol=0.01)
check(
    "Conformal PICP 90% (JSON vs thesis 90.0)", conf["absolute_picp_90"], 90.0, tol=0.1
)
check("Conformal q95 (JSON vs thesis 14.68)", conf["absolute_q_95"], 14.68, tol=0.01)
check(
    "Conformal PICP 95% (JSON vs thesis 95.01)",
    conf["absolute_picp_95"],
    95.01,
    tol=0.1,
)


# ============================================================
# SECTION 6: PIT Histogram Verification
# ============================================================
section("6. PIT Histogram — JSON Artifact Verification")

with open(str(ARTIFACT_DIR / "pit_t8.json"), "r") as f:
    pit = json.load(f)

with open(str(ARTIFACT_DIR / "pit_after_tempscaling_t8.json"), "r") as f:
    pit_ts = json.load(f)

print(f"  PIT mean (raw):          {pit['pit_mean']:.6f}")
print(f"  PIT KS stat (raw):       {pit['ks_test_subsample']['ks_stat']:.6f}")
print(f"  PIT first bin (raw):     {pit['histogram_density'][0]:.6f}")
print(f"  PIT KS stat (after TS):  {pit_ts['after_tempscaling']['ks_stat']:.6f}")
print(
    f"  PIT first bin (after TS):{pit_ts['after_tempscaling']['first_bin_density']:.6f}"
)
print()

check("PIT mean raw (JSON vs thesis 0.4331)", pit["pit_mean"], 0.4331, tol=1e-4)
check(
    "PIT KS stat raw (JSON vs thesis 0.245)",
    pit["ks_test_subsample"]["ks_stat"],
    0.245,
    tol=1e-3,
)
check(
    "PIT first bin raw (JSON vs thesis 0.2839)",
    pit["histogram_density"][0],
    0.2839,
    tol=1e-4,
)
check(
    "PIT KS stat after TS (JSON vs thesis 0.104)",
    pit_ts["after_tempscaling"]["ks_stat"],
    0.104,
    tol=1e-3,
)
check(
    "PIT KS reduction % (JSON vs thesis 57.4)",
    pit_ts["comparison"]["ks_stat_reduction_pct"],
    57.4,
    tol=0.5,
)
check(
    "PIT first bin after TS (JSON vs thesis 0.0879)",
    pit_ts["after_tempscaling"]["first_bin_density"],
    0.0879,
    tol=1e-4,
)

# Also recompute PIT values from raw NPZ
print("\n  --- PIT recomputed from raw arrays ---")
pit_values = sp_stats.norm.cdf(y_true, loc=y_pred_mean, scale=y_pred_std)
pit_mean_recomp = np.mean(pit_values)
print(f"  PIT mean (recomputed):  {pit_mean_recomp:.6f}")
check("PIT mean (recomputed vs JSON)", pit_mean_recomp, pit["pit_mean"], tol=1e-3)

# PIT KS test (subsample for tractability)
rng = np.random.RandomState(42)
pit_sub = rng.choice(pit_values, size=min(100000, len(pit_values)), replace=False)
ks_stat_recomp, ks_p_recomp = sp_stats.kstest(pit_sub, "uniform")
print(f"  PIT KS stat (recomputed, 100k subsample): {ks_stat_recomp:.6f}")
check(
    "PIT KS stat (recomputed vs JSON)",
    ks_stat_recomp,
    pit["ks_test_subsample"]["ks_stat"],
    tol=0.01,
)


# ============================================================
# SECTION 7: Winkler Scores — JSON Artifact Verification
# ============================================================
section("7. Winkler Scores — JSON Artifact Verification")

with open(str(ARTIFACT_DIR / "winkler_t8.json"), "r") as f:
    wink = json.load(f)

w90_gauss = wink["intervals"]["90pct"]["gaussian"]["mean_winkler"]
w90_conf_sigma = wink["intervals"]["90pct"]["conformal_sigma_scaled"]["mean_winkler"]

print(f"  Winkler 90% Gaussian:           {w90_gauss:.4f}")
print(f"  Winkler 90% Conformal sigma:    {w90_conf_sigma:.4f}")
print()

check("Winkler 90% Gaussian (JSON vs thesis 49.7)", w90_gauss, 49.7, tol=0.1)
check(
    "Winkler 90% Conformal sigma (JSON vs thesis 32.3)", w90_conf_sigma, 32.3, tol=0.1
)


# ============================================================
# SECTION 8: S-Convergence — JSON Artifact Verification
# ============================================================
section("8. S-Convergence — JSON Artifact Verification")

with open(str(ARTIFACT_DIR / "s_convergence_results.json"), "r") as f:
    sconv = json.load(f)

s_rho = {}
for entry in sconv["aggregate_convergence"]:
    s_rho[entry["S"]] = entry["spearman_rho"]

print(f"  S-convergence Spearman rho by S:")
for s_val in sorted(s_rho.keys()):
    print(f"    S={s_val:<4}: rho={s_rho[s_val]:.6f}")
print()

if 30 in s_rho and 50 in s_rho:
    check("S-conv S=30 rho (JSON vs thesis 0.4584)", s_rho[30], 0.4584, tol=1e-3)
    check("S-conv S=50 rho (JSON vs thesis 0.4632)", s_rho[50], 0.4632, tol=1e-3)
    diff_pct = abs(s_rho[50] - s_rho[30]) / s_rho[30] * 100
    print(f"  S30->S50 improvement: {diff_pct:.4f}%")
    check("S-conv: <1.5% improvement S30->S50", diff_pct, 0.0, tol=1.5)


# ============================================================
# SECTION 9: Conditional Coverage — JSON Artifact Verification
# ============================================================
section("9. Conditional Coverage (Adaptive Conformal) — JSON Verification")

with open(str(ARTIFACT_DIR / "conformal_conditional_coverage_t8.json"), "r") as f:
    cond_cov = json.load(f)

d1 = cond_cov["sigma_deciles"][0]
d10 = cond_cov["sigma_deciles"][9]

print(f"  Decile 1 adaptive coverage 90%:  {d1['adaptive_coverage_90'] * 100:.2f}%")
print(f"  Decile 10 adaptive coverage 90%: {d10['adaptive_coverage_90'] * 100:.2f}%")
print(f"  Decile 1 global coverage 90%:    {d1['global_coverage_90'] * 100:.2f}%")
print(f"  Decile 10 global coverage 90%:   {d10['global_coverage_90'] * 100:.2f}%")
print()

check(
    "Adaptive conformal D1 cov 90% (vs thesis 90.0)",
    d1["adaptive_coverage_90"] * 100,
    90.0,
    tol=0.5,
)
check(
    "Adaptive conformal D10 cov 90% (vs thesis 96.2)",
    d10["adaptive_coverage_90"] * 100,
    96.2,
    tol=0.5,
)
check(
    "Global conformal D1 cov 90% (vs thesis 98.6)",
    d1["global_coverage_90"] * 100,
    98.6,
    tol=0.5,
)
check(
    "Global conformal D10 cov 90% (vs thesis 62.9)",
    d10["global_coverage_90"] * 100,
    62.9,
    tol=0.5,
)


# ============================================================
# SECTION 10: Load ALL verified JSON files and print summary
# ============================================================
section("10. All Verified JSON Artifacts — Contents Summary")

json_files = sorted(ARTIFACT_DIR.glob("*.json"))
print(f"  Found {len(json_files)} JSON files in {ARTIFACT_DIR}\n")

for jf in json_files:
    try:
        with open(str(jf), "r", encoding="utf-8") as f:
            data = json.load(f)
        # Print top-level keys and select scalar values
        print(f"  --- {jf.name} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float, str, bool)):
                    print(f"    {k}: {v}")
                elif isinstance(v, list) and len(v) <= 5:
                    print(f"    {k}: {v}")
                elif isinstance(v, list):
                    print(f"    {k}: [{len(v)} items]")
                elif isinstance(v, dict):
                    print(f"    {k}: {{...}} ({len(v)} keys)")
                else:
                    print(f"    {k}: <{type(v).__name__}>")
        elif isinstance(data, list):
            print(f"    [list of {len(data)} items]")
        print()
    except Exception as e:
        print(f"  [ERROR] Failed to load {jf.name}: {e}")
        print()


# ============================================================
# SECTION 11: Verify all 32 figure PDF files exist
# ============================================================
section("11. Figure PDF Files — Existence Check")

EXPECTED_FIGURES = [
    "fig1_trial_comparison.pdf",
    "fig2_uq_ranking.pdf",
    "fig3_conformal_coverage.pdf",
    "fig3_feature_distributions.pdf",
    "fig5_feature_correlation.pdf",
    "fig6_with_without_uq.pdf",
    "fig7_calibration.pdf",
    "fig8_architecture.pdf",
    "fig9_policy_explanation.pdf",
    "fig10_node_vs_graph.pdf",
    "fig11_thesis_workflow.pdf",
    "fig12_trial_progression.pdf",
    "fig13_mc_dropout_inference.pdf",
    "fig14_conformal_workflow.pdf",
    "fig_network_intro.pdf",
    "pointnet_data_flow.pdf",
    "t7_calibration_curve.pdf",
    "t7_interval_width_comparison.pdf",
    "t7_selective_prediction_curve.pdf",
    "t7_vs_t8_uq_comparison.pdf",
    "t8_calibration_curve.pdf",
    "t8_conformal_conditional.pdf",
    "t8_error_detection_auroc.pdf",
    "t8_interval_width_comparison.pdf",
    "t8_per_graph_variation.pdf",
    "t8_pit_after_tempscaling.pdf",
    "t8_pit_histogram.pdf",
    "t8_reliability_diagram.pdf",
    "t8_s_convergence.pdf",
    "t8_selective_prediction_curve.pdf",
    "t8_stratified_uq.pdf",
    "t8_temperature_scaling.pdf",
]

# Also discover any extra PDFs
actual_pdfs = sorted([f.name for f in FIGURE_DIR.glob("*.pdf")])
expected_set = set(EXPECTED_FIGURES)
actual_set = set(actual_pdfs)

n_found = 0
n_missing = 0
for fig in EXPECTED_FIGURES:
    exists = (FIGURE_DIR / fig).exists()
    status = "[FOUND]" if exists else "[MISSING] ***"
    print(f"  {status}  {fig}")
    if exists:
        n_found += 1
    else:
        n_missing += 1

extra = actual_set - expected_set
if extra:
    print(f"\n  Extra PDFs not in expected list ({len(extra)}):")
    for e in sorted(extra):
        print(f"    [EXTRA] {e}")

print(
    f"\n  Summary: {n_found}/{len(EXPECTED_FIGURES)} expected figures found, {n_missing} missing"
)
print(f"  Total PDFs in figures dir: {len(actual_pdfs)}")


# ============================================================
# SECTION 12: Reliability Diagram & Stratified UQ — JSON Verification
# ============================================================
section("12. Reliability Diagram & Stratified UQ — JSON Verification")

with open(str(ARTIFACT_DIR / "reliability_diagram_t8.json"), "r") as f:
    rel_diag = json.load(f)

with open(str(ARTIFACT_DIR / "stratified_uq_t8.json"), "r") as f:
    strat = json.load(f)

print(f"  Reliability diagram:")
if isinstance(rel_diag, dict):
    for k, v in rel_diag.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, list) and len(v) <= 10:
            print(f"    {k}: {v}")
        elif isinstance(v, list):
            print(f"    {k}: [{len(v)} items]")
        elif isinstance(v, dict):
            print(f"    {k}: {{...}} ({len(v)} keys)")

print(f"\n  Stratified UQ:")
if isinstance(strat, dict):
    for k, v in strat.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, list) and len(v) <= 5:
            print(f"    {k}: {v}")
        elif isinstance(v, list):
            print(f"    {k}: [{len(v)} items]")
        elif isinstance(v, dict):
            print(f"    {k}: {{...}} ({len(v)} keys)")


# ============================================================
# SECTION 13: Bootstrap CI & NLL — JSON Verification
# ============================================================
section("13. Bootstrap CI & NLL — JSON Verification")

try:
    with open(str(ARTIFACT_DIR / "bootstrap_ci_results.json"), "r") as f:
        boot = json.load(f)
    print(f"  Bootstrap CI results:")
    for k, v in boot.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, dict):
            print(f"    {k}:")
            for k2, v2 in v.items():
                print(f"      {k2}: {v2}")
        elif isinstance(v, list) and len(v) <= 5:
            print(f"    {k}: {v}")
        else:
            print(f"    {k}: [{len(v)} items]")
except Exception as e:
    warn(f"bootstrap_ci_results.json: {e}")

print()

try:
    with open(str(ARTIFACT_DIR / "nll_results.json"), "r") as f:
        nll = json.load(f)
    print(f"  NLL results:")
    for k, v in nll.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, dict):
            print(f"    {k}:")
            for k2, v2 in v.items():
                if isinstance(v2, (int, float, str, bool)):
                    print(f"      {k2}: {v2}")
                else:
                    print(f"      {k2}: ...")
except Exception as e:
    warn(f"nll_results.json: {e}")


# ============================================================
# FINAL SUMMARY
# ============================================================
section("FINAL VERIFICATION SUMMARY")

total = n_pass + n_fail
print(f"  Total checks:  {total}")
print(f"  PASSED:        {n_pass}")
print(f"  FAILED:        {n_fail}")
print(f"  WARNINGS:      {n_warn}")
print(f"  Figure PDFs:   {n_found}/{len(EXPECTED_FIGURES)} present")
print()

if n_fail == 0:
    print(f"  *** ALL {total} METRIC CHECKS PASSED ***")
    print(f"  All recomputed values from raw NPZ arrays match JSON artifacts")
    print(f"  and thesis-cited numbers within tolerance.")
else:
    print(f"  *** {n_fail} CHECK(S) FAILED — SEE ABOVE ***")

# Save summary
summary = {
    "total_checks": total,
    "passed": n_pass,
    "failed": n_fail,
    "warnings": n_warn,
    "figures_found": n_found,
    "figures_expected": len(EXPECTED_FIGURES),
    "all_results": all_results,
}
summary_path = ARTIFACT_DIR / "verify_all_metrics_summary.json"
with open(str(summary_path), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved to: {summary_path}")
