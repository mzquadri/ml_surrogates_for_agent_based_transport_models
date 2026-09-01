"""
PART 4 AUDIT: Data integrity check for all NPZ and JSON files.
Confirms files exist, are readable, and have expected shapes/keys.
"""

import json
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA8 = os.path.join(
    BASE, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
UQ8 = os.path.join(DATA8, "uq_results")
PHASE3 = os.path.join(BASE, "docs", "verified", "phase3_results")
FIGURES = os.path.join(BASE, "thesis", "latex_tum_official", "figures")

n_pass = 0
n_fail = 0
checks = []


def check(name, passed, detail=""):
    global n_pass, n_fail
    status = "PASS" if passed else "FAIL"
    if passed:
        n_pass += 1
    else:
        n_fail += 1
    checks.append({"check": name, "status": status, "detail": detail})
    marker = "PASS" if passed else "**FAIL**"
    print(f"  [{marker}] {name}" + (f"  ({detail})" if detail else ""))


# ======================================================================
print("=" * 70)
print("1. NPZ DATA FILES — Integrity Check")
print("=" * 70)

# MC Dropout NPZ (the primary data source)
mc_npz_path = os.path.join(UQ8, "mc_dropout_full_100graphs_mc30.npz")
try:
    mc_data = np.load(mc_npz_path)
    keys = list(mc_data.keys())
    check("MC Dropout NPZ exists", True, f"keys={keys}")
    check(
        "MC Dropout NPZ predictions shape",
        mc_data["predictions"].shape == (3163500,),
        f"shape={mc_data['predictions'].shape}",
    )
    check(
        "MC Dropout NPZ uncertainties shape",
        mc_data["uncertainties"].shape == (3163500,),
        f"shape={mc_data['uncertainties'].shape}",
    )
    check(
        "MC Dropout NPZ targets shape",
        mc_data["targets"].shape == (3163500,),
        f"shape={mc_data['targets'].shape}",
    )
    check(
        "MC Dropout NPZ no NaN in predictions",
        not np.any(np.isnan(mc_data["predictions"])),
        "",
    )
    check(
        "MC Dropout NPZ no NaN in uncertainties",
        not np.any(np.isnan(mc_data["uncertainties"])),
        "",
    )
    check(
        "MC Dropout NPZ no NaN in targets", not np.any(np.isnan(mc_data["targets"])), ""
    )
    check(
        "MC Dropout NPZ all uncertainties > 0",
        np.all(mc_data["uncertainties"] > 0),
        f"min={mc_data['uncertainties'].min():.6f}",
    )
except Exception as e:
    check("MC Dropout NPZ exists", False, str(e))

# Deterministic NPZ
det_npz_path = os.path.join(UQ8, "deterministic_full_100graphs.npz")
try:
    det_data = np.load(det_npz_path)
    keys = list(det_data.keys())
    check("Deterministic NPZ exists", True, f"keys={keys}")
    pred_key = "predictions" if "predictions" in keys else keys[0]
    target_key = "targets" if "targets" in keys else keys[1]
    check(
        "Deterministic NPZ predictions shape",
        det_data[pred_key].shape[0] == 3163500,
        f"shape={det_data[pred_key].shape}",
    )
    check("Deterministic NPZ no NaN", not np.any(np.isnan(det_data[pred_key])), "")
except Exception as e:
    check("Deterministic NPZ exists", False, str(e))

# S-convergence NPZ
s_conv_path = os.path.join(PHASE3, "s_convergence_raw.npz")
try:
    s_data = np.load(s_conv_path)
    check("S-convergence NPZ exists", True, f"keys={list(s_data.keys())}")
except Exception as e:
    check("S-convergence NPZ exists", False, str(e))

# Ensemble Experiment A NPZ files (5 runs)
for i in range(5):
    ea_path = os.path.join(UQ8, "ensemble_experiments", f"exp_a_run_{i}.npz")
    try:
        ea_data = np.load(ea_path)
        check(f"Exp A run {i} NPZ exists", True, f"keys={list(ea_data.keys())}")
    except Exception as e:
        check(f"Exp A run {i} NPZ exists", False, str(e))

# ======================================================================
print("\n" + "=" * 70)
print("2. JSON ARTIFACT FILES — Integrity Check")
print("=" * 70)

# Phase 3 results
phase3_jsons = [
    "selective_prediction_s30.json",
    "temperature_scaling_t8.json",
    "reliability_diagram_t8.json",
    "crps_t8.json",
    "pit_t8.json",
    "pit_after_tempscaling_t8.json",
    "winkler_t8.json",
    "s_convergence_results.json",
    "conformal_conditional_coverage_t8.json",
    "per_graph_variation_t8.json",
    "stratified_uq_t8.json",
    "t7_error_detection.json",
    "nll_results.json",
    "bootstrap_ci_results.json",
    "ensemble_bug_root_cause.json",
    "ensemble_bug_diagnostic.json",
    "final_numeric_verification.json",
    "verify_all_metrics_summary.json",
]

for jf in phase3_jsons:
    jp = os.path.join(PHASE3, jf)
    try:
        with open(jp, "r") as f:
            data = json.load(f)
        check(f"phase3/{jf}", True, f"{len(data)} top-level keys")
    except Exception as e:
        check(f"phase3/{jf}", False, str(e))

# UQ results JSONs
uq_jsons = [
    "mc_dropout_full_metrics_model8_mc30_100graphs.json",
    "deterministic_metrics_100graphs.json",
    "conformal_standard.json",
]
for jf in uq_jsons:
    jp = os.path.join(UQ8, jf)
    try:
        with open(jp, "r") as f:
            data = json.load(f)
        check(f"uq_results/{jf}", True, f"{len(data)} top-level keys")
    except Exception as e:
        check(f"uq_results/{jf}", False, str(e))

# Ensemble JSONs
ens_jsons = [
    "experiment_a_fixed_results.json",
    "experiment_b_fixed_results.json",
]
for jf in ens_jsons:
    jp = os.path.join(UQ8, "ensemble_experiments", jf)
    try:
        with open(jp, "r") as f:
            data = json.load(f)
        check(f"ensemble/{jf}", True, f"{len(data)} top-level keys")
    except Exception as e:
        check(f"ensemble/{jf}", False, str(e))

# ======================================================================
print("\n" + "=" * 70)
print("3. FIGURE PDF FILES — Existence Check")
print("=" * 70)

expected_figures = [
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

for fig in expected_figures:
    fp = os.path.join(FIGURES, fig)
    exists = os.path.exists(fp)
    size = os.path.getsize(fp) if exists else 0
    check(f"Figure: {fig}", exists, f"size={size:,} bytes" if exists else "MISSING")

# ======================================================================
print("\n" + "=" * 70)
print(
    f"DATA INTEGRITY SUMMARY: {n_pass} PASS, {n_fail} FAIL out of {n_pass + n_fail} checks"
)
print("=" * 70)

fails = [c for c in checks if c["status"] == "FAIL"]
if fails:
    print("\nFAILED CHECKS:")
    for f in fails:
        print(f"  {f['check']}: {f['detail']}")
else:
    print("\nAll data files intact. No failures.")

out_path = os.path.join(PHASE3, "audit_data_integrity.json")
with open(out_path, "w") as f:
    json.dump(
        {"total": n_pass + n_fail, "pass": n_pass, "fail": n_fail, "checks": checks},
        f,
        indent=2,
    )
print(f"\nResults saved to: {out_path}")
