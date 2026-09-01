"""
PART 2 AUDIT: Verify all 9 HIGH-risk figures with hardcoded values
against their authoritative JSON/data sources.

Each hardcoded value in figure-generation scripts is checked against
the authoritative JSON artifact. Results printed + saved to JSON.
"""

import json
import os
import math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA8 = os.path.join(
    BASE, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
UQ8 = os.path.join(DATA8, "uq_results")
PHASE3 = os.path.join(BASE, "docs", "verified", "phase3_results")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ------- Load all authoritative sources -------
mc_json = load_json(
    os.path.join(UQ8, "mc_dropout_full_metrics_model8_mc30_100graphs.json")
)
det_json = load_json(os.path.join(UQ8, "deterministic_metrics_100graphs.json"))
conf_json = load_json(os.path.join(UQ8, "conformal_standard.json"))
sel_json = load_json(os.path.join(PHASE3, "selective_prediction_s30.json"))
t7_json = load_json(os.path.join(PHASE3, "t7_error_detection.json"))
ts_json = load_json(os.path.join(PHASE3, "temperature_scaling_t8.json"))
rel_json = load_json(os.path.join(PHASE3, "reliability_diagram_t8.json"))
diag_json = load_json(os.path.join(DATA8, "trial8_uq_diagnostics.json"))
exp_a_json = load_json(
    os.path.join(UQ8, "ensemble_experiments", "experiment_a_fixed_results.json")
)
exp_b_json = load_json(
    os.path.join(UQ8, "ensemble_experiments", "experiment_b_fixed_results.json")
)

# Feature analysis report (text)
feat_path = os.path.join(DATA8, "feature_analysis_plots", "feature_analysis_report.txt")
with open(feat_path, "r") as f:
    feat_text = f.read()

# -------- Check infrastructure --------
checks = []
n_pass = 0
n_fail = 0


def check(fig, name, hardcoded, json_val, tol=0.015, note=""):
    """Compare hardcoded figure value vs JSON source."""
    global n_pass, n_fail
    diff = abs(hardcoded - json_val)
    # For values > 1, use absolute tolerance; for small values, use relative
    if abs(json_val) > 0.1:
        ok = diff / abs(json_val) < tol  # relative tolerance
    else:
        ok = diff < 0.01  # absolute for near-zero values
    status = "PASS" if ok else "FAIL"
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    checks.append(
        {
            "figure": fig,
            "check": name,
            "hardcoded": hardcoded,
            "json_source": json_val,
            "diff": round(diff, 6),
            "status": status,
            "note": note,
        }
    )
    marker = "PASS" if ok else "**FAIL**"
    print(
        f"  [{marker}] {name}: hardcoded={hardcoded}, json={json_val:.6f}, diff={diff:.6f}"
        + (f"  ({note})" if note else "")
    )


# ======================================================================
# FIG 5.1: Trial Comparison (T2-T8)
# ======================================================================
print("=" * 70)
print("FIG 5.1 (fig1_trial_comparison.pdf) - Trial Comparison T2-T8")
print("=" * 70)

# Hardcoded in generate_all_thesis_figures.py:
fig51_r2 = [0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
fig51_mae = [4.33, 5.99, 6.08, 4.24, 4.32, 4.06, 3.96]
fig51_rmse = [8.15, 10.27, 10.15, 7.78, 8.06, 7.53, 7.12]
trials = ["T2", "T3", "T4", "T5", "T6", "T7", "T8"]

# T8 authoritative source
check("Fig 5.1", "T8 R2", 0.5957, det_json["r2"], note="det_metrics_100g")
check("Fig 5.1", "T8 MAE", 3.96, det_json["mae"], note="det_metrics_100g")
check("Fig 5.1", "T8 RMSE", 7.12, det_json["rmse"], note="det_metrics_100g")

# T7 from Exp B individual results
check(
    "Fig 5.1",
    "T7 R2",
    0.5471,
    exp_b_json["individual"]["7"]["r2"],
    note="exp_b_fixed individual T7",
)
check(
    "Fig 5.1",
    "T7 MAE",
    4.06,
    exp_b_json["individual"]["7"]["mae"],
    note="exp_b_fixed individual T7",
)
check(
    "Fig 5.1",
    "T7 RMSE",
    7.53,
    exp_b_json["individual"]["7"]["rmse"],
    note="exp_b_fixed individual T7",
)

# T2 from Exp B
check(
    "Fig 5.1",
    "T2 R2",
    0.5117,
    exp_b_json["individual"]["2"]["r2"],
    note="exp_b_fixed individual T2",
)
check(
    "Fig 5.1",
    "T2 MAE",
    4.33,
    exp_b_json["individual"]["2"]["mae"],
    note="exp_b_fixed individual T2",
)
check(
    "Fig 5.1",
    "T2 RMSE",
    8.15,
    exp_b_json["individual"]["2"]["rmse"],
    note="exp_b_fixed individual T2",
)

# T5 from Exp B
check(
    "Fig 5.1",
    "T5 R2",
    0.5553,
    exp_b_json["individual"]["5"]["r2"],
    note="exp_b_fixed individual T5",
)
check(
    "Fig 5.1",
    "T5 MAE",
    4.24,
    exp_b_json["individual"]["5"]["mae"],
    note="exp_b_fixed individual T5",
)
check(
    "Fig 5.1",
    "T5 RMSE",
    7.78,
    exp_b_json["individual"]["5"]["rmse"],
    note="exp_b_fixed individual T5",
)

# T6 from Exp B
check(
    "Fig 5.1",
    "T6 R2",
    0.5223,
    exp_b_json["individual"]["6"]["r2"],
    note="exp_b_fixed individual T6",
)
check(
    "Fig 5.1",
    "T6 MAE",
    4.32,
    exp_b_json["individual"]["6"]["mae"],
    note="exp_b_fixed individual T6",
)
check(
    "Fig 5.1",
    "T6 RMSE",
    8.06,
    exp_b_json["individual"]["6"]["rmse"],
    note="exp_b_fixed individual T6",
)

# T3, T4 have no individual JSON in exp_b (only T2,T5,T6,T7,T8 used in ensembles)
# Check against feature_analysis_report or TRIALS_SUMMARY
print("  [INFO] T3, T4: No individual JSON available (not used in ensembles).")
print(
    "         T3: R2=0.2246, MAE=5.99, RMSE=10.27 — from TRIALS_SUMMARY.csv (manual check)"
)
print(
    "         T4: R2=0.2426, MAE=6.08, RMSE=10.15 — from TRIALS_SUMMARY.csv (manual check)"
)

# ======================================================================
# FIG 5.2: UQ Ranking (Spearman rho)
# ======================================================================
print("\n" + "=" * 70)
print("FIG 5.2 (fig2_uq_ranking.pdf) - UQ Ranking by Spearman rho")
print("=" * 70)

# Hardcoded: rho = [0.4263, 0.4186, 0.4460, 0.4820, 0.4908, 0.4370, 0.4909, 0.4333]
# T8 standalone MC Dropout
check(
    "Fig 5.2", "T8 MC Dropout rho", 0.4820, mc_json["spearman"], note="mc_dropout_100g"
)

# T7 MC Dropout — from t7_error_detection.json
check(
    "Fig 5.2",
    "T7 MC Dropout rho",
    0.4460,
    t7_json["spearman_rho"],
    note="t7_error_detection.json",
)

# Exp A MC Dropout
check(
    "Fig 5.2",
    "Exp A MC Dropout rho",
    0.4908,
    exp_a_json["mc_dropout"]["spearman_rho"],
    note="experiment_a_fixed_results.json",
)

# Exp A Ensemble Variance
check(
    "Fig 5.2",
    "Exp A Ensemble Var rho",
    0.4370,
    exp_a_json["ensemble_variance"]["spearman_rho"],
    note="experiment_a_fixed_results.json",
)

# Exp A Combined
check(
    "Fig 5.2",
    "Exp A Combined rho",
    0.4909,
    exp_a_json["combined"]["spearman_rho"],
    note="experiment_a_fixed_results.json",
)

# Exp B Multi-Ensemble
check(
    "Fig 5.2",
    "Exp B Multi-Ens rho",
    0.4333,
    exp_b_json["ensemble"]["spearman_rho"],
    note="experiment_b_fixed_results.json",
)

# T5 and T6 MC Dropout — need to load their individual jsons
t5_mc_path = os.path.join(
    BASE,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_5th_try",
    "uq_results",
    "mc_dropout_full_metrics_model5_mc30_50graphs.json",
)
t6_mc_path = os.path.join(
    BASE,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_6th_trial_lower_lr",
    "uq_results",
    "mc_dropout_full_metrics_model6_mc30_50graphs.json",
)

if os.path.exists(t5_mc_path):
    t5_mc = load_json(t5_mc_path)
    check(
        "Fig 5.2",
        "T5 MC Dropout rho",
        0.4263,
        t5_mc["spearman"],
        note="T5 mc_metrics_50g",
    )
else:
    print(f"  [SKIP] T5 MC JSON not found: {t5_mc_path}")

if os.path.exists(t6_mc_path):
    t6_mc = load_json(t6_mc_path)
    check(
        "Fig 5.2",
        "T6 MC Dropout rho",
        0.4186,
        t6_mc["spearman"],
        note="T6 mc_metrics_50g",
    )
else:
    print(f"  [SKIP] T6 MC JSON not found: {t6_mc_path}")

# ======================================================================
# FIG 5.3: Conformal Prediction Coverage
# ======================================================================
print("\n" + "=" * 70)
print("FIG 5.3 (fig3_conformal_coverage.pdf) - Conformal Coverage")
print("=" * 70)

check(
    "Fig 5.3",
    "Achieved coverage 90%",
    90.02,
    conf_json["absolute_picp_90"],
    note="conformal_standard.json",
)
check(
    "Fig 5.3",
    "Achieved coverage 95%",
    95.01,
    conf_json["absolute_picp_95"],
    note="conformal_standard.json",
)
check(
    "Fig 5.3",
    "Interval width q90",
    9.92,
    conf_json["absolute_q_90"],
    note="conformal_standard.json (quantile, not width)",
)
check(
    "Fig 5.3",
    "Interval width q95",
    14.68,
    conf_json["absolute_q_95"],
    note="conformal_standard.json",
)

# ======================================================================
# FIG 5.5: Selective Prediction
# ======================================================================
print("\n" + "=" * 70)
print("FIG 5.5 (fig4_selective_prediction.pdf) - Selective Prediction")
print("=" * 70)

check(
    "Fig 5.5",
    "Baseline MAE (100%)",
    3.95,
    sel_json["baseline_mc_mae"],
    note="selective_prediction_s30.json",
)
# Find 90% and 50% from retention table
for entry in sel_json["retention_table"]:
    if entry["retained_pct"] == 90:
        check(
            "Fig 5.5",
            "MAE @ 90% retention",
            3.23,
            entry["MAE"],
            note="selective_prediction_s30.json",
        )
    if entry["retained_pct"] == 50:
        check(
            "Fig 5.5",
            "MAE @ 50% retention",
            2.32,
            entry["MAE"],
            note="selective_prediction_s30.json",
        )

check(
    "Fig 5.5",
    "Reduction @ 90%",
    18.3,
    sel_json["key_reductions"]["retain_90pct"]["mae_reduction_pct"],
    note="selective_prediction_s30.json",
)
check(
    "Fig 5.5",
    "Reduction @ 50%",
    41.2,
    sel_json["key_reductions"]["retain_50pct"]["mae_reduction_pct"],
    note="selective_prediction_s30.json",
)

# ======================================================================
# FIG 5.7 (fig5_feature_correlation.pdf) - Feature Correlation
# ======================================================================
print("\n" + "=" * 70)
print("FIG 5.7 (fig5_feature_correlation.pdf) - Feature Correlation")
print("=" * 70)

# From feature_analysis_report.txt:
# VOL_BASE_CASE        +0.3316
# CAPACITY_BASE_CASE   +0.2615
# CAPACITY_REDUCTION   -0.2286
# FREESPEED            +0.2110
# LENGTH               -0.0695
import re

feat_correlations = {}
for line in feat_text.splitlines():
    m = re.match(
        r"^\s*(VOL_BASE_CASE|CAPACITY_BASE_CASE|CAPACITY_REDUCTION|FREESPEED|LENGTH)\s+([+-]?\d+\.\d+)",
        line,
    )
    if m:
        feat_correlations[m.group(1)] = float(m.group(2))

# Hardcoded in figure: [0.332, 0.262, 0.211, -0.229, -0.070]
# Order: VOL, CAP, SPD, CAP_RED, LEN
check(
    "Fig 5.7",
    "VOL correlation",
    0.332,
    feat_correlations.get("VOL_BASE_CASE", 0),
    note="feature_analysis_report.txt",
)
check(
    "Fig 5.7",
    "CAP correlation",
    0.262,
    feat_correlations.get("CAPACITY_BASE_CASE", 0),
    note="feature_analysis_report.txt",
)
check(
    "Fig 5.7",
    "SPD correlation",
    0.211,
    feat_correlations.get("FREESPEED", 0),
    note="feature_analysis_report.txt",
)
check(
    "Fig 5.7",
    "CAP_RED correlation",
    -0.229,
    feat_correlations.get("CAPACITY_REDUCTION", 0),
    note="feature_analysis_report.txt",
)
check(
    "Fig 5.7",
    "LEN correlation",
    -0.070,
    feat_correlations.get("LENGTH", 0),
    note="feature_analysis_report.txt",
)

# ======================================================================
# FIG 5.20: Deterministic vs MC Dropout
# ======================================================================
print("\n" + "=" * 70)
print("FIG 5.20 (fig6_with_without_uq.pdf) - Det vs MC Dropout")
print("=" * 70)

# Hardcoded: det_r2=0.5957, mc_r2=0.5857, det_mae=3.96, mc_mae=3.948, det_rmse=7.12, mc_rmse=7.207
check("Fig 5.20", "Det R2", 0.5957, det_json["r2"], note="det_metrics_100g")
check("Fig 5.20", "Det MAE", 3.96, det_json["mae"], note="det_metrics_100g")
check("Fig 5.20", "Det RMSE", 7.12, det_json["rmse"], note="det_metrics_100g")
check("Fig 5.20", "MC R2", 0.5857, mc_json["r2"], note="mc_metrics_100g")
check("Fig 5.20", "MC MAE", 3.948, mc_json["mae"], note="mc_metrics_100g")
check("Fig 5.20", "MC RMSE", 7.207, mc_json["rmse"], note="mc_metrics_100g")

# Delta checks
det_r2, mc_r2 = 0.5957, 0.5857
det_mae, mc_mae = 3.96, 3.948
det_rmse, mc_rmse = 7.12, 7.207
print(f"  [INFO] Delta R2: {mc_r2 - det_r2:.4f} (hardcoded: -0.010)")
print(f"  [INFO] Delta MAE: {mc_mae - det_mae:.4f} (hardcoded: -0.012)")
print(f"  [INFO] Delta RMSE: {mc_rmse - det_rmse:.4f} (hardcoded: +0.087)")

# ======================================================================
# FIG 6.2 (fig7_calibration.pdf) - Calibration k95
# ======================================================================
print("\n" + "=" * 70)
print("FIG 6.2 (fig7_calibration.pdf) - Calibration k95")
print("=" * 70)

# Hardcoded: k95 = [1.96, 11.34]
check(
    "Fig 6.2",
    "k95 (MC Dropout)",
    11.34,
    diag_json["calibration_factors"]["k_95"],
    note="trial8_uq_diagnostics.json",
)
check(
    "Fig 6.2",
    "k95 (Gaussian ideal)",
    1.96,
    1.959963984540054,
    note="scipy.stats.norm.ppf(0.975) = 1.95996",
)

# Also cross-check k95 against temperature_scaling_t8.json
check(
    "Fig 6.2",
    "k95 cross-check (temp_scaling)",
    11.34,
    ts_json["conformal_k95_for_comparison"],
    note="temperature_scaling_t8.json",
)

# Also check against t7_error_detection.json t8_comparison
check(
    "Fig 6.2",
    "k95 cross-check (t7_err_det)",
    11.34,
    t7_json["t8_comparison"]["k95"],
    note="t7_error_detection.json t8_comparison",
)

# ======================================================================
# FIG 3.5 (fig12_trial_progression.pdf) - Trial Progression T1-T8
# ======================================================================
print("\n" + "=" * 70)
print("FIG 3.5 (fig12_trial_progression.pdf) - Trial Progression T1-T8")
print("=" * 70)

# Hardcoded: r2 = [0.7860, 0.5117, 0.2246, 0.2426, 0.5553, 0.5223, 0.5471, 0.5957]
#            mae = [2.97,   4.33,   5.99,   6.08,   4.24,   4.32,   4.06,   3.96]
# T8 (same as Fig 5.1)
check("Fig 3.5", "T8 R2", 0.5957, det_json["r2"], note="det_metrics_100g")
check("Fig 3.5", "T8 MAE", 3.96, det_json["mae"], note="det_metrics_100g")
# T1: check if eval_metrics_recomputed.json exists
t1_path = os.path.join(
    BASE,
    "data",
    "TR-C_Benchmarks",
    "pointnet_transf_gat_1st_bs32_5feat_seed42",
    "eval_metrics_recomputed.json",
)
if os.path.exists(t1_path):
    t1_json = load_json(t1_path)
    check(
        "Fig 3.5",
        "T1 R2",
        0.7860,
        t1_json.get("r2", 0),
        note="T1 eval_metrics_recomputed",
    )
    check(
        "Fig 3.5",
        "T1 MAE",
        2.97,
        t1_json.get("mae", 0),
        note="T1 eval_metrics_recomputed",
    )
else:
    # Try test_evaluation_complete.json
    t1_alt = os.path.join(
        BASE,
        "data",
        "TR-C_Benchmarks",
        "pointnet_transf_gat_1st_bs32_5feat_seed42",
        "test_evaluation_complete.json",
    )
    if os.path.exists(t1_alt):
        t1_json = load_json(t1_alt)
        check(
            "Fig 3.5",
            "T1 R2",
            0.7860,
            t1_json.get("r2", 0),
            note="T1 test_eval_complete",
        )
        check(
            "Fig 3.5",
            "T1 MAE",
            2.97,
            t1_json.get("mae", 0),
            note="T1 test_eval_complete",
        )
    else:
        print(f"  [SKIP] T1 JSON not found at {t1_path} or {t1_alt}")

# Reference lines hardcoded: ax1.axhline(y=0.5957), ax2.axhline(y=3.96) — same as T8, already checked

# ======================================================================
# FIG 3.6 (fig11_thesis_workflow.pdf) - Thesis Workflow
# ======================================================================
print("\n" + "=" * 70)
print("FIG 3.6 (fig11_thesis_workflow.pdf) - Thesis Workflow Diagram")
print("=" * 70)

# Hardcoded in box labels: R2=0.5957, MAE=3.96, rho=0.4820
check("Fig 3.6", "T8 R2 in workflow", 0.5957, det_json["r2"], note="det_metrics_100g")
check("Fig 3.6", "T8 MAE in workflow", 3.96, det_json["mae"], note="det_metrics_100g")
check(
    "Fig 3.6", "MC rho in workflow", 0.4820, mc_json["spearman"], note="mc_metrics_100g"
)

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print(f"HARDCODED FIGURE AUDIT SUMMARY: {n_pass} PASS, {n_fail} FAIL")
print("=" * 70)

# Print any failures
fails = [c for c in checks if c["status"] == "FAIL"]
if fails:
    print("\nFAILED CHECKS:")
    for f in fails:
        print(
            f"  {f['figure']} / {f['check']}: hardcoded={f['hardcoded']}, json={f['json_source']}"
        )
else:
    print("\nNo failures detected.")

# Save to JSON
out_path = os.path.join(PHASE3, "audit_hardcoded_figures.json")
summary = {
    "total_checks": len(checks),
    "pass": n_pass,
    "fail": n_fail,
    "checks": checks,
}
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to: {out_path}")
