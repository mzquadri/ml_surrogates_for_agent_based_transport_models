"""Verify Fig 5.14 values: AUROC and AUPRC for error detection."""

import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parent.parent.parent
CSV = (
    REPO
    / "data"
    / "TR-C_Benchmarks"
    / "point_net_transf_gat_8th_trial_lower_dropout"
    / "trial8_uq_ablation_results.csv"
)
JSON_T7 = REPO / "docs" / "verified" / "phase3_results" / "t7_error_detection.json"

print("Loading CSV...")
df = pd.read_csv(CSV)
print(f"  Rows: {len(df):,}")

sigma = df["pred_mc_std"].values.astype(np.float64)
errors = df["abs_error_det"].values.astype(np.float64)

p90 = float(np.percentile(errors, 90))
p80 = float(np.percentile(errors, 80))

print(f"\n  Top-10% cutoff (p90): {p90:.4f} veh/h")
print(f"  Top-20% cutoff (p80): {p80:.4f} veh/h")

# Top-10%
labels_10 = (errors >= p90).astype(int)
auroc_10 = float(roc_auc_score(labels_10, sigma))
auprc_10 = float(average_precision_score(labels_10, sigma))

# Top-20%
labels_20 = (errors >= p80).astype(int)
auroc_20 = float(roc_auc_score(labels_20, sigma))
auprc_20 = float(average_precision_score(labels_20, sigma))

print(f"\n=== COMPUTED FROM CSV ===")
print(f"  AUROC top-10%: {auroc_10:.4f}")
print(f"  AUROC top-20%: {auroc_20:.4f}")
print(f"  AUPRC top-10%: {auprc_10:.4f}")
print(f"  AUPRC top-20%: {auprc_20:.4f}")

# Load reference from JSON (t8_comparison inside t7_error_detection.json)
with open(JSON_T7) as f:
    ref = json.load(f)
t8_ref = ref["t8_comparison"]

print(f"\n=== REFERENCE (t7_error_detection.json -> t8_comparison) ===")
print(f"  AUROC top-10%: {t8_ref['auroc_top_10pct']}")
print(f"  AUROC top-20%: {t8_ref['auroc_top_20pct']}")

# Cross-check
checks = [
    ("AUROC top-10%", auroc_10, t8_ref["auroc_top_10pct"], 0.001),
    ("AUROC top-20%", auroc_20, t8_ref["auroc_top_20pct"], 0.001),
]

print(f"\n=== CROSS-CHECK ===")
all_pass = True
for name, computed, expected, tol in checks:
    match = abs(computed - expected) < tol
    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
    print(f"  {name}: computed={computed:.4f} expected={expected} -> {status}")

# Also print AUPRC (no reference in JSON t8_comparison, but figure shows 0.315 and 0.455)
print(f"\n  AUPRC top-10%: {auprc_10:.4f} (figure shows 0.315)")
print(f"  AUPRC top-20%: {auprc_20:.4f} (figure shows 0.455)")
auprc_10_match = abs(auprc_10 - 0.315) < 0.002
auprc_20_match = abs(auprc_20 - 0.455) < 0.002
print(f"  AUPRC top-10% match: {'PASS' if auprc_10_match else 'FAIL'}")
print(f"  AUPRC top-20% match: {'PASS' if auprc_20_match else 'FAIL'}")
if not (auprc_10_match and auprc_20_match):
    all_pass = False

print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
