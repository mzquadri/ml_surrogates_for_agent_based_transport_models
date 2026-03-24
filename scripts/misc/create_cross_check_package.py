#!/usr/bin/env python3
"""
Create cross-check verification package.
Contains: all JSON artifacts, all HD plots, and a summary.
"""

import json
import os
import zipfile
import glob

REPO = r"C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\ml_surrogates_for_agent_based_transport_models"
PHASE3 = os.path.join(REPO, "docs", "verified", "phase3_results")
UQ_DIR = os.path.join(
    REPO,
    "data",
    "TR-C_Benchmarks",
    "point_net_transf_gat_8th_trial_lower_dropout",
    "uq_results",
)
T8_DIR = os.path.join(
    REPO, "data", "TR-C_Benchmarks", "point_net_transf_gat_8th_trial_lower_dropout"
)
PLOTS_DIR = os.path.join(REPO, "docs", "hd_plots")
OUT_ZIP = os.path.join(REPO, "cross_check_package.zip")

print("Creating cross-check verification package...")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. All phase3 JSONs
    for f in sorted(glob.glob(os.path.join(PHASE3, "*.json"))):
        arcname = f"json_artifacts/phase3_results/{os.path.basename(f)}"
        zf.write(f, arcname)
        print(f"  + {arcname}")

    # 2. UQ results JSONs
    for f in sorted(glob.glob(os.path.join(UQ_DIR, "*.json"))):
        arcname = f"json_artifacts/uq_results/{os.path.basename(f)}"
        zf.write(f, arcname)
        print(f"  + {arcname}")

    # 3. test_evaluation_complete.json
    tec = os.path.join(T8_DIR, "test_evaluation_complete.json")
    if os.path.exists(tec):
        zf.write(tec, "json_artifacts/test_evaluation_complete.json")
        print(f"  + json_artifacts/test_evaluation_complete.json")

    # 4. All HD plots (PNG only for size)
    for f in sorted(glob.glob(os.path.join(PLOTS_DIR, "*.png"))):
        arcname = f"hd_plots/{os.path.basename(f)}"
        zf.write(f, arcname)
        print(f"  + {arcname}")

    # 5. All HD plots (PDF)
    for f in sorted(glob.glob(os.path.join(PLOTS_DIR, "*.pdf"))):
        arcname = f"hd_plots_pdf/{os.path.basename(f)}"
        zf.write(f, arcname)
        print(f"  + {arcname}")

    # 6. Plotting script for reproducibility
    script = os.path.join(REPO, "scripts", "generate_all_hd_plots.py")
    if os.path.exists(script):
        zf.write(script, "scripts/generate_all_hd_plots.py")
        print(f"  + scripts/generate_all_hd_plots.py")

print(f"\nPackage created: {OUT_ZIP}")
print(f"Size: {os.path.getsize(OUT_ZIP) / (1024 * 1024):.1f} MB")

# Count files
with zipfile.ZipFile(OUT_ZIP, "r") as zf:
    print(f"Total files: {len(zf.namelist())}")
