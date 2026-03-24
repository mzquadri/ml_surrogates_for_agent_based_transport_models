"""
run_phase5_all_scripts.py
=========================
Run all 11 individual figure-generation scripts for Phase 5.
Uses subprocess with PYTHONUTF8=1 to avoid conda cp1252 crashes on Windows.
Logs each script's output to a separate log file.
"""

import os
import sys
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIGURES_DIR = os.path.join(REPO, "scripts", "figure_generation")
EVAL_DIR = os.path.join(REPO, "scripts", "evaluation")
PYTHON = sys.executable

env = os.environ.copy()
env["PYTHONUTF8"] = "1"

# Each entry: (script_path, working_directory, description)
scripts = [
    # --- Scripts in scripts/figure_generation/ ---
    (
        os.path.join(FIGURES_DIR, "generate_network_intro_figure.py"),
        FIGURES_DIR,
        "fig_network_intro",
    ),
    (
        os.path.join(FIGURES_DIR, "generate_pointnet_dataflow_figure.py"),
        FIGURES_DIR,
        "pointnet_data_flow",
    ),
    # --- Scripts in scripts/evaluation/ ---
    (
        os.path.join(EVAL_DIR, "compute_pit.py"),
        REPO,
        "t8_pit_histogram",
    ),
    (
        os.path.join(EVAL_DIR, "compute_pit_after_tempscaling.py"),
        REPO,
        "t8_pit_after_tempscaling",
    ),
    (
        os.path.join(FIGURES_DIR, "generate_s_convergence_figure.py"),
        REPO,
        "t8_s_convergence",
    ),
    (
        os.path.join(FIGURES_DIR, "regenerate_fig58_s30.py"),
        REPO,
        "t8_selective_prediction_curve (S=30)",
    ),
    # --- run_part* scripts now live in scripts/evaluation/ ---
    (
        os.path.join(EVAL_DIR, "run_part2_uq_analyses.py"),
        REPO,
        "t8_error_detection_auroc + t8_selective_prediction_curve",
    ),
    (
        os.path.join(EVAL_DIR, "run_part3_calibration_audit.py"),
        REPO,
        "t8_calibration_curve + t8_interval_width_comparison",
    ),
    (
        os.path.join(EVAL_DIR, "run_part4_t7_crosscheck.py"),
        REPO,
        "t7_selective_prediction + t7_calibration + t7_interval_width",
    ),
]

passed = 0
failed = 0

for i, (script, cwd, desc) in enumerate(scripts, 1):
    name = os.path.basename(script).replace(".py", "")
    log_file = os.path.join(REPO, f"phase5_{name}.log")

    print(f"\n{'=' * 70}")
    print(f"[{i}/{len(scripts)}] {desc}")
    print(f"  Script: {os.path.relpath(script, REPO)}")
    print(f"  CWD:    {os.path.relpath(cwd, REPO) if cwd != REPO else '.'}")
    print(f"{'=' * 70}")

    with open(log_file, "w", encoding="utf-8") as lf:
        result = subprocess.run(
            [PYTHON, script],
            cwd=cwd,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            timeout=600,  # 10 min max per script
        )

    if result.returncode == 0:
        print(f"  SUCCESS (log: {os.path.basename(log_file)})")
        passed += 1
    else:
        print(f"  FAILED (rc={result.returncode})")
        failed += 1
        # Print last 30 lines of log for debugging
        with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
            lines = lf.readlines()
            for line in lines[-30:]:
                print(f"    {line.rstrip()}")

print(f"\n{'=' * 70}")
print(f"Phase 5 complete: {passed} passed, {failed} failed out of {len(scripts)}")
print(f"{'=' * 70}")
