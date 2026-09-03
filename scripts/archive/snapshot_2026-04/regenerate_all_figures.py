#!/usr/bin/env python3
"""Regenerate ALL 32 thesis figures from their respective scripts."""

import sys
import os
import importlib.util
import traceback

# Make sure we can import from the figures directory
FIGURES_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "thesis",
        "latex_tum_official",
        "figures",
    )
)
sys.path.insert(0, FIGURES_DIR)

# Also add repo root for other scripts
REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.insert(0, REPO_ROOT)


def run_module_function(module_path, func_name):
    """Import a module and run a specific function."""
    spec = importlib.util.spec_from_file_location("mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, func_name)
    fn()


def main():
    os.chdir(FIGURES_DIR)

    results = {}

    # === PHASE 1: generate_all_thesis_figures.py (10 figures) ===
    print("=" * 60)
    print("PHASE 1: generate_all_thesis_figures.py")
    print("=" * 60)

    phase1_script = os.path.join(FIGURES_DIR, "generate_all_thesis_figures.py")
    phase1_funcs = [
        ("fig1_trial_comparison", "Fig 5.1"),
        ("fig2_uq_ranking", "Fig 5.3"),
        ("fig3_conformal_coverage", "Fig 5.5"),
        ("fig4_selective_prediction", "Fig 5.8-alt"),
        ("fig3_feature_distributions", "Fig 3.1"),
        ("fig5_feature_correlation", "Fig 5.20"),
        ("fig6_with_without_uq", "Fig 5.2"),
        ("fig7_k95_calibration", "Fig 5.7"),
        ("fig8_architecture", "Fig 3.3"),
        ("fig9_policy_explanation", "Fig 6.2"),
        ("fig10_node_vs_graph", "Fig 3.2"),
    ]

    for func_name, thesis_fig in phase1_funcs:
        print(f"\n--- {thesis_fig}: {func_name} ---")
        try:
            run_module_function(phase1_script, func_name)
            results[thesis_fig] = "OK"
            print(f"  [OK] {thesis_fig}")
        except Exception as e:
            results[thesis_fig] = f"FAIL: {e}"
            print(f"  [FAIL] {thesis_fig}: {e}")
            traceback.print_exc()

    # === PHASE 2: generate_new_figures.py (4 figures) ===
    print("\n" + "=" * 60)
    print("PHASE 2: generate_new_figures.py")
    print("=" * 60)

    phase2_script = os.path.join(FIGURES_DIR, "generate_new_figures.py")
    phase2_funcs = [
        ("fig11_thesis_workflow", "Fig 1.2"),
        ("fig12_trial_progression", "Fig 3.5"),
        ("fig13_mc_dropout_inference", "Fig 3.6"),
        ("fig14_conformal_workflow", "Fig 5.4"),
    ]

    for func_name, thesis_fig in phase2_funcs:
        print(f"\n--- {thesis_fig}: {func_name} ---")
        try:
            run_module_function(phase2_script, func_name)
            results[thesis_fig] = "OK"
            print(f"  [OK] {thesis_fig}")
        except Exception as e:
            results[thesis_fig] = f"FAIL: {e}"
            print(f"  [FAIL] {thesis_fig}: {e}")
            traceback.print_exc()

    # === PHASE 3: generate_phase3_figures.py (6 figures) ===
    print("\n" + "=" * 60)
    print("PHASE 3: generate_phase3_figures.py")
    print("=" * 60)

    phase3_script = os.path.join(FIGURES_DIR, "generate_phase3_figures.py")
    phase3_funcs = [
        ("fig_conformal_conditional", "Fig 5.6"),
        ("fig_reliability_diagram", "Fig 5.11"),
        ("fig_temperature_scaling", "Fig 5.12"),
        ("fig_t7_vs_t8_comparison", "Fig 5.18"),
        ("fig_per_graph_variation", "Fig 5.19"),
        ("fig_stratified_uq", "Fig 5.21"),
    ]

    for func_name, thesis_fig in phase3_funcs:
        print(f"\n--- {thesis_fig}: {func_name} ---")
        try:
            run_module_function(phase3_script, func_name)
            results[thesis_fig] = "OK"
            print(f"  [OK] {thesis_fig}")
        except Exception as e:
            results[thesis_fig] = f"FAIL: {e}"
            print(f"  [FAIL] {thesis_fig}: {e}")
            traceback.print_exc()

    # === PHASE 4: Individual scripts ===
    print("\n" + "=" * 60)
    print("PHASE 4: Individual scripts")
    print("=" * 60)

    individual = [
        (
            os.path.join(FIGURES_DIR, "generate_network_intro_figure.py"),
            "main",
            "Fig 1.1",
        ),
        (
            os.path.join(FIGURES_DIR, "generate_pointnet_dataflow_figure.py"),
            "main",
            "Fig 3.4",
        ),
    ]

    for script, func_name, thesis_fig in individual:
        print(f"\n--- {thesis_fig}: {os.path.basename(script)} ---")
        try:
            run_module_function(script, func_name)
            results[thesis_fig] = "OK"
            print(f"  [OK] {thesis_fig}")
        except Exception as e:
            results[thesis_fig] = f"FAIL: {e}"
            print(f"  [FAIL] {thesis_fig}: {e}")
            traceback.print_exc()

    # === PHASE 5: Root-level scripts (need repo root as cwd) ===
    print("\n" + "=" * 60)
    print("PHASE 5: Root-level scripts")
    print("=" * 60)

    os.chdir(REPO_ROOT)

    root_scripts = [
        (
            os.path.join(REPO_ROOT, "scripts", "regenerate_fig58_s30.py"),
            "main",
            "Fig 5.8",
        ),
        (os.path.join(REPO_ROOT, "scripts", "compute_pit.py"), "main", "Fig 5.13"),
        (
            os.path.join(REPO_ROOT, "scripts", "compute_pit_after_tempscaling.py"),
            "main",
            "Fig 6.3",
        ),
        (
            os.path.join(REPO_ROOT, "scripts", "generate_s_convergence_figure.py"),
            "main",
            "Fig 6.1",
        ),
    ]

    for script, func_name, thesis_fig in root_scripts:
        print(f"\n--- {thesis_fig}: {os.path.basename(script)} ---")
        try:
            run_module_function(script, func_name)
            results[thesis_fig] = "OK"
            print(f"  [OK] {thesis_fig}")
        except Exception as e:
            results[thesis_fig] = f"FAIL: {e}"
            print(f"  [FAIL] {thesis_fig}: {e}")
            traceback.print_exc()

    # === PHASE 6: Part-level scripts ===
    print("\n" + "=" * 60)
    print("PHASE 6: Part-level scripts")
    print("=" * 60)

    part_scripts = [
        (os.path.join(REPO_ROOT, "run_part2_uq_analyses.py"), "main", "Fig 5.14"),
        (
            os.path.join(REPO_ROOT, "run_part3_calibration_audit.py"),
            "main",
            "Fig 5.9 + 5.10",
        ),
        (
            os.path.join(REPO_ROOT, "run_part4_t7_crosscheck.py"),
            "main",
            "Fig 5.15 + 5.16 + 5.17",
        ),
    ]

    for script, func_name, thesis_fig in part_scripts:
        print(f"\n--- {thesis_fig}: {os.path.basename(script)} ---")
        try:
            run_module_function(script, func_name)
            results[thesis_fig] = "OK"
            print(f"  [OK] {thesis_fig}")
        except Exception as e:
            results[thesis_fig] = f"FAIL: {e}"
            print(f"  [FAIL] {thesis_fig}: {e}")
            traceback.print_exc()

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("REGENERATION SUMMARY")
    print("=" * 60)

    ok_count = sum(1 for v in results.values() if v == "OK")
    fail_count = sum(1 for v in results.values() if v != "OK")

    for fig, status in sorted(results.items()):
        marker = "OK" if status == "OK" else "FAIL"
        print(f"  [{marker}] {fig}: {status}")

    print(f"\nTotal: {ok_count} OK, {fail_count} FAIL out of {len(results)}")


if __name__ == "__main__":
    main()
