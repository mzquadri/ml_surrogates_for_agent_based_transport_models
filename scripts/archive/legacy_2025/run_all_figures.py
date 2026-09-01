"""
RUN ALL FIGURES - MASTER SCRIPT
Execute all 6 figure generation scripts sequentially

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/run_all_figures.py
"""

import subprocess
import os

BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"

# List of all figure scripts
FIGURE_SCRIPTS = [
    "figure1_trials_overview.py",
    "figure2_detailed_analysis.py",
    "figure3_predictions_scatter.py",
    "figure4_residual_analysis.py",
    "figure5_generalization.py",
    "figure6_benchmark_comparison.py"
]

print("\n" + "="*80)
print(" MASTER SCRIPT: GENERATING ALL 6 FIGURES")
print("="*80)
print(f"\nBase Path: {BASE_PATH}")
print(f"Total Figures: {len(FIGURE_SCRIPTS)}\n")

# Track results
successful = []
failed = []

for i, script in enumerate(FIGURE_SCRIPTS, 1):
    script_path = os.path.join(BASE_PATH, script)
    
    print(f"\n[{i}/{len(FIGURE_SCRIPTS)}] Running: {script}")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=False,
            text=True,
            check=True
        )
        successful.append(script)
        print(f"[OK] {script} completed successfully")
    except subprocess.CalledProcessError as e:
        failed.append(script)
        print(f"[ERROR] {script} failed with error code {e.returncode}")
    except Exception as e:
        failed.append(script)
        print(f"[ERROR] {script} failed: {str(e)}")

# Final summary
print("\n" + "="*80)
print(" FINAL SUMMARY")
print("="*80)
print(f"\n[OK] Successful: {len(successful)}/{len(FIGURE_SCRIPTS)}")
for script in successful:
    print(f"     - {script}")

if failed:
    print(f"\n[ERROR] Failed: {len(failed)}/{len(FIGURE_SCRIPTS)}")
    for script in failed:
        print(f"     - {script}")
else:
    print(f"\n[OK] All figures generated successfully!")

print(f"\nOutput Directory: {BASE_PATH}/visualizations")
print("\n" + "="*80)
print(" VISUALIZATION GENERATION COMPLETE!")
print("="*80)
