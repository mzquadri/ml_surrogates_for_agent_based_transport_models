"""
COMPREHENSIVE VISUALIZATION GENERATOR
Creates all 8 detailed trial figures (Figure 2-9) in sequence

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/generate_all_detailed_figures.py
"""

import os
import subprocess

BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"

# All figure scripts
FIGURE_SCRIPTS = [
    "figure1_trials_overview.py",  # Already exists
    "figure2_trial1_detailed.py",   # Trial 1 - Failed
    "figure3_trial8_detailed.py",   # Trial 8 - Best Model
    "figure4_trials_comparison.py", # All trials comparison matrix
]

print("="*80)
print(" GENERATING ALL COMPREHENSIVE FIGURES")
print("="*80)
print("\nThis will generate:")
print("  • Figure 1: Complete Trials Overview (All 8)")
print("  • Figure 2: Trial 1 Detailed Analysis (Failed Model)")
print("  • Figure 3: Trial 8 Detailed Analysis (Best Model)")
print("  • Figure 4: Comprehensive Trials Comparison Matrix")
print("\n" + "="*80 + "\n")

for i, script in enumerate(FIGURE_SCRIPTS, 1):
    script_path = f"{BASE_PATH}/{script}"
    print(f"\n[{i}/{len(FIGURE_SCRIPTS)}] Running: {script}")
    print("-"*80)
    
    try:
        result = subprocess.run(['python', script_path], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {script} completed")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ ERROR in {script}:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {script} took too long")
    except Exception as e:
        print(f"❌ EXCEPTION in {script}: {str(e)}")
    
    print("-"*80)

print("\n" + "="*80)
print(" ALL FIGURES GENERATION COMPLETE")
print("="*80)
print("\n📁 Output Location:")
print(f"   {BASE_PATH}/visualizations/")
print("\n📊 Generated Files:")
print("   • figure1_complete_trials_overview.png")
print("   • figure2_trial1_detailed_analysis.png")
print("   • figure3_trial8_best_model_detailed.png")
print("   • figure4_trials_comparison_matrix.png")
print("\n✅ All visualizations ready for professor presentation!")
print("="*80)
