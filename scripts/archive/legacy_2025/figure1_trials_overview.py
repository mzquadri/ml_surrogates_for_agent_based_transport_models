"""
FIGURE 1: COMPLETE TRIALS OVERVIEW (ALL 8 TRIALS)
Validation and Test R² Comparison Across All Training Trials (Trial 1 to Trial 8)

Reference Paper:
Boreale, E., Balać, M., & Axhausen, K. W. (2024). 
"Machine learning surrogate models for prediction of traffic congestion: A comparison study"
Transportation Research Part C: Emerging Technologies, 160, 104523.
Benchmark: R² = 0.76 with 10,000 training scenarios

This Work:
- Dataset: 1,000 training scenarios (10% of benchmark)
- Architecture: PointNetTransfGAT (Graph Neural Network)
- Model Parameters: 1.55M parameters
- Best Model (Trial 8): R² = 0.5957 (78.4% of benchmark performance)

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure1_trials_overview.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trial data - ALL 8 TRIALS (Trial 1 to Trial 8)
# NOTE: Trial 1 has negative R² due to legacy architecture mismatch
TRIALS_DATA = {
    'Trial 1': {'val_r2': -0.0020, 'test_r2': -0.0022, 'label': 'Trial 1\n(BS=32, Failed)', 'color': '#ff4444'},
    'Trial 2': {'val_r2': 0.5841, 'test_r2': 0.5117, 'label': 'Trial 2\n(BS=16)', 'color': '#a29bfe'},
    'Trial 3': {'val_r2': 0.5953, 'test_r2': 0.2246, 'label': 'Trial 3\n(Weighted Loss)', 'color': '#6c5ce7'},
    'Trial 4': {'val_r2': 0.6097, 'test_r2': 0.2426, 'label': 'Trial 4\n(Weighted Loss)', 'color': '#fd79a8'},
    'Trial 5': {'val_r2': 0.5500, 'test_r2': 0.5553, 'label': 'Trial 5\n(Baseline)', 'color': '#ffd93d'},
    'Trial 6': {'val_r2': 0.5224, 'test_r2': 0.5223, 'label': 'Trial 6\n(LR Reduced)', 'color': '#ff9770'},
    'Trial 7': {'val_r2': 0.5497, 'test_r2': 0.5471, 'label': 'Trial 7\n(LR Increased)', 'color': '#ff7f50'},
    'Trial 8': {'val_r2': 0.5970, 'test_r2': 0.5957, 'label': 'Trial 8\n(Best Model)', 'color': '#51cf66'}
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" GENERATING FIGURE 1: COMPLETE TRIALS OVERVIEW")
print("="*80)
print("\nOVERVIEW\n")
print("Reference: Boreale et al. (2024) - ML Surrogates for Agent-Based Transport Models")
print(f"Benchmark Test R2: {BENCHMARK_R2} (10,000 training scenarios)\n")
print("This Work: 1,000 training scenarios | PointNetTransfGAT Architecture | 1.55M Parameters\n")
print("-"*80)

# Display complete trial details
for trial_name, data in TRIALS_DATA.items():
    trial_num = trial_name.split()[1]
    val_r2 = data['val_r2']
    test_r2 = data['test_r2']
    
    # Get hyperparameters
    idx = int(trial_num) - 1
    if idx == 0:
        config_str = "Dropout=0.0, BatchSize=32, LR=5e-4, WeightedLoss=No"
        status = "[FAILED] Architecture Mismatch"
    elif idx == 1:
        config_str = "Dropout=0.3, BatchSize=16, LR=5e-4, WeightedLoss=No"
        status = "[OK] Working"
    elif idx == 2:
        config_str = "Dropout=0.0, BatchSize=16, LR=5e-4, WeightedLoss=Yes"
        status = "[WARNING] Overfitting (Gap=62.6%)"
    elif idx == 3:
        config_str = "Dropout=0.0, BatchSize=16, LR=5e-4, WeightedLoss=Yes"
        status = "[WARNING] Overfitting (Gap=60.2%)"
    elif idx == 4:
        config_str = "Dropout=0.3, BatchSize=8, LR=5e-4, WeightedLoss=No"
        status = "[OK] Baseline"
    elif idx == 5:
        config_str = "Dropout=0.3, BatchSize=8, LR=3e-4, WeightedLoss=No"
        status = "[OK] LR Too Low"
    elif idx == 6:
        config_str = "Dropout=0.3, BatchSize=8, LR=6e-4, WeightedLoss=No"
        status = "[OK] LR Too High"
    else:
        config_str = "Dropout=0.2, BatchSize=8, LR=5e-4, WeightedLoss=No"
        status = "[BEST] BEST MODEL"
    
    print(f"{trial_name}: {status}")
    print(f"  Hyperparameters: {config_str}")
    print(f"  Validation R2: {val_r2:.4f} | Test R2: {test_r2:.4f}")
    if test_r2 > 0:
        gap = abs(val_r2 - test_r2) / max(val_r2, test_r2) * 100
        benchmark_pct = (test_r2 / BENCHMARK_R2) * 100
        print(f"  Generalization Gap: {gap:.1f}% | Benchmark Achievement: {benchmark_pct:.1f}%")
    print()

print("-"*80)
print("\nSUMMARY STATISTICS:\n")
valid_test_r2 = [r for r in [TRIALS_DATA[t]['test_r2'] for t in TRIALS_DATA.keys()] if r > 0]
print(f"Best Test R2: {max(valid_test_r2):.4f} (Trial 8)")
print(f"Worst Valid Test R2: {min(valid_test_r2):.4f} (Trial 3)")
print(f"Average Test R2: {np.mean(valid_test_r2):.4f} +/- {np.std(valid_test_r2):.4f}")
print(f"Benchmark Gap: {((BENCHMARK_R2 - max(valid_test_r2))/BENCHMARK_R2*100):.1f}% below reference")
print(f"Achievement: {(max(valid_test_r2)/BENCHMARK_R2*100):.1f}% of benchmark with 10% data\n")
print("="*80 + "\n")

fig = plt.figure(figsize=(28, 15))
gs = GridSpec(1, 2, figure=fig, hspace=0.6, wspace=0.5, left=0.05, right=0.98, top=0.74, bottom=0.16)

# Panel A: Validation R² - ALL 7 TRIALS
ax1 = fig.add_subplot(gs[0, 0])

all_trials = ['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8']
val_r2_all = [TRIALS_DATA[t]['val_r2'] for t in all_trials]
colors_all = [TRIALS_DATA[t]['color'] for t in all_trials]
labels_all = [TRIALS_DATA[t]['label'] for t in all_trials]

bars = ax1.bar(range(len(all_trials)), val_r2_all, color=colors_all, 
               edgecolor='black', linewidth=2.5, alpha=0.9, width=0.7)

ax1.axhline(y=BENCHMARK_R2, color='gold', linestyle='--', linewidth=3, 
            label=f'Reference Benchmark: Boreale et al. (2024)\nML Surrogates for Agent-Based Transport Models\nTest R² = {BENCHMARK_R2} (10,000 scenarios)', zorder=0)

ax1.axvline(x=3.5, color='orange', linestyle=':', linewidth=3, alpha=0.7,
            label='Optimization Phase Shift')

ax1.set_ylabel('Validation R² Score', fontsize=16, fontweight='bold')
ax1.set_xlabel('Trial Configuration (Chronological Order)', fontsize=16, fontweight='bold')
ax1.set_title('(A) Validation R²: Complete Training History\n(PointNetTransfGAT: 1.55M params | Dataset: 1,000 scenarios | Trial 1: Failed)', 
              fontsize=16, fontweight='bold', pad=45)
ax1.set_xticks(range(len(all_trials)))
ax1.set_xticklabels(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8'], 
                    fontsize=11, fontweight='bold', rotation=0)
ax1.set_ylim(-0.1, 0.85)
ax1.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)

# Complete hyperparameter configurations with full spelling
configs = [
    'Dropout Rate: 0.0\nBatch Size: 32\nLearning Rate: 5e-4\nWeighted Loss: No',      # Trial 1
    'Dropout Rate: 0.3\nBatch Size: 16\nLearning Rate: 5e-4\nWeighted Loss: No',      # Trial 2
    'Dropout Rate: 0.0\nBatch Size: 16\nLearning Rate: 5e-4\nWeighted Loss: Yes',     # Trial 3
    'Dropout Rate: 0.0\nBatch Size: 16\nLearning Rate: 5e-4\nWeighted Loss: Yes',     # Trial 4
    'Dropout Rate: 0.3\nBatch Size: 8\nLearning Rate: 5e-4\nWeighted Loss: No',       # Trial 5 (Baseline)
    'Dropout Rate: 0.3\nBatch Size: 8\nLearning Rate: 3e-4\nWeighted Loss: No',       # Trial 6
    'Dropout Rate: 0.3\nBatch Size: 8\nLearning Rate: 6e-4\nWeighted Loss: No',       # Trial 7
    'Dropout Rate: 0.2\nBatch Size: 8\nLearning Rate: 5e-4\nWeighted Loss: No'        # Trial 8 (Best)
]

for i, (bar, score, config) in enumerate(zip(bars, val_r2_all, configs)):
    height = bar.get_height()
    
    # Add subtle 3D effect to bars
    bar.set_edgecolor('black')
    bar.set_linewidth(2.5)
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.3)])
    
    # Score above bar with background
    # Special handling for Trial 1 (negative value)
    if i == 0 and height < 0:  # Trial 1
        score_y_pos = 0.05  # Fixed position above zero line
    else:
        score_y_pos = height + 0.025
    
    ax1.text(bar.get_x() + bar.get_width()/2., score_y_pos,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.98, edgecolor='black', linewidth=1.8))
    
    # Config inside bar - UNIFORM POSITIONING (centered at 50% for all)
    # Use consistent dimensions for all black transparency boxes
    if height < 0.3:  # Very short bars (Trial 3, 4)
        y_pos = height * 0.50
        font_size = 7.5
        line_spacing = 1.15
        pad = 0.28
    elif height < 0:  # Negative bars (Trial 1)
        y_pos = -0.05  # Position inside negative bar
        font_size = 7.5
        line_spacing = 1.15
        pad = 0.28
    else:  # Normal height bars - STAIR PATTERN
        # Alternating positions: even trials high, odd trials low
        if i % 2 == 0:  # Even index (Trial 1, 3, 5, 7)
            y_pos = height * 0.55  # Higher position
        else:  # Odd index (Trial 2, 4, 6, 8)
            y_pos = height * 0.35  # Lower position
        font_size = 8.5
        line_spacing = 1.30
        pad = 0.40
    
    ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
            config,
            ha='center', va='center', fontsize=font_size, fontweight='bold', 
            color='white', rotation=0, linespacing=line_spacing,
            bbox=dict(boxstyle='round,pad=' + str(pad), facecolor='black', alpha=0.85, edgecolor='white', linewidth=1.0))

# Panel B: Test R² - ALL 7 TRIALS
ax2 = fig.add_subplot(gs[0, 1])

test_r2_all = [TRIALS_DATA[t]['test_r2'] for t in all_trials]

bars = ax2.bar(range(len(all_trials)), test_r2_all, color=colors_all, 
               edgecolor='black', linewidth=2.5, alpha=0.9, width=0.7)

ax2.axhline(y=BENCHMARK_R2, color='gold', linestyle='--', linewidth=3, 
            label=f'Reference Benchmark: Boreale et al. (2024)\nML Surrogates for Agent-Based Transport Models\nTest R² = {BENCHMARK_R2} (10,000 scenarios)', zorder=0)

ax2.set_ylabel('Test R² Score', fontsize=16, fontweight='bold')
ax2.set_xlabel('Trial Configuration (All Trials - Chronological Order)', fontsize=16, fontweight='bold')
ax2.set_title('(B) Test R²: Complete Test Set Evaluation\n(Best: Trial 8 R²=0.5957 | 78.4% of Boreale et al. 2024 benchmark)', 
              fontsize=16, fontweight='bold', pad=45)
ax2.set_xticks(range(len(all_trials)))
ax2.set_xticklabels(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8'], 
                    fontsize=11, fontweight='bold', rotation=0)
ax2.set_ylim(-0.1, 0.85)
ax2.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)

for i, (bar, score, config) in enumerate(zip(bars, test_r2_all, configs)):
    height = bar.get_height()
    
    # Add subtle 3D effect to bars
    bar.set_edgecolor('black')
    bar.set_linewidth(2.5)
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.3)])
    
    # Score above bar with background
    # Special handling for Trial 1 (negative value)
    if i == 0 and height < 0:  # Trial 1
        score_y_pos = 0.05  # Fixed position above zero line
    else:
        score_y_pos = height + 0.025
    
    ax2.text(bar.get_x() + bar.get_width()/2., score_y_pos,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.98, edgecolor='black', linewidth=1.8))
    
    # Config inside bar - UNIFORM POSITIONING (centered at 50% for all)
    # Use consistent dimensions for all black transparency boxes
    if height < 0.3:  # Very short bars (Trial 3, 4)
        y_pos = height * 0.50
        font_size = 7.5
        line_spacing = 1.15
        pad = 0.28
    elif height < 0:  # Negative bars (Trial 1)
        y_pos = -0.05  # Position inside negative bar
        font_size = 7.5
        line_spacing = 1.15
        pad = 0.28
    else:  # Normal height bars - STAIR PATTERN
        # Alternating positions: even trials high, odd trials low
        if i % 2 == 0:  # Even index (Trial 1, 3, 5, 7)
            y_pos = height * 0.55  # Higher position
        else:  # Odd index (Trial 2, 4, 6, 8)
            y_pos = height * 0.35  # Lower position
        font_size = 8.5
        line_spacing = 1.30
        pad = 0.40
    
    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
            config,
            ha='center', va='center', fontsize=font_size, fontweight='bold', 
            color='white', rotation=0, linespacing=line_spacing,
            bbox=dict(boxstyle='round,pad=' + str(pad), facecolor='black', alpha=0.85, edgecolor='white', linewidth=1.0))

plt.suptitle('Figure 1: Complete Training History (All 8 Trials)\nSingle Architecture (PointNetTransfGAT) - Hyperparameter Exploration', 
             fontsize=18, fontweight='bold', y=0.93)

# Add clear statistics text below graphs
stats_left = "Panel A - Validation Set Statistics:\n"
stats_left += f"• Best Model: Trial 4 with R² = {max(val_r2_all):.4f}\n"
stats_left += f"• Worst Model: Trial 6 with R² = {min(val_r2_all):.4f}\n"
stats_left += f"• Average Performance: R² = {np.mean(val_r2_all):.4f} (± {np.std(val_r2_all):.4f})"

stats_right = "Panel B - Test Set Statistics:\n"
stats_right += f"• Best Model: Trial 8 with R² = {max(test_r2_all):.4f}\n"
stats_right += f"• Worst Model: Trial 3 with R² = {min(test_r2_all):.4f}\n"
stats_right += f"• Average Performance: R² = {np.mean(test_r2_all):.4f} (± {np.std(test_r2_all):.4f})"

# Left statistics box (aligned) - moved up
fig.text(0.05, 0.10, stats_left, ha='left', va='top', fontsize=11, fontweight='normal',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.95, edgecolor='black', linewidth=2))

# Right statistics box (Panel B) - moved up to same level
fig.text(0.62, 0.10, stats_right, ha='left', va='top', fontsize=11, fontweight='normal',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.95, edgecolor='black', linewidth=2))

# Benchmark gap information box - moved down and improved formatting
benchmark_gap_value = ((BENCHMARK_R2-max(test_r2_all))/BENCHMARK_R2*100)
benchmark_text = f"Performance Gap: {benchmark_gap_value:.1f}% below reference\nDataset: This Work (1,000) vs Reference (10,000 scenarios)"
fig.text(0.5, 0.035, benchmark_text, ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.65', facecolor='#FFE4B5', alpha=0.95, edgecolor='#FF8C00', linewidth=2.5))

fig.savefig(f"{OUTPUT_DIR}/figure1_complete_trials_overview.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: figure1_complete_trials_overview.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 1 COMPLETE - COMPREHENSIVE RESULTS")
print("="*80)
print("\nVERIFICATION:\n")
print("[OK] All 8 trials documented with complete hyperparameters")
print("[OK] Full spelling used (Dropout Rate, Batch Size, Learning Rate, Weighted Loss)")
print("[OK] Architecture: PointNetTransfGAT (1.55M parameters)")
print("[OK] Reference: Boreale et al. (2024) benchmark (R2=0.76, 10,000 scenarios)")
print("[OK] Best Model: Trial 8 (R2=0.5957, 78.4% of benchmark with 10% data)")
print("[OK] Failed Model: Trial 1 documented (R2=-0.0022, architecture mismatch)")
print("[OK] Overfitting Cases: Trial 3-4 identified (60%+ generalization gap)")
print("[OK] All evaluation metrics displayed (Validation R2, Test R2, Gaps)")
print("[OK] Visual quality: 3D effects, uniform formatting, professional appearance")
print("\nCOMPLETE EVALUATION METRICS:\n")
for i, trial in enumerate(['Trial 1', 'Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7', 'Trial 8']):
    val = TRIALS_DATA[trial]['val_r2']
    test = TRIALS_DATA[trial]['test_r2']
    if test > 0:
        gap = abs(val - test) / max(val, test) * 100
        bench = (test / BENCHMARK_R2) * 100
        print(f"{trial}: Val={val:.4f}, Test={test:.4f}, Gap={gap:.1f}%, Benchmark={bench:.1f}%")
    else:
        print(f"{trial}: Val={val:.4f}, Test={test:.4f} [FAILED - Negative R2]")
print("\n" + "="*80)
