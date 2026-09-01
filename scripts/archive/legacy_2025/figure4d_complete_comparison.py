"""
FIGURE 4D: COMPLETE VALIDATION VS TEST COMPARISON
Side-by-Side Bar Chart (All 8 Trials)

Reference: Boreale et al. (2024) - ML Surrogates for Agent-Based Transport Models
Complete R² comparison with benchmark reference

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure4d_complete_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Complete data (All 8 Trials)
TRIALS_COMPLETE = {
    'Trial 1': {'val': -0.0020, 'test': -0.0022, 'status': 'FAILED', 'color': '#ff4444'},
    'Trial 2': {'val': 0.5841, 'test': 0.5117, 'status': 'OK', 'color': '#a29bfe'},
    'Trial 3': {'val': 0.5953, 'test': 0.2246, 'status': 'OVERFIT', 'color': '#fd79a8'},
    'Trial 4': {'val': 0.6097, 'test': 0.2426, 'status': 'OVERFIT', 'color': '#6c5ce7'},
    'Trial 5': {'val': 0.5500, 'test': 0.5553, 'status': 'EXCELLENT', 'color': '#ffd93d'},
    'Trial 6': {'val': 0.5224, 'test': 0.5223, 'status': 'EXCELLENT', 'color': '#ff9770'},
    'Trial 7': {'val': 0.5497, 'test': 0.5471, 'status': 'EXCELLENT', 'color': '#ff7f50'},
    'Trial 8': {'val': 0.5970, 'test': 0.5957, 'status': 'BEST', 'color': '#51cf66'}
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" FIGURE 4D: COMPLETE VALIDATION VS TEST")
print("="*80)
print("\nSide-by-side comparison of all trials...")
print("Total Trials: 8")
print(f"Reference Benchmark: R² = {BENCHMARK_R2}")
print("\n" + "="*80 + "\n")

# Create figure
fig, ax = plt.subplots(figsize=(20, 11))
fig.patch.set_facecolor('white')

trial_names = list(TRIALS_COMPLETE.keys())
val_r2s = [TRIALS_COMPLETE[t]['val'] for t in trial_names]
test_r2s = [TRIALS_COMPLETE[t]['test'] for t in trial_names]
colors = [TRIALS_COMPLETE[t]['color'] for t in trial_names]

x = np.arange(len(trial_names))
width = 0.38

# Validation bars (blue)
bars_val = ax.bar(x - width/2, val_r2s, width, 
                 label='Validation R²',
                 color='#74b9ff', edgecolor='black', 
                 linewidth=2.5, alpha=0.9)

# Test bars (status colors)
bars_test = ax.bar(x + width/2, test_r2s, width, 
                  label='Test R²',
                  color=colors, edgecolor='black', 
                  linewidth=2.5, alpha=0.9)

# Add 3D effects
for bar in bars_val:
    bar.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(2, -2), shadow_rgbFace='blue', alpha=0.3)])
for bar in bars_test:
    bar.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(2, -2), shadow_rgbFace='gray', alpha=0.3)])

# Value labels on top
for i, bar in enumerate(bars_val):
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{val_r2s[i]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)

for i, bar in enumerate(bars_test):
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
               f'{test_r2s[i]:.4f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)

# Benchmark reference line
ax.axhline(y=BENCHMARK_R2, color='gold', linestyle='--', linewidth=4, 
          label=f'Reference Benchmark\n(Boreale et al. 2024, R²={BENCHMARK_R2})', 
          zorder=0)

# Good performance line
ax.axhline(y=0.5, color='green', linestyle=':', linewidth=2.5, 
          alpha=0.6, label='Good Performance (R² > 0.5)', zorder=0)

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels([f"Trial {t.split()[1]}" for t in trial_names], 
                   fontsize=13, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=16, fontweight='bold')
ax.set_xlabel('Trial Number (Chronological Order)', fontsize=16, fontweight='bold')
ax.set_title('Figure 4D: Complete Validation vs Test R² Comparison\n' +
            'All 8 Trials | Side-by-Side Analysis | Reference: Boreale et al. (2024)\n' +
            'Architecture: PointNetTransfGAT | Dataset: 1,000 scenarios', 
            fontsize=17, fontweight='bold', pad=25)

# Grid and legend
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)
ax.legend(fontsize=13, loc='upper left', ncol=2, framealpha=0.95)
ax.set_ylim(-0.12, 0.9)

# Status annotations
status_y = -0.08
for i, (trial, status) in enumerate(zip(trial_names, 
                                        [TRIALS_COMPLETE[t]['status'] for t in trial_names])):
    ax.text(i, status_y, status, ha='center', va='top', 
           fontsize=9, fontweight='bold', color=colors[i],
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor=colors[i], linewidth=1.5, alpha=0.9))

# Summary statistics box
summary_text = "SUMMARY STATISTICS:\n\n"
summary_text += "Validation R²:\n"
summary_text += f"  • Mean: {np.mean(val_r2s):.4f}\n"
summary_text += f"  • Best: {max(val_r2s):.4f} (Trial 4)\n\n"
summary_text += "Test R²:\n"
summary_text += f"  • Mean: {np.mean(test_r2s):.4f}\n"
summary_text += f"  • Best: {max(test_r2s):.4f} (Trial 8)\n\n"
summary_text += f"Usable Models: 5/8 (62.5%)"

ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
       fontsize=11, fontweight='bold', verticalalignment='top',
       horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.9', facecolor='lightcyan', 
                edgecolor='blue', linewidth=2.5, alpha=0.95),
       linespacing=1.6)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure4d_complete_comparison.png", 
           dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: figure4d_complete_comparison.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 4D COMPLETE")
print("="*80)
