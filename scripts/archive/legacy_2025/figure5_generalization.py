"""
FIGURE 5: GENERALIZATION PERFORMANCE ANALYSIS
Validation vs Test R² comparison and gap analysis

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure5_generalization.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trial data
TRIALS_DATA = {
    'Trial 5': {'val_r2': 0.5500, 'test_r2': 0.5553, 'val_test_gap': -0.96, 'label': 'Trial 5\n(Baseline)', 'color': '#ffd93d'},
    'Trial 6': {'val_r2': 0.5224, 'test_r2': 0.5223, 'val_test_gap': 0.01, 'label': 'Trial 6\n(LR=3e-4)', 'color': '#ff6b6b'},
    'Trial 7': {'val_r2': 0.5497, 'test_r2': 0.5471, 'val_test_gap': 0.47, 'label': 'Trial 7\n(LR=6e-4)', 'color': '#ff6b6b'},
    'Trial 8': {'val_r2': 0.5970, 'test_r2': 0.5957, 'val_test_gap': 0.21, 'label': 'Trial 8\n(Best Model)', 'color': '#51cf66'}
}

print("="*80)
print(" GENERATING FIGURE 5: GENERALIZATION ANALYSIS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 13))
plt.subplots_adjust(left=0.08, right=0.96, top=0.79, bottom=0.12, wspace=0.5)

# ============================================================================
# LEFT PANEL: Val vs Test R²
# ============================================================================
trials_list = ['Trial 5', 'Trial 6', 'Trial 7', 'Trial 8']
val_r2 = [TRIALS_DATA[t]['val_r2'] for t in trials_list]
test_r2 = [TRIALS_DATA[t]['test_r2'] for t in trials_list]

x = np.arange(len(trials_list))
width = 0.38

bars1 = ax1.bar(x - width/2, val_r2, width, label='Validation Set (100 samples)',
                color='#74b9ff', edgecolor='black', linewidth=2, alpha=0.9)
bars2 = ax1.bar(x + width/2, test_r2, width, label='Test Set (100 samples)',
                color='#fdcb6e', edgecolor='black', linewidth=2, alpha=0.9)

ax1.set_ylabel('R² Score (Coefficient of Determination)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Trial Configuration', fontsize=16, fontweight='bold')
ax1.set_title('(A) Validation vs Test R²: Generalization Assessment', 
              fontsize=17, fontweight='bold', pad=40)
ax1.set_xticks(x)
ax1.set_xticklabels([TRIALS_DATA[t]['label'] for t in trials_list], fontsize=13, fontweight='bold')
ax1.set_ylim(0.5, 0.62)
ax1.legend(fontsize=14, loc='lower right', framealpha=0.95)
ax1.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=1.2)

# Add value labels
for i, trial in enumerate(trials_list):
    val = val_r2[i]
    test = test_r2[i]
    gap = TRIALS_DATA[trial]['val_test_gap']
    
    ax1.text(i - width/2, val + 0.004, f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    color = 'green' if abs(gap) < 2 else 'orange' if abs(gap) < 5 else 'red'
    ax1.text(i + width/2, test + 0.004, f'{test:.4f}\n(gap: {gap:+.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

# ============================================================================
# RIGHT PANEL: Generalization Gap
# ============================================================================
gaps = [abs(TRIALS_DATA[t]['val_test_gap']) for t in trials_list]
colors_gap = ['green' if g < 2 else 'orange' if g < 5 else 'red' for g in gaps]

bars = ax2.bar(range(len(trials_list)), gaps, color=colors_gap,
               edgecolor='black', linewidth=2.5, alpha=0.9, width=0.65)

ax2.axhline(y=2, color='green', linestyle='--', linewidth=3, 
            label='Excellent (<2%)', alpha=0.75)
ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2.5, 
            label='Caution Zone (2-5%)', alpha=0.75)

ax2.set_ylabel('Absolute Validation-Test Gap (%)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Trial Configuration', fontsize=16, fontweight='bold')
ax2.set_title('(B) Generalization Gap Analysis: Overfitting Detection', 
              fontsize=17, fontweight='bold', pad=40)
ax2.set_xticks(range(len(trials_list)))
ax2.set_xticklabels([TRIALS_DATA[t]['label'] for t in trials_list], fontsize=13, fontweight='bold')
ax2.set_ylim(0, 6)
ax2.legend(fontsize=13, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=1.2)

# Add value labels
for i, (bar, gap) in enumerate(zip(bars, gaps)):
    height = bar.get_height()
    verdict = "Excellent" if gap < 2 else "Good" if gap < 5 else "Caution"
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.25,
            f'{gap:.2f}%\n{verdict}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('Figure 5: Generalization Performance Analysis\nValidation vs Test Set Comparison Across All Trials', 
             fontsize=19, fontweight='bold', y=0.93)

fig.savefig(f"{OUTPUT_DIR}/figure5_generalization_analysis.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: figure5_generalization_analysis.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 5 COMPLETE!")
print("="*80)
