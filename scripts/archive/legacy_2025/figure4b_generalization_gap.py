"""
FIGURE 4B: GENERALIZATION GAP ANALYSIS
Horizontal Bar Chart (All 7 Valid Trials)

Reference: Boreale et al. (2024) - ML Surrogates for Agent-Based Transport Models
Lower Gap = Better Generalization

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure4b_generalization_gap.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generalization gap data (Trial 1 excluded - failed)
TRIALS_GAP = {
    'Trial 2': {'gap': 12.4, 'status': 'OK', 'color': '#a29bfe'},
    'Trial 3': {'gap': 62.3, 'status': 'OVERFIT', 'color': '#fd79a8'},
    'Trial 4': {'gap': 60.2, 'status': 'OVERFIT', 'color': '#6c5ce7'},
    'Trial 5': {'gap': 1.0, 'status': 'EXCELLENT', 'color': '#ffd93d'},
    'Trial 6': {'gap': 0.0, 'status': 'EXCELLENT', 'color': '#ff9770'},
    'Trial 7': {'gap': 0.5, 'status': 'EXCELLENT', 'color': '#ff7f50'},
    'Trial 8': {'gap': 0.2, 'status': 'BEST', 'color': '#51cf66'}
}

print("="*80)
print(" FIGURE 4B: GENERALIZATION GAP ANALYSIS")
print("="*80)
print("\nAnalyzing generalization quality...")
print("Valid Trials: 7 (Trial 1 excluded - failed)")
print("Reference: Boreale et al. (2024)")
print("\n" + "="*80 + "\n")

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('white')

trial_names = list(TRIALS_GAP.keys())
gaps = [TRIALS_GAP[t]['gap'] for t in trial_names]
colors = [TRIALS_GAP[t]['color'] for t in trial_names]
statuses = [TRIALS_GAP[t]['status'] for t in trial_names]

# Sort by gap (ascending - best first)
sorted_indices = np.argsort(gaps)
trial_names_sorted = [trial_names[i] for i in sorted_indices]
gaps_sorted = [gaps[i] for i in sorted_indices]
colors_sorted = [colors[i] for i in sorted_indices]
statuses_sorted = [statuses[i] for i in sorted_indices]

y_pos = np.arange(len(trial_names_sorted))

# Horizontal bars
bars = ax.barh(y_pos, gaps_sorted, color=colors_sorted, 
              edgecolor='black', linewidth=2.5, alpha=0.9, height=0.65)

# Add 3D effect
for bar in bars:
    bar.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(2, -2), shadow_rgbFace='gray', alpha=0.3)])

# Value labels with status
for i, (bar, gap, status) in enumerate(zip(bars, gaps_sorted, statuses_sorted)):
    width = bar.get_width()
    label_x = width + 1.5
    
    # Gap percentage
    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
           f'{gap:.1f}%',
           ha='left', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='black', linewidth=1.5, alpha=0.95))
    
    # Status badge inside bar
    if width > 10:
        ax.text(width/2, bar.get_y() + bar.get_height()/2, 
               status,
               ha='center', va='center', fontsize=10, fontweight='bold', 
               color='white')

# Threshold lines
ax.axvline(x=1, color='green', linestyle='--', linewidth=3, alpha=0.8, 
          label='Excellent Threshold (<1%)', zorder=0)
ax.axvline(x=15, color='orange', linestyle='--', linewidth=3, alpha=0.8, 
          label='Acceptable Threshold (<15%)', zorder=0)
ax.axvline(x=50, color='red', linestyle='--', linewidth=3, alpha=0.8, 
          label='Critical Threshold (>50%)', zorder=0)

# Labels and title
ax.set_yticks(y_pos)
ax.set_yticklabels(trial_names_sorted, fontsize=13, fontweight='bold')
ax.set_xlabel('Generalization Gap (%)', fontsize=16, fontweight='bold')
ax.set_title('Figure 4B: Generalization Gap Analysis (7 Valid Trials)\n' +
            'Lower Gap = Better Generalization | Sorted by Performance\n' +
            'Reference: Boreale et al. (2024)', 
            fontsize=17, fontweight='bold', pad=25)

# Grid and legend
ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1.5)
ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
ax.set_xlim(0, 70)

# Summary statistics box
summary_text = "SUMMARY:\n"
summary_text += f"• Excellent (<1%): 4 trials (57.1%)\n"
summary_text += f"• Acceptable (<15%): 1 trial (14.3%)\n"
summary_text += f"• Critical (>50%): 2 trials (28.6%)\n"
summary_text += f"• Best: Trial 8 (Gap = 0.2%)"

ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
       fontsize=11, fontweight='bold', verticalalignment='top',
       horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', 
                edgecolor='blue', linewidth=2.5, alpha=0.95))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure4b_generalization_gap.png", 
           dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: figure4b_generalization_gap.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 4B COMPLETE")
print("="*80)
