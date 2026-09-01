"""
FIGURE 2: DETAILED 4-PANEL HYPERPARAMETER ANALYSIS
Multi-metric comparison, variance coverage, learning rate, and dropout effects

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure2_detailed_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trial data
TRIALS_DATA = {
    'Trial 2': {'test_r2': 0.5841, 'val_test_gap': 12.40, 'variance_coverage': 66.2, 'label': 'Trial 2\n(drop=0.3)', 'color': '#a29bfe'},
    'Trial 3': {'test_r2': 0.5953, 'val_test_gap': 62.26, 'variance_coverage': 106.6, 'label': 'Trial 3\n(drop=0.0+WL)', 'color': '#6c5ce7'},
    'Trial 5': {'test_r2': 0.5553, 'test_pearson': 0.7468, 'test_spearman': 0.2276, 'val_test_gap': -0.96, 'variance_coverage': 70.6, 'label': 'Trial 5\n(Baseline)', 'color': '#ffd93d'},
    'Trial 6': {'test_r2': 0.5223, 'test_pearson': 0.7262, 'test_spearman': 0.2006, 'val_test_gap': 0.01, 'variance_coverage': 65.9, 'label': 'Trial 6\n(LR=3e-4)', 'color': '#ff6b6b'},
    'Trial 7': {'test_r2': 0.5471, 'test_pearson': 0.7409, 'test_spearman': 0.2267, 'val_test_gap': 0.47, 'variance_coverage': 71.0, 'label': 'Trial 7\n(LR=6e-4)', 'color': '#ff6b6b'},
    'Trial 8': {'test_r2': 0.5957, 'test_pearson': 0.7726, 'test_spearman': 0.2929, 'val_test_gap': 0.21, 'variance_coverage': 74.5, 'label': 'Trial 8\n(Best Model)', 'color': '#51cf66'}
}

print("="*80)
print(" GENERATING FIGURE 2: DETAILED 4-PANEL ANALYSIS")
print("="*80)

fig = plt.figure(figsize=(28, 26))
gs = GridSpec(2, 2, figure=fig, hspace=0.6, wspace=0.5, left=0.08, right=0.96, top=0.88, bottom=0.06)

# ============================================================================
# PANEL A: Multi-Metric Performance (Trial 5 vs 8)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

metrics = ['R²', 'Pearson\nCorr.', 'Spearman\nCorr.']
trial5_metrics = [TRIALS_DATA['Trial 5']['test_r2'], TRIALS_DATA['Trial 5']['test_pearson'], TRIALS_DATA['Trial 5']['test_spearman']]
trial8_metrics = [TRIALS_DATA['Trial 8']['test_r2'], TRIALS_DATA['Trial 8']['test_pearson'], TRIALS_DATA['Trial 8']['test_spearman']]

x = np.arange(len(metrics))
width = 0.4

bars1 = ax1.bar(x - width/2, trial5_metrics, width, label='Trial 5 (Baseline, dropout=0.3)',
                color='#ffd93d', edgecolor='black', linewidth=2, alpha=0.9)
bars2 = ax1.bar(x + width/2, trial8_metrics, width, label='Trial 8 (Best, dropout=0.2)',
                color='#51cf66', edgecolor='black', linewidth=2, alpha=0.9)

ax1.set_ylabel('Score Value (Higher is Better)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Evaluation Metric', fontsize=16, fontweight='bold')
ax1.set_title('(A) Multi-Metric Performance Comparison', fontsize=17, fontweight='bold', pad=40)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=13, loc='lower right', framealpha=0.95)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)

for bars, metrics_vals in [(bars1, trial5_metrics), (bars2, trial8_metrics)]:
    for bar, val in zip(bars, metrics_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement analysis box
improvement_text = "Trial 8 Improvements over Trial 5:\n"
improvement_text += f"R²: {((trial8_metrics[0]-trial5_metrics[0])/trial5_metrics[0]*100):+.2f}%\n"
improvement_text += f"Pearson: {((trial8_metrics[1]-trial5_metrics[1])/trial5_metrics[1]*100):+.2f}%\n"
improvement_text += f"Spearman: {((trial8_metrics[2]-trial5_metrics[2])/trial5_metrics[2]*100):+.2f}%\n"
improvement_text += f"\nKey Change: Dropout 0.3 \u2192 0.2"
ax1.text(0.98, 0.98, improvement_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))

# ============================================================================
# PANEL B: Variance Coverage Analysis
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

trials_diag = ['Trial 5', 'Trial 6', 'Trial 7', 'Trial 8']
variance_coverage = [TRIALS_DATA[t]['variance_coverage'] for t in trials_diag]

bars = ax2.bar(range(len(trials_diag)), variance_coverage, 
               color=[TRIALS_DATA[t]['color'] for t in trials_diag],
               edgecolor='black', linewidth=2.5, alpha=0.9, width=0.65)

ax2.axhspan(70, 85, alpha=0.25, color='green', label='Optimal Range (70-85%)')
ax2.axhline(y=70, color='red', linestyle='--', linewidth=2.5, alpha=0.6)
ax2.axhline(y=85, color='red', linestyle='--', linewidth=2.5, alpha=0.6)

ax2.set_ylabel('Variance Coverage (%)\n(Prediction Std / Target Std × 100)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Trial Configuration', fontsize=16, fontweight='bold')
ax2.set_title('(B) Underfitting/Overfitting Diagnosis\nvia Variance Coverage', fontsize=17, fontweight='bold', pad=40)
ax2.set_xticks(range(len(trials_diag)))
ax2.set_xticklabels([TRIALS_DATA[t]['label'] for t in trials_diag], fontsize=13, fontweight='bold')
ax2.set_ylim(60, 120)
ax2.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)

for i, (bar, coverage) in enumerate(zip(bars, variance_coverage)):
    height = bar.get_height()
    if coverage < 70:
        diagnosis, text_color = "Underfitting", 'red'
    elif coverage > 85:
        diagnosis, text_color = "Overfitting", 'darkred'
    else:
        diagnosis, text_color = "Optimal", 'green'
    
    ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{coverage:.1f}%\n{diagnosis}',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color)

# Add diagnosis summary box
diag_text = "Variance Coverage Analysis:\n"
diag_text += f"Optimal Range: 70-85%\n"
diag_text += f"<70%: Underfitting (narrow predictions)\n"
diag_text += f">85%: Overfitting (too confident)\n\n"
diag_text += f"Best Trial: Trial 8 ({TRIALS_DATA['Trial 8']['variance_coverage']:.1f}%)\n"
diag_text += f"Status: Slightly High but Acceptable"
ax2.text(0.02, 0.98, diag_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))

# ============================================================================
# PANEL C: Learning Rate Sensitivity
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

lr_labels = ['3e-4\n(Trial 6)', '5e-4\n(Trials 5, 8)', '6e-4\n(Trial 7)']
r2_at_lr = [0.5223, 0.5755, 0.5471]
colors_lr = ['#ff6b6b', '#51cf66', '#ff6b6b']

bars = ax3.bar(range(len(lr_labels)), r2_at_lr, color=colors_lr,
               edgecolor='black', linewidth=2.5, alpha=0.9, width=0.6)

ax3.axvline(x=1, color='green', linestyle='--', linewidth=3.5, alpha=0.7,
            label='Optimal Learning Rate')

ax3.set_ylabel('Average Test R² Score', fontsize=16, fontweight='bold')
ax3.set_xlabel('Learning Rate Configuration', fontsize=16, fontweight='bold')
ax3.set_title('(C) Learning Rate Sensitivity Analysis', fontsize=17, fontweight='bold', pad=40)
ax3.set_xticks(range(len(lr_labels)))
ax3.set_xticklabels(lr_labels, fontsize=13, fontweight='bold')
ax3.set_ylim(0.5, 0.6)
ax3.legend(fontsize=12, loc='lower right', framealpha=0.95)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)

for bar, score in zip(bars, r2_at_lr):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{score:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

annotations = [
    (0, 0.515, 'Too slow\nUnderfits', '#ff6b6b'),
    (1, 0.515, 'Optimal\nConverges well', '#51cf66'),
    (2, 0.515, 'Overshoots\nUnstable', '#ff6b6b')
]
for x_pos, y_pos, text, color in annotations:
    ax3.text(x_pos, y_pos, text, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.4, edgecolor='black', linewidth=1.5))

# Add recommendation box
lr_rec_text = "Learning Rate Findings:\n"
lr_rec_text += f"3e-4: Too conservative (-9.3%)\n"
lr_rec_text += f"5e-4: Optimal baseline (100%)\n"
lr_rec_text += f"6e-4: Slightly unstable (-4.9%)\n\n"
lr_rec_text += f"Recommendation: 5e-4\n"
lr_rec_text += f"Used in: Trials 5, 8 (best results)"
ax3.text(0.98, 0.98, lr_rec_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=2))

# ============================================================================
# PANEL D: Dropout + Weighted Loss Effect
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

dropout_trials = ['Trial 2\n(drop=0.3)', 'Trial 3\n(drop=0.0+WL)', 'Trial 5\n(drop=0.3)', 'Trial 8\n(drop=0.2)']
dropout_r2 = [0.5841, 0.5953, 0.5500, 0.5957]
dropout_colors = ['#a29bfe', '#6c5ce7', '#ffd93d', '#51cf66']

bars = ax4.bar(range(len(dropout_trials)), dropout_r2, color=dropout_colors,
               edgecolor='black', linewidth=2.5, alpha=0.9, width=0.65)

ax4.axvline(x=1.5, color='orange', linestyle=':', linewidth=3.5, alpha=0.7,
            label='Phase Transition (Explore → Optimize)')

ax4.set_ylabel('R² Score (Val for 2-3, Test for 5&8)', fontsize=15, fontweight='bold')
ax4.set_xlabel('Trial and Dropout Configuration', fontsize=16, fontweight='bold')
ax4.set_title('(D) Dropout + Weighted Loss Effect Analysis', fontsize=17, fontweight='bold', pad=40)
ax4.set_xticks(range(len(dropout_trials)))
ax4.set_xticklabels(dropout_trials, fontsize=12, fontweight='bold')
ax4.set_ylim(0.54, 0.62)
ax4.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)

for i, (bar, score) in enumerate(zip(bars, dropout_r2)):
    height = bar.get_height()
    phase = "Early" if i < 2 else "Optimized"
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.006,
            f'{score:.4f}\n({phase})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add dropout findings box
dropout_text = "Dropout Effect Findings:\n"
dropout_text += f"Trial 2 (0.3): Val R²=0.5841\n"
dropout_text += f"Trial 3 (0.0+WL): Val R²=0.5953\n"
dropout_text += f"Trial 5 (0.3): Test R²=0.5553\n"
dropout_text += f"Trial 8 (0.2): Test R²=0.5957 \u2713\n\n"
dropout_text += f"Conclusion: 0.2 optimal\n"
dropout_text += f"Weighted Loss: No benefit"
ax4.text(0.02, 0.98, dropout_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='black', linewidth=2))

plt.suptitle('Figure 2: Detailed Hyperparameter Analysis (Trials 5-8)\nPointNetTransfGAT Architecture - Systematic Optimization', 
             fontsize=19, fontweight='bold', y=0.965)

fig.savefig(f"{OUTPUT_DIR}/figure2_detailed_analysis.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: figure2_detailed_analysis.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 2 COMPLETE!")
print("="*80)
