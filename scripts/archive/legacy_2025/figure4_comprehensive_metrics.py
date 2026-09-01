"""
FIGURE 4: COMPREHENSIVE METRICS ANALYSIS
Complete Evaluation Metrics Visualization (All 8 Trials)

Based on complete evaluation metrics:
- Validation vs Test R² comparison
- Generalization Gap Analysis
- Benchmark Achievement Percentage
- Trial Status Classification

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure4_comprehensive_metrics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Complete evaluation metrics from Figure 1 output
TRIALS_METRICS = {
    'Trial 1': {'val': -0.0020, 'test': -0.0022, 'gap': None, 'benchmark': None, 'status': 'FAILED'},
    'Trial 2': {'val': 0.5841, 'test': 0.5117, 'gap': 12.4, 'benchmark': 67.3, 'status': 'OK'},
    'Trial 3': {'val': 0.5953, 'test': 0.2246, 'gap': 62.3, 'benchmark': 29.6, 'status': 'OVERFIT'},
    'Trial 4': {'val': 0.6097, 'test': 0.2426, 'gap': 60.2, 'benchmark': 31.9, 'status': 'OVERFIT'},
    'Trial 5': {'val': 0.5500, 'test': 0.5553, 'gap': 1.0, 'benchmark': 73.1, 'status': 'EXCELLENT'},
    'Trial 6': {'val': 0.5224, 'test': 0.5223, 'gap': 0.0, 'benchmark': 68.7, 'status': 'EXCELLENT'},
    'Trial 7': {'val': 0.5497, 'test': 0.5471, 'gap': 0.5, 'benchmark': 72.0, 'status': 'EXCELLENT'},
    'Trial 8': {'val': 0.5970, 'test': 0.5957, 'gap': 0.2, 'benchmark': 78.4, 'status': 'BEST'}
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" FIGURE 4: COMPREHENSIVE METRICS ANALYSIS")
print("="*80)
print("\nAnalyzing complete evaluation metrics...")
print(f"Total Trials: {len(TRIALS_METRICS)}")
print(f"Reference Benchmark: R2 = {BENCHMARK_R2} (Boreale et al. 2024)")
print("\n" + "="*80 + "\n")

# Create comprehensive figure
fig = plt.figure(figsize=(28, 18))
fig.patch.set_facecolor('white')
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, left=0.06, right=0.96, top=0.92, bottom=0.06)

# Color scheme
colors_by_status = {
    'FAILED': '#ff4444',
    'OK': '#a29bfe',
    'OVERFIT': '#fd79a8',
    'EXCELLENT': '#ffd93d',
    'BEST': '#51cf66'
}

trial_names = list(TRIALS_METRICS.keys())
colors = [colors_by_status[TRIALS_METRICS[t]['status']] for t in trial_names]

# ============================================================================
# PANEL 1: VALIDATION VS TEST R² SCATTER PLOT
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

valid_trials = [t for t in trial_names if TRIALS_METRICS[t]['test'] > 0]
val_r2 = [TRIALS_METRICS[t]['val'] for t in valid_trials]
test_r2 = [TRIALS_METRICS[t]['test'] for t in valid_trials]
scatter_colors = [colors_by_status[TRIALS_METRICS[t]['status']] for t in valid_trials]

# Plot perfect correlation line
ax1.plot([0, 0.7], [0, 0.7], 'k--', linewidth=2, alpha=0.5, label='Perfect Generalization', zorder=1)

# Scatter plot
for i, trial in enumerate(valid_trials):
    ax1.scatter(val_r2[i], test_r2[i], s=300, color=scatter_colors[i], 
               edgecolor='black', linewidth=2.5, alpha=0.85, zorder=3)
    ax1.text(val_r2[i], test_r2[i], trial.split()[1], 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax1.set_xlabel('Validation R²', fontsize=14, fontweight='bold')
ax1.set_ylabel('Test R²', fontsize=14, fontweight='bold')
ax1.set_title('(A) Validation vs Test R² Scatter\nGeneralization Quality Assessment', 
             fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')
ax1.set_xlim(0.15, 0.65)
ax1.set_ylim(0.15, 0.65)

# Add Trial 1 separately (negative values)
ax1.text(0.20, 0.20, 'T1\n(Failed)', ha='center', va='center', fontsize=9, 
        fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ff4444', 
                 edgecolor='black', linewidth=2, alpha=0.85))

# ============================================================================
# PANEL 2: GENERALIZATION GAP HORIZONTAL BAR CHART
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

valid_gap_trials = [t for t in trial_names if TRIALS_METRICS[t]['gap'] is not None]
gaps = [TRIALS_METRICS[t]['gap'] for t in valid_gap_trials]
gap_colors = [colors_by_status[TRIALS_METRICS[t]['status']] for t in valid_gap_trials]

y_pos = np.arange(len(valid_gap_trials))
bars = ax2.barh(y_pos, gaps, color=gap_colors, edgecolor='black', 
               linewidth=2, alpha=0.85, height=0.6)

# Add value labels
for i, (bar, gap) in enumerate(zip(bars, gaps)):
    width = bar.get_width()
    label_x = width + 2
    ax2.text(label_x, bar.get_y() + bar.get_height()/2, f'{gap:.1f}%',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Add threshold zones
ax2.axvline(x=1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (<1%)')
ax2.axvline(x=15, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Acceptable (<15%)')
ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical (>50%)')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(valid_gap_trials, fontsize=11, fontweight='bold')
ax2.set_xlabel('Generalization Gap (%)', fontsize=14, fontweight='bold')
ax2.set_title('(B) Generalization Gap Analysis\n(Lower is Better)', 
             fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
ax2.set_xlim(0, 70)

# ============================================================================
# PANEL 3: BENCHMARK ACHIEVEMENT PERCENTAGE
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

valid_bench_trials = [t for t in trial_names if TRIALS_METRICS[t]['benchmark'] is not None]
benchmarks = [TRIALS_METRICS[t]['benchmark'] for t in valid_bench_trials]
bench_colors = [colors_by_status[TRIALS_METRICS[t]['status']] for t in valid_bench_trials]

bars = ax3.bar(range(len(valid_bench_trials)), benchmarks, color=bench_colors,
              edgecolor='black', linewidth=2.5, alpha=0.85, width=0.65)

# Add 3D effect
for bar in bars:
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), 
                          shadow_rgbFace='gray', alpha=0.3)])

# Add value labels
for i, (bar, bench) in enumerate(zip(bars, benchmarks)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 1.5,
            f'{bench:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.95, edgecolor='black', linewidth=1.5))

ax3.axhline(y=100, color='gold', linestyle='--', linewidth=3, 
           label='Reference Benchmark (100%)', zorder=0)
ax3.axhline(y=70, color='green', linestyle=':', linewidth=2, 
           alpha=0.7, label='Good Performance (>70%)', zorder=0)

ax3.set_xticks(range(len(valid_bench_trials)))
ax3.set_xticklabels([t.split()[1] for t in valid_bench_trials], 
                    fontsize=11, fontweight='bold')
ax3.set_ylabel('Benchmark Achievement (%)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Trial Number', fontsize=14, fontweight='bold')
ax3.set_title('(C) Benchmark Achievement\nBoreale et al. (2024) = 100%', 
             fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=10, loc='lower right')
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_ylim(0, 110)

# ============================================================================
# PANEL 4: ALL TRIALS R² COMPARISON (SIDE-BY-SIDE)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :2])

x = np.arange(len(trial_names))
width = 0.35

val_all = [TRIALS_METRICS[t]['val'] for t in trial_names]
test_all = [TRIALS_METRICS[t]['test'] for t in trial_names]

bars1 = ax4.bar(x - width/2, val_all, width, label='Validation R²',
               color='#74b9ff', edgecolor='black', linewidth=2, alpha=0.85)
bars2 = ax4.bar(x + width/2, test_all, width, label='Test R²',
               color=colors, edgecolor='black', linewidth=2, alpha=0.85)

# Add 3D effects
for bar in bars1:
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(1.5, -1.5), 
                          shadow_rgbFace='blue', alpha=0.2)])
for bar in bars2:
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(1.5, -1.5), 
                          shadow_rgbFace='gray', alpha=0.3)])

# Add value labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                f'{val_all[i]:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.015,
                f'{test_all[i]:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)

ax4.axhline(y=BENCHMARK_R2, color='gold', linestyle='--', linewidth=3, 
           label=f'Benchmark (R²={BENCHMARK_R2})', zorder=0)

ax4.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax4.set_xlabel('Trial Number', fontsize=14, fontweight='bold')
ax4.set_title('(D) Complete R² Comparison: Validation vs Test\nAll 8 Trials Side-by-Side', 
             fontsize=14, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels([t.split()[1] for t in trial_names], fontsize=11, fontweight='bold')
ax4.legend(fontsize=12, loc='upper left', ncol=3)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
ax4.set_ylim(-0.1, 0.85)

# ============================================================================
# PANEL 5: TRIAL STATUS CLASSIFICATION PIE CHART
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])

status_counts = {}
for trial in trial_names:
    status = TRIALS_METRICS[trial]['status']
    status_counts[status] = status_counts.get(status, 0) + 1

labels = list(status_counts.keys())
sizes = list(status_counts.values())
pie_colors = [colors_by_status[s] for s in labels]

wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=pie_colors,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'},
                                    wedgeprops={'edgecolor': 'black', 'linewidth': 2.5})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

ax5.set_title('(E) Trial Status Distribution\nQuality Classification', 
             fontsize=14, fontweight='bold', pad=15)

# Add counts in legend
legend_labels = [f"{label}: {count} trial{'s' if count > 1 else ''}" 
                for label, count in zip(labels, sizes)]
ax5.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=10)

# ============================================================================
# PANEL 6: COMPLETE METRICS TABLE
# ============================================================================
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

# Create table data
table_data = [['Trial', 'Validation R²', 'Test R²', 'Gap (%)', 
               'Benchmark (%)', 'Status']]

for trial in trial_names:
    metrics = TRIALS_METRICS[trial]
    gap_str = f"{metrics['gap']:.1f}" if metrics['gap'] is not None else "N/A"
    bench_str = f"{metrics['benchmark']:.1f}" if metrics['benchmark'] is not None else "N/A"
    
    row = [
        trial,
        f"{metrics['val']:.4f}",
        f"{metrics['test']:.4f}",
        gap_str,
        bench_str,
        metrics['status']
    ]
    table_data.append(row)

# Create table
table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                 bbox=[0.05, 0.1, 0.9, 0.85])

table.auto_set_font_size(False)
table.set_fontsize(11)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#2d3436')
    cell.set_text_props(weight='bold', color='white', fontsize=12)
    cell.set_edgecolor('white')
    cell.set_linewidth(2)

# Style data rows with colors
for i, trial in enumerate(trial_names, 1):
    status = TRIALS_METRICS[trial]['status']
    row_color = colors_by_status[status]
    
    for j in range(6):
        cell = table[(i, j)]
        if j == 5:  # Status column
            cell.set_facecolor(row_color)
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f8f9fa')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

ax6.set_title('(F) Complete Evaluation Metrics Table\nComprehensive Summary of All Trials', 
             fontsize=14, fontweight='bold', pad=20)

# Main title
fig.suptitle('Figure 4: Comprehensive Metrics Analysis (All 8 Trials)\nComplete Evaluation Results | Reference: Boreale et al. (2024)', 
             fontsize=18, fontweight='bold', y=0.97)

# Save figure
plt.savefig(f"{OUTPUT_DIR}/figure4_comprehensive_metrics.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: figure4_comprehensive_metrics.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 4 COMPLETE")
print("="*80)
print("\nVERIFICATION:")
print("   [OK] Scatter plot (Validation vs Test)")
print("   [OK] Generalization gap analysis")
print("   [OK] Benchmark achievement chart")
print("   [OK] Side-by-side R² comparison")
print("   [OK] Status classification")
print("   [OK] Complete metrics table")
print("\n" + "="*80)
