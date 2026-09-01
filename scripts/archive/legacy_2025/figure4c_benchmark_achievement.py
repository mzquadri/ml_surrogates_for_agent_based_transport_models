"""
FIGURE 4C: BENCHMARK ACHIEVEMENT PERCENTAGE
Comparison with Boreale et al. (2024) Reference (All 7 Valid Trials)

Reference: Boreale et al. (2024) R² = 0.76 = 100%
This Work: Best = 78.4% (Trial 8)

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure4c_benchmark_achievement.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Benchmark achievement data (Trial 1 excluded - failed)
TRIALS_BENCHMARK = {
    'Trial 2': {'benchmark': 67.3, 'test_r2': 0.5117, 'status': 'OK', 'color': '#a29bfe'},
    'Trial 3': {'benchmark': 29.6, 'test_r2': 0.2246, 'status': 'OVERFIT', 'color': '#fd79a8'},
    'Trial 4': {'benchmark': 31.9, 'test_r2': 0.2426, 'status': 'OVERFIT', 'color': '#6c5ce7'},
    'Trial 5': {'benchmark': 73.1, 'test_r2': 0.5553, 'status': 'EXCELLENT', 'color': '#ffd93d'},
    'Trial 6': {'benchmark': 68.7, 'test_r2': 0.5223, 'status': 'EXCELLENT', 'color': '#ff9770'},
    'Trial 7': {'benchmark': 72.0, 'test_r2': 0.5471, 'status': 'EXCELLENT', 'color': '#ff7f50'},
    'Trial 8': {'benchmark': 78.4, 'test_r2': 0.5957, 'status': 'BEST', 'color': '#51cf66'}
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" FIGURE 4C: BENCHMARK ACHIEVEMENT")
print("="*80)
print("\nComparing with reference benchmark...")
print(f"Reference: Boreale et al. (2024) R² = {BENCHMARK_R2}")
print("Valid Trials: 7 (Trial 1 excluded)")
print("\n" + "="*80 + "\n")

# Create figure
fig, ax = plt.subplots(figsize=(16, 11))
fig.patch.set_facecolor('white')

trial_names = list(TRIALS_BENCHMARK.keys())
benchmarks = [TRIALS_BENCHMARK[t]['benchmark'] for t in trial_names]
colors = [TRIALS_BENCHMARK[t]['color'] for t in trial_names]
statuses = [TRIALS_BENCHMARK[t]['status'] for t in trial_names]
test_r2s = [TRIALS_BENCHMARK[t]['test_r2'] for t in trial_names]

x_pos = np.arange(len(trial_names))

# Vertical bars
bars = ax.bar(x_pos, benchmarks, color=colors, 
             edgecolor='black', linewidth=3, alpha=0.9, width=0.7)

# Add 3D effect
for bar in bars:
    bar.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(3, -3), shadow_rgbFace='gray', alpha=0.4)])

# Value labels on top
for i, (bar, bench, status, test_r2) in enumerate(zip(bars, benchmarks, statuses, test_r2s)):
    height = bar.get_height()
    
    # Percentage
    ax.text(bar.get_x() + bar.get_width()/2, height + 2,
           f'{bench:.1f}%',
           ha='center', va='bottom', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='black', linewidth=2, alpha=0.95))
    
    # Test R² inside bar
    if height > 25:
        ax.text(bar.get_x() + bar.get_width()/2, height/2,
               f'R²={test_r2:.3f}\n{status}',
               ha='center', va='center', fontsize=11, fontweight='bold', 
               color='white', linespacing=1.4)

# Reference line (100%)
ax.axhline(y=100, color='gold', linestyle='--', linewidth=4, 
          label=f'Reference Benchmark: 100%\n(Boreale et al. 2024, R²={BENCHMARK_R2})', 
          zorder=0)

# Good performance threshold (70%)
ax.axhline(y=70, color='green', linestyle=':', linewidth=3, alpha=0.7,
          label='Good Performance Threshold (70%)', zorder=0)

# Labels and title
ax.set_xticks(x_pos)
ax.set_xticklabels([f"Trial {t.split()[1]}" for t in trial_names], 
                   fontsize=13, fontweight='bold')
ax.set_ylabel('Benchmark Achievement (%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Trial Number', fontsize=16, fontweight='bold')
ax.set_title('Figure 4C: Benchmark Achievement Comparison (7 Valid Trials)\n' +
            'Reference: Boreale et al. (2024) R² = 0.76 (10,000 scenarios) = 100%\n' +
            'This Work: 1,000 scenarios (10% data)', 
            fontsize=17, fontweight='bold', pad=25)

# Grid and legend
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)
ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.set_ylim(0, 115)

# Summary box
summary_text = "DATA EFFICIENCY:\n"
summary_text += f"• Best Model: Trial 8\n"
summary_text += f"• Achievement: 78.4%\n"
summary_text += f"• Using only 10% data\n"
summary_text += f"• Performance Gap: 21.6%\n\n"
summary_text += f"TRIALS > 70%: 4/7 (57.1%)"

ax.text(0.98, 0.55, summary_text, transform=ax.transAxes,
       fontsize=12, fontweight='bold', verticalalignment='top',
       horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.9', facecolor='lightyellow', 
                edgecolor='orange', linewidth=3, alpha=0.95))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure4c_benchmark_achievement.png", 
           dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: figure4c_benchmark_achievement.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 4C COMPLETE")
print("="*80)
