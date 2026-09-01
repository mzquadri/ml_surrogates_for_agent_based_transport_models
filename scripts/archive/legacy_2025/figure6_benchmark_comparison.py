"""
FIGURE 6: BENCHMARK COMPARISON WITH BOREALE ET AL. (2024)
Performance gap and data scaling analysis

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure6_benchmark_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trial 8 metrics
TRIAL8_R2 = 0.5957
TRIAL8_PEARSON = 0.7726

# Benchmark from Boreale et al. (2024)
BENCHMARK_R2 = 0.76
BENCHMARK_PEARSON = 0.87

print("="*80)
print(" GENERATING FIGURE 6: BENCHMARK COMPARISON")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 13))
plt.subplots_adjust(left=0.08, right=0.96, top=0.79, bottom=0.12, wspace=0.5)

# ============================================================================
# LEFT PANEL: R² vs Dataset Size
# ============================================================================
labels_bench = ['This Work\n(1,000 scenarios)', 'Boreale et al. (2024)\n(10,000 scenarios)']
r2_scores_data = [TRIAL8_R2, BENCHMARK_R2]
colors_bench = ['#51cf66', 'gold']

bars = ax1.bar(range(len(labels_bench)), r2_scores_data, color=colors_bench,
               edgecolor='black', linewidth=3, alpha=0.9, width=0.55)

ax1.set_ylabel('Test R² Score', fontsize=16, fontweight='bold')
ax1.set_xlabel('Dataset Configuration', fontsize=16, fontweight='bold')
ax1.set_title('(A) R² Performance vs Dataset Size\nScaling Analysis', 
              fontsize=17, fontweight='bold', pad=40)
ax1.set_xticks(range(len(labels_bench)))
ax1.set_xticklabels(labels_bench, fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.85)
ax1.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=1.2)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, r2_scores_data)):
    height = bar.get_height()
    percentage = (score / BENCHMARK_R2) * 100 if i == 0 else 100
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.025,
            f'R² = {score:.4f}\n({percentage:.1f}% of benchmark)',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

# Dataset info boxes
ax1.text(0, 0.08, 'Training: 800 samples\nTest: 100 samples', 
         ha='center', fontsize=12, fontweight='bold', transform=ax1.transData,
         bbox=dict(boxstyle='round', facecolor='#51cf66', alpha=0.4, edgecolor='black', linewidth=2))
ax1.text(1, 0.08, 'Training: 8,000 samples\nTest: ~2,000 samples', 
         ha='center', fontsize=12, fontweight='bold', transform=ax1.transData,
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.4, edgecolor='black', linewidth=2))

# ============================================================================
# RIGHT PANEL: Multi-Metric Comparison
# ============================================================================
metrics_comp = ['R²', 'Pearson\nCorrelation']
our_scores = [TRIAL8_R2, TRIAL8_PEARSON]
bench_scores = [BENCHMARK_R2, BENCHMARK_PEARSON]

x = np.arange(len(metrics_comp))
width = 0.38

bars1 = ax2.bar(x - width/2, our_scores, width, label='This Work (Trial 8)',
                color='#51cf66', edgecolor='black', linewidth=2, alpha=0.9)
bars2 = ax2.bar(x + width/2, bench_scores, width, label='Boreale et al. (2024)',
                color='gold', edgecolor='black', linewidth=2, alpha=0.9)

ax2.set_ylabel('Score Value', fontsize=16, fontweight='bold')
ax2.set_xlabel('Evaluation Metric', fontsize=16, fontweight='bold')
ax2.set_title('(B) Comprehensive Metrics Comparison\nThis Work vs Reference Benchmark', 
              fontsize=17, fontweight='bold', pad=40)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_comp, fontsize=15, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.legend(fontsize=14, loc='lower right', framealpha=0.95)
ax2.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=1.2)

# Add value labels
for i in range(len(metrics_comp)):
    height1 = our_scores[i]
    percentage = (our_scores[i] / bench_scores[i]) * 100
    ax2.text(i - width/2, height1 + 0.025, f'{our_scores[i]:.4f}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    height2 = bench_scores[i]
    ax2.text(i + width/2, height2 + 0.025, f'{bench_scores[i]:.4f}\n(100%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.suptitle('Figure 6: Benchmark Comparison with Boreale et al. (2024)\nPerformance Gap Analysis and Data Scaling Effects', 
             fontsize=19, fontweight='bold', y=0.93)

fig.savefig(f"{OUTPUT_DIR}/figure6_benchmark_comparison.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: figure6_benchmark_comparison.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 6 COMPLETE!")
print("="*80)
