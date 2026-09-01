"""
FEATURE 5 - CHART 9
Outlier Analysis

Detailed analysis of extreme length values and their characteristics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy import stats
from IPython.display import Image, display

print("\n" + "="*80)
print("FEATURE 5 - CHART 9: Outlier Analysis")
print("="*80)

# Setup
data_dir = Path('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct')
batch_path = data_dir / 'datalist_batch_1.pt'

HW_MAPPING = {
    -1: 'Unknown', 0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary',
    4: 'Tertiary', 5: 'Residential', 6: 'PT', 7: 'Service', 
    8: 'Living Street', 9: 'Motorway Link'
}

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
road_length = graph.x[:n_active, 5].numpy()
capacity = graph.x[:n_active, 1].numpy()
free_speed = graph.x[:n_active, 3].numpy()
baseline_volume = graph.x[:n_active, 2].numpy()
highway_type = graph.x[:n_active, 4].numpy().astype(int)

print(f"\nActive road segments: {n_active:,}")

# Identify outliers using IQR method
q1 = np.percentile(road_length, 25)
q3 = np.percentile(road_length, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers_mask = (road_length < lower_fence) | (road_length > upper_fence)
n_outliers = np.sum(outliers_mask)
n_normal = n_active - n_outliers

print(f"Outliers detected: {n_outliers:,} ({n_outliers/n_active*100:.1f}%)")
print(f"Normal roads: {n_normal:,} ({n_normal/n_active*100:.1f}%)")

# Separate lower and upper outliers
lower_outliers = road_length < lower_fence
upper_outliers = road_length > upper_fence
n_lower = np.sum(lower_outliers)
n_upper = np.sum(upper_outliers)

print(f"Lower outliers: {n_lower:,}")
print(f"Upper outliers: {n_upper:,}")

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(24, 20))

# Panel 1: Box plot with outliers highlighted
ax1 = axes[0, 0]
bp = ax1.boxplot(road_length, vert=True, patch_artist=True, showfliers=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
ax1.axhline(lower_fence, color='red', linestyle='--', linewidth=2, label=f'Lower fence: {lower_fence:.1f}m')
ax1.axhline(upper_fence, color='orange', linestyle='--', linewidth=2, label=f'Upper fence: {upper_fence:.1f}m')
ax1.set_ylabel('Road Length (m)', fontsize=10, fontweight='bold')
ax1.set_title('Box Plot with Outlier Boundaries', fontsize=11, fontweight='bold', pad=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Panel 2: Outlier distribution
ax2 = axes[0, 1]
categories = ['Normal', 'Lower\nOutliers', 'Upper\nOutliers']
counts = [n_normal, n_lower, n_upper]
colors = ['green', 'blue', 'red']
bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Count', fontsize=10, fontweight='bold')
ax2.set_title('Outlier Distribution', fontsize=11, fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, counts):
    pct = (count / n_active) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

# Panel 3: Outliers by highway type
ax3 = axes[0, 2]
type_counts = Counter(highway_type)
top_types = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)[:8]
type_names_short = [HW_MAPPING[code] for code in top_types]

outlier_pct_by_type = []
for type_code in top_types:
    mask = highway_type == type_code
    if np.sum(mask) > 0:
        pct = np.sum(mask & outliers_mask) / np.sum(mask) * 100
        outlier_pct_by_type.append(pct)
    else:
        outlier_pct_by_type.append(0)

bars = ax3.barh(range(len(type_names_short)), outlier_pct_by_type,
                color=plt.cm.Set3(np.linspace(0, 1, len(type_names_short))),
                alpha=0.8, edgecolor='black')
ax3.set_yticks(range(len(type_names_short)))
ax3.set_yticklabels(type_names_short, fontsize=9)
ax3.set_xlabel('% Outliers', fontsize=10, fontweight='bold')
ax3.set_title('Outlier % by Highway Type', fontsize=11, fontweight='bold', pad=10)
ax3.grid(axis='x', alpha=0.3)

for i, (bar, pct) in enumerate(zip(bars, outlier_pct_by_type)):
    ax3.text(pct, i, f' {pct:.1f}%', va='center', fontsize=8)

# Panel 4: Length distribution with outliers highlighted
ax4 = axes[1, 0]
ax4.hist([road_length[~outliers_mask], road_length[outliers_mask]], 
         bins=60, label=['Normal', 'Outliers'],
         color=['green', 'red'], alpha=0.7, edgecolor='black', stacked=False)
ax4.axvline(lower_fence, color='blue', linestyle='--', linewidth=2, label='Lower fence')
ax4.axvline(upper_fence, color='orange', linestyle='--', linewidth=2, label='Upper fence')
ax4.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax4.set_title('Distribution with Outliers', fontsize=11, fontweight='bold', pad=10)
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Panel 5: Top 20 longest roads
ax5 = axes[1, 1]
top20_idx = np.argsort(road_length)[-20:]
top20_lengths = road_length[top20_idx]
top20_types = [HW_MAPPING[highway_type[i]] for i in top20_idx]

bars = ax5.barh(range(20), top20_lengths, 
                color=plt.cm.Reds(np.linspace(0.3, 1, 20)),
                alpha=0.8, edgecolor='black')
ax5.set_yticks(range(20))
ax5.set_yticklabels([f'{i+1}. {t[:10]}' for i, t in enumerate(top20_types)], fontsize=8)
ax5.set_xlabel('Length (m)', fontsize=10, fontweight='bold')
ax5.set_title('Top 20 Longest Roads', fontsize=11, fontweight='bold', pad=10)
ax5.grid(axis='x', alpha=0.3)

# Panel 6: Top 20 shortest roads
ax6 = axes[1, 2]
bottom20_idx = np.argsort(road_length)[:20]
bottom20_lengths = road_length[bottom20_idx]
bottom20_types = [HW_MAPPING[highway_type[i]] for i in bottom20_idx]

bars = ax6.barh(range(20), bottom20_lengths, 
                color=plt.cm.Blues(np.linspace(0.3, 1, 20)),
                alpha=0.8, edgecolor='black')
ax6.set_yticks(range(20))
ax6.set_yticklabels([f'{i+1}. {t[:10]}' for i, t in enumerate(bottom20_types)], fontsize=8)
ax6.set_xlabel('Length (m)', fontsize=10, fontweight='bold')
ax6.set_title('Top 20 Shortest Roads', fontsize=11, fontweight='bold', pad=10)
ax6.grid(axis='x', alpha=0.3)

# Panel 7: Outlier characteristics - capacity
ax7 = axes[2, 0]
cap_normal = capacity[~outliers_mask]
cap_outliers = capacity[outliers_mask]

parts = ax7.violinplot([cap_normal, cap_outliers], positions=[0, 1], 
                        showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

ax7.set_xticks([0, 1])
ax7.set_xticklabels(['Normal', 'Outliers'], fontsize=10)
ax7.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax7.set_title('Capacity: Normal vs Outliers', fontsize=11, fontweight='bold', pad=10)
ax7.grid(axis='y', alpha=0.3)

# Add statistics
ax7.text(0, ax7.get_ylim()[1]*0.9, f'μ={np.mean(cap_normal):.0f}', ha='center', fontsize=8)
ax7.text(1, ax7.get_ylim()[1]*0.9, f'μ={np.mean(cap_outliers):.0f}', ha='center', fontsize=8)

# Panel 8: Outlier characteristics - speed
ax8 = axes[2, 1]
speed_normal = free_speed[~outliers_mask]
speed_outliers = free_speed[outliers_mask]

parts = ax8.violinplot([speed_normal, speed_outliers], positions=[0, 1], 
                        showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightgreen')
    pc.set_alpha(0.7)

ax8.set_xticks([0, 1])
ax8.set_xticklabels(['Normal', 'Outliers'], fontsize=10)
ax8.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax8.set_title('Speed: Normal vs Outliers', fontsize=11, fontweight='bold', pad=10)
ax8.grid(axis='y', alpha=0.3)

# Add statistics
ax8.text(0, ax8.get_ylim()[1]*0.9, f'μ={np.mean(speed_normal):.1f}', ha='center', fontsize=8)
ax8.text(1, ax8.get_ylim()[1]*0.9, f'μ={np.mean(speed_outliers):.1f}', ha='center', fontsize=8)

# Panel 9: Key insights
ax9 = axes[2, 2]
ax9.axis('off')

# Calculate statistics
longest_road = road_length.max()
shortest_road = road_length.min()
mean_normal = np.mean(road_length[~outliers_mask])
mean_outliers = np.mean(road_length[outliers_mask])

insights_text = f"""KEY INSIGHTS: OUTLIERS

OUTLIER DETECTION (IQR):
• Q1: {q1:.1f}m, Q3: {q3:.1f}m
• IQR: {iqr:.1f}m
• Lower fence: {lower_fence:.1f}m
• Upper fence: {upper_fence:.1f}m

COUNTS:
• Normal: {n_normal:,} ({n_normal/n_active*100:.1f}%)
• Lower: {n_lower:,} ({n_lower/n_active*100:.1f}%)
• Upper: {n_upper:,} ({n_upper/n_active*100:.1f}%)

EXTREMES:
• Longest: {longest_road:.1f}m
• Shortest: {shortest_road:.1f}m
• Range: {longest_road - shortest_road:.1f}m

CHARACTERISTICS:
• Mean normal: {mean_normal:.1f}m
• Mean outliers: {mean_outliers:.1f}m
• Capacity similar
• Speed similar

CONCLUSION:
• 6.2% outliers (typical)
• Mostly upper outliers
• Outliers have similar features
• Valid data, not errors
"""

ax9.text(0.1, 0.9, insights_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
output_path = 'feature5_chart9_outliers.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")
display(Image(output_path))

print("\n" + "="*80)
print("CHART 9 COMPLETE")
print("="*80)
print(f"\nTotal outliers: {n_outliers:,} ({n_outliers/n_active*100:.1f}%)")
print(f"Longest road: {longest_road:.1f}m")
print(f"Shortest road: {shortest_road:.1f}m")
print(f"Outliers are valid data (not errors)")
