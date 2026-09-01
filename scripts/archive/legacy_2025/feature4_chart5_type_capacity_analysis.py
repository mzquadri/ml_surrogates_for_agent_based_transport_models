"""
FEATURE 4 - CHART 5
Highway Type vs Capacity Analysis

Detailed analysis of capacity distribution across highway types
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from IPython.display import Image, display

# Setup
data_dir = Path('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct')
batch_path = data_dir / 'datalist_batch_1.pt'

HW_MAPPING = {
    -1: 'Unknown', 0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary',
    4: 'Tertiary', 5: 'Residential', 6: 'PT', 7: 'Service', 
    8: 'Living Street', 9: 'Motorway Link'
}

COLORS_11 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
             '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']

print("\nCHART 5: Highway Type vs Capacity Analysis")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
capacity = graph.x[:n_active, 1].numpy()

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Panel 1: Box plots
ax1 = axes[0, 0]
capacity_by_type = [capacity[highway_type == code] for code in type_codes_sorted]
bp = ax1.boxplot(capacity_by_type, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax1.set_title('Capacity Distribution by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)

# Add statistics
medians = [np.median(data) for data in capacity_by_type]
for i, median in enumerate(medians):
    ax1.text(i+1, median, f'{median:.0f}', ha='center', va='bottom', 
             fontsize=7, fontweight='bold', color='darkred')

# Panel 2: Mean and std bars
ax2 = axes[0, 1]
means = [np.mean(data) for data in capacity_by_type]
stds = [np.std(data) for data in capacity_by_type]
x_pos = np.arange(len(type_names_sorted))
bars = ax2.bar(x_pos, means, yerr=stds, color=COLORS_11[:len(type_names_sorted)], 
               alpha=0.8, capsize=5, edgecolor='black')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Mean Capacity (veh/h)', fontsize=10, fontweight='bold')
ax2.set_title('Mean Capacity with Standard Deviation', fontsize=11, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    ax2.text(i, mean + std, f'{mean:.0f}', ha='center', va='bottom', 
             fontsize=8, fontweight='bold')

# Panel 3: Capacity ranges by type
ax3 = axes[1, 0]
ranges = [(np.min(data), np.max(data), np.max(data) - np.min(data)) 
          for data in capacity_by_type]
mins, maxs, spans = zip(*ranges)

y_pos = np.arange(len(type_names_sorted))
ax3.barh(y_pos, spans, left=mins, color=COLORS_11[:len(type_names_sorted)], alpha=0.6)
ax3.scatter(medians, y_pos, color='red', s=100, zorder=3, marker='D', 
            edgecolors='black', linewidths=2, label='Median')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(type_names_sorted, fontsize=9)
ax3.set_xlabel('Capacity (veh/h)', fontsize=10, fontweight='bold')
ax3.set_title('Capacity Range by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3)
ax3.legend(fontsize=9)

# Add range labels
for i, (min_val, max_val) in enumerate(zip(mins, maxs)):
    ax3.text(max_val, i, f' {min_val:.0f}-{max_val:.0f}', va='center', fontsize=7)

# Panel 4: Statistics table
ax4 = axes[1, 1]
ax4.axis('off')

stats_data = []
stats_data.append(['Type', 'Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'CV'])
stats_data.append(['', '', '(veh/h)', '(veh/h)', '(veh/h)', '(veh/h)', '(veh/h)', ''])

for i, (code, name) in enumerate(zip(type_codes_sorted, type_names_sorted)):
    data = capacity_by_type[i]
    count = len(data)
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    cv = std / mean if mean > 0 else 0
    
    stats_data.append([
        name[:12], f'{count:,}', f'{mean:.0f}', f'{median:.0f}', 
        f'{std:.0f}', f'{min_val:.0f}', f'{max_val:.0f}', f'{cv:.2f}'
    ])

# Create table
table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Style header
for i in range(8):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')
    cell = table[(1, i)]
    cell.set_facecolor('#D9E1F2')
    cell.set_text_props(style='italic', fontsize=7)

# Color rows by type
for i, color in enumerate(COLORS_11[:len(type_names_sorted)]):
    for j in range(8):
        table[(i+2, j)].set_facecolor(color)
        table[(i+2, j)].set_alpha(0.3)

ax4.set_title('Capacity Statistics by Highway Type', fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
chart_path = 'feature4_chart5_type_capacity_analysis.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

# Summary
print("\nKey Findings:")
print(f"  Highest mean capacity: {type_names_sorted[np.argmax(means)]} ({max(means):.0f} veh/h)")
print(f"  Lowest mean capacity: {type_names_sorted[np.argmin(means)]} ({min(means):.0f} veh/h)")

# Calculate CV only for non-zero means
cvs = [s/m if m > 0 else 0 for s, m in zip(stds, means)]
valid_cvs = [(i, cv) for i, cv in enumerate(cvs) if cv > 0]
if valid_cvs:
    max_cv_idx, max_cv = max(valid_cvs, key=lambda x: x[1])
    min_cv_idx, min_cv = min(valid_cvs, key=lambda x: x[1])
    print(f"  Highest variability: {type_names_sorted[max_cv_idx]} (CV: {max_cv:.2f})")
    print(f"  Most consistent: {type_names_sorted[min_cv_idx]} (CV: {min_cv:.2f})")
