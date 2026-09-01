"""
FEATURE 4 - CHART 6
Highway Type vs Free Speed Analysis

Detailed analysis of speed distribution across highway types
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

print("\nCHART 6: Highway Type vs Free Speed Analysis")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
free_speed = graph.x[:n_active, 3].numpy()

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Panel 1: Violin plots
ax1 = axes[0, 0]
speed_by_type = [free_speed[highway_type == code] for code in type_codes_sorted]
parts = ax1.violinplot(speed_by_type, positions=range(len(type_names_sorted)), 
                        showmedians=True, showextrema=True, widths=0.7)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLORS_11[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

ax1.set_xticks(range(len(type_names_sorted)))
ax1.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax1.set_title('Speed Distribution by Highway Type (Violin)', fontsize=11, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: CDF by type
ax2 = axes[0, 1]
for i, (code, name) in enumerate(zip(type_codes_sorted[:6], type_names_sorted[:6])):  # Top 6 types
    data = free_speed[highway_type == code]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf, label=name, color=COLORS_11[i], linewidth=2, alpha=0.8)

ax2.set_xlabel('Free Speed (km/h)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
ax2.set_title('Speed CDF by Highway Type (Top 6)', fontsize=11, fontweight='bold', pad=15)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.3)

# Add percentile markers
for pct in [0.25, 0.5, 0.75]:
    ax2.axhline(y=pct, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(ax2.get_xlim()[1], pct, f' P{int(pct*100)}', va='center', fontsize=8)

# Panel 3: Mean speed comparison
ax3 = axes[1, 0]
means = [np.mean(data) for data in speed_by_type]
medians = [np.median(data) for data in speed_by_type]

y_pos = np.arange(len(type_names_sorted))
width = 0.35

bars1 = ax3.barh(y_pos - width/2, means, width, label='Mean', 
                 color=COLORS_11[:len(type_names_sorted)], alpha=0.8)
bars2 = ax3.barh(y_pos + width/2, medians, width, label='Median', 
                 color=COLORS_11[:len(type_names_sorted)], alpha=0.5)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(type_names_sorted, fontsize=9)
ax3.set_xlabel('Speed (km/h)', fontsize=10, fontweight='bold')
ax3.set_title('Mean vs Median Speed by Type', fontsize=11, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (mean, median) in enumerate(zip(means, medians)):
    ax3.text(mean, i - width/2, f' {mean:.1f}', va='center', fontsize=7)
    ax3.text(median, i + width/2, f' {median:.1f}', va='center', fontsize=7)

# Panel 4: Speed categories by type
ax4 = axes[1, 1]
speed_categories = {
    'Very Slow (0-20)': (0, 20),
    'Slow (20-40)': (20, 40),
    'Moderate (40-60)': (40, 60),
    'Fast (60-90)': (60, 90),
    'Very Fast (90-130)': (90, 130),
    'Highway (>130)': (130, np.inf)
}

category_data = []
for code in type_codes_sorted:
    speeds = free_speed[highway_type == code]
    counts = []
    for cat_name, (low, high) in speed_categories.items():
        count = np.sum((speeds >= low) & (speeds < high))
        pct = (count / len(speeds)) * 100
        counts.append(pct)
    category_data.append(counts)

category_data = np.array(category_data)
category_names = list(speed_categories.keys())

# Stacked bar chart
bottom = np.zeros(len(type_names_sorted))
category_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']

for i, (cat_name, color) in enumerate(zip(category_names, category_colors)):
    bars = ax4.barh(range(len(type_names_sorted)), category_data[:, i], 
                    left=bottom, label=cat_name, color=color, alpha=0.8)
    bottom += category_data[:, i]

ax4.set_yticks(range(len(type_names_sorted)))
ax4.set_yticklabels(type_names_sorted, fontsize=9)
ax4.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
ax4.set_title('Speed Categories Distribution by Type', fontsize=11, fontweight='bold', pad=15)
ax4.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
ax4.set_xlim(0, 100)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
chart_path = 'feature4_chart6_type_speed_analysis.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

# Summary
print("\nKey Findings:")
print(f"  Highest mean speed: {type_names_sorted[np.argmax(means)]} ({max(means):.1f} km/h)")
print(f"  Lowest mean speed: {type_names_sorted[np.argmin(means)]} ({min(means):.1f} km/h)")
print(f"  Speed range: {free_speed.min():.1f} - {free_speed.max():.1f} km/h")

# Find type with most highway-speed roads
highway_speed_pcts = category_data[:, -1]
print(f"  Most high-speed roads: {type_names_sorted[np.argmax(highway_speed_pcts)]} ({max(highway_speed_pcts):.1f}% >130 km/h)")
