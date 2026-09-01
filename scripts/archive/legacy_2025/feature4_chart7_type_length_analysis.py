"""
FEATURE 4 - CHART 7
Highway Type vs Road Length Analysis

Detailed analysis of road length distribution across highway types
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

print("\nCHART 7: Highway Type vs Road Length Analysis")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
road_length = graph.x[:n_active, 5].numpy()

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Panel 1: Box plots (log scale)
ax1 = axes[0, 0]
length_by_type = [road_length[highway_type == code] for code in type_codes_sorted]
bp = ax1.boxplot(length_by_type, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Road Length (m)', fontsize=10, fontweight='bold')
ax1.set_title('Road Length Distribution by Type (Log Scale)', fontsize=11, fontweight='bold', pad=15)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Add median labels
medians = [np.median(data) for data in length_by_type]
for i, median in enumerate(medians):
    ax1.text(i+1, median, f'{median:.0f}m', ha='center', va='bottom', 
             fontsize=7, fontweight='bold', color='darkred')

# Panel 2: Histogram of lengths by type (top 5)
ax2 = axes[0, 1]
for i, (code, name) in enumerate(zip(type_codes_sorted[:5], type_names_sorted[:5])):
    data = road_length[highway_type == code]
    ax2.hist(data, bins=50, alpha=0.5, label=name, color=COLORS_11[i], 
             edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax2.set_title('Length Distribution (Top 5 Types)', fontsize=11, fontweight='bold', pad=15)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 500)  # Focus on typical range

# Panel 3: Mean length with error bars
ax3 = axes[1, 0]
means = [np.mean(data) for data in length_by_type]
stds = [np.std(data) for data in length_by_type]

y_pos = np.arange(len(type_names_sorted))
bars = ax3.barh(y_pos, means, xerr=stds, color=COLORS_11[:len(type_names_sorted)], 
                alpha=0.8, capsize=5, edgecolor='black')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(type_names_sorted, fontsize=9)
ax3.set_xlabel('Mean Road Length (m)', fontsize=10, fontweight='bold')
ax3.set_title('Mean Road Length by Type', fontsize=11, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    ax3.text(mean + std, i, f' {mean:.1f}m', va='center', fontsize=8)

# Panel 4: Length categories by type
ax4 = axes[1, 1]
length_categories = {
    'Very Short (<50m)': (0, 50),
    'Short (50-100m)': (50, 100),
    'Medium (100-200m)': (100, 200),
    'Long (200-500m)': (200, 500),
    'Very Long (500-1000m)': (500, 1000),
    'Extra Long (>1000m)': (1000, np.inf)
}

category_data = []
for code in type_codes_sorted:
    lengths = road_length[highway_type == code]
    counts = []
    for cat_name, (low, high) in length_categories.items():
        count = np.sum((lengths >= low) & (lengths < high))
        pct = (count / len(lengths)) * 100
        counts.append(pct)
    category_data.append(counts)

category_data = np.array(category_data)
category_names = list(length_categories.keys())

# Stacked bar chart
bottom = np.zeros(len(type_names_sorted))
category_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']

for i, (cat_name, color) in enumerate(zip(category_names, category_colors)):
    bars = ax4.barh(range(len(type_names_sorted)), category_data[:, i], 
                    left=bottom, label=cat_name, color=color, alpha=0.8)
    bottom += category_data[:, i]
    
    # Add percentage labels for significant segments
    for j, val in enumerate(category_data[:, i]):
        if val > 5:  # Only label if >5%
            x = bottom[j] - val/2
            ax4.text(x, j, f'{val:.0f}%', ha='center', va='center', 
                    fontsize=6, fontweight='bold', color='white')

ax4.set_yticks(range(len(type_names_sorted)))
ax4.set_yticklabels(type_names_sorted, fontsize=9)
ax4.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
ax4.set_title('Length Categories Distribution by Type', fontsize=11, fontweight='bold', pad=15)
ax4.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
ax4.set_xlim(0, 100)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
chart_path = 'feature4_chart7_type_length_analysis.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

# Summary
print("\nKey Findings:")
print(f"  Longest mean: {type_names_sorted[np.argmax(means)]} ({max(means):.1f} m)")
print(f"  Shortest mean: {type_names_sorted[np.argmin(means)]} ({min(means):.1f} m)")
print(f"  Overall range: {road_length.min():.1f} - {road_length.max():.1f} m")

# Find type with most very short roads
very_short_pcts = category_data[:, 0]
print(f"  Most very short roads: {type_names_sorted[np.argmax(very_short_pcts)]} ({max(very_short_pcts):.1f}% <50m)")

# Find type with most long roads
long_pcts = category_data[:, -1]
print(f"  Most extra long roads: {type_names_sorted[np.argmax(long_pcts)]} ({max(long_pcts):.1f}% >1000m)")
