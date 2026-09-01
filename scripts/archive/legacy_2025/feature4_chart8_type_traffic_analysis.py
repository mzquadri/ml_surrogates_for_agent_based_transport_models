"""
FEATURE 4 - CHART 8
Highway Type vs Traffic Coverage Analysis

Detailed analysis of traffic distribution across highway types
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

print("\nCHART 8: Highway Type vs Traffic Coverage Analysis")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
baseline_volume = graph.x[:n_active, 2].numpy()

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Panel 1: Traffic coverage percentage
ax1 = axes[0, 0]
volume_by_type = [baseline_volume[highway_type == code] for code in type_codes_sorted]
traffic_pcts = [(np.sum(data != 0) / len(data)) * 100 for data in volume_by_type]

bars = ax1.barh(range(len(type_names_sorted)), traffic_pcts, 
                color=COLORS_11[:len(type_names_sorted)], alpha=0.8, edgecolor='black')
ax1.set_yticks(range(len(type_names_sorted)))
ax1.set_yticklabels(type_names_sorted, fontsize=9)
ax1.set_xlabel('Roads with Traffic (%)', fontsize=10, fontweight='bold')
ax1.set_title('Traffic Coverage by Highway Type', fontsize=11, fontweight='bold', pad=15)
ax1.set_xlim(0, 100)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, val in enumerate(traffic_pcts):
    ax1.text(val, i, f' {val:.1f}%', va='center', fontsize=8, fontweight='bold')

# Panel 2: Pie chart of total traffic by type
ax2 = axes[0, 1]
total_traffic_by_type = [np.sum(np.abs(data[data != 0])) for data in volume_by_type]
# Filter out types with zero traffic
non_zero_mask = np.array(total_traffic_by_type) > 0
filtered_names = [name for name, mask in zip(type_names_sorted, non_zero_mask) if mask]
filtered_traffic = [val for val, mask in zip(total_traffic_by_type, non_zero_mask) if mask]
filtered_colors = [color for color, mask in zip(COLORS_11[:len(type_names_sorted)], non_zero_mask) if mask]

wedges, texts, autotexts = ax2.pie(filtered_traffic, labels=filtered_names, 
                                     autopct='%1.1f%%', startangle=90,
                                     colors=filtered_colors)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)
ax2.set_title('Total Traffic Volume Share by Type', fontsize=11, fontweight='bold')

# Panel 3: Traffic vs no-traffic comparison
ax3 = axes[1, 0]
with_traffic = [np.sum(data != 0) for data in volume_by_type]
without_traffic = [len(data) - np.sum(data != 0) for data in volume_by_type]

y_pos = np.arange(len(type_names_sorted))
width = 0.35

bars1 = ax3.barh(y_pos, with_traffic, width, label='With Traffic', 
                 color='green', alpha=0.7)
bars2 = ax3.barh(y_pos, [-x for x in without_traffic], width, label='Without Traffic', 
                 color='red', alpha=0.7)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(type_names_sorted, fontsize=9)
ax3.set_xlabel('Number of Roads', fontsize=10, fontweight='bold')
ax3.set_title('Roads With vs Without Traffic', fontsize=11, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(axis='x', alpha=0.3)
ax3.axvline(x=0, color='black', linewidth=1)

# Add count labels
for i, (w_traffic, wo_traffic) in enumerate(zip(with_traffic, without_traffic)):
    if w_traffic > 0:
        ax3.text(w_traffic, i, f' {w_traffic:,}', va='center', fontsize=7)
    if wo_traffic > 0:
        ax3.text(-wo_traffic, i, f'{wo_traffic:,} ', ha='right', va='center', fontsize=7)

# Panel 4: Statistics table
ax4 = axes[1, 1]
ax4.axis('off')

stats_data = []
stats_data.append(['Type', 'Total', 'With Traffic', 'Coverage', 'Mean Vol', 'Traffic Share'])
stats_data.append(['', 'Roads', 'Roads', '(%)', '(veh/h)', '(%)'])

total_network_traffic = sum(total_traffic_by_type)

for i, (code, name) in enumerate(zip(type_codes_sorted, type_names_sorted)):
    data = volume_by_type[i]
    total_roads = len(data)
    with_traffic_count = np.sum(data != 0)
    coverage = (with_traffic_count / total_roads) * 100
    mean_vol = np.mean(np.abs(data[data != 0])) if with_traffic_count > 0 else 0
    traffic_share = (total_traffic_by_type[i] / total_network_traffic) * 100 if total_network_traffic > 0 else 0
    
    stats_data.append([
        name[:12], f'{total_roads:,}', f'{with_traffic_count:,}', 
        f'{coverage:.1f}', f'{mean_vol:.0f}', f'{traffic_share:.1f}'
    ])

# Create table
table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Style header
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')
    cell = table[(1, i)]
    cell.set_facecolor('#D9E1F2')
    cell.set_text_props(style='italic', fontsize=7)

# Color rows by type
for i, color in enumerate(COLORS_11[:len(type_names_sorted)]):
    for j in range(6):
        table[(i+2, j)].set_facecolor(color)
        table[(i+2, j)].set_alpha(0.3)

ax4.set_title('Traffic Statistics by Highway Type', fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
chart_path = 'feature4_chart8_type_traffic_analysis.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

# Summary
print("\nKey Findings:")
print(f"  Highest traffic coverage: {type_names_sorted[np.argmax(traffic_pcts)]} ({max(traffic_pcts):.1f}%)")
print(f"  Lowest traffic coverage: {type_names_sorted[np.argmin(traffic_pcts)]} ({min(traffic_pcts):.1f}%)")
print(f"  Overall network coverage: {(np.sum(baseline_volume != 0) / n_active) * 100:.1f}%")
print(f"  Types with traffic: {sum([1 for pct in traffic_pcts if pct > 0])}/{len(traffic_pcts)}")
