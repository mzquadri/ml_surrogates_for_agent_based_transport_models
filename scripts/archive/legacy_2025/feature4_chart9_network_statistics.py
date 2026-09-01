"""
FEATURE 4 - CHART 9
Highway Type Network Statistics

Analysis of network topology and connectivity by highway type
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

print("\nCHART 9: Highway Type Network Statistics")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
edge_index = graph.edge_index.numpy()

# Calculate degree (number of connections) for each road
degrees = np.zeros(n_active, dtype=int)
for i in range(edge_index.shape[1]):
    src, dst = edge_index[:, i]
    if src < n_active:
        degrees[src] += 1
    if dst < n_active:
        degrees[dst] += 1

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Panel 1: Mean degree by type
ax1 = axes[0, 0]
degree_by_type = [degrees[highway_type == code] for code in type_codes_sorted]
mean_degrees = [np.mean(data) for data in degree_by_type]

bars = ax1.barh(range(len(type_names_sorted)), mean_degrees, 
                color=COLORS_11[:len(type_names_sorted)], alpha=0.8, edgecolor='black')
ax1.set_yticks(range(len(type_names_sorted)))
ax1.set_yticklabels(type_names_sorted, fontsize=9)
ax1.set_xlabel('Mean Degree (Connections)', fontsize=10, fontweight='bold')
ax1.set_title('Average Network Connectivity by Type', fontsize=11, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, val in enumerate(mean_degrees):
    ax1.text(val, i, f' {val:.1f}', va='center', fontsize=8, fontweight='bold')

# Panel 2: Degree distribution by type (box plots)
ax2 = axes[0, 1]
bp = ax2.boxplot(degree_by_type, tick_labels=type_names_sorted, patch_artist=True)
for patch, color in zip(bp['boxes'], COLORS_11):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Degree (Connections)', fontsize=10, fontweight='bold')
ax2.set_title('Connectivity Distribution by Type', fontsize=11, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Network role classification
ax3 = axes[1, 0]

# Classify roads by degree
role_thresholds = {
    'Isolated (0-2)': (0, 2),
    'Low Conn (3-5)': (3, 5),
    'Medium Conn (6-10)': (6, 10),
    'High Conn (11-20)': (11, 20),
    'Hub (>20)': (21, np.inf)
}

role_data = []
for code in type_codes_sorted:
    deg = degrees[highway_type == code]
    counts = []
    for role_name, (low, high) in role_thresholds.items():
        count = np.sum((deg >= low) & (deg <= high))
        pct = (count / len(deg)) * 100
        counts.append(pct)
    role_data.append(counts)

role_data = np.array(role_data)
role_names = list(role_thresholds.keys())

# Stacked bar chart
bottom = np.zeros(len(type_names_sorted))
role_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

for i, (role_name, color) in enumerate(zip(role_names, role_colors)):
    bars = ax3.barh(range(len(type_names_sorted)), role_data[:, i], 
                    left=bottom, label=role_name, color=color, alpha=0.8)
    bottom += role_data[:, i]
    
    # Add percentage labels for significant segments
    for j, val in enumerate(role_data[:, i]):
        if val > 8:  # Only label if >8%
            x = bottom[j] - val/2
            ax3.text(x, j, f'{val:.0f}%', ha='center', va='center', 
                    fontsize=6, fontweight='bold', color='white')

ax3.set_yticks(range(len(type_names_sorted)))
ax3.set_yticklabels(type_names_sorted, fontsize=9)
ax3.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
ax3.set_title('Network Role Distribution by Type', fontsize=11, fontweight='bold', pad=15)
ax3.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
ax3.set_xlim(0, 100)
ax3.grid(axis='x', alpha=0.3)

# Panel 4: Statistics table
ax4 = axes[1, 1]
ax4.axis('off')

stats_data = []
stats_data.append(['Type', 'Count', 'Mean Deg', 'Median', 'Max', 'Hubs', 'Isolated'])
stats_data.append(['', 'Roads', '', 'Deg', 'Deg', '(>20)', '(0-2)'])

for i, (code, name) in enumerate(zip(type_codes_sorted, type_names_sorted)):
    deg = degrees[highway_type == code]
    count = len(deg)
    mean_deg = np.mean(deg)
    median_deg = np.median(deg)
    max_deg = np.max(deg)
    hubs = np.sum(deg > 20)
    isolated = np.sum(deg <= 2)
    
    stats_data.append([
        name[:12], f'{count:,}', f'{mean_deg:.1f}', f'{int(median_deg)}', 
        f'{max_deg}', f'{hubs}', f'{isolated}'
    ])

# Create table
table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Style header
for i in range(7):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')
    cell = table[(1, i)]
    cell.set_facecolor('#D9E1F2')
    cell.set_text_props(style='italic', fontsize=7)

# Color rows by type
for i, color in enumerate(COLORS_11[:len(type_names_sorted)]):
    for j in range(7):
        table[(i+2, j)].set_facecolor(color)
        table[(i+2, j)].set_alpha(0.3)

ax4.set_title('Network Connectivity Statistics', fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
chart_path = 'feature4_chart9_network_statistics.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

# Summary
print("\nKey Findings:")
print(f"  Highest connectivity: {type_names_sorted[np.argmax(mean_degrees)]} ({max(mean_degrees):.1f} avg connections)")
print(f"  Lowest connectivity: {type_names_sorted[np.argmin(mean_degrees)]} ({min(mean_degrees):.1f} avg connections)")
print(f"  Total network edges: {edge_index.shape[1]:,}")
print(f"  Average degree: {np.mean(degrees):.1f}")

# Find type with most hubs
hub_counts = [np.sum(degrees[highway_type == code] > 20) for code in type_codes_sorted]
print(f"  Most hubs: {type_names_sorted[np.argmax(hub_counts)]} ({max(hub_counts)} hubs with >20 connections)")
