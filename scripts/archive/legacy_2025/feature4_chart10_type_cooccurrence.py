"""
FEATURE 4 - CHART 10
Highway Type Co-occurrence Analysis

Analysis of how different highway types connect to each other
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

print("\nCHART 10: Highway Type Co-occurrence Analysis")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
edge_index = graph.edge_index.numpy()

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]
n_types = len(type_codes_sorted)

# Create co-occurrence matrix
cooccurrence = np.zeros((n_types, n_types), dtype=int)

for i in range(edge_index.shape[1]):
    src, dst = edge_index[:, i]
    if src < n_active and dst < n_active:
        src_type = highway_type[src]
        dst_type = highway_type[dst]
        
        # Find indices in sorted list
        try:
            src_idx = type_codes_sorted.index(src_type)
            dst_idx = type_codes_sorted.index(dst_type)
            cooccurrence[src_idx, dst_idx] += 1
            if src_idx != dst_idx:
                cooccurrence[dst_idx, src_idx] += 1
        except ValueError:
            pass

# Normalize to percentages (row-wise)
cooccurrence_pct = np.zeros_like(cooccurrence, dtype=float)
for i in range(n_types):
    row_sum = np.sum(cooccurrence[i, :])
    if row_sum > 0:
        cooccurrence_pct[i, :] = (cooccurrence[i, :] / row_sum) * 100

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 18))

# Panel 1: Co-occurrence heatmap (counts)
ax1 = axes[0, 0]
im1 = ax1.imshow(cooccurrence, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(n_types))
ax1.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax1.set_yticks(range(n_types))
ax1.set_yticklabels(type_names_sorted, fontsize=9)
ax1.set_title('Type Co-occurrence Matrix (Connection Counts)', fontsize=11, fontweight='bold', pad=15)

# Add text annotations for significant values
for i in range(n_types):
    for j in range(n_types):
        val = cooccurrence[i, j]
        if val > 100:  # Only show significant connections
            text = ax1.text(j, i, f'{val:,}', ha='center', va='center',
                           fontsize=6, color='white' if val > cooccurrence.max()/2 else 'black')

cbar1 = plt.colorbar(im1, ax=ax1, label='Connection Count')

# Panel 2: Co-occurrence heatmap (percentages)
ax2 = axes[0, 1]
im2 = ax2.imshow(cooccurrence_pct, cmap='RdYlGn', aspect='auto', vmin=0, vmax=50)
ax2.set_xticks(range(n_types))
ax2.set_xticklabels(type_names_sorted, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(n_types))
ax2.set_yticklabels(type_names_sorted, fontsize=9)
ax2.set_title('Type Co-occurrence (Row-wise %)', fontsize=11, fontweight='bold', pad=15)

# Add text annotations
for i in range(n_types):
    for j in range(n_types):
        val = cooccurrence_pct[i, j]
        if val > 5:  # Only show significant percentages
            text = ax2.text(j, i, f'{val:.1f}%', ha='center', va='center',
                           fontsize=6, color='white' if val > 25 else 'black')

cbar2 = plt.colorbar(im2, ax=ax2, label='Percentage (%)')

# Panel 3: Same-type vs different-type connections
ax3 = axes[1, 0]

same_type_pct = [cooccurrence_pct[i, i] for i in range(n_types)]
diff_type_pct = [100 - val for val in same_type_pct]

y_pos = np.arange(n_types)
width = 0.4

bars1 = ax3.barh(y_pos - width/2, same_type_pct, width, label='Same Type', 
                 color='green', alpha=0.7)
bars2 = ax3.barh(y_pos + width/2, diff_type_pct, width, label='Different Type', 
                 color='blue', alpha=0.7)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(type_names_sorted, fontsize=9)
ax3.set_xlabel('Percentage of Connections (%)', fontsize=10, fontweight='bold')
ax3.set_title('Same-Type vs Cross-Type Connectivity', fontsize=11, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(axis='x', alpha=0.3)
ax3.set_xlim(0, 100)

# Add value labels
for i, (same, diff) in enumerate(zip(same_type_pct, diff_type_pct)):
    ax3.text(same, i - width/2, f' {same:.1f}%', va='center', fontsize=7)
    ax3.text(diff, i + width/2, f' {diff:.1f}%', va='center', fontsize=7)

# Panel 4: Top connections for each type
ax4 = axes[1, 1]
ax4.axis('off')

# For each type, find top 3 connections
insights_text = "TOP CONNECTIONS BY TYPE:\n\n"
for i, name in enumerate(type_names_sorted):
    # Get top 3 connections (excluding self)
    row = cooccurrence[i, :]
    sorted_indices = np.argsort(row)[::-1]
    
    connections = []
    for idx in sorted_indices:
        if idx != i and row[idx] > 0:  # Exclude self
            connections.append((type_names_sorted[idx], row[idx], cooccurrence_pct[i, idx]))
        if len(connections) >= 3:
            break
    
    insights_text += f"{name}:\n"
    for conn_name, count, pct in connections:
        insights_text += f"  → {conn_name}: {count:,} ({pct:.1f}%)\n"
    insights_text += "\n"

ax4.text(0.05, 0.95, insights_text, fontsize=8, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax4.set_title('Top 3 Connections per Type', fontsize=11, fontweight='bold', pad=15)

plt.tight_layout()
chart_path = 'feature4_chart10_type_cooccurrence.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

# Summary
print("\nKey Findings:")
print(f"  Highest same-type connectivity: {type_names_sorted[np.argmax(same_type_pct)]} ({max(same_type_pct):.1f}%)")
print(f"  Most cross-type connectivity: {type_names_sorted[np.argmin(same_type_pct)]} ({max(diff_type_pct):.1f}% different types)")
print(f"  Total connections analyzed: {np.sum(cooccurrence)//2:,}")
