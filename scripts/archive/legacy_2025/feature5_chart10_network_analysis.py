"""
FEATURE 5 - CHART 10
Network Analysis by Length

Analysis of network connectivity and patterns based on road length.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from IPython.display import Image, display

print("\n" + "="*80)
print("FEATURE 5 - CHART 10: Network Analysis by Length")
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
highway_type = graph.x[:n_active, 4].numpy().astype(int)
edge_index = graph.edge_index[:, :n_active].numpy()

print(f"\nActive road segments: {n_active:,}")
print(f"Edge connections: {edge_index.shape[1]:,}")

# Calculate node degrees (connectivity)
degrees = np.zeros(n_active)
for i in range(edge_index.shape[1]):
    source = edge_index[0, i]
    target = edge_index[1, i]
    if source < n_active:
        degrees[source] += 1
    if target < n_active:
        degrees[target] += 1

print(f"Mean degree: {np.mean(degrees):.2f}")

# Define length categories
categories = ['<50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']
ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, np.inf)]

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(24, 20))

# Panel 1: Network coverage by length
ax1 = axes[0, 0]
total_network_length = np.sum(road_length)
coverage_by_cat = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    cat_length = np.sum(road_length[mask])
    coverage_by_cat.append(cat_length)

colors_cat = plt.cm.viridis(np.linspace(0, 1, len(categories)))
bars = ax1.bar(range(len(categories)), coverage_by_cat, color=colors_cat, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Total Length (m)', fontsize=10, fontweight='bold')
ax1.set_title('Network Coverage by Length Category', fontsize=11, fontweight='bold', pad=10)
ax1.grid(axis='y', alpha=0.3)

for i, (bar, cov) in enumerate(zip(bars, coverage_by_cat)):
    pct = (cov / total_network_length) * 100
    ax1.text(i, cov, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 2: Mean degree by length category
ax2 = axes[0, 1]
mean_degrees = []
std_degrees = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        mean_degrees.append(np.mean(degrees[mask]))
        std_degrees.append(np.std(degrees[mask]))
    else:
        mean_degrees.append(0)
        std_degrees.append(0)

bars = ax2.bar(range(len(categories)), mean_degrees, yerr=std_degrees, capsize=5,
               color=colors_cat, alpha=0.8, edgecolor='black')
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Mean Degree', fontsize=10, fontweight='bold')
ax2.set_title('Connectivity by Length Category', fontsize=11, fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)

for i, (bar, deg) in enumerate(zip(bars, mean_degrees)):
    ax2.text(i, deg, f'{deg:.2f}', ha='center', va='bottom', fontsize=8)

# Panel 3: Length vs Degree scatter
ax3 = axes[0, 2]
ax3.scatter(road_length, degrees, alpha=0.3, s=5, c='steelblue')
ax3.set_xlabel('Road Length (m)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Degree (connections)', fontsize=10, fontweight='bold')
ax3.set_title('Length vs Connectivity', fontsize=11, fontweight='bold', pad=10)
ax3.grid(alpha=0.3)

corr = np.corrcoef(road_length, degrees)[0, 1]
ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
         fontsize=10, fontweight='bold', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Panel 4: Degree distribution by category
ax4 = axes[1, 0]
degree_by_cat = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        degree_by_cat.append(degrees[mask])
    else:
        degree_by_cat.append([0])

bp = ax4.boxplot(degree_by_cat, tick_labels=categories, patch_artist=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Degree', fontsize=10, fontweight='bold')
ax4.set_title('Degree Distribution by Length', fontsize=11, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)

# Panel 5: Cumulative network length
ax5 = axes[1, 1]
sorted_lengths = np.sort(road_length)[::-1]
cumulative_length = np.cumsum(sorted_lengths)
cumulative_pct = cumulative_length / total_network_length * 100
x_pct = np.arange(len(sorted_lengths)) / len(sorted_lengths) * 100

ax5.plot(x_pct, cumulative_pct, linewidth=2, color='darkblue')
ax5.fill_between(x_pct, cumulative_pct, alpha=0.3, color='lightblue')
ax5.set_xlabel('% of Roads (sorted by length)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Cumulative Network Length %', fontsize=10, fontweight='bold')
ax5.set_title('Network Length Contribution', fontsize=11, fontweight='bold', pad=10)
ax5.grid(alpha=0.3)
ax5.axhline(50, color='red', linestyle='--', linewidth=2, label='50% network')
ax5.axhline(80, color='orange', linestyle='--', linewidth=2, label='80% network')
ax5.legend(fontsize=9)

# Panel 6: Network length by highway type
ax6 = axes[1, 2]
type_counts = Counter(highway_type)
top_types = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)[:8]
type_names_short = [HW_MAPPING[code] for code in top_types]

network_length_by_type = []
for type_code in top_types:
    mask = highway_type == type_code
    network_length_by_type.append(np.sum(road_length[mask]))

bars = ax6.barh(range(len(type_names_short)), network_length_by_type,
                color=plt.cm.Set3(np.linspace(0, 1, len(type_names_short))),
                alpha=0.8, edgecolor='black')
ax6.set_yticks(range(len(type_names_short)))
ax6.set_yticklabels(type_names_short, fontsize=9)
ax6.set_xlabel('Total Length (m)', fontsize=10, fontweight='bold')
ax6.set_title('Network Length by Type', fontsize=11, fontweight='bold', pad=10)
ax6.grid(axis='x', alpha=0.3)

for i, (bar, length) in enumerate(zip(bars, network_length_by_type)):
    pct = (length / total_network_length) * 100
    ax6.text(length, i, f' {pct:.1f}%', va='center', fontsize=8)

# Panel 7: Length efficiency (length per connection)
ax7 = axes[2, 0]
efficiency_by_cat = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0 and np.sum(degrees[mask]) > 0:
        total_len = np.sum(road_length[mask])
        total_deg = np.sum(degrees[mask])
        efficiency_by_cat.append(total_len / total_deg)
    else:
        efficiency_by_cat.append(0)

bars = ax7.bar(range(len(categories)), efficiency_by_cat, 
               color=colors_cat, alpha=0.8, edgecolor='black')
ax7.set_xticks(range(len(categories)))
ax7.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax7.set_ylabel('Length per Connection (m)', fontsize=10, fontweight='bold')
ax7.set_title('Network Efficiency by Category', fontsize=11, fontweight='bold', pad=10)
ax7.grid(axis='y', alpha=0.3)

for i, (bar, eff) in enumerate(zip(bars, efficiency_by_cat)):
    ax7.text(i, eff, f'{eff:.1f}', ha='center', va='bottom', fontsize=8)

# Panel 8: Statistics table
ax8 = axes[2, 1]
ax8.axis('off')

# Calculate statistics
total_km = total_network_length / 1000
mean_length = np.mean(road_length)
median_length = np.median(road_length)
mean_degree = np.mean(degrees)

stats_text = f"""NETWORK STATISTICS

TOTAL NETWORK:
Length:         {total_km:.2f} km
Roads:          {n_active:,}
Connections:    {int(np.sum(degrees)/2):,}

LENGTH METRICS:
Mean length:    {mean_length:.1f} m
Median length:  {median_length:.1f} m
Total range:    {road_length.min():.1f} - {road_length.max():.1f} m

CONNECTIVITY:
Mean degree:    {mean_degree:.2f}
Max degree:     {degrees.max():.0f}
Min degree:     {degrees.min():.0f}

COVERAGE:
Top 20% roads contribute
{cumulative_pct[int(0.2*len(cumulative_pct))]:.1f}% of network length

Most length in category:
{categories[np.argmax(coverage_by_cat)]}
"""

ax8.text(0.1, 0.9, stats_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax8.set_title('Network Statistics', fontsize=11, fontweight='bold', pad=10)

# Panel 9: Key insights
ax9 = axes[2, 2]
ax9.axis('off')

dominant_coverage_idx = np.argmax(coverage_by_cat)
dominant_coverage_cat = categories[dominant_coverage_idx]
dominant_coverage_pct = (coverage_by_cat[dominant_coverage_idx] / total_network_length) * 100

insights_text = f"""KEY INSIGHTS: NETWORK

COVERAGE:
• Total: {total_km:.2f} km
• {n_active:,} road segments
• Largest contribution: {dominant_coverage_cat}
  ({dominant_coverage_pct:.1f}% of network)

CONNECTIVITY:
• Mean degree: {mean_degree:.2f}
• Length-degree correlation: {corr:.3f}
• Longer roads similar connectivity
• Network well-connected

EFFICIENCY:
• Longer roads more efficient
  (more length per connection)
• Short roads: Dense network
• Long roads: Sparse network

CONTRIBUTION:
• 20% longest roads ≈ 
  {cumulative_pct[int(0.2*len(cumulative_pct))]:.1f}% of total length
• Network dominated by
  many short segments

CONCLUSION:
• Dense network of short roads
• Long roads span distances
• Length independent of
  local connectivity
"""

ax9.text(0.1, 0.9, insights_text, fontsize=7.5, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
output_path = 'feature5_chart10_network_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")
display(Image(output_path))

print("\n" + "="*80)
print("CHART 10 COMPLETE")
print("="*80)
print(f"\nTotal network: {total_km:.2f} km")
print(f"Mean connectivity: {mean_degree:.2f} connections per road")
print(f"Largest contribution: {dominant_coverage_cat} ({dominant_coverage_pct:.1f}%)")
print(f"Length-degree correlation: {corr:.3f}")
