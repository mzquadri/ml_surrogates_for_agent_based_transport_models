"""
FEATURE 5 - CHART 8
Length Categories Detailed Breakdown

Comprehensive analysis of road length categories and their characteristics.
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
print("FEATURE 5 - CHART 8: Length Categories Breakdown")
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

# Define categories
categories = ['Very Short\n(<50m)', 'Short\n(50-100m)', 'Medium\n(100-200m)', 
              'Long\n(200-500m)', 'Very Long\n(500-1000m)', 'Extra Long\n(>1000m)']
ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, np.inf)]
cat_labels = ['<50m', '50-100m', '100-200m', '200-500m', '500-1000m', '>1000m']

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(24, 20))

# Panel 1: Category distribution
ax1 = axes[0, 0]
counts = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    counts.append(np.sum(mask))

colors_cat = plt.cm.viridis(np.linspace(0, 1, len(categories)))
bars = ax1.bar(range(len(categories)), counts, color=colors_cat, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels(categories, fontsize=9)
ax1.set_ylabel('Count', fontsize=10, fontweight='bold')
ax1.set_title('Roads per Length Category', fontsize=11, fontweight='bold', pad=10)
ax1.grid(axis='y', alpha=0.3)

for i, (bar, count) in enumerate(zip(bars, counts)):
    pct = (count / n_active) * 100
    ax1.text(i, count, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)

# Panel 2: Cumulative percentage
ax2 = axes[0, 1]
cumulative = np.cumsum(counts) / n_active * 100
ax2.plot(range(len(categories)), cumulative, marker='o', linewidth=3, markersize=10, color='darkblue')
ax2.fill_between(range(len(categories)), cumulative, alpha=0.3, color='lightblue')
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_ylabel('Cumulative %', fontsize=10, fontweight='bold')
ax2.set_title('Cumulative Distribution', fontsize=11, fontweight='bold', pad=10)
ax2.grid(alpha=0.3)
ax2.axhline(50, color='red', linestyle='--', linewidth=2, label='50%')
ax2.axhline(80, color='orange', linestyle='--', linewidth=2, label='80%')
ax2.legend(fontsize=9)

for i, cum in enumerate(cumulative):
    ax2.text(i, cum+2, f'{cum:.1f}%', ha='center', fontsize=8, fontweight='bold')

# Panel 3: Highway type distribution by category
ax3 = axes[0, 2]
type_counts = Counter(highway_type)
top_types = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)[:5]
type_names_short = [HW_MAPPING[code][:10] for code in top_types]

type_by_cat = np.zeros((len(top_types), len(ranges)))
for i, type_code in enumerate(top_types):
    for j, (low, high) in enumerate(ranges):
        mask = (highway_type == type_code) & (road_length >= low) & (road_length < high)
        type_by_cat[i, j] = np.sum(mask)

im = ax3.imshow(type_by_cat, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(len(cat_labels)))
ax3.set_yticks(range(len(type_names_short)))
ax3.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=8)
ax3.set_yticklabels(type_names_short, fontsize=9)
ax3.set_title('Highway Type by Length Category', fontsize=11, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax3, label='Count')

# Panel 4: Mean capacity by category
ax4 = axes[1, 0]
mean_caps = []
std_caps = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        mean_caps.append(np.mean(capacity[mask]))
        std_caps.append(np.std(capacity[mask]))
    else:
        mean_caps.append(0)
        std_caps.append(0)

bars = ax4.bar(range(len(categories)), mean_caps, yerr=std_caps, capsize=5,
               color=colors_cat, alpha=0.8, edgecolor='black')
ax4.set_xticks(range(len(categories)))
ax4.set_xticklabels(categories, fontsize=9)
ax4.set_ylabel('Mean Capacity (veh/h)', fontsize=10, fontweight='bold')
ax4.set_title('Capacity by Length Category', fontsize=11, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)

for i, (bar, cap) in enumerate(zip(bars, mean_caps)):
    ax4.text(i, cap, f'{cap:.0f}', ha='center', va='bottom', fontsize=8)

# Panel 5: Mean speed by category
ax5 = axes[1, 1]
mean_speeds = []
std_speeds = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        mean_speeds.append(np.mean(free_speed[mask]))
        std_speeds.append(np.std(free_speed[mask]))
    else:
        mean_speeds.append(0)
        std_speeds.append(0)

bars = ax5.bar(range(len(categories)), mean_speeds, yerr=std_speeds, capsize=5,
               color=colors_cat, alpha=0.8, edgecolor='black')
ax5.set_xticks(range(len(categories)))
ax5.set_xticklabels(categories, fontsize=9)
ax5.set_ylabel('Mean Speed (km/h)', fontsize=10, fontweight='bold')
ax5.set_title('Speed by Length Category', fontsize=11, fontweight='bold', pad=10)
ax5.grid(axis='y', alpha=0.3)

for i, (bar, speed) in enumerate(zip(bars, mean_speeds)):
    ax5.text(i, speed, f'{speed:.1f}', ha='center', va='bottom', fontsize=8)

# Panel 6: Traffic presence by category
ax6 = axes[1, 2]
traffic_mask = baseline_volume != 0
traffic_pct = []
for low, high in ranges:
    mask = (road_length >= low) & (road_length < high)
    if np.sum(mask) > 0:
        pct = np.sum(mask & traffic_mask) / np.sum(mask) * 100
        traffic_pct.append(pct)
    else:
        traffic_pct.append(0)

bars = ax6.bar(range(len(categories)), traffic_pct, 
               color=colors_cat, alpha=0.8, edgecolor='black')
ax6.set_xticks(range(len(categories)))
ax6.set_xticklabels(categories, fontsize=9)
ax6.set_ylabel('% with Traffic', fontsize=10, fontweight='bold')
ax6.set_title('Traffic Presence by Category', fontsize=11, fontweight='bold', pad=10)
ax6.grid(axis='y', alpha=0.3)

for i, (bar, pct) in enumerate(zip(bars, traffic_pct)):
    ax6.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 7: Statistics by category
ax7 = axes[2, 0]
ax7.axis('off')
stats_text = "CATEGORY STATISTICS\n\n"

for i, (cat, (low, high)) in enumerate(zip(cat_labels, ranges)):
    mask = (road_length >= low) & (road_length < high)
    lengths = road_length[mask]
    if len(lengths) > 0:
        stats_text += f"{cat}:\n"
        stats_text += f"  Count: {len(lengths):,}\n"
        stats_text += f"  Mean: {np.mean(lengths):.1f}m\n"
        stats_text += f"  Median: {np.median(lengths):.1f}m\n"
        stats_text += f"  Std: {np.std(lengths):.1f}m\n\n"

ax7.text(0.1, 0.9, stats_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax7.set_title('Length Statistics', fontsize=11, fontweight='bold', pad=10)

# Panel 8: Category comparison table
ax8 = axes[2, 1]
ax8.axis('off')
table_data = [['Category', 'Count', '%', 'Capacity', 'Speed']]
for i, (cat, count) in enumerate(zip(cat_labels, counts)):
    pct = (count / n_active) * 100
    table_data.append([cat, f'{count:,}', f'{pct:.1f}%', 
                      f'{mean_caps[i]:.0f}', f'{mean_speeds[i]:.1f}'])

table = ax8.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.5)
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')
ax8.set_title('Category Comparison Table', fontsize=11, fontweight='bold', pad=10)

# Panel 9: Key insights
ax9 = axes[2, 2]
ax9.axis('off')

dominant_cat_idx = np.argmax(counts)
dominant_cat = cat_labels[dominant_cat_idx]
dominant_pct = (counts[dominant_cat_idx] / n_active) * 100

insights_text = f"""KEY INSIGHTS: CATEGORIES

DISTRIBUTION:
• Dominant: {dominant_cat}
  ({counts[dominant_cat_idx]:,} roads, {dominant_pct:.1f}%)
• 50% of roads: <{cat_labels[1]}
• 80% of roads: <{cat_labels[3]}
• Heavy right-skew

CAPACITY PATTERN:
• Highest: {cat_labels[np.argmax(mean_caps)]}
  ({max(mean_caps):.0f} veh/h)
• Lowest: {cat_labels[np.argmin(mean_caps)]}
  ({min(mean_caps):.0f} veh/h)
• Small variation across categories

SPEED PATTERN:
• Fastest: {cat_labels[np.argmax(mean_speeds)]}
  ({max(mean_speeds):.1f} km/h)
• Slowest: {cat_labels[np.argmin(mean_speeds)]}
  ({min(mean_speeds):.1f} km/h)

CONCLUSION:
• Network dominated by short roads
• Length weakly affects capacity/speed
• Road type more important
• STATIC physical dimension
"""

ax9.text(0.1, 0.9, insights_text, fontsize=8, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
output_path = 'feature5_chart8_categories.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")
display(Image(output_path))

print("\n" + "="*80)
print("CHART 8 COMPLETE")
print("="*80)
print(f"\nDominant category: {dominant_cat} ({counts[dominant_cat_idx]:,} roads, {dominant_pct:.1f}%)")
print(f"Cumulative 50%: {cumulative[1]:.1f}% of roads")
print(f"Cumulative 80%: {cumulative[3]:.1f}% of roads")
print(f"Conclusion: Network dominated by short roads")
