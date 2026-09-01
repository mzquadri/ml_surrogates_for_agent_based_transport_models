"""
FEATURE 4 - CHART 11
Highway Type Comprehensive Summary

Final comprehensive summary of highway type analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy import stats
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

print("\nCHART 11: Highway Type Comprehensive Summary")
print("=" * 60)

# Load data
graphs_list = torch.load(batch_path, weights_only=False)
graph = graphs_list[0]

n_active = 31635
highway_type = graph.x[:n_active, 4].numpy().astype(int)
capacity = graph.x[:n_active, 1].numpy()
free_speed = graph.x[:n_active, 3].numpy()
road_length = graph.x[:n_active, 5].numpy()
baseline_volume = graph.x[:n_active, 2].numpy()

# Sort types by frequency
type_counts = Counter(highway_type)
type_codes_sorted = sorted(type_counts.keys(), key=lambda x: type_counts[x], reverse=True)
type_names_sorted = [HW_MAPPING[code] for code in type_codes_sorted]

# Collect data by type
capacity_by_type = [capacity[highway_type == code] for code in type_codes_sorted]
speed_by_type = [free_speed[highway_type == code] for code in type_codes_sorted]
length_by_type = [road_length[highway_type == code] for code in type_codes_sorted]
volume_by_type = [baseline_volume[highway_type == code] for code in type_codes_sorted]

# Create figure with 9 panels
fig = plt.figure(figsize=(24, 20))

# Panel 1: Distribution summary (pie)
ax1 = plt.subplot(3, 3, 1)
type_counts_sorted = [type_counts[code] for code in type_codes_sorted]
colors_sorted = COLORS_11[:len(type_codes_sorted)]
wedges, texts, autotexts = ax1.pie(type_counts_sorted, labels=type_names_sorted, 
                                     autopct='%1.1f%%', startangle=90, colors=colors_sorted)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(7)
ax1.set_title('Type Distribution', fontsize=10, fontweight='bold')

# Panel 2: Capacity summary
ax2 = plt.subplot(3, 3, 2)
cap_means = [np.mean(data) for data in capacity_by_type]
bars = ax2.barh(range(len(type_names_sorted)), cap_means, color=colors_sorted, alpha=0.8)
ax2.set_yticks(range(len(type_names_sorted)))
ax2.set_yticklabels(type_names_sorted, fontsize=8)
ax2.set_xlabel('Mean Capacity (veh/h)', fontsize=9, fontweight='bold')
ax2.set_title('Capacity by Type', fontsize=10, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Panel 3: Speed summary
ax3 = plt.subplot(3, 3, 3)
speed_means = [np.mean(data) for data in speed_by_type]
bars = ax3.barh(range(len(type_names_sorted)), speed_means, color=colors_sorted, alpha=0.8)
ax3.set_yticks(range(len(type_names_sorted)))
ax3.set_yticklabels(type_names_sorted, fontsize=8)
ax3.set_xlabel('Mean Speed (km/h)', fontsize=9, fontweight='bold')
ax3.set_title('Speed by Type', fontsize=10, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Panel 4: Length summary
ax4 = plt.subplot(3, 3, 4)
length_means = [np.mean(data) for data in length_by_type]
bars = ax4.barh(range(len(type_names_sorted)), length_means, color=colors_sorted, alpha=0.8)
ax4.set_yticks(range(len(type_names_sorted)))
ax4.set_yticklabels(type_names_sorted, fontsize=8)
ax4.set_xlabel('Mean Length (m)', fontsize=9, fontweight='bold')
ax4.set_title('Road Length by Type', fontsize=10, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Panel 5: Traffic coverage
ax5 = plt.subplot(3, 3, 5)
traffic_pcts = [(np.sum(data != 0) / len(data)) * 100 for data in volume_by_type]
bars = ax5.barh(range(len(type_names_sorted)), traffic_pcts, color=colors_sorted, alpha=0.8)
ax5.set_yticks(range(len(type_names_sorted)))
ax5.set_yticklabels(type_names_sorted, fontsize=8)
ax5.set_xlabel('Traffic Coverage (%)', fontsize=9, fontweight='bold')
ax5.set_title('Roads with Traffic', fontsize=10, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
ax5.set_xlim(0, 100)

# Panel 6: Feature correlations radar chart
ax6 = plt.subplot(3, 3, 6, projection='polar')

# Calculate correlations for top 5 types
top5_types = type_codes_sorted[:5]
top5_names = type_names_sorted[:5]
categories = ['Capacity', 'Speed', 'Length', 'Traffic']
n_cats = len(categories)

angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]

for i, (code, name, color) in enumerate(zip(top5_types, top5_names, colors_sorted[:5])):
    mask = highway_type == code
    
    # Calculate correlations (normalized to 0-1)
    values = [
        abs(np.corrcoef(capacity[mask], free_speed[mask])[0, 1]),
        abs(np.corrcoef(capacity[mask], road_length[mask])[0, 1]),
        abs(np.corrcoef(free_speed[mask], road_length[mask])[0, 1]),
        abs(stats.pointbiserialr((baseline_volume[mask] != 0).astype(int), capacity[mask])[0])
    ]
    values += values[:1]
    
    ax6.plot(angles, values, 'o-', linewidth=2, label=name, color=color, alpha=0.7)
    ax6.fill(angles, values, alpha=0.15, color=color)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontsize=8)
ax6.set_ylim(0, 1)
ax6.set_title('Feature Correlations (Top 5)', fontsize=10, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)
ax6.grid(True)

# Panel 7: Capacity-Speed relationship scatter
ax7 = plt.subplot(3, 3, 7)
for i, (code, name, color) in enumerate(zip(type_codes_sorted[:5], type_names_sorted[:5], colors_sorted[:5])):
    mask = highway_type == code
    ax7.scatter(capacity[mask], free_speed[mask], alpha=0.3, s=5, color=color, label=name)
ax7.set_xlabel('Capacity (veh/h)', fontsize=9, fontweight='bold')
ax7.set_ylabel('Free Speed (km/h)', fontsize=9, fontweight='bold')
ax7.set_title('Capacity vs Speed (Top 5)', fontsize=10, fontweight='bold')
ax7.legend(fontsize=7, loc='best')
ax7.grid(alpha=0.3)

# Panel 8: Type hierarchy and importance
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')

# Calculate importance score (combination of count, capacity, speed, traffic)
importance_scores = []
for i, code in enumerate(type_codes_sorted):
    count_score = type_counts_sorted[i] / n_active
    cap_score = cap_means[i] / max(cap_means)
    speed_score = speed_means[i] / max(speed_means)
    traffic_score = traffic_pcts[i] / 100
    
    importance = (count_score * 0.4 + cap_score * 0.2 + 
                  speed_score * 0.2 + traffic_score * 0.2)
    importance_scores.append(importance)

# Sort by importance
importance_sorted = sorted(zip(type_names_sorted, importance_scores, type_counts_sorted, 
                               cap_means, speed_means, traffic_pcts),
                          key=lambda x: x[1], reverse=True)

hierarchy_text = "TYPE HIERARCHY (by Importance):\n\n"
for rank, (name, score, count, cap, speed, traffic) in enumerate(importance_sorted[:8], 1):
    hierarchy_text += f"{rank}. {name}\n"
    hierarchy_text += f"   Score: {score:.3f}\n"
    hierarchy_text += f"   Roads: {count:,} | Cap: {cap:.0f}\n"
    hierarchy_text += f"   Speed: {speed:.1f} | Traffic: {traffic:.1f}%\n\n"

ax8.text(0.05, 0.95, hierarchy_text, fontsize=8, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax8.set_title('Type Importance Ranking', fontsize=10, fontweight='bold', pad=15)

# Panel 9: Key insights summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

dominant_type = type_names_sorted[0]
highest_cap = type_names_sorted[np.argmax(cap_means)]
highest_speed = type_names_sorted[np.argmax(speed_means)]
most_traffic = type_names_sorted[np.argmax(traffic_pcts)]
longest = type_names_sorted[np.argmax(length_means)]

insights_text = f"""HIGHWAY TYPE - KEY INSIGHTS

DISTRIBUTION:
• {len(type_counts)} unique types present
• {dominant_type} dominant ({type_counts_sorted[0]:,} roads, {type_counts_sorted[0]/n_active*100:.1f}%)
• Top 3 types: {sum(type_counts_sorted[:3])/n_active*100:.1f}% coverage

CHARACTERISTICS:
• Highest capacity: {highest_cap} ({max(cap_means):.0f} veh/h)
• Highest speed: {highest_speed} ({max(speed_means):.1f} km/h)
• Longest roads: {longest} ({max(length_means):.1f} m)
• Most traffic: {most_traffic} ({max(traffic_pcts):.1f}%)

DATA QUALITY:
• Unknown type: {type_counts.get(-1, 0):,} roads ({type_counts.get(-1, 0)/n_active*100:.1f}%)
• Overall traffic coverage: {(np.sum(baseline_volume != 0) / n_active) * 100:.1f}%

FEATURE STATUS:
• Type: STATIC (design parameter)
• Does NOT change across scenarios
• Strong correlation with capacity & speed
• Determines road functional class
"""

ax9.text(0.05, 0.95, insights_text, fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
ax9.set_title('Summary & Insights', fontsize=10, fontweight='bold', pad=15)

plt.tight_layout()
chart_path = 'feature4_chart11_comprehensive_summary.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {chart_path}")
display(Image(chart_path))

print("\n" + "="*60)
print("FEATURE 4 ANALYSIS COMPLETE")
print("="*60)
print(f"\nTotal charts created: 11")
print(f"  • Parts 1-2: Distribution & characteristics")
print(f"  • Charts 5-11: Detailed individual analysis")
print(f"\nHighway type is STATIC and determines:")
print(f"  - Road capacity (strong correlation)")
print(f"  - Speed limits (strong correlation)")
print(f"  - Network functional hierarchy")
print(f"  - Traffic patterns (moderate correlation)")
