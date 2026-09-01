import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Load first scenario
data = torch.load('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt', weights_only=False)
scenario_0 = data[0]

# Extract VOL_BASE_CASE (Feature 0)
vol_base_case = scenario_0.x[:, 0].numpy()

# Calculate statistics
total_roads = len(vol_base_case)
roads_with_traffic = np.sum(vol_base_case > 0)
avg_traffic = np.mean(vol_base_case[vol_base_case > 0])
median_traffic = np.median(vol_base_case[vol_base_case > 0])
max_traffic = np.max(vol_base_case)

# Categorize roads
no_traffic = np.sum(vol_base_case == 0)
low_traffic = np.sum((vol_base_case > 0) & (vol_base_case <= 50))
medium_traffic = np.sum((vol_base_case > 50) & (vol_base_case <= 200))
high_traffic = np.sum(vol_base_case > 200)

# Create figure
fig = plt.figure(figsize=(16, 9))

# Create grid: 2 rows - top row for charts, bottom row for text
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.5, wspace=0.3,
                       top=0.94, bottom=0.08, left=0.08, right=0.95)

# Main title
fig.suptitle('Feature 1: VOL_BASE_CASE (Baseline Traffic Volume)', 
             fontsize=20, fontweight='bold')

# Top row: Charts
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ====== LEFT: Histogram ======
n, bins, patches = ax1.hist(vol_base_case[vol_base_case > 0], bins=50, 
                             color='#4A90E2', edgecolor='black', alpha=0.85)

ax1.set_xlabel('Daily Traffic Volume (vehicles/day)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Road Segments', fontsize=13, fontweight='bold')
ax1.set_title('Traffic Distribution Across Roads', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# ====== RIGHT: Bar Chart ======
categories = ['NO Traffic\n(0)', 'LOW Traffic\n(1-50)', 'MEDIUM Traffic\n(50-200)', 'HIGH Traffic\n(>200)']
counts = [no_traffic, low_traffic, medium_traffic, high_traffic]
colors = ['#95A5A6', '#5DADE2', '#F39C12', '#E74C3C']

bars = ax2.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)

# Add values on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    percentage = (count / total_roads) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:,}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Number of Road Segments', fontsize=13, fontweight='bold')
ax2.set_xlabel('Traffic Category (cars/day range)', fontsize=13, fontweight='bold')
ax2.set_title('Roads Grouped by Traffic Level', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ====== BOTTOM ROW: Explanation Text ======
ax_text = fig.add_subplot(gs[1, :])
ax_text.axis('off')

explanation = f"""
INTERPRETATION & KEY STATISTICS:

Total Network: {total_roads:,} road segments in Paris

LEFT CHART (Histogram): Shows distribution of traffic volumes across all roads with traffic
    • Most roads (shown by tall left bars) have LOW traffic (under 100 vehicles/day)
    • Distribution is right-skewed: few roads handle very high traffic
    • Statistics: Average = {avg_traffic:.0f} vehicles/day | Median = {median_traffic:.0f} | Maximum = {max_traffic:.0f}

RIGHT CHART (Bar Chart): Roads categorized by traffic level
    • NO Traffic: {no_traffic:,} roads ({(no_traffic/total_roads*100):.1f}%) - Pedestrian/bike/public transport only
    • LOW Traffic: {low_traffic:,} roads ({(low_traffic/total_roads*100):.1f}%) - Residential streets, local access
    • MEDIUM Traffic: {medium_traffic:,} roads ({(medium_traffic/total_roads*100):.1f}%) - Secondary roads, neighborhood avenues  
    • HIGH Traffic: {high_traffic:,} roads ({(high_traffic/total_roads*100):.1f}%) - Major boulevards, highways, arterials

Key Insight: 76% of roads serve vehicles, but only 5% handle high traffic volumes. The network is dominated by low-traffic residential streets.
"""

ax_text.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=10,
             family='monospace', wrap=True,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=15))

# Save
os.makedirs('data/visualisation', exist_ok=True)
plt.savefig('data/visualisation/feature0_chart1_CLEAN.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("CHART 1: VOL_BASE_CASE")
print("="*70)
print(f"Total roads: {total_roads:,}")
print(f"Average traffic: {avg_traffic:.0f} vehicles/day")
print("Chart saved!")
print("="*70 + "\n")
