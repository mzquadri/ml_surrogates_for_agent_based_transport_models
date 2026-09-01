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
roads_without_traffic = np.sum(vol_base_case == 0)

avg_traffic = np.mean(vol_base_case[vol_base_case > 0])
median_traffic = np.median(vol_base_case[vol_base_case > 0])
max_traffic = np.max(vol_base_case)

# Categorize roads
no_traffic = np.sum(vol_base_case == 0)
low_traffic = np.sum((vol_base_case > 0) & (vol_base_case <= 50))
medium_traffic = np.sum((vol_base_case > 50) & (vol_base_case <= 200))
high_traffic = np.sum(vol_base_case > 200)

# Create figure with LOTS of space at bottom
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
plt.subplots_adjust(top=0.92, bottom=0.28, left=0.08, right=0.96, wspace=0.35)

# Main title
fig.suptitle('Feature 1: VOL_BASE_CASE (Baseline Traffic Volume)', 
             fontsize=18, fontweight='bold')

# ====== LEFT PANEL: Histogram ======
n, bins, patches = ax1.hist(vol_base_case[vol_base_case > 0], bins=50, color='#4A90E2', 
                             edgecolor='black', alpha=0.85, linewidth=1.2)

# Label first 3 bars
for i in range(3):
    height = patches[i].get_height()
    ax1.text(patches[i].get_x() + patches[i].get_width()/2., height + 300,
            f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Axes and labels
ax1.set_xlabel('Daily Traffic Volume (vehicles/day)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Road Segments', fontsize=12, fontweight='bold')
ax1.set_title('Traffic Distribution', fontsize=12, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=10)

# ====== RIGHT PANEL: Bar Chart ======
categories = ['NO\nTraffic', 'LOW\nTraffic', 'MEDIUM\nTraffic', 'HIGH\nTraffic']
counts = [no_traffic, low_traffic, medium_traffic, high_traffic]
colors = ['#95A5A6', '#5DADE2', '#F39C12', '#E74C3C']

bars = ax2.bar(categories, counts, color=colors, edgecolor='black', 
               linewidth=1.5, alpha=0.9, width=0.7)

# Add labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    percentage = (count / total_roads) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height + 400,
             f'{count:,}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Axes and labels
ax2.set_ylabel('Number of Road Segments', fontsize=12, fontweight='bold')
ax2.set_xlabel('Traffic Category', fontsize=12, fontweight='bold')
ax2.set_title('Roads by Category', fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.tick_params(labelsize=10)

# Add range labels below x-axis
for i, range_label in enumerate(['0 cars/day', '1-50 cars/day', '50-200 cars/day', '>200 cars/day']):
    ax2.text(i, -800, range_label, ha='center', va='top', fontsize=9, color='#555555')

ax2.set_ylim(-1200, max(counts) * 1.12)

# ====== EXPLANATION BOXES BELOW CHARTS ======
# Left chart stats
text1 = f"Statistics: {roads_with_traffic:,} roads analyzed | Average: {avg_traffic:.0f} vehicles/day | Median: {median_traffic:.0f} | Max: {max_traffic:.0f}\nExamples: 20 cars/day = Small street | 50 cars/day = Residential | 200 cars/day = Avenue | 500+ = Major road"
fig.text(0.27, 0.14, text1, ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=10))

# Right chart summary
text2 = f"Summary: {total_roads:,} total roads | {roads_with_traffic:,} with traffic ({(roads_with_traffic/total_roads*100):.1f}%)\nNO: {no_traffic:,} ({(no_traffic/total_roads*100):.1f}%) Pedestrian | LOW: {low_traffic:,} ({(low_traffic/total_roads*100):.1f}%) Residential | MEDIUM: {medium_traffic:,} ({(medium_traffic/total_roads*100):.1f}%) Secondary | HIGH: {high_traffic:,} ({(high_traffic/total_roads*100):.1f}%) Major"
fig.text(0.73, 0.14, text2, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=10))

# Create directory if it doesn't exist
os.makedirs('data/visualisation', exist_ok=True)

plt.savefig('data/visualisation/feature0_chart1_CLEAN.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("CHART 1: VOL_BASE_CASE - BASELINE TRAFFIC VOLUME")
print("="*70)
print(f"\nTotal roads: {total_roads:,}")
print(f"With traffic: {roads_with_traffic:,} ({roads_with_traffic/total_roads*100:.1f}%)")
print(f"Average: {avg_traffic:.0f} vehicles/day | Median: {median_traffic:.0f} | Max: {max_traffic:.0f}")
print("\n" + "="*70)
print("Chart saved!")
print("="*70 + "\n")
