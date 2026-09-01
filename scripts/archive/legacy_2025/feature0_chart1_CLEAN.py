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

# Create figure with clean layout and more space at bottom
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.3,
                       top=0.93, bottom=0.22, left=0.08, right=0.95)

# Main title
fig.suptitle('Feature 1: VOL_BASE_CASE (Baseline Traffic Volume)', 
             fontsize=20, fontweight='bold', y=0.97)

# Create subplots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ====== LEFT PANEL: Distribution Histogram ======
n, bins, patches = ax1.hist(vol_base_case[vol_base_case > 0], bins=50, color='#4A90E2', 
                             edgecolor='black', alpha=0.85, linewidth=1.2)

# Add count labels with bin ranges on first 3 bars only
for i in range(min(3, len(patches))):
    height = patches[i].get_height()
    bin_start = int(bins[i])
    bin_end = int(bins[i+1])
    ax1.text(patches[i].get_x() + patches[i].get_width()/2., height + 250,
            f'{int(height):,}\n({bin_start}-{bin_end})', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Daily Traffic Volume (vehicles/day)', fontsize=13, fontweight='bold', labelpad=8)
ax1.set_ylabel('Number of Road Segments', fontsize=13, fontweight='bold', labelpad=8)
ax1.set_title('Distribution of Traffic Across All Roads', fontsize=13, pad=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='both', labelsize=11)

# Statistics box BELOW the chart
stats_text = f"""Statistics: Roads analyzed: {roads_with_traffic:,} | Average: {avg_traffic:.0f} vehicles/day | Median: {median_traffic:.0f} | Max: {max_traffic:.0f}
Examples: 20 cars/day = Small street | 50 cars/day = Residential | 200 cars/day = Avenue | 500+ cars/day = Major road"""

ax1.text(0.5, -0.22, stats_text, transform=ax1.transAxes, 
         fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1))

# ====== RIGHT PANEL: Category Bar Chart ======
# Simple category names for x-axis
categories = ['NO\nTraffic', 'LOW\nTraffic', 'MEDIUM\nTraffic', 'HIGH\nTraffic']
counts = [no_traffic, low_traffic, medium_traffic, high_traffic]
colors = ['#95A5A6', '#5DADE2', '#F39C12', '#E74C3C']

x_pos = np.arange(len(categories))
bars = ax2.bar(x_pos, counts, color=colors, edgecolor='black', 
               linewidth=1.5, alpha=0.9, width=0.65)

# Add count and percentage labels on bars with better spacing
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    percentage = (count / total_roads) * 100
    
    # Count number on top - moved MUCH higher to avoid overlap
    ax2.text(bar.get_x() + bar.get_width()/2., height + 800,
             f'{count:,}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Percentage below count - good spacing
    ax2.text(bar.get_x() + bar.get_width()/2., height + 250,
             f'({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold', style='italic')

# X-axis: Set positions and labels
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=11, fontweight='bold')

# Add range definitions below x-axis labels (clean positioning)
range_labels = ['(0 cars/day)', '(1-50 cars/day)', '(50-200 cars/day)', '(>200 cars/day)']
for i, range_label in enumerate(range_labels):
    ax2.text(i, -1000, range_label, ha='center', va='top', 
             fontsize=9, color='#555555', style='italic')

# Axis labels and title
ax2.set_ylabel('Number of Road Segments', fontsize=13, fontweight='bold', labelpad=8)
ax2.set_xlabel('Traffic Level Category', fontsize=13, fontweight='bold', labelpad=30)
ax2.set_title('Roads Grouped by Traffic Level', fontsize=13, pad=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.tick_params(axis='y', labelsize=11)

# Set y-axis limit
max_height = max(counts)
ax2.set_ylim(-1400, max_height * 1.15)

# Summary box BELOW the chart
summary_text = f"""Summary: Total Roads = {total_roads:,} | With Traffic = {roads_with_traffic:,} ({(roads_with_traffic/total_roads*100):.1f}%)
Breakdown: NO Traffic: {no_traffic:,} ({(no_traffic/total_roads*100):.1f}%) - Pedestrian only | LOW: {low_traffic:,} ({(low_traffic/total_roads*100):.1f}%) - Residential | MEDIUM: {medium_traffic:,} ({(medium_traffic/total_roads*100):.1f}%) - Secondary | HIGH: {high_traffic:,} ({(high_traffic/total_roads*100):.1f}%) - Major roads"""

ax2.text(0.5, -0.22, summary_text, transform=ax2.transAxes, 
         fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))

# Create directory if it doesn't exist
os.makedirs('data/visualisation', exist_ok=True)

plt.savefig('data/visualisation/feature0_chart1_CLEAN.png', dpi=150, bbox_inches='tight')
plt.show()  # Display the chart in Colab
print("\n" + "="*70)
print("CHART 1: VOL_BASE_CASE - BASELINE TRAFFIC VOLUME")
print("="*70)
print(f"\nTotal roads in network: {total_roads:,}")
print(f"Roads WITH traffic: {roads_with_traffic:,} ({roads_with_traffic/total_roads*100:.1f}%)")
print(f"Roads WITHOUT traffic: {roads_without_traffic:,} ({roads_without_traffic/total_roads*100:.1f}%)")
print(f"\nAverage traffic: {avg_traffic:.0f} vehicles/day")
print(f"Median traffic: {median_traffic:.0f} vehicles/day")
print(f"Busiest road: {max_traffic:.0f} vehicles/day")
print("\n" + "="*70)
print("Chart saved successfully!")
print("="*70 + "\n")
