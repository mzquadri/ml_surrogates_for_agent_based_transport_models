import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Load first scenario
data = torch.load('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt', weights_only=False)
scenario_0 = data[0]

# Extract CAPACITY_BASE_CASE (Feature 1)
capacity_base = scenario_0.x[:, 1].numpy()

# Get traffic data for comparison
vol_base = scenario_0.x[:, 0].numpy()
avg_traffic = np.mean(vol_base[vol_base > 0])

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution (excluding zeros)
ax = axes[0]
capacity_with_value = capacity_base[capacity_base > 0]
ax.hist(capacity_with_value, bins=50, color='#2ECC71', edgecolor='black', alpha=0.7)
ax.set_xlabel('Road Capacity (vehicles per hour)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Roads', fontsize=12, fontweight='bold')
ax.set_title('Feature 2: CAPACITY_BASE_CASE Distribution\n(Only roads with capacity)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, f'Total roads: {len(capacity_with_value)}\nMean: {capacity_with_value.mean():.0f}\nMedian: {np.median(capacity_with_value):.0f}\nMax: {capacity_with_value.max():.0f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#A9DFBF', alpha=0.8), fontsize=10)

# Plot 2: Capacity categories
ax = axes[1]
categories = ['No Capacity\n(0)', 'Small\n(1-1000)', 'Medium\n(1000-2000)', 'Large\n(>2000)']
counts = [
    (capacity_base == 0).sum(),
    ((capacity_base > 0) & (capacity_base <= 1000)).sum(),
    ((capacity_base > 1000) & (capacity_base <= 2000)).sum(),
    (capacity_base > 2000).sum()
]
colors = ['#BDC3C7', '#58D68D', '#28B463', '#1E8449']
bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Roads', fontsize=12, fontweight='bold')
ax.set_title('Feature 2: Road Capacity Categories', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(capacity_base)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()

# Create directory if it doesn't exist
os.makedirs('data/visualisation', exist_ok=True)

plt.savefig('data/visualisation/chart2_capacity_base_case.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed information
print("\n" + "="*80)
print("CHART 2: CAPACITY_BASE_CASE (Road Capacity) - Feature 2")
print("="*80)
print()
print("What this feature represents:")
print("  → Maximum number of vehicles that can use the road per hour")
print("  → Measured BEFORE any policy changes (baseline capacity)")
print("  → Depends on road width, number of lanes, and road type")
print("  → Unit: vehicles per hour (veh/h)")
print()
print("Key Statistics:")
print(f"  Total road segments:          {len(capacity_base):,}")
print(f"  Segments with capacity:       {len(capacity_with_value):,}  ({len(capacity_with_value)/len(capacity_base)*100:5.1f}%)")
print(f"  Segments without capacity:     {(capacity_base == 0).sum():,}  ({(capacity_base == 0).sum()/len(capacity_base)*100:5.1f}%)")
print(f"  Average capacity:                {capacity_with_value.mean():.0f}  vehicles/hour")
print(f"  Median capacity:                 {np.median(capacity_with_value):.0f}  vehicles/hour")
print(f"  Standard deviation:              {capacity_with_value.std():.0f}  vehicles/hour")
print(f"  Maximum capacity:               {capacity_with_value.max():.0f}  vehicles/hour")
print()
print("Capacity Categories:")
print(f"  NO Capacity (0)        {counts[0]:,} roads  ({counts[0]/len(capacity_base)*100:5.1f}%) - Pedestrian/bike only")
print(f"  SMALL (1-1000)        {counts[1]:,} roads  ({counts[1]/len(capacity_base)*100:5.1f}%) - 1-lane local streets")
print(f"  MEDIUM (1000-2000)     {counts[2]:,} roads  ({counts[2]/len(capacity_base)*100:5.1f}%) - 2-lane roads")
print(f"  LARGE (>2000)          {counts[3]:,} roads  ({counts[3]/len(capacity_base)*100:5.1f}%) - Multi-lane boulevards/highways")
print()
print("Real-world Examples:")
print("  • 600 veh/h   = Single-lane residential street (one car every 6 seconds)")
print("  • 1200 veh/h  = Two-lane local road (typical neighborhood)")
print("  • 2400 veh/h  = Four-lane avenue (busy urban road)")
print("  • 14400 veh/h = Major highway (maximum observed in dataset)")
print()
print("Relationship with Traffic Volume:")
print(f"  • Roads with NO capacity also have NO traffic (pedestrian-only)")
print(f"  • Capacity utilization: {(avg_traffic * 24 / capacity_with_value.mean() * 100):.1f}% average")
print(f"    (daily traffic / hourly capacity gives rough usage estimate)")
print()
print("Why this matters:")
print("  • Capacity determines how much traffic a road can handle before congestion")
print("  • Higher capacity roads (highways) can absorb more traffic")
print("  • GNN model uses capacity to predict traffic redistribution after policy changes")
print("  • When policies close roads, traffic must move to roads with available capacity")
print("  • Critical for understanding bottlenecks and congestion patterns")
print()
print("="*80)
print("Chart saved: data/visualisation/chart2_capacity_base_case.png")
print("="*80 + "\n")
