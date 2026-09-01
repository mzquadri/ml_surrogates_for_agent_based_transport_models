import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Load first scenario
data = torch.load('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt', weights_only=False)
scenario_0 = data[0]

# Extract data
vol_base = scenario_0.x[:, 0].numpy()

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution (excluding zeros)
ax = axes[0]
vol_with_traffic = vol_base[vol_base > 0]
ax.hist(vol_with_traffic, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Traffic Volume (vehicles per day)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Roads', fontsize=12, fontweight='bold')
ax.set_title('VOL_BASE_CASE Distribution\n(Only roads with traffic)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.97, f'Total roads: {len(vol_with_traffic)}\nMean: {vol_with_traffic.mean():.1f}\nMax: {vol_with_traffic.max():.0f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# Plot 2: Traffic categories
ax = axes[1]
categories = ['No Traffic\n(0)', 'Low\n(1-50)', 'Medium\n(50-200)', 'High\n(>200)']
counts = [
    (vol_base == 0).sum(),
    ((vol_base > 0) & (vol_base <= 50)).sum(),
    ((vol_base > 50) & (vol_base <= 200)).sum(),
    (vol_base > 200).sum()
]
colors = ['lightgray', 'lightblue', 'orange', 'red']
bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Roads', fontsize=12, fontweight='bold')
ax.set_title('Traffic Volume Categories', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(vol_base)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()

# Create directory if it doesn't exist
os.makedirs('data/visualisation', exist_ok=True)

plt.savefig('data/visualisation/chart1_vol_base_case.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed information
print("\n" + "="*80)
print("CHART 1: VOL_BASE_CASE (Baseline Traffic Volume)")
print("="*80)
print()
print("What this feature represents:")
print("  → Number of vehicles using each road segment per day")
print("  → Measured BEFORE any policy changes (baseline conditions)")
print("  → Critical for understanding which roads carry most traffic")
print()
print("Key Statistics:")
print(f"  Total road segments:          {len(vol_base):,}")
print(f"  Segments with traffic:        {len(vol_with_traffic):,}  ({len(vol_with_traffic)/len(vol_base)*100:5.1f}%)")
print(f"  Segments without traffic:      {(vol_base == 0).sum():,}  ({(vol_base == 0).sum()/len(vol_base)*100:5.1f}%)")
print(f"  Average daily traffic:            {vol_with_traffic.mean():.0f}  vehicles/day")
print(f"  Busiest road:                   {vol_with_traffic.max():.0f}  vehicles/day")
print()
print("Traffic Categories:")
print(f"  NO Traffic (0)         {counts[0]:,} roads  ({counts[0]/len(vol_base)*100:5.1f}%)")
print(f"  LOW (1-50)            {counts[1]:,} roads  ({counts[1]/len(vol_base)*100:5.1f}%)")
print(f"  MEDIUM (50-200)        {counts[2]:,} roads  ({counts[2]/len(vol_base)*100:5.1f}%)")
print(f"  HIGH (>200)            {counts[3]:,} roads  ({counts[3]/len(vol_base)*100:5.1f}%)")
print()
print("Why this matters:")
print("  • Shows realistic Paris traffic pattern: few busy roads, many quiet streets")
print("  • GNN model will learn from these baseline patterns")
print("  • Policy impacts will be measured as CHANGE from these baseline values")
print()
print("="*80)
print("Chart saved: data/visualisation/chart1_vol_base_case.png")
print("="*80 + "\n")
