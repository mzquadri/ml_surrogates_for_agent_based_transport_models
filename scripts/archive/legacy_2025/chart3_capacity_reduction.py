import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Load first scenario
data = torch.load('/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt', weights_only=False)
scenario_0 = data[0]

# Extract CAPACITY_REDUCTION (Feature 2) - THE POLICY FEATURE
capacity_reduction = scenario_0.x[:, 2].numpy()

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution of reductions (only roads with reduction)
ax = axes[0]
roads_with_reduction = capacity_reduction[capacity_reduction < 0]
ax.hist(roads_with_reduction, bins=50, color='#E74C3C', edgecolor='black', alpha=0.7)
ax.set_xlabel('Capacity Reduction (vehicles per hour)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Roads', fontsize=12, fontweight='bold')
ax.set_title('Feature 3: CAPACITY_REDUCTION Distribution\n(Only roads affected by policy)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.97, f'Total affected: {len(roads_with_reduction)}\nMean: {roads_with_reduction.mean():.0f}\nMedian: {np.median(roads_with_reduction):.0f}\nMin: {roads_with_reduction.min():.0f}',
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8), fontsize=10)

# Plot 2: Policy impact categories
ax = axes[1]
categories = ['No Change\n(0)', 'Small Reduction\n(-1000 to 0)', 'Medium Reduction\n(-2000 to -1000)', 'Large Reduction\n(< -2000)']
counts = [
    (capacity_reduction == 0).sum(),
    ((capacity_reduction < 0) & (capacity_reduction >= -1000)).sum(),
    ((capacity_reduction < -1000) & (capacity_reduction >= -2000)).sum(),
    (capacity_reduction < -2000).sum()
]
colors = ['#D5D8DC', '#F5B7B1', '#E74C3C', '#C0392B']
bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Roads', fontsize=12, fontweight='bold')
ax.set_title('Feature 3: Policy Impact Categories', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(capacity_reduction)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()

# Create directory if it doesn't exist
os.makedirs('data/visualisation', exist_ok=True)

plt.savefig('data/visualisation/chart3_capacity_reduction.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed information
print("\n" + "="*80)
print("CHART 3: CAPACITY_REDUCTION (Policy Feature) - Feature 3")
print("="*80)
print()
print("What this feature represents:")
print("  → THIS IS THE POLICY INPUT - shows which roads are affected by interventions")
print("  → Amount of capacity REMOVED from each road due to policy changes")
print("  → Unit: vehicles per hour (veh/h)")
print()
print("⚠️  IMPORTANT - How to Read the Values:")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Value = 0      → No policy applied (road normal)                  │")
print("  │  Value = -600   → 600 veh/h capacity REMOVED (one lane closed)     │")
print("  │  Value = -1200  → 1200 veh/h capacity REMOVED (road mostly closed) │")
print("  │  Value = -4800  → 4800 veh/h capacity REMOVED (highway closed)     │")
print("  │                                                                     │")
print("  │  ➤ NEGATIVE (−) sign = LOSS of capacity                            │")
print("  │  ➤ More negative = More strict policy                              │")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("Concrete Example:")
print("  Suppose a road has CAPACITY_BASE_CASE = 1200 veh/h (original capacity)")
print()
print("  ┌──────────────┬─────────────────┬──────────────────┬─────────────────┐")
print("  │  Scenario    │  Policy Applied │  CAPACITY_REDUC. │  Final Capacity │")
print("  ├──────────────┼─────────────────┼──────────────────┼─────────────────┤")
print("  │  Normal      │  No policy      │        0         │  1200 veh/h     │")
print("  │  Scenario 1  │  Close 1 lane   │      -600        │   600 veh/h     │")
print("  │  Scenario 2  │  Close road     │     -1200        │     0 veh/h     │")
print("  └──────────────┴─────────────────┴──────────────────┴─────────────────┘")
print()
print("  Formula: Final Capacity = CAPACITY_BASE_CASE + CAPACITY_REDUCTION")
print("           (e.g., 1200 + (-600) = 600 veh/h remaining)")
print()
print("Key Statistics:")
print(f"  Total road segments:          {len(capacity_reduction):,}")
print(f"  Roads AFFECTED by policy:      {(capacity_reduction < 0).sum():,}  ({(capacity_reduction < 0).sum()/len(capacity_reduction)*100:5.1f}%)")
print(f"  Roads NOT affected:           {(capacity_reduction == 0).sum():,}  ({(capacity_reduction == 0).sum()/len(capacity_reduction)*100:5.1f}%)")
print(f"  Average reduction (affected):    {roads_with_reduction.mean():.0f}  veh/h")
print(f"  Median reduction (affected):     {np.median(roads_with_reduction):.0f}  veh/h")
print(f"  Standard deviation:              {roads_with_reduction.std():.0f}  veh/h")
print(f"  Maximum reduction:              {roads_with_reduction.min():.0f}  veh/h (most restrictive)")
print()
print("Policy Impact Categories:")
print(f"  NO Change (0)                {counts[0]:,} roads  ({counts[0]/len(capacity_reduction)*100:5.1f}%) - Not affected")
print(f"  SMALL Reduction (-1000 to 0)  {counts[1]:,} roads  ({counts[1]/len(capacity_reduction)*100:5.1f}%) - Minor restrictions")
print(f"  MEDIUM Reduction (-2000 to -1000) {counts[2]:,} roads  ({counts[2]/len(capacity_reduction)*100:5.1f}%) - Moderate restrictions")
print(f"  LARGE Reduction (< -2000)     {counts[3]:,} roads  ({counts[3]/len(capacity_reduction)*100:5.1f}%) - Severe restrictions/closures")
print()
print("Real-world Examples (What these negative values mean):")
print("  • -600 veh/h   = Close ONE lane on two-lane road")
print("                   (Before: 1200 veh/h → After: 600 veh/h)")
print()
print("  • -1200 veh/h  = Close ENTIRE single-lane street to cars")
print("                   (Before: 1200 veh/h → After: 0 veh/h = CLOSED)")
print()
print("  • -2400 veh/h  = Close TWO lanes on four-lane avenue")
print("                   (Before: 4800 veh/h → After: 2400 veh/h)")
print()
print("  • -4800 veh/h  = Close MAJOR highway (complete closure)")
print("                   (Before: 4800 veh/h → After: 0 veh/h = CLOSED)")
print()
print("How Policies Work:")
print("  • Each scenario tests DIFFERENT policy combinations")
print("  • Policies might close specific lanes, roads, or entire zones")
print("  • This feature tells the GNN WHERE policies are applied")
print("  • The GNN must predict HOW traffic redistributes to other roads")
print()
print("Simple Summary:")
print("  ✓ More NEGATIVE value = More STRICT policy = More capacity REMOVED")
print("  ✓ Zero (0) = No policy = Road operating normally")
print("  ✓ 91.9% roads unaffected, only 8.1% have policies applied")
print("  ✓ GNN must predict: 'Traffic kahen jayegi jab roads band hongi?'")
print()
print("Why this is THE MOST IMPORTANT feature:")
print("  • This is the ONLY feature that changes between scenarios")
print("  • VOL_BASE_CASE and CAPACITY_BASE_CASE are the SAME across all 1000 scenarios")
print("  • CAPACITY_REDUCTION defines what makes each scenario unique")
print("  • The GNN learns: 'Given these capacity reductions, how will traffic change?'")
print("  • Model success depends on learning the relationship between this input and traffic changes")
print()
print("="*80)
print("Chart saved: data/visualisation/chart3_capacity_reduction.png")
print("="*80 + "\n")
