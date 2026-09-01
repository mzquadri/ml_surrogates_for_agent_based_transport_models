"""
FEATURE MAPPING VERIFICATION

Cross-checks actual data values against expected feature definitions:
- Feature 0: Should be VOL_BASE_CASE (baseline volume)
- Feature 1: Should be CAPACITY_BASE_CASE (road capacity)
- Feature 2: Should be CAPACITY_REDUCTION (policy impact)
- Feature 3: Should be FREESPEED (free-flow speed)
- Feature 4: Should be HIGHWAY (road type)
- Feature 5: Should be LENGTH (segment length)
"""

import torch
import numpy as np

# Load data
batch_path = '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt'
data_list = torch.load(batch_path, weights_only=False, map_location='cpu')

print("="*80)
print("FEATURE MAPPING VERIFICATION")
print("="*80)

# Get data from first scenario
data = data_list[0]

print(f"\nTotal features in data.x: {data.x.shape[1]}")
print(f"Total nodes: {data.x.shape[0]}")

# Extract all features
features = []
for i in range(data.x.shape[1]):
    features.append(data.x[:, i].numpy())

print("\n" + "="*80)
print("ANALYZING EACH FEATURE")
print("="*80)

for i, feat in enumerate(features):
    print(f"\n{'='*80}")
    print(f"FEATURE {i}:")
    print(f"{'='*80}")
    
    # Basic statistics
    print(f"  Min: {feat.min():.2f}")
    print(f"  Max: {feat.max():.2f}")
    print(f"  Mean: {feat.mean():.2f}")
    print(f"  Median: {np.median(feat):.2f}")
    print(f"  Std: {feat.std():.2f}")
    print(f"  Zeros: {(feat == 0).sum()} ({(feat == 0).sum()/len(feat)*100:.2f}%)")
    print(f"  Negative values: {(feat < 0).sum()} ({(feat < 0).sum()/len(feat)*100:.2f}%)")
    print(f"  Unique values: {len(np.unique(feat))}")
    
    # Determine likely feature type
    print(f"\n  Analysis:")
    
    # Check if it's length-like (positive, meter range)
    if feat.min() >= 0 and feat.max() > 100 and feat.max() < 20000 and feat.mean() < 200:
        print(f"  → Likely LENGTH: Range 0-{feat.max():.0f}m, mean {feat.mean():.1f}m")
    
    # Check if it's capacity-like (positive, veh/h range)
    elif feat.min() >= 0 and feat.max() > 1000 and feat.max() < 10000 and feat.mean() > 400:
        print(f"  → Likely CAPACITY: Range 0-{feat.max():.0f} veh/h, mean {feat.mean():.0f} veh/h")
    
    # Check if it's volume-like (negative values, veh/h range)
    elif feat.min() < 0 and feat.max() == 0 and abs(feat.min()) < 10000:
        print(f"  → Likely BASELINE_VOLUME: Negative values (traffic present)")
        print(f"    Range: {feat.min():.0f} to 0 veh/h")
        print(f"    Mean (non-zero): {feat[feat < 0].mean():.1f} veh/h")
    
    # Check if it's percentage-like (0-100 range)
    elif feat.min() >= 0 and feat.max() <= 100 and feat.mean() < 50:
        print(f"  → Likely CAPACITY_REDUCTION or FREESPEED %: Range 0-{feat.max():.1f}%")
    
    # Check if it's speed-like (km/h range)
    elif feat.min() > 0 and feat.max() < 200 and feat.mean() > 20:
        print(f"  → Likely FREESPEED: Range {feat.min():.0f}-{feat.max():.0f} km/h")
    
    # Check if it's categorical (integer codes)
    elif len(np.unique(feat)) < 20 and feat.min() >= -1 and feat.max() < 20:
        print(f"  → Likely HIGHWAY TYPE: {len(np.unique(feat))} categories")
        print(f"    Values: {sorted(np.unique(feat))[:15]}")

# Expected mapping
print("\n" + "="*80)
print("EXPECTED FEATURE MAPPING (from supervisor)")
print("="*80)
print("""
Feature 0 = VOL_BASE_CASE (baseline volume) - negative values
Feature 1 = CAPACITY_BASE_CASE (road capacity) - veh/h
Feature 2 = CAPACITY_REDUCTION (policy impact) - percentage
Feature 3 = FREESPEED (free-flow speed) - km/h or percentage
Feature 4 = HIGHWAY (road type) - categorical codes
Feature 5 = LENGTH (segment length) - meters
""")

# Final verification
print("\n" + "="*80)
print("CROSS-CHECK RESULTS")
print("="*80)

print("\nBased on data analysis:")
print(f"  Feature 0: {features[0].min():.1f} to {features[0].max():.1f}, mean {features[0].mean():.1f}")
print(f"  Feature 1: {features[1].min():.1f} to {features[1].max():.1f}, mean {features[1].mean():.1f}")
print(f"  Feature 2: {features[2].min():.1f} to {features[2].max():.1f}, mean {features[2].mean():.1f}")
print(f"  Feature 3: {features[3].min():.1f} to {features[3].max():.1f}, mean {features[3].mean():.1f}")
print(f"  Feature 4: {features[4].min():.1f} to {features[4].max():.1f}, mean {features[4].mean():.1f}")
print(f"  Feature 5: {features[5].min():.1f} to {features[5].max():.1f}, mean {features[5].mean():.1f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
Please verify which features we analyzed:
1. Check if our 'Feature 0: LENGTH' analysis matches actual Feature 5 patterns
2. Check if our 'Feature 2: BASELINE_VOLUME' matches actual Feature 0 patterns
3. Confirm correct feature index → name mapping for thesis documentation
""")

print("="*80)
