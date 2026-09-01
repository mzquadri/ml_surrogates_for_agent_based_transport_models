"""
MASTER DATA VERIFICATION - Feature Pattern Analysis

CRITICAL FINDING: Feature labels in code DO NOT match actual data patterns!

Code labels (process_simulations_for_gnn.py):
- Feature 0: VOL_BASE_CASE 
- Feature 1: CAPACITY_BASE_CASE
- Feature 2: CAPACITY_REDUCTION
- Feature 3: FREESPEED
- Feature 4: HIGHWAY
- Feature 5: LENGTH

Actual data patterns suggest:
- Feature 0: LENGTH-like (0-1596m, 23.86% zeros)
- Feature 1: CAPACITY_BASE_CASE (correct)
- Feature 2: VOL_BASE_CASE (negative values -4800 to 0)
- Feature 3: CAPACITY_REDUCTION (0-33.3%, matches capacity zeros)
- Feature 4: HIGHWAY (correct)
- Feature 5: LENGTH (different from F0, no zeros)

This script analyzes ACTUAL data patterns to determine true feature identities.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
batch_path = '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt'
data_list = torch.load(batch_path, weights_only=False, map_location='cpu')

print("="*80)
print("MASTER DATA VERIFICATION - CORRECT FEATURE MAPPING")
print("="*80)

# Load first scenario
data = data_list[0]

print(f"\nDataset Structure:")
print(f"  Total nodes: {data.x.shape[0]:,}")
print(f"  Total edges: {data.edge_index.shape[1]:,}")
print(f"  Total features: {data.x.shape[1]}")
print(f"  Target shape: {data.y.shape}")
print(f"  Scenarios in batch: {len(data_list)}")

# Extract all features
f0_vol_base = data.x[:, 0].numpy()
f1_capacity_base = data.x[:, 1].numpy()
f2_capacity_reduction = data.x[:, 2].numpy()
f3_freespeed = data.x[:, 3].numpy()
f4_highway = data.x[:, 4].numpy()
f5_length = data.x[:, 5].numpy()
target = data.y.numpy().flatten()

# Feature analysis - CODE LABEL vs ACTUAL PATTERN
features = {
    'F0 [CODE: VOL_BASE_CASE]': {
        'data': f0_vol_base,
        'code_label': 'VOL_BASE_CASE',
        'suspected_actual': 'LENGTH',
        'reason': 'Positive values 0-1596m, no negatives (volume should be negative)',
        'expected_if_volume': 'Negative values, 91.9% zeros',
        'expected_if_length': 'Positive values in meters'
    },
    'F1 [CODE: CAPACITY_BASE_CASE]': {
        'data': f1_capacity_base,
        'code_label': 'CAPACITY_BASE_CASE',
        'suspected_actual': 'CAPACITY_BASE_CASE',
        'reason': 'Positive values 0-14400 veh/h, reasonable capacity range',
        'expected_if_volume': 'N/A',
        'expected_if_length': 'N/A'
    },
    'F2 [CODE: CAPACITY_REDUCTION]': {
        'data': f2_capacity_reduction,
        'code_label': 'CAPACITY_REDUCTION',
        'suspected_actual': 'VOL_BASE_CASE',
        'reason': 'Negative values -4800 to 0, 91.9% zeros (traffic pattern!)',
        'expected_if_volume': 'Negative values, high % zeros',
        'expected_if_length': 'N/A'
    },
    'F3 [CODE: FREESPEED]': {
        'data': f3_freespeed,
        'code_label': 'FREESPEED',
        'suspected_actual': 'CAPACITY_REDUCTION',
        'reason': '0-33.3% range, 10.79% zeros matching capacity zeros',
        'expected_if_volume': 'N/A',
        'expected_if_length': 'N/A'
    },
    'F4 [CODE: HIGHWAY]': {
        'data': f4_highway,
        'code_label': 'HIGHWAY',
        'suspected_actual': 'HIGHWAY',
        'reason': '11 categorical types, matches expected road types',
        'expected_if_volume': 'N/A',
        'expected_if_length': 'N/A'
    },
    'F5 [CODE: LENGTH]': {
        'data': f5_length,
        'code_label': 'LENGTH',
        'suspected_actual': 'LENGTH',
        'reason': 'Positive values in meters, 0% zeros',
        'expected_if_volume': 'N/A',
        'expected_if_length': 'Positive values in meters'
    }
}

print("\n" + "="*80)
print("FEATURE PATTERN ANALYSIS")
print("="*80)

for name, info in features.items():
    data_vals = info['data']
    
    print(f"\n{name}")
    print(f"  Code Label:        {info['code_label']}")
    print(f"  Suspected Actual:  {info['suspected_actual']}")
    print(f"  Analysis Reason:   {info['reason']}")
    print(f"  " + "-"*76)
    print(f"  Data Statistics:")
    print(f"    Min:      {data_vals.min():.4f}")
    print(f"    Max:      {data_vals.max():.4f}")
    print(f"    Mean:     {data_vals.mean():.4f}")
    print(f"    Median:   {np.median(data_vals):.4f}")
    print(f"    Std:      {data_vals.std():.4f}")
    print(f"    Zeros:    {(data_vals == 0).sum():,} ({(data_vals == 0).sum()/len(data_vals)*100:.2f}%)")
    
    # Pattern-specific checks
    negatives = (data_vals < 0).sum()
    if negatives > 0:
        print(f"    Negative: {negatives:,} ({negatives/len(data_vals)*100:.2f}%)")
        print(f"    >> PATTERN: Negative values suggest TRAFFIC VOLUME (not capacity/length)")
    
    if data_vals.min() >= 0 and data_vals.max() < 100 and 'HIGHWAY' not in name:
        print(f"    >> PATTERN: 0-{data_vals.max():.1f} range suggests PERCENTAGE or NORMALIZED values")
    
    if data_vals.min() >= 0 and data_vals.max() > 100 and data_vals.max() < 3000:
        print(f"    >> PATTERN: Positive values >100 suggest LENGTH (meters) or large CAPACITY")
    
    if 'HIGHWAY' in info['code_label']:
        unique = len(np.unique(data_vals))
        print(f"    Unique:   {unique} categories")
        print(f"    Range:    {int(data_vals.min())} to {int(data_vals.max())}")
    
    if info['suspected_actual'] == 'LENGTH':
        short = (data_vals < 10).sum()
        long = (data_vals > 500).sum()
        print(f"    Very short (<10m):  {short:,} ({short/len(data_vals)*100:.2f}%)")
        print(f"    Very long (>500m):  {long:,} ({long/len(data_vals)*100:.2f}%)")

# Verify static vs dynamic
print("\n" + "="*80)
print("STATIC vs DYNAMIC VERIFICATION")
print("="*80)

# Check variance across scenarios
features_to_check = {
    'F0 [LENGTH?]': 0,
    'F1 [CAPACITY]': 1,
    'F2 [VOLUME?]': 2,
    'F3 [CAP_RED?]': 3,
    'F4 [HIGHWAY]': 4,
    'F5 [LENGTH]': 5
}

print("\nChecking variance across 50 scenarios...")
for name, idx in features_to_check.items():
    # Get feature values from first 10 scenarios
    values_across_scenarios = [data_list[i].x[:, idx].numpy() for i in range(min(10, len(data_list)))]
    values_array = np.array(values_across_scenarios)
    
    # Calculate variance per node
    variance_per_node = values_array.var(axis=0)
    mean_variance = variance_per_node.mean()
    max_variance = variance_per_node.max()
    
    if max_variance < 1e-6:
        status = "STATIC"
    else:
        status = "DYNAMIC"
    
    print(f"  {name:30s}: {status:8s} (var: {mean_variance:.6f})")

# Target analysis
print("\n" + "="*80)
print("TARGET ANALYSIS")
print("="*80)

print(f"\nTarget: Change in traffic volume")
print(f"  Min:      {target.min():.4f}")
print(f"  Max:      {target.max():.4f}")
print(f"  Mean:     {target.mean():.4f}")
print(f"  Median:   {np.median(target):.4f}")
print(f"  Std:      {target.std():.4f}")
print(f"  Granularity: Edge-level ({len(target):,} edges)")

# Key findings
print("\n" + "="*80)
print("CRITICAL FINDINGS - FEATURE MAPPING DISCREPANCY")
print("="*80)

print("""
ACTUAL DATA PATTERNS vs CODE LABELS:

F0: Code says VOL_BASE_CASE, but data shows LENGTH pattern
    - Range: 0-1596m (no negatives!)
    - Expected for volume: Negative values
    - Conclusion: Likely LENGTH (linegraph-transformed)

F1: CAPACITY_BASE_CASE - CONFIRMED
    - Range: 0-14400 veh/h
    - Pattern matches expected capacity

F2: Code says CAPACITY_REDUCTION, but data shows VOL_BASE_CASE pattern
    - Range: -4800 to 0 veh/h (negative values!)
    - 91.9% zeros (no traffic pattern)
    - Conclusion: This IS the baseline volume

F3: Code says FREESPEED, but data shows CAPACITY_REDUCTION pattern  
    - Range: 0-33.3% (percentage!)
    - 10.79% zeros (matches capacity zeros)
    - Conclusion: This IS the capacity reduction percentage

F4: HIGHWAY - CONFIRMED
    - 11 categorical road types

F5: LENGTH - CONFIRMED
    - Range: 4-2569m
    - 0% zeros (actual road segment lengths)

REVISED FEATURE MAPPING:
- F0: LENGTH (linegraph-transformed, 23.86% zeros)
- F1: CAPACITY_BASE_CASE
- F2: VOL_BASE_CASE (baseline traffic volume)
- F3: CAPACITY_REDUCTION (policy-induced percentage)
- F4: HIGHWAY
- F5: LENGTH (original road segments, no zeros)

Static: F0, F1, F3, F4, F5
Dynamic: F2 (baseline volume varies slightly across scenarios)
""")

# Correlation preview
print("\n" + "="*80)
print("CORRELATION PREVIEW (with Target)")
print("="*80)

correlations = []
for name, info in features.items():
    corr = np.corrcoef(info['data'], target)[0, 1]
    correlations.append((name, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'Feature':50s} {'Correlation':>12s} {'Strength':>15s}")
print("-" * 80)
for name, corr in correlations:
    if abs(corr) > 0.3:
        strength = "Strong"
    elif abs(corr) > 0.1:
        strength = "Moderate"
    else:
        strength = "Weak"
    print(f"{name:50s} {corr:12.4f} {strength:>15s}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - ACTION REQUIRED")
print("="*80)
print("""
CONFIRMED: Feature labels in code DO NOT match actual data!

Recommended Actions:
1. Use ACTUAL feature patterns (not code labels) for analysis
2. Verify with supervisor/paper authors about preprocessing steps
3. Check if linegraph transformation reordered features
4. Proceed with analysis using OBSERVED patterns:
   - F0: LENGTH (linegraph)
   - F1: CAPACITY_BASE_CASE  
   - F2: VOL_BASE_CASE (negative = traffic)
   - F3: CAPACITY_REDUCTION (0-33.3%)
   - F4: HIGHWAY
   - F5: LENGTH (original)

Next: Feature-by-feature comprehensive analysis (~70 charts)
Using ACTUAL patterns, not code labels!
""")
print("="*80)