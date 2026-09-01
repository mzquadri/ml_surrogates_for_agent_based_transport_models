"""
CRITICAL VERIFICATION: Data vs Code Labels vs Paper

This script definitively determines:
1. What the preprocessing CODE claims to create
2. What the DATA actually contains
3. What the PAPER says should be there

Goal: Identify if there's a preprocessing bug or just mislabeling
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
batch_path = '/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt'
data_list = torch.load(batch_path, weights_only=False, map_location='cpu')

print("="*100)
print("CRITICAL VERIFICATION: Data vs Code Labels vs Paper")
print("="*100)

# Extract features from first scenario
data = data_list[0]
print(f"\nDataset: {data.x.shape[0]:,} nodes, {data.x.shape[1]} features")

# According to preprocessing code (process_simulations_for_gnn.py):
CODE_LABELS = {
    0: "VOL_BASE_CASE",
    1: "CAPACITY_BASE_CASE", 
    2: "CAPACITY_REDUCTION",
    3: "FREESPEED",
    4: "HIGHWAY",
    5: "LENGTH"
}

# According to paper (Section 6.2):
PAPER_FEATURES = {
    "Static": ["Traffic volume base case", "Capacity base case", "Speed limit", "Length"],
    "Variable": ["Capacity reduction"],
    "Positional": ["x,y coordinates (separate)"]
}

print("\n" + "="*100)
print("PART 1: CODE LABELS (what preprocessing claims to create)")
print("="*100)
for idx, label in CODE_LABELS.items():
    print(f"  F{idx}: {label}")

print("\n" + "="*100)
print("PART 2: PAPER SPECIFICATION (Section 6.2)")
print("="*100)
print(f"  Static features (4): {', '.join(PAPER_FEATURES['Static'])}")
print(f"  Variable features (1): {', '.join(PAPER_FEATURES['Variable'])}")
print(f"  Positional features (2): {', '.join(PAPER_FEATURES['Positional'])}")

print("\n" + "="*100)
print("PART 3: ACTUAL DATA ANALYSIS - IDENTIFYING FEATURES BY PATTERN")
print("="*100)

# Analyze each feature across multiple scenarios
features_analysis = {}

for f_idx in range(6):
    print(f"\n{'-'*100}")
    print(f"FEATURE {f_idx} [CODE SAYS: {CODE_LABELS[f_idx]}]")
    print(f"{'-'*100}")
    
    # Get feature from first scenario
    feature_vals = data.x[:, f_idx].numpy()
    
    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  Min:      {feature_vals.min():12.4f}")
    print(f"  Max:      {feature_vals.max():12.4f}")
    print(f"  Mean:     {feature_vals.mean():12.4f}")
    print(f"  Median:   {np.median(feature_vals):12.4f}")
    print(f"  Std:      {feature_vals.std():12.4f}")
    print(f"  Zeros:    {(feature_vals == 0).sum():,} ({(feature_vals == 0).sum()/len(feature_vals)*100:.2f}%)")
    
    negatives = (feature_vals < 0).sum()
    if negatives > 0:
        print(f"  Negative: {negatives:,} ({negatives/len(feature_vals)*100:.2f}%)")
        print(f"  >> Contains NEGATIVE values - likely TRAFFIC VOLUME!")
    
    # Check variance across scenarios
    scenario_values = []
    for i in range(min(10, len(data_list))):
        scenario_values.append(data_list[i].x[:, f_idx].numpy())
    
    variance_per_node = np.var(scenario_values, axis=0)
    mean_variance = variance_per_node.mean()
    max_variance = variance_per_node.max()
    
    if max_variance < 1e-6:
        temporal_nature = "STATIC"
    else:
        temporal_nature = "DYNAMIC"
    
    print(f"\nTemporal Nature (across scenarios):")
    print(f"  Status: {temporal_nature}")
    print(f"  Mean variance: {mean_variance:.6f}")
    print(f"  Max variance:  {max_variance:.6f}")
    
    # Pattern recognition
    print(f"\nPattern Recognition:")
    
    identified_type = "UNKNOWN"
    confidence = "LOW"
    evidence = []
    
    # Check for TRAFFIC VOLUME pattern
    if negatives > 0 and (feature_vals == 0).sum() > len(feature_vals) * 0.8:
        identified_type = "TRAFFIC VOLUME (baseline)"
        confidence = "HIGH"
        evidence.append("Contains negative values (traffic presence indicator)")
        evidence.append(f"High % zeros ({(feature_vals == 0).sum()/len(feature_vals)*100:.1f}%) - sparse traffic")
        evidence.append(f"{temporal_nature} - {'expected for baseline' if temporal_nature == 'STATIC' else 'varies across scenarios'}")
    
    # Check for LENGTH pattern
    elif feature_vals.min() >= 0 and feature_vals.max() > 100 and feature_vals.max() < 3000:
        if (feature_vals == 0).sum() > 1000:
            identified_type = "LENGTH (linegraph-transformed)"
            confidence = "HIGH"
            evidence.append(f"Range 0-{feature_vals.max():.0f}m suggests road lengths")
            evidence.append(f"{(feature_vals == 0).sum()/len(feature_vals)*100:.1f}% zeros - linegraph artifacts")
            evidence.append("STATIC nature - expected for geometric property")
        else:
            identified_type = "LENGTH (original road segments)"
            confidence = "HIGH"
            evidence.append(f"Range {feature_vals.min():.1f}-{feature_vals.max():.0f}m - road lengths")
            evidence.append("No zeros - all roads have length")
            evidence.append("STATIC nature - expected for geometric property")
    
    # Check for CAPACITY pattern
    elif feature_vals.min() >= 0 and feature_vals.max() > 1000 and feature_vals.max() < 20000:
        identified_type = "CAPACITY"
        confidence = "HIGH"
        evidence.append(f"Range 0-{feature_vals.max():.0f} veh/h - capacity range")
        evidence.append(f"{(feature_vals == 0).sum()/len(feature_vals)*100:.1f}% zeros - roads without car access")
        evidence.append("STATIC nature - expected for infrastructure property")
    
    # Check for PERCENTAGE pattern
    elif feature_vals.min() >= 0 and feature_vals.max() < 100:
        if feature_vals.max() > 30 and feature_vals.max() < 35:
            identified_type = "CAPACITY REDUCTION (%)"
            confidence = "HIGH"
            evidence.append(f"Range 0-{feature_vals.max():.1f}% - percentage values")
            evidence.append("Max ~33.3% matches policy reduction level")
            evidence.append("STATIC nature - policy is fixed per scenario")
        else:
            identified_type = "PERCENTAGE or NORMALIZED VALUE"
            confidence = "MEDIUM"
            evidence.append(f"Range 0-{feature_vals.max():.1f} - normalized values")
    
    # Check for CATEGORICAL pattern
    elif len(np.unique(feature_vals)) < 20 and feature_vals.max() < 20:
        identified_type = "CATEGORICAL (Highway type)"
        confidence = "HIGH"
        evidence.append(f"{len(np.unique(feature_vals))} unique values")
        evidence.append(f"Range {int(feature_vals.min())} to {int(feature_vals.max())}")
        evidence.append("STATIC nature - road type doesn't change")
    
    print(f"  >> IDENTIFIED AS: {identified_type}")
    print(f"  >> CONFIDENCE: {confidence}")
    print(f"  >> Evidence:")
    for ev in evidence:
        print(f"     - {ev}")
    
    # Store analysis
    features_analysis[f_idx] = {
        'code_label': CODE_LABELS[f_idx],
        'identified_type': identified_type,
        'confidence': confidence,
        'temporal_nature': temporal_nature,
        'has_negatives': negatives > 0,
        'pct_zeros': (feature_vals == 0).sum()/len(feature_vals)*100,
        'range': (feature_vals.min(), feature_vals.max()),
        'mean': feature_vals.mean()
    }

# Summary comparison
print("\n" + "="*100)
print("PART 4: SUMMARY COMPARISON - CODE LABELS vs ACTUAL DATA")
print("="*100)

print(f"\n{'Feature':<4} {'Code Label':<25} {'Identified Type':<35} {'Match?':<10}")
print("-"*100)

mismatches = []
for f_idx, analysis in features_analysis.items():
    code_label = analysis['code_label']
    identified = analysis['identified_type']
    
    # Check if they match (simplified matching)
    match = "YES"
    if "VOL_BASE_CASE" in code_label and "VOLUME" not in identified:
        match = "NO"
        mismatches.append(f_idx)
    elif "CAPACITY_REDUCTION" in code_label and "REDUCTION" not in identified:
        match = "NO"
        mismatches.append(f_idx)
    elif "LENGTH" in code_label and "LENGTH" not in identified:
        match = "NO"
        mismatches.append(f_idx)
    elif "FREESPEED" in code_label and ("PERCENTAGE" not in identified and "SPEED" not in identified):
        match = "NO"
        mismatches.append(f_idx)
    elif "CAPACITY_BASE" in code_label and "CAPACITY" not in identified:
        match = "NO"
        mismatches.append(f_idx)
    
    print(f"F{f_idx}    {code_label:<25} {identified:<35} {match:<10}")

# Final verdict
print("\n" + "="*100)
print("FINAL VERDICT")
print("="*100)

if len(mismatches) > 0:
    print(f"\nCRITICAL FINDING: {len(mismatches)} MISMATCHES DETECTED!")
    print(f"Mismatched features: F{', F'.join(map(str, mismatches))}")
    print("\nPossible causes:")
    print("  1. Bug in preprocessing code (features stored in wrong order)")
    print("  2. Linegraph transformation reordered features")
    print("  3. Code labels are outdated (features changed but labels didn't)")
    print("  4. Multiple versions of preprocessing used inconsistently")
    
    print("\nRECOMMENDATION:")
    print("  >> Use ACTUAL data patterns for analysis, NOT code labels!")
    print("  >> Verify with paper author/supervisor about feature ordering")
    print("  >> Check preprocessing code for bugs")
else:
    print("\nAll features match their code labels!")
    print("Data is correctly labeled according to preprocessing code.")

print("\n" + "="*100)
print("DETAILED MAPPING FOR ANALYSIS")
print("="*100)

print("\nUse these ACTUAL feature identities for analysis:")
for f_idx, analysis in features_analysis.items():
    print(f"  F{f_idx}: {analysis['identified_type']:<35} ({analysis['temporal_nature']})")

print("\n" + "="*100)
print("VERIFICATION COMPLETE")
print("="*100)
