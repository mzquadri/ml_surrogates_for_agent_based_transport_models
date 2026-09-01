"""
GeoJSON Analysis for Colab
Copy this code to your Colab notebook after running the batch analysis
"""
import json
import pandas as pd
from pathlib import Path

# Define GeoJSON file path (UPDATE THIS PATH IN COLAB)
geojson_path = Path("/content/drive/MyDrive/Zamin_thesis/ml_surrogates_for_agent_based_transport_models/data/visualisation/districts_paris.geojson")

print("="*80)
print("GEOJSON FILE ANALYSIS - PARIS DISTRICTS")
print("="*80)

# Load GeoJSON file
with open(geojson_path, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

print(f"\nFile Type: {geojson_data.get('type', 'Unknown')}")
print(f"Number of Features: {len(geojson_data.get('features', []))}")

# Analyze each district
print("\n" + "-"*80)
print("DISTRICT INFORMATION")
print("-"*80)

districts = []
for idx, feature in enumerate(geojson_data.get('features', [])):
    props = feature.get('properties', {})
    geom = feature.get('geometry', {})
    
    district_info = {
        'index': idx,
        'geometry_type': geom.get('type', 'Unknown'),
        'num_coordinates': len(geom.get('coordinates', [])) if geom.get('coordinates') else 0
    }
    
    # Add all properties
    district_info.update(props)
    districts.append(district_info)
    
    # Print first few to understand structure
    if idx < 5:
        print(f"\nDistrict {idx + 1}:")
        print(f"  Properties: {props}")
        print(f"  Geometry Type: {geom.get('type')}")
        print(f"  Coordinate Arrays: {len(geom.get('coordinates', []))}")

# Create DataFrame for analysis
df_districts = pd.DataFrame(districts)

print("\n" + "="*80)
print("DISTRICT SUMMARY")
print("="*80)
print(f"\nTotal Districts: {len(districts)}")
print(f"\nAvailable Properties:")
for col in df_districts.columns:
    if col not in ['index', 'geometry_type', 'num_coordinates']:
        print(f"  - {col}")
        if df_districts[col].dtype in ['int64', 'float64']:
            print(f"    Range: [{df_districts[col].min()}, {df_districts[col].max()}]")
        else:
            unique_vals = df_districts[col].unique()
            if len(unique_vals) <= 10:
                print(f"    Values: {list(unique_vals)}")
            else:
                print(f"    Unique values: {len(unique_vals)}")

print("\n" + "-"*80)
print("FULL DISTRICT TABLE")
print("-"*80)
print(df_districts.to_string())

print("\n" + "="*80)
print("GEOJSON ANALYSIS COMPLETE")
print("="*80)
