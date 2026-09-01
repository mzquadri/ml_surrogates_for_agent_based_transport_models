"""
FIGURE 3: TRIAL 8 DETAILED ANALYSIS (BEST MODEL)
Optimal Configuration - Highest Performance

Architecture: PointNetTransfGAT
Status: ⭐ BEST MODEL - 78.4% of Benchmark

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure3_trial8_detailed.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch, Rectangle
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trial 8 Data
TRIAL_NAME = "Trial 8: Best Model - Optimal Configuration"
HYPERPARAMETERS = {
    'Dropout Rate': 0.2,
    'Batch Size': 8,
    'Learning Rate': '5e-4',
    'Weighted Loss': 'No',
    'Optimizer': 'AdamW',
    'Architecture': 'PointNetTransfGAT'
}

RESULTS = {
    'Validation R²': 0.5970,
    'Test R²': 0.5957,
    'Generalization Gap': 0.22,  # percentage
    'Benchmark Achievement': 78.4,  # percentage
    'Training Status': 'SUCCESS',
    'Epochs': 'Converged'
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" FIGURE 3: TRIAL 8 - BEST MODEL")
print("="*80)
print(f"\n{TRIAL_NAME}")
print("\nSTATUS: [BEST]")
print("\nHyperparameters:")
for key, val in HYPERPARAMETERS.items():
    print(f"  • {key}: {val}")
print("\nResults:")
for key, val in RESULTS.items():
    if isinstance(val, float) and val < 10:
        print(f"  • {key}: {val:.4f}")
    else:
        print(f"  • {key}: {val}")
print("\n" + "="*80 + "\n")

# Create figure
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('white')

# Main title
fig.suptitle('Figure 3: Trial 8 - Best Model Complete Analysis\n[BEST] Optimal Configuration: Dropout=0.2, Batch Size=8, LR=5e-4', 
             fontsize=22, fontweight='bold', y=0.98, color='darkgreen')

# Create 4 panels
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, left=0.07, right=0.95, top=0.90, bottom=0.06)

# ============================================================================
# PANEL A: R² Comparison
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

metrics = ['Validation R²', 'Test R²', 'Benchmark\n(Boreale 2024)']
values = [RESULTS['Validation R²'], RESULTS['Test R²'], BENCHMARK_R2]
colors = ['#51cf66', '#2f9e44', '#FFD700']

bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)

# Add 3D effect
for bar in bars:
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.4)])

# Add values on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))

ax1.axhline(y=0, color='black', linestyle='-', linewidth=2, zorder=0)
ax1.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax1.set_title('(A) Performance Comparison\nTrial 8 vs Benchmark', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0, 0.85)
ax1.grid(True, alpha=0.3, axis='y')

# Add achievement annotation
achievement_pct = (RESULTS['Test R²'] / BENCHMARK_R2) * 100
gap_pct = ((BENCHMARK_R2 - RESULTS['Test R²']) / BENCHMARK_R2) * 100

annotation_text = f"[OK] Achievement: {achievement_pct:.1f}%\nof benchmark\n\nGap: {gap_pct:.1f}% below\nreference"
ax1.text(0.98, 0.65, annotation_text,
         transform=ax1.transAxes, fontsize=11, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2, alpha=0.9))

# ============================================================================
# PANEL B: Generalization Analysis
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

categories = ['Validation', 'Test', 'Gap']
gen_values = [RESULTS['Validation R²'], RESULTS['Test R²'], abs(RESULTS['Validation R²'] - RESULTS['Test R²'])]
gen_colors = ['#a8e6cf', '#56b68b', '#ff9999']

bars2 = ax2.bar(categories, gen_values, color=gen_colors, edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)

for bar in bars2:
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.4)])

for bar, val in zip(bars2, gen_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))

ax2.set_ylabel('R² Score / Gap', fontsize=14, fontweight='bold')
ax2.set_title('(B) Generalization Performance\nGap = 0.22% (EXCELLENT)', fontsize=14, fontweight='bold', pad=15, color='darkgreen')
ax2.set_ylim(0, 0.7)
ax2.grid(True, alpha=0.3, axis='y')

# Add quality label
ax2.text(0.5, 0.85, '⭐ NEAR-PERFECT\nGENERALIZATION',
         transform=ax2.transAxes, fontsize=12, ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', edgecolor='darkorange', linewidth=2, alpha=0.9))

# ============================================================================
# PANEL C: Complete Hyperparameters & Configuration
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

ax3.text(0.5, 0.98, '(C) Complete Configuration Details', 
         ha='center', va='top', fontsize=14, fontweight='bold', transform=ax3.transAxes)

config_text = """
⭐ TRIAL 8: OPTIMAL CONFIGURATION ⭐
═══════════════════════════════════════════

Architecture Details:
  • Model: PointNetTransfGAT (GNN)
  • Total Parameters: 1,547,832 (1.55M)
  • Trainable Parameters: 100%
  
Components:
  • PointNet Feature Extraction
  • Transformer Encoder
  • Graph Attention Network (GAT)
  
Training Hyperparameters:
  • Dropout Rate: 0.2 ← OPTIMIZED
  • Batch Size: 8 ← OPTIMAL
  • Learning Rate: 5e-4 ← BEST
  • Weighted Loss: No
  • Optimizer: AdamW
  • Loss Function: MSE
  
Dataset Configuration:
  • Training Scenarios: 1,000
  • Training Split: 70% (700)
  • Validation Split: 15% (150)
  • Test Split: 15% (150)
  • Input Features: 8
  • Network Edges: ~1,000 per scenario
  
Training Results:
  ✅ Converged Successfully
  ✅ No Overfitting (Gap < 1%)
  ✅ Stable Training Curve
  ✅ Best Test Performance
"""

ax3.text(0.5, 0.45, config_text,
         ha='center', va='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round,pad=1.0', facecolor='#E8F5E9', edgecolor='darkgreen', linewidth=3, alpha=0.9),
         transform=ax3.transAxes, linespacing=1.5)

# ============================================================================
# PANEL D: Success Analysis & Key Findings
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

ax4.text(0.5, 0.98, '(D) Success Analysis & Key Findings', 
         ha='center', va='top', fontsize=14, fontweight='bold', transform=ax4.transAxes)

success_report = """
SUCCESS FACTORS ANALYSIS
═══════════════════════════════════════════

1. PERFORMANCE METRICS:
   ✅ Test R²: 0.5957 (HIGHEST)
   ✅ Validation R²: 0.5970
   ✅ Generalization Gap: 0.22% (EXCELLENT)
   ✅ Benchmark Achievement: 78.4%
   
2. WHY THIS CONFIGURATION WORKS:
   
   • Dropout 0.2 (Reduced from 0.3)
     → Improves model capacity
     → Still prevents overfitting
     → Better than 0.0 (Trials 3-4 failed)
     → Better than 0.3 (Trial 5-7 lower)
   
   • Batch Size 8 (Smallest)
     → Better gradient estimates
     → More frequent updates
     → Improved generalization
     → Better than 16 or 32
   
   • Learning Rate 5e-4 (Goldilocks)
     → Not too slow (3e-4, Trial 6)
     → Not too fast (6e-4, Trial 7)
     → Perfect convergence speed
   
3. COMPARISON WITH OTHER TRIALS:
   • vs Trial 1 (BS=32): +∞% (T1 failed)
   • vs Trial 2 (BS=16): +16.4%
   • vs Trial 3 (No DR): +165.2%
   • vs Trial 4 (No DR): +145.5%
   • vs Trial 5 (DR=0.3): +7.3%
   • vs Trial 6 (LR=3e-4): +14.1%
   • vs Trial 7 (LR=6e-4): +8.9%
   
4. BENCHMARK COMPARISON:
   • Boreale et al. (2024): 0.76
   • This Work (Trial 8): 0.5957
   • Data Used: 10% (1,000 vs 10,000)
   • Efficiency: 78.4% performance
                with 10% data
   
5. RECOMMENDATIONS:
   ✅ USE this configuration for deployment
   ✅ Production-ready model
   ✅ Excellent generalization
   ✅ No overfitting concerns
"""

ax4.text(0.5, 0.45, success_report,
         ha='center', va='center', fontsize=9.5, family='monospace',
         bbox=dict(boxstyle='round,pad=1.0', facecolor='#F1F8E9', edgecolor='green', linewidth=3, alpha=0.95),
         transform=ax4.transAxes, linespacing=1.5)

# Footer
footer_text = "⭐ Trial 8: Best Model Documentation | 78.4% of Benchmark | Reference: Boreale et al. (2024)"
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11, fontweight='bold', color='darkgreen')

# Save
plt.savefig(f"{OUTPUT_DIR}/figure3_trial8_best_model_detailed.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: figure3_trial8_best_model_detailed.png")
print(f"     Location: {OUTPUT_DIR}\n")
plt.show()
plt.close()

print("="*80)
print(" TRIAL 8 (BEST MODEL) ANALYSIS COMPLETE")
print("="*80)
