"""
FIGURE 2: TRIAL 1 DETAILED ANALYSIS
Failed Model - Architecture Mismatch Case Study

Architecture: PointNetTransfGAT (Incompatible Checkpoint)
Status:  FAILED - Negative R² Score

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure2_trial1_detailed.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trial 1 Data
TRIAL_NAME = "Trial 1: Batch Size 32 Experiment"
HYPERPARAMETERS = {
    'Dropout Rate': 0.0,
    'Batch Size': 32,
    'Learning Rate': '5e-4',
    'Weighted Loss': 'No',
    'Optimizer': 'AdamW',
    'Architecture': 'PointNetTransfGAT'
}

RESULTS = {
    'Validation R²': -0.0020,
    'Test R²': -0.0022,
    'Training Status': 'FAILED',
    'Reason': 'Architecture Mismatch'
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" FIGURE 2: TRIAL 1 - FAILED MODEL")
print("="*80)
print(f"\n{TRIAL_NAME}")
print("\nSTATUS: [FAILED]")
print("\nHyperparameters:")
for key, val in HYPERPARAMETERS.items():
    print(f"  • {key}: {val}")
print("\nResults:")
for key, val in RESULTS.items():
    print(f"  • {key}: {val}")
print("\n" + "="*80 + "\n")

# Create figure
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('white')

# Main title
fig.suptitle('Figure 2: Trial 1 - Complete Failure Analysis\nBatch Size 32 Experiment with Architecture Mismatch', 
             fontsize=22, fontweight='bold', y=0.98)

# Create 3 panels
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.08)

# ============================================================================
# PANEL A: R² Comparison
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

metrics = ['Validation R²', 'Test R²', 'Benchmark\n(Boreale 2024)']
values = [RESULTS['Validation R²'], RESULTS['Test R²'], BENCHMARK_R2]
colors = ['#ff4444', '#cc0000', '#FFD700']

bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85, width=0.6)

# Add 3D effect
for bar in bars[:2]:  # Failed bars
    bar.set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='darkred', alpha=0.4)])
bars[2].set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.3)])

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    if i < 2:  # Negative values
        y_pos = 0.05
        va = 'bottom'
    else:
        y_pos = height + 0.02
        va = 'bottom'
    
    ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:.4f}',
            ha='center', va=va, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))

ax1.axhline(y=0, color='black', linestyle='-', linewidth=2, zorder=0)
ax1.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax1.set_title('(A) Performance Comparison\nTrial 1 vs Benchmark', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(-0.1, 0.85)
ax1.grid(True, alpha=0.3, axis='y')

# Add failure annotation
ax1.annotate('[X] NEGATIVE R2\nWorse than mean prediction', 
            xy=(0.5, -0.002), xytext=(0.5, 0.3),
            fontsize=11, ha='center', fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE4E1', edgecolor='red', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# ============================================================================
# PANEL B: Hyperparameters Box
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, '(B) Complete Hyperparameters Configuration', 
         ha='center', va='top', fontsize=14, fontweight='bold', transform=ax2.transAxes)

# Hyperparameters box
hyperparam_text = "TRIAL 1 CONFIGURATION\n" + "="*45 + "\n\n"
hyperparam_text += f"Architecture: {HYPERPARAMETERS['Architecture']}\n"
hyperparam_text += f"Model Parameters: 1.55 Million\n\n"
hyperparam_text += "Training Hyperparameters:\n"
hyperparam_text += f"  • Dropout Rate: {HYPERPARAMETERS['Dropout Rate']}\n"
hyperparam_text += f"  • Batch Size: {HYPERPARAMETERS['Batch Size']}\n"
hyperparam_text += f"  • Learning Rate: {HYPERPARAMETERS['Learning Rate']}\n"
hyperparam_text += f"  • Weighted Loss: {HYPERPARAMETERS['Weighted Loss']}\n"
hyperparam_text += f"  • Optimizer: {HYPERPARAMETERS['Optimizer']}\n\n"
hyperparam_text += "Dataset:\n"
hyperparam_text += "  • Training Scenarios: 1,000\n"
hyperparam_text += "  • Validation Split: 15%\n"
hyperparam_text += "  • Test Split: 15%\n\n"
hyperparam_text += "Status:  FAILED\n"
hyperparam_text += "Reason: Architecture Mismatch\n"
hyperparam_text += "(Legacy checkpoint incompatible)"

ax2.text(0.5, 0.45, hyperparam_text,
         ha='center', va='center', fontsize=11, family='monospace',
         bbox=dict(boxstyle='round,pad=1.0', facecolor='#FFE4E1', edgecolor='red', linewidth=3, alpha=0.9),
         transform=ax2.transAxes)

# ============================================================================
# PANEL C: Failure Analysis
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')

ax3.text(0.5, 0.95, '(C) Detailed Failure Analysis & Technical Report', 
         ha='center', va='top', fontsize=14, fontweight='bold', transform=ax3.transAxes)

failure_report = """
FAILURE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════════════

1. PROBLEM IDENTIFICATION:
   • Validation R²: -0.0020 (NEGATIVE)
   • Test R²: -0.0022 (NEGATIVE)
   • Status: Model performs WORSE than simply predicting mean value
   
2. ROOT CAUSE:
   ✗ Architecture Mismatch - Legacy checkpoint incompatible with current PointNetTransfGAT
   ✗ State Dict Load Errors:
      - 20 missing keys in checkpoint
      - 18 unexpected keys in current architecture
   ✗ Batch Size 32 was experimental configuration from early development phase
   
3. TECHNICAL DETAILS:
   • Model Type: PointNetTransfGAT (Graph Neural Network)
   • Total Parameters: 1,547,832 (1.55M)
   • Checkpoint Source: Early experiment with different architecture version
   • Error Type: torch.nn.modules.module.ModuleAttributeError
   
4. WHY NEGATIVE R²?
   R² = 1 - (SS_res / SS_tot)
   
   • SS_res = Σ(y_true - y_pred)²  [Sum of squared residuals]
   • SS_tot = Σ(y_true - y_mean)²  [Total sum of squares]
   
   When SS_res > SS_tot, R² becomes negative
   This means: Model predictions are WORSE than just using mean value
   
5. LESSONS LEARNED:
   ✓ Batch Size 32 abandoned in favor of smaller batch sizes (8, 16)
   ✓ Architecture compatibility must be verified before training
   ✓ Checkpoint versioning critical for reproducibility
   ✓ Early experiments showed BS=32 was suboptimal even when working
   
6. COMPARISON WITH BENCHMARK:
   • Boreale et al. (2024): R² = 0.76 (10,000 scenarios)
   • Trial 1: R² = -0.0022 (FAILED)
   • Gap: N/A (model non-functional)
   
7. RECOMMENDATION:
   ✗ DO NOT USE this configuration
   ✓ Use Trial 8 (Best Model): R² = 0.5957 with BS=8, Dropout=0.2
   ✓ Smaller batch sizes (8) provide better generalization
   ✓ Dropout regularization (0.2-0.3) is CRITICAL
"""

ax3.text(0.02, 0.50, failure_report,
         ha='left', va='center', fontsize=9.5, family='monospace',
         bbox=dict(boxstyle='round,pad=1.2', facecolor='#FFF5EE', edgecolor='darkred', linewidth=3, alpha=0.95),
         transform=ax3.transAxes, linespacing=1.6)

# Footer
footer_text = "Trial 1 Documentation | Architecture Mismatch Case Study | Reference: Boreale et al. (2024)"
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10, style='italic', color='gray')

# Save
plt.savefig(f"{OUTPUT_DIR}/figure2_trial1_detailed_analysis.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: figure2_trial1_detailed_analysis.png")
print(f"     Location: {OUTPUT_DIR}\n")
plt.show()
plt.close()

print("="*80)
print(" TRIAL 1 ANALYSIS COMPLETE")
print("="*80)
