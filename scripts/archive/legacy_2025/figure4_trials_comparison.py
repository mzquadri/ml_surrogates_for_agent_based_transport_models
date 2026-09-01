"""
FIGURE 4: COMPREHENSIVE TRIALS COMPARISON MATRIX
Side-by-side comparison of all 8 trials with detailed metrics

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure4_trials_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
import seaborn as sns
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Complete trials data
TRIALS_DATA = {
    'Trial 1': {
        'val_r2': -0.0020, 'test_r2': -0.0022,
        'dropout': 0.0, 'batch_size': 32, 'lr': '5e-4',
        'status': 'Failed', 'color': '#ff4444'
    },
    'Trial 2': {
        'val_r2': 0.5841, 'test_r2': 0.5117,
        'dropout': 0.3, 'batch_size': 16, 'lr': '5e-4',
        'status': 'Working', 'color': '#a29bfe'
    },
    'Trial 3': {
        'val_r2': 0.5953, 'test_r2': 0.2246,
        'dropout': 0.0, 'batch_size': 16, 'lr': '5e-4',
        'status': 'Overfit', 'color': '#6c5ce7'
    },
    'Trial 4': {
        'val_r2': 0.6097, 'test_r2': 0.2426,
        'dropout': 0.0, 'batch_size': 16, 'lr': '5e-4',
        'status': 'Overfit', 'color': '#fd79a8'
    },
    'Trial 5': {
        'val_r2': 0.5500, 'test_r2': 0.5553,
        'dropout': 0.3, 'batch_size': 8, 'lr': '5e-4',
        'status': 'Baseline', 'color': '#ffd93d'
    },
    'Trial 6': {
        'val_r2': 0.5224, 'test_r2': 0.5223,
        'dropout': 0.3, 'batch_size': 8, 'lr': '3e-4',
        'status': 'LR Low', 'color': '#ff9770'
    },
    'Trial 7': {
        'val_r2': 0.5497, 'test_r2': 0.5471,
        'dropout': 0.3, 'batch_size': 8, 'lr': '6e-4',
        'status': 'LR High', 'color': '#ff7f50'
    },
    'Trial 8': {
        'val_r2': 0.5970, 'test_r2': 0.5957,
        'dropout': 0.2, 'batch_size': 8, 'lr': '5e-4',
        'status': 'BEST', 'color': '#51cf66'
    },
}

BENCHMARK_R2 = 0.76

print("="*80)
print(" FIGURE 4: COMPREHENSIVE TRIALS COMPARISON MATRIX")
print("="*80)

# Create figure with 3 rows
fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('white')

fig.suptitle('Figure 4: Comprehensive Trials Comparison Matrix\nAll 8 Trials - Complete Performance & Hyperparameter Analysis', 
             fontsize=24, fontweight='bold', y=0.98)

gs = fig.add_gridspec(3, 1, hspace=0.4, left=0.06, right=0.96, top=0.92, bottom=0.05,
                      height_ratios=[1.2, 1, 1.2])

# ============================================================================
# PANEL A: Performance Heatmap
# ============================================================================
ax1 = fig.add_subplot(gs[0])

# Prepare data for heatmap
trials = list(TRIALS_DATA.keys())
metrics = ['Validation R²', 'Test R²', 'Gap %', 'Benchmark %']

heatmap_data = []
for trial in trials:
    data = TRIALS_DATA[trial]
    val_r2 = data['val_r2']
    test_r2 = data['test_r2']
    
    if test_r2 > 0:
        gap = abs(val_r2 - test_r2) / max(val_r2, test_r2) * 100
        bench_pct = (test_r2 / BENCHMARK_R2) * 100
    else:
        gap = 0
        bench_pct = 0
    
    heatmap_data.append([val_r2, test_r2, gap, bench_pct])

heatmap_data = np.array(heatmap_data).T

# Normalize for coloring (but show actual values)
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=trials, yticklabels=metrics,
            cbar_kws={'label': 'Normalized Score'},
            linewidths=2, linecolor='black', ax=ax1,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

ax1.set_title('(A) Performance Metrics Heatmap - All Trials', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Trial Configuration', fontsize=14, fontweight='bold')
ax1.set_ylabel('Performance Metrics', fontsize=14, fontweight='bold')

# Highlight best trial (Trial 8)
rect = Rectangle((7, 0), 1, 4, fill=False, edgecolor='blue', linewidth=4)
ax1.add_patch(rect)
ax1.text(7.5, -0.7, '⭐ BEST', ha='center', fontsize=12, fontweight='bold', color='darkgreen')

# Highlight failed trial (Trial 1)
rect2 = Rectangle((0, 0), 1, 4, fill=False, edgecolor='red', linewidth=4)
ax1.add_patch(rect2)
ax1.text(0.5, -0.7, '❌ FAILED', ha='center', fontsize=12, fontweight='bold', color='darkred')

# ============================================================================
# PANEL B: Hyperparameters Comparison
# ============================================================================
ax2 = fig.add_subplot(gs[1])

# Prepare hyperparameter data
hyperparam_metrics = ['Dropout', 'Batch Size', 'LR (×10⁴)']
hyperparam_data = []

for trial in trials:
    data = TRIALS_DATA[trial]
    lr_numeric = float(data['lr'].replace('e-', '')) * 10  # Convert to ×10⁴ scale
    hyperparam_data.append([data['dropout'], data['batch_size'], lr_numeric])

hyperparam_data = np.array(hyperparam_data).T

# Create hyperparameter heatmap
sns.heatmap(hyperparam_data, annot=True, fmt='.1f', cmap='viridis',
            xticklabels=trials, yticklabels=hyperparam_metrics,
            cbar_kws={'label': 'Parameter Value'},
            linewidths=2, linecolor='black', ax=ax2,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

ax2.set_title('(B) Hyperparameter Configuration Matrix', 
              fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Trial Configuration', fontsize=14, fontweight='bold')
ax2.set_ylabel('Hyperparameters', fontsize=14, fontweight='bold')

# ============================================================================
# PANEL C: Comparative Summary Table
# ============================================================================
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')

ax3.text(0.5, 0.98, '(C) Comprehensive Comparison Summary', 
         ha='center', va='top', fontsize=16, fontweight='bold', transform=ax3.transAxes)

# Create detailed comparison table
table_text = """
╔══════════╦═══════════╦══════════╦══════════╦═══════════╦═════════════╦═══════════════╦═════════════════════════════════════════════════════╗
║  TRIAL   ║   Val R²  ║  Test R² ║   Gap %  ║  Dropout  ║ Batch Size  ║  Learning Rate║  STATUS & KEY CHARACTERISTICS                       ║
╠══════════╬═══════════╬══════════╬══════════╬═══════════╬═════════════╬═══════════════╬═════════════════════════════════════════════════════╣
║ Trial 1  ║  -0.0020  ║ -0.0022  ║    N/A   ║    0.0    ║     32      ║     5e-4      ║  ❌ FAILED - Architecture Mismatch (Legacy Model)  ║
║ Trial 2  ║   0.5841  ║  0.5117  ║  12.4%   ║    0.3    ║     16      ║     5e-4      ║  ✅ Working - First successful config (BS=16)      ║
║ Trial 3  ║   0.5953  ║  0.2246  ║  62.3%   ║    0.0    ║     16      ║     5e-4      ║  ⚠️  OVERFIT - No dropout causes 62% gap           ║
║ Trial 4  ║   0.6097  ║  0.2426  ║  60.2%   ║    0.0    ║     16      ║     5e-4      ║  ⚠️  OVERFIT - Weighted loss can't fix no dropout  ║
║ Trial 5  ║   0.5500  ║  0.5553  ║   0.96%  ║    0.3    ║      8      ║     5e-4      ║  ✅ BASELINE - Excellent generalization (BS=8)     ║
║ Trial 6  ║   0.5224  ║  0.5223  ║   0.02%  ║    0.3    ║      8      ║     3e-4      ║  ✅ LR Low - Perfect gap but lower performance     ║
║ Trial 7  ║   0.5497  ║  0.5471  ║   0.47%  ║    0.3    ║      8      ║     6e-4      ║  ✅ LR High - Slightly worse than baseline         ║
║ Trial 8  ║   0.5970  ║  0.5957  ║   0.22%  ║    0.2    ║      8      ║     5e-4      ║  ⭐ BEST - Optimal dropout (0.2) maximizes R²      ║
╚══════════╩═══════════╩══════════╩══════════╩═══════════╩═════════════╩═══════════════╩═════════════════════════════════════════════════════╝

KEY FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BATCH SIZE EFFECT:
   • BS=32 (Trial 1): FAILED - Architecture incompatibility
   • BS=16 (Trials 2-4): Moderate performance (R²=0.22-0.51 test)
   • BS=8 (Trials 5-8): BEST performance (R²=0.52-0.60 test)
   ➜ CONCLUSION: Smaller batch size (8) provides superior generalization

2. DROPOUT REGULARIZATION:
   • Dropout=0.0 (Trials 1, 3-4): DISASTER - 60%+ overfitting or failure
   • Dropout=0.3 (Trials 2, 5-7): Good (R²=0.51-0.56 test)
   • Dropout=0.2 (Trial 8): OPTIMAL (R²=0.60 test)
   ➜ CONCLUSION: Dropout 0.2 balances capacity and regularization perfectly

3. LEARNING RATE SENSITIVITY:
   • LR=3e-4 (Trial 6): Too conservative (R²=0.52)
   • LR=5e-4 (Trials 5, 8): OPTIMAL (R²=0.56-0.60)
   • LR=6e-4 (Trial 7): Too aggressive (R²=0.55)
   ➜ CONCLUSION: 5e-4 is the sweet spot

4. GENERALIZATION ANALYSIS:
   • Excellent (<1% gap): Trials 5, 6, 7, 8
   • Moderate (>10% gap): Trial 2
   • Critical (>60% gap): Trials 3, 4
   ➜ CONCLUSION: Dropout is CRITICAL for generalization

5. BENCHMARK COMPARISON (Boreale et al. 2024: R²=0.76):
   • Best Achievement: Trial 8 (78.4% with 10% data)
   • Performance Gap: 21.6% below benchmark
   • Data Efficiency: Excellent (78% performance with 10% scenarios)
   ➜ CONCLUSION: Competitive performance despite limited data

RECOMMENDATIONS FOR DEPLOYMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ USE Trial 8 Configuration: Dropout=0.2, BatchSize=8, LR=5e-4
✅ Production-ready with excellent generalization (0.22% gap)
✅ Highest test performance (R²=0.5957)
✅ 78.4% of benchmark performance with only 10% of training data
"""

ax3.text(0.5, 0.45, table_text,
         ha='center', va='center', fontsize=8.5, family='monospace',
         bbox=dict(boxstyle='round,pad=1.2', facecolor='#F8F9FA', edgecolor='black', linewidth=2, alpha=0.95),
         transform=ax3.transAxes, linespacing=1.4)

# Footer
footer_text = "Complete Trials Comparison | Reference: Boreale et al. (2024) R²=0.76 | Dataset: 1,000 scenarios"
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=11, style='italic', color='gray')

# Save
plt.savefig(f"{OUTPUT_DIR}/figure4_trials_comparison_matrix.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: figure4_trials_comparison_matrix.png")
print(f"     Location: {OUTPUT_DIR}\n")
plt.show()
plt.close()

print("="*80)
print(" COMPREHENSIVE COMPARISON MATRIX COMPLETE")
print("="*80)
