"""
FIGURE 4A: VALIDATION VS TEST R² SCATTER PLOT
Generalization Quality Assessment (All 8 Trials)

Reference: Boreale et al. (2024) - ML Surrogates for Agent-Based Transport Models
Benchmark: R² = 0.76 (10,000 scenarios)

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/figure4a_val_vs_test_scatter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Complete evaluation metrics (All 8 Trials)
TRIALS_DATA = {
    'Trial 1': {'val': -0.0020, 'test': -0.0022, 'status': 'FAILED', 'color': '#ff4444'},
    'Trial 2': {'val': 0.5841, 'test': 0.5117, 'status': 'OK', 'color': '#a29bfe'},
    'Trial 3': {'val': 0.5953, 'test': 0.2246, 'status': 'OVERFIT', 'color': '#fd79a8'},
    'Trial 4': {'val': 0.6097, 'test': 0.2426, 'status': 'OVERFIT', 'color': '#6c5ce7'},
    'Trial 5': {'val': 0.5500, 'test': 0.5553, 'status': 'EXCELLENT', 'color': '#ffd93d'},
    'Trial 6': {'val': 0.5224, 'test': 0.5223, 'status': 'EXCELLENT', 'color': '#ff9770'},
    'Trial 7': {'val': 0.5497, 'test': 0.5471, 'status': 'EXCELLENT', 'color': '#ff7f50'},
    'Trial 8': {'val': 0.5970, 'test': 0.5957, 'status': 'BEST', 'color': '#51cf66'}
}

print("="*80)
print(" FIGURE 4A: VALIDATION VS TEST R² SCATTER")
print("="*80)
print("\nGeneralization Quality Assessment")
print("Total Trials: 8")
print("Reference: Boreale et al. (2024)")
print("\n" + "="*80 + "\n")

# Create figure with modern styling
fig, ax = plt.subplots(figsize=(20, 15))
fig.patch.set_facecolor('#f5f6fa')
ax.set_facecolor('#ffffff')

# Add comprehensive ML zones with better spacing
from matplotlib.patches import Rectangle, Polygon

# 1. EXCELLENT ZONE (top right) - Both high, good generalization
excellent_zone = Rectangle((0.55, 0.55), 0.10, 0.10, 
                          facecolor='#d4edda', alpha=0.4, zorder=0,
                          edgecolor='#28a745', linewidth=2, linestyle='--')
ax.add_patch(excellent_zone)
ax.text(0.60, 0.60, 'EXCELLENT\nZONE\n(High Val & Test)', ha='center', va='center',
       fontsize=8, fontweight='bold', color='#155724', alpha=0.8)

# 2. OVERFITTING ZONE (upper left triangle - high val, low test)
overfit_vertices = np.array([[0.45, 0.20], [0.65, 0.20], [0.65, 0.45]])
overfit_zone = Polygon(overfit_vertices, facecolor='#f8d7da', alpha=0.35, 
                      edgecolor='#dc3545', linewidth=2, linestyle=':', zorder=0)
ax.add_patch(overfit_zone)
ax.text(0.58, 0.28, 'OVERFITTING\n(High Val, Low Test)\nMEMORIZATION', 
       ha='center', va='center', fontsize=8, fontweight='bold', 
       color='#721c24', alpha=0.75, style='italic')

# 3. UNDERFITTING ZONE (lower left - both low)
underfit_zone = Rectangle((0.15, 0.15), 0.15, 0.15, 
                         facecolor='#fff3cd', alpha=0.35, zorder=0,
                         edgecolor='#ffc107', linewidth=2, linestyle='-.')
ax.add_patch(underfit_zone)
ax.text(0.225, 0.225, 'UNDERFITTING\n(Both Low)\nPOOR MODEL', 
       ha='center', va='center', fontsize=8, fontweight='bold', 
       color='#856404', alpha=0.75)

# 4. ACCEPTABLE ZONE (mid-range)
acceptable_zone = Rectangle((0.45, 0.45), 0.10, 0.10, 
                           facecolor='#cce5ff', alpha=0.3, zorder=0,
                           edgecolor='#007bff', linewidth=1.5, linestyle='--')
ax.add_patch(acceptable_zone)
ax.text(0.50, 0.50, 'ACCEPTABLE', ha='center', va='center',
       fontsize=7, fontweight='bold', color='#004085', alpha=0.7)

# Perfect generalization line with enhanced glow effect
line = ax.plot([0.15, 0.65], [0.15, 0.65], 'k-', linewidth=5, alpha=0.9, 
        label='Perfect Generalization\n(Ideal: Val = Test)', zorder=2)
line[0].set_path_effects([
    path_effects.withSimplePatchShadow(offset=(0, 0), shadow_rgbFace='gold', alpha=0.6, rho=0.9),
    path_effects.withStroke(linewidth=8, foreground='gold', alpha=0.25)
])

# Plot valid trials with FIXED positioning - NO OVERLAP
valid_trials = [t for t in TRIALS_DATA.keys() if TRIALS_DATA[t]['test'] > 0]

# CUSTOM OFFSETS - carefully positioned to avoid ALL overlaps
offset_configs = {
    'Trial 2': {'x': 0.06, 'y': -0.025, 'ha': 'left', 'va': 'center'},
    'Trial 3': {'x': 0.06, 'y': 0.00, 'ha': 'left', 'va': 'center'},
    'Trial 4': {'x': 0.02, 'y': 0.05, 'ha': 'center', 'va': 'bottom'},
    'Trial 5': {'x': -0.05, 'y': 0.01, 'ha': 'right', 'va': 'center'},
    'Trial 6': {'x': -0.06, 'y': -0.025, 'ha': 'right', 'va': 'center'},
    'Trial 7': {'x': -0.02, 'y': 0.05, 'ha': 'center', 'va': 'bottom'},
    'Trial 8': {'x': 0.02, 'y': 0.055, 'ha': 'center', 'va': 'bottom'}
}

for trial in valid_trials:
    data = TRIALS_DATA[trial]
    val = data['val']
    test = data['test']
    color = data['color']
    status = data['status']
    trial_num = trial.split()[1]
    
    # Triple-layer glow effect
    ax.scatter(val, test, s=1100, color=color, alpha=0.06, zorder=2, edgecolors='none')
    ax.scatter(val, test, s=900, color=color, alpha=0.10, zorder=2, edgecolors='none')
    ax.scatter(val, test, s=750, color=color, alpha=0.15, zorder=2, edgecolors='none')
    
    # Main scatter point with premium styling
    ax.scatter(val, test, s=750, color='white', edgecolor='none', alpha=1.0, zorder=4)
    ax.scatter(val, test, s=700, color=color, edgecolor='white', linewidth=6, alpha=0.98, zorder=5)
    scatter_main = ax.scatter(val, test, s=700, color=color, edgecolor='black', linewidth=3, alpha=1.0, zorder=6)
    
    # Enhanced shadow
    scatter_main.set_path_effects([
        path_effects.withSimplePatchShadow(offset=(5, -5), shadow_rgbFace='black', alpha=0.4),
        path_effects.withSimplePatchShadow(offset=(3, -3), shadow_rgbFace='gray', alpha=0.3),
        path_effects.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace=color, alpha=0.2)
    ])
    
    # Label inside circle
    text = ax.text(val, test, trial_num, 
           ha='center', va='center', fontsize=20, fontweight='bold', 
           color='white', zorder=7)
    text.set_path_effects([
        path_effects.withStroke(linewidth=5, foreground='black', alpha=0.7),
        path_effects.withStroke(linewidth=2, foreground=color, alpha=0.4)
    ])
    
    # Status badge with FIXED positioning
    offset_cfg = offset_configs[trial]
    
    # Calculate gap
    gap = abs(val - test) / max(val, test) * 100
    
    # Badge with quality indicator
    if gap < 1:
        quality = "⭐ EXCELLENT"
        badge_color = '#28a745'
    elif gap < 15:
        quality = "✓ GOOD"
        badge_color = '#17a2b8'
    elif gap < 50:
        quality = "⚠ WARNING"
        badge_color = '#ffc107'
    else:
        quality = "✗ CRITICAL"
        badge_color = '#dc3545'
    
    badge_text = f'{trial}\n{status}\nGap: {gap:.1f}%\n{quality}'
    
    annotation = ax.annotate(badge_text, 
               xy=(val, test), 
               xytext=(val + offset_cfg['x'], test + offset_cfg['y']),
               fontsize=9.5, fontweight='bold', color=badge_color,
               bbox=dict(boxstyle='round,pad=0.65', facecolor='white', 
                        edgecolor=badge_color, linewidth=3, alpha=0.98),
               zorder=10, ha=offset_cfg['ha'], va=offset_cfg['va'],
               arrowprops=dict(arrowstyle='->', color=badge_color, lw=2.5, 
                             alpha=0.85, shrinkA=18, shrinkB=5,
                             connectionstyle='arc3,rad=0.15'),
               linespacing=1.3)

# Add Trial 1 (failed) - separate positioning
trial1_box = ax.text(0.18, 0.18, 'Trial 1\nFAILED\nNegative R²\n✗ UNUSABLE', 
       ha='center', va='center', 
       fontsize=10, fontweight='bold', color='white', zorder=11,
       bbox=dict(boxstyle='round,pad=0.75', facecolor='#dc3545', 
                edgecolor='white', linewidth=4, alpha=0.97),
       linespacing=1.3)
trial1_box.get_bbox_patch().set_path_effects([
    path_effects.withSimplePatchShadow(offset=(4, -4), shadow_rgbFace='darkred', alpha=0.6),
    path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='#721c24', alpha=0.4)
])

# Labels and title with modern premium styling
ax.set_xlabel('Validation R² Score', fontsize=20, fontweight='bold', 
             color='#2c3e50', labelpad=15)
ax.set_ylabel('Test R² Score', fontsize=20, fontweight='bold', 
             color='#2c3e50', labelpad=15)

# Multi-line title with proper spacing
title_line1 = 'Figure 4A: Validation vs Test R² Scatter Plot'
title_line2 = 'Generalization Quality Assessment (All 8 Trials)'
title_line3 = 'Reference: Boreale et al. (2024) | Benchmark R² = 0.76 (10,000 scenarios)'
title = ax.set_title(f'{title_line1}\n{title_line2}\n{title_line3}', 
            fontsize=17, fontweight='bold', pad=35, color='#1a1a1a',
            linespacing=1.6)

# Premium grid styling
ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.3, color='#7f8c8d', which='major')
ax.grid(True, alpha=0.12, linestyle=':', linewidth=0.8, color='#bdc3c7', which='minor')
ax.minorticks_on()
ax.set_axisbelow(True)

# Enhanced legend with premium styling
legend = ax.legend(fontsize=13, loc='upper left', framealpha=0.98, 
                  edgecolor='#2c3e50', fancybox=True, shadow=True,
                  borderpad=1.2, labelspacing=1.0)
legend.get_frame().set_linewidth(2.5)
legend.get_frame().set_facecolor('#f8f9fa')

# Set limits with padding
ax.set_xlim(0.14, 0.66)
ax.set_ylim(0.14, 0.66)

# Modern reference zone lines with proper spacing
zone_line_h = ax.axhline(y=0.5, color='#27ae60', linestyle=':', 
                        linewidth=3, alpha=0.7, zorder=1)
zone_line_v = ax.axvline(x=0.5, color='#27ae60', linestyle=':', 
                        linewidth=3, alpha=0.7, zorder=1)
zone_line_h.set_path_effects([path_effects.withStroke(linewidth=5, 
                              foreground='#d4edda', alpha=0.3)])
zone_line_v.set_path_effects([path_effects.withStroke(linewidth=5, 
                              foreground='#d4edda', alpha=0.3)])

# Zone label with premium badge - positioned to avoid overlap
zone_label = ax.text(0.625, 0.525, 'GOOD\nPERFORMANCE\nZONE\n(R² > 0.5)', 
       fontsize=10, fontweight='bold', color='white', ha='center', va='bottom',
       bbox=dict(boxstyle='round,pad=0.7', facecolor='#27ae60', 
                edgecolor='white', linewidth=3, alpha=0.95),
       linespacing=1.3, zorder=8)
zone_label.get_bbox_patch().set_path_effects([
    path_effects.withSimplePatchShadow(offset=(3, -3), shadow_rgbFace='#155724', alpha=0.5)
])

# Detailed info box - LEFT SIDE (no overlap)
info_box_props = dict(boxstyle='round,pad=1.0', facecolor='#fff9e6', 
                     edgecolor='#ffc107', linewidth=3.5, alpha=0.98)

info_text = " ML QUALITY CRITERIA:\n"
info_text += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
info_text += " EXCELLENT (Gap <1%):\n"
info_text += "   • Near perfect\n"
info_text += "   • Production ready\n"
info_text += "   • Trials: 5,6,7,8\n\n"
info_text += "✓ GOOD (Gap <15%):\n"
info_text += "   • Usable model\n"
info_text += "   • Minor improvement\n"
info_text += "   • Trial: 2\n\n"
info_text += "⚠ WARNING (Gap 15-50%):\n"
info_text += "   • Needs tuning\n"
info_text += "   • Use with caution\n\n"
info_text += "✗ CRITICAL (Gap >50%):\n"
info_text += "   • Severe overfitting\n"
info_text += "   • Don't use\n"
info_text += "   • Trials: 3,4"

info_textbox = ax.text(0.145, 0.655, info_text, 
       fontsize=9, fontweight='bold',
       bbox=info_box_props, verticalalignment='top', 
       color='#856404', linespacing=1.35, zorder=12,
       family='monospace')
info_textbox.get_bbox_patch().set_path_effects([
    path_effects.withSimplePatchShadow(offset=(4, -4), shadow_rgbFace='#f0ad4e', alpha=0.5)
])

# Add ML concepts box - BOTTOM LEFT (no overlap)
concepts_text = "🎓 ML CONCEPTS:\n"
concepts_text += "━━━━━━━━━━━━━━━━━\n\n"
concepts_text += "OVERFITTING:\n"
concepts_text += "Model memorizes\n"
concepts_text += "training data.\n"
concepts_text += "Val>>Test ✗\n\n"
concepts_text += "UNDERFITTING:\n"
concepts_text += "Model too simple.\n"
concepts_text += "Both R² low ✗\n\n"
concepts_text += "GOOD FIT:\n"
concepts_text += "Balanced learning.\n"
concepts_text += "Val≈Test ✓"

concepts_box = ax.text(0.145, 0.305, concepts_text,
       fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.85', facecolor='#e8f4f8',
                edgecolor='#17a2b8', linewidth=3, alpha=0.98),
       verticalalignment='top', horizontalalignment='left',
       color='#0c5460', linespacing=1.35, zorder=12,
       family='monospace')
concepts_box.get_bbox_patch().set_path_effects([
    path_effects.withSimplePatchShadow(offset=(3, -3), shadow_rgbFace='#17a2b8', alpha=0.4)
])

# Statistical summary - BOTTOM RIGHT (no overlap)
stats_text = "📈 FINAL STATISTICS:\n"
stats_text += "━━━━━━━━━━━━━━━━━━━\n\n"
stats_text += f"Total Trials: 8\n"
stats_text += f"Valid: 7 (87.5%)\n"
stats_text += f"Failed: 1 (12.5%)\n\n"
stats_text += "QUALITY BREAKDOWN:\n"
stats_text += f"• Excellent: 4 (57%)\n"
stats_text += f"• Good: 1 (14%)\n"
stats_text += f"• Critical: 2 (29%)\n\n"
stats_text += "🏆 BEST: Trial 8\n"
stats_text += "R²=0.5957, Gap=0.2%\n\n"
stats_text += "DEPLOYMENT READY:\n"
stats_text += "Trials 5,6,7,8 ✓"

stats_box = ax.text(0.645, 0.145, stats_text,
       fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.9', facecolor='#d4edda',
                edgecolor='#28a745', linewidth=3, alpha=0.98),
       verticalalignment='bottom', horizontalalignment='right',
       color='#155724', linespacing=1.35, zorder=12,
       family='monospace')
stats_box.get_bbox_patch().set_path_effects([
    path_effects.withSimplePatchShadow(offset=(3, -3), shadow_rgbFace='#28a745', alpha=0.4)
])

# Model quality legend - TOP RIGHT (no overlap)
quality_text = "MODEL QUALITY GUIDE:\n"
quality_text += "━━━━━━━━━━━━━━━━━━━\n\n"
quality_text += "⭐ EXCELLENT:\n"
quality_text += "Gap <1%\n"
quality_text += "Deploy immediately\n\n"
quality_text += "✓ GOOD:\n"
quality_text += "Gap 1-15%\n"
quality_text += "Usable with care\n\n"
quality_text += "⚠ WARNING:\n"
quality_text += "Gap 15-50%\n"
quality_text += "Needs improvement\n\n"
quality_text += "✗ CRITICAL:\n"
quality_text += "Gap >50%\n"
quality_text += "Do not use"

quality_box = ax.text(0.645, 0.655, quality_text,
       fontsize=9, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.9', facecolor='#f8d7da',
                edgecolor='#dc3545', linewidth=3, alpha=0.98),
       verticalalignment='top', horizontalalignment='right',
       color='#721c24', linespacing=1.35, zorder=12,
       family='monospace')
quality_box.get_bbox_patch().set_path_effects([
    path_effects.withSimplePatchShadow(offset=(3, -3), shadow_rgbFace='#dc3545', alpha=0.4)
])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure4a_val_vs_test_scatter.png", 
           dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: figure4a_val_vs_test_scatter.png")
print(f"     Location: {OUTPUT_DIR}")
plt.show()
plt.close()

print("\n" + "="*80)
print(" FIGURE 4A COMPLETE")
print("="*80)
