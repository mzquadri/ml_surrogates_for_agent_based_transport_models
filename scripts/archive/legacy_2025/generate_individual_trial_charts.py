"""
COMPLETE TRIAL ANALYSIS CHARTS GENERATOR
Creates detailed individual charts for ALL 8 trials

Generates:
- Figure 5-12: Individual trial detailed analyses (Trials 2-7)
- Each figure includes: Performance metrics, hyperparameters, analysis

Usage in Google Colab:
!python /content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main/generate_individual_trial_charts.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import os

# Configuration
BASE_PATH = "/content/drive/MyDrive/Zamin-thesis/ml_surrogates_for_agent_based_transport_models-main"
OUTPUT_DIR = f"{BASE_PATH}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Complete trials database
TRIALS_DATABASE = {
    'Trial 2': {
        'name': 'Trial 2: First Working Configuration',
        'hyperparameters': {
            'Dropout Rate': 0.3,
            'Batch Size': 16,
            'Learning Rate': '5e-4',
            'Weighted Loss': 'No',
            'Optimizer': 'AdamW',
            'Architecture': 'PointNetTransfGAT'
        },
        'results': {
            'Validation R2': 0.5841,
            'Test R2': 0.5117,
            'Generalization Gap': 12.4,
            'Benchmark Achievement': 67.3,
            'Training Status': 'SUCCESS'
        },
        'analysis': {
            'status': '[OK] WORKING MODEL',
            'key_points': [
                'First successful working model after Trial 1 failure',
                'Batch size 16 shows promise but not optimal',
                'Dropout 0.3 provides adequate regularization',
                'Good generalization with 12.4% gap (acceptable)',
                'Achieves 67.3% of benchmark performance'
            ],
            'strengths': [
                'Successfully converged',
                'Reasonable generalization gap',
                'Stable training curve',
                'No catastrophic overfitting'
            ],
            'weaknesses': [
                'Lower than optimal performance',
                'Batch size 16 suboptimal vs BS=8',
                '12.4% gap indicates room for improvement',
                'Can be improved with hyperparameter tuning'
            ],
            'recommendation': 'Good baseline but Trial 8 (BS=8) performs better'
        },
        'color': '#a29bfe',
        'figure_num': 5
    },
    'Trial 3': {
        'name': 'Trial 3: Overfitting Case Study (Weighted Loss)',
        'hyperparameters': {
            'Dropout Rate': 0.0,
            'Batch Size': 16,
            'Learning Rate': '5e-4',
            'Weighted Loss': 'Yes',
            'Optimizer': 'AdamW',
            'Architecture': 'PointNetTransfGAT'
        },
        'results': {
            'Validation R2': 0.5953,
            'Test R2': 0.2246,
            'Generalization Gap': 62.3,
            'Benchmark Achievement': 29.6,
            'Training Status': 'OVERFITTING'
        },
        'analysis': {
            'status': '[WARNING] SEVERE OVERFITTING',
            'key_points': [
                'Zero dropout causes severe overfitting',
                '62.3% generalization gap (CRITICAL)',
                'Model memorizes training data',
                'Fails on unseen test set',
                'Weighted loss cannot compensate for lack of regularization'
            ],
            'strengths': [
                'High validation R2 (0.5953)',
                'Successfully completed training',
                'Demonstrates importance of dropout'
            ],
            'weaknesses': [
                'Catastrophic test performance (R2=0.2246)',
                '62.3% generalization gap (highest among all trials)',
                'Model not usable for deployment',
                'Overfits to training distribution'
            ],
            'recommendation': 'DO NOT USE - Demonstrates critical need for dropout regularization'
        },
        'color': '#6c5ce7',
        'figure_num': 6
    },
    'Trial 4': {
        'name': 'Trial 4: Overfitting Validation (Weighted Loss Repeat)',
        'hyperparameters': {
            'Dropout Rate': 0.0,
            'Batch Size': 16,
            'Learning Rate': '5e-4',
            'Weighted Loss': 'Yes',
            'Optimizer': 'AdamW',
            'Architecture': 'PointNetTransfGAT'
        },
        'results': {
            'Validation R2': 0.6097,
            'Test R2': 0.2426,
            'Generalization Gap': 60.2,
            'Benchmark Achievement': 31.9,
            'Training Status': 'OVERFITTING'
        },
        'analysis': {
            'status': '[WARNING] SEVERE OVERFITTING',
            'key_points': [
                'Confirms Trial 3 findings - zero dropout fails',
                'Highest validation R2 (0.6097) but poor generalization',
                '60.2% gap confirms dropout necessity',
                'Weighted loss does NOT solve overfitting',
                'Validates critical importance of regularization'
            ],
            'strengths': [
                'Best validation R2 among all trials',
                'Confirms experimental hypothesis',
                'Provides clear evidence for dropout necessity'
            ],
            'weaknesses': [
                'Second-worst test R2 (0.2426)',
                '60.2% generalization gap (second-highest)',
                'Not suitable for deployment',
                'Misleading high validation score'
            ],
            'recommendation': 'DO NOT USE - Validates that high validation R2 without dropout is misleading'
        },
        'color': '#fd79a8',
        'figure_num': 7
    },
    'Trial 5': {
        'name': 'Trial 5: Baseline Model (Optimal Batch Size)',
        'hyperparameters': {
            'Dropout Rate': 0.3,
            'Batch Size': 8,
            'Learning Rate': '5e-4',
            'Weighted Loss': 'No',
            'Optimizer': 'AdamW',
            'Architecture': 'PointNetTransfGAT'
        },
        'results': {
            'Validation R2': 0.5500,
            'Test R2': 0.5553,
            'Generalization Gap': 0.96,
            'Benchmark Achievement': 73.1,
            'Training Status': 'SUCCESS'
        },
        'analysis': {
            'status': '[OK] STABLE BASELINE',
            'key_points': [
                'Smallest batch size (8) improves generalization',
                'Near-perfect generalization (gap < 1%)',
                'Test R2 EXCEEDS validation R2 (rare achievement)',
                'Establishes baseline for further optimization',
                'Batch size 8 proves optimal for this dataset'
            ],
            'strengths': [
                'Excellent generalization (0.96% gap)',
                'Test outperforms validation (robust model)',
                'Stable training process',
                '73.1% of benchmark achievement'
            ],
            'weaknesses': [
                'Slightly lower absolute R2 than Trial 8',
                'Dropout 0.3 may be too aggressive',
                'Room for capacity improvement'
            ],
            'recommendation': 'Excellent baseline - Use as reference for optimization'
        },
        'color': '#ffd93d',
        'figure_num': 8
    },
    'Trial 6': {
        'name': 'Trial 6: Learning Rate Sensitivity (Reduced)',
        'hyperparameters': {
            'Dropout Rate': 0.3,
            'Batch Size': 8,
            'Learning Rate': '3e-4',
            'Weighted Loss': 'No',
            'Optimizer': 'AdamW',
            'Architecture': 'PointNetTransfGAT'
        },
        'results': {
            'Validation R2': 0.5224,
            'Test R2': 0.5223,
            'Generalization Gap': 0.02,
            'Benchmark Achievement': 68.7,
            'Training Status': 'SUCCESS'
        },
        'analysis': {
            'status': '[OK] LR TOO CONSERVATIVE',
            'key_points': [
                'Lower LR (3e-4) trains slower',
                'Perfect generalization (0.02% gap)',
                'Both val and test R2 lower than baseline',
                'Too conservative learning rate underperforms',
                '5e-4 proves better than 3e-4'
            ],
            'strengths': [
                'Near-perfect generalization',
                'Extremely stable training',
                'Val and test R2 almost identical',
                'No overfitting concerns'
            ],
            'weaknesses': [
                'Lower absolute performance (R2=0.52)',
                'Trains too slowly',
                'May not reach optimal capacity',
                '4.4% below baseline performance'
            ],
            'recommendation': 'Good stability but 5e-4 LR (baseline) performs better'
        },
        'color': '#ff9770',
        'figure_num': 9
    },
    'Trial 7': {
        'name': 'Trial 7: Learning Rate Sensitivity (Increased)',
        'hyperparameters': {
            'Dropout Rate': 0.3,
            'Batch Size': 8,
            'Learning Rate': '6e-4',
            'Weighted Loss': 'No',
            'Optimizer': 'AdamW',
            'Architecture': 'PointNetTransfGAT'
        },
        'results': {
            'Validation R2': 0.5497,
            'Test R2': 0.5471,
            'Generalization Gap': 0.47,
            'Benchmark Achievement': 72.0,
            'Training Status': 'SUCCESS'
        },
        'analysis': {
            'status': '[OK] LR SLIGHTLY HIGH',
            'key_points': [
                'Higher LR (6e-4) trains faster',
                'Excellent generalization (0.47% gap)',
                'Performance slightly below baseline',
                '6e-4 too aggressive compared to 5e-4',
                'Validates 5e-4 as sweet spot'
            ],
            'strengths': [
                'Excellent generalization',
                'Faster convergence',
                'Stable final performance',
                '72% of benchmark achievement'
            ],
            'weaknesses': [
                'Slightly worse than baseline (5e-4)',
                'May overshoot optimal weights',
                'Less fine-tuned convergence',
                '1% below baseline performance'
            ],
            'recommendation': 'Good but 5e-4 LR (baseline) is optimal'
        },
        'color': '#ff7f50',
        'figure_num': 10
    }
}

BENCHMARK_R2 = 0.76

def generate_trial_chart(trial_key, trial_data):
    """Generate detailed chart for a single trial"""
    
    print(f"\\n{'='*80}")
    print(f" GENERATING FIGURE {trial_data['figure_num']}: {trial_data['name'].upper()}")
    print(f"{'='*80}")
    
    # Create figure
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('white')
    
    # Main title
    fig.suptitle(f"Figure {trial_data['figure_num']}: {trial_data['name']}\\nComplete Performance Analysis and Configuration Details", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create 4 panels
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, left=0.07, right=0.95, top=0.90, bottom=0.06)
    
    # PANEL A: R2 Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = ['Validation R2', 'Test R2', 'Benchmark\\n(Boreale 2024)']
    values = [trial_data['results']['Validation R2'], 
              trial_data['results']['Test R2'], 
              BENCHMARK_R2]
    colors = [trial_data['color'], trial_data['color'], '#FFD700']
    alphas = [0.7, 0.9, 0.85]
    
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', 
                   linewidth=2.5, width=0.6)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
        bar.set_path_effects([path_effects.withSimplePatchShadow(
            offset=(2, -2), shadow_rgbFace='gray', alpha=0.4)])
    
    # Add values
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.95, edgecolor='black', linewidth=2))
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=2, zorder=0)
    ax1.set_ylabel('R2 Score', fontsize=14, fontweight='bold')
    ax1.set_title(f'(A) Performance Comparison\\n{trial_key} vs Benchmark', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 0.85)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add status box
    gap = trial_data['results']['Generalization Gap']
    bench_ach = trial_data['results']['Benchmark Achievement']
    status_text = f"{trial_data['analysis']['status']}\\n\\n"
    status_text += f"Gap: {gap:.1f}%\\n"
    status_text += f"Benchmark: {bench_ach:.1f}%"
    
    ax1.text(0.98, 0.95, status_text, transform=ax1.transAxes, 
             fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                      edgecolor='black', linewidth=2, alpha=0.9))
    
    # PANEL B: Hyperparameters Configuration
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    ax2.text(0.5, 0.98, '(B) Complete Hyperparameters Configuration', 
             ha='center', va='top', fontsize=14, fontweight='bold', 
             transform=ax2.transAxes)
    
    config_text = f"{trial_key.upper()} CONFIGURATION\\n"
    config_text += "="*50 + "\\n\\n"
    config_text += f"Architecture: {trial_data['hyperparameters']['Architecture']}\\n"
    config_text += "Model Parameters: 1.55 Million\\n\\n"
    config_text += "Training Hyperparameters:\\n"
    config_text += f"  Dropout Rate: {trial_data['hyperparameters']['Dropout Rate']}\\n"
    config_text += f"  Batch Size: {trial_data['hyperparameters']['Batch Size']}\\n"
    config_text += f"  Learning Rate: {trial_data['hyperparameters']['Learning Rate']}\\n"
    config_text += f"  Weighted Loss: {trial_data['hyperparameters']['Weighted Loss']}\\n"
    config_text += f"  Optimizer: {trial_data['hyperparameters']['Optimizer']}\\n\\n"
    config_text += "Results:\\n"
    config_text += f"  Validation R2: {trial_data['results']['Validation R2']:.4f}\\n"
    config_text += f"  Test R2: {trial_data['results']['Test R2']:.4f}\\n"
    config_text += f"  Generalization Gap: {trial_data['results']['Generalization Gap']:.2f}%\\n"
    config_text += f"  Benchmark Achievement: {trial_data['results']['Benchmark Achievement']:.1f}%\\n"
    config_text += f"  Status: {trial_data['results']['Training Status']}"
    
    ax2.text(0.5, 0.45, config_text, ha='center', va='center', 
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round,pad=1.0', facecolor='#F5F5F5', 
                      edgecolor='black', linewidth=2, alpha=0.9),
             transform=ax2.transAxes, linespacing=1.5)
    
    # PANEL C: Key Analysis Points
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    ax3.text(0.5, 0.98, '(C) Key Analysis Points', 
             ha='center', va='top', fontsize=14, fontweight='bold', 
             transform=ax3.transAxes)
    
    analysis_text = "KEY FINDINGS:\\n"
    analysis_text += "-"*50 + "\\n\\n"
    for i, point in enumerate(trial_data['analysis']['key_points'], 1):
        analysis_text += f"{i}. {point}\\n\\n"
    
    analysis_text += "\\nSTRENGTHS:\\n"
    for strength in trial_data['analysis']['strengths']:
        analysis_text += f"  [+] {strength}\\n"
    
    analysis_text += "\\nWEAKNESSES:\\n"
    for weakness in trial_data['analysis']['weaknesses']:
        analysis_text += f"  [-] {weakness}\\n"
    
    ax3.text(0.5, 0.45, analysis_text, ha='center', va='center', 
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round,pad=1.0', facecolor='#FFFAF0', 
                      edgecolor='black', linewidth=2, alpha=0.9),
             transform=ax3.transAxes, linespacing=1.6)
    
    # PANEL D: Recommendation and Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    ax4.text(0.5, 0.98, '(D) Recommendation and Context', 
             ha='center', va='top', fontsize=14, fontweight='bold', 
             transform=ax4.transAxes)
    
    rec_text = "RECOMMENDATION:\\n"
    rec_text += "="*50 + "\\n\\n"
    rec_text += trial_data['analysis']['recommendation'] + "\\n\\n"
    rec_text += "-"*50 + "\\n\\n"
    rec_text += "CONTEXT IN COMPLETE STUDY:\\n\\n"
    rec_text += f"This trial is part of a comprehensive\\n"
    rec_text += f"hyperparameter exploration study with\\n"
    rec_text += f"8 total trials.\\n\\n"
    rec_text += f"Dataset: 1,000 training scenarios\\n"
    rec_text += f"Architecture: PointNetTransfGAT (1.55M params)\\n"
    rec_text += f"Reference: Boreale et al. (2024)\\n"
    rec_text += f"Benchmark: R2 = 0.76 (10,000 scenarios)\\n\\n"
    rec_text += f"TRIAL RANKINGS:\\n"
    rec_text += f"  1st: Trial 8 (R2=0.5957) - BEST\\n"
    rec_text += f"  2nd: Trial 5 (R2=0.5553) - Baseline\\n"
    rec_text += f"  3rd: Trial 7 (R2=0.5471)\\n"
    rec_text += f"  4th: Trial 6 (R2=0.5223)\\n"
    rec_text += f"  5th: Trial 2 (R2=0.5117)\\n"
    rec_text += f"  6th: Trial 4 (R2=0.2426) - Overfit\\n"
    rec_text += f"  7th: Trial 3 (R2=0.2246) - Overfit\\n"
    rec_text += f"  8th: Trial 1 (R2=-0.0022) - Failed"
    
    ax4.text(0.5, 0.45, rec_text, ha='center', va='center', 
             fontsize=9.5, family='monospace',
             bbox=dict(boxstyle='round,pad=1.0', facecolor='#F0FFF0', 
                      edgecolor='darkgreen', linewidth=2, alpha=0.9),
             transform=ax4.transAxes, linespacing=1.5)
    
    # Footer
    footer_text = f"{trial_key} Documentation | Reference: Boreale et al. (2024) | Dataset: 1,000 scenarios"
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10, 
             style='italic', color='gray')
    
    # Save
    output_file = f"figure{trial_data['figure_num']}_{trial_key.lower().replace(' ', '_')}_detailed.png"
    plt.savefig(f"{OUTPUT_DIR}/{output_file}", 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {output_file}")
    print(f"     Location: {OUTPUT_DIR}")
    plt.close()
    
    print(f"{'='*80}\\n")

# Generate all trial charts
print("="*80)
print(" GENERATING FIGURES 5-10: TRIALS 2-7")
print("="*80)

for trial_key in ['Trial 2', 'Trial 3', 'Trial 4', 'Trial 5', 'Trial 6', 'Trial 7']:
    generate_trial_chart(trial_key, TRIALS_DATABASE[trial_key])

print("\\n" + "="*80)
print(" FIGURES 5-10 COMPLETE")
print("="*80)
print("\nGenerated:")
print("  - Figure 5: Trial 2")
print("  - Figure 6: Trial 3 (Overfitting)")
print("  - Figure 7: Trial 4 (Overfitting)")
print("  - Figure 8: Trial 5 (Baseline)")
print("  - Figure 9: Trial 6 (LR Low)")
print("  - Figure 10: Trial 7 (LR High)")
print("="*80)
