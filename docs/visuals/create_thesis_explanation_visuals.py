"""
Thesis Explanation Visuals
==========================
Ye script saari thesis ko samjhane ke liye charts banata hai.
Har chart ek concept explain karta hai - simple aur clear.

Author: Nazim
Date: February 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Output folder
OUTPUT_DIR = 'docs/visuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors - Light pastel theme (aankhen na thakein)
COLORS = {
    'blue': '#89CFF0',
    'pink': '#FFB6C1', 
    'green': '#98FB98',
    'yellow': '#FFEAA7',
    'purple': '#E6E6FA',
    'orange': '#FFDAB9',
    'dark': '#2C3E50',
    'white': '#FFFFFF',
    'gray': '#F5F5F5'
}


def chart_1_problem_solution():
    """
    Chart 1: Problem aur Solution
    - Traffic simulation slow hai
    - ML fast hai
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Problem
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('PROBLEM: Traffic Simulation Bahut Slow Hai', 
                  fontsize=16, fontweight='bold', color='#e74c3c', pad=20)
    
    # Computer icon
    computer = FancyBboxPatch((1, 4), 3, 2.5, boxstyle="round,pad=0.1",
                               facecolor=COLORS['gray'], edgecolor=COLORS['dark'], linewidth=2)
    ax1.add_patch(computer)
    ax1.text(2.5, 5.25, '💻', fontsize=30, ha='center', va='center')
    ax1.text(2.5, 3.2, 'MATSim\nSimulation', fontsize=10, ha='center', va='top', fontweight='bold')
    
    # Clock showing time
    clock = Circle((7, 5.25), 1.5, facecolor=COLORS['pink'], edgecolor=COLORS['dark'], linewidth=2)
    ax1.add_patch(clock)
    ax1.text(7, 5.25, '⏰', fontsize=35, ha='center', va='center')
    ax1.text(7, 3.2, '2-4 HOURS\nper scenario!', fontsize=11, ha='center', va='top', 
             fontweight='bold', color='#e74c3c')
    
    # Arrow
    ax1.annotate('', xy=(5.2, 5.25), xytext=(4.3, 5.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Sad planner
    ax1.text(5, 1.5, '😫 City Planner:\n"Mujhe 100 scenarios test karne hain,\n400 hours lagenge!"', 
             fontsize=10, ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.8))
    
    # Right: Solution
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('SOLUTION: ML Model Seconds Mein Answer Deta Hai', 
                  fontsize=16, fontweight='bold', color='#27ae60', pad=20)
    
    # Brain/ML icon
    ml_box = FancyBboxPatch((1, 4), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor=COLORS['blue'], edgecolor=COLORS['dark'], linewidth=2)
    ax2.add_patch(ml_box)
    ax2.text(2.5, 5.25, '🧠', fontsize=30, ha='center', va='center')
    ax2.text(2.5, 3.2, 'GNN Model\n(ML)', fontsize=10, ha='center', va='top', fontweight='bold')
    
    # Fast clock
    fast_clock = Circle((7, 5.25), 1.5, facecolor=COLORS['green'], edgecolor=COLORS['dark'], linewidth=2)
    ax2.add_patch(fast_clock)
    ax2.text(7, 5.25, '⚡', fontsize=35, ha='center', va='center')
    ax2.text(7, 3.2, '2-3 SECONDS\nper scenario!', fontsize=11, ha='center', va='top', 
             fontweight='bold', color='#27ae60')
    
    # Arrow
    ax2.annotate('', xy=(5.2, 5.25), xytext=(4.3, 5.25),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Happy planner
    ax2.text(5, 1.5, '😊 City Planner:\n"100 scenarios sirf 5 minutes mein!\nAur uncertainty bhi pata chal gayi!"', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.5))
    
    # Speedup comparison at bottom
    fig.text(0.5, 0.02, '🚀 Speedup: 5000x Faster!', fontsize=14, ha='center', 
             fontweight='bold', color=COLORS['dark'],
             bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    save_path = os.path.join(OUTPUT_DIR, '01_problem_solution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_2_what_is_uncertainty():
    """
    Chart 2: Uncertainty kya hai?
    Doctor example se samjhao
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Uncertainty Kya Hai? (Doctor Example)', fontsize=18, fontweight='bold', pad=20)
    
    # Left side: Bad (No uncertainty)
    bad_box = FancyBboxPatch((0.5, 2), 6, 6, boxstyle="round,pad=0.2",
                              facecolor='#ffcccc', edgecolor='#e74c3c', linewidth=3)
    ax.add_patch(bad_box)
    ax.text(3.5, 7.5, '❌ BURA Doctor', fontsize=14, ha='center', fontweight='bold', color='#e74c3c')
    
    # Doctor says
    ax.text(3.5, 6, '👨‍⚕️', fontsize=40, ha='center', va='center')
    ax.text(3.5, 4.5, '"Aapko cancer hai."', fontsize=12, ha='center', va='center',
            style='italic', fontweight='bold')
    ax.text(3.5, 3.5, 'Bas itna bola.\nKitna sure hai? Pata nahi!', fontsize=10, ha='center', 
            color='#666')
    ax.text(3.5, 2.5, '😰 Patient confused', fontsize=10, ha='center')
    
    # Right side: Good (With uncertainty)
    good_box = FancyBboxPatch((7.5, 2), 6, 6, boxstyle="round,pad=0.2",
                               facecolor='#ccffcc', edgecolor='#27ae60', linewidth=3)
    ax.add_patch(good_box)
    ax.text(10.5, 7.5, '✅ ACHA Doctor', fontsize=14, ha='center', fontweight='bold', color='#27ae60')
    
    # Good doctor says
    ax.text(10.5, 6, '👩‍⚕️', fontsize=40, ha='center', va='center')
    ax.text(10.5, 4.5, '"70% chance cancer hai,\n30% chance nahi hai."', fontsize=12, 
            ha='center', va='center', style='italic', fontweight='bold')
    ax.text(10.5, 3.3, 'Confidence level bataya!\nAb patient samajh sakta hai.', fontsize=10, 
            ha='center', color='#666')
    ax.text(10.5, 2.5, '😌 Patient informed', fontsize=10, ha='center')
    
    # Bottom explanation
    ax.text(7, 1, '🎯 Humara ML Model bhi aisa hai - sirf answer nahi deta,\n'
                  'ye bhi batata hai ki "mujhe apne answer pe kitna bharosa hai"',
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.8))
    
    save_path = os.path.join(OUTPUT_DIR, '02_what_is_uncertainty.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_3_paris_network():
    """
    Chart 3: Paris road network ka data
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Humara Data: Paris Road Network', fontsize=18, fontweight='bold', pad=20)
    
    # Paris map (simplified)
    paris_box = FancyBboxPatch((0.5, 3), 5, 5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['blue'], edgecolor=COLORS['dark'], linewidth=2, alpha=0.3)
    ax.add_patch(paris_box)
    
    # Roads as lines
    np.random.seed(42)
    for _ in range(30):
        x1, y1 = np.random.uniform(0.8, 5.2), np.random.uniform(3.3, 7.7)
        x2, y2 = x1 + np.random.uniform(-1, 1), y1 + np.random.uniform(-1, 1)
        ax.plot([x1, x2], [y1, y2], color=COLORS['dark'], linewidth=1.5, alpha=0.6)
    
    # Nodes
    for _ in range(40):
        x, y = np.random.uniform(0.8, 5.2), np.random.uniform(3.3, 7.7)
        ax.scatter(x, y, color=COLORS['pink'], s=30, edgecolor=COLORS['dark'], zorder=5)
    
    ax.text(3, 2.5, '🗼 Paris, France', fontsize=14, ha='center', fontweight='bold')
    
    # Stats boxes on right
    stats = [
        ('🛣️ Roads (Links)', '31,635', COLORS['blue']),
        ('📊 Scenarios', '500 total', COLORS['green']),
        ('🧪 Test Set', '100 scenarios', COLORS['pink']),
        ('📈 Total Predictions', '3.16 Million', COLORS['yellow']),
    ]
    
    for i, (label, value, color) in enumerate(stats):
        y_pos = 7.5 - i * 1.5
        stat_box = FancyBboxPatch((6.5, y_pos - 0.5), 5, 1.2, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor=COLORS['dark'], linewidth=2)
        ax.add_patch(stat_box)
        ax.text(7, y_pos, label, fontsize=11, va='center', fontweight='bold')
        ax.text(11, y_pos, value, fontsize=12, va='center', ha='right', fontweight='bold')
    
    # Input features
    ax.text(9, 2, '5 Input Features per Road:', fontsize=12, ha='center', fontweight='bold')
    features = ['1. Base Traffic Volume', '2. Road Capacity', '3. Capacity Reduction (Policy)', 
                '4. Speed Limit', '5. Road Length']
    for i, f in enumerate(features):
        ax.text(9, 1.5 - i*0.4, f, fontsize=9, ha='center')
    
    save_path = os.path.join(OUTPUT_DIR, '03_paris_data.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_4_model_architecture():
    """
    Chart 4: Model architecture - simple flow
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Model Architecture: PointNetTransfGAT', fontsize=18, fontweight='bold', pad=20)
    
    # Boxes for each component
    components = [
        (1, 'INPUT\n(5 features)', COLORS['gray'], '📊'),
        (3.5, 'PointNet\nEncoder', COLORS['blue'], '🔷'),
        (6.5, 'Transformer\n(4-head)', COLORS['purple'], '🔮'),
        (9.5, 'GAT\nLayers', COLORS['green'], '🕸️'),
        (12, 'OUTPUT\nΔVolume', COLORS['pink'], '📈'),
    ]
    
    for x, label, color, emoji in components:
        box = FancyBboxPatch((x-0.8, 3), 1.8, 2.5, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=COLORS['dark'], linewidth=2)
        ax.add_patch(box)
        ax.text(x+0.1, 4.8, emoji, fontsize=20, ha='center', va='center')
        ax.text(x+0.1, 3.5, label, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Arrows between components
    for i in range(len(components)-1):
        x1 = components[i][0] + 1
        x2 = components[i+1][0] - 0.8
        ax.annotate('', xy=(x2, 4.25), xytext=(x1, 4.25),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Explanations below
    explanations = [
        (1.1, '5 numbers\nper road'),
        (3.6, 'Features\nextract karo'),
        (6.6, 'Global\ncontext'),
        (9.6, 'Road network\nstructure'),
        (12.1, 'Traffic\nchange'),
    ]
    
    for x, text in explanations:
        ax.text(x, 2.3, text, fontsize=8, ha='center', va='top', color='#666')
    
    # MC Dropout explanation
    ax.text(7, 0.8, '🎲 MC Dropout: Inference time pe bhi dropout ON rakhte hain\n'
                    '→ Har baar thoda alag answer aata hai\n'
                    '→ Answers ke spread se pata chalta hai model kitna sure hai',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.7))
    
    save_path = os.path.join(OUTPUT_DIR, '04_model_architecture.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_5_mc_dropout_explained():
    """
    Chart 5: MC Dropout step by step
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('MC Dropout Kaise Kaam Karta Hai?', fontsize=18, fontweight='bold', pad=20)
    
    # Step 1: Input
    ax.text(1, 8.5, 'Step 1:', fontsize=12, fontweight='bold', color=COLORS['dark'])
    ax.text(1, 7.8, 'Same input do model ko', fontsize=10)
    input_box = FancyBboxPatch((0.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                                facecolor=COLORS['gray'], edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 7, 'Road Data', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Step 2: Multiple runs
    ax.text(4, 8.5, 'Step 2:', fontsize=12, fontweight='bold', color=COLORS['dark'])
    ax.text(4, 7.8, '30 baar run karo\n(dropout ON)', fontsize=10)
    
    for i in range(5):
        y = 7 - i * 0.5
        run_box = FancyBboxPatch((3.5, y-0.2), 1.5, 0.4, boxstyle="round,pad=0.05",
                                  facecolor=COLORS['blue'], edgecolor=COLORS['dark'], linewidth=1)
        ax.add_patch(run_box)
        ax.text(4.25, y, f'Run {i+1}', fontsize=8, ha='center', va='center')
    ax.text(4.25, 4.3, '...', fontsize=12, ha='center')
    ax.text(4.25, 3.8, 'Run 30', fontsize=8, ha='center')
    
    # Arrow to step 2
    ax.annotate('', xy=(3.3, 7), xytext=(2.7, 7),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Step 3: Different outputs
    ax.text(7, 8.5, 'Step 3:', fontsize=12, fontweight='bold', color=COLORS['dark'])
    ax.text(7, 7.8, 'Har baar alag answer!', fontsize=10)
    
    outputs = [102, 98, 105, 95, 108, 101, 99, 103]
    for i, val in enumerate(outputs):
        y = 7 - i * 0.5
        out_box = FancyBboxPatch((6.5, y-0.2), 1.5, 0.4, boxstyle="round,pad=0.05",
                                  facecolor=COLORS['pink'], edgecolor=COLORS['dark'], linewidth=1)
        ax.add_patch(out_box)
        ax.text(7.25, y, f'{val} veh/h', fontsize=8, ha='center', va='center')
    
    # Arrow to step 3
    ax.annotate('', xy=(6.3, 6), xytext=(5.2, 6),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Step 4: Statistics
    ax.text(10, 8.5, 'Step 4:', fontsize=12, fontweight='bold', color=COLORS['dark'])
    ax.text(10, 7.8, 'Statistics nikalo', fontsize=10)
    
    mean_box = FancyBboxPatch((9.5, 6), 2.5, 1.2, boxstyle="round,pad=0.1",
                               facecolor=COLORS['green'], edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(mean_box)
    ax.text(10.75, 6.8, 'Mean = 101', fontsize=11, ha='center', fontweight='bold')
    ax.text(10.75, 6.3, '(Prediction)', fontsize=9, ha='center', color='#666')
    
    std_box = FancyBboxPatch((9.5, 4.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                              facecolor=COLORS['yellow'], edgecolor=COLORS['dark'], linewidth=2)
    ax.add_patch(std_box)
    ax.text(10.75, 5.3, 'Std = 4.2', fontsize=11, ha='center', fontweight='bold')
    ax.text(10.75, 4.8, '(Uncertainty)', fontsize=9, ha='center', color='#666')
    
    # Arrow to step 4
    ax.annotate('', xy=(9.3, 6), xytext=(8.2, 6),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Bottom explanation
    ax.text(7, 2, '💡 Agar model sure hai → Har baar same answer → Low Std (Low Uncertainty)\n'
                  '💡 Agar model confused hai → Har baar alag answer → High Std (High Uncertainty)',
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['purple'], alpha=0.3))
    
    ax.text(7, 0.5, '🎯 Simple idea: "Answers ka spread = Confidence level"',
            fontsize=12, ha='center', fontweight='bold')
    
    save_path = os.path.join(OUTPUT_DIR, '05_mc_dropout_explained.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_6_training_results():
    """
    Chart 6: 8 trials ke results
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data
    trials = list(range(1, 9))
    r2_scores = [0.9277, 0.9253, 0.9313, 0.9328, 0.9156, 0.9228, 0.9345, 0.9374]
    mae_scores = [98.79, 100.79, 96.95, 94.45, 115.65, 103.47, 93.56, 91.55]
    
    # Bar colors - winner is green
    colors = [COLORS['blue']] * 7 + [COLORS['green']]
    
    bars = ax.bar(trials, r2_scores, color=colors, edgecolor=COLORS['dark'], linewidth=2)
    
    # Highlight winner
    bars[7].set_hatch('///')
    
    ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Training Results: 8 Trials (Alag Hyperparameters)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0.90, 0.95)
    ax.set_xticks(trials)
    
    # Add value labels
    for i, (trial, r2) in enumerate(zip(trials, r2_scores)):
        color = '#27ae60' if i == 7 else COLORS['dark']
        weight = 'bold' if i == 7 else 'normal'
        ax.text(trial, r2 + 0.002, f'{r2:.4f}', ha='center', fontsize=10, 
                fontweight=weight, color=color)
    
    # Winner annotation
    ax.annotate('🏆 WINNER!\nR² = 0.9374\nMAE = 91.55', xy=(8, 0.9374), xytext=(6.5, 0.944),
                fontsize=11, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
                bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.5))
    
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    ax.text(1, 0.903, '📌 Trial 8 settings: batch=32, dropout=0.15, lr=0.001, split=80/10/10',
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '06_training_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_7_uq_comparison():
    """
    Chart 7: UQ methods comparison
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = ['MC Dropout\n(Winner! 🏆)', 'Multi-Model\nEnsemble', 'Ensemble\nVariance']
    spearman = [0.160, 0.117, 0.103]
    colors = [COLORS['green'], COLORS['blue'], COLORS['pink']]
    
    bars = ax.barh(methods, spearman, color=colors, edgecolor=COLORS['dark'], linewidth=2, height=0.6)
    
    ax.set_xlabel('Spearman Correlation (ρ)\n(Higher = Uncertainty better predicts error)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Kaun Sa UQ Method Best Hai?', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 0.2)
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Value labels
    for bar, val in zip(bars, spearman):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'ρ = {val:.3f}',
                va='center', fontsize=12, fontweight='bold')
    
    # Explanation
    ax.text(0.1, -0.8, '❓ Spearman ρ kya hai?\n'
                       '→ Ye batata hai ki jab model "unsure" bolta hai, toh kya waqai mein galat hota hai?\n'
                       '→ Higher ρ = Model ki uncertainty zyada reliable hai',
            fontsize=10, ha='center', transform=ax.transData,
            bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.7))
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '07_uq_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_8_calibration_fix():
    """
    Chart 8: Temperature scaling fix
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before
    ax1 = axes[0]
    ax1.set_title('PEHLE: Model Jhooth Bolta Tha 😬', fontsize=14, fontweight='bold', color='#e74c3c')
    
    # Pie chart - before
    labels = ['Sahi Predictions', 'Galat Predictions']
    sizes_before = [33, 67]
    explode = (0.05, 0)
    colors_pie = [COLORS['green'], COLORS['pink']]
    
    ax1.pie(sizes_before, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.0f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.text(0, -1.4, 'Model kehta tha:\n"68% predictions sahi hongi"', fontsize=11, ha='center')
    ax1.text(0, -1.9, '❌ Actually sirf 33% sahi thi!', fontsize=12, ha='center', 
             color='#e74c3c', fontweight='bold')
    
    # After
    ax2 = axes[1]
    ax2.set_title('BAAD MEIN: Model Sach Bolta Hai 😊', fontsize=14, fontweight='bold', color='#27ae60')
    
    sizes_after = [68, 32]
    ax2.pie(sizes_after, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.0f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.text(0, -1.4, 'Model kehta hai:\n"68% predictions sahi hongi"', fontsize=11, ha='center')
    ax2.text(0, -1.9, '✅ Actually 68% sahi hain!', fontsize=12, ha='center', 
             color='#27ae60', fontweight='bold')
    
    # Overall explanation
    fig.text(0.5, 0.02, '🔧 Fix: Temperature Scaling (T = 2.92) → ECE: 0.35 → 0.03 (90% better!)',
             fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    save_path = os.path.join(OUTPUT_DIR, '08_calibration_fix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_9_practical_use():
    """
    Chart 9: Practical use case
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Real-World Use: Flagging Unreliable Predictions', fontsize=18, fontweight='bold', pad=20)
    
    # Scenario
    ax.text(7, 9, '🎯 Scenario: City planner 100 predictions check kar raha hai', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Left: Without UQ
    left_box = FancyBboxPatch((0.5, 2), 6, 5.5, boxstyle="round,pad=0.2",
                               facecolor='#ffeeee', edgecolor='#e74c3c', linewidth=3)
    ax.add_patch(left_box)
    ax.text(3.5, 7, '❌ Bina Uncertainty Ke', fontsize=14, ha='center', fontweight='bold', color='#e74c3c')
    
    ax.text(3.5, 6, '100 predictions', fontsize=11, ha='center')
    ax.text(3.5, 5.3, '↓', fontsize=20, ha='center')
    ax.text(3.5, 4.5, 'Sab same dikhti hain\nKaunsi galat hai? 🤷', fontsize=10, ha='center')
    ax.text(3.5, 3.5, '↓', fontsize=20, ha='center')
    ax.text(3.5, 2.8, '😰 Risk: Galat decision\nle sakta hai', fontsize=10, ha='center', color='#e74c3c')
    
    # Right: With UQ
    right_box = FancyBboxPatch((7.5, 2), 6, 5.5, boxstyle="round,pad=0.2",
                                facecolor='#eeffee', edgecolor='#27ae60', linewidth=3)
    ax.add_patch(right_box)
    ax.text(10.5, 7, '✅ Uncertainty Ke Saath', fontsize=14, ha='center', fontweight='bold', color='#27ae60')
    
    ax.text(10.5, 6, '100 predictions', fontsize=11, ha='center')
    ax.text(10.5, 5.3, '↓', fontsize=20, ha='center')
    ax.text(10.5, 4.5, '90 confident (✅ trust karo)\n10 uncertain (⚠️ check karo)', fontsize=10, ha='center')
    ax.text(10.5, 3.5, '↓', fontsize=20, ha='center')
    ax.text(10.5, 2.8, '😊 18% less errors!\nSafe decisions', fontsize=10, ha='center', color='#27ae60')
    
    # Result box
    ax.text(7, 1, '📊 Result: Sirf 10% predictions manually check karke, 18% errors bach gaye!',
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['yellow'], alpha=0.8))
    
    save_path = os.path.join(OUTPUT_DIR, '09_practical_use.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def chart_10_summary():
    """
    Chart 10: Final summary
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('🎓 Thesis Summary: Kya Seekha Humne?', fontsize=20, fontweight='bold', pad=20)
    
    # Key findings
    findings = [
        ('1', '🚀 GNN Model Works!', 'R² = 0.9374\n93.74% accuracy', COLORS['green']),
        ('2', '🏆 MC Dropout Best Hai', 'Spearman ρ = 0.160\n55% better than ensemble', COLORS['blue']),
        ('3', '❌ Same Architecture Ensemble Fail', '8 same models\n= No diversity = No benefit', COLORS['pink']),
        ('4', '✅ Calibration Fixed', 'ECE: 0.35 → 0.03\n90% improvement', COLORS['yellow']),
        ('5', '📊 Practical Value', '10% flag → 18% error reduction', COLORS['purple']),
    ]
    
    for i, (num, title, detail, color) in enumerate(findings):
        y = 10 - i * 1.8
        
        # Number circle
        circle = Circle((1, y), 0.5, facecolor=color, edgecolor=COLORS['dark'], linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, num, fontsize=14, ha='center', va='center', fontweight='bold')
        
        # Content box
        box = FancyBboxPatch((2, y-0.6), 11, 1.2, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=COLORS['dark'], linewidth=2, alpha=0.4)
        ax.add_patch(box)
        ax.text(2.5, y+0.2, title, fontsize=12, fontweight='bold', va='center')
        ax.text(2.5, y-0.3, detail, fontsize=10, va='center', color='#444')
    
    # Bottom message
    ax.text(7, 0.8, '🎯 Conclusion: MC Dropout + Temperature Scaling = \n'
                    'Fast, Accurate, and Trustworthy Traffic Predictions!',
            fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.5))
    
    save_path = os.path.join(OUTPUT_DIR, '10_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("="*60)
    print("Creating Thesis Explanation Visuals")
    print("="*60)
    
    chart_1_problem_solution()
    chart_2_what_is_uncertainty()
    chart_3_paris_network()
    chart_4_model_architecture()
    chart_5_mc_dropout_explained()
    chart_6_training_results()
    chart_7_uq_comparison()
    chart_8_calibration_fix()
    chart_9_practical_use()
    chart_10_summary()
    
    print("\n" + "="*60)
    print("Done! 10 charts created in docs/visuals/")
    print("="*60)
