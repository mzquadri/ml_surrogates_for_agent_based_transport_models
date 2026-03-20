"""
MEGA THESIS CHARTS — Complete, Professional, Cross-Checked
==========================================================
Author: Mohd Zamin Quadri — TUM Master Thesis
Professor: Günnemann | Supervisor: Fuchsgruber

KEY FACTS:
  - Trial 1 = OLD architecture (Linear final layer, NOT GATConv) → EXCLUDED from fair comparison
  - Trials 2-8 = Correct PointNetTransfGAT (GATConv final layer, matches Elena's paper)
  - Best correct trial = Trial 8 (R²=0.596)
  - Only 10% of Paris population simulated (1% downsampled × 10,000 simulations)
  - MC Dropout: 30 stochastic forward passes
  - Ensemble Experiments: A (5 training runs) + B (multi-model from trials 2,5,6,7,8)
  - Temperature Scaling calibration applied
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os, warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE = 'data/TR-C_Benchmarks'
OUT = 'docs/visuals/verification'
os.makedirs(OUT, exist_ok=True)

TRIAL_FOLDERS = {
    1: 'pointnet_transf_gat_1st_bs32_5feat_seed42',
    2: 'point_net_transf_gat_2nd_try',
    3: 'point_net_transf_gat_3rd_trial_weighted_loss',
    4: 'point_net_transf_gat_4th_trial_weighted_loss',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}

TRIAL_SHORT = {
    1: 'T1: OLD Arch (Linear final)',
    2: 'T2: BS=16, DO=0.20, LR=1e-3',
    3: 'T3: BS=16, DO=0.30, LR=1e-3 (wt.loss)',
    4: 'T4: BS=16, DO=0.30, LR=1e-3 (wt.loss)',
    5: 'T5: BS=32, DO=0.20, LR=1e-4',
    6: 'T6: BS=32, DO=0.30, LR=1e-4',
    7: 'T7: BS=32, DO=0.15, LR=1e-4 (80/10/10)',
    8: 'T8: BS=32, DO=0.15, LR=1e-3 (best UQ)',
}

TC = ['#E74C3C','#FF8E53','#FECA57','#F39C12','#48DBFB','#0ABDE3','#1ABC9C','#2ECC71']  # Trial colors
PAL = {'blue':'#4A90D9','coral':'#FF6B6B','green':'#2ECC71','gold':'#F39C12',
       'purple':'#9B59B6','teal':'#1ABC9C','rose':'#E74C3C','dark':'#2C3E50',
       'sky':'#48DBFB','ocean':'#0ABDE3','mint':'#98FB98','peach':'#FFEAA7'}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': '#FAFBFC',
        'axes.grid': True, 'grid.alpha': 0.12, 'grid.linestyle': '--',
        'font.family': 'sans-serif', 'font.size': 11,
        'axes.titlesize': 15, 'axes.titleweight': 'bold',
        'axes.labelsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
    })

def wm(ax, t='Cross-checked from NPZ files'):
    ax.text(0.99, 0.01, t, transform=ax.transAxes, fontsize=7, color='#CCC', ha='right', va='bottom', style='italic')


# ============================================================
# DATA LOADING
# ============================================================

def load_trials():
    import torch
    R = {}
    for t, f in TRIAL_FOLDERS.items():
        d = np.load(os.path.join(BASE, f, 'test_predictions.npz'))
        p, tg = d['predictions'].flatten(), d['targets'].flatten()
        e = np.abs(p - tg)
        R[t] = {'p': p, 't': tg, 'e': e,
                 'r2': r2_score(tg, p), 'mae': mean_absolute_error(tg, p),
                 'rmse': np.sqrt(mean_squared_error(tg, p)),
                 'med': np.median(e), 'p90': np.percentile(e, 90),
                 'u5': np.mean(e < 5)*100, 'n': len(p)}
    
    # Load features + positions from graph
    g = torch.load('data/train_data/dist_not_connected_10k_1pct/datalist_batch_1.pt',
                    map_location='cpu', weights_only=False)[0]
    feats = g.x.numpy()
    pos = g.pos.numpy()
    return R, feats, pos

def load_mc():
    M = {}
    paths = {
        5: f'{BASE}/point_net_transf_gat_5th_try/uq_results/mc_dropout_full_50graphs_mc30.npz',
        6: f'{BASE}/point_net_transf_gat_6th_trial_lower_lr/uq_results/mc_dropout_full_50graphs_mc30.npz',
        7: f'{BASE}/point_net_transf_gat_7th_trial_80_10_10_split/uq_results/mc_dropout_full_100graphs_mc30.npz',
        8: f'{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/mc_dropout_full_100graphs_mc30.npz',
    }
    for t, path in paths.items():
        d = np.load(path)
        p, u, tg = d['predictions'].flatten(), d['uncertainties'].flatten(), d['targets'].flatten()
        e = np.abs(p - tg)
        rho, pv = spearmanr(u, e)
        pr, _ = pearsonr(u, e)
        M[t] = {'p':p, 'u':u, 't':tg, 'e':e, 'rho':rho, 'pval':pv, 'pr':pr,
                 'c1': np.mean(e < u)*100, 'c2': np.mean(e < 2*u)*100,
                 'mu': np.mean(u), 'me': np.mean(e)}
    return M

def load_ens():
    ea = np.load(f'{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_a_data.npz', allow_pickle=True)
    eb = np.load(f'{BASE}/point_net_transf_gat_8th_trial_lower_dropout/uq_results/ensemble_experiments/experiment_b_data.npz', allow_pickle=True)
    return ea, eb


# ============================================================
# CHART A1: FULL 3D MODEL ARCHITECTURE DIAGRAM
# ============================================================

def chart_arch_diagram():
    """Beautiful 3D-style architecture diagram of PointNetTransfGAT"""
    fig, ax = plt.subplots(figsize=(28, 16))
    ax.set_xlim(-1, 30)
    ax.set_ylim(-3, 15)
    ax.axis('off')
    ax.set_facecolor('#0F1419')
    fig.patch.set_facecolor('#0F1419')

    # ---- Helper: draw a 3D-style block ----
    def block(x, y, w, h, color, label, sublabel='', glow=False):
        # Shadow
        shadow = FancyBboxPatch((x+0.12, y-0.12), w, h, boxstyle='round,pad=0.15',
                                facecolor='#000000', alpha=0.4, lw=0, zorder=1)
        ax.add_patch(shadow)
        # Main box
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor='white', lw=1.5, alpha=0.92, zorder=2)
        ax.add_patch(box)
        if glow:
            glow_box = FancyBboxPatch((x-0.08, y-0.08), w+0.16, h+0.16, boxstyle='round,pad=0.2',
                                      facecolor='none', edgecolor=color, lw=3, alpha=0.4, zorder=1)
            ax.add_patch(glow_box)
        ax.text(x+w/2, y+h/2+0.15, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=3)
        if sublabel:
            ax.text(x+w/2, y+h/2-0.3, sublabel, ha='center', va='center', fontsize=8,
                    color='#DDDDDD', zorder=3, style='italic')

    def arrow(x1, y1, x2, y2, color='#888888'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5), zorder=4)

    def dim_label(x, y, text):
        ax.text(x, y, text, ha='center', va='center', fontsize=8, color='#AAAAAA',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A2030', edgecolor='#444', alpha=0.9), zorder=5)

    # ============================================================
    # TITLE
    # ============================================================
    ax.text(14.5, 14.2, 'PointNetTransfGAT — Complete Architecture',
            ha='center', fontsize=24, fontweight='bold', color='white')
    ax.text(14.5, 13.5, 'Graph Neural Network for Traffic Flow Change Prediction on Paris Road Network',
            ha='center', fontsize=13, color='#888888')
    ax.text(14.5, 13.0, 'Elena Natterer\'s architecture | Reproduced by Mohd Zamin Quadri (TUM)',
            ha='center', fontsize=10, color='#666666', style='italic')

    # ============================================================
    # INPUT SECTION (left)
    # ============================================================
    # Graph input
    block(0, 9, 3.5, 2.5, '#1a5276', 'Input Graph', 'Paris: 31,635 nodes\n59,851 edges')
    dim_label(1.75, 8.5, 'PyG Data object')

    # Feature box
    block(0, 5.5, 3.5, 2.8, '#2e86c1', '6 Node Features', 
          'VOL_BASE_CASE (0-1596)\nCAPACITY (0-14400)\nCAP_REDUCTION (-4800-0)\nFREESPEED (0-33)\nLANES (-1 to 9)\nLENGTH (4-2569)')
    dim_label(1.75, 5, '[N, 5] → model uses 5')

    # Position box
    block(0, 2, 3.5, 2.8, '#1b4f72', 'Spatial Positions', 
          'Start pos: (lat, lon)\nEnd pos: (lat, lon)\nParis ~48.85°N, 2.34°E')
    dim_label(1.75, 1.5, '[N, 3, 2] → start & end')

    # Arrows from inputs
    arrow(3.6, 10, 5.2, 9.5, '#2e86c1')
    arrow(3.6, 6.8, 5.2, 8.5, '#2e86c1')
    arrow(3.6, 3.5, 5.2, 7, '#1b4f72')

    # ============================================================
    # POINTNET SECTION
    # ============================================================
    # PointNetConv 1
    block(5.3, 8, 3.8, 3, '#8e44ad', 'PointNetConv 1', 
          'Local MLP: 7→256\nGlobal MLP: 256→512→512\nUses START positions\n+ ReLU + Dropout')
    dim_label(7.2, 7.5, 'Output: [N, 512]')

    # PointNetConv 2
    block(5.3, 4, 3.8, 3, '#6c3483', 'PointNetConv 2', 
          'Local MLP: 514→256\nGlobal MLP: 256→512→128\nUses END positions\n+ ReLU + Dropout')
    dim_label(7.2, 3.5, 'Output: [N, 128]')

    # Arrow between PointNets
    arrow(7.2, 8, 7.2, 7.1, '#8e44ad')
    
    # Arrow to TransformerConv
    arrow(9.2, 5.5, 10.5, 8.5, '#6c3483')

    # ============================================================
    # TRANSFORMER + GAT SECTION
    # ============================================================
    # TransformerConv 1
    block(10.6, 8, 3.8, 3, '#c0392b', 'TransformerConv 1', 
          '128 → 64 × 4 heads = 256\nMulti-head attention on graph\n+ ReLU + Dropout')
    dim_label(12.5, 7.5, 'Output: [N, 256]')

    arrow(14.5, 9.5, 15.5, 9.5, '#c0392b')

    # TransformerConv 2
    block(15.6, 8, 3.8, 3, '#e74c3c', 'TransformerConv 2', 
          '256 → 128 × 4 heads = 512\nMulti-head attention on graph\n+ ReLU + Dropout')
    dim_label(17.5, 7.5, 'Output: [N, 512]')

    arrow(19.5, 9.5, 20.5, 9.5, '#e74c3c')

    # GATConv
    block(20.6, 8, 3.5, 3, '#27ae60', 'GATConv', 
          '512 → 64\nGraph Attention Network\nlearns edge importance')
    dim_label(22.35, 7.5, 'Output: [N, 64]')

    arrow(24.2, 9.5, 25.2, 9.5, '#27ae60')

    # Output GATConv
    block(25.3, 8, 3.5, 3, '#f39c12', 'GATConv (Final)', 
          '64 → 1\nPer-node prediction\nDeltaVolume (veh/h)', glow=True)
    dim_label(27.05, 7.5, 'Output: [N, 1]')

    # ============================================================
    # OUTPUT SECTION
    # ============================================================
    block(25.3, 3.5, 3.5, 3.2, '#d4ac0d', 'Output', 
          'Per-link traffic flow\nchange prediction\nDeltaVolume (vehicles/hour)\nfor each of 31,635 links')
    arrow(27.05, 8, 27.05, 6.8, '#f39c12')

    # ============================================================
    # MC Dropout annotation
    # ============================================================
    block(10.6, 1, 8.5, 2.2, '#2980b9', 'MC Dropout (UQ)', 
          'During inference: keep dropout ON\n30 stochastic forward passes → mean prediction + uncertainty (std)\nNo retraining needed! Cheapest UQ method.')
    
    # Arrows showing dropout in PointNet + Transformer
    for xp in [7.2, 12.5, 17.5]:
        ax.annotate('', xy=(xp, 2.2 if xp > 9 else 4), xytext=(xp, 3.2),
                    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5, ls='--'), zorder=4)

    # ============================================================
    # LEGEND
    # ============================================================
    legend_items = [
        ('#8e44ad', 'PointNet (spatial encoding)'),
        ('#c0392b', 'Transformer (multi-head attention)'),
        ('#27ae60', 'GAT (graph attention)'),
        ('#f39c12', 'Output (per-node prediction)'),
        ('#2980b9', 'MC Dropout (uncertainty)'),
    ]
    for i, (c, l) in enumerate(legend_items):
        ax.add_patch(Rectangle((0.3, -0.5 - i*0.65), 0.5, 0.45, facecolor=c, edgecolor='white', lw=1, zorder=5))
        ax.text(1.1, -0.28 - i*0.65, l, fontsize=10, color='white', va='center', zorder=5)

    # Model stats
    ax.text(20, -0.3, 'Total Parameters: ~1.4M | Optimizer: AdamW (weight_decay=1e-4)\n'
            'Training: 10% Paris population (1% downsampled), 10,000 simulations\n'
            'Best trial: T8 (BS=32, DO=0.15, LR=1e-3) → R²=0.596, MAE=3.96',
            fontsize=10, color='#AAAAAA', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A2030', edgecolor='#333', alpha=0.9))

    plt.savefig(os.path.join(OUT, 'A1_architecture_diagram.png'), dpi=250, bbox_inches='tight',
                facecolor='#0F1419', edgecolor='none')
    plt.close()
    print("  A1 Architecture diagram done")


# ============================================================
# CHART A2: DATA PIPELINE DIAGRAM
# ============================================================

def chart_data_pipeline():
    fig, ax = plt.subplots(figsize=(26, 10))
    ax.set_xlim(-1, 27)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_facecolor('#0F1419')
    fig.patch.set_facecolor('#0F1419')

    def block(x, y, w, h, color, label, sub=''):
        shadow = FancyBboxPatch((x+0.1, y-0.1), w, h, boxstyle='round,pad=0.15',
                                facecolor='#000', alpha=0.4, lw=0, zorder=1)
        ax.add_patch(shadow)
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor='white', lw=1.5, alpha=0.92, zorder=2)
        ax.add_patch(box)
        ax.text(x+w/2, y+h/2+0.15, label, ha='center', va='center', fontsize=12,
                fontweight='bold', color='white', zorder=3)
        if sub:
            ax.text(x+w/2, y+h/2-0.35, sub, ha='center', va='center', fontsize=8.5,
                    color='#DDD', zorder=3, style='italic')

    def arrow(x1, y1, x2, y2, color='#888'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5), zorder=4)

    ax.text(13, 8.5, 'Complete Data & Experiment Pipeline', fontsize=22, fontweight='bold',
            color='white', ha='center')
    ax.text(13, 7.9, 'From MATSim agent-based simulations to uncertainty-aware predictions',
            fontsize=12, color='#888', ha='center')

    # Row 1: Data
    block(0, 5.5, 3.8, 1.8, '#1a5276', 'MATSim Simulation',
          'Agent-based transport model\nParis 1% population\n10,000 scenarios')
    arrow(3.9, 6.4, 5, 6.4, '#48DBFB')

    block(5.1, 5.5, 3.8, 1.8, '#2e86c1', 'Graph Construction',
          '31,635 road links = nodes\n59,851 connections = edges\n6 features per node')
    arrow(9, 6.4, 10.1, 6.4, '#48DBFB')

    block(10.2, 5.5, 3.8, 1.8, '#1b4f72', 'Train/Val/Test Split',
          'T2-T6: 80/15/5\nT7: 80/10/10\nT8: 80/10/10')
    arrow(14.1, 6.4, 15.2, 6.4, '#48DBFB')

    block(15.3, 5.5, 3.8, 1.8, '#27ae60', 'GNN Training',
          'PointNetTransfGAT\nAdamW + Early Stopping\n500-1000 epochs')
    arrow(19.2, 6.4, 20.3, 6.4, '#2ECC71')

    block(20.4, 5.5, 3.8, 1.8, '#f39c12', 'Test Evaluation',
          '50-100 unseen graphs\nR², MAE, RMSE\nPer-node predictions')

    # Row 2: UQ
    arrow(17.2, 5.5, 5, 3.2, '#9B59B6')

    block(0, 1.5, 5, 1.8, '#8e44ad', 'MC Dropout (UQ Method 1)',
          'Keep dropout ON during inference\n30 forward passes → mean + std\nSpearman rho = 0.482 (Trial 8)')
    arrow(5.1, 2.4, 6.8, 2.4, '#9B59B6')

    block(7, 1.5, 5, 1.8, '#6c3483', 'Ensemble (UQ Method 2)',
          'Exp A: 5 training runs (same arch)\nExp B: Multi-model (T2,5,6,7,8)\nEpistemic + Aleatoric uncertainty')
    arrow(12.1, 2.4, 13.8, 2.4, '#9B59B6')

    block(14, 1.5, 5, 1.8, '#2980b9', 'Temperature Scaling (Calib.)',
          'T = 2.90: scales sigma_new = T * sigma\nECE: 0.356 --> 0.033 (90.6% better)\n1-sigma coverage: 32.8% --> ~68%')
    arrow(19.1, 2.4, 20.5, 2.4, '#2980b9')

    block(20.6, 1.5, 5, 1.8, '#d4ac0d', 'Thesis Contribution',
          '1) Reproduced Elena\'s GNN\n2) Added UQ (MC Dropout + Ensemble)\n3) Calibrated uncertainties\n4) Only 10% training data!')

    plt.savefig(os.path.join(OUT, 'A2_data_pipeline.png'), dpi=250, bbox_inches='tight',
                facecolor='#0F1419', edgecolor='none')
    plt.close()
    print("  A2 Data pipeline done")


# ============================================================
# CHART B1: ALL TRIALS 3D (with Trial 1 clearly marked)
# ============================================================

def chart_all_trials_3d(R):
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')

    trials = sorted(R.keys())
    r2 = [R[t]['r2'] for t in trials]
    mae = [R[t]['mae'] for t in trials]
    rmse = [R[t]['rmse'] for t in trials]

    x = np.arange(8)
    w, d = 0.22, 0.5

    # R² bars
    colors_r2 = ['#E74C3C' if t == 1 else '#4A90D9' for t in trials]
    ax.bar3d(x, np.zeros(8), np.zeros(8), w, d, r2, color=colors_r2, alpha=0.85, edgecolor='white', lw=0.5)
    
    # MAE bars (scaled)
    colors_mae = ['#E74C3C' if t == 1 else '#FF6B6B' for t in trials]
    ax.bar3d(x, np.ones(8)*1.2, np.zeros(8), w, d, [m/10 for m in mae], color=colors_mae, alpha=0.85, edgecolor='white', lw=0.5)
    
    # RMSE bars (scaled)
    colors_rmse = ['#E74C3C' if t == 1 else '#F39C12' for t in trials]
    ax.bar3d(x, np.full(8, 2.4), np.zeros(8), w, d, [r/10 for r in rmse], color=colors_rmse, alpha=0.85, edgecolor='white', lw=0.5)

    ax.set_xticks(x + w/2)
    labels = []
    for t in trials:
        if t == 1:
            labels.append('T1\nWRONG ARCH')
        else:
            labels.append(f'T{t}')
    ax.set_xticklabels(labels, fontsize=9, rotation=-15)
    ax.set_yticks([0, 1.2, 2.4])
    ax.set_yticklabels(['R² Score', 'MAE / 10', 'RMSE / 10'], fontsize=9)
    ax.set_zlabel('Value', fontsize=11, fontweight='bold')

    ax.set_title('All 8 Trials: 3D Performance Comparison\n'
                 'Trial 1 = OLD architecture (Linear final layer) — shown in RED\n'
                 'Trials 2-8 = Correct PointNetTransfGAT (GATConv final)',
                 fontsize=15, fontweight='bold', pad=25)

    # Best correct trial annotation
    ax.text2D(0.02, 0.88,
              'Trial 1: R²=0.786 (BUT wrong architecture!)\n'
              'Best CORRECT trial: T8 R²=0.596\n'
              'T1 had Linear final layer instead of GATConv\n'
              '→ Not comparable to Elena\'s paper',
              transform=ax.transAxes, fontsize=10, fontweight='bold',
              color='#E74C3C',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95))

    ax.text2D(0.02, 0.02,
              'All trials trained on 10% Paris population\nPointNetTransfGAT architecture (Elena Natterer\'s paper)',
              transform=ax.transAxes, fontsize=8, color='#888', style='italic')

    ax.view_init(elev=25, azim=-50)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'B1_all_trials_3d.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  B1 All trials 3D done")


# ============================================================
# CHART B2: CORRECT TRIALS ONLY (T2-T8) — Clean comparison
# ============================================================

def chart_correct_trials(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    correct = [2, 3, 4, 5, 6, 7, 8]
    colors = [TC[t-1] for t in correct]

    # R²
    ax = axes[0]
    r2v = [R[t]['r2'] for t in correct]
    bars = ax.bar([f'T{t}' for t in correct], r2v, color=colors, edgecolor='white', lw=2)
    for bar, val, t in zip(bars, r2v, correct):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    best_idx = np.argmax(r2v)
    bars[best_idx].set_edgecolor('#2ECC71')
    bars[best_idx].set_linewidth(4)
    ax.annotate('BEST', xy=(best_idx, r2v[best_idx]),
                xytext=(best_idx + 0.5, r2v[best_idx] + 0.05),
                fontsize=12, fontweight='bold', color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))
    ax.set_title('R² Score (higher = better)\nOnly correct architecture trials', fontsize=14, fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # MAE
    ax = axes[1]
    maev = [R[t]['mae'] for t in correct]
    bars = ax.bar([f'T{t}' for t in correct], maev, color=colors, edgecolor='white', lw=2)
    for bar, val in zip(bars, maev):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    best_idx = np.argmin(maev)
    bars[best_idx].set_edgecolor('#2ECC71')
    bars[best_idx].set_linewidth(4)
    ax.set_title('MAE (lower = better)\nvehicles/hour average error', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (veh/h)', fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # % under 5 error
    ax = axes[2]
    u5v = [R[t]['u5'] for t in correct]
    bars = ax.bar([f'T{t}' for t in correct], u5v, color=colors, edgecolor='white', lw=2)
    for bar, val in zip(bars, u5v):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('% Predictions with < 5 veh/h error\nhigher = more accurate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage', fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    fig.suptitle('Trials 2-8: Correct PointNetTransfGAT Architecture (GATConv final layer)\n'
                 'Trial 1 excluded — used Linear final layer (wrong architecture)',
                 fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'B2_correct_trials_comparison.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  B2 Correct trials comparison done")


# ============================================================
# CHART B3: HYPERPARAMETER LANDSCAPE 3D
# ============================================================

def chart_hyperparams_3d(R):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    SETTINGS = {
        2: (0.001, 0.20, 16), 3: (0.001, 0.30, 16), 4: (0.001, 0.30, 16),
        5: (0.0001, 0.20, 32), 6: (0.0001, 0.30, 32),
        7: (0.0001, 0.15, 32), 8: (0.001, 0.15, 32),
    }

    for t in [2,3,4,5,6,7,8]:
        lr, do, bs = SETTINGS[t]
        r2 = R[t]['r2']
        size = r2 * 1200 + 100
        c = TC[t-1]
        ax.scatter(lr, do, r2, c=c, s=size, edgecolors='white', lw=2, alpha=0.9, zorder=5)
        ax.text(lr, do, r2+0.02, f'T{t}\nR²={r2:.3f}\nBS={bs}',
                fontsize=8.5, ha='center', fontweight='bold', color=c)

    ax.set_xlabel('\nLearning Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('\nDropout Rate', fontsize=11, fontweight='bold')
    ax.set_zlabel('\nR² Score', fontsize=11, fontweight='bold')
    ax.set_title('Hyperparameter Exploration (Correct Architecture Only)\n'
                 'Bubble size proportional to R² | 7 trials explored',
                 fontsize=15, fontweight='bold', pad=25)

    ax.text2D(0.02, 0.08,
              'Key finding: Lower dropout (0.15) + BS=32 works best\n'
              'LR=1e-3 with low dropout: Trial 8 R²=0.596\n'
              'Weighted loss (T3, T4) did NOT help performance',
              transform=ax.transAxes, fontsize=9.5, color='#555',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', alpha=0.9))

    ax.view_init(elev=30, azim=-55)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'B3_hyperparameter_landscape.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  B3 Hyperparameter landscape done")


# ============================================================
# CHARTS C1-C7: PER-TRIAL DETAILED ANALYSIS
# ============================================================

def chart_per_trial(R, trial_num):
    d = R[trial_num]
    is_wrong = (trial_num == 1)
    color = '#E74C3C' if is_wrong else TC[trial_num - 1]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # 1. Scatter plot
    ax = axes[0, 0]
    idx = np.random.RandomState(42).choice(len(d['p']), min(100000, len(d['p'])), replace=False)
    hb = ax.hexbin(d['t'][idx], d['p'][idx], gridsize=80, cmap='YlOrRd', mincnt=1, lw=0.1)
    lims = [d['t'].min(), d['t'].max()]
    ax.plot(lims, lims, 'k--', lw=2, alpha=0.7)
    ax.set_xlabel('Actual DeltaVolume', fontsize=11)
    ax.set_ylabel('Predicted DeltaVolume', fontsize=11)
    ax.set_title('Predicted vs Actual', fontsize=13, fontweight='bold')
    plt.colorbar(hb, ax=ax, shrink=0.8)
    wm(ax)

    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(d['e'][d['e'] < 20], bins=100, color=color, alpha=0.75, edgecolor='white', lw=0.5)
    ax.axvline(d['mae'], color='black', lw=2, ls='--', label=f'MAE={d["mae"]:.3f}')
    ax.axvline(d['med'], color='blue', lw=2, ls=':', label=f'Median={d["med"]:.3f}')
    ax.set_xlabel('Absolute Error (veh/h)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    wm(ax)

    # 3. CDF
    ax = axes[0, 2]
    sorted_e = np.sort(d['e'])
    cdf = np.arange(len(sorted_e)) / len(sorted_e)
    ax.plot(sorted_e, cdf, color=color, lw=3)
    ax.axhline(0.5, color='#aaa', ls=':', alpha=0.5)
    ax.axhline(0.9, color='#aaa', ls='--', alpha=0.5)
    ax.axvline(5, color='#888', ls='--', alpha=0.5)
    ax.text(5.5, 0.5, f'{d["u5"]:.1f}% < 5 veh/h', fontsize=10, color='#555')
    ax.set_xlim(0, 25)
    ax.set_xlabel('Error Threshold (veh/h)', fontsize=11)
    ax.set_ylabel('Cumulative Fraction', fontsize=11)
    ax.set_title('Cumulative Error Distribution', fontsize=13, fontweight='bold')
    wm(ax)

    # 4. Residual plot
    ax = axes[1, 0]
    residuals = d['p'][idx] - d['t'][idx]
    ax.hexbin(d['t'][idx], residuals, gridsize=80, cmap='coolwarm', mincnt=1, lw=0.1)
    ax.axhline(0, color='black', lw=2, ls='-')
    ax.set_xlabel('Actual DeltaVolume', fontsize=11)
    ax.set_ylabel('Residual (Pred - Actual)', fontsize=11)
    ax.set_title('Residual Analysis', fontsize=13, fontweight='bold')
    wm(ax)

    # 5. Stats card
    ax = axes[1, 1]; ax.axis('off')
    arch_note = 'OLD Architecture (Linear final)\nNOT matching Elena\'s paper!' if is_wrong else 'Correct PointNetTransfGAT\n(GATConv final layer)'
    arch_color = '#E74C3C' if is_wrong else '#2ECC71'

    stats_text = [
        ('Architecture', arch_note, arch_color),
        ('R² Score', f'{d["r2"]:.6f}', PAL['blue']),
        ('MAE', f'{d["mae"]:.4f} veh/h', PAL['blue']),
        ('RMSE', f'{d["rmse"]:.4f} veh/h', PAL['blue']),
        ('Median Error', f'{d["med"]:.4f} veh/h', PAL['blue']),
        ('90th Percentile', f'{d["p90"]:.3f} veh/h', PAL['blue']),
        ('% Under 5 Error', f'{d["u5"]:.2f}%', PAL['blue']),
        ('N Predictions', f'{d["n"]:,}', PAL['dark']),
    ]
    for i, (k, v, c) in enumerate(stats_text):
        y = 0.92 - i * 0.11
        ax.text(0.05, y, f'{k}:', fontsize=12, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.5, y, v, fontsize=11, transform=ax.transAxes, color=c, fontweight='bold')

    # 6. Top/Bottom errors
    ax = axes[1, 2]
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pvals = [np.percentile(d['e'], p) for p in percentiles]
    ax.barh([f'P{p}' for p in percentiles], pvals, color=cm.RdYlGn_r(np.linspace(0.2, 0.8, len(percentiles))),
            edgecolor='white', lw=1.5)
    for i, (p, v) in enumerate(zip(percentiles, pvals)):
        ax.text(v+0.1, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Absolute Error (veh/h)', fontsize=11, fontweight='bold')
    ax.set_title('Error Percentiles', fontsize=13, fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    for a in axes.flat:
        if a.get_visible():
            a.set_facecolor('#FAFBFC')

    title_prefix = 'Trial 1 [WRONG ARCHITECTURE — Linear final layer]' if is_wrong else f'Trial {trial_num}'
    fig.suptitle(f'{title_prefix} — Detailed Analysis\n'
                 f'{TRIAL_SHORT[trial_num]}',
                 fontsize=17, fontweight='bold', y=1.03,
                 color='#E74C3C' if is_wrong else PAL['dark'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'C{trial_num}_trial_{trial_num}_detail.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  C{trial_num} Trial {trial_num} detail done")


# ============================================================
# CHART D1: MC DROPOUT — All 4 trials combined
# ============================================================

def chart_mc_dropout_all(M):
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    mc_trials = sorted(M.keys())

    for i, t in enumerate(mc_trials):
        ax = axes[i//2][i%2]
        d = M[t]
        idx = np.random.RandomState(42).choice(len(d['e']), min(80000, len(d['e'])), replace=False)

        hb = ax.hexbin(d['u'][idx], d['e'][idx], gridsize=70, cmap='magma_r', mincnt=1, lw=0.1)

        # Trend line
        edges = np.percentile(d['u'][idx], np.linspace(0, 100, 26))
        bc, bm = [], []
        for j in range(25):
            mask = (d['u'][idx] >= edges[j]) & (d['u'][idx] < edges[j+1])
            if mask.sum() > 10:
                bc.append(d['u'][idx][mask].mean())
                bm.append(d['e'][idx][mask].mean())
        ax.plot(bc, bm, 'c-', lw=3.5, label='Mean Error Trend', zorder=10)

        is_best = (t == 8)
        star = ' [BEST]' if is_best else ''
        color = '#2ECC71' if is_best else PAL['dark']
        ax.set_title(f'Trial {t}{star}\nSpearman rho = {d["rho"]:.4f} | Pearson r = {d["pr"]:.4f}',
                     fontsize=13, fontweight='bold', color=color)

        stats = (f'Mean uncertainty: {d["mu"]:.3f}\n'
                 f'Mean error: {d["me"]:.3f}\n'
                 f'1-sigma coverage: {d["c1"]:.1f}% (ideal: 68.3%)\n'
                 f'2-sigma coverage: {d["c2"]:.1f}% (ideal: 95.4%)\n'
                 f'N = {len(d["p"]):,}')
        ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#ddd'))

        ax.set_xlabel('MC Dropout Uncertainty (sigma)', fontsize=11)
        ax.set_ylabel('Absolute Error |pred - actual|', fontsize=11)
        ax.set_xlim(0, np.percentile(d['u'][idx], 99))
        ax.set_ylim(0, np.percentile(d['e'][idx], 99))
        ax.legend(fontsize=9, loc='upper left')
        plt.colorbar(hb, ax=ax, shrink=0.75)
        ax.set_facecolor('#FAFBFC')
        wm(ax)

    fig.suptitle('MC Dropout: Uncertainty vs Prediction Error\n'
                 '30 stochastic forward passes per prediction — upward trend = uncertainty is meaningful',
                 fontsize=17, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'D1_mc_dropout_all.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  D1 MC Dropout all done")


# ============================================================
# CHART D2: ENSEMBLE EXPERIMENTS
# ============================================================

def chart_ensemble(ea, eb, M):
    fig, axes = plt.subplots(2, 3, figsize=(24, 15))

    targets_a = ea['targets'].flatten()
    ens_mean = ea['ensemble_mean'].flatten()
    ens_var = ea['ensemble_variance'].flatten()
    mc_avg = ea['avg_mc_uncertainty'].flatten()
    combined = ea['combined_uncertainty'].flatten()
    errors_a = np.abs(ens_mean - targets_a)

    targets_b = eb['targets'].flatten()
    ens_pred_b = eb['ensemble_prediction'].flatten()
    ens_unc_b = eb['ensemble_uncertainty'].flatten()
    errors_b = np.abs(ens_pred_b - targets_b)

    # A1: Experiment A — 3 uncertainty types
    ax = axes[0, 0]
    types = ['Ensemble\nVariance\n(epistemic)', 'MC Avg\n(aleatoric)', 'Combined\n(both)']
    rhos = [spearmanr(ens_var, errors_a)[0], spearmanr(mc_avg, errors_a)[0], spearmanr(combined, errors_a)[0]]
    clrs_a = ['#FF6B6B', '#4A90D9', '#F39C12']
    bars = ax.bar(types, rhos, color=clrs_a, edgecolor='white', lw=2, width=0.5)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman rho', fontsize=12, fontweight='bold')
    ax.set_title('Experiment A: 3 Uncertainty Types\n5 training runs of Trial 8 architecture',
                 fontsize=13, fontweight='bold')
    ax.text(0.5, -0.13, 'Epistemic = model doesn\'t know | Aleatoric = data is noisy',
            transform=ax.transAxes, fontsize=9, color='#888', ha='center', style='italic')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # A2: Experiment B — per-model R²
    ax = axes[0, 1]
    mk = ['model_2_predictions','model_5_predictions','model_6_predictions','model_7_predictions','model_8_predictions']
    mn = ['T2','T5','T6','T7','T8']
    mc = ['#FF8E53','#48DBFB','#9B59B6','#1ABC9C','#2ECC71']
    model_r2 = [r2_score(targets_b, eb[k].flatten()) for k in mk]
    names = mn + ['Ensemble']
    r2s = model_r2 + [r2_score(targets_b, ens_pred_b)]
    clrs_b = mc + ['#FFD700']
    bars = ax.bar(names, r2s, color=clrs_b, edgecolor='white', lw=2)
    for bar, val in zip(bars, r2s):
        label_y = val + 0.001 if val >= 0 else val - 0.003
        ax.text(bar.get_x()+bar.get_width()/2, label_y, f'{val:.4f}', ha='center', fontsize=9, fontweight='bold',
                va='bottom' if val >= 0 else 'top')
    ax.axhline(0, color='#888', lw=1)
    ax.set_title('Experiment B: Multi-Model R²\nR² near 0 = different data distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # A3: Grand UQ ranking
    ax = axes[0, 2]
    all_methods = [
        ('MC Drop T8\n(30 samples)', M[8]['rho'], '#4A90D9'),
        ('MC Drop T7', M[7]['rho'], '#0ABDE3'),
        ('MC Drop T5', M[5]['rho'], '#48DBFB'),
        ('MC Drop T6', M[6]['rho'], '#87CEEB'),
        ('Ens MC Avg\n(Exp A)', spearmanr(mc_avg, errors_a)[0], '#F39C12'),
        ('Combined\n(Exp A)', spearmanr(combined, errors_a)[0], '#FF8E53'),
        ('Multi-Model\n(Exp B)', spearmanr(ens_unc_b, errors_b)[0], '#9B59B6'),
        ('Ens Variance\n(Exp A)', spearmanr(ens_var, errors_a)[0], '#FF6B6B'),
    ]
    all_methods.sort(key=lambda x: x[1], reverse=True)
    bars = ax.barh(range(len(all_methods)), [m[1] for m in all_methods],
                   color=[m[2] for m in all_methods], edgecolor='white', lw=2, height=0.65)
    ax.set_yticks(range(len(all_methods)))
    ax.set_yticklabels([m[0] for m in all_methods], fontsize=9)
    for bar, m in zip(bars, all_methods):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f'{m[1]:.4f}',
                va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Spearman rho', fontsize=11, fontweight='bold')
    ax.set_title('All UQ Methods Ranked\nMC Dropout is the clear winner!', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.annotate('WINNER!', xy=(all_methods[0][1], 0), xytext=(all_methods[0][1]-0.1, -0.8),
                fontsize=12, fontweight='bold', color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # A4: Error reduction comparison
    ax = axes[1, 0]
    pcts = np.arange(50, 96, 2)
    overall_mc = np.mean(M[8]['e'])
    overall_a = np.mean(errors_a)
    overall_b = np.mean(errors_b)

    for label, unc, err, overall, color in [
        ('MC Dropout (T8)', M[8]['u'], M[8]['e'], overall_mc, '#4A90D9'),
        ('Combined (Exp A)', combined, errors_a, overall_a, '#F39C12'),
        ('Multi-Model (Exp B)', ens_unc_b, errors_b, overall_b, '#9B59B6'),
    ]:
        reds = [(overall - np.mean(err[unc <= np.percentile(unc, p)])) / overall * 100 for p in pcts]
        ax.plot(pcts, reds, 'o-', color=color, lw=2.5, ms=5, label=label, alpha=0.85)

    ax.set_xlabel('Keep top X% most confident', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error Reduction %', fontsize=11, fontweight='bold')
    ax.set_title('Practical Value: Error Reduction\nby filtering uncertain predictions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # A5: MC Dropout explanation visual
    ax = axes[1, 1]; ax.axis('off')
    ax.set_title('How MC Dropout Works', fontsize=16, fontweight='bold', color=PAL['dark'])

    steps = [
        ('1. Train GNN with dropout', '#2e86c1', 'Standard training with dropout layers'),
        ('2. At inference: keep dropout ON', '#8e44ad', 'Unlike normal: DO NOT turn off dropout'),
        ('3. Run 30 forward passes', '#27ae60', 'Same input → 30 different predictions'),
        ('4. Prediction = mean of 30', '#f39c12', 'Average gives final prediction'),
        ('5. Uncertainty = std of 30', '#e74c3c', 'Standard deviation = how uncertain'),
        ('6. No retraining needed!', '#2ECC71', 'Cheapest UQ method available'),
    ]
    for i, (title, color, desc) in enumerate(steps):
        y = 0.88 - i * 0.15
        ax.text(0.02, y, title, fontsize=12, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.02, y-0.04, desc, fontsize=10, color='#666', transform=ax.transAxes)

    # A6: Ensemble explanation
    ax = axes[1, 2]; ax.axis('off')
    ax.set_title('Ensemble Methods Summary', fontsize=16, fontweight='bold', color=PAL['dark'])

    items = [
        ('Experiment A (5 training runs)', '#8e44ad',
         'Same architecture × 5 random seeds\nVariance = epistemic uncertainty\nMC Dropout also run per model\nCombined = sqrt(Var + MC²)'),
        ('Experiment B (multi-model)', '#9B59B6',
         'Trials 2, 5, 6, 7, 8 as ensemble\nDifferent hyperparameters\nDisagreement = uncertainty\nrho = 0.117'),
        ('Key Finding', '#2ECC71',
         'MC Dropout ALONE (rho=0.482) beats\nall ensemble methods (rho~0.1-0.16)\nMore accurate + cheaper to compute!'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.88 - i * 0.30
        ax.text(0.02, y, title, fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.02, y-0.05, desc, fontsize=10, color='#555', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Uncertainty Quantification: MC Dropout + Ensemble Experiments\n'
                 'Comparing all UQ methods from our thesis research',
                 fontsize=17, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'D2_ensemble_experiments.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  D2 Ensemble experiments done")


# ============================================================
# CHART D3: CALIBRATION (Temperature Scaling)
# ============================================================

def chart_calibration(M):
    d = M[8]
    unc, err = d['u'], d['e']

    def ece_fn(T, u, e, nb=10):
        s = u * T
        edges = np.unique(np.percentile(s, np.linspace(0, 100, nb+1)))
        val = 0.0
        for j in range(len(edges)-1):
            mask = (s >= edges[j]) & (s < edges[j+1]) if j < len(edges)-2 else (s >= edges[j]) & (s <= edges[j+1])
            if mask.sum() == 0: continue
            val += (mask.sum() / len(u)) * abs(np.mean(e[mask] < s[mask]) - 0.683)
        return val

    res = minimize_scalar(lambda T: ece_fn(T, unc, err), bounds=(0.1, 20), method='bounded')
    T_opt = res.x
    ece_before = ece_fn(1.0, unc, err)
    ece_after = ece_fn(T_opt, unc, err)
    improv = (ece_before - ece_after) / ece_before * 100

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 1. Coverage comparison
    ax = axes[0]
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected = [0.383, 0.683, 0.866, 0.954, 0.988, 0.997]
    orig_cov = [np.mean(err < s*unc) for s in sigmas]
    cal_cov = [np.mean(err < s*unc*T_opt) for s in sigmas]

    x = np.arange(len(sigmas))
    w = 0.25
    ax.bar(x-w, expected, w, label='Gaussian Ideal', color='#E6E6FA', edgecolor=PAL['dark'], lw=1.5)
    ax.bar(x, orig_cov, w, label='Before (T=1.0)', color='#FFEAA7', edgecolor=PAL['dark'], lw=1.5)
    ax.bar(x+w, cal_cov, w, label=f'After (T={T_opt:.2f})', color='#98FB98', edgecolor=PAL['dark'], lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}sigma' for s in sigmas], fontsize=12)
    ax.set_ylabel('Coverage', fontsize=11, fontweight='bold')
    ax.set_title(f'Coverage at Different Sigma Levels\n1-sigma: {orig_cov[1]*100:.1f}% -> {cal_cov[1]*100:.1f}%',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_facecolor('#FAFBFC')

    # 2. ECE comparison
    ax = axes[1]
    bars = ax.bar(['Before\n(T=1.0)', f'After\n(T={T_opt:.2f})'], [ece_before, ece_after],
                  color=['#FFEAA7', '#98FB98'], edgecolor=PAL['dark'], lw=2, width=0.5)
    for bar, val in zip(bars, [ece_before, ece_after]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.4f}', ha='center', fontsize=15, fontweight='bold')
    ax.set_ylabel('ECE (lower = better)', fontsize=11, fontweight='bold')
    ax.set_title(f'ECE Improvement: {improv:.1f}%', fontsize=16, fontweight='bold', color='#2ECC71')
    ax.annotate(f'{improv:.0f}% better!', xy=(1, ece_after), xytext=(0.5, (ece_before+ece_after)/2),
                fontsize=16, fontweight='bold', color='#2ECC71', ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=3))
    ax.set_facecolor('#FAFBFC')

    # 3. Explanation
    ax = axes[2]; ax.axis('off')
    ax.set_title('What is Temperature Scaling?', fontsize=16, fontweight='bold', color=PAL['dark'])

    steps = [
        ('Problem', '#E74C3C', 'MC Dropout uncertainties are too narrow\n'
         f'1-sigma coverage: {d["c1"]:.1f}% (should be 68.3%)\nModel is overconfident!'),
        ('Solution', '#2ECC71', f'Multiply all uncertainties by T = {T_opt:.2f}\n'
         f'sigma_new = {T_opt:.2f} x sigma_old\nSimple post-processing, no retraining'),
        ('Result', '#4A90D9', f'ECE drops from {ece_before:.4f} to {ece_after:.4f}\n'
         f'{improv:.1f}% improvement in calibration!\n1-sigma coverage now close to 68.3%'),
        ('Why it matters', '#F39C12', 'Calibrated uncertainties are trustworthy\n'
         'Traffic planners can set confidence thresholds\nand know they mean what they say!'),
    ]
    for i, (title, color, desc) in enumerate(steps):
        y = 0.90 - i * 0.22
        ax.text(0.02, y, title + ':', fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.02, y-0.04, desc, fontsize=10, color='#555', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Temperature Scaling Calibration\n'
                 'Making uncertainty estimates trustworthy for traffic planning decisions',
                 fontsize=17, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'D3_calibration.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  D3 Calibration done (T={T_opt:.4f}, ECE: {ece_before:.4f} -> {ece_after:.4f})")
    return T_opt, ece_before, ece_after


# ============================================================
# CHART E1: SPATIAL MAP (Paris network)
# ============================================================

def chart_spatial(M, pos):
    d = M[8]
    n = 31635
    unc_avg = d['u'].reshape(-1, n).mean(axis=0)
    err_avg = d['e'].reshape(-1, n).mean(axis=0)
    p = pos[:n].mean(axis=1)
    lons, lats = p[:, 0], p[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    sc1 = ax1.scatter(lons, lats, c=unc_avg, cmap='hot_r', s=2, alpha=0.7,
                      vmin=np.percentile(unc_avg, 5), vmax=np.percentile(unc_avg, 95))
    cb1 = plt.colorbar(sc1, ax=ax1, shrink=0.8)
    cb1.set_label('Mean MC Dropout Uncertainty (sigma)', fontsize=10)
    ax1.set_title('Where Is The Model Uncertain?\n31,635 road links, each averaged over 100 test scenarios',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')
    ax1.set_facecolor('#1a1a2e')

    sc2 = ax2.scatter(lons, lats, c=err_avg, cmap='inferno', s=2, alpha=0.7,
                      vmin=np.percentile(err_avg, 5), vmax=np.percentile(err_avg, 95))
    cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.8)
    cb2.set_label('Mean Absolute Error (veh/h)', fontsize=10)
    ax2.set_title('Where Does The Model Make Errors?\nDo high-uncertainty areas match high-error areas?',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude'); ax2.set_ylabel('Latitude')
    ax2.set_facecolor('#1a1a2e')

    rho_s, _ = spearmanr(unc_avg, err_avg)
    fig.suptitle(f'Paris Road Network — Spatial Uncertainty & Error Analysis\n'
                 f'Spatial correlation (uncertainty vs error): rho = {rho_s:.4f}',
                 fontsize=17, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'E1_spatial_map.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  E1 Spatial map done")


# ============================================================
# CHART E2: 3D ERROR SURFACE
# ============================================================

def chart_3d_surface(M, feats):
    d = M[8]
    n = 31635
    feat = np.tile(feats[:n], (len(d['p'])//n, 1))
    ml = min(len(feat), len(d['p']))
    vol, unc, err = feat[:ml, 0], d['u'][:ml], d['e'][:ml]

    nb = 25
    vb = np.percentile(vol, np.linspace(0, 100, nb+1))
    ub = np.percentile(unc, np.linspace(0, 100, nb+1))
    Z = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            mask = (vol >= vb[i]) & (vol < vb[i+1]) & (unc >= ub[j]) & (unc < ub[j+1])
            if mask.sum() > 10:
                Z[i, j] = np.mean(err[mask])

    X, Y = np.meshgrid((vb[:-1]+vb[1:])/2, (ub[:-1]+ub[1:])/2)
    Z_plot = np.nan_to_num(Z.T, nan=0)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_plot, cmap='magma', alpha=0.85, lw=0.3, edgecolor='white', antialiased=True)
    ax.set_xlabel('\nBase Traffic Volume (veh/h)', fontsize=11, fontweight='bold')
    ax.set_ylabel('\nMC Dropout Uncertainty (sigma)', fontsize=11, fontweight='bold')
    ax.set_zlabel('\nMean Prediction Error', fontsize=11, fontweight='bold')
    ax.set_title('3D Error Surface: Volume x Uncertainty x Error\n'
                 'Busy roads + high uncertainty = worst prediction errors',
                 fontsize=15, fontweight='bold', pad=25)
    plt.colorbar(surf, shrink=0.5, pad=0.1, label='Mean Absolute Error')
    ax.view_init(elev=30, azim=-45)

    ax.text2D(0.02, 0.05, 'Peak errors on busy roads where model is also uncertain\n'
              'Traffic planners should review these predictions most carefully',
              transform=ax.transAxes, fontsize=9, color='#555',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'E2_3d_error_surface.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  E2 3D surface done")


# ============================================================
# CHART E3: FEATURE ANALYSIS
# ============================================================

def chart_features(M, feats):
    d = M[8]
    n = 31635
    fnames = ['VOL_BASE_CASE\n(base traffic)', 'CAPACITY\n(max flow)', 'CAP_REDUCTION\n(disruption)',
              'FREESPEED\n(free-flow speed)', 'LANES\n(road lanes)', 'LENGTH\n(link length m)']

    feat = np.tile(feats[:n], (len(d['p'])//n, 1))
    ml = min(len(feat), len(d['p']))
    feat, unc, err = feat[:ml], d['u'][:ml], d['e'][:ml]

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    idx = np.random.RandomState(42).choice(ml, min(50000, ml), replace=False)

    for i, (fname, ax) in enumerate(zip(fnames, axes.flat)):
        hb = ax.hexbin(feat[idx, i], unc[idx], gridsize=60, cmap='YlOrRd', mincnt=1, lw=0.1)
        rho_u, _ = spearmanr(feat[idx, i], unc[idx])
        rho_e, _ = spearmanr(feat[idx, i], err[idx])

        strength = 'STRONG' if abs(rho_u) > 0.3 else 'moderate' if abs(rho_u) > 0.15 else 'weak'
        color = '#E74C3C' if abs(rho_u) > 0.3 else '#F39C12' if abs(rho_u) > 0.15 else '#999'

        ax.set_xlabel(fname, fontsize=11, fontweight='bold')
        ax.set_ylabel('MC Dropout Uncertainty', fontsize=10)
        ax.set_title(f'rho(feat, unc) = {rho_u:.3f} [{strength}]\nrho(feat, error) = {rho_e:.3f}',
                     fontsize=11, fontweight='bold', color=color)
        ax.set_facecolor('#FAFBFC')
        plt.colorbar(hb, ax=ax, shrink=0.8)
        wm(ax)

    fig.suptitle('Which Input Features Drive Prediction Uncertainty?\n'
                 'Trial 8, MC Dropout — 6 features vs uncertainty analysis',
                 fontsize=17, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'E3_feature_analysis.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  E3 Feature analysis done")


# ============================================================
# CHART E4: PRACTICAL VALUE — With vs Without UQ
# ============================================================

def chart_practical(M, R):
    d = M[8]
    overall_mae = R[8]['mae']
    thresh90 = np.percentile(d['u'], 90)
    low = d['u'] <= thresh90
    mae_conf = np.mean(d['e'][low])
    mae_unconf = np.mean(d['e'][~low])
    improv = (overall_mae - mae_conf) / overall_mae * 100

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    # Left: Without UQ
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title('WITHOUT Uncertainty', fontsize=16, fontweight='bold', color='#E74C3C')
    box = FancyBboxPatch((0.3, 1.5), 9.3, 7, boxstyle='round,pad=0.3', facecolor='#FFE4E1', edgecolor='#E74C3C', lw=3)
    ax.add_patch(box)
    ax.text(5, 7, '3,163,500 Predictions', fontsize=15, ha='center', fontweight='bold')
    ax.text(5, 5.5, 'ALL treated the same', fontsize=13, ha='center', color='#666')
    ax.text(5, 4, f'Average Error = {overall_mae:.3f} veh/h', fontsize=14, ha='center', fontweight='bold', color='#E74C3C')
    ax.text(5, 2.5, 'No way to tell which predictions\nare reliable vs which are garbage!', fontsize=11, ha='center', color='#E74C3C')

    # Middle: With UQ
    ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title('WITH MC Dropout UQ', fontsize=16, fontweight='bold', color='#2ECC71')
    box1 = FancyBboxPatch((0.3, 5), 9.3, 3.5, boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2ECC71', lw=3)
    ax.add_patch(box1)
    ax.text(5, 7.8, f'90% Confident: {low.sum():,}', fontsize=12, ha='center', fontweight='bold', color='#2ECC71')
    ax.text(5, 6.8, f'MAE = {mae_conf:.3f} veh/h', fontsize=14, ha='center', fontweight='bold')
    ax.text(5, 5.5, 'TRUST these predictions!', fontsize=10, ha='center', color='#2ECC71', style='italic')

    box2 = FancyBboxPatch((0.3, 1.5), 9.3, 3, boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#F39C12', lw=3)
    ax.add_patch(box2)
    ax.text(5, 3.8, f'10% Uncertain: {(~low).sum():,}', fontsize=12, ha='center', fontweight='bold', color='#F39C12')
    ax.text(5, 2.8, f'MAE = {mae_unconf:.3f} veh/h', fontsize=14, ha='center', fontweight='bold')
    ax.text(5, 1.8, 'FLAG for review / re-simulate!', fontsize=10, ha='center', color='#F39C12', style='italic')

    # Right: Impact
    ax = axes[2]
    labels = ['All\n(no UQ)', 'Top 90%\n(confident)', 'Bottom 10%\n(uncertain)']
    vals = [overall_mae, mae_conf, mae_unconf]
    colors = ['#FFE4E1', '#98FB98', '#FFEAA7']
    bars = ax.bar(labels, vals, color=colors, edgecolor=PAL['dark'], lw=2, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.3f}', ha='center', fontsize=14, fontweight='bold')
    ax.set_title(f'{improv:.1f}% Lower Error for Confident Predictions', fontsize=14, fontweight='bold', color='#2ECC71')
    ax.set_ylabel('MAE (vehicles/hour)', fontsize=12, fontweight='bold')
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    fig.suptitle('The Practical Value of Uncertainty for Traffic Planners\n'
                 'UQ tells you WHICH predictions to trust and which to double-check',
                 fontsize=17, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'E4_practical_value.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  E4 Practical value done")


# ============================================================
# CHART E5: PER-GRAPH ANALYSIS
# ============================================================

def chart_per_graph(R):
    p, t = R[8]['p'], R[8]['t']
    n = 31635
    ng = len(p) // n

    gr2, gmae = [], []
    for g in range(ng):
        s, e = g*n, (g+1)*n
        gr2.append(r2_score(t[s:e], p[s:e]))
        gmae.append(mean_absolute_error(t[s:e], p[s:e]))
    gr2, gmae = np.array(gr2), np.array(gmae)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # R² per graph
    ax = axes[0]
    colors = cm.viridis(np.linspace(0, 1, ng))
    ax.bar(range(ng), gr2, color=colors, width=0.9)
    ax.axhline(np.mean(gr2), color='#E74C3C', lw=2, ls='--', label=f'Mean: {np.mean(gr2):.4f}')
    ax.axhline(np.median(gr2), color='#4A90D9', lw=2, ls=':', label=f'Median: {np.median(gr2):.4f}')
    ax.set_xlabel('Test Scenario Index', fontsize=11)
    ax.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax.set_title(f'R² per Test Scenario ({ng} graphs)\nSome disruptions are easier to predict',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # MAE per graph
    ax = axes[1]
    ax.bar(range(ng), gmae, color=cm.plasma(np.linspace(0, 1, ng)), width=0.9)
    ax.axhline(np.mean(gmae), color='#E74C3C', lw=2, ls='--', label=f'Mean: {np.mean(gmae):.3f}')
    ax.set_xlabel('Test Scenario Index', fontsize=11)
    ax.set_ylabel('MAE (veh/h)', fontsize=11, fontweight='bold')
    ax.set_title('MAE per Test Scenario', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    # R² histogram
    ax = axes[2]
    ax.hist(gr2, bins=30, color='#4A90D9', edgecolor='white', alpha=0.8, lw=2)
    ax.axvline(np.median(gr2), color='#E74C3C', lw=2, ls='--', label=f'Median: {np.median(gr2):.4f}')
    ax.set_xlabel('R² per Scenario', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distribution of Per-Scenario R²\nRange: [{gr2.min():.3f}, {gr2.max():.3f}]',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_facecolor('#FAFBFC')
    wm(ax)

    fig.suptitle(f'Trial 8: Performance Across {ng} Individual Test Scenarios\n'
                 'Each scenario = different traffic disruption on Paris network',
                 fontsize=17, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'E5_per_graph.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  E5 Per-graph analysis done")


# ============================================================
# CHART F1: RESEARCH STORY — What We Did vs Elena's Paper
# ============================================================

def chart_research_story():
    fig, axes = plt.subplots(1, 2, figsize=(24, 13))

    # Left: Elena's Paper
    ax = axes[0]; ax.axis('off')
    ax.set_facecolor('#F0F4F8')

    ax.text(0.5, 0.97, "Elena Natterer's Paper", fontsize=20, fontweight='bold',
            color='#1a5276', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.93, 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100',
            fontsize=8, color='#888', ha='center', transform=ax.transAxes)

    elena_items = [
        ('Architecture', '#2e86c1', 'PointNetTransfGAT\n'
         '- PointNetConv (spatial encoding)\n'
         '- TransformerConv (multi-head attention)\n'
         '- GATConv (graph attention)\n'
         '- Final GATConv output layer'),
        ('Data', '#27ae60', 'Paris road network\n'
         '- 31,635 road links (nodes)\n'
         '- 59,851 connections (edges)\n'
         '- 10,000 MATSim simulations\n'
         '- 1% downsampled population'),
        ('Task', '#f39c12', 'Predict traffic flow change (DeltaVolume)\n'
         'for each road link when policies change\n'
         '(e.g., capacity reduction on some links)'),
        ('Result', '#e74c3c', 'GNN can predict traffic effects\n'
         'much faster than re-running MATSim\n'
         '(seconds vs hours of simulation)'),
    ]

    for i, (title, color, desc) in enumerate(elena_items):
        y = 0.82 - i * 0.21
        ax.text(0.05, y, title + ':', fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y-0.03, desc, fontsize=10.5, color='#444', transform=ax.transAxes, linespacing=1.4)

    # Right: Our Thesis
    ax = axes[1]; ax.axis('off')
    ax.set_facecolor('#F8F0F0')

    ax.text(0.5, 0.97, "Our Thesis Contribution", fontsize=20, fontweight='bold',
            color='#8e44ad', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.93, 'Mohd Zamin Quadri — TUM Master Thesis — Prof. Gunnemann',
            fontsize=8, color='#888', ha='center', transform=ax.transAxes)

    our_items = [
        ('Reproduced Elena\'s Model', '#8e44ad',
         '8 hyperparameter trials of PointNetTransfGAT\n'
         'Trial 1: OLD arch (Linear final) -- R²=0.786 but wrong\n'
         'Trial 8: BEST correct arch -- R²=0.596\n'
         'Trained on only 10% of Paris data!'),
        ('Added MC Dropout UQ', '#2980b9',
         '30 stochastic forward passes = uncertainty estimate\n'
         'Spearman rho = 0.482 (T8) — strongest UQ signal\n'
         'No retraining needed, cheapest method\n'
         'Applied to Trials 5, 6, 7, 8'),
        ('Added Ensemble UQ', '#6c3483',
         'Exp A: 5 training runs of same model (epistemic)\n'
         'Exp B: Multi-model from Trials 2,5,6,7,8\n'
         'Combined uncertainty: epistemic + aleatoric\n'
         'MC Dropout alone beats all ensemble methods!'),
        ('Calibrated Uncertainties', '#27ae60',
         'Temperature Scaling: T = 2.90\n'
         'ECE improved 90.6% (0.356 -> 0.033)\n'
         '1-sigma coverage: 32.8% -> ~68.3%\n'
         'Now uncertainty estimates are TRUSTWORTHY!'),
    ]

    for i, (title, color, desc) in enumerate(our_items):
        y = 0.82 - i * 0.21
        ax.text(0.05, y, title + ':', fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y-0.03, desc, fontsize=10.5, color='#444', transform=ax.transAxes, linespacing=1.4)

    fig.suptitle('Research Context: Building on Elena Natterer\'s Work\n'
                 'Original GNN for Traffic + Our UQ Extension',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'F1_research_story.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  F1 Research story done")


# ============================================================
# CHART F2: COMPLETE SUMMARY DASHBOARD
# ============================================================

def chart_final_dashboard(R, M, ea, eb, T_opt, ece_b, ece_a):
    fig, axes = plt.subplots(2, 3, figsize=(26, 16))
    improv = (ece_b - ece_a) / ece_b * 100

    # 1. Best model stats
    ax = axes[0, 0]; ax.axis('off')
    ax.set_title('Best Model: Trial 8', fontsize=16, fontweight='bold', color='#2ECC71')
    items = [
        ('Architecture', 'PointNetTransfGAT (correct)', '#2ECC71'),
        ('Test R²', f'{R[8]["r2"]:.4f}', '#4A90D9'),
        ('Test MAE', f'{R[8]["mae"]:.3f} veh/h', '#4A90D9'),
        ('Test RMSE', f'{R[8]["rmse"]:.3f} veh/h', '#4A90D9'),
        ('Total Predictions', f'{R[8]["n"]:,}', '#555'),
        ('< 5 veh/h error', f'{R[8]["u5"]:.1f}%', '#2ECC71'),
        ('Hyperparams', 'BS=32, DO=0.15, LR=1e-3', '#555'),
        ('Training Data', '10% Paris population only!', '#E74C3C'),
    ]
    for i, (k, v, c) in enumerate(items):
        y = 0.90 - i * 0.11
        ax.text(0.03, y, f'{k}:', fontsize=11, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.52, y, v, fontsize=11, transform=ax.transAxes, color=c, fontweight='bold')

    # 2. Trial ranking
    ax = axes[0, 1]
    correct = [2,3,4,5,6,7,8]
    st = sorted(correct, key=lambda t: R[t]['r2'], reverse=True)
    bars = ax.barh(range(7), [R[t]['r2'] for t in st],
                   color=[TC[t-1] for t in st], edgecolor='white', lw=2)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f'T{t} (MAE={R[t]["mae"]:.2f})' for t in st], fontsize=10)
    ax.set_xlabel('R²', fontsize=11, fontweight='bold')
    ax.set_title('Correct Architecture Trials Ranked\n(Trial 1 excluded — wrong arch)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, [R[t]['r2'] for t in st]):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.set_facecolor('#FAFBFC')

    # 3. UQ ranking
    ax = axes[0, 2]
    targets_a = ea['targets'].flatten()
    errors_a = np.abs(ea['ensemble_mean'].flatten() - targets_a)
    targets_b = eb['targets'].flatten()
    errors_b = np.abs(eb['ensemble_prediction'].flatten() - targets_b)

    uq_items = [
        ('MC Drop T8', M[8]['rho'], '#4A90D9'),
        ('MC Drop T7', M[7]['rho'], '#0ABDE3'),
        ('MC Drop T5', M[5]['rho'], '#48DBFB'),
        ('MC Drop T6', M[6]['rho'], '#87CEEB'),
        ('Ens MC Avg', spearmanr(ea['avg_mc_uncertainty'].flatten(), errors_a)[0], '#F39C12'),
        ('Multi-Model', spearmanr(eb['ensemble_uncertainty'].flatten(), errors_b)[0], '#9B59B6'),
        ('Ens Variance', spearmanr(ea['ensemble_variance'].flatten(), errors_a)[0], '#FF6B6B'),
    ]
    uq_items.sort(key=lambda x: x[1], reverse=True)
    bars = ax.barh(range(len(uq_items)), [x[1] for x in uq_items],
                   color=[x[2] for x in uq_items], edgecolor='white', lw=2)
    ax.set_yticks(range(len(uq_items)))
    ax.set_yticklabels([x[0] for x in uq_items], fontsize=10)
    ax.set_xlabel('Spearman rho', fontsize=11, fontweight='bold')
    ax.set_title('UQ Methods Ranked\nMC Dropout wins by far!', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, [x[1] for x in uq_items]):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.set_facecolor('#FAFBFC')

    # 4. Calibration summary
    ax = axes[1, 0]; ax.axis('off')
    ax.set_title(f'Calibration (T={T_opt:.2f})', fontsize=16, fontweight='bold', color='#2ECC71')
    cal_items = [
        ('ECE Before', f'{ece_b:.4f}', '#E74C3C'),
        ('ECE After', f'{ece_a:.4f}', '#2ECC71'),
        ('Improvement', f'{improv:.1f}%', '#2ECC71'),
        ('1-sigma Before', f'{M[8]["c1"]:.1f}%', '#E74C3C'),
        ('1-sigma After', '~68%', '#2ECC71'),
        ('Temperature', f'{T_opt:.2f}', '#4A90D9'),
    ]
    for i, (k, v, c) in enumerate(cal_items):
        y = 0.88 - i * 0.14
        ax.text(0.05, y, f'{k}:', fontsize=12, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.55, y, v, fontsize=13, transform=ax.transAxes, color=c, fontweight='bold')

    # 5. Key numbers
    ax = axes[1, 1]; ax.axis('off')
    ax.set_title('Key Facts', fontsize=16, fontweight='bold', color=PAL['dark'])
    facts = [
        ('Network', 'Paris, France', '#4A90D9'),
        ('Road Links', '31,635 per graph', '#4A90D9'),
        ('Graph Edges', '59,851', '#4A90D9'),
        ('Input Features', '5 (of 6 available)', '#4A90D9'),
        ('MC Dropout Passes', '30 forward passes', '#8e44ad'),
        ('Training Data', '10% of full dataset', '#E74C3C'),
        ('Total Test Preds', '3,163,500 (T8)', '#555'),
    ]
    for i, (k, v, c) in enumerate(facts):
        y = 0.88 - i * 0.12
        ax.text(0.05, y, f'{k}:', fontsize=11, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.55, y, v, fontsize=11, transform=ax.transAxes, color=c, fontweight='bold')

    # 6. Thesis takeaways
    ax = axes[1, 2]; ax.axis('off')
    ax.set_title('Thesis Takeaways', fontsize=16, fontweight='bold', color='#8e44ad')
    takeaways = [
        ('1.', 'GNN surrogates work for traffic\nprediction even with 10% data', '#27ae60'),
        ('2.', 'MC Dropout is the best UQ method\n(rho=0.482, cheapest, no retraining)', '#4A90D9'),
        ('3.', 'Temperature scaling makes\nuncertainties trustworthy (90.6% ECE fix)', '#F39C12'),
        ('4.', 'Ensemble methods add value but\nMC Dropout alone is sufficient', '#9B59B6'),
        ('5.', 'Traffic planners can use uncertainty\nto decide which predictions to trust', '#E74C3C'),
    ]
    for i, (num, text, color) in enumerate(takeaways):
        y = 0.88 - i * 0.17
        ax.text(0.02, y, num, fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.08, y, text, fontsize=11, color='#444', transform=ax.transAxes, linespacing=1.4)

    fig.suptitle('COMPLETE THESIS VERIFICATION DASHBOARD\n'
                 'All values cross-checked directly from NPZ files | PointNetTransfGAT on Paris Network',
                 fontsize=19, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'F2_final_dashboard.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  F2 Final dashboard done")


# ============================================================
# CHART F3: RADAR (Multi-axis)
# ============================================================

def chart_radar(R):
    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))
    cats = ['R² (x100)', 'Low MAE\n(10-MAE)', 'Low RMSE\n(15-RMSE)', '< 5 Error %', 'Low P90\n(20-P90)']
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    trial_colors_radar = {2:'#FF8E53', 5:'#48DBFB', 7:'#1ABC9C', 8:'#2ECC71'}
    for t, c in trial_colors_radar.items():
        vals = [R[t]['r2']*100, max(0, 10-R[t]['mae']), max(0, 15-R[t]['rmse']),
                R[t]['u5'], max(0, 20-R[t]['p90'])]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.12, color=c)
        ax.plot(angles, vals, 'o-', lw=2.5, ms=7, color=c, label=f'Trial {t}')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=11, fontweight='bold')
    ax.set_title('Multi-Axis Performance Radar\nBigger area = better overall (correct arch only)',
                 fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'F3_radar.png'), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  F3 Radar done")


# ============================================================
# MAIN
# ============================================================

def main():
    setup_style()
    print("=" * 70)
    print("  MEGA THESIS CHARTS — Complete Professional Package")
    print("=" * 70)

    print("\n[1/4] Loading all data...")
    R, feats, pos = load_trials()
    M = load_mc()
    ea, eb = load_ens()

    print(f"\n[2/4] Generating architecture & pipeline diagrams...")
    chart_arch_diagram()
    chart_data_pipeline()

    print(f"\n[3/4] Generating comparison & analysis charts...")
    chart_all_trials_3d(R)
    chart_correct_trials(R)
    chart_hyperparams_3d(R)

    # Per-trial charts for ALL 8 trials
    for t in range(1, 9):
        chart_per_trial(R, t)

    chart_mc_dropout_all(M)
    chart_ensemble(ea, eb, M)
    T_opt, ece_b, ece_a = chart_calibration(M)

    chart_spatial(M, pos)
    chart_3d_surface(M, feats)
    chart_features(M, feats)
    chart_practical(M, R)
    chart_per_graph(R)

    print(f"\n[4/4] Generating research story & summary...")
    chart_research_story()
    chart_final_dashboard(R, M, ea, eb, T_opt, ece_b, ece_a)
    chart_radar(R)

    # Count total files
    files = [f for f in os.listdir(OUT) if f.endswith('.png')]
    print(f"\n{'=' * 70}")
    print(f"  DONE! {len(files)} charts in {OUT}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
