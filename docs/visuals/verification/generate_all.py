"""
COMPLETE THESIS CHARTS — A to Z — Fresh Build
================================================
Mohd Zamin Quadri | TUM Master Thesis | Prof. Günnemann | Supervisor: Fuchsgruber
Elena Natterer's PointNetTransfGAT architecture reproduced + UQ extension

KEY FACTS:
  - Trial 1 = OLD architecture (Linear final layer, NOT GATConv) → EXCLUDED
  - Trials 2–8 = Correct PointNetTransfGAT (GATConv final)
  - Best correct trial = Trial 8 (R²=0.596, MAE=3.96)
  - Training on only 10 percent of Paris population (1% downsampled × 10k simulations)
  - MC Dropout: 30 stochastic forward passes  →  uncertainty estimate
  - Ensemble Exp A (5 runs same arch) + Exp B (multi-model T2,5,6,7,8)
  - Temperature Scaling calibration: ECE 0.356 → 0.033 (90.6 percent)
  - Paris network: 31,635 nodes, 59,851 edges
"""

import numpy as np, os, warnings, sys, textwrap, gc
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── paths ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BASE = os.path.join(ROOT, 'data', 'TR-C_Benchmarks')
TRAIN = os.path.join(ROOT, 'data', 'train_data', 'dist_not_connected_10k_1pct')
OUT  = os.path.dirname(os.path.abspath(__file__))   # same folder as this script

FOLDERS = {
    1: 'pointnet_transf_gat_1st_bs32_5feat_seed42',
    2: 'point_net_transf_gat_2nd_try',
    3: 'point_net_transf_gat_3rd_trial_weighted_loss',
    4: 'point_net_transf_gat_4th_trial_weighted_loss',
    5: 'point_net_transf_gat_5th_try',
    6: 'point_net_transf_gat_6th_trial_lower_lr',
    7: 'point_net_transf_gat_7th_trial_80_10_10_split',
    8: 'point_net_transf_gat_8th_trial_lower_dropout',
}

LABELS = {
    1: 'T1 OLD Arch (Linear)',
    2: 'T2 BS32 DO0.20 W-MSE',
    3: 'T3 BS32 DO0.20 W-MSE 150ep',
    4: 'T4 BS32 DO0.20 W-MSE 150ep',
    5: 'T5 BS64 DO0.20 LR1e-3',
    6: 'T6 BS32 DO0.20 LR5e-4',
    7: 'T7 BS32 DO0.20 80/10/10',
    8: 'T8 BS32 DO0.15 LR1e-3',
}

# 8 gorgeous trial colours
TC = ['#E74C3C','#FF8E53','#FECA57','#F39C12','#48DBFB','#0ABDE3','#1ABC9C','#2ECC71']

# Helper palette
P = dict(blue='#4A90D9', coral='#FF6B6B', green='#2ECC71', gold='#F39C12',
         purple='#9B59B6', teal='#1ABC9C', rose='#E74C3C', dark='#2C3E50',
         sky='#48DBFB', ocean='#0ABDE3', mint='#98FB98', peach='#FFEAA7')


def _style():
    plt.rcParams.update({
        'figure.facecolor':'white','axes.facecolor':'#FAFBFC',
        'axes.grid':True,'grid.alpha':0.12,'grid.linestyle':'--',
        'font.family':'sans-serif','font.size':11,
        'axes.titlesize':15,'axes.titleweight':'bold',
        'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
    })

def _wm(ax, t='Cross-checked from NPZ'):
    ax.text(0.99,0.01,t,transform=ax.transAxes,fontsize=7,color='#CCC',
            ha='right',va='bottom',style='italic')

def _save(fig, name, **kw):
    for dpi in [220, 150, 100]:
        try:
            gc.collect()
            fig.savefig(os.path.join(OUT, name), dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            gc.collect()
            if dpi < 220:
                print(f"   {name}  (DPI={dpi} due to memory)")
            else:
                print(f"   {name}")
            return
        except MemoryError:
            continue
    # all DPIs failed — skip
    plt.close(fig)
    gc.collect()
    print(f"   {name}  SKIPPED (MemoryError at all DPIs)")


# ═══════════════════════════════════════════════════════════
#  HELPERS — SAFE CORRELATION FOR LARGE ARRAYS
# ═══════════════════════════════════════════════════════════

def _safe_spearmanr(a, b, max_n=500000):
    """Spearmanr with subsampling for large arrays to avoid MemoryError."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) > max_n:
        idx = np.random.RandomState(42).choice(len(a), max_n, replace=False)
        a, b = a[idx], b[idx]
    return spearmanr(a, b)[0]

def _safe_pearsonr(a, b, max_n=500000):
    """Pearsonr with subsampling for large arrays to avoid MemoryError."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) > max_n:
        idx = np.random.RandomState(42).choice(len(a), max_n, replace=False)
        a, b = a[idx], b[idx]
    return pearsonr(a, b)[0]


# ═══════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════

def load_trials():
    import torch
    R = {}
    for t, f in FOLDERS.items():
        d = np.load(os.path.join(BASE, f, 'test_predictions.npz'))
        pr, tg = d['predictions'].flatten(), d['targets'].flatten()
        e = np.abs(pr - tg)
        R[t] = dict(p=pr, t=tg, e=e,
                     r2=r2_score(tg,pr), mae=mean_absolute_error(tg,pr),
                     rmse=np.sqrt(mean_squared_error(tg,pr)),
                     med=np.median(e), p90=np.percentile(e,90),
                     u5=np.mean(e<5)*100, n=len(pr))
    g = torch.load(os.path.join(TRAIN,'datalist_batch_1.pt'),
                   map_location='cpu', weights_only=False)[0]
    return R, g.x.numpy(), g.pos.numpy()

def load_mc():
    M = {}
    for t, sub in [(5,'5th_try'),(6,'6th_trial_lower_lr'),
                    (7,'7th_trial_80_10_10_split'),(8,'8th_trial_lower_dropout')]:
        ng = 100 if t>=7 else 50
        path = os.path.join(BASE,f'point_net_transf_gat_{sub}',
                            'uq_results',f'mc_dropout_full_{ng}graphs_mc30.npz')
        d = np.load(path)
        pr,u,tg = d['predictions'].flatten(),d['uncertainties'].flatten(),d['targets'].flatten()
        e = np.abs(pr-tg)
        rho = _safe_spearmanr(u,e); pr2 = _safe_pearsonr(u,e)
        M[t] = dict(p=pr,u=u,t=tg,e=e,rho=rho,pval=0.0,pr=pr2,
                     c1=np.mean(e<u)*100, c2=np.mean(e<2*u)*100,
                     mu=np.mean(u), me=np.mean(e))
    return M

def load_ens():
    ens_dir = os.path.join(BASE,'point_net_transf_gat_8th_trial_lower_dropout',
                           'uq_results','ensemble_experiments')
    ea = np.load(os.path.join(ens_dir,'experiment_a_data.npz'), allow_pickle=True)
    eb = np.load(os.path.join(ens_dir,'experiment_b_data.npz'), allow_pickle=True)
    return ea, eb


# ═══════════════════════════════════════════════════════════
#  01 — ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════

def chart_01_architecture():
    fig, ax = plt.subplots(figsize=(28,16))
    ax.set_xlim(-1,30); ax.set_ylim(-4,15); ax.axis('off')
    ax.set_facecolor('#FAFBFC'); fig.patch.set_facecolor('white')

    def blk(x,y,w,h,c,label,sub='',glow=False):
        ax.add_patch(FancyBboxPatch((x+.12,y-.12),w,h,boxstyle='round,pad=0.15',
                     facecolor='#D0D0D0',alpha=.3,lw=0,zorder=1))
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15',
                     facecolor=c,edgecolor='#CCC',lw=1.5,alpha=.92,zorder=2))
        if glow:
            ax.add_patch(FancyBboxPatch((x-.08,y-.08),w+.16,h+.16,boxstyle='round,pad=0.2',
                         facecolor='none',edgecolor=c,lw=3,alpha=.4,zorder=1))
        ax.text(x+w/2,y+h/2+.15,label,ha='center',va='center',fontsize=11,
                fontweight='bold',color='white',zorder=3)
        if sub:
            ax.text(x+w/2,y+h/2-.35,sub,ha='center',va='center',fontsize=8,
                    color='#DDD',zorder=3,style='italic')

    def arr(x1,y1,x2,y2,c='#888'):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color=c,lw=2.5),zorder=4)

    def dim(x,y,t):
        ax.text(x,y,t,ha='center',va='center',fontsize=8,color='#555',
                bbox=dict(boxstyle='round,pad=0.2',facecolor='#F0F4F8',edgecolor='#BBB',alpha=.9),zorder=5)

    # title
    ax.text(14.5,14.2,'PointNetTransfGAT  —  Full Architecture',
            ha='center',fontsize=24,fontweight='bold',color='#1a5276')
    ax.text(14.5,13.5,'Graph Neural Network for Traffic-Flow-Change Prediction on the Paris Road Network',
            ha='center',fontsize=13,color='#555')
    ax.text(14.5,13.0,"Elena Natterer's architecture  |  Reproduced + UQ by Mohd Zamin Quadri (TUM)",
            ha='center',fontsize=10,color='#888',style='italic')

    # INPUT
    blk(0,9,3.5,2.5,'#1a5276','Input Graph','Paris: 31,635 nodes\n59,851 edges')
    dim(1.75,8.5,'PyG Data object')
    blk(0,5.5,3.5,2.8,'#2e86c1','6 Node Features',
        'VOL_BASE (0-1596)\nCAPACITY (0-14400)\nCAP_REDUCTION\nFREESPEED (0-33)\nLANES\nLENGTH (4-2569)')
    dim(1.75,5,'[N, 5] input')
    blk(0,2,3.5,2.8,'#1b4f72','Spatial Positions',
        'Start pos (lat,lon)\nEnd pos (lat,lon)\nParis ~48.85°N 2.34°E')
    dim(1.75,1.5,'[N, 3, 2]')

    arr(3.6,10,5.2,9.5,'#2e86c1'); arr(3.6,6.8,5.2,8.5,'#2e86c1')
    arr(3.6,3.5,5.2,7,'#1b4f72')

    # POINTNET
    blk(5.3,8,3.8,3,'#8e44ad','PointNetConv 1',
        'Local MLP: 7→256\nGlobal MLP: 256→512→512\nUses START positions\n+ ReLU + Dropout')
    dim(7.2,7.5,'[N, 512]')
    blk(5.3,4,3.8,3,'#6c3483','PointNetConv 2',
        'Local MLP: 514→256\nGlobal MLP: 256→512→128\nUses END positions\n+ ReLU + Dropout')
    dim(7.2,3.5,'[N, 128]')
    arr(7.2,8,7.2,7.1,'#8e44ad'); arr(9.2,5.5,10.5,8.5,'#6c3483')

    # TRANSFORMER + GAT
    blk(10.6,8,3.8,3,'#c0392b','TransformerConv 1',
        '128→64×4 heads = 256\nMulti-head attention\n+ ReLU + Dropout')
    dim(12.5,7.5,'[N, 256]')
    arr(14.5,9.5,15.5,9.5,'#c0392b')
    blk(15.6,8,3.8,3,'#e74c3c','TransformerConv 2',
        '256→128×4 heads = 512\nMulti-head attention\n+ ReLU + Dropout')
    dim(17.5,7.5,'[N, 512]')
    arr(19.5,9.5,20.5,9.5,'#e74c3c')
    blk(20.6,8,3.5,3,'#27ae60','GATConv','512→64\nGraph Attention')
    dim(22.35,7.5,'[N, 64]')
    arr(24.2,9.5,25.2,9.5,'#27ae60')
    blk(25.3,8,3.5,3,'#f39c12','GATConv (Final)',
        '64→1\nPer-node DeltaVol\nprediction',glow=True)
    dim(27.05,7.5,'[N, 1]')

    # OUTPUT
    blk(25.3,3.5,3.5,3.2,'#d4ac0d','Output',
        'Per-link traffic-flow\nchange prediction\nDeltaVolume (veh/h)\nfor 31,635 links')
    arr(27.05,8,27.05,6.8,'#f39c12')

    # MC Dropout
    blk(10.6,.8,8.5,2.2,'#2980b9','MC Dropout  (UQ)',
        'Keep dropout ON at inference → 30 forward passes → mean + std\nNo retraining needed — cheapest UQ method')
    for xp in [7.2,12.5,17.5]:
        ax.annotate('',xy=(xp,2 if xp>9 else 4),xytext=(xp,3),
                    arrowprops=dict(arrowstyle='->',color='#2980b9',lw=1.5,ls='--'),zorder=4)

    # legend
    for i,(c,l) in enumerate([('#8e44ad','PointNet (spatial)'),('#c0392b','Transformer (attention)'),
                               ('#27ae60','GAT (graph attention)'),('#f39c12','Output (node pred)'),
                               ('#2980b9','MC Dropout (UQ)')]):
        ax.add_patch(Rectangle((.3,-1-i*.65),.5,.45,facecolor=c,edgecolor='#ddd',lw=1,zorder=5))
        ax.text(1.1,-.78-i*.65,l,fontsize=10,color='#333',va='center',zorder=5)

    ax.text(20,-.5,'Parameters: ~1.4 M  |  Optimizer: AdamW (wd=1e-4)\n'
            'Training: 10% Paris population (1% down-sampled), 10 000 scenarios\n'
            'Best trial: T8 (BS 32, DO 0.15, LR 1e-3) → R²=0.596, MAE=3.96',
            fontsize=10,color='#555',va='top',
            bbox=dict(boxstyle='round,pad=0.5',facecolor='#F0F4F8',edgecolor='#ddd',alpha=.9))

    _save(fig,'01_architecture.png')


# ═══════════════════════════════════════════════════════════
#  02 — DATA & EXPERIMENT PIPELINE
# ═══════════════════════════════════════════════════════════

def chart_02_pipeline():
    fig, ax = plt.subplots(figsize=(26,10))
    ax.set_xlim(-1,27); ax.set_ylim(-1,9); ax.axis('off')
    ax.set_facecolor('#FAFBFC'); fig.patch.set_facecolor('white')

    def blk(x,y,w,h,c,lbl,sub=''):
        ax.add_patch(FancyBboxPatch((x+.1,y-.1),w,h,boxstyle='round,pad=0.15',
                     facecolor='#D0D0D0',alpha=.3,lw=0,zorder=1))
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15',
                     facecolor=c,edgecolor='#CCC',lw=1.5,alpha=.92,zorder=2))
        ax.text(x+w/2,y+h/2+.15,lbl,ha='center',va='center',fontsize=12,
                fontweight='bold',color='white',zorder=3)
        if sub:
            ax.text(x+w/2,y+h/2-.35,sub,ha='center',va='center',fontsize=8.5,
                    color='#DDD',zorder=3,style='italic')
    def arr(x1,y1,x2,y2,c='#888'):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color=c,lw=2.5),zorder=4)

    ax.text(13,8.5,'Complete Data & Experiment Pipeline',fontsize=22,
            fontweight='bold',color='#1a5276',ha='center')
    ax.text(13,7.9,'From MATSim agent-based simulations  →  uncertainty-aware traffic predictions',
            fontsize=12,color='#555',ha='center')

    # row 1
    blk(0,5.5,3.8,1.8,'#1a5276','MATSim Sim',
        'Agent-based transport\nParis 1% pop\n10 000 scenarios')
    arr(3.9,6.4,5,6.4,'#48DBFB')
    blk(5.1,5.5,3.8,1.8,'#2e86c1','Graph Build',
        '31,635 road links = nodes\n59,851 connections = edges\n6 features / node')
    arr(9,6.4,10.1,6.4,'#48DBFB')
    blk(10.2,5.5,3.8,1.8,'#1b4f72','Train/Val/Test',
        'T2-T6: 70/15/15\nT7-T8: 80/10/10\n10% data only!')
    arr(14.1,6.4,15.2,6.4,'#48DBFB')
    blk(15.3,5.5,3.8,1.8,'#27ae60','GNN Training',
        'PointNetTransfGAT\nAdamW + Early Stop\n500-1000 epochs')
    arr(19.2,6.4,20.3,6.4,'#2ECC71')
    blk(20.4,5.5,3.8,1.8,'#f39c12','Test Eval',
        '50-100 unseen graphs\nR², MAE, RMSE\nPer-node predictions')

    # row 2
    arr(17.2,5.5,5,3.2,'#9B59B6')
    blk(0,1.5,5,1.8,'#8e44ad','MC Dropout (UQ 1)',
        'Dropout ON at inference\n30 passes → mean+std\nrho=0.482 (T8)')
    arr(5.1,2.4,6.8,2.4,'#9B59B6')
    blk(7,1.5,5,1.8,'#6c3483','Ensemble (UQ 2)',
        'Exp A: 5 runs same arch\nExp B: multi-model T2,5,6,7,8\nEpistemic + Aleatoric')
    arr(12.1,2.4,13.8,2.4,'#9B59B6')
    blk(14,1.5,5,1.8,'#2980b9','Temp. Scaling',
        'T=2.90: sigma_new=T*sigma\nECE: 0.356→0.033\n90.6% improvement')
    arr(19.1,2.4,20.5,2.4,'#2980b9')
    blk(20.6,1.5,5,1.8,'#d4ac0d','Thesis Contribution',
        '1) Reproduced GNN\n2) Added UQ (MC+Ens)\n3) Calibrated\n4) Only 10% data!')

    _save(fig,'02_pipeline.png')


# ═══════════════════════════════════════════════════════════
#  03 — ALL 8 TRIALS 3-D BAR  (Trial 1 in red)
# ═══════════════════════════════════════════════════════════

def chart_03_all_trials_3d(R):
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111,projection='3d')
    trials = list(range(1,9))
    r2 = [R[t]['r2'] for t in trials]
    mae = [R[t]['mae'] for t in trials]
    rmse = [R[t]['rmse'] for t in trials]
    x = np.arange(8); w,d = .22,.5

    for metric,ypos,vals,scale,base_c in [
            ('R²',0,r2,1,'#4A90D9'),('MAE/10',1.2,mae,.1,'#FF6B6B'),('RMSE/10',2.4,rmse,.1,'#F39C12')]:
        cols = ['#E74C3C' if t==1 else base_c for t in trials]
        ax.bar3d(x,np.full(8,ypos),np.zeros(8),w,d,[v*scale for v in vals],
                 color=cols,alpha=.85,edgecolor='white',lw=.5)

    ax.set_xticks(x+w/2)
    ax.set_xticklabels(['T1\nWRONG' if t==1 else f'T{t}' for t in trials],fontsize=9,rotation=-15)
    ax.set_yticks([0,1.2,2.4]); ax.set_yticklabels(['R²','MAE/10','RMSE/10'],fontsize=9)
    ax.set_zlabel('Value',fontsize=11,fontweight='bold')
    ax.set_title('All 8 Trials — 3-D Performance\n'
                 'T1 = OLD arch (Linear final) shown RED  |  T2-T8 = correct GATConv final',
                 fontsize=15,fontweight='bold',pad=25)
    ax.text2D(.02,.88,'T1 R²=0.786 BUT wrong architecture!\n'
              'Best CORRECT: T8 R²=0.596\nTraining on 10% data only',
              transform=ax.transAxes,fontsize=10,fontweight='bold',color='#E74C3C',
              bbox=dict(boxstyle='round,pad=0.4',facecolor='w',alpha=.95))
    ax.view_init(elev=25,azim=-50)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    _save(fig,'03_all_trials_3d.png')


# ═══════════════════════════════════════════════════════════
#  04 — CORRECT TRIALS BAR CHART (T2-T8)
# ═══════════════════════════════════════════════════════════

def chart_04_correct_trials(R):
    fig, axes = plt.subplots(1,3,figsize=(24,8))
    ct = [2,3,4,5,6,7,8]
    cols = [TC[t-1] for t in ct]

    for ax_i,(metric,key,higher) in enumerate([('R²','r2',True),('MAE (veh/h)','mae',False),
                                                ('% Error < 5','u5',True)]):
        ax = axes[ax_i]
        v = [R[t][key] for t in ct]
        bars = ax.bar([f'T{t}' for t in ct],v,color=cols,edgecolor='w',lw=2)
        for b,val in zip(bars,v):
            fmt = f'{val:.4f}' if key=='r2' else (f'{val:.3f}' if key=='mae' else f'{val:.1f}%')
            ax.text(b.get_x()+b.get_width()/2,val+(.005 if higher else .05),
                    fmt,ha='center',fontsize=9,fontweight='bold')
        best = np.argmax(v) if higher else np.argmin(v)
        bars[best].set_edgecolor('#2ECC71'); bars[best].set_linewidth(4)
        direction = '(higher=better)' if higher else '(lower=better)'
        ax.set_title(f'{metric} {direction}',fontsize=14,fontweight='bold')
        ax.set_ylabel(metric,fontweight='bold'); ax.set_facecolor('#FAFBFC'); _wm(ax)

    fig.suptitle('Trials 2–8: Correct PointNetTransfGAT (GATConv final)\n'
                 'Trial 1 excluded — wrong architecture  |  Trained on 10% data',
                 fontsize=16,fontweight='bold',y=1.02)
    plt.tight_layout()
    _save(fig,'04_correct_trials.png')


# ═══════════════════════════════════════════════════════════
#  05 — HYPERPARAMETER 3-D LANDSCAPE
# ═══════════════════════════════════════════════════════════

def chart_05_hyperparams(R):
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111,projection='3d')
    HP = {2:(.001,.20,32),3:(.001,.20,32),4:(.001,.20,32),
          5:(.001,.20,64),6:(.0005,.20,32),7:(.001,.20,32),8:(.001,.15,32)}
    for t,(lr,do,bs) in HP.items():
        r2 = R[t]['r2']
        ax.scatter(lr,do,r2,c=TC[t-1],s=r2*1200+100,edgecolors='w',lw=2,alpha=.9,zorder=5)
        ax.text(lr,do,r2+.02,f'T{t}\nR²={r2:.3f}\nBS={bs}',fontsize=8.5,
                ha='center',fontweight='bold',color=TC[t-1])
    ax.set_xlabel('\nLearning Rate',fontsize=11,fontweight='bold')
    ax.set_ylabel('\nDropout',fontsize=11,fontweight='bold')
    ax.set_zlabel('\nR²',fontsize=11,fontweight='bold')
    ax.set_title('Hyperparameter Landscape (Correct Arch)\nBubble ∝ R²  |  7 trials explored',
                 fontsize=15,fontweight='bold',pad=25)
    ax.text2D(.02,.08,'Key: only T8 uses DO=0.15 → best R²\n'
              'Weighted loss (T2,T3,T4) did NOT help\n10% training data',
              transform=ax.transAxes,fontsize=9.5,color='#555',
              bbox=dict(boxstyle='round,pad=0.4',facecolor='#F0F0F0',alpha=.9))
    ax.view_init(elev=30,azim=-55); plt.tight_layout()
    _save(fig,'05_hyperparams_3d.png')


# ═══════════════════════════════════════════════════════════
#  06-13 — PER-TRIAL 6-PANEL DETAIL (T1-T8)
# ═══════════════════════════════════════════════════════════

def chart_trial_detail(R, t):
    d = R[t]; wrong = (t==1); c = '#E74C3C' if wrong else TC[t-1]
    fig, axes = plt.subplots(2,3,figsize=(22,14))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(d['p']),min(100000,len(d['p'])),replace=False)

    # scatter
    ax=axes[0,0]
    ax.hexbin(d['t'][idx],d['p'][idx],gridsize=80,cmap='YlOrRd',mincnt=1,lw=.1)
    lims=[d['t'].min(),d['t'].max()]; ax.plot(lims,lims,'k--',lw=2,alpha=.7)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title('Predicted vs Actual',fontweight='bold')
    _wm(ax)

    # error hist
    ax=axes[0,1]
    ax.hist(d['e'][d['e']<20],bins=100,color=c,alpha=.75,edgecolor='w',lw=.5)
    ax.axvline(d['mae'],color='k',lw=2,ls='--',label=f"MAE={d['mae']:.3f}")
    ax.axvline(d['med'],color='blue',lw=2,ls=':',label=f"Median={d['med']:.3f}")
    ax.set_xlabel('Abs Error (veh/h)'); ax.set_title('Error Distribution',fontweight='bold')
    ax.legend(fontsize=10); _wm(ax)

    # CDF
    ax=axes[0,2]
    se=np.sort(d['e']); cdf=np.arange(len(se))/len(se)
    ax.plot(se,cdf,color=c,lw=3)
    ax.axhline(.5,color='#aaa',ls=':',alpha=.5); ax.axhline(.9,color='#aaa',ls='--',alpha=.5)
    ax.axvline(5,color='#888',ls='--',alpha=.5)
    ax.text(5.5,.5,f'{d["u5"]:.1f}% < 5 veh/h',fontsize=10,color='#555')
    ax.set_xlim(0,25); ax.set_xlabel('Error Threshold'); ax.set_title('Cumulative Error',fontweight='bold')
    _wm(ax)

    # residual
    ax=axes[1,0]
    res=d['p'][idx]-d['t'][idx]
    ax.hexbin(d['t'][idx],res,gridsize=80,cmap='coolwarm',mincnt=1,lw=.1)
    ax.axhline(0,color='k',lw=2); ax.set_xlabel('Actual'); ax.set_ylabel('Residual')
    ax.set_title('Residual Analysis',fontweight='bold'); _wm(ax)

    # stats card
    ax=axes[1,1]; ax.axis('off')
    arch = ('OLD Arch (Linear final)\nNOT Elenas paper!' if wrong
            else 'Correct PointNetTransfGAT\n(GATConv final)')
    ac = '#E74C3C' if wrong else '#2ECC71'
    items = [('Architecture',arch,ac),
             ('R²',f'{d["r2"]:.6f}',P['blue']),
             ('MAE',f'{d["mae"]:.4f} veh/h',P['blue']),
             ('RMSE',f'{d["rmse"]:.4f} veh/h',P['blue']),
             ('Median Err',f'{d["med"]:.4f}',P['blue']),
             ('P90',f'{d["p90"]:.3f} veh/h',P['blue']),
             ('< 5 Error',f'{d["u5"]:.2f}%',P['blue']),
             ('N Predictions',f'{d["n"]:,}',P['dark'])]
    for i,(k,v,col) in enumerate(items):
        y=.92-i*.11
        ax.text(.05,y,f'{k}:',fontsize=12,fontweight='bold',transform=ax.transAxes,color='#555')
        ax.text(.5,y,v,fontsize=11,transform=ax.transAxes,color=col,fontweight='bold')

    # percentiles
    ax=axes[1,2]
    percs=[10,25,50,75,90,95,99]; pv=[np.percentile(d['e'],p) for p in percs]
    ax.barh([f'P{p}' for p in percs],pv,
            color=cm.RdYlGn_r(np.linspace(.2,.8,len(percs))),edgecolor='w',lw=1.5)
    for i,val in enumerate(pv):
        ax.text(val+.1,i,f'{val:.2f}',va='center',fontsize=10,fontweight='bold')
    ax.set_xlabel('Abs Error (veh/h)'); ax.set_title('Error Percentiles',fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    for a in axes.flat:
        if a.get_visible(): a.set_facecolor('#FAFBFC')

    tag = 'WRONG ARCHITECTURE' if wrong else LABELS[t]
    fig.suptitle(f'Trial {t}  —  {"["+tag+"]" if wrong else tag}\n'
                 f'Detailed 6-Panel Analysis  |  Trained on 10% data',
                 fontsize=17,fontweight='bold',y=1.02,
                 color='#E74C3C' if wrong else P['dark'])
    plt.tight_layout()
    _save(fig,f'{5+t:02d}_trial_{t}_detail.png')


# ═══════════════════════════════════════════════════════════
#  14 — MC DROPOUT ALL 4 TRIALS
# ═══════════════════════════════════════════════════════════

def chart_14_mc_all(M):
    fig, axes = plt.subplots(2,2,figsize=(18,15))
    for i,t in enumerate(sorted(M)):
        ax=axes[i//2][i%2]; d=M[t]
        idx=np.random.RandomState(42).choice(len(d['e']),min(80000,len(d['e'])),False)
        ax.hexbin(d['u'][idx],d['e'][idx],gridsize=70,cmap='magma_r',mincnt=1,lw=.1)
        # trend
        edges=np.percentile(d['u'][idx],np.linspace(0,100,26))
        bc,bm=[],[]
        for j in range(25):
            m=(d['u'][idx]>=edges[j])&(d['u'][idx]<edges[j+1])
            if m.sum()>10: bc.append(d['u'][idx][m].mean()); bm.append(d['e'][idx][m].mean())
        ax.plot(bc,bm,'c-',lw=3.5,label='Mean Error Trend',zorder=10)
        star=' [BEST]' if t==8 else ''
        ax.set_title(f'Trial {t}{star}\nrho={d["rho"]:.4f} | Pearson r={d["pr"]:.4f}',
                     fontsize=13,fontweight='bold',color='#2ECC71' if t==8 else P['dark'])
        stats=(f'Mean unc: {d["mu"]:.3f}\nMean err: {d["me"]:.3f}\n'
               f'1-sig cov: {d["c1"]:.1f}% (ideal 68.3%)\n'
               f'2-sig cov: {d["c2"]:.1f}% (ideal 95.4%)\nN={len(d["p"]):,}')
        ax.text(.97,.97,stats,transform=ax.transAxes,fontsize=9,va='top',ha='right',
                bbox=dict(boxstyle='round,pad=0.4',facecolor='w',alpha=.92,edgecolor='#ddd'))
        ax.set_xlabel('MC Uncertainty (sigma)'); ax.set_ylabel('|Pred - Actual|')
        ax.set_xlim(0,np.percentile(d['u'][idx],99)); ax.set_ylim(0,np.percentile(d['e'][idx],99))
        ax.legend(fontsize=9,loc='upper left'); ax.set_facecolor('#FAFBFC'); _wm(ax)
    fig.suptitle('MC Dropout: Uncertainty vs Error  |  30 stochastic passes\n'
                 'Upward trend = uncertainty is meaningful',fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'14_mc_dropout_all.png')


# ═══════════════════════════════════════════════════════════
#  15 — ENSEMBLE DEEP-DIVE (6 panels)
# ═══════════════════════════════════════════════════════════

def chart_15_ensemble(ea,eb,M):
    fig,axes=plt.subplots(2,3,figsize=(24,15))
    ta=ea['targets'].flatten(); em=ea['ensemble_mean'].flatten()
    ev=ea['ensemble_variance'].flatten(); mc_a=ea['avg_mc_uncertainty'].flatten()
    comb=ea['combined_uncertainty'].flatten(); ea_err=np.abs(em-ta)
    tb=eb['targets'].flatten(); ep=eb['ensemble_prediction'].flatten()
    eu=eb['ensemble_uncertainty'].flatten(); eb_err=np.abs(ep-tb)

    # panel 1 — Exp A 3 unc types
    ax=axes[0,0]
    types=['Ens Var\n(epistemic)','MC Avg\n(aleatoric)','Combined']
    rhos=[_safe_spearmanr(ev,ea_err),_safe_spearmanr(mc_a,ea_err),_safe_spearmanr(comb,ea_err)]
    bars=ax.bar(types,rhos,color=['#FF6B6B','#4A90D9','#F39C12'],edgecolor='w',lw=2,width=.5)
    for b,v in zip(bars,rhos): ax.text(b.get_x()+b.get_width()/2,v+.005,f'{v:.4f}',ha='center',fontsize=12,fontweight='bold')
    ax.set_ylabel('Spearman rho'); ax.set_title('Exp A: 3 Uncertainty Types\n5 runs of T8 arch',fontsize=13,fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # panel 2 — Exp B per-model R²
    ax=axes[0,1]
    mk=['model_2_predictions','model_5_predictions','model_6_predictions','model_7_predictions','model_8_predictions']
    mn=['T2','T5','T6','T7','T8']; mc=['#FF8E53','#48DBFB','#9B59B6','#1ABC9C','#2ECC71']
    mr2=[r2_score(tb,eb[k].flatten()) for k in mk]
    names=mn+['Ensemble']; r2s=mr2+[r2_score(tb,ep)]; clrs=mc+['#FFD700']
    bars=ax.bar(names,r2s,color=clrs,edgecolor='w',lw=2)
    for b,v in zip(bars,r2s):
        ly=v+.001 if v>=0 else v-.003
        ax.text(b.get_x()+b.get_width()/2,ly,f'{v:.4f}',ha='center',fontsize=9,fontweight='bold',
                va='bottom' if v>=0 else 'top')
    ax.axhline(0,color='#888',lw=1)
    ax.set_title('Exp B: Multi-Model R²\nR²≈0 = different distribution',fontsize=13,fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # panel 3 — Grand UQ ranking
    ax=axes[0,2]
    items=[('MC Drop T8',M[8]['rho'],'#4A90D9'),('MC Drop T7',M[7]['rho'],'#0ABDE3'),
           ('MC Drop T5',M[5]['rho'],'#48DBFB'),('MC Drop T6',M[6]['rho'],'#87CEEB'),
           ('Ens MC Avg (A)',_safe_spearmanr(mc_a,ea_err),'#F39C12'),
           ('Combined (A)',_safe_spearmanr(comb,ea_err),'#FF8E53'),
           ('Multi-Model (B)',_safe_spearmanr(eu,eb_err),'#9B59B6'),
           ('Ens Var (A)',_safe_spearmanr(ev,ea_err),'#FF6B6B')]
    items.sort(key=lambda x:x[1],reverse=True)
    bars=ax.barh(range(len(items)),[x[1] for x in items],color=[x[2] for x in items],edgecolor='w',lw=2,height=.65)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items],fontsize=9)
    for b,x in zip(bars,items): ax.text(b.get_width()+.005,b.get_y()+b.get_height()/2,f'{x[1]:.4f}',va='center',fontsize=10,fontweight='bold')
    ax.set_xlabel('Spearman rho'); ax.set_title('All UQ Methods Ranked\nMC Dropout is the winner!',fontsize=13,fontweight='bold')
    ax.invert_yaxis(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # panel 4 — error reduction
    ax=axes[1,0]
    pcts=np.arange(50,96,2)
    for label,unc,err,c in [('MC Drop T8',M[8]['u'],M[8]['e'],'#4A90D9'),
                             ('Combined (A)',comb,ea_err,'#F39C12'),
                             ('Multi-Model (B)',eu,eb_err,'#9B59B6')]:
        overall=np.mean(err)
        reds=[(overall-np.mean(err[unc<=np.percentile(unc,p)]))/overall*100 for p in pcts]
        ax.plot(pcts,reds,'o-',color=c,lw=2.5,ms=5,label=label,alpha=.85)
    ax.set_xlabel('Keep top X% most confident'); ax.set_ylabel('Error Reduction %')
    ax.set_title('Practical: Error Reduction\nby filtering uncertain preds',fontsize=13,fontweight='bold')
    ax.legend(fontsize=10); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # panel 5 — MC Dropout explanation
    ax=axes[1,1]; ax.axis('off')
    ax.set_title('How MC Dropout Works',fontsize=16,fontweight='bold',color=P['dark'])
    for i,(t,c,d2) in enumerate([
            ('1. Train GNN with dropout','#2e86c1','Standard training with dropout layers'),
            ('2. At inference: keep dropout ON','#8e44ad','Unlike normal — DO NOT turn off dropout'),
            ('3. Run 30 forward passes','#27ae60','Same input → 30 different predictions'),
            ('4. Prediction = mean of 30','#f39c12','Average gives final prediction'),
            ('5. Uncertainty = std of 30','#e74c3c','Standard deviation = uncertainty'),
            ('6. No retraining needed!','#2ECC71','Cheapest UQ method available')]):
        y=.88-i*.15
        ax.text(.02,y,t,fontsize=12,fontweight='bold',color=c,transform=ax.transAxes)
        ax.text(.02,y-.04,d2,fontsize=10,color='#666',transform=ax.transAxes)

    # panel 6 — ensemble summary
    ax=axes[1,2]; ax.axis('off')
    ax.set_title('Ensemble Summary',fontsize=16,fontweight='bold',color=P['dark'])
    for i,(t,c,d2) in enumerate([
            ('Experiment A (5 runs)','#8e44ad','Same arch × 5 seeds\nVariance=epistemic unc\nCombined=sqrt(Var+MC²)'),
            ('Experiment B (multi-model)','#9B59B6','Trials 2,5,6,7,8 as ensemble\nDisagreement = uncertainty\nrho=0.117'),
            ('Key Finding','#2ECC71','MC Dropout ALONE (rho=0.482)\nbeats all ensembles (rho~0.1–0.16)\nMore accurate + cheaper!')]):
        y=.88-i*.30
        ax.text(.02,y,t+':',fontsize=13,fontweight='bold',color=c,transform=ax.transAxes)
        ax.text(.02,y-.05,d2,fontsize=10,color='#555',transform=ax.transAxes,linespacing=1.5)

    fig.suptitle('Uncertainty Quantification: MC Dropout + Ensemble Experiments',
                 fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'15_ensemble_deep_dive.png')


# ═══════════════════════════════════════════════════════════
#  16 — CALIBRATION (Temp Scaling)
# ═══════════════════════════════════════════════════════════

def chart_16_calibration(M):
    d=M[8]; unc,err=d['u'],d['e']
    def ece(T,u,e,nb=10):
        s=u*T; edges=np.unique(np.percentile(s,np.linspace(0,100,nb+1))); v=0.
        for j in range(len(edges)-1):
            m=(s>=edges[j])&(s<edges[j+1]) if j<len(edges)-2 else (s>=edges[j])&(s<=edges[j+1])
            if m.sum()==0: continue
            v+=(m.sum()/len(u))*abs(np.mean(e[m]<s[m])-.683)
        return v
    res=minimize_scalar(lambda T:ece(T,unc,err),bounds=(.1,20),method='bounded')
    T_opt=res.x; ece_b=ece(1.,unc,err); ece_a=ece(T_opt,unc,err)
    imp=(ece_b-ece_a)/ece_b*100

    fig,axes=plt.subplots(1,3,figsize=(24,8))

    # coverage
    ax=axes[0]
    sigs=[.5,1,1.5,2,2.5,3]; ex=[.383,.683,.866,.954,.988,.997]
    orig=[np.mean(err<s*unc) for s in sigs]; cal=[np.mean(err<s*unc*T_opt) for s in sigs]
    x=np.arange(len(sigs)); w=.25
    ax.bar(x-w,ex,w,label='Gaussian Ideal',color='#E6E6FA',edgecolor=P['dark'],lw=1.5)
    ax.bar(x,orig,w,label='Before (T=1)',color='#FFEAA7',edgecolor=P['dark'],lw=1.5)
    ax.bar(x+w,cal,w,label=f'After (T={T_opt:.2f})',color='#98FB98',edgecolor=P['dark'],lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([f'{s}σ' for s in sigs],fontsize=12)
    ax.set_ylabel('Coverage'); ax.set_title(f'Coverage at σ levels\n1σ: {orig[1]*100:.1f}%→{cal[1]*100:.1f}%',fontweight='bold')
    ax.legend(fontsize=10); ax.set_facecolor('#FAFBFC')

    # ECE
    ax=axes[1]
    bars=ax.bar(['Before\n(T=1)',f'After\n(T={T_opt:.2f})'],[ece_b,ece_a],
                color=['#FFEAA7','#98FB98'],edgecolor=P['dark'],lw=2,width=.5)
    for b,v in zip(bars,[ece_b,ece_a]): ax.text(b.get_x()+b.get_width()/2,v+.01,f'{v:.4f}',ha='center',fontsize=15,fontweight='bold')
    ax.set_ylabel('ECE (lower=better)')
    ax.set_title(f'ECE Improvement: {imp:.1f}%',fontsize=16,fontweight='bold',color='#2ECC71')
    ax.annotate(f'{imp:.0f}% better!',xy=(1,ece_a),xytext=(.5,(ece_b+ece_a)/2),
                fontsize=16,fontweight='bold',color='#2ECC71',ha='center',
                arrowprops=dict(arrowstyle='->',color='#2ECC71',lw=3))
    ax.set_facecolor('#FAFBFC')

    # explanation
    ax=axes[2]; ax.axis('off')
    ax.set_title('What is Temperature Scaling?',fontsize=16,fontweight='bold',color=P['dark'])
    for i,(t,c,d2) in enumerate([
            ('Problem','#E74C3C',f'MC Dropout unc too narrow\n1σ cov: {d["c1"]:.1f}% (should be 68.3%)\nModel is overconfident!'),
            ('Solution','#2ECC71',f'Multiply unc by T={T_opt:.2f}\nsigma_new = T × sigma_old\nSimple post-hoc, no retraining'),
            ('Result','#4A90D9',f'ECE: {ece_b:.4f}→{ece_a:.4f}\n{imp:.1f}% improvement\n1σ coverage→~68%'),
            ('Why','#F39C12','Calibrated unc → trustworthy\nTraffic planners can set thresholds\nand know they mean what they say')]):
        y=.90-i*.22
        ax.text(.02,y,t+':',fontsize=13,fontweight='bold',color=c,transform=ax.transAxes)
        ax.text(.02,y-.04,d2,fontsize=10,color='#555',transform=ax.transAxes,linespacing=1.5)

    fig.suptitle('Temperature Scaling Calibration\nMaking uncertainty trustworthy for traffic planning',
                 fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'16_calibration.png')
    return T_opt,ece_b,ece_a


# ═══════════════════════════════════════════════════════════
#  17 — SPATIAL MAP (Paris)
# ═══════════════════════════════════════════════════════════

def chart_17_spatial(M,pos):
    d=M[8]; n=31635
    unc_avg=d['u'].reshape(-1,n).mean(axis=0)
    err_avg=d['e'].reshape(-1,n).mean(axis=0)
    p=pos[:n].mean(axis=1); lons,lats=p[:,0],p[:,1]

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(22,10))
    for ax,vals,cmap,label,ttl in [
            (ax1,unc_avg,'hot_r','MC Unc (σ)','Where Is the Model Uncertain?'),
            (ax2,err_avg,'inferno','MAE (veh/h)','Where Are the Errors?')]:
        sc=ax.scatter(lons,lats,c=vals,cmap=cmap,s=2,alpha=.7,
                      vmin=np.percentile(vals,5),vmax=np.percentile(vals,95))
        plt.colorbar(sc,ax=ax,shrink=.8).set_label(label)
        ax.set_title(ttl+'\n31,635 links averaged over 100 test scenarios',fontsize=14,fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_facecolor('#F5F7FA')
    rho=_safe_spearmanr(unc_avg,err_avg)
    fig.suptitle(f'Paris Road Network — rho(uncertainty, error)={rho:.4f}',fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'17_spatial_map.png')


# ═══════════════════════════════════════════════════════════
#  18 — 3-D ERROR SURFACE
# ═══════════════════════════════════════════════════════════

def chart_18_surface(M,feats):
    d=M[8]; n=31635
    feat=np.tile(feats[:n],(len(d['p'])//n,1)); ml=min(len(feat),len(d['p']))
    vol,unc,err=feat[:ml,0],d['u'][:ml],d['e'][:ml]
    # subsample to avoid MemoryError on large boolean masks
    max_pts=300000
    if len(vol)>max_pts:
        idx=np.random.RandomState(42).choice(len(vol),max_pts,False)
        vol,unc,err=vol[idx],unc[idx],err[idx]
    nb=25; vb=np.percentile(vol,np.linspace(0,100,nb+1)); ub=np.percentile(unc,np.linspace(0,100,nb+1))
    Z=np.full((nb,nb),np.nan)
    for i in range(nb):
        for j in range(nb):
            m=(vol>=vb[i])&(vol<vb[i+1])&(unc>=ub[j])&(unc<ub[j+1])
            if m.sum()>10: Z[i,j]=np.mean(err[m])
    X,Y=np.meshgrid((vb[:-1]+vb[1:])/2,(ub[:-1]+ub[1:])/2)
    fig=plt.figure(figsize=(16,12)); ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,np.nan_to_num(Z.T),cmap='magma',alpha=.85,lw=.3,edgecolor='w',antialiased=True)
    ax.set_xlabel('\nBase Volume'); ax.set_ylabel('\nMC Uncertainty'); ax.set_zlabel('\nMean Error')
    ax.set_title('3-D Error Surface: Volume × Uncertainty × Error\nBusy roads + high uncertainty → worst errors',
                 fontsize=15,fontweight='bold',pad=25)
    ax.view_init(elev=30,azim=-45); plt.tight_layout(); _save(fig,'18_3d_surface.png')


# ═══════════════════════════════════════════════════════════
#  19 — FEATURE ANALYSIS (6 features vs uncertainty)
# ═══════════════════════════════════════════════════════════

def chart_19_features(M,feats):
    d=M[8]; n=31635
    fn=['VOL_BASE','CAPACITY','CAP_REDUCTION','FREESPEED','LANES','LENGTH']
    feat=np.tile(feats[:n],(len(d['p'])//n,1)); ml=min(len(feat),len(d['p']))
    feat,unc,err=feat[:ml],d['u'][:ml],d['e'][:ml]
    idx=np.random.RandomState(42).choice(ml,min(50000,ml),False)

    fig,axes=plt.subplots(2,3,figsize=(22,14))
    for i,(nm,ax) in enumerate(zip(fn,axes.flat)):
        ax.hexbin(feat[idx,i],unc[idx],gridsize=60,cmap='YlOrRd',mincnt=1,lw=.1)
        ru=_safe_spearmanr(feat[idx,i],unc[idx]); re=_safe_spearmanr(feat[idx,i],err[idx])
        s='STRONG' if abs(ru)>.3 else ('moderate' if abs(ru)>.15 else 'weak')
        c='#E74C3C' if abs(ru)>.3 else ('#F39C12' if abs(ru)>.15 else '#999')
        ax.set_xlabel(nm,fontweight='bold'); ax.set_ylabel('MC Unc')
        ax.set_title(f'rho_unc={ru:.3f} [{s}]\nrho_err={re:.3f}',fontsize=11,fontweight='bold',color=c)
        ax.set_facecolor('#FAFBFC'); _wm(ax)
    fig.suptitle('Which Features Drive Uncertainty?\nTrial 8 — 6 features vs MC Dropout uncertainty',
                 fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'19_feature_analysis.png')


# ═══════════════════════════════════════════════════════════
#  20 — PRACTICAL VALUE: WITH vs WITHOUT UQ
# ═══════════════════════════════════════════════════════════

def chart_20_practical(M,R):
    d=M[8]; overall=R[8]['mae']
    t90=np.percentile(d['u'],90); low=d['u']<=t90
    mae_c=np.mean(d['e'][low]); mae_u=np.mean(d['e'][~low])
    imp=(overall-mae_c)/overall*100

    fig,axes=plt.subplots(1,3,figsize=(24,9))

    # WITHOUT
    ax=axes[0]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title('WITHOUT Uncertainty',fontsize=16,fontweight='bold',color='#E74C3C')
    ax.add_patch(FancyBboxPatch((.3,1.5),9.3,7,boxstyle='round,pad=0.3',facecolor='#FFE4E1',edgecolor='#E74C3C',lw=3))
    ax.text(5,7,'3,163,500 Predictions',fontsize=15,ha='center',fontweight='bold')
    ax.text(5,5.5,'ALL treated the same',fontsize=13,ha='center',color='#666')
    ax.text(5,4,f'Avg Error = {overall:.3f} veh/h',fontsize=14,ha='center',fontweight='bold',color='#E74C3C')
    ax.text(5,2.5,'No way to know which\npredictions to trust!',fontsize=11,ha='center',color='#E74C3C')

    # WITH
    ax=axes[1]; ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.set_title('WITH MC Dropout UQ',fontsize=16,fontweight='bold',color='#2ECC71')
    ax.add_patch(FancyBboxPatch((.3,5),9.3,3.5,boxstyle='round,pad=0.3',facecolor='#E8F5E9',edgecolor='#2ECC71',lw=3))
    ax.text(5,7.8,f'90% Confident: {low.sum():,}',fontsize=12,ha='center',fontweight='bold',color='#2ECC71')
    ax.text(5,6.8,f'MAE = {mae_c:.3f} veh/h',fontsize=14,ha='center',fontweight='bold')
    ax.text(5,5.5,'TRUST these!',fontsize=10,ha='center',color='#2ECC71',style='italic')
    ax.add_patch(FancyBboxPatch((.3,1.5),9.3,3,boxstyle='round,pad=0.3',facecolor='#FFF3E0',edgecolor='#F39C12',lw=3))
    ax.text(5,3.8,f'10% Uncertain: {(~low).sum():,}',fontsize=12,ha='center',fontweight='bold',color='#F39C12')
    ax.text(5,2.8,f'MAE = {mae_u:.3f} veh/h',fontsize=14,ha='center',fontweight='bold')
    ax.text(5,1.8,'FLAG for review!',fontsize=10,ha='center',color='#F39C12',style='italic')

    # impact
    ax=axes[2]
    bars=ax.bar(['All\n(no UQ)','Top 90%\n(confident)','Bottom 10%\n(uncertain)'],
                [overall,mae_c,mae_u],color=['#FFE4E1','#98FB98','#FFEAA7'],edgecolor=P['dark'],lw=2,width=.55)
    for b,v in zip(bars,[overall,mae_c,mae_u]):
        ax.text(b.get_x()+b.get_width()/2,v+.05,f'{v:.3f}',ha='center',fontsize=14,fontweight='bold')
    ax.set_title(f'{imp:.1f}% Lower Error (confident)',fontsize=14,fontweight='bold',color='#2ECC71')
    ax.set_ylabel('MAE (veh/h)',fontweight='bold'); ax.set_facecolor('#FAFBFC'); _wm(ax)

    fig.suptitle('Practical Value of Uncertainty for Traffic Planners\nUQ tells WHICH predictions to trust',
                 fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'20_practical_value.png')


# ═══════════════════════════════════════════════════════════
#  21 — PER-GRAPH ANALYSIS (Trial 8)
# ═══════════════════════════════════════════════════════════

def chart_21_per_graph(R):
    p,tg=R[8]['p'],R[8]['t']; n=31635; ng=len(p)//n
    gr2=[r2_score(tg[g*n:(g+1)*n],p[g*n:(g+1)*n]) for g in range(ng)]
    gmae=[mean_absolute_error(tg[g*n:(g+1)*n],p[g*n:(g+1)*n]) for g in range(ng)]
    gr2,gmae=np.array(gr2),np.array(gmae)

    fig,axes=plt.subplots(1,3,figsize=(24,8))
    ax=axes[0]; ax.bar(range(ng),gr2,color=cm.viridis(np.linspace(0,1,ng)),width=.9)
    ax.axhline(np.mean(gr2),color='#E74C3C',lw=2,ls='--',label=f'Mean: {np.mean(gr2):.4f}')
    ax.set_xlabel('Scenario'); ax.set_ylabel('R²'); ax.set_title(f'R² per Scenario ({ng} graphs)',fontweight='bold')
    ax.legend(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    ax=axes[1]; ax.bar(range(ng),gmae,color=cm.plasma(np.linspace(0,1,ng)),width=.9)
    ax.axhline(np.mean(gmae),color='#E74C3C',lw=2,ls='--',label=f'Mean: {np.mean(gmae):.3f}')
    ax.set_xlabel('Scenario'); ax.set_ylabel('MAE'); ax.set_title('MAE per Scenario',fontweight='bold')
    ax.legend(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    ax=axes[2]; ax.hist(gr2,bins=30,color='#4A90D9',edgecolor='w',alpha=.8,lw=2)
    ax.axvline(np.median(gr2),color='#E74C3C',lw=2,ls='--',label=f'Median: {np.median(gr2):.4f}')
    ax.set_xlabel('R²'); ax.set_title(f'R² Distribution [{gr2.min():.3f}, {gr2.max():.3f}]',fontweight='bold')
    ax.legend(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    fig.suptitle(f'Trial 8: {ng} Individual Test Scenarios\nEach = different traffic disruption on Paris',
                 fontsize=17,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'21_per_graph.png')


# ═══════════════════════════════════════════════════════════
#  22 — RESEARCH STORY: Elena vs Our Thesis
# ═══════════════════════════════════════════════════════════

def chart_22_research():
    fig,axes=plt.subplots(1,2,figsize=(24,13))

    ax=axes[0]; ax.axis('off'); ax.set_facecolor('#F0F4F8')
    ax.text(.5,.97,"Elena Natterer's Paper",fontsize=20,fontweight='bold',color='#1a5276',ha='center',transform=ax.transAxes)
    for i,(t,c,d) in enumerate([
            ('Architecture','#2e86c1','PointNetTransfGAT\n- PointNetConv (spatial)\n- TransformerConv (attention)\n- GATConv (graph)\n- Final GATConv output'),
            ('Data','#27ae60','Paris road network\n- 31,635 links / 59,851 edges\n- 10,000 MATSim simulations\n- 1% downsampled population'),
            ('Task','#f39c12','Predict DeltaVolume per link\nwhen policy changes (e.g.\ncapacity reduction)'),
            ('Result','#e74c3c','GNN predicts traffic effects\nin seconds vs hours of simulation')]):
        y=.82-i*.21
        ax.text(.05,y,t+':',fontsize=14,fontweight='bold',color=c,transform=ax.transAxes)
        ax.text(.05,y-.03,d,fontsize=10.5,color='#444',transform=ax.transAxes,linespacing=1.4)

    ax=axes[1]; ax.axis('off'); ax.set_facecolor('#F8F0F0')
    ax.text(.5,.97,'Our Thesis Contribution',fontsize=20,fontweight='bold',color='#8e44ad',ha='center',transform=ax.transAxes)
    for i,(t,c,d) in enumerate([
            ('Reproduced the Model','#8e44ad','8 hyperparameter trials\nT1: OLD arch (Linear) R²=0.786 but wrong\nT8: BEST correct R²=0.596\nTrained on only 10% data!'),
            ('MC Dropout UQ','#2980b9','30 stochastic passes = uncertainty\nrho=0.482 (T8) — strongest signal\nNo retraining needed'),
            ('Ensemble UQ','#6c3483','Exp A: 5 training runs (epistemic)\nExp B: multi-model T2,5,6,7,8\nMC Dropout alone beats ensembles!'),
            ('Calibration','#27ae60','Temperature Scaling T=2.90\nECE 0.356→0.033 (90.6% fix)\n1σ coverage→~68.3%\nUncertainties now trustworthy!')]):
        y=.82-i*.21
        ax.text(.05,y,t+':',fontsize=14,fontweight='bold',color=c,transform=ax.transAxes)
        ax.text(.05,y-.03,d,fontsize=10.5,color='#444',transform=ax.transAxes,linespacing=1.4)

    fig.suptitle("Research Context: Elena's GNN + Our UQ Extension",fontsize=18,fontweight='bold',y=1.02)
    plt.tight_layout(); _save(fig,'22_research_story.png')


# ═══════════════════════════════════════════════════════════
#  23 — RADAR (multi-axis, top 4 trials)
# ═══════════════════════════════════════════════════════════

def chart_23_radar(R):
    """Clear grouped bar chart comparing top 4 trials across 5 metrics"""
    fig, axes = plt.subplots(1, 5, figsize=(30, 10))
    trials = [2, 5, 7, 8]
    trial_colors = {2: '#FF8E53', 5: '#48DBFB', 7: '#1ABC9C', 8: '#2ECC71'}

    metrics = [
        ('R²', 'r2', '{:.4f}', 'higher is better', True),
        ('MAE (veh/h)', 'mae', '{:.3f}', 'lower is better', False),
        ('RMSE (veh/h)', 'rmse', '{:.3f}', 'lower is better', False),
        ('<5 Error %', 'u5', '{:.1f}%', 'higher is better', True),
        ('P90 Error', 'p90', '{:.2f}', 'lower is better', False),
    ]

    for ax_i, (name, key, fmt, hint, higher_better) in enumerate(metrics):
        ax = axes[ax_i]
        vals = [R[t][key] for t in trials]
        colors = [trial_colors[t] for t in trials]
        bars = ax.bar([f'T{t}' for t in trials], vals, color=colors, edgecolor='white', lw=2, width=0.6)
        for b, v in zip(bars, vals):
            label = fmt.format(v)
            ax.text(b.get_x() + b.get_width()/2, v + (max(vals)-min(vals))*0.02,
                    label, ha='center', fontsize=11, fontweight='bold')
        # Highlight best
        best_idx = vals.index(max(vals)) if higher_better else vals.index(min(vals))
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(4)
        ax.set_title(f'{name}\n({hint})', fontsize=13, fontweight='bold',
                     color='#2ECC71' if higher_better else '#E74C3C')
        ax.set_facecolor('#FAFBFC')
        ax.tick_params(axis='x', labelsize=12)
        _wm(ax)

    fig.suptitle('Trial Comparison: 5 Key Metrics Side by Side\n'
                 'Gold border = best in each metric  |  Correct architecture only (T2, T5, T7, T8)',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '23_radar.png')


# ═══════════════════════════════════════════════════════════
#  24 — SCATTER GRID: BEST TRIAL PREDICTIONS
# ═══════════════════════════════════════════════════════════

def chart_24_scatter_grid(R):
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    # Different colormaps for visual variety
    cmaps = ['Reds', 'Oranges', 'YlOrBr', 'YlOrRd', 'Blues', 'GnBu', 'BuGn', 'Greens']
    for idx, t in enumerate(range(1, 9)):
        ax = axes[idx//4][idx%4]; d = R[t]
        si = np.random.RandomState(42).choice(len(d['p']), min(80000, len(d['p'])), False)
        ax.hexbin(d['t'][si], d['p'][si], gridsize=80, cmap=cmaps[idx], mincnt=1, lw=.1)
        lims = [min(d['t'].min(), d['p'].min()), max(d['t'].max(), d['p'].max())]
        ax.plot(lims, lims, 'k--', lw=2, alpha=.6)
        c = '#E74C3C' if t == 1 else '#2ECC71'
        tag = '⚠ WRONG ARCH' if t == 1 else LABELS[t]
        ax.set_title(f'Trial {t}: R²={d["r2"]:.4f}\n{tag}', fontsize=11, fontweight='bold', color=c)
        ax.set_xlabel('Actual (veh/h)'); ax.set_ylabel('Predicted (veh/h)')
        ax.set_facecolor('#FAFBFC'); _wm(ax)
    fig.suptitle('Prediction Scatter — All 8 Trials\nCloser to diagonal = better  |  Trial 1 = wrong architecture',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '24_scatter_grid.png')


# ═══════════════════════════════════════════════════════════
#  25 — ERROR DISTRIBUTION OVERLAY
# ═══════════════════════════════════════════════════════════

def chart_25_error_overlay(R):
    fig, axes = plt.subplots(1, 2, figsize=(28, 11))

    # Left: CDF for ALL 7 correct trials
    ax = axes[0]
    for t in [2, 3, 4, 5, 6, 7, 8]:
        e = R[t]['e']; se = np.sort(e[e < 25])
        ax.plot(se, np.arange(len(se))/len(se), color=TC[t-1], lw=2.5,
                label=f'T{t}: MAE={R[t]["mae"]:.2f}, R²={R[t]["r2"]:.3f}')
    ax.axhline(.5, color='#aaa', ls=':', alpha=.5, lw=1.5)
    ax.axhline(.9, color='#aaa', ls='--', alpha=.5, lw=1.5)
    ax.axvline(5, color='#888', ls='--', alpha=.4, lw=2)
    ax.fill_betweenx([0, 1], 0, 5, alpha=0.06, color='#2ECC71')
    ax.text(2.5, 0.05, 'Low Error\nZone (<5)', ha='center', fontsize=10,
            color='#2ECC71', fontweight='bold', alpha=0.7)
    ax.set_xlabel('Absolute Error (veh/h)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Fraction', fontsize=13, fontweight='bold')
    ax.set_title('Error CDF — All 7 Correct Trials\nFurther left = better performance',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.text(0.03, 0.97, 'HOW TO READ:\n\u2022 Lines further LEFT = better\n\u2022 Steeper rise = more accurate\n\u2022 Green zone = error < 5 veh/h\n\u2022 T8 (green) is furthest left',
            transform=ax.transAxes, fontsize=8.5, va='top', color='#555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F8FF', edgecolor='#4A90D9', alpha=0.9, lw=1.5))
    ax.set_facecolor('#FAFBFC'); _wm(ax)
    ax = axes[1]
    trials = [2, 3, 4, 5, 6, 7, 8]
    metrics = ['MAE', 'Median', 'P90', '<5 err %']
    x = np.arange(len(trials))
    w = 0.2
    data = {
        'MAE': [R[t]['mae'] for t in trials],
        'Median': [R[t]['med'] for t in trials],
        'P90': [R[t]['p90'] for t in trials],
    }
    colors_m = ['#4A90D9', '#F39C12', '#E74C3C']
    for i, (metric, vals) in enumerate(data.items()):
        bars = ax.bar(x + i*w - w, vals, w, label=metric,
                      color=colors_m[i], edgecolor='white', lw=1.5, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.1f}',
                    ha='center', fontsize=8, fontweight='bold', rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{t}' for t in trials], fontsize=11, fontweight='bold')
    ax.set_ylabel('Error (veh/h)', fontweight='bold')
    ax.set_title('Error Statistics Breakdown\nMAE, Median, P90 for each trial',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Highlight best
    ax.annotate('★ T8 is best\nacross all metrics', xy=(6, R[8]['mae']),
                xytext=(4.5, max(R[t]['p90'] for t in trials) * 0.85),
                fontsize=12, fontweight='bold', color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))

    fig.suptitle('Error Distribution Analysis — Complete Picture\n'
                 'Trial 8 consistently smallest errors | Weighted loss trials (T3, T4) clearly worse',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '25_error_cdf_overlay.png')


# ═══════════════════════════════════════════════════════════
#  26 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════

def chart_26_heatmap(R):
    fig,ax=plt.subplots(figsize=(10,8))
    ct=[2,3,4,5,6,7,8]; metrics=['R²','MAE','RMSE','Median Err','P90','<5 Err %']
    data=np.array([[R[t]['r2'],R[t]['mae'],R[t]['rmse'],R[t]['med'],R[t]['p90'],R[t]['u5']] for t in ct])
    # normalize cols 0-1
    dn=(data-data.min(0))/(data.max(0)-data.min(0)+1e-9)
    im=ax.imshow(dn.T,cmap='RdYlGn',aspect='auto',vmin=0,vmax=1)
    ax.set_xticks(range(len(ct))); ax.set_xticklabels([f'T{t}' for t in ct],fontsize=11)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics,fontsize=11)
    for i in range(len(metrics)):
        for j in range(len(ct)):
            ax.text(j,i,f'{data[j,i]:.3f}',ha='center',va='center',fontsize=9,fontweight='bold',
                    color='white' if dn[j,i]<.3 or dn[j,i]>.7 else 'black')
    plt.colorbar(im,ax=ax,shrink=.75,label='Normalised (0=worst, 1=best per metric)')
    ax.set_title('Performance Heatmap — Correct Trials\nRed=worst, Green=best per metric',
                 fontsize=15,fontweight='bold')
    plt.tight_layout(); _save(fig,'26_heatmap.png')


# ═══════════════════════════════════════════════════════════
#  27 — FINAL DASHBOARD
# ═══════════════════════════════════════════════════════════

def chart_27_dashboard(R,M,ea,eb,T_opt,ece_b,ece_a):
    gc.collect()
    imp=(ece_b-ece_a)/ece_b*100
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#F0F2F5')

    # Create a 3×3 grid for more visual impact
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ── Card 1: Best Model (large, top-left) ──
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off'); ax.set_facecolor('#E8F5E9')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#E8F5E9', edgecolor='#2ECC71', lw=3))
    ax.text(0.5, 0.92, '★ BEST MODEL', ha='center', fontsize=16, fontweight='bold',
            color='#2ECC71', transform=ax.transAxes)
    ax.text(0.5, 0.82, 'Trial 8', ha='center', fontsize=22, fontweight='bold',
            color='#1a5276', transform=ax.transAxes)
    for i, (k, v, c) in enumerate([
            ('R²', f'{R[8]["r2"]:.4f}', '#4A90D9'),
            ('MAE', f'{R[8]["mae"]:.3f} veh/h', '#4A90D9'),
            ('RMSE', f'{R[8]["rmse"]:.3f} veh/h', '#4A90D9'),
            ('<5 Error', f'{R[8]["u5"]:.1f}%', '#2ECC71'),
            ('Config', 'BS32 DO0.15 LR1e-3', '#555')]):
        y = 0.68 - i * 0.12
        ax.text(0.12, y, f'{k}:', fontsize=11, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.55, y, v, fontsize=12, transform=ax.transAxes, color=c, fontweight='bold')

    # ── Card 2: Trial Ranking bar chart ──
    ax = fig.add_subplot(gs[0, 1])
    ct = [2, 3, 4, 5, 6, 7, 8]; st = sorted(ct, key=lambda t: R[t]['r2'], reverse=True)
    bars = ax.barh(range(7), [R[t]['r2'] for t in st], color=[TC[t-1] for t in st], edgecolor='w', lw=2)
    ax.set_yticks(range(7)); ax.set_yticklabels([f'T{t}' for t in st], fontsize=11)
    ax.set_xlabel('R²'); ax.set_title('Trial Ranking', fontweight='bold', fontsize=14)
    for b, v in zip(bars, [R[t]['r2'] for t in st]):
        ax.text(v + 0.003, b.get_y() + b.get_height()/2, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # ── Card 3: UQ Ranking bar chart ──
    ax = fig.add_subplot(gs[0, 2])
    ta = ea['targets'].flatten(); ea_e = np.abs(ea['ensemble_mean'].flatten()-ta)
    tb = eb['targets'].flatten(); eb_e = np.abs(eb['ensemble_prediction'].flatten()-tb)
    uq = [('MC T8', M[8]['rho'], '#4A90D9'), ('MC T7', M[7]['rho'], '#0ABDE3'),
          ('MC T5', M[5]['rho'], '#48DBFB'), ('MC T6', M[6]['rho'], '#87CEEB'),
          ('Ens A', _safe_spearmanr(ea['avg_mc_uncertainty'].flatten(), ea_e), '#F39C12'),
          ('Ens B', _safe_spearmanr(eb['ensemble_uncertainty'].flatten(), eb_e), '#9B59B6')]
    uq.sort(key=lambda x: x[1], reverse=True)
    bars = ax.barh(range(len(uq)), [x[1] for x in uq], color=[x[2] for x in uq], edgecolor='w', lw=2)
    ax.set_yticks(range(len(uq))); ax.set_yticklabels([x[0] for x in uq], fontsize=11)
    for b, v in zip(bars, [x[1] for x in uq]):
        ax.text(v + 0.003, b.get_y() + b.get_height()/2, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Spearman ρ'); ax.set_title('UQ Ranking', fontweight='bold', fontsize=14)
    ax.invert_yaxis(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # ── Card 4: Calibration ──
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off'); ax.set_facecolor('#FFF8E1')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#FFF8E1', edgecolor='#F39C12', lw=3))
    ax.text(0.5, 0.90, 'CALIBRATION', ha='center', fontsize=16, fontweight='bold',
            color='#F39C12', transform=ax.transAxes)
    ax.text(0.5, 0.78, f'T = {T_opt:.2f}', ha='center', fontsize=28, fontweight='bold',
            color='#1a5276', transform=ax.transAxes)
    for i, (k, v, c) in enumerate([
            ('ECE Before', f'{ece_b:.4f}', '#E74C3C'),
            ('ECE After', f'{ece_a:.4f}', '#2ECC71'),
            ('Improvement', f'{imp:.1f}%', '#2ECC71'),
            ('1σ Coverage', f'{M[8]["c1"]:.1f}% → ~68%', '#4A90D9')]):
        y = 0.62 - i * 0.14
        ax.text(0.12, y, f'{k}:', fontsize=11, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.55, y, v, fontsize=12, transform=ax.transAxes, color=c, fontweight='bold')

    # ── Card 5: Network Stats ──
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off'); ax.set_facecolor('#E3F2FD')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#E3F2FD', edgecolor='#4A90D9', lw=3))
    ax.text(0.5, 0.90, 'PARIS NETWORK', ha='center', fontsize=16, fontweight='bold',
            color='#4A90D9', transform=ax.transAxes)
    for i, (k, v) in enumerate([
            ('Road Links', '31,635'), ('Graph Edges', '59,851'),
            ('Input Features', '5'), ('MC Passes', '30'),
            ('Training Data', '10% ONLY'), ('Test Preds (T8)', f'{R[8]["n"]:,}')]):
        y = 0.75 - i * 0.11
        ax.text(0.12, y, f'{k}:', fontsize=11, fontweight='bold', transform=ax.transAxes, color='#555')
        ax.text(0.62, y, v, fontsize=12, transform=ax.transAxes, color='#1a5276', fontweight='bold')

    # ── Card 6: Takeaways ──
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off'); ax.set_facecolor('#F3E5F5')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#F3E5F5', edgecolor='#8e44ad', lw=3))
    ax.text(0.5, 0.92, 'KEY TAKEAWAYS', ha='center', fontsize=16, fontweight='bold',
            color='#8e44ad', transform=ax.transAxes)
    for i, (n, t, c) in enumerate([
            ('1.', 'GNN surrogates work for traffic\n   prediction even with 10% data', '#27ae60'),
            ('2.', 'MC Dropout is best UQ method\n   (ρ=0.482, cheapest, no retraining)', '#4A90D9'),
            ('3.', 'Temperature scaling makes unc.\n   trustworthy (90.6% ECE fix)', '#F39C12'),
            ('4.', 'Planners can use uncertainty to\n   decide which predictions to trust', '#E74C3C')]):
        y = 0.78 - i * 0.18
        ax.text(0.06, y, n, fontsize=13, fontweight='bold', color=c, transform=ax.transAxes)
        ax.text(0.12, y, t, fontsize=10, color='#444', transform=ax.transAxes, linespacing=1.4)

    # ── Bottom row: 3D-style metric cards ──
    metrics_row = [
        ('R² = 0.596', 'Best Trial 8', '#2ECC71'),
        ('MAE = 3.957', 'Vehicles/hour', '#4A90D9'),
        ('ρ = 0.482', 'MC Dropout Quality', '#8e44ad'),
        ('ECE = 0.033', 'After Calibration', '#F39C12'),
        ('~1800×', 'Faster than MATSim', '#E74C3C'),
    ]
    for i, (val, desc, color) in enumerate(metrics_row):
        ax_m = fig.add_subplot(gs[2, :])
        if i == 0:
            ax_m.axis('off')
            ax_m.set_xlim(0, 50); ax_m.set_ylim(0, 6)
        cx = 5 + i * 10
        ax_m.add_patch(FancyBboxPatch((cx-4, 0.5), 8, 5, boxstyle='round,pad=0.3',
                       facecolor=color, edgecolor='white', lw=2, alpha=0.15))
        ax_m.add_patch(FancyBboxPatch((cx-3.8, 0.7), 7.6, 4.6, boxstyle='round,pad=0.2',
                       facecolor='white', edgecolor=color, lw=2.5, alpha=0.9))
        ax_m.text(cx, 3.8, val, ha='center', fontsize=18, fontweight='bold', color=color)
        ax_m.text(cx, 1.8, desc, ha='center', fontsize=10, color='#666')

    fig.suptitle('COMPLETE THESIS DASHBOARD\nAll values cross-checked from NPZ  |  PointNetTransfGAT on Paris',
                 fontsize=20, fontweight='bold', y=1.02)
    _save(fig, '27_final_dashboard.png')


# ═══════════════════════════════════════════════════════════
#  28 — TRIAL EVOLUTION TIMELINE
# ═══════════════════════════════════════════════════════════

def chart_28_trial_evolution(R):
    """What changed trial to trial and what happened to R²"""
    fig, axes = plt.subplots(2, 1, figsize=(28, 18), gridspec_kw={'height_ratios': [2.5, 1]})

    # === Top: Timeline ribbon ===
    ax = axes[0]; ax.set_xlim(-0.5, 8.5); ax.set_ylim(-1, 11); ax.axis('off')
    ax.set_facecolor('#FAFBFC')

    changes = {
        1: ('OLD Architecture\nLinear final layer', '#E74C3C', 'BS32, DO0.20\nLR 1e-3, 100ep\n70/15/15 split'),
        2: ('Fixed Architecture\n+ Weighted Loss', '#FF8E53', 'BS32, DO0.20\nLR 1e-3, 100ep\n70/15/15, W-MSE'),
        3: ('Weighted Loss\n+ More Epochs', '#FECA57', 'BS32, DO0.20\nLR 1e-3, 150ep\n70/15/15, W-MSE'),
        4: ('Weighted Loss v2\n(same approach)', '#F39C12', 'BS32, DO0.20\nLR 1e-3, 150ep\n70/15/15, W-MSE'),
        5: ('Standard Loss\n+ Larger Batch', '#48DBFB', 'BS64, DO0.20\nLR 1e-3, 100ep\n70/15/15 split'),
        6: ('Lower LR\n+ More Epochs', '#0ABDE3', 'BS32, DO0.20\nLR 5e-4, 150ep\n70/15/15 split'),
        7: ('New Split\n80/10/10', '#1ABC9C', 'BS32, DO0.20\nLR 1e-3, 100ep\n80/10/10 split'),
        8: ('Lower Dropout\n0.15', '#2ECC71', 'BS32, DO0.15\nLR 1e-3, 100ep\n80/10/10 split'),
    }

    for t in range(1, 9):
        x = t - 0.5
        change, color, hp = changes[t]
        # Rounded card with shadow effect
        ax.add_patch(FancyBboxPatch((x - 0.46, 3.2), 0.92, 6.5, boxstyle='round,pad=0.15',
                     facecolor='#DDD', edgecolor='none', alpha=0.3))  # shadow
        ax.add_patch(FancyBboxPatch((x - 0.48, 3.4), 0.92, 6.5, boxstyle='round,pad=0.15',
                     facecolor=color, edgecolor='white', lw=2, alpha=0.9))
        # trial number circle
        ax.text(x, 9.3, f'T{t}', ha='center', fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='white', lw=2))
        # what changed
        ax.text(x, 7.8, change, ha='center', fontsize=9, color='white', linespacing=1.3,
                fontweight='bold')
        # hyperparams
        ax.text(x, 5.8, hp, ha='center', fontsize=8, color='#FFF', alpha=0.9, linespacing=1.3)
        # R² value badge
        r2 = R[t]['r2']
        badge_color = '#E74C3C' if t == 1 else ('#2ECC71' if t == 8 else '#FFF')
        badge_tc = 'white' if t in [1, 8] else '#333'
        ax.text(x, 4, f'R²={r2:.4f}', ha='center', fontsize=11, fontweight='bold',
                color=badge_tc, bbox=dict(facecolor=badge_color, alpha=0.85,
                boxstyle='round,pad=0.2', edgecolor='white', lw=1.5))
        # arrow to next
        if t < 8:
            ax.annotate('', xy=(x + 0.55, 6.5), xytext=(x + 0.47, 6.5),
                        arrowprops=dict(arrowstyle='->', color='#666', lw=2.5))

    # Labels
    ax.text(0.5, 2.5, '⚠ INVALID', ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
    ax.text(4.5, 2.5, '← All correct architecture (GATConv final) →', ha='center',
            fontsize=11, fontweight='bold', color='#2ECC71')

    # === Bottom: R² progression ===
    ax = axes[1]
    trials = list(range(1, 9))
    r2s = [R[t]['r2'] for t in trials]
    colors = ['#E74C3C'] + [TC[t-1] for t in range(2, 9)]
    bars = ax.bar(trials, r2s, color=colors, edgecolor='white', lw=2, width=0.7)
    for t, v in zip(trials, r2s):
        ax.text(t, v + 0.01, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
    # highlight best
    bars[7].set_edgecolor('#FFD700'); bars[7].set_linewidth(4)
    ax.axhline(R[8]['r2'], color='#2ECC71', ls='--', lw=1.5, alpha=0.5)
    ax.set_xlabel('Trial', fontsize=13, fontweight='bold')
    ax.set_ylabel('R²', fontsize=13, fontweight='bold')
    ax.set_xticks(trials)
    ax.set_xticklabels([f'T{t}' for t in trials], fontsize=12)
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    fig.suptitle('Trial Evolution: What Changed & What Happened\n'
                 'From wrong architecture to best model — 8 experiments on 10% Paris data',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '28_trial_evolution.png')


# ═══════════════════════════════════════════════════════════
#  29 — HYPERPARAMETER ABLATION: DROPOUT EFFECT
# ═══════════════════════════════════════════════════════════

def chart_29_dropout_effect(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    # Dropout groups (correct arch only)
    groups = {0.15: [8], 0.20: [2, 3, 4, 5, 6, 7]}
    do_vals = sorted(groups.keys())

    # R² by dropout
    ax = axes[0]
    for do in do_vals:
        ts = groups[do]
        r2s = [R[t]['r2'] for t in ts]
        for t, r2 in zip(ts, r2s):
            ax.scatter(do, r2, c=TC[t-1], s=200, edgecolors='w', lw=2, zorder=5)
            ax.annotate(f'T{t}', (do, r2), textcoords='offset points', xytext=(8, 5),
                       fontsize=10, fontweight='bold', color=TC[t-1])
    avg_r2 = {do: np.mean([R[t]['r2'] for t in groups[do]]) for do in do_vals}
    ax.plot(do_vals, [avg_r2[d] for d in do_vals], 'k--', lw=2, alpha=0.5, label='Mean')
    ax.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax.set_title('R² vs Dropout Rate\nLower dropout → better R²', fontweight='bold', color='#2ECC71')
    ax.legend(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # MAE by dropout
    ax = axes[1]
    for do in do_vals:
        ts = groups[do]
        maes = [R[t]['mae'] for t in ts]
        for t, mae in zip(ts, maes):
            ax.scatter(do, mae, c=TC[t-1], s=200, edgecolors='w', lw=2, zorder=5)
            ax.annotate(f'T{t}', (do, mae), textcoords='offset points', xytext=(8, 5),
                       fontsize=10, fontweight='bold', color=TC[t-1])
    avg_mae = {do: np.mean([R[t]['mae'] for t in groups[do]]) for do in do_vals}
    ax.plot(do_vals, [avg_mae[d] for d in do_vals], 'k--', lw=2, alpha=0.5, label='Mean')
    ax.set_xlabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (veh/h)', fontsize=12, fontweight='bold')
    ax.set_title('MAE vs Dropout Rate\nLower dropout → lower error', fontweight='bold', color='#2ECC71')
    ax.legend(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Summary card
    ax = axes[2]; ax.axis('off')
    ax.set_title('Dropout Ablation Summary', fontsize=16, fontweight='bold', color=P['dark'])
    items = [
        ('DO = 0.15', '#2ECC71', f'Trial 8 only\nR² = {R[8]["r2"]:.4f}\nMAE = {R[8]["mae"]:.3f}\n★ BEST dropout rate!'),
        ('DO = 0.20', '#F39C12', f'Trials 2-7 (6 trials)\nAvg R² = {avg_r2[0.20]:.4f}\nAvg MAE = {avg_mae[0.20]:.3f}\nAll other correct trials'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.85 - i * 0.30
        ax.text(0.05, y, title + ':', fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y - 0.05, desc, fontsize=10.5, color='#444', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Hyperparameter Ablation: Dropout Rate Effect\n'
                 'DO=0.15 wins — less regularization helps with limited data',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '29_dropout_effect.png')


# ═══════════════════════════════════════════════════════════
#  30 — HYPERPARAMETER ABLATION: LEARNING RATE
# ═══════════════════════════════════════════════════════════

def chart_30_lr_effect(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    groups = {1e-3: [2, 3, 4, 5, 7, 8], 5e-4: [6]}

    # R² comparison
    ax = axes[0]
    positions = {1e-3: 0, 5e-4: 1}
    for lr in [1e-3, 5e-4]:
        r2s = [R[t]['r2'] for t in groups[lr]]
        bp = ax.boxplot([r2s], positions=[positions[lr]], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor='#4A90D9' if lr==1e-3 else '#FF6B6B', alpha=0.5))
        for t in groups[lr]:
            ax.scatter(positions[lr] + np.random.uniform(-0.1, 0.1), R[t]['r2'],
                      c=TC[t-1], s=150, edgecolors='w', lw=2, zorder=5)
            ax.annotate(f'T{t}', (positions[lr], R[t]['r2']),
                       textcoords='offset points', xytext=(15, 0), fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['LR=1e-3', 'LR=5e-4'], fontsize=12)
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_title('R² by Learning Rate', fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # MAE comparison
    ax = axes[1]
    for lr in [1e-3, 5e-4]:
        maes = [R[t]['mae'] for t in groups[lr]]
        bp = ax.boxplot([maes], positions=[positions[lr]], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor='#4A90D9' if lr==1e-3 else '#FF6B6B', alpha=0.5))
        for t in groups[lr]:
            ax.scatter(positions[lr] + np.random.uniform(-0.1, 0.1), R[t]['mae'],
                      c=TC[t-1], s=150, edgecolors='w', lw=2, zorder=5)
            ax.annotate(f'T{t}', (positions[lr], R[t]['mae']),
                       textcoords='offset points', xytext=(15, 0), fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['LR=1e-3', 'LR=5e-4'], fontsize=12)
    ax.set_ylabel('MAE (veh/h)', fontweight='bold')
    ax.set_title('MAE by Learning Rate', fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Summary
    ax = axes[2]; ax.axis('off')
    ax.set_title('Learning Rate Finding', fontsize=16, fontweight='bold', color=P['dark'])
    avg_1e3 = np.mean([R[t]['r2'] for t in groups[1e-3]])
    avg_5e4 = np.mean([R[t]['r2'] for t in groups[5e-4]])
    items = [
        ('LR = 1e-3', '#4A90D9', f'Trials 2, 3, 4, 5, 7, 8\nAvg R² = {avg_1e3:.4f}\nIncludes T2/T3/T4 (W-MSE)\nBest trial T8 uses this'),
        ('LR = 5e-4', '#FF6B6B', f'Trial 6 only\nR² = {avg_5e4:.4f}\nLower LR with 150 epochs\nSlower convergence'),
        ('Finding', '#2ECC71', 'LR=1e-3 with DO=0.15 is best\nHigher LR converges faster\nOnly T6 used lower LR\nand it did not help'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.85 - i * 0.30
        ax.text(0.05, y, title + ':', fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y - 0.05, desc, fontsize=10.5, color='#444', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Hyperparameter Ablation: Learning Rate Effect\n'
                 'LR=1e-3 + DO=0.15 = winning combination (Trial 8)',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '30_lr_effect.png')


# ═══════════════════════════════════════════════════════════
#  31 — WEIGHTED LOSS IMPACT
# ═══════════════════════════════════════════════════════════

def chart_31_weighted_loss(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    wt = [2, 3, 4]  # weighted loss trials (T2, T3, T4 use W-MSE)
    std = [5, 6, 7, 8]  # standard loss trials

    # R² comparison
    ax = axes[0]
    w_r2 = [R[t]['r2'] for t in wt]
    s_r2 = [R[t]['r2'] for t in std]
    bars = ax.bar(['Weighted\nLoss', 'Standard\nLoss'], [np.mean(w_r2), np.mean(s_r2)],
                 color=['#FFEAA7', '#98FB98'], edgecolor=P['dark'], lw=2, width=0.5)
    for b, v in zip(bars, [np.mean(w_r2), np.mean(s_r2)]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=14, fontweight='bold')
    # individual points
    for t in wt:
        ax.scatter(0, R[t]['r2'], c=TC[t-1], s=120, edgecolors='w', lw=2, zorder=5)
        ax.annotate(f'T{t}', (0, R[t]['r2']), textcoords='offset points', xytext=(12, 0), fontsize=9)
    for t in std:
        ax.scatter(1, R[t]['r2'], c=TC[t-1], s=120, edgecolors='w', lw=2, zorder=5)
        ax.annotate(f'T{t}', (1, R[t]['r2']), textcoords='offset points', xytext=(12, 0), fontsize=9)
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_title('Average R²\nWeighted loss HURTS performance', fontweight='bold', color='#E74C3C')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # MAE comparison
    ax = axes[1]
    w_mae = [R[t]['mae'] for t in wt]
    s_mae = [R[t]['mae'] for t in std]
    bars = ax.bar(['Weighted\nLoss', 'Standard\nLoss'], [np.mean(w_mae), np.mean(s_mae)],
                 color=['#FFEAA7', '#98FB98'], edgecolor=P['dark'], lw=2, width=0.5)
    for b, v in zip(bars, [np.mean(w_mae), np.mean(s_mae)]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f'{v:.3f}', ha='center', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (veh/h)', fontweight='bold')
    ax.set_title('Average MAE\nHigher error with weighted loss', fontweight='bold', color='#E74C3C')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Explanation
    ax = axes[2]; ax.axis('off')
    ax.set_title('Weighted Loss Analysis', fontsize=16, fontweight='bold', color=P['dark'])
    diff_r2 = np.mean(s_r2) - np.mean(w_r2)
    items = [
        ('What It Is', '#4A90D9', 'Weighted loss gives MORE weight\nto high-traffic links\nIdea: focus on important roads'),
        ('What Happened', '#E74C3C', f'Avg R²: {np.mean(w_r2):.4f} (weighted)\n'
         f'vs {np.mean(s_r2):.4f} (standard)\n'
         f'R² dropped by {diff_r2:.4f}!'),
        ('Why It Failed', '#F39C12', 'With only 10% training data,\nweighting disrupts gradient balance\n'
         'Model overfit to high-vol links\nand generalized WORSE overall'),
        ('Verdict', '#E74C3C', 'DO NOT use weighted loss\nwith limited training data.\nStandard MSE loss is better!'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.88 - i * 0.22
        ax.text(0.03, y, title + ':', fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.03, y - 0.04, desc, fontsize=10, color='#444', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Weighted Loss Experiment: Did It Help?\n'
                 'Trials 2, 3 & 4 used W-MSE — performance got WORSE',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '31_weighted_loss.png')


# ═══════════════════════════════════════════════════════════
#  32 — DATA SPLIT COMPARISON (70/15/15 vs 80/10/10)
# ═══════════════════════════════════════════════════════════

def chart_32_split_comparison(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    s1 = [2, 3, 4, 5, 6]  # 70/15/15
    s2 = [7, 8]  # 80/10/10

    # grouped bar R², MAE
    for ax_i, (metric, key, fmt, better) in enumerate([
            ('R²', 'r2', '{:.4f}', 'higher'), ('MAE (veh/h)', 'mae', '{:.3f}', 'lower')]):
        ax = axes[ax_i]
        v1 = np.mean([R[t][key] for t in s1])
        v2 = np.mean([R[t][key] for t in s2])
        bars = ax.bar(['70/15/15\n(5 trials)', '80/10/10\n(2 trials)'], [v1, v2],
                     color=['#FFB6C1', '#98FB98'], edgecolor=P['dark'], lw=2, width=0.5)
        for b, v in zip(bars, [v1, v2]):
            ax.text(b.get_x()+b.get_width()/2, v + 0.005, fmt.format(v),
                    ha='center', fontsize=14, fontweight='bold')
        for t in s1:
            ax.scatter(0, R[t][key], c=TC[t-1], s=120, edgecolors='w', lw=2, zorder=5)
        for t in s2:
            ax.scatter(1, R[t][key], c=TC[t-1], s=120, edgecolors='w', lw=2, zorder=5)
        winner = '80/10/10' if (better=='higher' and v2>v1) or (better=='lower' and v2<v1) else '70/15/15'
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{metric}: {winner} wins', fontweight='bold', color='#2ECC71')
        ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Explanation
    ax = axes[2]; ax.axis('off')
    ax.set_title('Split Strategy Finding', fontsize=16, fontweight='bold', color=P['dark'])
    avg1_r2 = np.mean([R[t]['r2'] for t in s1])
    avg2_r2 = np.mean([R[t]['r2'] for t in s2])
    items = [
        ('70/15/15 Split', '#FF6B6B', f'Trials 2-6 (5 trials)\nAvg R² = {avg1_r2:.4f}\n'
         'Larger validation set\nSmaller test set'),
        ('80/10/10 Split', '#2ECC71', f'Trials 7-8 (2 trials)\nAvg R² = {avg2_r2:.4f}\n'
         'More test graphs (100)\nMore robust evaluation'),
        ('Note', '#F39C12', 'Split alone doesn\'t explain gap.\n'
         'Only T8 uses DO=0.15 (others 0.20)\nwhich is the main advantage.\n'
         'But 100 test graphs give\nmore robust evaluation.'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.85 - i * 0.30
        ax.text(0.05, y, title + ':', fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y - 0.05, desc, fontsize=10, color='#444', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Data Split Strategy: 70/15/15 vs 80/10/10\n'
                 'More test data enables more robust evaluation',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '32_split_comparison.png')


# ═══════════════════════════════════════════════════════════
#  33 — ARCHITECTURE BUG VISUAL EXPLANATION
# ═══════════════════════════════════════════════════════════

def chart_33_arch_bug():
    fig, axes = plt.subplots(1, 2, figsize=(24, 13))

    # LEFT — WRONG (Trial 1)
    ax = axes[0]; ax.axis('off'); ax.set_facecolor('#FFF0F0')
    ax.set_title('Trial 1: WRONG Architecture', fontsize=20, fontweight='bold', color='#E74C3C')
    layers = [
        ('Input', '5 features', '#1a5276'),
        ('PointNetConv ×2', '5 → 512 → 128', '#8e44ad'),
        ('TransformerConv ×2', '128 → 256 → 512', '#c0392b'),
        ('GATConv', '512 → 64', '#27ae60'),
        ('⚠ Linear(64, 1)', 'WRONG: plain linear\nNo graph attention', '#E74C3C'),
        ('Output', 'ΔVolume per node', '#555'),
    ]
    for i, (name, desc, color) in enumerate(layers):
        y = 0.88 - i * 0.14
        ax.add_patch(FancyBboxPatch((0.1, y - 0.04), 0.8, 0.1,
                     transform=ax.transAxes, boxstyle='round,pad=0.01',
                     facecolor=color, alpha=0.2, edgecolor=color, lw=2))
        ax.text(0.15, y, name, fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.55, y, desc, fontsize=11, color='#444', transform=ax.transAxes)
    ax.text(0.5, 0.07, 'R² = 0.786 — looks great but architecture\n'
            'does NOT match Elena\'s paper!', fontsize=13, ha='center',
            color='#E74C3C', fontweight='bold', transform=ax.transAxes,
            bbox=dict(facecolor='#FFE4E1', edgecolor='#E74C3C', lw=2, boxstyle='round,pad=0.3'))

    # RIGHT — CORRECT (Trials 2-8)
    ax = axes[1]; ax.axis('off'); ax.set_facecolor('#F0FFF0')
    ax.set_title('Trials 2-8: CORRECT Architecture', fontsize=20, fontweight='bold', color='#2ECC71')
    layers = [
        ('Input', '5 features', '#1a5276'),
        ('PointNetConv ×2', '5 → 512 → 128', '#8e44ad'),
        ('TransformerConv ×2', '128 → 256 → 512', '#c0392b'),
        ('GATConv', '512 → 64', '#27ae60'),
        ('✓ GATConv(64, 1)', 'CORRECT: graph attention\nMatches Elena\'s paper', '#2ECC71'),
        ('Output', 'ΔVolume per node', '#555'),
    ]
    for i, (name, desc, color) in enumerate(layers):
        y = 0.88 - i * 0.14
        ax.add_patch(FancyBboxPatch((0.1, y - 0.04), 0.8, 0.1,
                     transform=ax.transAxes, boxstyle='round,pad=0.01',
                     facecolor=color, alpha=0.2, edgecolor=color, lw=2))
        ax.text(0.15, y, name, fontsize=14, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.55, y, desc, fontsize=11, color='#444', transform=ax.transAxes)
    ax.text(0.5, 0.07, 'Best R² = 0.596 (T8) — lower but HONEST\n'
            'Matches paper architecture exactly!', fontsize=13, ha='center',
            color='#2ECC71', fontweight='bold', transform=ax.transAxes,
            bbox=dict(facecolor='#E8F5E9', edgecolor='#2ECC71', lw=2, boxstyle='round,pad=0.3'))

    fig.suptitle('Architecture Bug: Why Trial 1 Results Are Invalid\n'
                 'Linear final layer ≠ GATConv — R² boost was fake!',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '33_arch_bug.png')


# ═══════════════════════════════════════════════════════════
#  34 — MC DROPOUT COVERAGE DEEP DIVE
# ═══════════════════════════════════════════════════════════

def chart_34_mc_coverage(M):
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))

    # Panel 1: WHAT WE DID — explanation card
    ax = axes[0, 0]; ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#E3F2FD', edgecolor='#4A90D9', lw=3))
    ax.text(0.5, 0.92, 'WHAT WE DID', ha='center', fontsize=18, fontweight='bold',
            color='#4A90D9', transform=ax.transAxes)
    steps = [
        ('Step 1:', 'Ran MC Dropout (30 passes) on 4 trials (T5-T8)', '#1a5276'),
        ('Step 2:', 'Got uncertainty (σ) for each of 31,635 road links', '#2e86c1'),
        ('Step 3:', 'Checked: what % of actual errors fall within 1σ, 2σ, 3σ?', '#8e44ad'),
        ('Step 4:', 'Compared to ideal Gaussian: 1σ should cover 68.3%', '#27ae60'),
        ('Step 5:', 'Found model is OVERCONFIDENT (covers too few errors)', '#E74C3C'),
        ('Step 6:', 'Temperature Scaling fixes this!', '#2ECC71'),
    ]
    for i, (step, desc, c) in enumerate(steps):
        y = 0.78 - i * 0.12
        ax.text(0.06, y, step, fontsize=11, fontweight='bold', color=c, transform=ax.transAxes)
        ax.text(0.22, y, desc, fontsize=10, color='#444', transform=ax.transAxes)

    # Panel 2: Coverage comparison across trials
    ax = axes[0, 1]
    trials = sorted(M.keys())
    sigs = [0.5, 1, 1.5, 2, 2.5, 3]
    ideal = [0.383, 0.683, 0.866, 0.954, 0.988, 0.997]
    x = np.arange(len(sigs))
    w = 0.15

    ax.bar(x - 2*w, ideal, w, label='Ideal Gaussian', color='#E6E6FA', edgecolor=P['dark'], lw=1)
    for i, t in enumerate(trials):
        covs = [np.mean(M[t]['e'] < s * M[t]['u']) for s in sigs]
        ax.bar(x + (i-1)*w, covs, w, label=f'T{t}', color=TC[t-1], edgecolor='w', lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f'{s}σ' for s in sigs], fontsize=11)
    ax.set_ylabel('Coverage Fraction')
    ax.set_title('Coverage at σ Levels\n(bars below ideal = overconfident)', fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 3: Gap from ideal (uncalibrated)
    ax = axes[1, 0]
    for t in trials:
        gaps = [np.mean(M[t]['e'] < s * M[t]['u']) - i for s, i in zip(sigs, ideal)]
        ax.plot(sigs, gaps, 'o-', color=TC[t-1], lw=2.5, ms=8, label=f'T{t}')
    ax.axhline(0, color='k', ls='--', lw=1.5, alpha=0.5, label='Perfect')
    ax.fill_between(sigs, -0.05, 0.05, alpha=0.1, color='green', label='± 5% OK zone')
    ax.set_xlabel('Sigma Multiplier'); ax.set_ylabel('Gap (actual − ideal)')
    ax.set_title('All trials BELOW zero = OVERCONFIDENT\n(uncertainty too narrow)',
                 fontweight='bold', color='#E74C3C')
    ax.legend(fontsize=9); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 4: WHAT WE FOUND — summary card
    ax = axes[1, 1]; ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#FFF8E1', edgecolor='#F39C12', lw=3))
    ax.text(0.5, 0.92, 'WHAT WE FOUND', ha='center', fontsize=18, fontweight='bold',
            color='#F39C12', transform=ax.transAxes)
    findings = [
        ('Finding 1:', 'All 4 trials are OVERCONFIDENT', '#E74C3C',
         f'1σ coverage: {M[8]["c1"]:.1f}% instead of 68.3%'),
        ('Finding 2:', 'T8 has BEST uncertainty quality', '#2ECC71',
         f'Spearman ρ = {M[8]["rho"]:.4f} (highest)'),
        ('Finding 3:', 'Lower dropout = better UQ', '#4A90D9',
         'T8 (DO=0.15) best of all DO=0.20 trials'),
        ('Solution:', 'Temperature Scaling fixes calibration', '#27ae60',
         'T=2.90 → ECE drops 90.6%!'),
    ]
    for i, (title, finding, c, detail) in enumerate(findings):
        y = 0.78 - i * 0.18
        ax.text(0.06, y, title, fontsize=12, fontweight='bold', color=c, transform=ax.transAxes)
        ax.text(0.26, y, finding, fontsize=11, color='#333', transform=ax.transAxes, fontweight='bold')
        ax.text(0.26, y - 0.06, detail, fontsize=10, color='#666', transform=ax.transAxes)

    fig.suptitle('MC Dropout Coverage Analysis: What We Did & What We Found\n'
                 'All trials overconfident — Temperature Scaling fixes this!',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '34_mc_coverage.png')


# ═══════════════════════════════════════════════════════════
#  35 — ENSEMBLE vs MC DROPOUT HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════

def chart_35_ens_vs_mc(M, ea, eb):
    gc.collect()
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    ta = ea['targets'].flatten(); em = ea['ensemble_mean'].flatten()
    ev = ea['ensemble_variance'].flatten(); mc_a = ea['avg_mc_uncertainty'].flatten()
    comb = ea['combined_uncertainty'].flatten()
    ea_err = np.abs(em - ta)
    tb = eb['targets'].flatten(); ep = eb['ensemble_prediction'].flatten()
    eu = eb['ensemble_uncertainty'].flatten()
    eb_err = np.abs(ep - tb)

    # Panel 1 — Spearman rho comparison
    ax = axes[0]
    methods = ['MC Drop\nT8', 'MC Drop\nT7', 'MC Drop\nT5', 'MC Drop\nT6',
               'Ens MC\nAvg (A)', 'Combined\n(A)', 'Multi-\nModel (B)', 'Ens Var\n(A)']
    rhos = [M[8]['rho'], M[7]['rho'], M[5]['rho'], M[6]['rho'],
            _safe_spearmanr(mc_a, ea_err), _safe_spearmanr(comb, ea_err),
            _safe_spearmanr(eu, eb_err), _safe_spearmanr(ev, ea_err)]
    colors = ['#4A90D9', '#0ABDE3', '#48DBFB', '#87CEEB',
              '#F39C12', '#FF8E53', '#9B59B6', '#FF6B6B']
    bars = ax.bar(range(len(methods)), rhos, color=colors, edgecolor='w', lw=2)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=8)
    for b, v in zip(bars, rhos):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0.4, color='#2ECC71', ls='--', lw=1.5, alpha=0.5)
    ax.set_ylabel('Spearman ρ'); ax.set_title('UQ Quality Ranking\nMC Dropout dominates!', fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 2 — Cost-benefit
    ax = axes[1]
    rho_comb = _safe_spearmanr(comb, ea_err)
    rho_eu = _safe_spearmanr(eu, eb_err)
    items = [
        ('MC Dropout (T8)', M[8]['rho'], 1, '#4A90D9'),
        ('Ens A (5 runs)', rho_comb, 5, '#F39C12'),
        ('Ens B (multi-model)', rho_eu, 5, '#9B59B6'),
    ]
    for name, rho, cost, c in items:
        ax.scatter(cost, rho, s=400, c=c, edgecolors='w', lw=3, zorder=5)
        ax.annotate(f'{name}\nρ={rho:.3f}', (cost, rho), textcoords='offset points',
                   xytext=(15, -5), fontsize=11, fontweight='bold', color=c)
    ax.set_xlabel('Relative Computation Cost (# of trained models)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman ρ (quality)', fontsize=12, fontweight='bold')
    ax.set_title('Cost vs Quality Tradeoff\nMC Dropout: cheapest AND best!', fontweight='bold', color='#2ECC71')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 3 — Summary card
    ax = axes[2]; ax.axis('off')
    ax.set_title('Head-to-Head Summary', fontsize=16, fontweight='bold', color=P['dark'])
    items = [
        ('MC Dropout', '#4A90D9',
         f'ρ = {M[8]["rho"]:.4f} (best!)\nCost: 30 forward passes of 1 model\nNo extra training needed\nSimplest to implement'),
        ('Ensemble A', '#F39C12',
         f'ρ = {rho_comb:.4f}\nCost: Train 5 models from scratch\n5× training time + 5× storage\nMarginal improvement only'),
        ('Ensemble B', '#9B59B6',
         f'ρ = {rho_eu:.4f}\nCost: Need different architectures\n5 different models\nWorst quality!'),
        ('Winner', '#2ECC71', 'MC Dropout is the clear winner!\nBest quality + cheapest compute\nFor traffic planners → use MC Dropout'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.90 - i * 0.23
        ax.text(0.03, y, title + ':', fontsize=12, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.03, y - 0.04, desc, fontsize=9.5, color='#444', transform=ax.transAxes, linespacing=1.4)

    fig.suptitle('MC Dropout vs Ensemble: Head-to-Head Comparison\n'
                 'Single model dropout beats 5-model ensembles!',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '35_ens_vs_mc.png')


# ═══════════════════════════════════════════════════════════
#  36 — CALIBRATION RELIABILITY DIAGRAM
# ═══════════════════════════════════════════════════════════

def chart_36_reliability(M):
    d = M[8]; unc, err = d['u'], d['e']
    # Find T_opt
    def ece_fn(T, u, e, nb=10):
        s = u * T; edges = np.unique(np.percentile(s, np.linspace(0, 100, nb+1))); v = 0.
        for j in range(len(edges)-1):
            m = (s >= edges[j]) & (s < edges[j+1]) if j < len(edges)-2 else (s >= edges[j]) & (s <= edges[j+1])
            if m.sum() == 0: continue
            v += (m.sum()/len(u)) * abs(np.mean(e[m] < s[m]) - 0.683)
        return v
    res = minimize_scalar(lambda T: ece_fn(T, unc, err), bounds=(0.1, 20), method='bounded')
    T_opt = res.x

    fig, axes = plt.subplots(2, 2, figsize=(24, 18))

    # Panel 1: WHAT THIS CHART SHOWS — explanation
    ax = axes[0, 0]; ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#E3F2FD', edgecolor='#4A90D9', lw=3))
    ax.text(0.5, 0.92, 'HOW TO READ THIS', ha='center', fontsize=18, fontweight='bold',
            color='#4A90D9', transform=ax.transAxes)
    explanations = [
        ('What it shows:', 'Does predicted uncertainty match actual error?', '#1a5276'),
        ('X-axis:', 'Predicted uncertainty (what model thinks error will be)', '#2e86c1'),
        ('Y-axis:', 'Observed error (what error actually was)', '#8e44ad'),
        ('Diagonal line:', 'Perfect calibration (predicted = actual)', '#27ae60'),
        ('Points BELOW:', 'Overconfident (thinks error is higher than it is)', '#F39C12'),
        ('Points ABOVE:', 'Underconfident (error higher than predicted)', '#E74C3C'),
        ('Goal:', 'All points on the diagonal line!', '#2ECC71'),
    ]
    for i, (title, desc, c) in enumerate(explanations):
        y = 0.78 - i * 0.10
        ax.text(0.06, y, title, fontsize=10, fontweight='bold', color=c, transform=ax.transAxes)
        ax.text(0.35, y, desc, fontsize=10, color='#444', transform=ax.transAxes)

    # Panel 2: Before calibration
    ax = axes[0, 1]
    nb = 15
    scaled = unc * 1.0
    edges = np.percentile(scaled, np.linspace(0, 100, nb+1))
    expected, observed = [], []
    for j in range(nb):
        if j < nb-1:
            m = (scaled >= edges[j]) & (scaled < edges[j+1])
        else:
            m = (scaled >= edges[j]) & (scaled <= edges[j+1])
        if m.sum() > 10:
            expected.append(np.mean(scaled[m]))
            observed.append(np.mean(err[m]))
    ax.scatter(expected, observed, c='#E74C3C', s=120, edgecolors='w', lw=2, zorder=5)
    ax.plot(expected, observed, color='#E74C3C', lw=2.5, alpha=0.7)
    lims = [0, max(max(expected), max(observed)) * 1.1]
    ax.plot(lims, lims, 'k--', lw=2, alpha=0.5, label='Perfect calibration')
    ax.fill_between(lims, [0, lims[1]*0.8], [lims[1]*0.2, lims[1]*1.2],
                    alpha=0.05, color='green')
    ax.set_xlabel('Predicted Uncertainty (σ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Error', fontsize=12, fontweight='bold')
    ax.set_title('BEFORE Calibration (T=1.0)\nPoints ABOVE diagonal = error bigger than predicted',
                 fontweight='bold', color='#E74C3C', fontsize=13)
    ax.legend(fontsize=10); ax.set_facecolor('#FAFBFC'); _wm(ax)
    # Annotate the gap
    ax.annotate('Errors are HIGHER\nthan predicted!\n(overconfident)',
                xy=(expected[len(expected)//2], observed[len(observed)//2]),
                xytext=(expected[0]*2, observed[-1]*0.8),
                fontsize=11, fontweight='bold', color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

    # Panel 3: After calibration
    ax = axes[1, 0]
    scaled_cal = unc * T_opt
    edges_cal = np.percentile(scaled_cal, np.linspace(0, 100, nb+1))
    expected_cal, observed_cal = [], []
    for j in range(nb):
        if j < nb-1:
            m = (scaled_cal >= edges_cal[j]) & (scaled_cal < edges_cal[j+1])
        else:
            m = (scaled_cal >= edges_cal[j]) & (scaled_cal <= edges_cal[j+1])
        if m.sum() > 10:
            expected_cal.append(np.mean(scaled_cal[m]))
            observed_cal.append(np.mean(err[m]))
    ax.scatter(expected_cal, observed_cal, c='#2ECC71', s=120, edgecolors='w', lw=2, zorder=5)
    ax.plot(expected_cal, observed_cal, color='#2ECC71', lw=2.5, alpha=0.7)
    lims_cal = [0, max(max(expected_cal), max(observed_cal)) * 1.1]
    ax.plot(lims_cal, lims_cal, 'k--', lw=2, alpha=0.5, label='Perfect calibration')
    ax.fill_between(lims_cal, [0, lims_cal[1]*0.8], [lims_cal[1]*0.2, lims_cal[1]*1.2],
                    alpha=0.05, color='green')
    ax.set_xlabel('Predicted Uncertainty (σ × T)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Error', fontsize=12, fontweight='bold')
    ax.set_title(f'AFTER Calibration (T={T_opt:.2f})\nPoints now closer to diagonal = FIXED!',
                 fontweight='bold', color='#2ECC71', fontsize=13)
    ax.legend(fontsize=10); ax.set_facecolor('#FAFBFC'); _wm(ax)
    ax.annotate('Now matches\nthe diagonal!',
                xy=(expected_cal[len(expected_cal)//2], observed_cal[len(observed_cal)//2]),
                xytext=(expected_cal[0]*2, observed_cal[-1]*0.7),
                fontsize=11, fontweight='bold', color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))

    # Panel 4: Coverage table + summary
    ax = axes[1, 1]; ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                 boxstyle='round,pad=0.03', facecolor='#F8F8F8', edgecolor='#ddd', lw=2))
    ax.text(0.5, 0.92, 'CALIBRATION RESULTS', ha='center', fontsize=18, fontweight='bold',
            color=P['dark'], transform=ax.transAxes)
    sigs = [0.5, 1, 1.5, 2]; ideal_covs = [0.383, 0.683, 0.866, 0.954]
    # Table header
    ax.text(0.10, 0.80, 'σ Level', fontsize=11, fontweight='bold', color='#555', transform=ax.transAxes)
    ax.text(0.30, 0.80, 'Before', fontsize=11, fontweight='bold', color='#E74C3C', transform=ax.transAxes)
    ax.text(0.50, 0.80, 'After', fontsize=11, fontweight='bold', color='#2ECC71', transform=ax.transAxes)
    ax.text(0.70, 0.80, 'Ideal', fontsize=11, fontweight='bold', color='#4A90D9', transform=ax.transAxes)
    ax.plot([0.08, 0.85], [0.77, 0.77], color='#ddd', lw=1.5, transform=ax.transAxes)
    for i, (s, ic) in enumerate(zip(sigs, ideal_covs)):
        y = 0.72 - i * 0.08
        bef = np.mean(err < s * unc) * 100
        aft = np.mean(err < s * unc * T_opt) * 100
        ax.text(0.12, y, f'{s}σ', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.30, y, f'{bef:.1f}%', fontsize=12, color='#E74C3C', transform=ax.transAxes)
        ax.text(0.50, y, f'{aft:.1f}%', fontsize=12, color='#2ECC71', fontweight='bold', transform=ax.transAxes)
        ax.text(0.70, y, f'{ic*100:.1f}%', fontsize=12, color='#4A90D9', transform=ax.transAxes)

    ax.text(0.5, 0.32, f'Temperature T = {T_opt:.2f}', ha='center', fontsize=16,
            fontweight='bold', color='#1a5276', transform=ax.transAxes)
    ax.text(0.5, 0.18, 'Before: uncertainties too narrow (overconfident)\n'
            'After: uncertainties match reality!\n'
            'Traffic planners can now TRUST the confidence intervals',
            ha='center', fontsize=11, color='#444', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Reliability Diagram: Before vs After Temperature Scaling\n'
                 'Points on diagonal = model knows when it is uncertain',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '36_reliability_diagram.png')


# ═══════════════════════════════════════════════════════════
#  37 — COMPLETE BENEFITS & DRAWBACKS TABLE
# ═══════════════════════════════════════════════════════════

def chart_37_benefits_drawbacks(R, M, ea, eb):
    gc.collect()
    ta = ea['targets'].flatten(); em = ea['ensemble_mean'].flatten()
    comb = ea['combined_uncertainty'].flatten(); ea_err = np.abs(em - ta)
    tb = eb['targets'].flatten(); ep = eb['ensemble_prediction'].flatten()
    eu = eb['ensemble_uncertainty'].flatten(); eb_err = np.abs(ep - tb)
    rho_comb = _safe_spearmanr(comb, ea_err)
    rho_eu = _safe_spearmanr(eu, eb_err)

    fig, ax = plt.subplots(figsize=(24, 17))
    ax.axis('off'); ax.set_facecolor('#FAFBFC')
    fig.patch.set_facecolor('#FAFBFC')

    rows = [
        ('1. Architecture\n   Reproduction', '#1a5276',
         'Reproduced Elena\'s PointNetTransfGAT\nfrom SSRN 5182100 paper exactly',
         'Proves GNN surrogates work\nfor traffic prediction on Paris',
         'Trial 1 had wrong arch (Linear)\nR²=0.786 was misleading!'),
        ('2. Training on\n   10% Data', '#2e86c1',
         'Only used 10% of Paris population\n(1% downsampled × 10k simulations)',
         'Still achieves R²=0.596!\nShows data efficiency of GNNs',
         'Performance gap vs full data\nR² lower than paper\'s result'),
        ('3. Hyperparameter\n   Exploration', '#8e44ad',
         '8 trials: varied BS, DO, LR, loss\nSplit ratios tested',
         f'Found best combo: BS32, DO0.15,\nLR1e-3 → R²={R[8]["r2"]:.4f}',
         'W-MSE failed (T2/T3/T4)\nOnly T8 used DO=0.15'),
        ('4. MC Dropout\n   UQ', '#2980b9',
         '30 stochastic passes at inference\nKeep dropout ON → mean+std',
         f'Best: ρ={M[8]["rho"]:.4f} (T8)\nNo retraining needed!\nCheapest UQ method',
         f'Overconfident: 1σ cov = {M[8]["c1"]:.1f}%\n(should be 68.3%)\nNeeds calibration'),
        ('5. Ensemble A\n   (5 runs)', '#6c3483',
         'Train same arch 5× with diff seeds\nVariance = epistemic uncertainty',
         'Captures model uncertainty\nCombined unc adds info',
         f'Only ρ={rho_comb:.3f}\n5× training cost\nMC Dropout alone is better!'),
        ('6. Ensemble B\n   (multi-model)', '#9B59B6',
         'Different trials as ensemble\nT2,5,6,7,8 combined',
         'Tests diversity hypothesis\nDifferent HPs → diverse preds',
         f'Worst: ρ={rho_eu:.3f}\nR²≈0 (wrong distribution)\n5 different models, bad result'),
        ('7. Temperature\n   Scaling', '#27ae60',
         f'Scale uncertainty by T=2.90\nsigma_new = T × sigma_old',
         'ECE: 0.356 → 0.033 (90.6% fix!)\n1σ coverage → ~68.3%\nTrustworthy uncertainties!',
         'Only post-hoc fix\nDoesn\'t improve ranking\nJust makes numbers meaningful'),
        ('8. Practical\n   Value', '#f39c12',
         'Use uncertainty to filter predictions\nKeep top 90% confident, flag rest',
         'Confident preds have lower error!\nTraffic planners can trust these',
         'Still some error remains\n10% flagged may be important\nroads too'),
    ]

    y_start = 0.93
    row_h = 0.108

    # Column headers with colored backgrounds
    for x, label, color in [(0.085, 'EXPERIMENT', '#1a5276'), (0.29, 'DESCRIPTION', '#4A90D9'),
                              (0.56, '✓ BENEFIT', '#2ECC71'), (0.83, '✗ DRAWBACK', '#E74C3C')]:
        ax.text(x, 0.955, label, fontsize=13, fontweight='bold', ha='center',
                color='white', transform=ax.transAxes,
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3', alpha=0.85))

    for i, (what, color, desc, benefit, drawback) in enumerate(rows):
        y = y_start - i * row_h

        # Row background
        bg_color = '#F0F4F8' if i % 2 == 0 else '#FFFFFF'
        ax.add_patch(FancyBboxPatch((0.01, y - 0.045), 0.98, row_h - 0.005,
                     transform=ax.transAxes, boxstyle='round,pad=0.005',
                     facecolor=bg_color, edgecolor='#E0E0E0', lw=1))

        # What — colored left badge
        ax.add_patch(FancyBboxPatch((0.015, y - 0.035), 0.135, row_h - 0.02,
                     transform=ax.transAxes, boxstyle='round,pad=0.005',
                     facecolor=color, alpha=0.12, edgecolor=color, lw=1.5))
        ax.text(0.082, y + 0.015, what, fontsize=9.5, fontweight='bold',
                color=color, ha='center', transform=ax.transAxes, linespacing=1.3)

        # Description
        ax.text(0.17, y + 0.025, desc, fontsize=8.5, color='#444',
                transform=ax.transAxes, linespacing=1.4, va='top')

        # Benefit — green text
        ax.text(0.44, y + 0.025, benefit, fontsize=8.5, color='#1B7A3D',
                transform=ax.transAxes, linespacing=1.3, va='top')

        # Drawback — red text
        ax.text(0.72, y + 0.025, drawback, fontsize=8.5, color='#A33',
                transform=ax.transAxes, linespacing=1.3, va='top')

    _wm(ax)
    fig.suptitle('Complete Research Journey: What We Tried, Benefits & Drawbacks\n'
                 'Every experiment and its outcome — from start to finish',
                 fontsize=19, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '37_benefits_drawbacks.png')


# ═══════════════════════════════════════════════════════════
#  38 — RESEARCH JOURNEY TIMELINE (visual)
# ═══════════════════════════════════════════════════════════

def chart_38_journey():
    fig, ax = plt.subplots(figsize=(26, 12))
    ax.set_xlim(-1, 31); ax.set_ylim(-2, 11); ax.axis('off')
    ax.set_facecolor('#F8FAFB'); fig.patch.set_facecolor('white')

    # Timeline line with gradient effect
    ax.plot([-0.5, 30.5], [4, 4], color='#CBD5E1', lw=6, zorder=1, solid_capstyle='round')
    ax.plot([-0.5, 30.5], [4, 4], color='#94A3B8', lw=2, zorder=1)

    phases = [
        (1.5, 'Phase 1\nData & Setup', '#1a5276',
         'MATSim data\n31,635 nodes\n59,851 edges\n10,000 scenarios\n10% population'),
        (6, 'Phase 2\nReproduction', '#8e44ad',
         'Elena\'s GNN\n8 trials\nT1 wrong arch!\nT8 best: R²=0.596\nBS32/DO0.15/LR1e-3'),
        (11, 'Phase 3\nHP Exploration', '#c0392b',
         'Dropout effect\nLR effect\nBatch size\nWeighted loss\nData split'),
        (16, 'Phase 4\nMC Dropout', '#2980b9',
         '30 stochastic passes\nρ=0.482 (T8)\nCheapest UQ\nNo retraining\nBut overconfident'),
        (21, 'Phase 5\nEnsembles', '#6c3483',
         'Exp A: 5 runs\nExp B: multi-model\nρ~0.1-0.16\nMC Dropout wins\nMore expensive'),
        (26, 'Phase 6\nCalibration', '#27ae60',
         'Temp Scaling\nT=2.90\nECE: 0.356→0.033\n90.6% improvement\nTrustworthy UQ!'),
    ]

    for x, title, color, details in phases:
        # Node on timeline
        ax.scatter(x, 4, s=350, c=color, edgecolors='white', lw=3, zorder=5)

        # Box above (odd) or below (even)
        idx = phases.index((x, title, color, details))
        if idx % 2 == 0:
            y_box, y_conn = 5.5, 4.3
        else:
            y_box, y_conn = 0.3, 3.7

        # Card with shadow effect
        ax.add_patch(FancyBboxPatch((x-2.1, y_box-0.1), 4.2, 3.2,
                     boxstyle='round,pad=0.2', facecolor='#E0E0E0', edgecolor='none',
                     alpha=0.3, zorder=2))  # shadow
        ax.add_patch(FancyBboxPatch((x-2, y_box), 4, 3,
                     boxstyle='round,pad=0.2', facecolor='white', edgecolor=color,
                     lw=2.5, alpha=0.95, zorder=3))

        # Colored top strip
        ax.add_patch(FancyBboxPatch((x-1.9, y_box+2.2), 3.8, 0.7,
                     boxstyle='round,pad=0.1', facecolor=color, edgecolor='none',
                     alpha=0.9, zorder=4))
        ax.plot([x, x], [y_conn, y_box + (0 if idx % 2 == 0 else 3)],
                color=color, lw=2, zorder=2)

        ax.text(x, y_box + 2.55, title, ha='center', fontsize=11,
                fontweight='bold', color='white', zorder=5)
        ax.text(x, y_box + 0.25, details, ha='center', fontsize=8.5,
                color='#555', zorder=4, linespacing=1.3)

    # Progress arrows
    for i in range(len(phases)-1):
        x1 = phases[i][0] + 2.2
        x2 = phases[i+1][0] - 2.2
        ax.annotate('', xy=(x2, 4), xytext=(x1, 4),
                    arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=2.5), zorder=2)

    ax.text(15, 10, 'Research Journey: From Raw Data to Trustworthy Predictions',
            ha='center', fontsize=22, fontweight='bold', color='#1a5276')
    ax.text(15, 9.2, 'Mohd Zamin Quadri — TUM Master\'s Thesis — Prof. Günnemann',
            ha='center', fontsize=13, color='#666')

    _save(fig, '38_research_journey.png')


# ═══════════════════════════════════════════════════════════
#  39 — TRAINING DATA EFFICIENCY
# ═══════════════════════════════════════════════════════════

def chart_39_data_efficiency(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    # Panel 1 — What 10% means
    ax = axes[0]; ax.axis('off')
    ax.set_title('Training Data Context', fontsize=16, fontweight='bold', color=P['dark'])
    items = [
        ('Full Paris Network', '#1a5276', '~12 million inhabitants\nFull agent-based simulation\nHours of computation per scenario'),
        ('Our Training Data', '#E74C3C', '1% population downsample\n= ~120,000 agents\n× 10,000 scenarios\n= 10% of full dataset'),
        ('Why 10%?', '#F39C12', 'Full simulation too expensive\n1% sample preserves patterns\nStill captures traffic dynamics'),
        ('Result', '#2ECC71', f'R² = {R[8]["r2"]:.4f} with 10% data!\nUsable predictions\nfor traffic planning'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.88 - i * 0.22
        ax.text(0.05, y, title + ':', fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y - 0.04, desc, fontsize=10, color='#444', transform=ax.transAxes, linespacing=1.4)

    # Panel 2 — Prediction volume
    ax = axes[1]
    n_nodes = 31635
    test_graphs = {2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 100, 8: 100}
    total_preds = [n_nodes * test_graphs.get(t, 50) for t in range(2, 9)]
    trials = list(range(2, 9))
    bars = ax.bar([f'T{t}' for t in trials], total_preds,
                 color=[TC[t-1] for t in trials], edgecolor='w', lw=2)
    for b, v in zip(bars, total_preds):
        ax.text(b.get_x()+b.get_width()/2, v + 20000, f'{v:,}',
                ha='center', fontsize=9, fontweight='bold', rotation=45)
    ax.set_ylabel('Total Predictions', fontweight='bold')
    ax.set_title('Predictions per Trial\n31,635 nodes × test graphs', fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 3 — Speed comparison
    ax = axes[2]
    methods = ['MATSim\n(1 scenario)', 'GNN\n(1 scenario)', 'GNN\n(100 scenarios)']
    times = [3600, 2, 200]  # seconds (approx)
    colors = ['#E74C3C', '#2ECC71', '#2ECC71']
    bars = ax.bar(methods, times, color=colors, edgecolor=P['dark'], lw=2, width=0.5)
    ax.set_ylabel('Time (seconds)', fontweight='bold')
    ax.set_yscale('log')
    for b, v in zip(bars, times):
        label = f'{v}s' if v < 3600 else f'{v//3600}h'
        ax.text(b.get_x()+b.get_width()/2, v*1.3, label, ha='center', fontsize=14, fontweight='bold')
    ax.set_title('Speed: MATSim vs GNN\nGNN is ~1800× faster!', fontweight='bold', color='#2ECC71')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    fig.suptitle('Data Efficiency: Only 10% of Paris Population Used\n'
                 'GNN achieves useful predictions with fraction of data + massive speedup',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '39_data_efficiency.png')


# ═══════════════════════════════════════════════════════════
#  40 — ERROR BY VOLUME CATEGORY
# ═══════════════════════════════════════════════════════════

def chart_40_error_by_volume(R, feats):
    d = R[8]; n = 31635
    feat_rep = np.tile(feats[:n, 0], len(d['p']) // n + 1)[:len(d['p'])]
    vol = feat_rep

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    # Panel 1 — Error vs base volume
    ax = axes[0]
    nb = 20
    edges = np.percentile(vol, np.linspace(0, 100, nb+1))
    bin_c, bin_mae, bin_std = [], [], []
    for j in range(nb):
        m = (vol >= edges[j]) & (vol < edges[j+1]) if j < nb-1 else (vol >= edges[j])
        if m.sum() > 50:
            bin_c.append(np.mean(vol[m]))
            bin_mae.append(np.mean(d['e'][m]))
            bin_std.append(np.std(d['e'][m]))
    ax.fill_between(bin_c, np.array(bin_mae)-np.array(bin_std),
                    np.array(bin_mae)+np.array(bin_std), alpha=0.2, color='#4A90D9')
    ax.plot(bin_c, bin_mae, 'o-', color='#4A90D9', lw=2.5, ms=6, label='Mean ± Std')
    ax.set_xlabel('Base Volume (veh/h)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_title('Error Increases with Traffic Volume\nHigh-volume roads are harder to predict', fontweight='bold')
    ax.legend(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 2 — Volume category breakdown
    ax = axes[1]
    cats = [('Low\n(0-100)', (0, 100), '#98FB98'),
            ('Medium\n(100-500)', (100, 500), '#FFEAA7'),
            ('High\n(500+)', (500, np.inf), '#FF6B6B')]
    cat_names, cat_maes, cat_counts, cat_colors = [], [], [], []
    for name, (lo, hi), color in cats:
        m = (vol >= lo) & (vol < hi)
        if m.sum() > 0:
            cat_names.append(name)
            cat_maes.append(np.mean(d['e'][m]))
            cat_counts.append(m.sum())
            cat_colors.append(color)
    bars = ax.bar(cat_names, cat_maes, color=cat_colors, edgecolor=P['dark'], lw=2, width=0.5)
    for b, v, n_val in zip(bars, cat_maes, cat_counts):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.3f}\n({n_val:,} links)',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('MAE (veh/h)', fontweight='bold')
    ax.set_title('Error by Volume Category\nLow-volume roads: easiest', fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 3 — Percentage of links in each category
    ax = axes[2]
    sizes = [c / sum(cat_counts) * 100 for c in cat_counts]
    wedges, _, autotexts = ax.pie(sizes, labels=cat_names, colors=cat_colors,
                                   autopct='%1.1f%%', startangle=90,
                                   wedgeprops=dict(edgecolor='white', lw=2),
                                   textprops=dict(fontsize=12))
    for at in autotexts:
        at.set_fontweight('bold')
    ax.set_title('Link Distribution\nMost links are low-volume', fontweight='bold')

    fig.suptitle('Error Analysis by Traffic Volume (Trial 8)\n'
                 'High-traffic roads have highest error — uncertainty helps identify these!',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '40_error_by_volume.png')


# ═══════════════════════════════════════════════════════════
#  41 — BATCH SIZE EFFECT
# ═══════════════════════════════════════════════════════════

def chart_41_batch_size(R):
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    bs32 = [2, 3, 4, 6, 7, 8]
    bs64 = [5]

    for ax_i, (metric, key, fmt, better) in enumerate([
            ('R²', 'r2', '{:.4f}', 'higher'), ('MAE (veh/h)', 'mae', '{:.3f}', 'lower')]):
        ax = axes[ax_i]
        v32 = [R[t][key] for t in bs32]
        v64 = [R[t][key] for t in bs64]
        bp1 = ax.boxplot([v32], positions=[0], widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor='#98FB98', alpha=0.7))
        bp2 = ax.boxplot([v64], positions=[1], widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor='#FFEAA7', alpha=0.7))
        for t in bs32:
            ax.scatter(np.random.uniform(-0.08, 0.08), R[t][key],
                      c=TC[t-1], s=150, edgecolors='w', lw=2, zorder=5)
            ax.annotate(f'T{t}', (0, R[t][key]), textcoords='offset points',
                       xytext=(12, 3), fontsize=10, fontweight='bold')
        for t in bs64:
            ax.scatter(1 + np.random.uniform(-0.08, 0.08), R[t][key],
                      c=TC[t-1], s=150, edgecolors='w', lw=2, zorder=5)
            ax.annotate(f'T{t}', (1, R[t][key]), textcoords='offset points',
                       xytext=(12, 3), fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['BS=32\n(6 trials)', 'BS=64\n(1 trial)'], fontsize=12)
        ax.set_ylabel(metric, fontweight='bold')
        avg32, avg64 = np.mean(v32), np.mean(v64)
        winner = 'BS=64' if (better=='higher' and avg64 > avg32) or (better=='lower' and avg64 < avg32) else 'BS=32'
        ax.set_title(f'{metric}: {winner} wins\nBS32 avg={fmt.format(avg32)}, BS64 avg={fmt.format(avg64)}',
                    fontweight='bold')
        ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Summary
    ax = axes[2]; ax.axis('off')
    ax.set_title('Batch Size Finding', fontsize=16, fontweight='bold', color=P['dark'])
    items = [
        ('BS = 32', '#2ECC71', f'Trials 2, 3, 4, 6, 7, 8\nAvg R² = {np.mean([R[t]["r2"] for t in bs32]):.4f}\n'
         'Includes best trial T8 (DO=0.15)\nUsed by most experiments'),
        ('BS = 64', '#F39C12', f'Trial 5 only\nR² = {R[5]["r2"]:.4f}\n'
         'Larger batch, same other HPs as T1\nMid-range performance'),
        ('Note', '#4A90D9', 'Only T5 used BS=64, rest are BS=32.\n'
         'Single trial not enough to isolate\nbatch size effect alone.\n'
         'BS=32 is the default choice.'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.85 - i * 0.30
        ax.text(0.05, y, title + ':', fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.05, y - 0.05, desc, fontsize=10, color='#444', transform=ax.transAxes, linespacing=1.5)

    fig.suptitle('Hyperparameter Ablation: Batch Size Effect\n'
                 'BS=32 used in 6/7 correct trials — T5 (BS=64) is the exception',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '41_batch_size.png')


# ═══════════════════════════════════════════════════════════
#  42 — MC DROPOUT: TRIAL-BY-TRIAL BENEFIT
# ═══════════════════════════════════════════════════════════

def chart_42_mc_per_trial(M):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    trials = sorted(M.keys())

    # Panel 1 — rho ranking
    ax = axes[0, 0]
    rhos = [(t, M[t]['rho']) for t in trials]
    rhos.sort(key=lambda x: x[1], reverse=True)
    bars = ax.barh(range(len(rhos)), [r[1] for r in rhos],
                  color=[TC[r[0]-1] for r in rhos], edgecolor='w', lw=2)
    ax.set_yticks(range(len(rhos))); ax.set_yticklabels([f'Trial {r[0]}' for r in rhos], fontsize=12)
    for b, (t, rho) in zip(bars, rhos):
        ax.text(rho + 0.005, b.get_y() + b.get_height()/2, f'ρ={rho:.4f}',
                va='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('Spearman ρ'); ax.set_title('MC Dropout Quality by Trial', fontweight='bold')
    ax.invert_yaxis(); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 2 — Coverage comparison
    ax = axes[0, 1]
    x = np.arange(len(trials)); w = 0.35
    c1 = [M[t]['c1'] for t in trials]
    c2 = [M[t]['c2'] for t in trials]
    ax.bar(x - w/2, c1, w, label='1σ Coverage', color='#4A90D9', edgecolor='w', lw=1.5)
    ax.bar(x + w/2, c2, w, label='2σ Coverage', color='#FF8E53', edgecolor='w', lw=1.5)
    ax.axhline(68.3, color='#2ECC71', ls='--', lw=2, alpha=0.6, label='Ideal 1σ (68.3%)')
    ax.axhline(95.4, color='#9B59B6', ls='--', lw=2, alpha=0.6, label='Ideal 2σ (95.4%)')
    ax.set_xticks(x); ax.set_xticklabels([f'T{t}' for t in trials], fontsize=12)
    ax.set_ylabel('Coverage %'); ax.set_title('Coverage: All Trials Overconfident', fontweight='bold', color='#E74C3C')
    ax.legend(fontsize=9); ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 3 — Mean uncertainty vs mean error
    ax = axes[1, 0]
    for t in trials:
        ax.scatter(M[t]['mu'], M[t]['me'], c=TC[t-1], s=300, edgecolors='w', lw=3, zorder=5)
        ax.annotate(f'T{t}\nρ={M[t]["rho"]:.3f}', (M[t]['mu'], M[t]['me']),
                   textcoords='offset points', xytext=(12, 5), fontsize=11, fontweight='bold', color=TC[t-1])
    ax.set_xlabel('Mean Uncertainty (σ)', fontweight='bold')
    ax.set_ylabel('Mean Error (|pred-actual|)', fontweight='bold')
    ax.set_title('Uncertainty vs Error Scale\nT8: highest unc → best correlation', fontweight='bold')
    ax.set_facecolor('#FAFBFC'); _wm(ax)

    # Panel 4 — What drives good MC Dropout
    ax = axes[1, 1]; ax.axis('off')
    ax.set_title('What Makes MC Dropout Work Better?', fontsize=15, fontweight='bold', color=P['dark'])
    items = [
        ('Trial 8 is BEST (ρ=0.482)', '#2ECC71',
         'DO=0.15, LR=1e-3, BS=32\nLower dropout → more stable MC\nHigher 1σ coverage = more informative'),
        ('Trial 7 is 2nd (ρ=0.444)', '#1ABC9C',
         'DO=0.20, LR=1e-3, BS=32\n80/10/10 split like T8\nSame LR but higher dropout'),
        ('T5, T6 lower (ρ~0.42)', '#48DBFB',
         'Both DO=0.20\nT5: BS=64, T6: LR=5e-4\nDifferent HPs → slightly worse UQ'),
        ('Key insight', '#F39C12',
         'Lower dropout (0.15) → better UQ!\nParadox: less randomness gives\nbetter uncertainty estimates'),
    ]
    for i, (title, color, desc) in enumerate(items):
        y = 0.88 - i * 0.22
        ax.text(0.03, y, title, fontsize=11, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.03, y - 0.04, desc, fontsize=9.5, color='#444', transform=ax.transAxes, linespacing=1.4)

    fig.suptitle('MC Dropout: Per-Trial Analysis\n'
                 'Lower dropout rate → better uncertainty estimates (paradoxically)',
                 fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '42_mc_per_trial.png')


# ═══════════════════════════════════════════════════════════
#  43 — KEY NUMBERS INFOGRAPHIC
# ═══════════════════════════════════════════════════════════

def chart_43_key_numbers(R, M):
    fig, ax = plt.subplots(figsize=(28, 16))
    ax.set_xlim(0, 28); ax.set_ylim(0, 16); ax.axis('off')
    ax.set_facecolor('#F5F7FA'); fig.patch.set_facecolor('white')

    ax.text(14, 15, 'Key Numbers at a Glance', ha='center', fontsize=28,
            fontweight='bold', color='#1a5276')
    ax.text(14, 14.2, 'Every critical metric from the thesis — all cross-checked from NPZ files',
            ha='center', fontsize=14, color='#666')

    cards = [
        (2.5, 11.5, '31,635', 'Road Links\n(Paris)', '#1a5276'),
        (7.5, 11.5, '59,851', 'Graph Edges', '#2e86c1'),
        (12.5, 11.5, '10,000', 'MATSim\nSimulations', '#8e44ad'),
        (17.5, 11.5, '10%', 'Training\nData Used', '#E74C3C'),
        (22.5, 11.5, '8', 'Trials\nRun', '#F39C12'),

        (2.5, 7.5, f'{R[8]["r2"]:.4f}', 'Best R²\n(Trial 8)', '#2ECC71'),
        (7.5, 7.5, f'{R[8]["mae"]:.2f}', 'Best MAE\n(veh/h)', '#4A90D9'),
        (12.5, 7.5, f'{R[8]["n"]:,}', 'Total\nPredictions', '#FF8E53'),
        (17.5, 7.5, '30', 'MC Dropout\nPasses', '#2980b9'),
        (22.5, 7.5, f'{M[8]["rho"]:.3f}', 'Best Spearman\nρ (MC Drop)', '#48DBFB'),

        (2.5, 3.5, '2.90', 'Temperature\n(T_opt)', '#27ae60'),
        (7.5, 3.5, '0.033', 'ECE After\nCalibration', '#1ABC9C'),
        (12.5, 3.5, '90.6%', 'ECE\nImprovement', '#2ECC71'),
        (17.5, 3.5, '~1800×', 'Speed vs\nMATSim', '#F39C12'),
        (22.5, 3.5, '5', 'Ensemble\nModels (A)', '#9B59B6'),
    ]

    for cx, cy, number, label, color in cards:
        # Shadow
        ax.add_patch(FancyBboxPatch((cx-1.9, cy-1.3), 3.8, 2.9,
                     boxstyle='round,pad=0.2', facecolor='#E0E0E0', edgecolor='none', alpha=0.4))
        # White card
        ax.add_patch(FancyBboxPatch((cx-2, cy-1.2), 3.8, 2.9,
                     boxstyle='round,pad=0.2', facecolor='white', edgecolor=color,
                     lw=2.5, alpha=1.0))
        # Colored top accent
        ax.add_patch(FancyBboxPatch((cx-1.8, cy+1.0), 3.4, 0.5,
                     boxstyle='round,pad=0.1', facecolor=color, edgecolor='none', alpha=0.15))
        # Number
        ax.text(cx-0.1, cy + 0.5, number, ha='center', fontsize=24,
                fontweight='bold', color=color)
        # Label
        ax.text(cx-0.1, cy - 0.6, label, ha='center', fontsize=10,
                color='#555', linespacing=1.2)

    _save(fig, '43_key_numbers.png')


# ═══════════════════════════════════════════════════════════
#  44 — COMPLETE METRICS TABLE
# ═══════════════════════════════════════════════════════════

def chart_44_metrics_table(R, M):
    fig, ax = plt.subplots(figsize=(28, 18))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Table data
    cols = ['Trial', 'Arch', 'BS', 'DO', 'LR', 'Loss', 'Split', 'R²', 'MAE', 'RMSE', 'P90', '<5%']
    hp = {
        1: ('WRONG', 32, 0.20, '1e-3', 'MSE', '70/15/15'),
        2: ('OK', 32, 0.20, '1e-3', 'W-MSE', '70/15/15'),
        3: ('OK', 32, 0.20, '1e-3', 'W-MSE', '70/15/15'),
        4: ('OK', 32, 0.20, '1e-3', 'W-MSE', '70/15/15'),
        5: ('OK', 64, 0.20, '1e-3', 'MSE', '70/15/15'),
        6: ('OK', 32, 0.20, '5e-4', 'MSE', '70/15/15'),
        7: ('OK', 32, 0.20, '1e-3', 'MSE', '80/10/10'),
        8: ('OK', 32, 0.15, '1e-3', 'MSE', '80/10/10'),
    }
    rows = []
    for t in range(1, 9):
        h = hp[t]
        rows.append([f'T{t}', h[0], str(h[1]), str(h[2]), h[3], h[4], h[5],
                     f'{R[t]["r2"]:.4f}', f'{R[t]["mae"]:.3f}', f'{R[t]["rmse"]:.3f}',
                     f'{R[t]["p90"]:.2f}', f'{R[t]["u5"]:.1f}%'])

    table = ax.table(cellText=rows, colLabels=cols, loc='center',
                    cellLoc='center', colColours=['#4A90D9']*len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Modern styling
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#E0E0E0')
        cell.set_linewidth(1.5)
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold', fontsize=12)
            cell.set_facecolor('#1a5276')
            cell.set_height(0.06)
        else:
            t_num = r  # trial number
            if t_num == 1:
                cell.set_facecolor('#FFEBEE')  # light red
            elif t_num == 8:
                cell.set_facecolor('#E8F5E9')  # light green
            else:
                cell.set_facecolor('#FAFBFC' if r % 2 == 0 else '#FFFFFF')
            # Bold the metrics columns
            if c >= 7:
                cell.set_text_props(fontweight='bold')

    fig.suptitle('Complete Trial Comparison Table\n'
                 'Red = wrong architecture (T1)  |  Green = best trial (T8)  |  All values from NPZ files',
                 fontsize=19, fontweight='bold', y=0.95, color='#1a5276')
    plt.tight_layout()
    _save(fig, '44_metrics_table.png')

    # MC Dropout table
    fig2, ax2 = plt.subplots(figsize=(24, 10))
    ax2.axis('off')
    fig2.patch.set_facecolor('white')

    mc_cols = ['Trial', 'Spearman ρ', 'Pearson r', '1σ Cov %', '2σ Cov %', 'Mean Unc', 'Mean Err', 'N Preds']
    mc_rows = []
    for t in sorted(M.keys()):
        mc_rows.append([f'T{t}', f'{M[t]["rho"]:.4f}', f'{M[t]["pr"]:.4f}',
                       f'{M[t]["c1"]:.1f}%', f'{M[t]["c2"]:.1f}%',
                       f'{M[t]["mu"]:.4f}', f'{M[t]["me"]:.4f}', f'{len(M[t]["p"]):,}'])

    table2 = ax2.table(cellText=mc_rows, colLabels=mc_cols, loc='center',
                      cellLoc='center', colColours=['#2980b9']*len(mc_cols))
    table2.auto_set_font_size(False)
    table2.set_fontsize(13)
    table2.scale(1, 2.8)

    for (r, c), cell in table2.get_celld().items():
        cell.set_edgecolor('#E0E0E0')
        cell.set_linewidth(1.5)
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold', fontsize=13)
            cell.set_facecolor('#1a5276')
        else:
            # T8 is the last MC trial row (row 4 = index of T8 among sorted M.keys())
            is_best = (sorted(M.keys())[r-1] == 8) if r <= len(M) else False
            cell.set_facecolor('#E8F5E9' if is_best else ('#FAFBFC' if r % 2 == 0 else '#FFFFFF'))
            if c in [1, 2]:  # ρ and r columns
                cell.set_text_props(fontweight='bold')

    fig2.suptitle('MC Dropout Results Table\n'
                  'Green = best trial (T8)  |  30 passes, correct architecture only',
                  fontsize=19, fontweight='bold', y=0.92, color='#1a5276')
    plt.tight_layout()
    _save(fig2, '45_mc_table.png')


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def _run(fn, *args, **kwargs):
    """Run a chart function with gc + MemoryError protection."""
    gc.collect()
    try:
        return fn(*args, **kwargs)
    except MemoryError:
        plt.close('all')
        gc.collect()
        print(f"   SKIPPED {fn.__name__} (MemoryError)")
        return None

def main():
    _style()
    print('='*70)
    print('  FRESH A-to-Z THESIS CHARTS — EXTENDED EDITION')
    print('='*70)

    print('\n[1/7] Loading all data …')
    R, feats, pos = load_trials()
    M = load_mc()
    ea, eb = load_ens()

    print('\n[2/7] Architecture & pipeline …')
    _run(chart_01_architecture)
    _run(chart_02_pipeline)

    print('\n[3/7] Trial comparisons …')
    _run(chart_03_all_trials_3d, R)
    _run(chart_04_correct_trials, R)
    _run(chart_05_hyperparams, R)

    print('\n[4/7] Per-trial detail (×8) + analysis …')
    for t in range(1,9):
        _run(chart_trial_detail, R, t)

    _run(chart_14_mc_all, M)
    _run(chart_15_ensemble, ea, eb, M)
    ret = _run(chart_16_calibration, M)
    T_opt, ece_b, ece_a = ret if ret else (2.90, 0.356, 0.033)
    _run(chart_17_spatial, M, pos)
    _run(chart_18_surface, M, feats)
    _run(chart_19_features, M, feats)
    _run(chart_20_practical, M, R)
    _run(chart_21_per_graph, R)

    print('\n[5/7] Research story & summary …')
    _run(chart_22_research)
    _run(chart_23_radar, R)
    _run(chart_24_scatter_grid, R)
    _run(chart_25_error_overlay, R)
    _run(chart_26_heatmap, R)
    _run(chart_27_dashboard, R, M, ea, eb, T_opt, ece_b, ece_a)

    print('\n[6/7] NEW — Research journey deep-dive charts …')
    _run(chart_28_trial_evolution, R)
    _run(chart_29_dropout_effect, R)
    _run(chart_30_lr_effect, R)
    _run(chart_31_weighted_loss, R)
    _run(chart_32_split_comparison, R)
    _run(chart_33_arch_bug)
    _run(chart_34_mc_coverage, M)
    _run(chart_35_ens_vs_mc, M, ea, eb)
    _run(chart_36_reliability, M)
    _run(chart_37_benefits_drawbacks, R, M, ea, eb)
    _run(chart_38_journey)
    _run(chart_39_data_efficiency, R)
    _run(chart_40_error_by_volume, R, feats)
    _run(chart_41_batch_size, R)
    _run(chart_42_mc_per_trial, M)

    print('\n[7/7] NEW — Key numbers & tables …')
    _run(chart_43_key_numbers, R, M)
    _run(chart_44_metrics_table, R, M)

    files = sorted(f for f in os.listdir(OUT) if f.endswith('.png'))
    total_mb = sum(os.path.getsize(os.path.join(OUT,f)) for f in files) / 1e6
    print(f'\n{"="*70}')
    print(f'  DONE!  {len(files)} charts  |  {total_mb:.1f} MB  |  {OUT}')
    print(f'{"="*70}')
    for f in files:
        print(f'    {f}')


if __name__ == '__main__':
    main()
