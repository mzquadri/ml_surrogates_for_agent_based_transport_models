import os, datetime, glob

BASE = os.path.join('data', 'TR-C_Benchmarks')
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

def fdate(path):
    if os.path.exists(path):
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    return "NOT FOUND"

print("=== TRAINING TIMELINE ===")
for t, f in sorted(FOLDERS.items()):
    d = os.path.join(BASE, f)
    model = os.path.join(d, 'trained_model', 'model.pth')
    test  = os.path.join(d, 'test_predictions.npz')
    print(f"T{t}: model.pth={fdate(model)}  test_pred={fdate(test)}")

print("\n=== UQ RESULTS TIMELINE ===")
for t in [2, 5, 6, 7, 8]:
    f = FOLDERS[t]
    uq_dir = os.path.join(BASE, f, 'uq_results')
    if os.path.isdir(uq_dir):
        files = sorted(glob.glob(os.path.join(uq_dir, '*.npz')) + glob.glob(os.path.join(uq_dir, '*.json')))
        for fp in files:
            print(f"  T{t}: {os.path.basename(fp)} = {fdate(fp)}")

print("\n=== ENSEMBLE EXPERIMENTS ===")
ens_dir = os.path.join(BASE, FOLDERS[8], 'uq_results', 'ensemble_experiments')
if os.path.isdir(ens_dir):
    for fp in sorted(glob.glob(os.path.join(ens_dir, '*'))):
        if os.path.isfile(fp):
            print(f"  {os.path.basename(fp)} = {fdate(fp)}")

print("\n=== WANDB ===")
wandb_dir = 'wandb'
if os.path.isdir(wandb_dir):
    for item in os.listdir(wandb_dir):
        fp = os.path.join(wandb_dir, item)
        print(f"  {item} = {fdate(fp)}")

print("\n=== KEY SCRIPT DATES ===")
scripts = [
    'scripts/training/train.py',
    'scripts/gnn/pointnet_transf_gat.py',
    'scripts/evaluation/evaluate.py',
    'scripts/data_preprocessing/preprocess.py',
    'docs/visuals/verification/generate_all.py',
    'thesis/THESIS_MAIN.md',
]
for s in scripts:
    if os.path.exists(s):
        print(f"  {s} = {fdate(s)}")
    else:
        # Try to find it
        matches = glob.glob(f"**/{os.path.basename(s)}", recursive=True)
        if matches:
            print(f"  {matches[0]} = {fdate(matches[0])}")

print("\n=== THESIS LATEX DATES ===")
for fp in sorted(glob.glob('thesis/latex_tum_official/chapters/*.tex')):
    print(f"  {os.path.basename(fp)} = {fdate(fp)}")

print("\n=== FOLDER CREATION DATES ===")
for t, f in sorted(FOLDERS.items()):
    d = os.path.join(BASE, f)
    if os.path.isdir(d):
        ct = datetime.datetime.fromtimestamp(os.path.getctime(d)).strftime("%Y-%m-%d")
        print(f"  T{t} folder created: {ct}")
