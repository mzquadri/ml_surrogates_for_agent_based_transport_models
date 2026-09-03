=============================================================
HETEROSCEDASTIC REGRESSION - IMPLEMENTATION PLAN
For: MSc Thesis - Uncertainty Quantification for GNN Surrogates
Author: Mohd Zamin Quadri
=============================================================

WHAT IS IT:
- Modify the PointNetTransfGAT model to output TWO values per node:
  (1) mean prediction (same as now), (2) learned log-variance (aleatoric uncertainty)
- Retrain with Gaussian Negative Log-Likelihood (NLL) loss instead of MSE
- This captures aleatoric (data) uncertainty: which road segments have
  inherently noisy/variable traffic predictions

WHY IT'S RELEVANT:
- Some roads (arterials near intersections) have inherently more variable traffic
- Some roads (residential streets) are stable
- Current MC Dropout captures epistemic uncertainty (model ignorance)
- Heteroscedastic regression captures aleatoric uncertainty (data noise)
- Together they give a complete uncertainty picture

CURRENT MODEL (T8):
- Architecture: PointNetTransfGAT
- Output layer: self.gat_final = GATConv(64, 1)  --> outputs 1 value per node
- Loss: MSE (torch.nn.MSELoss)
- Hyperparams: batch_size=8, grad_accum=3, dropout=0.2, lr=0.0005, split=80/10/10
- Training: ~1000 max epochs, early stopping patience=25
- Hardware: NVIDIA T4 GPU (Google Colab)
- Results: R2=0.5957, MAE=3.96 veh/h

WHAT NEEDS TO CHANGE:

1. MODEL CODE (scripts/gnn/models/point_net_transf_gat.py):
   - Line 95: Change `self.gat_final = GATConv(64, 1)` to `GATConv(64, 2)`
   - In forward(): split output into mean and log_variance:
     out = self.gat_final(x, edge_index)  # shape: [N, 2]
     mean = out[:, 0]
     log_var = out[:, 1]
     return mean, log_var

2. LOSS FUNCTION (scripts/gnn/help_functions.py):
   - Add Gaussian NLL loss alongside GNN_Loss class:
     def heteroscedastic_loss(mean, log_var, target):
         precision = torch.exp(-log_var)
         return torch.mean(precision * (target - mean)**2 + log_var)
   - This is: (1/sigma^2) * (y - mu)^2 + log(sigma^2)

3. TRAINING LOOP (scripts/gnn/models/base_gnn.py):
   - Modify to handle 2-output model
   - Use heteroscedastic NLL loss instead of MSE
   - Keep everything else same (optimizer, scheduler, early stopping)

4. TRAINING SCRIPT (scripts/training/run_models.py):
   - Add flag: --heteroscedastic True
   - Command to run:
     python run_models.py \
       --gnn_arch point_net_transf_gat \
       --unique_model_description point_net_transf_gat_9th_trial_heteroscedastic \
       --in_channels 5 --use_all_features False \
       --num_epochs 1000 --lr 0.0005 \
       --early_stopping_patience 25 \
       --use_dropout True --dropout 0.2 \
       --batch_size 8 --gradient_accumulation_steps 3

5. EVALUATION:
   - MC Dropout inference: run S=30 passes, get mean predictions + learned variance
   - Now you have TWO uncertainty sources:
     * Epistemic: variance across MC passes (same as before)
     * Aleatoric: learned log_var output (new)
   - Total uncertainty = epistemic + aleatoric
   - Rerun all UQ metrics (Spearman rho, CRPS, ECE, conformal, selective, etc.)

6. THESIS UPDATES:
   - Chapter 3: Add Method 7 (Heteroscedastic Regression) section
   - Chapter 5: Add results section comparing aleatoric vs epistemic
   - Chapter 6: Update discussion with decomposition analysis
   - Recompile PDF

TIME ESTIMATE:
- Code changes: ~1-2 hours
- Retraining on Colab T4: ~2-3 hours
- MC Dropout inference (S=30, 100 graphs): ~4 hours
- UQ evaluation pipeline: ~1-2 hours
- Figure generation: ~1 hour
- Thesis text updates: ~2-3 hours
- PDF recompile + delivery folder update: ~30 min
- TOTAL: ~12-16 hours (2-3 days realistically)

RISKS:
- NLL loss is harder to optimize; variance can collapse
- R2 may drop below 0.5957 (point accuracy vs calibration tradeoff)
- Colab Pro session = 24hr max (training+inference = ~7hr, should fit)
- All existing thesis numbers remain valid (this is a NEW trial T9)

KEY FILES TO MODIFY:
- scripts/gnn/models/point_net_transf_gat.py  (output layer)
- scripts/gnn/help_functions.py                (loss function)
- scripts/gnn/models/base_gnn.py               (training loop)
- scripts/training/run_models.py               (training entry point)

KEY FILES TO READ FOR CONTEXT:
- data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/
    test_evaluation_complete.json              (T8 baseline metrics)
- scripts/gnn/models/point_net_transf_gat.py   (full model architecture)
- scripts/gnn/help_functions.py                (GNN_Loss class, MC dropout)
- thesis/latex_tum_official/chapters/03_methodology.tex (Method descriptions)
- thesis/latex_tum_official/bibliography.bib   (already has amini2020deep ref)

REFERENCE PAPER:
- Nix & Weigend (1994) "Estimating the mean and variance of the target
  probability distribution" - foundational heteroscedastic regression
- Already cited in thesis: Lakshminarayanan et al. (2017) uses
  heteroscedastic NLL for deep ensembles

REPO ROOT: C:\Users\zamin\OneDrive\Desktop\Nazim_thesis\
           ml_surrogates_for_agent_based_transport_models\
=============================================================
