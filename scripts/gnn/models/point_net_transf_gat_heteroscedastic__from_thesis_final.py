"""
Heteroscedastic PointNetTransfGAT Model
========================================
Extension of PointNetTransfGAT for heteroscedastic regression.

Outputs [mean, log_var] for each node prediction, enabling:
1. Aleatoric uncertainty quantification (learned variance)
2. Epistemic uncertainty via MC Dropout (as before)
3. Total uncertainty decomposition: sigma_total^2 = sigma_aleatoric^2 + sigma_epistemic^2

Key changes from original PointNetTransfGAT:
- Line 95: GATConv(64, 1) -> GATConv(64, 2)
- forward() returns (mean, log_var) instead of single prediction

References:
- Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
  https://arxiv.org/abs/1703.04977
- Original PointNetTransfGAT: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100

Author: Mohd Zamin Quadri (heteroscedastic extension for thesis UQ)
"""

import wandb

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import Sequential as GeoSequential, TransformerConv, GATConv, PointNetConv, global_mean_pool

from gnn.models.base_gnn import BaseGNN


class PointNetTransfGATHeteroscedastic(BaseGNN):
    """
    Heteroscedastic version of PointNetTransfGAT.
    
    Architecture identical to original T8 model, except:
    - Final GAT layer outputs 2 channels: [mean, log_var]
    - forward() returns tuple: (mean, log_var)
    
    Use with HeteroscedasticGaussianLoss for training.
    """
    
    def __init__(self, 
                in_channels: int = 5, 
                out_channels: int = 2,  # CHANGED: 2 outputs (mean, log_var) instead of 1
                point_net_conv_layer_structure_local_mlp: list = [256], 
                point_net_conv_layer_structure_global_mlp: list = [512], 
                gat_conv_layer_structure: list = [128, 256, 512], 
                dropout: float = 0.2,  # T8 uses 0.2 dropout
                use_dropout: bool = True,
                predict_mode_stats: bool = False,
                dtype: torch.dtype = torch.float32,
                log_to_wandb: bool = False):
        
        """
        Initialize heteroscedastic GNN model.
        
        Parameters:
        - in_channels (int): Number of input features (5 for T8)
        - out_channels (int): Number of output channels (2 for [mean, log_var])
        - point_net_conv_layer_structure_local_mlp (list): Local MLP structure [256]
        - point_net_conv_layer_structure_global_mlp (list): Global MLP structure [512]
        - gat_conv_layer_structure (list): GAT layer structure [128, 256, 512]
        - dropout (float): Dropout rate (0.2 for T8)
        - use_dropout (bool): Enable dropout (True for MC Dropout)
        - predict_mode_stats (bool): Enable mode stats prediction (False for T8)
        - dtype (torch.dtype): Data type (float32)
        - log_to_wandb (bool): Enable W&B logging
        """
        # Call parent class constructor
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)
        
        # Architecture-specific parameters (same as T8)
        self.pnc_local = point_net_conv_layer_structure_local_mlp
        self.pnc_global = point_net_conv_layer_structure_global_mlp
        self.gat_conv = gat_conv_layer_structure

        if self.log_to_wandb:
            wandb.config.update({
                "pnc_local": self.pnc_local,
                "pnc_global": self.pnc_global,
                "gat_conv": self.gat_conv,
                "heteroscedastic": True
            }, allow_val_change=True)

        # Define layers
        self.define_layers()

        # Initialize weights
        self.initialize_weights()
        
    def define_layers(self):
        """Define model layers (identical to T8 except final GAT output size)."""
        
        # Initialize dropout if needed
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        # PointNet layers (use start + end pos)
        self.point_net_conv_1 = self.create_point_net_layer(
            gat_conv_starts_with_layer=self.gat_conv[0], 
            is_first_layer=True, 
            is_last_layer=False
        )
        self.point_net_conv_2 = self.create_point_net_layer(
            gat_conv_starts_with_layer=self.gat_conv[0], 
            is_first_layer=False, 
            is_last_layer=True
        )
        
        # GAT layers (same as T8)
        layers_global = self.define_gat_layers()
        self.gat_graph_layers = GeoSequential('x, edge_index', layers_global)
        
        # *** KEY CHANGE: Output layer now outputs 2 values (mean, log_var) ***
        self.gat_final = GATConv(64, 2)  # CHANGED from GATConv(64, 1)
        
        # Mode stats predictor (if enabled, not used in T8)
        if self.predict_mode_stats:
            self.mode_stat_predictor = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=2),
                nn.Linear(64, 2)
            )

    def forward(self, data):
        """
        Forward pass for heteroscedastic GNN.
        
        Parameters:
        - data (Data): Input graph data with node features, edge indices, positions
        
        Returns:
        - mean (torch.Tensor): Predicted means, shape (N, 1)
        - log_var (torch.Tensor): Predicted log variances, shape (N, 1)
        
        OR if predict_mode_stats=True:
        - (mean, log_var, mode_stats_pred)
        """
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        # Use start + end pos (pos shape: [N, 3, 2])
        pos1 = data.pos[:, 0, :]  # Start position
        pos2 = data.pos[:, 1, :]  # End position
        x = self.point_net_conv_1(x, pos1, edge_index)
        x = self.point_net_conv_2(x, pos2, edge_index)
        
        # GAT layers
        x = self.gat_graph_layers(x, edge_index)
        
        # Final output: [mean, log_var]
        node_predictions = self.gat_final(x, edge_index)  # Shape: (N, 2)
        
        # Split into mean and log_var
        mean = node_predictions[:, 0:1]      # Shape: (N, 1)
        log_var = node_predictions[:, 1:2]   # Shape: (N, 1)
        
        # NUMERICAL SAFETY: Check for NaN/Inf in outputs
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            raise ValueError(
                f"Mean predictions contain NaN or Inf! "
                f"mean range: [{mean.min().item():.2f}, {mean.max().item():.2f}]"
            )
        if torch.isnan(log_var).any() or torch.isinf(log_var).any():
            raise ValueError(
                f"Log variance predictions contain NaN or Inf! "
                f"log_var range: [{log_var.min().item():.2f}, {log_var.max().item():.2f}]"
            )
        
        # Mode stats prediction (if enabled)
        if self.predict_mode_stats:
            mode_stats = data.mode_stats
            batch = data.batch
            pooled_node_predictions = global_mean_pool(x, batch)
            shape_node_preds = pooled_node_predictions.shape[0]
            shape_mode_stats = int(mode_stats.shape[0] / shape_node_preds)
            
            tensor_for_pooling = torch.repeat_interleave(
                torch.arange(shape_node_preds), shape_mode_stats
            ).to(x.device)
            mode_stats_pooled = global_mean_pool(mode_stats, tensor_for_pooling)
            
            mode_stats_pred = self.mode_stat_predictor(mode_stats_pooled)
            mode_stats_pred = mode_stats_pred.repeat_interleave(shape_mode_stats, dim=0)
            
            return mean, log_var, mode_stats_pred
        
        return mean, log_var
    
    def define_gat_layers(self):
        """
        Define GAT layers (identical to T8).
        
        Returns:
        - List: Layers for GATConv
        """
        layers = []
        for idx in range(len(self.gat_conv) - 1):      
            # Transformer layer
            layers.append((
                TransformerConv(self.gat_conv[idx], int(self.gat_conv[idx + 1]/4), heads=4), 
                'x, edge_index -> x'
            ))
            layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                layers.append(self.dropout_layer)
        layers.append((GATConv(self.gat_conv[-1], 64), 'x, edge_index -> x'))
        return layers
    
    def create_point_net_layer(self, gat_conv_starts_with_layer:int, 
                               is_first_layer:bool=False, is_last_layer:bool=False):
        """
        Create PointNetConv layers (identical to T8).
        
        Parameters:
        - gat_conv_starts_with_layer (int): Starting layer size
        - is_first_layer (bool): First PointNet layer flag
        - is_last_layer (bool): Last PointNet layer flag
        
        Returns:
        - PointNetConv: Configured PointNet layer
        """
        offset_due_to_pos = 2
        local_MLP_layers = []
        
        if is_first_layer:
            local_MLP_layers.append(nn.Linear(self.in_channels + offset_due_to_pos, self.pnc_local[0]))
        else:
            local_MLP_layers.append(nn.Linear(self.pnc_global[-1] + offset_due_to_pos, self.pnc_local[0]))
        
        local_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            local_MLP_layers.append(self.dropout_layer)
        
        for idx in range(len(self.pnc_local)-1):
            local_MLP_layers.append(nn.Linear(self.pnc_local[idx], self.pnc_local[idx + 1]))
            local_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                local_MLP_layers.append(self.dropout_layer)
        
        local_MLP = nn.Sequential(*local_MLP_layers)
        
        # Global MLP
        global_MLP_layers = []
        global_MLP_layers.append(nn.Linear(self.pnc_local[-1], self.pnc_global[0]))
        global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        
        for idx in range(len(self.pnc_global) - 1):
            global_MLP_layers.append(nn.Linear(self.pnc_global[idx], self.pnc_global[idx + 1]))
            global_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                global_MLP_layers.append(self.dropout_layer)
                
        if is_last_layer:
            global_MLP_layers.append(nn.Linear(self.pnc_global[-1], gat_conv_starts_with_layer))
        else:
            global_MLP_layers.append(nn.Linear(self.pnc_global[-1], self.pnc_global[-1]))
        
        global_MLP_layers.append(nn.ReLU())
        if self.use_dropout:
            global_MLP_layers.append(self.dropout_layer)
        
        global_MLP = nn.Sequential(*global_MLP_layers)
        return PointNetConv(local_nn=local_MLP, global_nn=global_MLP)
    
    # WEIGHT INITIALIZATION (Override, identical to T8)
    def initialize_weights(self):
        """Initialize model weights using Xavier and Kaiming initialization."""
        super().initialize_weights()  # Call parent class method for Linear layers
        for m in self.modules():
            if isinstance(m, PointNetConv):
                self._initialize_pointnetconv(m)
            elif isinstance(m, GATConv):
                self._initialize_gatconv(m)

    def _initialize_pointnetconv(self, m: PointNetConv):
        """Initialize weights for PointNetConv layers."""
        for name, param in m.local_nn.named_parameters():
            if param.dim() > 1:  # weight parameters
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:  # bias parameters
                init.zeros_(param)
        for name, param in m.global_nn.named_parameters():
            if param.dim() > 1:  # weight parameters
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:  # bias parameters
                init.zeros_(param)

    def _initialize_gatconv(self, m: GATConv):
        """Initialize weights for GATConv layers."""
        if hasattr(m, 'lin') and m.lin is not None:
            init.xavier_normal_(m.lin.weight)
            if m.lin.bias is not None:
                init.zeros_(m.lin.bias)
        if hasattr(m, 'att_src') and m.att_src is not None:
            init.xavier_normal_(m.att_src)
        if hasattr(m, 'att_dst') and m.att_dst is not None:
            init.xavier_normal_(m.att_dst)
