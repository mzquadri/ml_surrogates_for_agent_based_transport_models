import wandb

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import Sequential as GeoSequential, TransformerConv, GATConv, PointNetConv, global_mean_pool

from gnn.models.base_gnn import BaseGNN

"""
PointNetTransfGATQuantile
=========================
Identical architecture to PointNetTransfGAT (T8) except the final GATConv
layer outputs 2 channels instead of 1:
  channel 0 -> q_lo  (lower quantile, tau = alpha/2)
  channel 1 -> q_hi  (upper quantile, tau = 1 - alpha/2)

These raw outputs are returned as separate tensors of shape [N, 1] each.
Monotonicity ordering (min/max) is NOT applied here; it is applied
at inference time in the evaluation and calibration scripts, following
the yromano/cqr reference implementation (LearnerOptimizedCrossing.predict).

Used for Conformalized Quantile Regression (CQR).
Reference: Romano, Patterson & Candes (2019), arXiv:1905.03222
           yromano/cqr (github.com/yromano/cqr)

Author: Mohd Zamin Quadri (CQR UQ extension)
"""


class PointNetTransfGATQuantile(BaseGNN):
    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 2,
                 point_net_conv_layer_structure_local_mlp: list = [256],
                 point_net_conv_layer_structure_global_mlp: list = [512],
                 gat_conv_layer_structure: list = [128, 256, 512],
                 dropout: float = 0.3,
                 use_dropout: bool = False,
                 predict_mode_stats: bool = False,
                 dtype: torch.dtype = torch.float32,
                 log_to_wandb: bool = False):

        """
        Initialize the quantile GNN model.

        Parameters:
        - in_channels (int): Number of input channels (5 for T8 features).
        - out_channels (int): Must be 2 (q_lo and q_hi). Fixed at 2.
        - point_net_conv_layer_structure_local_mlp (list): Local MLP layer sizes.
        - point_net_conv_layer_structure_global_mlp (list): Global MLP layer sizes.
        - gat_conv_layer_structure (list): GAT intermediate layer sizes.
        - dropout (float): Dropout rate. T8 uses 0.2.
        - use_dropout (bool): Whether to apply dropout.
        - predict_mode_stats (bool): Not used. Kept for BaseGNN compatibility.
        - dtype (torch.dtype): Data type for computations.
        - log_to_wandb (bool): Whether to log to Weights & Biases.
        """
        # Force out_channels=2 for quantile outputs
        super().__init__(
            in_channels=in_channels,
            out_channels=2,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            log_to_wandb=log_to_wandb)

        # Architecture-specific parameters
        self.pnc_local  = point_net_conv_layer_structure_local_mlp
        self.pnc_global = point_net_conv_layer_structure_global_mlp
        self.gat_conv   = gat_conv_layer_structure

        if self.log_to_wandb:
            wandb.config.update({"pnc_local": self.pnc_local,
                                 "pnc_global": self.pnc_global,
                                 "gat_conv": self.gat_conv},
                                allow_val_change=True)

        # Define the layers of the model
        self.define_layers()

        # Initialize weights
        self.initialize_weights()

    def define_layers(self):

        # Initialize dropout if needed
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(self.dropout)

        # PointNet layers
        # Use start + end pos (pos shape: [N, 3, 2])
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

        # GAT layers
        layers_global = self.define_gat_layers()
        self.gat_graph_layers = GeoSequential('x, edge_index', layers_global)

        # Output layer: 2 channels (q_lo, q_hi)
        # CHANGE from T8: GATConv(64, 1) -> GATConv(64, 2)
        self.gat_final = GATConv(64, 2)

    def forward(self, data):
        """
        Forward pass for the quantile GNN model.

        Parameters:
        - data (Data): Input data containing node features and edge indices.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]:
            q_lo: Lower quantile predictions, shape [N, 1]
            q_hi: Upper quantile predictions, shape [N, 1]

        NOTE: No min/max ordering applied here. Monotonicity enforcement
        is applied at inference time in evaluation/calibration scripts.
        """
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        # Use start + end pos (pos shape: [N, 3, 2] for start, end, midpoint)
        pos1 = data.pos[:, 0, :]  # Start position
        pos2 = data.pos[:, 1, :]  # End position

        x = self.point_net_conv_1(x, pos1, edge_index)
        x = self.point_net_conv_2(x, pos2, edge_index)

        x = self.gat_graph_layers(x, edge_index)

        # node_predictions shape: [N, 2]
        node_predictions = self.gat_final(x, edge_index)

        # Return as separate tensors, each [N, 1]
        q_lo = node_predictions[:, 0:1]   # lower quantile
        q_hi = node_predictions[:, 1:2]   # upper quantile

        return q_lo, q_hi

    def define_gat_layers(self):
        """
        Define layers for GATConv based on configuration.
        Identical to PointNetTransfGAT.define_gat_layers().

        Returns:
        - List: Layers for GATConv.
        """
        layers = []
        for idx in range(len(self.gat_conv) - 1):
            # Transformer layer
            layers.append((TransformerConv(self.gat_conv[idx], int(self.gat_conv[idx + 1] / 4), heads=4), 'x, edge_index -> x'))
            layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                layers.append(self.dropout_layer)
        layers.append((GATConv(self.gat_conv[-1], 64), 'x, edge_index -> x'))
        return layers

    def create_point_net_layer(self, gat_conv_starts_with_layer: int, is_first_layer: bool = False, is_last_layer: bool = False):
        """
        Create PointNetConv layers with specified configurations.
        Identical to PointNetTransfGAT.create_point_net_layer().

        Parameters:
        - gat_conv_starts_with_layer (int): Starting layer size for GATConv.
        - is_first_layer (bool): Whether this is the first PointNet layer.
        - is_last_layer (bool): Whether this is the last PointNet layer.

        Returns:
        - PointNetConv: Configured PointNet layer.
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
        for idx in range(len(self.pnc_local) - 1):
            local_MLP_layers.append(nn.Linear(self.pnc_local[idx], self.pnc_local[idx + 1]))
            local_MLP_layers.append(nn.ReLU())
            if self.use_dropout:
                local_MLP_layers.append(self.dropout_layer)
        local_MLP = nn.Sequential(*local_MLP_layers)

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

    # WEIGHT INITIALIZATION (Override)
    def initialize_weights(self):
        """
        Initialize model weights using Xavier and Kaiming initialization.
        Identical to PointNetTransfGAT.initialize_weights().
        """
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
