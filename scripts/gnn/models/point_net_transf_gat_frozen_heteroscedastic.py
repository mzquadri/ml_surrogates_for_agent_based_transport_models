"""
PointNetTransfGATFrozenHeteroscedastic
======================================
Trial 9-v2 Frozen Backbone Heteroscedastic model.

Architecture:
  - Backbone: identical to T8 (point_net_conv_1, point_net_conv_2, gat_graph_layers)
    Loaded from T8 checkpoint and frozen (no gradient updates).
  - Head: new GATConv(64 -> 2) (trainable only)
    channel 0 -> mu       (predicted mean)
    channel 1 -> log_var  (predicted log variance)
    Returns (mu, log_var) each shape [N, 1].

CRITICAL DIFFERENCE from PointNetTransfGATFrozenCQR (T11):
  - T11 overrides train() to force backbone into eval() mode, so dropout is
    always OFF in the backbone.  This is correct for CQR where MC Dropout
    through the backbone is not needed.
  - This class does NOT override train().  When model.train() is called,
    backbone dropout layers are ACTIVE.  This is required for:
      1. MC Dropout inference (epistemic uncertainty via stochastic backbone)
      2. Regularization during head training (backbone dropout acts as noise)
  - Backbone weights are still FROZEN (requires_grad=False).
  - Forward still runs backbone under torch.no_grad() (saves memory -- no
    autograd graph for intermediate backbone tensors).
  - torch.no_grad() does NOT disable dropout; dropout is controlled by the
    module training mode (model.train() vs model.eval()).

Key remapping:
  Old PyG checkpoint stores GATConv weights as '.lin.weight'.
  PyG 2.3.1 expects '.lin_src.weight' and '.lin_dst.weight'.
  Remapping is applied automatically inside _load_backbone().

T8 checkpoint (READ-ONLY):
  data/TR-C_Benchmarks/point_net_transf_gat_8th_trial_lower_dropout/trained_model/model.pth

Used for Heteroscedastic Gaussian NLL + MC Dropout UQ, Trial 9-v2.
References:
  - Kendall & Gal (2017) NIPS: "What Uncertainties Do We Need in Bayesian DL?"
  - Seitzer (2022): faithful heteroscedastic regression (beta-NLL)

Author: Mohd Zamin Quadri
"""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    Sequential as GeoSequential,
    TransformerConv,
    GATConv,
    PointNetConv,
)


class PointNetTransfGATFrozenHeteroscedastic(nn.Module):
    """
    Frozen-backbone heteroscedastic GNN for Trial 9-v2.

    Backbone (point_net_conv_1, point_net_conv_2, gat_graph_layers) is loaded
    from T8 checkpoint and frozen:
      - all backbone parameters have requires_grad=False
      - forward() runs backbone under torch.no_grad() for memory efficiency

    UNLIKE PointNetTransfGATFrozenCQR (T11), this class does NOT override
    train() to force backbone into eval mode.  Backbone dropout remains active
    in train mode, which is required for MC Dropout epistemic uncertainty.

    Only gat_heteroscedastic_head (GATConv 64->2) is trainable.
    Outputs: (mu, log_var), each shape [N, 1].
    """

    # Architecture constants (must match T8 exactly)
    _IN_CHANNELS = 5
    _PNC_LOCAL   = [256]
    _PNC_GLOBAL  = [512]
    _GAT_CONV    = [128, 256, 512]
    _DROPOUT     = 0.2

    def __init__(self, t8_model_path: str, device=None):
        """
        Parameters
        ----------
        t8_model_path : str
            Absolute path to T8 model.pth checkpoint.
        device : torch.device, optional
            Device to map checkpoint weights to during loading. Defaults to CPU.
        """
        super().__init__()

        if device is None:
            device = torch.device('cpu')

        # Shared dropout layer (matches T8 structure -- no parameters, not in state_dict)
        self.dropout_layer = nn.Dropout(self._DROPOUT)

        # ------------------------------------------------------------------
        # Backbone (frozen after loading)
        # ------------------------------------------------------------------
        self.point_net_conv_1 = self._create_point_net_layer(
            gat_conv_starts_with_layer=self._GAT_CONV[0],
            is_first_layer=True,
            is_last_layer=False,
        )
        self.point_net_conv_2 = self._create_point_net_layer(
            gat_conv_starts_with_layer=self._GAT_CONV[0],
            is_first_layer=False,
            is_last_layer=True,
        )
        gat_layers = self._define_gat_layers()
        self.gat_graph_layers = GeoSequential('x, edge_index', gat_layers)

        # ------------------------------------------------------------------
        # Trainable heteroscedastic head (newly initialised, NOT loaded from T8)
        # ------------------------------------------------------------------
        self.gat_heteroscedastic_head = GATConv(64, 2)

        # Load T8 backbone weights, validate, then freeze
        self._load_backbone(t8_model_path, device)
        self._freeze_backbone()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    # NOTE: We intentionally do NOT override train().
    # When model.train() is called, backbone dropout layers become active.
    # This is required for MC Dropout (epistemic uncertainty estimation).
    # T11 PointNetTransfGATFrozenCQR overrides train() to force backbone eval
    # -- that pattern is NOT appropriate here.

    def forward(self, data):
        """
        Forward pass.

        Backbone runs under torch.no_grad() (backbone params also have
        requires_grad=False, but no_grad additionally avoids building the
        autograd graph for the intermediate tensors -- saves memory).

        Note: torch.no_grad() does NOT disable dropout.  Dropout is controlled
        by the module training mode (model.train() vs model.eval()).

        Returns
        -------
        mu      : Tensor, shape [N, 1]   predicted mean
        log_var : Tensor, shape [N, 1]   predicted log variance
        """
        edge_index = data.edge_index
        pos1 = data.pos[:, 0, :]   # start position [N, 2]
        pos2 = data.pos[:, 1, :]   # end position   [N, 2]

        with torch.no_grad():
            x = data.x.to(torch.float32)
            x = self.point_net_conv_1(x, pos1, edge_index)
            x = self.point_net_conv_2(x, pos2, edge_index)
            x = self.gat_graph_layers(x, edge_index)
        # x: [N, 64], requires_grad=False

        out = self.gat_heteroscedastic_head(x, edge_index)  # [N, 2]
        mu      = out[:, 0:1]   # [N, 1]
        log_var = out[:, 1:2]   # [N, 1]
        return mu, log_var

    # ----------------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------------

    def _load_backbone(self, model_path: str, device):
        """
        Load T8 checkpoint with GATConv key remapping, then validate.

        Expected mismatches (both intentional):
          - Unexpected keys in checkpoint: 'gat_final.*'  (T8 output head, replaced)
          - Missing keys not in checkpoint: 'gat_heteroscedastic_head.*' (new head)
        Any other unexpected or missing keys raise RuntimeError.
        """
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # Remap old PyG GATConv keys for gat_final only:
        #   gat_final.lin.weight -> gat_final.lin_src.weight + gat_final.lin_dst.weight
        # Scoped to gat_final.* only; gat_graph_layers keys are already in correct format.
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('gat_final.') and '.lin.weight' in k:
                remapped[k.replace('.lin.weight', '.lin_src.weight')] = v
                remapped[k.replace('.lin.weight', '.lin_dst.weight')] = v
            else:
                remapped[k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)

        # Validate unexpected keys: only gat_final.* are allowed
        bad_unexpected = [k for k in unexpected if not k.startswith('gat_final.')]
        if bad_unexpected:
            raise RuntimeError(
                f"Unexpected backbone keys in T8 checkpoint (not gat_final.*): "
                f"{bad_unexpected}"
            )

        # Validate missing keys: only gat_heteroscedastic_head.* are allowed
        bad_missing = [
            k for k in missing if not k.startswith('gat_heteroscedastic_head.')
        ]
        if bad_missing:
            raise RuntimeError(
                f"Backbone keys not loaded from T8 checkpoint: {bad_missing}"
            )

        n_skipped = len([k for k in unexpected if k.startswith('gat_final.')])
        n_head    = len([k for k in missing if k.startswith('gat_heteroscedastic_head.')])
        print(f"  T8 backbone loaded from: {model_path}")
        print(f"  Skipped gat_final keys (T8 output head, replaced by new head): {n_skipped}")
        print(f"  New gat_heteroscedastic_head keys (randomly initialised): {n_head}")

    def _freeze_backbone(self):
        """
        Freeze all backbone parameters. Only gat_heteroscedastic_head remains trainable.
        """
        for name, param in self.named_parameters():
            if not name.startswith('gat_heteroscedastic_head.'):
                param.requires_grad_(False)

        n_frozen = sum(
            p.numel() for n, p in self.named_parameters()
            if not n.startswith('gat_heteroscedastic_head.')
        )
        n_trainable = sum(p.numel() for p in self.gat_heteroscedastic_head.parameters())
        print(f"  Backbone frozen:  {n_frozen:,} parameters (requires_grad=False)")
        print(f"  Head trainable:   {n_trainable:,} parameters (requires_grad=True)")

    def _define_gat_layers(self):
        """
        Define GAT layers -- identical to PointNetTransfGAT.define_gat_layers().
        gat_conv = [128, 256, 512]:
          TransformerConv(128 -> 64, heads=4) -> 256
          TransformerConv(256 -> 128, heads=4) -> 512
          GATConv(512 -> 64)
        """
        layers = []
        for idx in range(len(self._GAT_CONV) - 1):
            layers.append((
                TransformerConv(
                    self._GAT_CONV[idx],
                    int(self._GAT_CONV[idx + 1] / 4),
                    heads=4,
                ),
                'x, edge_index -> x',
            ))
            layers.append(nn.ReLU(inplace=True))
            layers.append(self.dropout_layer)
        layers.append((GATConv(self._GAT_CONV[-1], 64), 'x, edge_index -> x'))
        return layers

    def _create_point_net_layer(
        self,
        gat_conv_starts_with_layer: int,
        is_first_layer: bool = False,
        is_last_layer: bool = False,
    ):
        """
        Create PointNetConv layer -- identical to PointNetTransfGAT.create_point_net_layer().
        """
        offset = 2  # pos contributes 2 extra features to local MLP input

        local_layers = []
        in_dim = (
            self._IN_CHANNELS + offset if is_first_layer
            else self._PNC_GLOBAL[-1] + offset
        )
        local_layers.append(nn.Linear(in_dim, self._PNC_LOCAL[0]))
        local_layers.append(nn.ReLU())
        local_layers.append(self.dropout_layer)
        for i in range(len(self._PNC_LOCAL) - 1):
            local_layers.append(nn.Linear(self._PNC_LOCAL[i], self._PNC_LOCAL[i + 1]))
            local_layers.append(nn.ReLU())
            local_layers.append(self.dropout_layer)
        local_mlp = nn.Sequential(*local_layers)

        global_layers = []
        global_layers.append(nn.Linear(self._PNC_LOCAL[-1], self._PNC_GLOBAL[0]))
        global_layers.append(nn.ReLU())
        global_layers.append(self.dropout_layer)
        for i in range(len(self._PNC_GLOBAL) - 1):
            global_layers.append(nn.Linear(self._PNC_GLOBAL[i], self._PNC_GLOBAL[i + 1]))
            global_layers.append(nn.ReLU())
            global_layers.append(self.dropout_layer)
        if is_last_layer:
            global_layers.append(
                nn.Linear(self._PNC_GLOBAL[-1], gat_conv_starts_with_layer)
            )
        else:
            global_layers.append(
                nn.Linear(self._PNC_GLOBAL[-1], self._PNC_GLOBAL[-1])
            )
        global_layers.append(nn.ReLU())
        global_layers.append(self.dropout_layer)
        global_mlp = nn.Sequential(*global_layers)

        return PointNetConv(local_nn=local_mlp, global_nn=global_mlp)
