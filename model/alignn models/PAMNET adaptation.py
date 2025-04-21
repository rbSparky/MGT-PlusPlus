# TRYING PAMNET
# Epoch loss: 3.7040
# Training time: 31.90 s
# /home/rishabh/miniconda3/envs/mgt_py310/lib/python3.10/site-packages/pymatgen/core/structure.py:3090: UserWarning: Issues encountered while parsing CIF: 2 fractional coordinates rounded to ideal values to avoid issues with finite precision.
#   struct = parser.parse_structures(primitive=primitive)[0]
# Validation error: 1.3422
# Epoch 1 validation complete
# New best model saved to /home/rishabh/saved_models/lowest.ckpt
# Completed Epoch 1/100 in 34.35 s
# Epoch loss: 3.8454
# Training time: 26.26 s
# Validation error: 1.3329
# Epoch 2 validation complete
# New best model saved to /home/rishabh/saved_models/lowest.ckpt
# Completed Epoch 2/100 in 28.86 s
# Epoch loss: 10.0560
# Training time: 30.24 s
# Validation error: 2.6527
# Epoch 3 validation complete
# Completed Epoch 3/100 in 32.61 s
# Epoch loss: 11.8653
# Training time: 32.19 s
# Validation error: 2.6400
# Epoch 4 validation complete
# Completed Epoch 4/100 in 35.43 s
# Epoch loss: 1.7769
# Training time: 30.73 s
# Validation error: 0.3165
# Epoch 5 validation complete
# New best model saved to /home/rishabh/saved_models/lowest.ckpt
# Completed Epoch 5/100 in 33.51 s
# Epoch loss: 1.4997
# Training time: 27.48 s
# Validation error: 0.8607
# Epoch 6 validation complete
# Completed Epoch 6/100 in 30.04 s
# Epoch loss: 0.6790
# Training time: 31.69 s
# Validation error: 0.3207
# Epoch 7 validation complete
# Completed Epoch 7/100 in 34.04 s
# Epoch loss: 0.4934
# Training time: 26.32 s
# Validation error: 0.4456
# Epoch 8 validation complete
# Completed Epoch 8/100 in 28.37 s
# Epoch loss: 0.7783
# Training time: 33.30 s
# Validation error: 0.6173
# Epoch 9 validation complete
# Completed Epoch 9/100 in 36.10 s
# Epoch loss: 0.4238
# Training time: 30.13 s
# Validation error: 0.2991
# Epoch 10 validation complete
# New best model saved to /home/rishabh/saved_models/lowest.ckpt
# Completed Epoch 10/100 in 33.30 s
# Epoch loss: 0.2630
# Training time: 29.07 s
# Validation error: 0.2961
# Epoch 11 validation complete
# New best model saved to /home/rishabh/saved_models/lowest.ckpt
# Completed Epoch 11/100 in 31.88 s
# Epoch loss: 0.4541
# Training time: 32.07 s
# Validation error: 0.3156
# Epoch 12 validation complete
# Completed Epoch 12/100 in 36.25 s
# Epoch loss: 0.4419
# Training time: 32.42 s
# Validation error: 0.3331
# Epoch 13 validation complete
# Completed Epoch 13/100 in 34.80 s
# Epoch loss: 0.3222
# Training time: 35.65 s
# Validation error: 0.2798
# Epoch 14 validation complete
# New best model saved to /home/rishabh/saved_models/lowest.ckpt
# Completed Epoch 14/100 in 39.00 s
# Epoch loss: 0.4664
# Training time: 26.23 s
# Validation error: 0.3321
# Epoch 15 validation complete
# Completed Epoch 15/100 in 28.39 s
# Epoch loss: 0.6839
# Training time: 24.93 s
# Validation error: 0.5035
# Epoch 16 validation complete
# Completed Epoch 16/100 in 27.61 s
# Epoch loss: 0.5517
# Training time: 36.34 s
# Validation error: 0.4510
# Epoch 17 validation complete
# Completed Epoch 17/100 in 39.58 s
# Epoch loss: 0.5898
# Training time: 30.11 s
# Validation error: 0.4014
# Epoch 18 validation complete
# Completed Epoch 18/100 in 32.98 s

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Tuple, Union

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

Tensor = torch.Tensor

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_factor: int = 4, bias: bool = True):
        super().__init__()
        if rank_factor <= 0 or in_features % rank_factor != 0 or out_features % rank_factor != 0:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
        else:
            r = in_features // rank_factor
            self.layer = nn.Sequential(
                nn.Linear(in_features, r, bias=False),
                nn.Linear(r, out_features, bias=bias),
            )

    def forward(self, x):
        return self.layer(x)

class _SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def _mlp(channels):
    layers = []
    for i in range(1, len(channels)):
        layers += [LowRankLinear(channels[i-1], channels[i]), _SiLU()]
    return nn.Sequential(*layers)

class _Res(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = _mlp([dim, dim, dim])

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(x)

class EdgeGatedGraphConv(nn.Module):
    def __init__(
        self,
        feature_dims: int,
        norm: Union[bool, Tuple[bool, bool]] = True,
        residual: bool = True,
        rank_factor: int = 4,
    ):
        super().__init__()
        self.residual = residual
        if isinstance(norm, bool):
            norm = (norm, norm)
        self.norm_nodes, self.norm_edges = norm
        D = feature_dims

        self.mlp_msg = _mlp([3*D, D])
        self.W_e     = LowRankLinear(D, D)

        self.mlp_x2  = _mlp([D, D])
        self.res1 = _Res(D)
        self.res2 = _Res(D)
        self.res3 = _Res(D)

        if self.norm_nodes:
            self.ln_node = nn.LayerNorm(D)
        if self.norm_edges:
            self.ln_edge = nn.LayerNorm(D)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: Tensor,
        edge_feats: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x0, e0 = node_feats, edge_feats

        src, dst = g.edges(form="uv")

        h = node_feats
        h_j = h[src]
        h_i = h[dst]
        m = self.mlp_msg(torch.cat((h_i, h_j, edge_feats), dim=1))
        m = m * self.W_e(edge_feats)

        agg = scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")

        x = h + agg
        x = self.mlp_x2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        if self.norm_nodes:
            x = self.ln_node(x)
        x = F.silu(x)
        if self.residual:
            x = x0 + x

        e = e0 + self.W_e(e0)
        if self.norm_edges:
            e = self.ln_edge(e)
        e = F.silu(e)

        return x, e

    def __repr__(self):
        D = self.W_e.layer[-1].out_features if isinstance(self.W_e.layer, nn.Sequential) \
             else self.W_e.layer.out_features
        return (
            f"{self.__class__.__name__}"
            f"(dim={D}, norm={(self.norm_nodes, self.norm_edges)}, residual={self.residual})"
        )

class ALIGNNLayer(nn.Module):
    def __init__(
        self,
        feature_dims: int,
        edge_norm: Union[bool, Tuple[bool, bool]] = True,
        node_norm: Union[bool, Tuple[bool, bool]] = True,
        rank_factor: int = 4,
    ):
        super().__init__()
        if isinstance(edge_norm, bool):
            edge_norm = (edge_norm, edge_norm)
        if isinstance(node_norm, bool):
            node_norm = (node_norm, node_norm)

        self.edge_update = EdgeGatedGraphConv(
            feature_dims, norm=edge_norm, residual=True, rank_factor=rank_factor
        )
        self.atom_update = EdgeGatedGraphConv(
            feature_dims, norm=node_norm, residual=True, rank_factor=rank_factor
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: Tensor,
        y: Tensor,
        z: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        y, z = self.edge_update(lg, y, z)
        x, y = self.atom_update(g, x, y)
        return x, y, z
