from typing import Tuple, Union
import dgl_patch as dgl
import numpy
import torch
import dgl.function as fn
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from torch import Tensor
from typing import Union, Tuple

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_factor: int = 4, bias: bool = True):
        super().__init__()
        if rank_factor <= 0:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
        elif in_features % rank_factor != 0 or out_features % rank_factor != 0:
            print(f"Warning: feature_dims ({in_features}, {out_features}) not divisible by rank_factor ({rank_factor}). Using standard Linear layer.")
            self.layer = nn.Linear(in_features, out_features, bias=bias)
        else:
            rank = in_features // rank_factor
            self.layer = nn.Sequential(
                nn.Linear(in_features, rank, bias=False),
                nn.Linear(rank, out_features, bias=bias)
            )

    def forward(self, x):
        return self.layer(x)

class EdgeGatedGraphConv(nn.Module):
    def __init__(self, feature_dims: int, norm: Union[bool, Tuple[bool, bool]] = True, residual: bool = True, rank_factor: int = 4):
        super().__init__()
        self.residual = residual
        self.rank_factor = rank_factor
        if isinstance(norm, bool):
            norm = (norm, norm)
        self.norm = norm
        self.src_gate = LowRankLinear(feature_dims, feature_dims, rank_factor=rank_factor)
        self.dst_gate = LowRankLinear(feature_dims, feature_dims, rank_factor=rank_factor)
        self.edge_gate = LowRankLinear(feature_dims, feature_dims, rank_factor=rank_factor)
        if norm[0]:
            self.norm_nodes = nn.LayerNorm(feature_dims)
        self.src_update = LowRankLinear(feature_dims, feature_dims, rank_factor=rank_factor)
        self.dst_update = LowRankLinear(feature_dims, feature_dims, rank_factor=rank_factor)
        if norm[1]:
            self.norm_edges = nn.LayerNorm(feature_dims)

    def forward(self, g: dgl.DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> Tuple[Tensor, Tensor]:
        g = g.local_var()
        if self.residual:
            node_feats_residual = node_feats
            edge_feats_residual = edge_feats
        e_src = self.src_gate(node_feats)
        e_dst = self.dst_gate(node_feats)
        e_edge = self.edge_gate(edge_feats)
        g.ndata["e_src"] = e_src
        g.ndata["e_dst"] = e_dst
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + e_edge
        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        sum_sigma_h = g.ndata.pop("sum_sigma_h")
        sum_sigma = g.ndata.pop("sum_sigma")
        h = sum_sigma_h / (sum_sigma + 1e-6)
        x = self.src_update(node_feats) + h
        if self.norm[0]:
            x = self.norm_nodes(x)
        x = F.silu(x)
        if self.residual:
            x = node_feats_residual + x
        y = m
        if self.norm[1]:
            y = self.norm_edges(y)
        y = F.silu(y)
        if self.residual:
            y = edge_feats_residual + y
        return x, y

    def __repr__(self):
        return f"{self.__class__.__name__}(feature_dims={self.src_gate.layer[0].in_features}, rank_factor={self.rank_factor}, norm={self.norm}, residual={self.residual})"

class ALIGNNLayer(nn.Module):
    def __init__(self, feature_dims: int, edge_norm: Union[bool, Tuple[bool, bool]] = True, node_norm: Union[bool, Tuple[bool, bool]] = True, rank_factor: int = 0):
        super(ALIGNNLayer, self).__init__()
        if isinstance(edge_norm, bool):
            edge_norm = (edge_norm, edge_norm)
        if isinstance(node_norm, bool):
            node_norm = (node_norm, node_norm)
        self.edge_update = EdgeGatedGraphConv(feature_dims=feature_dims, norm=edge_norm)
        self.atom_update = EdgeGatedGraphConv(feature_dims=feature_dims, norm=node_norm)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, x: Tensor, y: Tensor, z: Tensor):
        y, z = self.edge_update(g=lg, node_feats=y, edge_feats=z)
        x, y = self.atom_update(g=g, node_feats=x, edge_feats=y)
        return x, y, z