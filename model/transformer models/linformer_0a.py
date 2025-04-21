import math
from typing import Union, Tuple, Optional
import dgl
import torch
import dgl.function as fn
import torch.nn.functional as F
from torch import nn
from dgl.nn.functional import edge_softmax

class multiheaded(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 concat: bool = True, dropout: float = 0.2, bias: bool = True, 
                 residual: bool = True, norm: Union[bool, Tuple[bool, bool]] = True,
                 k: int = 128, max_nodes: int = 512):
        super(multiheaded, self).__init__()
        if isinstance(norm, bool):
            norm = (norm, norm)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.residual = residual
        self.norm = norm
        self.k = k
        self.max_nodes = max_nodes

        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_edge = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.E = nn.Parameter(torch.Tensor(heads, k, max_nodes))
        self.F = nn.Parameter(torch.Tensor(heads, k, max_nodes))

        if self.concat:
            if norm[0]:
                self.norm_node = nn.LayerNorm(heads * out_channels)
            if norm[1]:
                self.norm_edge = nn.LayerNorm(heads * out_channels)
        else:
            if norm[0]:
                self.norm_node = nn.LayerNorm(out_channels)
            if norm[1]:
                self.norm_edge = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        Q = self.lin_query(node_feats).view(-1, self.heads, self.out_channels)
        K = self.lin_key(node_feats).view(-1, self.heads, self.out_channels)
        V = self.lin_value(node_feats).view(-1, self.heads, self.out_channels)
        N = Q.shape[0]

        K_trans = K.transpose(0, 1)
        V_trans = V.transpose(0, 1)
        E_proj = self.E[:, :, :N]
        F_proj = self.F[:, :, :N]
        projected_K = torch.bmm(E_proj, K_trans)
        projected_V = torch.bmm(F_proj, V_trans)

        Q_trans = Q.transpose(0, 1)
        scores = torch.bmm(Q_trans, projected_K.transpose(1, 2)) / math.sqrt(self.out_channels)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, projected_V)
        out = out.transpose(0, 1)

        if self.concat:
            x = out.reshape(N, self.heads * self.out_channels)
        else:
            x = out.mean(dim=1)

        m = self.lin_edge(edge_feats).view(-1, self.heads, self.out_channels)
        if self.concat:
            m = m.reshape(-1, self.heads * self.out_channels)
        else:
            m = m.mean(dim=1)

        if self.norm[0]:
            x = self.norm_node(x)
        if self.norm[1]:
            m = self.norm_edge(m)

        if self.residual:
            x = x + node_feats
            y = m + edge_feats
        else:
            y = m

        return x, y
