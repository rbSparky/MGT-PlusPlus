# OUR SOTA

# QMOF
# Epoch loss: 59.8125
# Training time: 22.283740282058716 seconds
# Validation error: 6.4995
# Validation time: 4.473360776901245 seconds
# Completed Epoch 1 of 100 in 26.76 s
# /home/rishabh/miniconda3/envs/mgt_py310/lib/python3.10/site-packages/torch/distributed/fsdp/_state_dict_utils.py:768: UserWarning: When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict willbe returned.
#   warnings.warn(
# /home/rishabh/miniconda3/envs/mgt_py310/lib/python3.10/site-packages/torch/distributed/fsdp/_state_dict_utils.py:711: UserWarning: When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict willbe returned.
#   warnings.warn(
# Epoch loss: 14.1758
# Training time: 41.63847804069519 seconds
# Validation error: 0.1367
# Validation time: 5.381483793258667 seconds
# Completed Epoch 2 of 100 in 47.02 s
# Epoch loss: 25.1374
# Training time: 46.313060998916626 seconds
# Validation error: 3.8112
# Validation time: 5.110134601593018 seconds
# Completed Epoch 3 of 100 in 51.42 s
# Epoch loss: 20.9168
# Training time: 48.72316813468933 seconds
# Validation error: 2.9795
# Validation time: 5.080740928649902 seconds
# Completed Epoch 4 of 100 in 53.80 s
# Epoch loss: 5.3201
# Training time: 42.15045619010925 seconds
# Validation error: 0.5555
# Validation time: 5.55467414855957 seconds
# Completed Epoch 5 of 100 in 47.71 s
# Epoch loss: 2.3637
# Training time: 44.84124183654785 seconds
# Validation error: 1.5290
# Validation time: 5.015815496444702 seconds
# Completed Epoch 6 of 100 in 49.86 s
# Epoch loss: 7.5707
# Training time: 42.69744682312012 seconds
# Validation error: 2.3053
# Validation time: 5.094756603240967 seconds
# Completed Epoch 7 of 100 in 47.79 s
# Epoch loss: 7.7421
# Training time: 47.859519481658936 seconds
# Validation error: 2.1640
# Validation time: 6.20598578453064 seconds
# Completed Epoch 8 of 100 in 54.07 s
# Epoch loss: 3.7778
# Training time: 51.24882626533508 seconds
# Validation error: 1.4691
# Validation time: 5.2103962898254395 seconds
# Completed Epoch 9 of 100 in 56.46 s
# Epoch loss: 0.9143
# Training time: 48.923983573913574 seconds
# Validation error: 0.5196
# Validation time: 4.9715330600738525 seconds
# Completed Epoch 10 of 100 in 53.90 s
# Epoch loss: 0.7440
# Training time: 49.69083642959595 seconds
# Validation error: 0.3798
# Validation time: 5.017044544219971 seconds
# Completed Epoch 11 of 100 in 54.71 s
# Epoch loss: 3.3074
# Training time: 51.38764977455139 seconds
# Validation error: 1.0613
# Validation time: 5.470114469528198 seconds
# Completed Epoch 12 of 100 in 56.86 s
# Epoch loss: 3.9594
# Training time: 52.56816792488098 seconds
# Validation error: 1.2391
# Validation time: 5.324229955673218 seconds
# Completed Epoch 13 of 100 in 57.89 s

import math
from typing import Union, Tuple
import dgl
import torch
import dgl.function as fn
import torch.nn.functional as F
from torch import nn
from dgl.nn.functional import edge_softmax

from linformer import LinformerSelfAttention

class multiheaded(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 heads: int = 1, 
                 concat: bool = True, 
                 dropout: float = 0.2, 
                 bias: bool = True, 
                 residual: bool = True, 
                 norm: Union[bool, Tuple[bool, bool]] = True,
                 seq_len: int = 1024, 
                 k: int = 256,
                 rank_factor: int = 0         
                ):
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

        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_edge = nn.Linear(in_channels, heads * out_channels, bias=bias)

        if self.concat:
            if self.norm[0]:
                self.norm_node = nn.LayerNorm(heads * out_channels)
            if self.norm[1]:
                self.norm_edge = nn.LayerNorm(heads * out_channels)
        else:
            if self.norm[0]:
                self.norm_node = nn.LayerNorm(out_channels)
            if self.norm[1]:
                self.norm_edge = nn.LayerNorm(out_channels)

        self.linformer_attn = LinformerSelfAttention(
            dim = heads * out_channels,
            seq_len = 1024,
            heads = heads,
            k = k,
            one_kv_head = True,
            share_kv = True
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        r"""
        Args:
            g: dgl.DGLGraph
                (Retained for API compatibility; note that in this linformer version the graph structure is not used.)
            node_feats: torch.Tensor
                Node features (shape: [num_nodes, in_channels])
            edge_feats: torch.Tensor
                Edge features (shape: [num_edges, in_channels])
        Returns:
            x: torch.Tensor
                Updated node features.
            y: torch.Tensor
                Updated edge features.
        """
        x_proj = self.lin_query(node_feats)  # [num_nodes, heads*out_channels]
        x_seq = x_proj.unsqueeze(0)  # [1, num_nodes, heads*out_channels]
        x_attn = self.linformer_attn(x_seq)  # [1, num_nodes, heads*out_channels]
        x = x_attn.squeeze(0)  # [num_nodes, heads*out_channels]

        m = self.lin_edge(edge_feats)  # [num_edges, heads*out_channels]

        x = F.dropout(x, p=self.dropout, training=self.training)
        m = F.dropout(m, p=self.dropout, training=self.training)

        if self.concat:
            # e hhh?? [*, heads*out_channels].
            x = x.view(-1, self.heads * self.out_channels)
            m = m.view(-1, self.heads * self.out_channels)
        else:
            x = x.view(-1, self.heads, self.out_channels).mean(dim=1)
            m = m.view(-1, self.heads, self.out_channels).mean(dim=1)

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