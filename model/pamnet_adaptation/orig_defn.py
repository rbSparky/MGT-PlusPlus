import dgl_patch as dgl
import torch
from torch import nn, Tensor
from dgl.nn.pytorch import AvgPooling
from typing import Optional
import model.alignn as alignn
import model.transformer as transformer
import modules.modules as modules
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool, global_add_pool, radius, knn
from torch_geometric.utils import remove_self_loops

from model.pamnet_adaptation.megik import Global_MessagePassing, Local_MessagePassing, BesselBasisLayer, SphericalBasisLayer, MLP,Res

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_factor: int = 4, bias: bool = True):
        super().__init__()
        self.rank_factor = rank_factor
        if rank_factor <= 0 or in_features % rank_factor != 0 or out_features % rank_factor != 0:
             if rank_factor > 0:
                 print(f"Warning: feature_dims ({in_features}, {out_features}) not divisible by rank_factor ({rank_factor}). Using standard Linear layer.")
             self.layer = nn.Linear(in_features, out_features, bias=bias)
             self.is_low_rank = False
        else:
            rank = in_features // rank_factor
            self.layer = nn.Sequential(
                nn.Linear(in_features, rank, bias=False),
                nn.Linear(rank, out_features, bias=bias)
            )
            self.is_low_rank = True

    def forward(self, x):
        return self.layer(x)

    def __repr__(self):
         if self.is_low_rank:
              in_f = self.layer[0].in_features
              out_f = self.layer[1].out_features
              rank = self.layer[0].out_features
              return f"LowRankLinear(in={in_f}, out={out_f}, rank={rank})"
         else:
              return f"{self.layer}"


class Config(object):
    def __init__(self, dataset, dim, n_layer, cutoff_l, cutoff_g, flow='source_to_target'):
        self.dataset = dataset
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff_l = cutoff_l
        self.cutoff_g = cutoff_g
        self.flow = flow

class PAMNet(nn.Module):
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5):
        super(PAMNet, self).__init__()

        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g
        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        # self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        # self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)
        self.mlp_rbf_g = MLP([16, self.dim])
        # self.mlp_rbf_l = MLP([16, self.dim])    
        # self.mlp_sbf1 = MLP([num_spherical * num_radial, self.dim])
        # self.mlp_sbf2 = MLP([num_spherical * num_radial, self.dim])

        self.mlp_edge_res = Res(self.dim)
        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))
        self.local_layer = torch.nn.ModuleList()
        # for _ in range(config.n_layer):
        #     self.local_layer.append(Local_MessagePassing(config))
        self.softmax = nn.Softmax(dim=-1)

    def indices(self, g:dgl.DGLGraph):
        row, col = g.edges(form="uv")
        # print(row, col)
        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,sparse_sizes=(g.num_nodes(), g.num_nodes()))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    # def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph,
    #             x: Tensor, y: Tensor, z: Tensor, f: Tensor):
    #     ###
    def forward(self, g:dgl.DGLGraph,
                 x, y):
        g_edge_index = g.edges(form="uv")
        # print(g_edge_index)
        # lg_edge_index = lg.edge_index
        # fg_edge_index = fg.edge_index
        
        ## one-hop and two-hop neighbors
        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(g)
        # idx_i_l, idx_j_l, idx_k_l, idx_kj_l, idx_ji_l, idx_i_pair_l, idx_j1_pair_l, idx_j2_pair_l, idx_jj_pair_l, idx_ji_pair_l = self.indices(lg)
        # idx_i_f, idx_j_f, idx_k_f, idx_kj_f, idx_ji_f, idx_i_pair_f, idx_j1_pair_f, idx_j2_pair_f, idx_jj_pair_f, idx_ji_pair_f = self.indices(fg)

        ## obtaining the distance matrix
        row, col = g_edge_index
        g_dist = torch.norm(x[row] - x[col], dim=-1)
        # row, col = lg_edge_index
        # lg_dist = torch.norm(x[row] - x[col], dim=-1)
        # row, col = fg_edge_index
        # fg_dist = torch.norm(x[row] - x[col], dim=-1)

        ## obtaining bessel basis
        g_rbf = self.rbf_g(g_dist)
        # lg_rbf = self.rbf_l(lg_dist)
        # fg_rbf = self.rbf_l(fg_dist)

        ### edge feauture combination idea. ### 
        # print(f"g_rbf shape: {g_rbf.shape}")
        # print(f" y shape: {y.shape}")
        # g_rbf = torch.cat((g_rbf, y), dim=-1)

        g_rbf = self.mlp_rbf_g(g_rbf)
        # lg_rbf = self.mlp_rbf_l(lg_rbf)
        # fg_rbf = self.mlp_rbf_l(fg_rbf)

        ## TODO: obtaining spherical basis

        ## passing through the global message passing
        ## creating a copy for line graph.
        # lx = x
        global_attn_score =[]
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")
        for i in range(self.n_layer):
            x, out, att_score_g = self.global_layer[i](x,g_rbf,g_edge_index)
            # print(f"out shape: {out.shape}")
            # print(f"att_score_g shape: {att_score_g.shape}")
            # print(f" x shape: {x.shape}")
            global_attn_score.append(att_score_g)
            # f = self.global_layer[i](f, fg_edge_index, fg_rbf)
            # lx = self.global_layer[i](lx, lg_edge_index, lg_rbf)
        ## now returning the updated x, f, and lx
        out = out.squeeze(0)
        # global_attn_score = torch.stack(global_attn_score, dim=0)
        # print(f"out shape: {out.shape} , x shape: {x.shape}")   
        ## pooling the graph
        out = global_mean_pool(out,torch.zeros(out.shape[0], device=out.device,dtype=torch.long))
        # global_attn_score = F.leaky_relu(global_attn_score, negative_slope=0.2)
        # global_attn_score = self.softmax(global_attn_score)
        # out = out * global_attn_score
        y = self.mlp_edge_res(y)
        return out ,y



        
## changing up the encoder block to include a dual-graph split
## also incorporating angle blocks
class encoder(nn.Module):
    def __init__(self, encoder_dims: int, n_heads: int = 4, n_mha: int = 1, n_alignn: int = 4, n_gnn: int = 4,
                 residual: bool = True, norm: bool = True, last: bool = False, seq_len: int = 256,
                 rank_factor: int = 0):
        super(encoder, self).__init__()
        assert encoder_dims % n_heads == 0
        assert n_heads >= 1
        assert n_gnn >= 1
        self.head_dim = int(encoder_dims / n_heads)
        self.residual = residual
        self.norm = norm
        self.rank_factor = rank_factor

        mha_layers = [transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads, seq_len=seq_len, rank_factor=rank_factor)
                      for _ in range(n_mha - 1)]
        mha_layers.append(transformer.multiheaded(encoder_dims, self.head_dim, heads=n_heads,
                                                    norm=(True, False), seq_len=seq_len, rank_factor=rank_factor))

        alignns = [alignn.ALIGNNLayer(encoder_dims, rank_factor=rank_factor) for _ in range(n_alignn - 1)]
        alignns.append(alignn.ALIGNNLayer(encoder_dims, edge_norm=(True, False), rank_factor=rank_factor))

        pamnets = [PAMNet(Config(encoder_dims, encoder_dims, n_gnn, 0.5, 3)) for _ in range(n_gnn)]

        self.mha_layers = nn.ModuleList(mha_layers)
        self.alignn_layers = nn.ModuleList(alignns)
        self.pamnet_layers  = nn.ModuleList(pamnets)

        self.lin_block = nn.Sequential(
            LowRankLinear(encoder_dims, encoder_dims, rank_factor=rank_factor),
            nn.ReLU(inplace=True),
            LowRankLinear(encoder_dims, encoder_dims, rank_factor=rank_factor)
        )

        if self.norm:
            self.normalizer = nn.LayerNorm(encoder_dims)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph,
                x: Tensor, y: Tensor, z: Tensor, f: Tensor):
        x_residual = x
        for mha in self.mha_layers:
            x, f = mha(fg, x, f)
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)
        for pamnet_layer in self.pamnet_layers:
            x, y = pamnet_layer(g ,x ,y)

        out = self.lin_block(x)
        if self.norm:
            out = self.normalizer(out)
        if self.residual:
            out = out + x_residual
        return out, y, z, f

class FusedGraphformer(nn.Module):
    def __init__(self, args):
        super(FusedGraphformer, self).__init__()
        self.rank_factor = getattr(args, "rank_factor", 0)
        self.atom_embedding = modules.MLPLayer(args.num_atom_fea, args.hidden_dims, rank_factor=self.rank_factor)
        self.positional_embedding = modules.MLPLayer(args.num_pe_fea, args.hidden_dims, rank_factor=self.rank_factor)

        self.edge_expansion = modules.RBFExpansion(vmin=0, vmax=args.local_radius, bins=args.num_edge_bins)
        self.edge_embedding = nn.Sequential(
            modules.MLPLayer(args.num_edge_bins, args.embedding_dims, rank_factor=self.rank_factor),
            modules.MLPLayer(args.embedding_dims, args.hidden_dims, rank_factor=self.rank_factor)
        )

        self.angle_expansion = modules.RBFExpansion(vmin=-1, vmax=1, bins=args.num_angle_bins)
        self.angle_embedding = nn.Sequential(
            modules.MLPLayer(args.num_angle_bins, args.embedding_dims, rank_factor=self.rank_factor),
            modules.MLPLayer(args.embedding_dims, args.hidden_dims, rank_factor=self.rank_factor)
        )

        self.fc_embedding = nn.Sequential(
            modules.MLPLayer(1, args.num_clmb_bins, rank_factor=self.rank_factor),
            modules.MLPLayer(args.num_clmb_bins, args.embedding_dims, rank_factor=self.rank_factor),
            modules.MLPLayer(args.embedding_dims, args.hidden_dims, rank_factor=self.rank_factor)
        )

        encoders = [encoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha,
                            n_alignn=args.n_alignn, n_gnn=args.n_gnn, seq_len=512, rank_factor=self.rank_factor)
                    for _ in range(args.num_layers - 1)]
        encoders.append(encoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha,
                                n_alignn=args.n_alignn, n_gnn=args.n_gnn, last=True, seq_len=512, rank_factor=self.rank_factor))

        self.encoders = nn.ModuleList(encoders)
        self.global_pool = AvgPooling()
        self.final_fc = nn.Linear(args.hidden_dims, args.out_dims)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, fg: dgl.DGLGraph):
        atom_attr = g.ndata.pop('node_feats') if 'node_feats' in g.ndata else None
        edge_attr = g.edata.pop('edge_feats') if 'edge_feats' in g.edata else None
        angle_attr = lg.edata.pop('angle_feats') if 'angle_feats' in lg.edata else None
        fc_attr = fg.edata.pop('fc_feats') if 'fc_feats' in fg.edata else None
        pe_attr = g.ndata.pop('pes') if 'pes' in g.ndata else None

        if atom_attr is None or edge_attr is None or angle_attr is None or fc_attr is None or pe_attr is None:
             raise ValueError("One or more required features (node_feats, edge_feats, angle_feats, fc_feats, pes) not found in input graphs.")

        atom_attr = self.atom_embedding(atom_attr)
        pe_attr = self.positional_embedding(pe_attr)
        atom_attr = atom_attr + pe_attr

        edge_attr = self.edge_expansion(edge_attr)
        if edge_attr.dim() == 3 and edge_attr.shape[1] == 1:
             edge_attr = edge_attr.squeeze(1)
        edge_attr = self.edge_embedding(edge_attr)

        angle_attr = self.angle_expansion(angle_attr)
        if angle_attr.dim() == 3 and angle_attr.shape[1] == 1:
             angle_attr = angle_attr.squeeze(1)
        angle_attr = self.angle_embedding(angle_attr)

        fc_attr = self.fc_embedding(fc_attr)

        for encdr in self.encoders:
            atom_attr, edge_attr, angle_attr, fc_attr = encdr(g, lg, fg, atom_attr, edge_attr, angle_attr, fc_attr)
            # print(f"atom_attr shape: {atom_attr.shape}")
            # print(f"edge_attr shape: {edge_attr.shape}")
            # print(f"angle_attr shape: {angle_attr.shape}")
            # print(f"fc_attr shape: {fc_attr.shape}")
        graph_attr = self.global_pool(g, atom_attr)
        out = self.final_fc(graph_attr)
        # print("Final output shape: ", out.shape)
        # print("________-------------------------------------------")
        return out, atom_attr, edge_attr, angle_attr, fc_attr

    def freeze_pretrain(self):
        for name, param in self.named_parameters():
            if 'final_fc' not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_train(self):
        for name, param in self.named_parameters():
            if 'final_fc' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
