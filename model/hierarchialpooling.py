from torch_geometric.nn import GCNConv
import dgl_patch as dgl
import torch
from torch import nn, Tensor
from dgl.nn.pytorch import AvgPooling
from typing import Optional
import model.alignn as alignn
import model.transformer as transformer
import modules.modules as modules
import torch.nn.functional as F

class DiffPool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, assign_channels, num_clusters,
                adj_pool=False):
        super().__init__()
        self.gnn_embed = GCNConv(in_channels, hidden_channels)
        self.gnn_assign = GCNConv(in_channels, assign_channels)
        self.lin_assign = torch.nn.Linear(assign_channels, num_clusters)
        self.adj_pool = adj_pool  
    def forward(self, x, edge_index, batch=None):
        z = F.relu(self.gnn_embed(x, edge_index)) ## just an embedding        
        s = F.relu(self.gnn_assign(x, edge_index)) ## embeddings to calculate assignment scores
        s = self.lin_assign(s)                     ## assignment scores to each cluster      
        s = F.softmax(s, dim=-1)                   ## probabilities of each node to each cluster
        x_pool = torch.matmul(s.transpose(0,1), z)       ## pooled node features
        if adj_pool:
            adj = torch.zeros(x.size(0), x.size(0), device=x.device) ## adjacency matrix
            adj[edge_index[0], edge_index[1]] = 1 
            adj_pool = s.transpose(0,1) @ adj @ s
            output = (x_pool, adj_pool,s)     
        else:
            output = x_pool
        return output


class DiffPoolLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, assign_channels, num_clusters,
                adj_pool=False):
        super().__init__()
        self.diffpool = DiffPool(in_channels, hidden_channels, assign_channels, num_clusters, adj_pool)
        self.lin = torch.nn.Linear(hidden_channels, in_channels)
        self.norm = torch.nn.LayerNorm(in_channels)
        self.activation = torch.nn.SiLU()
        self.adj_pool = adj_pool
    def forward(self, x, edge_index, batch=None):
        x = self.diffpool(x, edge_index, batch)
        if self.adj_pool:
            x, adj_pool, s = x
            # x = torch.cat([x, adj_pool], dim=1)
        x = self.lin(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

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

        EGGC_Class = getattr(alignn, "OptimizedEdgeGatedGraphConvV1", alignn.EdgeGatedGraphConv)
        eggcs = [EGGC_Class(encoder_dims, rank_factor=rank_factor) for _ in range(n_gnn - 1)]
        eggcs.append(EGGC_Class(encoder_dims, norm=(True, False), rank_factor=rank_factor))

        self.mha_layers = nn.ModuleList(mha_layers)
        self.alignn_layers = nn.ModuleList(alignns)
        self.eggc_layers = nn.ModuleList(eggcs)

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
        for eggc_layer in self.eggc_layers:
            x, y = eggc_layer(g, x, y)
        out = self.lin_block(x)
        if self.norm:
            out = self.normalizer(out)
        if self.residual:
            out = out + x_residual
        return out, y, z, f

class Graphformer(nn.Module):
    def __init__(self, args):
        super(Graphformer, self).__init__()
        self.rank_factor = getattr(args, "rank_factor", 0)

        if(args.graph_coarsening):
            self.diffpool = DiffPoolLayer(args.num_atom_fea, args.hidden_dims, args.hidden_dims, args.num_clusters,
                                          adj_pool=args.adj_pool)
            self.diffpool2 = DiffPoolLayer(args.hidden_dims, args.hidden_dims, args.hidden_dims, args.num_clusters,
                                          adj_pool=args.adj_pool)
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

        graph_attr = self.global_pool(g, atom_attr)
        out = self.final_fc(graph_attr)
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
