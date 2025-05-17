import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Beta
import dgl_patch as dgl
import torch
from torch import nn, Tensor
from dgl.nn.pytorch import AvgPooling
from typing import Optional
import model.alignn as alignn
import model.transformer as transformer
import modules.modules as modules

def _align_dimensions(rec_adj, adj, pos_weight):
    """
    Ensures that rec_adj, adj, and pos_weight have the correct dimensions.
    """
    # Assert that rec_adj has either 3 or 4 dimensions
    assert rec_adj.ndim in [3, 4]
    
    # If rec_adj has 3 dimensions, add an extra dimension at the beginning
    if rec_adj.ndim == 3:
        rec_adj = rec_adj.unsqueeze(0)
    
    # Unpack the shape of rec_adj
    P, B, N, N = rec_adj.shape

    # If pos_weight is not None, align its dimensions
    if pos_weight is not None:
        pos_weight = pos_weight.unsqueeze(0).expand(P, -1, -1, -1)

    # Align the dimensions of adj
    adj = adj.unsqueeze(0).expand(P, -1, -1, -1)

    # Return the aligned tensors
    return rec_adj, adj, pos_weight


class BNPool(nn.Module):
    """
    Bayesian Nonparametric Pooling layer.

    Args:
        emb_size (int): Size of the input embeddings.
        n_clusters (int): Maximum number of clusters.
        n_particles (int): Number of particles for the Stick Breaking Process.
        alpha_DP (float): Concentration parameter of the Dirichlet Process.
        sigma_K (float): Variance of the Gaussian prior for the cluster-cluster prob. matrix.
        mu_K (float): Mean of the Gaussian prior for the cluster-cluster prob. matrix.
        k_init (float): Initial value for the cluster-cluster prob. matrix.
        eta (float): Coefficient for the KL divergence loss.
        rescale_loss (bool): Whether to rescale the loss by the number of nodes.
        balance_links (bool): Whether to balance the links in the adjacency matrix.
        train_K (bool): Whether to train the cluster-cluster prob. matrix.
    """
    def __init__(self,
                 emb_size: int,
                 n_clusters: int = 50,
                 n_particles:int = 1,
                 alpha_DP: float = 10,
                 sigma_K: float = 1.0,
                 mu_K: float = 10.0,
                 k_init: float = 1.0,
                 eta: float = 1.0,
                 rescale_loss:bool = True,
                 balance_links: bool = True,
                 train_K: bool = True):
        super(BNPool, self).__init__()
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.rescale_loss = rescale_loss
        self.balance_links = balance_links
        self.train_K = train_K
        self.eta = eta  # coefficient for the kl_loss

        # Prior for the Stick Breaking Process
        self.register_buffer('ones_C', th.ones(self.n_clusters - 1))
        self.register_buffer('alpha_DP', th.ones(self.n_clusters - 1) * alpha_DP)
        
        # Prior for the cluster-cluster prob. matrix
        self.register_buffer('sigma_K', th.tensor(sigma_K))
        self.register_buffer('mu_K', mu_K * th.eye(self.n_clusters, self.n_clusters) -
                             mu_K * (1 - th.eye(self.n_clusters, self.n_clusters)))
        
        # Posterior distributions for the sticks
        self.W = th.nn.Linear(emb_size, 2*(self.n_clusters-1), bias=False)

        # Posterior cluster-cluster prob matrix
        self.mu_tilde = th.nn.Parameter(k_init * th.eye(self.n_clusters, self.n_clusters) -
                                        k_init * (1-th.eye(self.n_clusters, self.n_clusters)), requires_grad=train_K)


    @staticmethod
    def _compute_pi_given_sticks(sticks):
        device = sticks.device
        log_v = th.concat([th.log(sticks), th.zeros(*sticks.shape[:-1], 1, device=device)], dim=-1)
        log_one_minus_v = th.concat([th.zeros(*sticks.shape[:-1], 1, device=device), th.log(1 - sticks)], dim=-1)
        pi = th.exp(log_v + th.cumsum(log_one_minus_v, dim=-1))  # has shape [n_particles, batch, n_nodes, n_clusters]
        return pi

    def get_S(self, node_embs):
        out = th.clamp(F.softplus(self.W(node_embs)), min=1e-3, max=1e3) 
        alpha_tilde, beta_tilde = th.split(out, self.n_clusters-1, dim=-1)
        self.alpha_tilde = alpha_tilde  # Stored for logging purposes
        self.beta_tilde = beta_tilde    # Stored for logging purposes
        q_pi = Beta(alpha_tilde, beta_tilde)
        stick_fractions = q_pi.rsample([self.n_particles])
        S = self._compute_pi_given_sticks(stick_fractions)
        return S, q_pi

    def _compute_dense_coarsened_graph(self, S, adj, x, mask):
        """
        Compute the coarsened graph by applying the clustering matrix S 
        to the adjacency matrix and the node embeddings.
        """
        x_pool = th.einsum('bnk,bnf->bkf', S, x)

        adj_pool = th.matmul(th.matmul(S.transpose(1, 2), adj), S) # has shape B x K x K
        nonempty_clust = ((S*mask.unsqueeze(-1)) >= 0.2).sum(axis=1) > 0 # Tracks which clusters are non-empty

        # remove element on the diagonal
        ind = th.arange(self.n_clusters, device=adj_pool.device) # [0,1,2,3,...,K-1]
        adj_pool[:, ind, ind] = 0
        deg = th.einsum('ijk->ij', adj_pool)
        deg = th.sqrt(deg+1e-4)[:, None]
        adj_pool = (adj_pool / deg) / deg.transpose(1, 2)

        return adj_pool, x_pool, nonempty_clust
    
    def forward(self, node_embs, adj, node_mask, 
                pos_weight=None,    
                return_coarsened_graph=None):

        # Compute the node assignments and the reconstructed adjacency matrix
        S, q_z = self.get_S(node_embs)  # S has shape P x B x N x K

        # Compute the losses
        rec_loss = self.dense_rec_loss(S, adj, pos_weight)   # has shape P x B x N x N
        kl_loss = self.eta * self.pi_prior_loss(q_z)  # has shape B x N

        K_prior_loss = self.K_prior_loss() if self.train_K else 0  # has shape 1
        
        # Sum losses over nodes by considering the actual number of nodes for each graph
        if not th.all(node_mask):
            edge_mask = th.einsum('bn,bm->bnm', node_mask, node_mask).unsqueeze(0)  # has shape 1 x B x N x N
            rec_loss = rec_loss * edge_mask
            kl_loss = kl_loss * node_mask
        rec_loss = rec_loss.sum((-1, -2))  # has shape P x B
        kl_loss = kl_loss.sum(-1)          # has shape B

        # Normalize the losses
        if self.rescale_loss:
            N = node_mask.sum(-1)
            rec_loss = rec_loss / N.unsqueeze(0)
            kl_loss = kl_loss / N
            K_prior_loss = K_prior_loss / N

        # Build the output dictionary
        loss_d = {'quality': rec_loss.mean(),
                  'kl': self.eta * kl_loss.mean()}
        if self.train_K:
            loss_d['K_prior'] = K_prior_loss.mean()

        S = S[0] # Take only the first particle

        if return_coarsened_graph:
            out_adj, out_x, nonempty_clust = self._compute_dense_coarsened_graph(S, adj, node_embs, node_mask)
            return S, out_x, out_adj, nonempty_clust, loss_d
        else:
            return S, loss_d

    def dense_rec_loss(self, S, adj, pos_weight):
        """
        BCE loss between the reconstructed adjacency matrix and the true one.
        We use with logits because K is no longer the indentiy matrix.
        """
        p_adj = S @ self.mu_tilde @ S.transpose(-1, -2)

        if self.balance_links and pos_weight is None:
            raise ValueError("pos_weight must be provided when balance_links is True")
        p_adj, adj, pos_weight = _align_dimensions(p_adj, adj, pos_weight)

        loss = F.binary_cross_entropy_with_logits(p_adj, adj, weight=pos_weight, reduction='none')

        return loss  # has shape P x B x N x N

    def pi_prior_loss(self, q_pi):
        """
        KL divergence between the posterior and the prior of the Stick Breaking Process.
        """
        p_pi = Beta(self.get_buffer('ones_C'), self.get_buffer('alpha_DP'))
        loss = kl_divergence(q_pi, p_pi).sum(-1)
        return loss  # has shape B x N

    def K_prior_loss(self):
        """
        KL divergence between the posterior and the prior of the cluster-cluster prob. matrix.
        """
        mu_K, sigma_K = self.get_buffer('mu_K'), self.get_buffer('sigma_K')
        K_prior_loss = (0.5 * (self.mu_tilde - mu_K) ** 2 / sigma_K).sum()
        return K_prior_loss  # has shape 1
    

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

class Poolencoder(nn.Module):
    def __init__(self, encoder_dims: int, n_heads: int = 4, n_mha: int = 1, n_alignn: int = 4, n_gnn: int = 4,
                 residual: bool = True, norm: bool = True, last: bool = False, seq_len: int = 256,
                 rank_factor: int = 0):
        super(Poolencoder, self).__init__()
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

class GraphPoolformer(nn.Module):
    def __init__(self, args):
        super(GraphPoolformer, self).__init__()
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

        encoders = [Poolencoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha,
                            n_alignn=args.n_alignn, n_gnn=args.n_gnn, seq_len=512, rank_factor=self.rank_factor)
                    for _ in range(args.num_layers - 1)]
        encoders.append(Poolencoder(args.hidden_dims, n_heads=args.n_heads, n_mha=args.n_mha,
                                n_alignn=args.n_alignn, n_gnn=args.n_gnn, last=True, seq_len=512, rank_factor=self.rank_factor))

        self.encoders = nn.ModuleList(encoders)
        if(args.pooling == 'bnpool'):
            self.global_pool = BNPool(args.hidden_dims, n_clusters=args.n_clusters, n_particles=args.n_particles,
                                      alpha_DP=args.alpha_dp, sigma_K=args.sigma_k, mu_K=args.mu_k,
                                      k_init=args.k_init, eta=args.eta, rescale_loss=args.rescale_loss,
                                      balance_links=args.balance_links, train_K=args.train_k)
            self.bnpool = True
        else:
            self.bnpool = False
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

        if self.bnpool:
            adj_scipy = g.adjacency_matrix()
            print(adj_scipy.shape)
            adj = torch.tensor(adj_scipy.to_dense(), dtype=torch.float32).to(atom_attr.device)
            node_mask = th.zeros((adj.shape[0], 1000), device=atom_attr.device)
            node_mask[:, :adj.shape[0]] = 1
            adj = th.zeros((1000, 1000), device=atom_attr.device)
            adj[:adj.shape[0], :adj.shape[0]] = adj
            S, out_x, out_adj, nonempty_clust, loss_d = self.global_pool(
                node_embs=atom_attr,  # Node embeddings from the encoder
                adj=adj,              # Adjacency matrix
                node_mask=node_mask,  # Node mask
                return_coarsened_graph=True # Request the coarsened graph features
            )
            graph_attr = out_x 
        else:
            graph_attr = self.global_pool(g, atom_attr)
            loss_d = {} # Initialize an empty loss dictionary if not using bnpool

        out = self.final_fc(graph_attr)
        
        # Modified return signature to include loss_d from BNPool
        return out, atom_attr, edge_attr, angle_attr, fc_attr, loss_d

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
