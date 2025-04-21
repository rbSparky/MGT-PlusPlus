import torch
import numpy as np
from typing import Optional
from torch import nn, Tensor

class LowRankLinear(nn.Module):
    """A low-rank linear layer using a bottleneck."""
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
                # Optional: Add activation like ReLU or GeLU here if desired within the bottleneck
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


class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_factor=0):
        super(MLPLayer, self).__init__()

        # Use LowRankLinear(in_dim, out_dim, rank_factor) if rank_factor > 0
        # else use nn.Linear(in_dim, out_dim)
        if rank_factor > 0:
            self.linear = LowRankLinear(in_features, out_features, rank_factor)
        else:
            self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.SiLU = nn.SiLU(inplace=True)        

    def forward(self, x: Tensor):
        x = self.linear(x)
        x = self.norm(x)
        return self.SiLU(x)


class RBFExpansion(nn.Module):
    def __init__(self, vmin: float = 0, vmax: float = 8, bins: int = 40, lenghtscale: Optional[float] = None):
        super(RBFExpansion, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer('centers', torch.linspace(vmin, vmax, bins))

        if lenghtscale is None:
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lenghtscale
            self.gamma = 1 / (lenghtscale ** 2)

    def forward(self, x: Tensor):
        return torch.exp(-self.gamma * (x.unsqueeze(1) - self.centers) ** 2)

