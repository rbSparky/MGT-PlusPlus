import torch
import torch.nn as nn
from torch.nn import Sequential, Linear
import sympy as sym
from math import pi as PI
from model.pamnet_adaptation.utils import bessel_basis, real_sph_harm
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def MLP(channels):
    return Sequential(*[
        Sequential(Linear(channels[i - 1], channels[i]), SiLU())
        for i in range(1, len(channels))])


class Res(nn.Module):
    def __init__(self, dim):
        super(Res, self).__init__()
        self.mlp = MLP([dim, dim, dim])

    def forward(self, x):
        x_out = self.mlp(x)
        x_out = x_out + x
        return x_out


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        env_val = 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x

        zero = torch.zeros_like(x)
        return torch.where(x < 1, env_val, zero)

class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=6):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        ##
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out
    

class Global_MessagePassing(MessagePassing):
    def __init__(self, config):
        super(Global_MessagePassing, self).__init__(flow=config.flow)
        self.dim = config.dim

        self.mlp_x1 = MLP([self.dim, self.dim])
        self.mlp_x2 = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.mlp_m = MLP([self.dim * 3, self.dim])
        self.W_edge_attr = nn.Linear(self.dim, self.dim, bias=False)

        self.mlp_out = MLP([self.dim, self.dim, self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, self.dim)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x, edge_attr, edge_index):
        res_x = x
        x = self.mlp_x1(x)

       
        edge_index = torch.stack(edge_index, dim=0)
        x = x + self.propagate(edge_index, x=x, num_nodes=x.size(0), edge_attr=edge_attr)
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)

        return x, out, att_score
    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        m = torch.cat((x_i, x_j, edge_attr), -1)
        m = self.mlp_m(m)

        ##the below self.W_edge_attr@edge_attr,is supposed to represent an attention computation,
        ## based on edge feautures, i'm gonna try a direct attention_score
        return m * self.W_edge_attr(edge_attr)
    def update(self, aggr_out):
        return aggr_out


class Local_MessagePassing(torch.nn.Module):
    def __init__(self, config):
        super(Local_MessagePassing, self).__init__()
        self.dim = config.dim

        self.mlp_x1 = MLP([self.dim, self.dim])
        self.mlp_m_ji = MLP([3 * self.dim, self.dim])
        self.mlp_m_kj = MLP([3 * self.dim, self.dim])
        self.mlp_sbf = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)
        self.mlp_x2 = MLP([self.dim, self.dim])
        
        self.mlp_out = MLP([self.dim, self.dim, self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x, rbf, sbf2, sbf1, idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index):
        j, i = edge_index
        idx = torch.cat((idx_kj, idx_jj_pair), 0)
        idx_scatter = torch.cat((idx_ji, idx_ji_pair), 0)
        sbf = torch.cat((sbf2, sbf1), 0)

        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        m = torch.cat([x[i], x[j], rbf], dim=-1)
        m_ji = self.mlp_m_ji(m)
        m_neighbor = self.mlp_m_kj(m) * self.lin_rbf(rbf)
        m_other = m_neighbor[idx] * self.mlp_sbf(sbf)
        m_other = scatter(m_other, idx_scatter, dim=0, dim_size=m.size(0), reduce='add')
        m = m_ji + m_other

        m = self.lin_rbf_out(rbf) * m
        x = x + scatter(m, i, dim=0, dim_size=x.size(0), reduce='add')
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)

        return x, out, att_score

