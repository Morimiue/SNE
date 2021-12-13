import torch
import torch.nn as nn
import torch.nn.functional as F


class GMLPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, out_dim):
        super(GMLPModel, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim_1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden_dim_1)
        self.dp = nn.Dropout(0.35)
        self.linear2 = nn.Linear(hidden_dim_1, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.normal_(self.linear2.bias, std=1e-6)

    def _get_z_sim(self, z):
        dot_products = z @ z.T
        magnitudes = torch.sum(z**2, 1).reshape(-1, 1)
        magnitudes = torch.sqrt(magnitudes)
        magnitudes = magnitudes @ magnitudes.T
        z_sim = dot_products * (magnitudes**(-1))
        # z_sim = z_sim.fill_diagonal_(0.)
        return z_sim

    def _get_z_dist(self, z):
        z_dist = torch.norm(z[:, None] - z, dim=2, p=2)
        return z_dist

    def _get_z_dist_triu(self, z):
        z_dist = torch.pdist(z, p=2)
        return z_dist

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = self.dp(x)
        x = self.linear2(x)

        # return x, self._get_z_sim(x)
        if self.training:
            return x, self._get_z_dist_triu(x)
        else:
            return x, self._get_z_dist(x)
