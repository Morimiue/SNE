from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMLPModel(nn.Module):
    def __init__(self, in_dim: int, hid_dim1: int, out_dim: int) -> None:
        super(GMLPModel, self).__init__()

        self.linear1 = nn.Linear(in_dim, hid_dim1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hid_dim1)
        self.dp = nn.Dropout(0.35)
        self.linear2 = nn.Linear(hid_dim1, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.normal_(self.linear2.bias, std=1e-6)

    def _get_z_sim(self, z: torch.Tensor) -> torch.Tensor:
        dot_products = z @ z.T
        magnitudes = torch.sqrt(torch.sum(z**2, 1))[:, None]
        magnitude_products = magnitudes @ magnitudes.T
        z_sim = dot_products * magnitude_products**(-1)
        return z_sim

    def _get_z_dist(self, z: torch.Tensor) -> torch.Tensor:
        z_dist = torch.norm(z[:, None] - z, dim=2, p=2)
        return z_dist

    def _get_z_dist_triu(self, z: torch.Tensor) -> torch.Tensor:
        z_dist = torch.pdist(z, p=2)
        return z_dist

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = self.dp(x)
        x = self.linear2(x)

        if self.training:
            return x, self._get_z_dist_triu(x)
        else:
            return x
            # return x, self._get_z_dist(x)
