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

    def _get_y_hat(self, z):
        y_hat = z @ z.T
        z_sum = torch.sum(z**2, 1).reshape(-1, 1)
        z_sum = torch.sqrt(z_sum).reshape(-1, 1)
        z_sum = z_sum @ z_sum.T
        y_hat = y_hat * (z_sum**(-1))
        # y_hat = y_hat.fill_diagonal_(0.)
        return y_hat

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = self.dp(x)
        x = self.linear2(x)

        return x, self._get_y_hat(x)
