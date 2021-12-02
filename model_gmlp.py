import torch
import torch.nn as nn
import torch.nn.functional as F


def get_y_hat(z):
    y_hat = z@z.T
    mask = torch.eye(y_hat.shape[0])
    z_sum = torch.sum(z**2, 1).reshape(-1, 1)
    z_sum = torch.sqrt(z_sum).reshape(-1, 1)
    z_sum = z_sum @ z_sum.T
    y_hat = y_hat * (z_sum**(-1))
    y_hat = (1-mask) * y_hat
    return y_hat


class GMLPModel(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(GMLPModel, self).__init__()
        self.linear1 = nn.Linear(in_dim, n_hidden_1)
        self.act = nn.PReLU()
        self.ln = nn.LayerNorm(n_hidden_1, eps=1e-6)
        self.dp = nn.Dropout(0.35)
        self.linear2 = nn.Linear(n_hidden_1, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.normal_(self.linear2.bias, std=1e-6)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = self.dp(x)
        x = self.linear2(x)

        if self.training:
            return x, get_y_hat(x)
        else:
            return x
