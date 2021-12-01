import torch
import torch.nn as nn
import torch.nn.functional as F


def get_y_hat(z):
    b = z.shape[0]
    y_hat = torch.zeros((b, b))
    for i in range(b):
        for j in range(b):
            y_hat[i, j] = F.cosine_similarity(z[i], z[j], dim=0)
    return y_hat


class GMLPModel(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(GMLPModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.PReLU(),
            nn.LayerNorm(n_hidden_1, eps=1e-6),
            nn.Dropout(0.35))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        if self.training:
            return x, get_y_hat(x)
        else:
            return x
