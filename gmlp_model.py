import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data


def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0])
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis


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
            return x, get_feature_dis(x)
        else:
            return x
