import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'

raw_smpl_path = './data/real/raw/samples_raw_diabetes.csv'
raw_intr_path = './data/real/raw/interactions_raw_trrust.csv'
smpl_path = './data/real/processed/samples_trrust_dia.csv'
intr_path = './data/real/processed/interactions_trrust_dia.csv'

feature_size = 16
batch_size = 512
epoch_num = 200
contrasive_loss_m = 700.
potential_loss_l = 2.
potential_loss_k = 3.
learning_rate = 1e-4
weight_decay = 1e-3
delta = 0.10

is_use_gpu = torch.cuda.is_available()
is_save_model = True


class NContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                y_hat: torch.Tensor,
                y: torch.Tensor,
                tau: int = 1) -> torch.Tensor:
        y_hat = torch.exp(y_hat * tau)
        y_match_sum = torch.sum(y_hat * y, 1)
        y_sum = torch.sum(y_hat, 1)
        loss = -torch.log(y_match_sum * (y_sum)**(-1) + 1e-8).mean()
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                z_dist: torch.Tensor,
                y: torch.Tensor,
                m: float = contrasive_loss_m) -> torch.Tensor:
        zeros = torch.zeros(y.shape)
        if is_use_gpu:
            zeros = zeros.cuda()
        ls = z_dist**2
        ld = torch.maximum(zeros, m - z_dist)**2
        loss = y * ls + (1 - y) * ld
        return loss.mean() / 2


class PotentialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                z_dist: torch.Tensor,
                y: torch.Tensor,
                l: float = potential_loss_l,
                k: float = potential_loss_k) -> torch.Tensor:
        ls = (z_dist - l)**2
        ld = k * (z_dist + 1e-8)**(-1)
        loss = y * ls + (1 - y) * ld
        return loss.mean() / 2


# train model
def train_model(model, data_loader):
    # define criterion and optimizer
    # criterion = NContrastLoss()
    # criterion = ContrastiveLoss()
    criterion = PotentialLoss()
    # or use the
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay)

    # train for epochs
    min_loss = float('inf')
    for epoch in range(epoch_num):
        # initialize loss and accuracy
        train_loss = 0.
        train_acc = 0.

        # train on batches
        for _, data in enumerate(data_loader):
            # get batch
            x, y = data
            i, j = torch.triu_indices(y.shape[0], y.shape[1], 1)
            y = y[i, j]
            if is_use_gpu:
                x, y = x.cuda(), y.cuda()
            # forward
            z, y_hat = model(x)
            # backward
            optimizer.zero_grad()
            loss = criterion(y_hat, y) + F.mse_loss(y_hat, y) * 100
            # loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            # accumulate loss and accuracy
            train_loss += loss
            train_acc += (abs(y - y_hat) < delta).float().mean()
            # true_y_hat = F.relu(1 - y_hat / contrasive_loss_m)
            zeros = torch.zeros(y.shape)
            if is_use_gpu:
                zeros = zeros.cuda()
            true_y_hat = torch.maximum(zeros,
                                       -(y_hat / potential_loss_l - 1)**2 + 1)
            train_acc += (abs(y - true_y_hat) < delta).float().mean()

        # get loss and accuracy of this epoch
        loader_step = len(data_loader)
        train_loss = train_loss / loader_step
        train_acc = train_acc / loader_step
        min_loss = min(min_loss, train_loss)
        # print training stats
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f'--- Epoch: {epoch+1}, Loss: {train_loss:.6f}, Acc: {train_acc:.6f}'
            )

        # print some data for debug
        if (epoch + 1) == epoch_num:
            # print('z', z)
            print('y_hat', y_hat)
            print('true_y_hat', true_y_hat)
            print('y', y)

    # save last model
    if is_save_model:
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    train_data = get_dataset(smpl_path, intr_path, feature_size)
    # train_data = get_cora_dataset(train=True)

    data_loader = GMLPDataLoader(data=train_data,
                                 batch_size=batch_size,
                                 shuffle=True)

    model = Model(feature_size, 256, 256)
    if is_use_gpu:
        model = model.cuda()

    # model.load_state_dict(torch.load(model_path))
    train_model(model, data_loader)
