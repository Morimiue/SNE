import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

from mlp_model import MLPModel as Model
from utils import *


otu_path = './data/train_otu.csv'
adj_path = './data/train_adj.csv'
model_path = './models/mlp_model.pth'

batch_size = 64
epoch_num = 10000
learning_rate = 1e-4
delta = 0.10

is_using_gpu = torch.cuda.is_available()
is_saving_model = True


# Train model
def train_model(model, data_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train on every epoch
    for epoch in range(epoch_num):
        # Initialize loss and accuracy
        train_loss = 0.
        train_acc = 0.

        for _, data in enumerate(data_loader):
            x, y = data
            if is_using_gpu:
                x = x.cuda()
                y = y.cuda()
            # Forward
            out = model(x)
            loss = criterion(out, y)
            train_loss += loss.item()
            train_acc += (abs(y-out) < delta).float().mean()
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss and accuracy
        loader_step = len(data_loader)
        if epoch == 0 or (epoch+1) % 100 == 0:
            print(
                f'---Epoch: {epoch+1}, Loss: {train_loss/loader_step:.6f}, Acc: {train_acc/loader_step:.6f}')
            # np.set_printoptions(precision=3, suppress=True)
            # print(y.numpy().T)
            # print(out.detach().numpy().T)

        if (epoch+1) == epoch_num:
            np.set_printoptions(precision=3, suppress=True)
            if is_using_gpu:
                print()
                print(y.cpu().numpy().T)
                print()
                print(out.detach().cpu().numpy().T)
            else:
                print()
                print(y.numpy().T)
                print()
                print(out.detach().numpy().T)

    # Save model
    if is_saving_model:
        torch.save(model.state_dict(), model_path)


otu, adj = read_raw_data(otu_path, adj_path)
sample_num = otu.shape[1]

train_dataset = get_coe_dateset(otu, adj)

loader = Data.DataLoader(dataset=train_dataset,
                         batch_size=batch_size, shuffle=True)

model = Model(sample_num * 2, 1024, 256, 32, 1)

if is_using_gpu:
    model = model.cuda()

train_model(model, loader)
