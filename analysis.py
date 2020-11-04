import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from scipy.stats import spearmanr

from mlp_model import MLPModel as Model
from data_processor import *


otu_path = './data/test_otu.csv'
adj_path = './data/test_adj.csv'
model_path = './models/mlp_model.pth'

batch_size = 64
delta = 0.10

is_using_gpu = torch.cuda.is_available()


def get_deep_accuracy(model, data_loader):
    criterion = nn.MSELoss()

    # Initialize loss and accuracy
    eval_loss = 0.
    eval_acc = 0.

    for _, data in enumerate(data_loader):
        x, y = data
        if is_using_gpu:
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            out = model(x)
            loss = criterion(out, y)
        eval_loss += loss.item()
        eval_acc += (abs(y-out) < delta).float().mean()

    # Print loss and accuracy
    loader_step = len(data_loader)
    print(
        f'---Loss: {eval_loss/loader_step:.6f}, Acc: {eval_acc/loader_step:.6f}')


def get_accuracy(coef):
    a = (abs(adj-coef) < delta)
    acc = np.mean(a)
    print(f'{acc:.6f}')


otu, adj = read_raw_data(otu_path, adj_path)
sample_num = otu.shape[1]

test_dataset = get_dateset(otu, adj)

loader = Data.DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)

model = Model(sample_num * 2, 1024, 256, 32, 1)
model.load_state_dict(torch.load(model_path))
model.eval()
if is_using_gpu:
    model = model.cuda()

get_deep_accuracy(model, loader)

pearson_coef = np.corrcoef(otu)
spearman_coef = spearmanr(otu, axis=1)[0]
get_accuracy(pearson_coef)
get_accuracy(spearman_coef)
