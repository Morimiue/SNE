import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from scipy.stats import spearmanr

from model_mlp import MLPModel as Model
from utils import *

otu_path = './data/otu0.csv'
adj_path = './data/wtd_adj0.csv'
model_path = './models/mlp_model.pth'

batch_size = 64
delta = 0.1

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
spieces_num = otu.shape[0]
sample_num = otu.shape[1]
# print(spieces_num)
# print(sample_num)

test_dataset = get_coe_dateset(otu, adj)

loader = Data.DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)

model = Model(sample_num * 2, 1024, 256, 32, 1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
if is_using_gpu:
    model = model.cuda()

deep_coef = np.zeros((spieces_num, spieces_num))
for i in range(spieces_num):
    for j in range(spieces_num):
        deep_coef[i][j] = model(torch.from_numpy(
            np.append(otu[i], otu[j])))

# get_deep_accuracy(model, loader)

pearson_coef = np.corrcoef(otu)
spearman_coef = spearmanr(otu, axis=1)[0]

get_accuracy(deep_coef)
get_accuracy(pearson_coef)
get_accuracy(spearman_coef)


# Graw graphs
node_size = np.average(otu, axis=1)
node_min, node_max = np.min(node_size), np.max(node_size)
node_size = (node_size - node_min) / (node_max - node_min)
node_color = node_size
node_size *= 800
node_size += 200

G = nx.DiGraph()
G.add_nodes_from(range(spieces_num))
for i in range(spieces_num):
    for j in range(spieces_num):
        if adj[i][j] and i != j:
            G.add_edge(i, j)
print(G.size())
pos = nx.drawing.spring_layout(G)
draw_raw_graph(G, pos, 500, node_color)
draw_graph(G, pos, node_size, node_color)

G = nx.DiGraph()
G.add_nodes_from(range(spieces_num))
for i in range(spieces_num):
    for j in range(spieces_num):
        if abs(deep_coef[i, j]) > 0.1 and i != j:
            G.add_edge(i, j)
print(G.size())
draw_graph(G, pos, node_size, node_color)
