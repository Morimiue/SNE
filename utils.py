import csv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data


class GMLPDataLoader():
    def __init__(self, data: Data, batch_size: int, shuffle: bool = False, drop_last: bool = False):
        self.x = torch.as_tensor(data.x)
        self.y = torch.as_tensor(coo_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(
            data.x.shape[0], data.x.shape[0])).toarray())
        self.batch_size = batch_size
        self.shuffle = shuffle
        if drop_last:
            self.sample_num = len(self.x) // self.batch_size
        else:
            self.sample_num = (
                len(self.x) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for _ in range(2):
            random_index = torch.as_tensor(np.random.choice(
                np.arange(len(self.x)), self.batch_size))
            x_batch = self.x[random_index]
            y_batch = self.y[random_index, :][:, random_index]
            yield x_batch, y_batch

    def __len__(self):
        return self.sample_num


# Read data from file
def read_raw_data(otu_path, adj_path):
    with open(otu_path) as f:
        reader = csv.reader(f)
        otu = np.array(list(reader), dtype=np.float32)

    with open(adj_path) as f:
        reader = csv.reader(f)
        adj = np.array(list(reader), dtype=np.float32)

    return otu, adj


# Get dataset
def get_real_dataset():
    samples_df = pd.read_csv('./data/real/samples.csv')

    x = samples_df.values[:, 1:].astype(np.float32)
    x = torch.as_tensor(x)

    interactions_df = pd.read_csv('./data/real/interactions.csv')

    gene1 = interactions_df.gene1.to_list()
    gene2 = interactions_df.gene2.to_list()

    gene_names = samples_df.name.to_list()

    gene1_index = np.array([gene_names.index(i)
                           for i in gene1], dtype=np.int32)
    gene2_index = np.array([gene_names.index(i)
                           for i in gene2], dtype=np.int32)

    # edge_num = len(gene1)
    # coo = coo_matrix((np.ones(edge_num), (gene1_index, gene2_index)))
    edge_index = np.vstack((gene1_index, gene2_index))
    edge_index = torch.as_tensor(edge_index)

    return Data(x=x, edge_index=edge_index)


def get_coe_dateset(otu, adj):
    spieces_num = otu.shape[0]
    sample_num = otu.shape[1]

    x_train = np.zeros((spieces_num * (spieces_num - 1), sample_num * 2))
    y_train = np.zeros((spieces_num * (spieces_num - 1), 1))

    cnt = 0
    for i in range(spieces_num):
        for j in range(spieces_num):
            if i == j:
                continue
            x_train[cnt] = np.append(otu[i], otu[j])
            cnt += 1

    cnt = 0
    for i in range(spieces_num):
        for j in range(spieces_num):
            if i == j:
                continue
            y_train[cnt] = adj[i][j]
            cnt += 1

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    return TensorDataset(x_train, y_train)


def get_emb_dateset():
    for i in range(100):
        with open(f'./data/synthetic/otu{i}.csv') as f:
            reader = csv.reader(f)
            otu = np.array(list(reader), dtype=np.float32)
            if i == 0:
                x_train = otu
            else:
                x_train = np.vstack((x_train, otu))

    for i in range(100):
        with open(f'./data/synthetic/adj{i}.csv') as f:
            reader = csv.reader(f)
            adj = np.array(list(reader), dtype=np.float32)
            if i == 0:
                y_train = adj
            else:
                y_train = np.vstack((y_train, adj))

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    return TensorDataset(x_train, y_train)


# Draw graph
def draw_graph(G, pos, node_size, node_color):
    nx.draw_networkx_nodes(
        G,
        pos,
        alpha=0.9,
        node_size=node_size,
        node_color=node_color,
        cmap=plt.cm.Wistia,
        edgecolors='tab:gray',
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle='-',
        alpha=0.5,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle='-',
        alpha=0.5,
        width=7,
        edge_color='tab:blue',
    )
    nx.draw_networkx_labels(
        G,
        pos,
        alpha=0.8,
        # font_color='whitesmoke'
    )

    plt.axis('off')
    plt.show()


# Draw raw graph
def draw_raw_graph(G, pos, node_size, node_color):
    nx.draw_networkx_nodes(
        G,
        pos,
        alpha=0.9,
        node_size=node_size,
        cmap=plt.cm.Wistia,
        edgecolors='tab:gray',
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle='-',
        alpha=0.5,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle='-',
        alpha=0.5,
        width=7,
        edge_color='tab:blue',
    )
    nx.draw_networkx_labels(
        G,
        pos,
        alpha=0.8,
        # font_color='whitesmoke'
    )

    plt.axis('off')
    plt.show()
