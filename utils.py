import csv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


class GMLPDataLoader():
    def __init__(self,
                 data: Data,
                 batch_size: int,
                 shuffle: bool = False,
                 drop_last: bool = False):
        self.x = data.x.type(torch.float32)
        self.y = torch.sparse_coo_tensor(data.edge_index,
                                         torch.ones(data.edge_index.shape[1]),
                                         (data.x.shape[0], data.x.shape[0]))
        self.y = self.y.to_dense().type(torch.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        if drop_last:
            self.sample_num = len(self.x) // self.batch_size
        else:
            self.sample_num = (len(self.x) + self.batch_size -
                               1) // self.batch_size

    def __iter__(self):
        for _ in range(self.sample_num):
            random_index = torch.randperm(len(self.x))[:self.batch_size]
            x_batch = self.x[random_index]
            y_batch = self.y[random_index, :][:, random_index]
            yield x_batch, y_batch

    def __len__(self):
        return self.sample_num


# data cleaning
def clean_data():
    # read raw data
    samples_df = pd.read_csv('./data/real/raw_samples.csv')
    interactions_df = pd.read_csv('./data/real/raw_interactions.csv',
                                  usecols=[0, 1])

    # get intersection of genes
    gene_in_samples = samples_df.iloc[:, 0].to_numpy()
    gene_in_interactions = interactions_df.to_numpy()
    name_intersection = np.intersect1d(gene_in_samples, gene_in_interactions)

    # get cleaned samples
    raw_samples = samples_df.to_numpy()
    new_samples = samples_df.columns.to_numpy()

    for x in raw_samples:
        if x[0] in name_intersection:
            new_samples = np.vstack((new_samples, x))

    pd.DataFrame(new_samples).to_csv('./data/real/samples.csv',
                                     header=False,
                                     index=False)

    # get cleaned interactions
    raw_interactions = interactions_df.to_numpy()
    new_interactions = interactions_df.columns.to_numpy()

    for x in raw_interactions:
        if x[0] in name_intersection and x[1] in name_intersection:
            new_interactions = np.vstack((new_interactions, x))

    pd.DataFrame(new_interactions).to_csv('./data/real/interactions.csv',
                                          header=False,
                                          index=False)


# read data from file
def read_raw_data(otu_path, adj_path):
    with open(otu_path) as f:
        reader = csv.reader(f)
        otu = np.array(list(reader), dtype=np.float32)

    with open(adj_path) as f:
        reader = csv.reader(f)
        adj = np.array(list(reader), dtype=np.float32)

    return otu, adj


# get dataset
def get_cora_dataset():
    dataset = Planetoid('./data', 'Cora')
    data = dataset[0]

    return Data(x=data.x, edge_index=data.edge_index)


def get_real_dataset():
    samples_df = pd.read_csv('./data/real/samples.csv')
    interactions_df = pd.read_csv('./data/real/interactions.csv')

    x = samples_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    x = torch.as_tensor(x, dtype=torch.float32)

    gene_names = samples_df.name.to_list()
    gene1 = interactions_df.gene1.to_list()
    gene2 = interactions_df.gene2.to_list()

    gene1_index = torch.as_tensor([gene_names.index(i) for i in gene1],
                                  dtype=torch.int32)
    gene2_index = torch.as_tensor([gene_names.index(i) for i in gene2],
                                  dtype=torch.int32)
    edge_index = torch.vstack((gene1_index, gene2_index))

    return Data(x=x, edge_index=edge_index)


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


# draw graph
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


# draw raw graph
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
