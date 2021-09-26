import csv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.utils.data as Data


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
def get_dateset(otu, adj):
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

    return Data.TensorDataset(x_train, y_train)


# Draw Graph
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
