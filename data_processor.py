import csv
import numpy as np
import torch
import torch.utils.data as Data


# Read data from file
def read_raw_data(otu_path, adj_path):
    with open(otu_path) as f:
        reader = csv.reader(f)
        otu = np.array(list(reader), dtype=np.float32)
        otu = otu.T

    with open(adj_path) as f:
        reader = csv.reader(f)
        adj = np.array(list(reader), dtype=np.float32)

    return otu, adj


# Get dataset
def get_dateset(otu, adj):
    unit_num = otu.shape[0]
    sample_num = otu.shape[1]

    x_train = np.zeros((unit_num * (unit_num - 1), sample_num * 2))
    y_train = np.zeros((unit_num * (unit_num - 1), 1))

    cnt = 0
    for i in range(unit_num):
        for j in range(unit_num):
            if i == j:
                continue
            x_train[cnt] = np.append(otu[i], otu[j])
            cnt += 1

    cnt = 0
    for i in range(unit_num):
        for j in range(unit_num):
            if i == j:
                continue
            y_train[cnt] = adj[i][j]
            cnt += 1

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    return Data.TensorDataset(x_train, y_train)
