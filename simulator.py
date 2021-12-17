import numpy as np
import random
import csv
import math

hub_num = 20
speicies_num = 2000
samples_num = 50
Huge_Enough = 10


def generate_simulated_data(hub_num, speicies_num, samples_num):

    mean = np.random.rand(speicies_num)
    mean -= 1
    hubs = random.sample(range(speicies_num), hub_num)
    # 生成协方差矩阵
    cov = np.zeros([speicies_num, speicies_num])
    for i in range(speicies_num):
        for j in range(i - 1):
            if ((i in hubs) or (j in hubs)):
                cov[i, j] = np.random.choice([0, 0.2], p=[.93, .07])
            else:
                cov[i, j] = np.random.choice([0, 0.2], p=[.992, .008])
    cov = cov + cov.T
    np.fill_diagonal(cov, Huge_Enough)
    # 获得邻接矩阵
    adj = np.array(cov, dtype=bool)
    # 归一化，以得到正定矩阵
    for co in cov:
        _range = np.max(co) - np.min(co)
        co = (co - np.min(co)) / _range
    # lny 服从多元高斯分布，我们要获得y
    otus = np.random.multivariate_normal(mean, cov, (samples_num), 'raise')
    otus = np.exp(otus)
    otu_data_path = 'data/sim/samples.csv'
    with open(otu_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(otus.T)
    interactions_path = 'data/sim/interactions.csv'
    with open(interactions_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(adj)


if __name__ == '__main__':

    generate_simulated_data(hub_num, speicies_num, samples_num)