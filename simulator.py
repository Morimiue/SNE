import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from utils import save_dense_to_interactions

smpl_path = './data/synthetic/samples2.csv'
intr_path = './data/synthetic/interactions2.csv'

species_num = 1000
hub_num = 10
sample_num = 40

diagonal_value = 10.


def generate_simulated_data(species_num: int, hub_num: int, sample_num: int):
    """"""
    # generate the covariance matrix with hub model
    hubs = np.random.choice(np.arange(species_num), hub_num, replace=False)
    adj = np.zeros([species_num, species_num])
    for i in range(species_num):
        for j in range(i):
            if i in hubs or j in hubs:
                adj[i, j] = np.random.choice([0., .2], p=[.92, .08])
            else:
                adj[i, j] = np.random.choice([0., .2], p=[.992, .008])
    cov = adj + adj.T
    np.fill_diagonal(cov, diagonal_value)
    # save interactions
    save_dense_to_interactions(adj, intr_path)
    # get the otu table
    mean = np.random.rand(species_num) - .5
    otus = np.random.multivariate_normal(mean, cov, (sample_num), 'raise')
    otus = np.exp(otus).T
    # save samples
    otu_names = np.char.add('OTU', np.arange(species_num).astype(str))
    otus = np.hstack((otu_names[:, None], otus))
    df = pd.DataFrame(otus,
                      columns=['name'] + [f's{i}' for i in range(sample_num)])
    df.to_csv(smpl_path, index=False)


if __name__ == '__main__':
    generate_simulated_data(species_num, hub_num, sample_num)
