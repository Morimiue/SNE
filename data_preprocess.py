import numpy as np
import pandas as pd


# get intersection of genes
samples_df = pd.read_csv('./data/real/raw_samples.csv')
gene_in_samples = samples_df.name.to_list()
gene_in_samples = set(gene_in_samples)

interactions_df = pd.read_csv(
    './data/real/raw_interactions.csv', usecols=[0, 1])
gene1 = interactions_df.gene1.to_list()
gene2 = interactions_df.gene2.to_list()
gene_in_interactions = set(gene1 + gene2)

name_intersection = gene_in_samples & gene_in_interactions

# get new samples
raw_samples = samples_df.values
new_samples = np.array(samples_df.columns)

for x in raw_samples:
    if x[0] in name_intersection:
        new_samples = np.vstack((new_samples, x))

pd.DataFrame(new_samples).to_csv(
    './data/real/samples.csv', header=False, index=False)

# get new interactions
raw_interactions = interactions_df.values
new_interactions = np.array(interactions_df.columns)

for x in raw_interactions:
    if x[0] in name_intersection and x[1] in name_intersection:
        new_interactions = np.vstack((new_interactions, x))

pd.DataFrame(new_interactions).to_csv(
    './data/real/interactions.csv', header=False, index=False)
