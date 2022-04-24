from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minepy import cstats, pstats
from scipy.sparse import coo_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model_sim.pth'
smpl_path = './data/synthetic/samples2.csv'
intr_path = './data/synthetic/interactions2.csv'
results_dir_path = './data/inference_sim/'

# model_path = './models/gmlp_model_sim.pth'
# smpl_path = './data/real/processed/samples_trrust_tb.csv'
# intr_path = './data/real/processed/interactions_trrust_tb.csv'
# results_dir_path = './data/inference_gene/'

# model_path = './models/gmlp_model_sim_675.pth'
# smpl_path = './data/microbiome/microbiome2.csv'
# results_dir_path = './data/inference_microb/'

feature_size = 40
contrasive_loss_m = 10.
potential_loss_l = 10.

is_use_gpu = torch.cuda.is_available()
is_save_results = True


def sne(x):
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    if is_use_gpu:
        torch_x = torch_x.cuda()
        y_hat = model(torch_x).cpu().detach().numpy()
    else:
        y_hat = model(torch_x).detach().numpy()
    y_hat = euclidean_distances(y_hat)
    return np.maximum(0., -(y_hat / potential_loss_l - 1)**2 + 1)


def pcc(x):
    return np.corrcoef(x)


def spm(x):
    return spearmanr(x, axis=1)[0]


def _lss_pair(o1: np.ndarray, o2: np.ndarray, D: int) -> float:
    N = len(o1)
    o1 = o1.reshape(-1, 1)
    o2 = o2.reshape(1, -1)
    dot_product = np.dot(o1, o2)
    psm = np.zeros([N, N])
    nsm = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if abs(i - j) <= D:
                psm[i][j] = max(0, psm[i - 1][j - 1] + dot_product[i][j])
                nsm[i][j] = max(0, nsm[i - 1][j - 1] - dot_product[i][j])
    max_p = psm.max()
    max_n = nsm.max()
    max_score = max(max_p, max_n)
    flag = np.sign(max_p - max_n)
    return flag * max_score / N


def lss(data: np.ndarray, D: int) -> np.ndarray:
    N = len(data)
    res = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            res[i][j] = _lss_pair(data[i], data[j], D)
    res = res + res.T
    for i in range(N):
        res[i][i] = _lss_pair(data[i], data[i], D)
    return res


def lsa(x):
    return lss(x, 3)


def mic(x):
    return cstats(x, x, alpha=0.6, c=15, est="mic_e")[0]


@torch.no_grad()
def predict(x: np.ndarray,
            score_thresh: float = 0.8,
            p_val_thresh: float = 0.05,
            perm_num: int = 0) -> None:
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''

    init_scores = np.empty((len(x), len(x)))
    if perm_num > 0:
        perm_scores = np.empty((len(x), len(x)))
        p_values = np.zeros((len(x), len(x)))
    cor_matrixs = np.empty((len(x), len(x)))
    cor_matrixs_top = np.zeros((len(x), len(x)))

    # initial test statistics
    init_scores = sne(x)
    np.fill_diagonal(init_scores, np.nan)

    # permutated statistics
    if perm_num > 0:
        for i in range(perm_num):
            x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
            perm_scores = sne(x)
            p_values += np.abs(perm_scores) > np.abs(init_scores)
            print(f'permutation test: {i+1} / {perm_num}', end='\r')
        p_values /= perm_num

    # compute the correlation matrix
    if perm_num > 0:
        init_scores = np.multiply(init_scores, p_values <= p_val_thresh)

    if perm_num > 0:
        cor_matrixs = np.multiply(
            np.abs(init_scores) >= score_thresh, p_values <= p_val_thresh)
    else:
        cor_matrixs = np.abs(init_scores) >= score_thresh
    intr_counts = np.count_nonzero(cor_matrixs)

    np.fill_diagonal(init_scores, 0.)
    init_scores[np.tril_indices_from(init_scores, -1000)] = 0.
    xs, ys = np.unravel_index(
        np.argsort(init_scores.ravel())[-1000:], init_scores.shape)
    for i in range(len(xs)):
        cor_matrixs_top[xs[i]][ys[i]] = init_scores[xs[i]][ys[i]]

    # save results
    print(intr_counts)
    if is_save_results:
        np.save(results_dir_path + 'init_scores.npy', init_scores)
        np.save(results_dir_path + 'cor_matrixs_top.npy', cor_matrixs_top)

        if perm_num:
            p_counts = np.empty(10, dtype=np.int_)
            for i in range(10):
                p_counts[i] = np.count_nonzero(p_values < (i + 1) / 10)
            for i in range(10 - 1, 0, -1):
                p_counts[i] -= p_counts[i - 1]

            np.save(results_dir_path + 'p_values.npy', p_values)
            np.save(results_dir_path + 'p_counts.npy', p_counts)
            np.savetxt(results_dir_path + 'p_counts.txt', p_counts, fmt='%d')

        save_dense_to_interactions(cor_matrixs_top,
                                   node_names=node_names,
                                   path=results_dir_path + 'interactions.csv')


@torch.no_grad()
def predict_and_compare(x: np.ndarray,
                        y: np.ndarray,
                        methods: tuple[function, ...],
                        score_thresh: tuple[float, ...],
                        p_val_thresh: float = 0.05,
                        perm_num: int = 0) -> None:
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    np.fill_diagonal(y, 0.)
    # y = get_triu_items(y)
    ground_truth_intr_count = np.count_nonzero(y)

    init_scores = np.empty((len(methods), len(x), len(x)))
    if perm_num > 0:
        perm_scores = np.empty((len(methods), len(x), len(x)))
        p_values = np.zeros((len(methods), len(x), len(x)))
    cor_matrixs = np.empty((len(methods), len(x), len(x)))

    intr_counts = np.empty(len(methods), dtype=np.int_)
    true_posits = np.empty(len(methods), dtype=np.int_)
    precisions = np.empty(len(methods))
    recalls = np.empty(len(methods))

    # initial test statistics
    for i in range(len(methods)):
        init_scores[i] = methods[i](x)
        np.fill_diagonal(init_scores[i], np.nan)

    # permutated statistics
    if perm_num > 0:
        for i in range(perm_num):
            x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
            for j in range(len(methods)):
                perm_scores[j] = methods[j](x)
                p_values[j] += np.abs(perm_scores[j]) > np.abs(init_scores[j])
            print(f'permutation test: {i+1} / {perm_num}', end='\r')
        p_values /= perm_num

    # compute the correlation matrix for each method
    for i in range(len(methods)):
        if perm_num > 0:
            cor_matrixs[i] = np.multiply(
                np.abs(init_scores[i]) >= score_thresh[i],
                p_values[i] <= p_val_thresh)
        else:
            cor_matrixs[i] = np.abs(init_scores[i]) >= score_thresh[i]
        intr_counts[i] = np.count_nonzero(cor_matrixs[i])
        true_posits[i] = np.count_nonzero(np.multiply(cor_matrixs[i], y))
        precisions[i] = true_posits[i] / intr_counts[i]
        recalls[i] = true_posits[i] / ground_truth_intr_count

    # print results
    print(f'{"":16}', end='')
    for x in map(lambda x: x.__name__, methods):
        print(f'{x:12}', end='')
    print('Ground Truth')

    print(f'{"Interactions":16}', end='')
    for x in intr_counts:
        print(f'{x:<12}', end='')
    print(ground_truth_intr_count)

    print(f'{"True Positives":16}', end='')
    for x in true_posits:
        print(f'{x:<12}', end='')
    print()

    print(f'{"Precision":16}', end='')
    for x in precisions:
        print(f'{x:<12.2%}', end='')
    print()

    print(f'{"Recall":16}', end='')
    for x in recalls:
        print(f'{x:<12.2%}', end='')
    print()

    # save results
    if is_save_results:
        np.save(results_dir_path + 'precisions.npy', precisions)
        np.save(results_dir_path + 'recalls.npy', recalls)

        if perm_num:
            p_counts = np.empty(10, dtype=np.int_)
            for i in range(10):
                p_counts[i] = np.count_nonzero(p_values[0] < (i + 1) / 10)
            for i in range(10 - 1, 0, -1):
                p_counts[i] -= p_counts[i - 1]

            np.save(results_dir_path + 'p_values.npy', p_values[0])
            np.save(results_dir_path + 'p_counts.npy', p_counts)
            np.savetxt(results_dir_path + 'p_counts.txt', p_counts, fmt='%d')

        save_dense_to_interactions(cor_matrixs[0],
                                   results_dir_path + 'interactions.csv')


if __name__ == '__main__':
    torch_data = get_dataset(smpl_path, intr_path, feature_size)
    # torch_data = get_dataset(smpl_path)
    # torch_data = get_cora_dataset(train=False)

    x = np.asarray(torch_data.x)
    edge_index = np.asarray(torch_data.edge_index)
    y = coo_matrix((np.ones(edge_index.shape[1]), edge_index),
                   (x.shape[0], x.shape[0])).todense()
    node_names = torch_data.node_names

    model = Model(feature_size, 256, 256)
    if is_use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # x = x[:300].copy()
    # y = y[:300, :300].copy()
    # x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)

    # predict(x, score_thresh=0.8, p_val_thresh=0.05, perm_num=200)

    predict_and_compare(
        x,
        y,
        methods=(sne, pcc, spm),
        score_thresh=(0.0, 0.4, 0.4),
        # score_thresh=(0.8, 0.8, 0.8),
        p_val_thresh=0.05,
        perm_num=100)
