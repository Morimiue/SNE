import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
from scipy.sparse import coo_matrix

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'

is_using_gpu = torch.cuda.is_available()

# calculate the permutation p-value of diffrernt methods
# x: the series1 of the correlation calculation
# y: the series2 of the correlation calculation
# num_permutation: after how many permutations we get the p-value
# method: which methods' p-value are we calculating?
# y_hat: the predicted correlation value of our method between x and y, single value, not a matrix


def get_perm_pvalue(x, y, num_permutation, method, y_hat):
    initial_test_statistic = np.abs(cal_corvalue(x, y, method, y_hat))
    Scores_after_perm = np.zeros(num_permutation, dtype='float')
    for i in range(0, num_permutation):
        x = np.random.permutation(x)
        y = np.random.permutation(y)
        Scores_after_perm[i] = float(cal_corvalue(x, y, method, y_hat))
    p_two_tail = np.sum(
        np.abs(Scores_after_perm) >= initial_test_statistic)/np.float(num_permutation)
    return p_two_tail

# x: the series1 of the correlation calculation
# y: the series2 of the correlation calculation
# method: which methods' p-value are we calculating?


def cal_corvalue(x, y, method, y_hat):
    x = np.asarray(x)
    y = np.asarray(y)
    if method == 'Pearson':
        return stats.pearsonr(x, y)[0]
    elif(method == 'Spearman'):
        return stats.spearmanr(x, y)[0]
    elif(method == 'nbd'):
        return y_hat
    else:
        print("Error: 相关度计算参数不正确，可选参数为：Pearson,Spearman,nbd")
    return


    # significance: if the p-value is less than the threshold
    # we reject the null-hypothesis, and accept that our judgement is true
    # which means these two series is correlated
if __name__ == '__main__':
    torch_data = get_real_dataset()
    # torch_data = get_cora_dataset()

    x = np.asarray(torch_data.x)
    edge_index = np.asarray(torch_data.edge_index)
    y = coo_matrix((np.ones(edge_index.shape[1]), edge_index),
                   (x.shape[0], x.shape[0])).todense()

    model = Model(12, 256, 256)
    if is_using_gpu:
        model = model.cuda()

    model.eval()

    torch_z, torch_y_hat = model(torch_data.x)
    print(torch_y_hat)
