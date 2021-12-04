import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'

is_using_gpu = torch.cuda.is_available()


def get_p_values(x, perm_num, threshold, model):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
    pcc_init_score = np.corrcoef(x)
    spm_init_score = spearmanr(x, axis=1)[0]
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    if is_using_gpu:
        torch_x.cuda()
    sne_init_score = model(torch_x)[1].detach().numpy()
    # permutated statistics
    pcc_score = np.zeros((perm_num, x.shape[0], x.shape[0]))
    spm_score = np.zeros((perm_num, x.shape[0], x.shape[0]))
    sne_score = np.zeros((perm_num, x.shape[0], x.shape[0]))
    for i in range(perm_num):
        x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
        pcc_score[i] = np.corrcoef(x)
        spm_score[i] = spearmanr(x, axis=1)[0]
        torch_x = torch.as_tensor(x, dtype=torch.float32)
        if is_using_gpu:
            torch_x.cuda()
        sne_score[i] = model(torch_x)[1].detach().numpy()

        if i % 100 == 0:
            print(f'{i}th computation done')
    # compute p values
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(pcc_init_score),
                   axis=0) / perm_num
    spm_p = np.sum(np.abs(spm_score) > np.abs(spm_init_score),
                   axis=0) / perm_num
    sne_p = np.sum(np.abs(sne_score) > np.abs(sne_init_score),
                   axis=0) / perm_num
    #
    pcc_ok = np.sum(pcc_p < threshold) / 2
    spm_ok = np.sum(spm_p < threshold) / 2
    sne_ok = np.sum(sne_p < threshold) / 2
    print(pcc_ok, spm_ok, sne_ok)


if __name__ == '__main__':
    torch_data = get_real_dataset()
    # torch_data = get_cora_dataset()
    x = np.asarray(torch_data.x)
    edge_index = np.asarray(torch_data.edge_index)
    y = coo_matrix((np.ones(edge_index.shape[1]), edge_index),
                   (x.shape[0], x.shape[0])).todense()
    if is_using_gpu:
        torch_data = torch_data.cuda()

    model = Model(12, 256, 256)
    if is_using_gpu:
        model = model.cuda()

    model.eval()

    get_p_values(x[:100], 1000, 0.01, model)
