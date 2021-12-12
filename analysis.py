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


@torch.no_grad()
def get_p_values(x, y, perm_num, threshold, cut_threshold, model):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
    pcc_init_score = np.corrcoef(x)
    spm_init_score = spearmanr(x, axis=1)[0]
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    if is_using_gpu:
        torch_x = torch_x.cuda()
        sne_init_score = model(torch_x)[1].detach().cpu().numpy()
    else:
        sne_init_score = model(torch_x)[1].numpy()
    n = y.shape[0]
    # 设置对角线元素为1
    y[range(n), range(n)] = 1
    pcc_init_score[range(n), range(n)] = 1
    spm_init_score[range(n), range(n)] = 1
    sne_init_score[range(n), range(n)] = 1
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
            torch_x = torch_x.cuda()
            sne_score[i] = model(torch_x)[1].detach().cpu().numpy()
        else:
            sne_score[i] = model(torch_x)[1].numpy()

        if i % 100 == 0:
            print(f'{i}th computation done')
    # compute p values
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(pcc_init_score),
                   axis=0) / perm_num
    spm_p = np.sum(np.abs(spm_score) > np.abs(spm_init_score),
                   axis=0) / perm_num
    sne_p = np.sum(np.abs(sne_score) > np.abs(sne_init_score),
                   axis=0) / perm_num

    # the element with p value less than the threshold
    pcc_ok = np.sum(pcc_p < threshold) / 2
    spm_ok = np.sum(spm_p < threshold) / 2
    sne_ok = np.sum(sne_p < threshold) / 2
    print(pcc_ok, spm_ok, sne_ok)
    # print(np.count_nonzero(pcc_init_score), np.count_nonzero(spm_init_score),
    #   np.count_nonzero(sne_init_score), np.count_nonzero(y))
    # 保留 p-value < 0.1 的元素
    pcc_init_score *= (pcc_p < threshold)
    spm_init_score *= (spm_p < threshold)
    sne_init_score *= (sne_p < threshold)
    # print(np.count_nonzero(pcc_init_score), np.count_nonzero(spm_init_score),
    #   np.count_nonzero(sne_init_score), np.count_nonzero(y))
    # 判断是否相关，大于阈值则视为相关
    pcc_init_score = (np.abs(pcc_init_score) > cut_threshold).astype(float)
    spm_init_score = (np.abs(spm_init_score) > cut_threshold).astype(float)
    sne_init_score = (np.abs(sne_init_score) > cut_threshold).astype(float)

    # print(np.count_nonzero(pcc_init_score), np.count_nonzero(spm_init_score),
    #   np.count_nonzero(sne_init_score), np.count_nonzero(y))
    # 统计与原始矩阵相同元素比例，评判模型优劣
    pcc_final_all = (pcc_init_score == y).astype(float).mean()
    spm_final_all = (spm_init_score == y).astype(float).mean()
    sne_final_all = (sne_init_score == y).astype(float).mean()

    # 计算精确率，评判模型优劣
    pcc_equal_y = (pcc_init_score * y).astype(float).sum()
    spm_equal_y = (spm_init_score * y).astype(float).sum()
    sne_equal_y = (sne_init_score * y).astype(float).sum()

    pcc_precision = pcc_equal_y / np.count_nonzero(pcc_init_score)
    spm_precision = spm_equal_y / np.count_nonzero(spm_init_score)
    sne_precision = sne_equal_y / np.count_nonzero(sne_init_score)
    print(pcc_precision, spm_precision, sne_precision)
    # return pcc_final_all, spm_final_all, sne_final_all
    # 计算召回率，评判模型优劣

    pcc_call_back = pcc_equal_y / np.count_nonzero(pcc_init_score)
    spm_call_back = spm_equal_y / np.count_nonzero(spm_init_score)
    sne_call_back = sne_equal_y / np.count_nonzero(sne_init_score)
    print(pcc_call_back, spm_call_back, sne_call_back)


if __name__ == '__main__':
    # torch_data = get_real_dataset()
    torch_data = get_cora_dataset()
    x = np.asarray(torch_data.x)
    edge_index = np.asarray(torch_data.edge_index)
    y = coo_matrix((np.ones(edge_index.shape[1]), edge_index),
                   (x.shape[0], x.shape[0])).todense()
    if is_using_gpu:
        torch_data = torch_data.cuda()

    model = Model(12, 256, 256)
    model.load_state_dict(torch.load(model_path))
    if is_using_gpu:
        model = model.cuda()

    model.eval()
    t = y[100:200, 100:200].copy()
    get_p_values(x[100:200], t, 1000, 0.01, 0.9, model)
    # get_p_values(x, y, 30, 0.01, 0.9, model)
    # length_of_pieces = int(x.shape[0])
    # print(length_of_pieces)
    # pcc = 0
    # spm = 0
    # sne = 0
    # for i in range(0, length_of_pieces - 1):
    #     if (i % 100 == 0):
    #         piece_of_y = y[i:i + 100, i:i + 100].copy()
    #         a, b, c = get_p_values(x[i:i + 100], piece_of_y, 300, 0.01, 0.8,
    #                                model)
    #         pcc += a / length_of_pieces
    #         spm += b / length_of_pieces
    #         sne += c / length_of_pieces

    #         print(f'{i*100} to {(i+1)*100}th elements done')
    # print(pcc, spm, sne)
