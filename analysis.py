import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'
samples = './data/real/samples.csv'
interactions = './data/real/interactions.csv'

is_using_gpu = torch.cuda.is_available()


@torch.no_grad()
def evaluate_our_model(x, y, perm_num, threshold, cut_threshold, model):
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

        if i % 10 == 0:
            print(f'{i}th computation done')
    # compute p values
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(pcc_init_score),
                   axis=0) / perm_num
    spm_p = np.sum(np.abs(spm_score) > np.abs(spm_init_score),
                   axis=0) / perm_num
    sne_p = np.sum(np.abs(sne_score) > np.abs(sne_init_score),
                   axis=0) / perm_num

    # the element with p value less than the threshold
    # pcc_ok = np.sum(pcc_p < threshold) / 2
    # spm_ok = np.sum(spm_p < threshold) / 2
    # sne_ok = np.sum(sne_p < threshold) / 2
    # 保留 p-value < 0.1 的元素
    pcc_init_score *= (pcc_p < threshold)
    spm_init_score *= (spm_p < threshold)
    sne_init_score *= (sne_p < threshold)
    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    pcc_cor_matrix = (np.abs(pcc_init_score) > cut_threshold).astype(float)
    spm_cor_matrix = (np.abs(spm_init_score) > cut_threshold).astype(float)
    sne_cor_matrix = (np.abs(sne_init_score) > cut_threshold).astype(float)

    print("四者非0元素个数")
    print(np.count_nonzero(pcc_cor_matrix), np.count_nonzero(spm_cor_matrix),
          np.count_nonzero(sne_cor_matrix), np.count_nonzero(y))
    # 我们预测相关，且p-value < threshold的元素数量
    # np.count_nonzero 仅统计非对角线元素
    # np.multiply(pcc_cor_matrix, y) 得到对位相乘结果，而 * 不行，离谱
    pcc_equal_y = np.count_nonzero(np.multiply(pcc_cor_matrix, y))
    spm_equal_y = np.count_nonzero(np.multiply(spm_cor_matrix, y))
    sne_equal_y = np.count_nonzero(np.multiply(sne_cor_matrix, y))
    print('三者真阳性数')
    print(pcc_equal_y, spm_equal_y, sne_equal_y)
    # 计算精确率，此时只需要考虑预测正确，且真的正确的值
    pcc_precision = pcc_equal_y / np.count_nonzero(pcc_cor_matrix)
    spm_precision = spm_equal_y / np.count_nonzero(spm_cor_matrix)
    sne_precision = sne_equal_y / np.count_nonzero(sne_cor_matrix)
    print('三者精确率')
    # pcc_precision = 1
    # spm_precision = 1
    print(pcc_precision, spm_precision, sne_precision)
    # 计算召回率，此时只需要考虑真的正确，且我们预测正确的值

    pcc_call_back = pcc_equal_y / np.count_nonzero(y)
    spm_call_back = spm_equal_y / np.count_nonzero(y)
    sne_call_back = sne_equal_y / np.count_nonzero(y)
    print('三者召回率')
    print(pcc_call_back, spm_call_back, sne_call_back)


if __name__ == '__main__':
    torch_data = get_real_dataset(samples, interactions)
    # torch_data = get_cora_dataset()
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
    t = y[:100, :100].copy()
    evaluate_our_model(x[:100], t, 1000, 0.1, 0.9, model)
    # evaluate_our_model(x, y, 100, 0.1, 0.9, model)
