import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from minepy import cstats, pstats
from scipy.sparse import coo_matrix
from scipy.stats import spearmanr

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'

smpl_path = './data/real/processed/samples_trrust_tb.csv'
intr_path = './data/real/processed/interactions_trrust_tb.csv'
# smpl_path = './data/synthetic/samples.csv'
# intr_path = './data/synthetic/interactions.csv'

feature_size = 40
contrasive_loss_m = 10.
potential_loss_l = 10.

is_use_gpu = torch.cuda.is_available()
is_use_gpu = False


def sne(x):
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    if is_use_gpu:
        torch_x = torch_x.cuda()
        y_hat = model(torch_x)[1].cpu()
    else:
        y_hat = model(torch_x)[1]
    zeros = torch.zeros(1)
    return torch.maximum(zeros, -(y_hat / potential_loss_l - 1)**2 + 1).numpy()


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
    return np.corrcoef(x)
    return lss(x, 3)


def mic(x):
    return np.corrcoef(x)
    return cstats(x, x, alpha=0.6, c=15, est="mic_e")[0]


@torch.no_grad()
def predict(x, y, score_thresh):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
    sne_init_score = sne(x)

    save_dense_to_interactions(sne_init_score,
                               './data/predicted/interactions.csv')

    # use upper triangular matrix to calculate
    y = get_triu_items(y)
    sne_init_score = get_triu_items(sne_init_score)

    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    sne_cor_matrix = np.abs(sne_init_score) > score_thresh

    print("计算相关对数……")
    sne_intr_num = np.count_nonzero(sne_cor_matrix)
    ground_truth_intr_num = np.count_nonzero(y)

    print('计算真阳性数……')
    # np.multiply(pcc_cor_matrix, y) 得到对位相乘结果，而 * 不行，离谱
    sne_true_pos = np.count_nonzero(np.multiply(sne_cor_matrix, y))

    # 计算精确率，此时只需要考虑预测正确，且真的正确的值
    print('计算精确率……')
    sne_precision = sne_true_pos / sne_intr_num

    # 计算召回率，此时只需要考虑真的正确，且我们预测正确的值
    print('计算召回率……')
    sne_recall = sne_true_pos / ground_truth_intr_num

    print(f'''{"":16}{"SNE":8}{"Ground Truth":12}
{"Interactions":16}{sne_intr_num:<8}{ground_truth_intr_num:<12}
{"True Pos":16}{sne_true_pos:<8}
{"Precision":16}{sne_precision:<8.2%}
{"Recall":16}{sne_recall:<8.2%}''')


@torch.no_grad()
def predict_and_compare(x, y, perm_num, p_val_thresh, score_thresh):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
    sne_init_score = sne(x)
    pcc_init_score = pcc(x)
    spm_init_score = spm(x)
    lsa_init_score = lsa(x)
    mic_init_score = mic(x)

    np.fill_diagonal(y, 0.)
    np.fill_diagonal(sne_init_score, np.nan)
    np.fill_diagonal(pcc_init_score, np.nan)
    np.fill_diagonal(spm_init_score, np.nan)
    np.fill_diagonal(lsa_init_score, np.nan)
    np.fill_diagonal(mic_init_score, np.nan)

    # permutated statistics
    sne_p_count = 0
    pcc_p_count = 0
    spm_p_count = 0
    lsa_p_count = 0
    mic_p_count = 0
    for i in range(perm_num):
        x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
        sne_perm_score = sne(x)
        pcc_perm_score = pcc(x)
        spm_perm_score = spm(x)
        lsa_perm_score = lsa(x)
        mic_perm_score = mic(x)

        sne_p_count += np.abs(sne_perm_score) > np.abs(sne_init_score)
        pcc_p_count += np.abs(pcc_perm_score) > np.abs(pcc_init_score)
        spm_p_count += np.abs(spm_perm_score) > np.abs(spm_init_score)
        lsa_p_count += np.abs(lsa_perm_score) > np.abs(lsa_init_score)
        mic_p_count += np.abs(mic_perm_score) > np.abs(mic_init_score)

        if i % 10 == 0:
            print(f'{i}th computation done')

    # compute p values
    sne_p = sne_p_count / perm_num
    pcc_p = pcc_p_count / perm_num
    spm_p = spm_p_count / perm_num
    lsa_p = lsa_p_count / perm_num
    mic_p = mic_p_count / perm_num

    # the elements with p value less than the threshold
    sne_init_score *= (sne_p < p_val_thresh)
    pcc_init_score *= (pcc_p < p_val_thresh)
    spm_init_score *= (spm_p < p_val_thresh)
    lsa_init_score *= (lsa_p < p_val_thresh)
    mic_init_score *= (mic_p < p_val_thresh)

    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    sne_cor_matrix = np.abs(sne_init_score) > score_thresh
    pcc_cor_matrix = np.abs(pcc_init_score) > score_thresh
    spm_cor_matrix = np.abs(spm_init_score) > score_thresh
    lsa_cor_matrix = np.abs(lsa_init_score) > score_thresh
    mic_cor_matrix = np.abs(mic_init_score) > score_thresh

    print("计算相关对数……")
    sne_intr_num = np.count_nonzero(sne_cor_matrix)
    pcc_intr_num = np.count_nonzero(pcc_cor_matrix)
    spm_intr_num = np.count_nonzero(spm_cor_matrix)
    lsa_intr_num = np.count_nonzero(lsa_cor_matrix)
    mic_intr_num = np.count_nonzero(mic_cor_matrix)
    ground_truth_intr_num = np.count_nonzero(y)

    print('计算真阳性数……')
    # np.multiply(pcc_cor_matrix, y) 得到对位相乘结果，而 * 不行，离谱
    sne_true_pos = np.count_nonzero(np.multiply(sne_cor_matrix, y))
    pcc_true_pos = np.count_nonzero(np.multiply(pcc_cor_matrix, y))
    spm_true_pos = np.count_nonzero(np.multiply(spm_cor_matrix, y))
    lsa_true_pos = np.count_nonzero(np.multiply(lsa_cor_matrix, y))
    mic_true_pos = np.count_nonzero(np.multiply(mic_cor_matrix, y))

    # 计算精确率，此时只需要考虑预测正确，且真的正确的值
    print('计算精确率……')
    sne_precision = sne_true_pos / sne_intr_num
    pcc_precision = pcc_true_pos / pcc_intr_num
    spm_precision = spm_true_pos / spm_intr_num
    lsa_precision = lsa_true_pos / lsa_intr_num
    mic_precision = mic_true_pos / mic_intr_num

    # 计算召回率，此时只需要考虑真的正确，且我们预测正确的值
    print('计算召回率……')
    sne_recall = sne_true_pos / ground_truth_intr_num
    pcc_recall = pcc_true_pos / ground_truth_intr_num
    spm_recall = spm_true_pos / ground_truth_intr_num
    lsa_recall = lsa_true_pos / ground_truth_intr_num
    mic_recall = mic_true_pos / ground_truth_intr_num

    print(
        f'''{"":16}{"SNE":8}{"PCC":8}{"SPM":8}{"LSA":8}{"MIC":8}{"Ground Truth":12}
{"Interactions":16}{sne_intr_num:<8}{pcc_intr_num:<8}{spm_intr_num:<8}{lsa_intr_num:<8}{mic_intr_num:<8}{ground_truth_intr_num:<12}
{"True Pos":16}{sne_true_pos:<8}{pcc_true_pos:<8}{spm_true_pos:<8}{lsa_true_pos:<8}{mic_true_pos:<8}
{"Precision":16}{sne_precision:<8.2%}{pcc_precision:<8.2%}{spm_precision:<8.2%}{lsa_precision:<8.2%}{mic_precision:<8.2%}
{"Recall":16}{sne_recall:<8.2%}{pcc_recall:<8.2%}{spm_recall:<8.2%}{lsa_recall:<8.2%}{mic_recall:<8.2%}'''
    )


if __name__ == '__main__':
    torch_data = get_dataset(smpl_path, intr_path, feature_size)
    # torch_data = get_cora_dataset(train=False)

    x = np.asarray(torch_data.x)
    edge_index = np.asarray(torch_data.edge_index)
    y = coo_matrix((np.ones(edge_index.shape[1]), edge_index),
                   (x.shape[0], x.shape[0])).todense()

    model = Model(feature_size, 256, 256)
    if is_use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # ---------- quick evaluate ----------
    predict(x, y, model, score_thresh=0.)

    # ---------- fully evaluate ----------
    # x_tmp = x[:300].copy()
    # y_tmp = y[:300, :300].copy()
    # predict_and_compare(x_tmp,
    #                     y_tmp,
    #                     model,
    #                     perm_num=100,
    #                     p_val_thresh=0.1,
    #                     score_thresh=0.8)
