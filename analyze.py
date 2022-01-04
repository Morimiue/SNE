import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.stats import spearmanr

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'

raw_smpl_path = './data/real/raw/samples_raw_tuberculosis.csv'
raw_intr_path = './data/real/raw/interactions_raw_trrust.csv'
smpl_path = './data/real/processed/samples_trrust_tuber.csv'
intr_path = './data/real/processed/interactions_trrust_tuber.csv'

feature_size = 16
contrasive_loss_m = 700.
potential_loss_l = 2.

is_use_gpu = torch.cuda.is_available()


@torch.no_grad()
def evaluate_model(x, y, model, perm_num, p_val_thresh, score_thresh):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
    pcc_init_score = np.corrcoef(x)
    spm_init_score = spearmanr(x, axis=1)[0]
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    zeros = torch.zeros(y.shape)
    if is_use_gpu:
        torch_x = torch_x.cuda()
        y_hat = model(torch_x)[1].cpu()
    else:
        y_hat = model(torch_x)[1]
    sne_init_score = torch.maximum(zeros, -(y_hat / potential_loss_l - 1)**2 +
                                   1).numpy()

    # 设置对角线元素为0,以排除对角线元素干扰
    n = y.shape[0]
    y[range(n), range(n)] = 0
    pcc_init_score[range(n), range(n)] = 0
    spm_init_score[range(n), range(n)] = 0
    sne_init_score[range(n), range(n)] = 0

    # permutated statistics
    pcc_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    spm_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    sne_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    for i in range(perm_num):
        x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
        pcc_score[i] = np.corrcoef(x)
        spm_score[i] = spearmanr(x, axis=1)[0]
        torch_x = torch.as_tensor(x, dtype=torch.float32)
        if is_use_gpu:
            torch_x = torch_x.cuda()
            y_hat = model(torch_x)[1].cpu()
        else:
            y_hat = model(torch_x)[1]
        sne_score[i] = torch.maximum(
            zeros, -(y_hat / potential_loss_l - 1)**2 + 1).numpy()

        if i % 10 == 0:
            print(f'{i}th computation done')

    # compute p values
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(pcc_init_score),
                   axis=0) / perm_num
    spm_p = np.sum(np.abs(spm_score) > np.abs(spm_init_score),
                   axis=0) / perm_num
    sne_p = np.sum(np.abs(sne_score) > np.abs(sne_init_score),
                   axis=0) / perm_num

    # the elements with p value less than the threshold
    pcc_init_score *= (pcc_p < p_val_thresh)
    spm_init_score *= (spm_p < p_val_thresh)
    sne_init_score *= (sne_p < p_val_thresh)

    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    pcc_cor_matrix = (np.abs(pcc_init_score) > score_thresh).astype(float)
    spm_cor_matrix = (np.abs(spm_init_score) > score_thresh).astype(float)
    sne_cor_matrix = (np.abs(sne_init_score) > score_thresh).astype(float)

    print("计算相关对数……")
    pcc_intr_num = np.count_nonzero(pcc_cor_matrix)
    spm_intr_num = np.count_nonzero(spm_cor_matrix)
    sne_intr_num = np.count_nonzero(sne_cor_matrix)
    ground_truth_intr_num = np.count_nonzero(y)

    print('计算真阳性数……')
    # np.multiply(pcc_cor_matrix, y) 得到对位相乘结果，而 * 不行，离谱
    pcc_true_pos = np.count_nonzero(np.multiply(pcc_cor_matrix, y))
    spm_true_pos = np.count_nonzero(np.multiply(spm_cor_matrix, y))
    sne_true_pos = np.count_nonzero(np.multiply(sne_cor_matrix, y))

    # 计算精确率，此时只需要考虑预测正确，且真的正确的值
    print('计算精确率……')
    pcc_precision = pcc_true_pos / pcc_intr_num
    spm_precision = spm_true_pos / spm_intr_num
    sne_precision = sne_true_pos / sne_intr_num

    # 计算召回率，此时只需要考虑真的正确，且我们预测正确的值
    print('计算召回率……')
    pcc_recall = pcc_true_pos / ground_truth_intr_num
    spm_recall = spm_true_pos / ground_truth_intr_num
    sne_recall = sne_true_pos / ground_truth_intr_num

    print(f'''{"":16}{"PCC":8}{"SPM":8}{"SNE":8}{"Ground Truth":12}
{"Interactions":16}{pcc_intr_num:<8}{spm_intr_num:<8}{sne_intr_num:<8}{ground_truth_intr_num:<12}
{"True Pos":16}{pcc_true_pos:<8}{spm_true_pos:<8}{sne_true_pos:<8}
{"Precision":16}{pcc_precision:<8.2%}{spm_precision:<8.2%}{sne_precision:<8.2%}
{"Recall":16}{pcc_recall:<8.2%}{spm_recall:<8.2%}{sne_recall:<8.2%}''')


def evaluate_all_batches(p_val_thresh,
                         score_thresh,
                         pcc_init_score,
                         spm_init_score,
                         sne_init_score,
                         p_sum_list,
                         perm_num=300,
                         in_path='p_value/scores.pkl'):
    # compute p values
    print(p_sum_list.shape)
    print(type(p_sum_list))
    pcc_p = p_sum_list[0] / perm_num
    spm_p = p_sum_list[1] / perm_num
    sne_p = p_sum_list[2] / perm_num
    print(sne_p.shape)
    print(sne_init_score.shape)
    # the element with p value less than the threshold
    # pcc_ok = np.sum(pcc_p < threshold) / 2
    # spm_ok = np.sum(spm_p < threshold) / 2
    # sne_ok = np.sum(sne_p < threshold) / 2
    # 保留 p-value < 0.1 的元素
    pcc_init_score *= (pcc_p < p_val_thresh)
    spm_init_score *= (spm_p < p_val_thresh)
    sne_init_score *= (sne_p < p_val_thresh)
    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    pcc_cor_matrix = (np.abs(pcc_init_score) > score_thresh).astype(float)
    spm_cor_matrix = (np.abs(spm_init_score) > score_thresh).astype(float)
    sne_cor_matrix = (np.abs(sne_init_score) > score_thresh).astype(float)

    print("计算相关对数……")
    pcc_intr_num = np.count_nonzero(pcc_cor_matrix)
    spm_intr_num = np.count_nonzero(spm_cor_matrix)
    sne_intr_num = np.count_nonzero(sne_cor_matrix)
    ground_truth_intr_num = np.count_nonzero(y)

    print('计算真阳性数……')
    # np.multiply(pcc_cor_matrix, y) 得到对位相乘结果，而 * 不行，离谱
    pcc_true_pos = np.count_nonzero(np.multiply(pcc_cor_matrix, y))
    spm_true_pos = np.count_nonzero(np.multiply(spm_cor_matrix, y))
    sne_true_pos = np.count_nonzero(np.multiply(sne_cor_matrix, y))

    # 计算精确率，此时只需要考虑预测正确，且真的正确的值
    print('计算精确率……')
    pcc_precision = pcc_true_pos / pcc_intr_num
    spm_precision = spm_true_pos / spm_intr_num
    sne_precision = sne_true_pos / sne_intr_num

    # 计算召回率，此时只需要考虑真的正确，且我们预测正确的值
    print('计算召回率……')
    pcc_recall = pcc_true_pos / ground_truth_intr_num
    spm_recall = spm_true_pos / ground_truth_intr_num
    sne_recall = sne_true_pos / ground_truth_intr_num

    print(f'''{"":16}{"PCC":8}{"SPM":8}{"SNE":8}{"Ground Truth":12}
{"Interactions":16}{pcc_intr_num:<8}{spm_intr_num:<8}{sne_intr_num:<8}{ground_truth_intr_num:<12}
{"True Pos":16}{pcc_true_pos:<8}{spm_true_pos:<8}{sne_true_pos:<8}
{"Precision":16}{pcc_precision:<8.2%}{spm_precision:<8.2%}{sne_precision:<8.2%}
{"Recall":16}{pcc_recall:<8.2%}{spm_recall:<8.2%}{sne_recall:<8.2%}''')


def calculate_p_in_batches(x,
                           perm_num,
                           model,
                           p_sum_list,
                           out_path='p_value/scores.pkl'):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated


    '''
    pcc_score = np.zeros((perm_num, x.shape[0], x.shape[0]))
    spm_score = np.zeros((perm_num, x.shape[0], x.shape[0]))
    sne_score = np.zeros((perm_num, x.shape[0], x.shape[0]))
    for i in range(perm_num):
        x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
        pcc_score[i] = np.corrcoef(x)
        spm_score[i] = spearmanr(x, axis=1)[0]
        torch_x = torch.as_tensor(x, dtype=torch.float32)
        if is_use_gpu:
            torch_x = torch_x.cuda()
            sne_score[i] = model(torch_x)[1].detach().cpu().numpy()
        else:
            sne_score[i] = model(torch_x)[1].numpy()
    # with open(out_path, 'a', newline='') as f:
    #     pickle.dump(scores, f)
    # get the sum times when current_score is larger than init_score at this batch
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(pcc_init_score), axis=0)
    spm_p = np.sum(np.abs(spm_score) > np.abs(spm_init_score), axis=0)
    sne_p = np.sum(np.abs(sne_score) > np.abs(sne_init_score), axis=0)
    # get the sum times when current_score is larger than init_score at all batches
    p_sum_list[0] += pcc_p
    p_sum_list[1] += spm_p
    p_sum_list[2] += sne_p
    return p_sum_list


def calculate_init_scores(x, y, model):
    # initial test statistics
    pcc_init_score = np.corrcoef(x)
    spm_init_score = spearmanr(x, axis=1)[0]
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    zeros = torch.zeros(y.shape)
    if is_use_gpu:
        torch_x = torch_x.cuda()
        zeros = zeros.cuda()
        y_hat = model(torch_x)[1].cpu()
    else:
        y_hat = model(torch_x)[1]
    y_hat = y_hat.detach().numpy()
    sne_init_score = np.abs(y_hat)
    # sne_init_score = F.relu(1 - y_hat / contrasive_loss_m).numpy()
    # sne_init_score = torch.maximum(
    #     zeros.cpu(), -(y_hat - potential_loss_l)**2 / 10000 + 1).numpy()
    n = y.shape[0]
    sne_init_score = np.array(sne_init_score)
    # 设置对角线元素为0,以排除对角线元素干扰
    y[range(n), range(n)] = 0
    pcc_init_score[range(n), range(n)] = 0
    spm_init_score[range(n), range(n)] = 0
    sne_init_score[range(n), range(n)] = 0
    return pcc_init_score, spm_init_score, sne_init_score


if __name__ == '__main__':
    # clean_data(
    #     in_smpl_path=raw_smpl_path,
    #     in_intr_path=raw_intr_path,
    #     out_smpl_path=smpl_path,
    #     out_intr_path=intr_path,
    # )

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

    # quick evaluate
    # x_tmp = x[:500].copy()
    # y_tmp = y[:500, :500].copy()
    # evaluate_model(x_tmp,
    #                y_tmp,
    #                model,
    #                perm_num=100,
    #                p_val_thresh=0.2,
    #                score_thresh=0.8)

    # fully evaluate
    pcc_init_score, spm_init_score, sne_init_score = calculate_init_scores(
        x, y, model)
    # 3 means methods that in our comparing experiment
    p_sum_list = np.zeros((3, x.shape[0], x.shape[0]))
    for i in range(10):
        p_sum_list = calculate_p_in_batches(x, 10, model, p_sum_list)
        print(f'{i*10}th computation done')
    with open('p_value/scores.pkl', 'wb') as f:
        pickle.dump(p_sum_list, f)
    with open('p_value/scores.pkl', 'rb') as f:
        p_sum_list = pickle.load(f)
    evaluate_all_batches(0.1, 0.8, pcc_init_score, spm_init_score,
                         sne_init_score, p_sum_list)
