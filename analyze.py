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


@torch.no_grad()
def evaluate_model_only(x, y, model, score_thresh):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
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
    # np.fill_diagonal(y, 0.)
    # np.fill_diagonal(sne_init_score, 0.)

    save_dense_to_interactions(sne_init_score,
                               './data/predicted/interactions.csv')

    # use upper triangular matrix to calculate
    y = get_triu_items(y)
    sne_init_score = get_triu_items(sne_init_score)

    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    sne_cor_matrix = (np.abs(sne_init_score) > score_thresh).astype(float)

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
def evaluate_model(x, y, model, perm_num, p_val_thresh, score_thresh):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    # initial test statistics
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    zeros = torch.zeros(y.shape)
    if is_use_gpu:
        torch_x = torch_x.cuda()
        y_hat = model(torch_x)[1].cpu()
    else:
        y_hat = model(torch_x)[1]
    sne_init_score = torch.maximum(zeros, -(y_hat / potential_loss_l - 1)**2 +
                                   1).numpy()
    pcc_init_score = np.corrcoef(x)
    spm_init_score = spearmanr(x, axis=1)[0]
    lsa_init_score = lsa(x, 3)
    mic_init_score, _ = cstats(x, x, alpha=0.6, c=15, est="mic_e")

    # 设置对角线元素为0,以排除对角线元素干扰
    n = y.shape[0]
    y[range(n), range(n)] = 0
    sne_init_score[range(n), range(n)] = 0
    pcc_init_score[range(n), range(n)] = 0
    spm_init_score[range(n), range(n)] = 0
    lsa_init_score[range(n), range(n)] = 0
    mic_init_score[range(n), range(n)] = 0

    # permutated statistics
    sne_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    pcc_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    spm_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    lsa_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    mic_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    for i in range(perm_num):
        x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
        torch_x = torch.as_tensor(x, dtype=torch.float32)
        if is_use_gpu:
            torch_x = torch_x.cuda()
            y_hat = model(torch_x)[1].cpu()
        else:
            y_hat = model(torch_x)[1]
        sne_score[i] = torch.maximum(
            zeros, -(y_hat / potential_loss_l - 1)**2 + 1).numpy()
        pcc_score[i] = np.corrcoef(x)
        spm_score[i] = spearmanr(x, axis=1)[0]
        lsa_score[i] = lsa(x, 3)
        mic_score[i] = np.corrcoef(x)
        # mic_score[i], _ = cstats(x, x, alpha=0.6, c=15, est="mic_e")
        sne_score[i][range(x.shape[0]), range(x.shape[0])] = 0
        pcc_score[i][range(x.shape[0]), range(x.shape[0])] = 0
        spm_score[i][range(x.shape[0]), range(x.shape[0])] = 0
        lsa_score[i][range(x.shape[0]), range(x.shape[0])] = 0
        mic_score[i][range(x.shape[0]), range(x.shape[0])] = 0

        if i % 10 == 0:
            print(f'{i}th computation done')

    # compute p values
    sne_p = np.sum(np.abs(sne_score) > np.abs(sne_init_score),
                   axis=0) / perm_num
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(pcc_init_score),
                   axis=0) / perm_num
    spm_p = np.sum(np.abs(spm_score) > np.abs(spm_init_score),
                   axis=0) / perm_num
    lsa_p = np.sum(np.abs(lsa_score) > np.abs(lsa_init_score),
                   axis=0) / perm_num
    mic_p = np.sum(np.abs(mic_score) > np.abs(mic_init_score),
                   axis=0) / perm_num

    # the elements with p value less than the threshold
    sne_init_score *= (sne_p < p_val_thresh)
    pcc_init_score *= (pcc_p < p_val_thresh)
    spm_init_score *= (spm_p < p_val_thresh)
    lsa_init_score *= (lsa_p < p_val_thresh)
    mic_init_score *= (mic_p < p_val_thresh)

    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    sne_cor_matrix = (np.abs(sne_init_score) > score_thresh).astype(float)
    pcc_cor_matrix = (np.abs(pcc_init_score) > score_thresh).astype(float)
    spm_cor_matrix = (np.abs(spm_init_score) > score_thresh).astype(float)
    lsa_cor_matrix = (np.abs(lsa_init_score) > score_thresh).astype(float)
    mic_cor_matrix = (np.abs(mic_init_score) > score_thresh).astype(float)

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


def evaluate_all_batches(p_val_thresh,
                         score_thresh,
                         init_score_list,
                         p_sum_list,
                         perm_num=300,
                         in_path='p_value/scores.pkl'):
    # compute p values
    p_list = p_sum_list / perm_num
    # the element with p value less than the threshold
    init_score_list *= (p_list < p_val_thresh)
    # 判断是否相关，大于阈值则视为相关，得到相关矩阵
    cor_matrix_list = (np.abs(init_score_list) > score_thresh).astype(float)
    ground_truth_intr_num = np.count_nonzero(y)
    intr_num_list = np.zeros((5, 1))
    for i in range(5):
        intr_num_list[i] = np.count_nonzero(cor_matrix_list[i])
    print('计算真阳性数……')
    # np.multiply(pcc_cor_matrix, y) 得到对位相乘结果，而 * 不行，离谱
    true_pos_list = np.zeros((5, 1))
    for i in range(5):
        true_pos_list[i] = np.count_nonzero(np.multiply(cor_matrix_list[i], y))

    # 计算精确率，此时只需要考虑预测正确，且真的正确的值
    print('计算精确率……')
    precision_list = np.zeros((5, 1))
    for i in range(5):
        precision_list[i] = true_pos_list[i] / intr_num_list[i]
    # 计算召回率，此时只需要考虑真的正确，且我们预测正确的值
    print('计算召回率……')
    recall_list = np.zeros((5, 1))
    for i in range(5):
        recall_list[i] = true_pos_list[i] / ground_truth_intr_num
    intr_num_list = intr_num_list.astype(float)
    print(
        f'''{"":16}{"PCC":8}{"SPM":8}{"SNE":8}{"LSA":8}{"MIC":8}{"Ground Truth":12}
{"Interactions":16}{intr_num_list[0][0]:<8}{intr_num_list[1][0]:<8}{intr_num_list[2][0]:<8}{intr_num_list[3][0]:<8}{intr_num_list[4][0]:<8}{ground_truth_intr_num:<12}
{"True Pos":16}{true_pos_list[0][0]:<8}{true_pos_list[1][0]:<8}{true_pos_list[2][0]:<8}{true_pos_list[3][0]:<8}{true_pos_list[4][0]:<8}
{"Precision":16}{precision_list[0][0]:<8.2%}{precision_list[1][0]:<8.2%}{precision_list[2][0]:<8.2%}{precision_list[3][0]:<8.2%}{precision_list[4][0]:<8.2%}
{"Recall":16}{recall_list[0][0]:<8.2%}{recall_list[1][0]:<8.2%}{recall_list[2][0]:<8.2%}{recall_list[3][0]:<8.2%}{recall_list[4][0]:<8.2%}'''
    )


def calculate_p_in_batches(x,
                           perm_num,
                           model,
                           init_score_list,
                           p_sum_list,
                           out_path='p_value/scores.pkl'):
    '''
    if the p-value is less than the threshold we reject the null-hypothesis,
    and accept that our judgement is true i.e. these series are correlated
    '''
    sne_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    pcc_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    spm_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    lsa_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    mic_score = np.empty((perm_num, x.shape[0], x.shape[0]))
    zeros = torch.zeros(y.shape)
    for i in range(perm_num):
        x = np.apply_along_axis(np.random.permutation, axis=1, arr=x)
        torch_x = torch.as_tensor(x, dtype=torch.float32)
        if is_use_gpu:
            torch_x = torch_x.cuda()
            y_hat = model(torch_x)[1].cpu()
        else:
            y_hat = model(torch_x)[1]
        sne_score[i] = torch.maximum(
            zeros, -(y_hat / potential_loss_l - 1)**2 + 1).detach().numpy()
        pcc_score[i] = np.corrcoef(x)
        spm_score[i] = spearmanr(x, axis=1)[0]
        # lsa_score = lsa(x, 3)
        lsa_score[i] = np.corrcoef(x)
        # mic_score = cstats(x, x, alpha=0.6, c=15, est="mic_e")
        mic_score[i] = np.corrcoef(x)
    # get the sum times when current_score is larger than init_score at this batch
    pcc_p = np.sum(np.abs(pcc_score) > np.abs(init_score_list[0]), axis=0)
    spm_p = np.sum(np.abs(spm_score) > np.abs(init_score_list[1]), axis=0)
    sne_p = np.sum(np.abs(sne_score) > np.abs(init_score_list[2]), axis=0)
    lsa_p = np.sum(np.abs(lsa_score) > np.abs(init_score_list[3]), axis=0)
    mic_p = np.sum(np.abs(mic_score) > np.abs(init_score_list[4]), axis=0)
    pcc_p[range(x.shape[0]), range(x.shape[0])] = 0
    spm_p[range(x.shape[0]), range(x.shape[0])] = 0
    sne_p[range(x.shape[0]), range(x.shape[0])] = 0
    lsa_p[range(x.shape[0]), range(x.shape[0])] = 0
    mic_p[range(x.shape[0]), range(x.shape[0])] = 0

    # get the sum times when current_score is larger than init_score at all batches
    p_sum_list[0] += pcc_p
    p_sum_list[1] += spm_p
    p_sum_list[2] += sne_p
    p_sum_list[3] += lsa_p
    p_sum_list[4] += mic_p
    return p_sum_list


def calculate_init_scores(x, y, model):
    # initial test statistics
    pcc_init_score = np.corrcoef(x)
    spm_init_score = spearmanr(x, axis=1)[0]
    # lsa_init_score = lsa(x, 3)
    lsa_init_score = np.corrcoef(x)
    # mic_init_score, _ = cstats(x, x, alpha=0.6, c=15, est="mic_e")
    mic_init_score = np.corrcoef(x)
    torch_x = torch.as_tensor(x, dtype=torch.float32)
    zeros = torch.zeros(y.shape)
    if is_use_gpu:
        torch_x = torch_x.cuda()
        y_hat = model(torch_x)[1].cpu()
    else:
        y_hat = model(torch_x)[1]
    sne_init_score = torch.maximum(zeros, -(y_hat / potential_loss_l - 1)**2 +
                                   1).detach().numpy()
    n = y.shape[0]
    sne_init_score = np.array(sne_init_score)
    # 设置对角线元素为0,以排除对角线元素干扰
    y[range(n), range(n)] = 0
    pcc_init_score[range(n), range(n)] = 0
    spm_init_score[range(n), range(n)] = 0
    sne_init_score[range(n), range(n)] = 0
    lsa_init_score[range(n), range(n)] = 0
    mic_init_score[range(n), range(n)] = 0
    init_score_list = np.stack((pcc_init_score, spm_init_score, sne_init_score,
                                lsa_init_score, mic_init_score),
                               axis=0)
    return init_score_list


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

    # ---------- super quick evaluate ----------
    evaluate_model_only(x, y, model, score_thresh=0.)

    # ---------- quick evaluate ----------
    # x_tmp = x[:300].copy()
    # y_tmp = y[:300, :300].copy()
    # evaluate_model(x_tmp,
    #                y_tmp,
    #                model,
    #                perm_num=100,
    #                p_val_thresh=0.1,
    #                score_thresh=0.8)

    # ---------- fully evaluate ----------
    # init_score_list = calculate_init_scores(x, y, model)
    # # 5 means methods that in our comparing experiment
    # p_sum_list = np.zeros((5, x.shape[0], x.shape[0]))
    # batch_of_perm = 10
    # batchsize_of_perm = 3
    # for i in range(batch_of_perm):
    #     p_sum_list = calculate_p_in_batches(x, batchsize_of_perm, model,
    #                                         init_score_list, p_sum_list)
    #     print(f'{i*batch_of_perm}th computation done')
    # # with open('p_value/scores.pkl', 'wb') as f:
    # #     pickle.dump(p_sum_list, f)
    # # with open('p_value/scores.pkl', 'rb') as f:
    # #     p_sum_list = pickle.load(f)
    # evaluate_all_batches(0.1, 0.8, init_score_list, p_sum_list,
    #                      batch_of_perm * batchsize_of_perm)
