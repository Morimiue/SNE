import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix

from model_gmlp import GMLPModel as Model
from utils import *

model_path = './models/gmlp_model.pth'

is_using_gpu = torch.cuda.is_available()

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
