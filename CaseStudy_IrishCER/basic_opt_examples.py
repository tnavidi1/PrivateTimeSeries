import torch
from tqdm import tqdm
import processData
import nets
import sys
sys.path.append("..")
import OptMiniModule.util as optMini_util

import numpy as np
desired_width = 300
np.set_printoptions(linewidth=desired_width)

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=100)



def check(dataloader):
    ## multiple iterations
    T = 24  # time horizon : 24 hours
    G = optMini_util.construct_G_batt_raw(T)
    h = optMini_util.construct_h_batt_raw(T)
    A = optMini_util.construct_A_batt_raw(T)
    b = optMini_util.construct_b_batt_raw(T)
    print(G)
    print(G.shape)
    print(h)
    print(h.shape)
    print(A)
    print(A.shape)
    print(b)
    print(b.shape)

    ################################
    raise NotImplementedError
    with tqdm(dataloader) as pbar:
        for k, (X, Y) in enumerate(pbar):
            print(k, X, optMini_util.convert_binary_label(Y, 1500.0))

            if k > 8:
                raise NotImplementedError



check(dataloader_dict['train'])