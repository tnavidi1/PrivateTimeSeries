import torch
import torch.nn.functional as F
from tqdm import tqdm
import processData
import nets
import numpy as np
import logging
import argparse
import os
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('paper', style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 2.5}, )


import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib
import basic_util as bUtil

# ==== printing config ==== #
desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150
torch.set_printoptions(profile="full", linewidth=400)

dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=1, bsz=128, split=0.15)


def run_train(dataloader, params, p_opt='TOU', seed=1):

    _default_horizon_ = 48
    torch.manual_seed(seed)

    price = None
    if p_opt == 'TOU':
        price = bUtil.create_price()
    elif p_opt == 'LMP':
        price = bUtil.create_LMP('../Data_LMP/8-18-2019_8days_Menlo_LMPs.csv')
    else:
        price = torch.rand((_default_horizon_, 1))  # price is a column vector

    # print(price)
    # raise NotImplementedError()

    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price)



"""
    "learning_rate": 1e-3,
    "batch_size": 64,
    "iter_max": 1002,
    "iter_save": 50,
    "num_workers": 10,
    "tradeoff_beta1": 3,
    "tradeoff_beta2": 2,
    "c_i": 0.3,
    "c_o": 0.3,
    "eta_eff": 0.95,
    "T":48,
    "B": 0.4,
    "beta1": 0.06,
    "beta2": 0.04,
    "beta3": 0.05,
    "alpha": 0.2,
    "xi": 1
"""

# ========= start run ========= # 


params = dict(learning_rate=1e-3, batch_size=64,
              iter_max=1002, iter_save=50, num_workers=10,
              tradeoff_beta1 = 3, tradeoff_beta2 = 2,
              c_i=0.99, c_o=0.99, eta_eff=0.97,
              T=24, B=1.5,
              beta1=0.6, beta2=0.4, beta3=0.5,
              alpha=0.2)



run_train(dataloader_dict['train'], params=params, p_opt='TOU', seed=1)

