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


def run_train(dataloader, params, p_opt='TOU', iter_max=500, lr=1e-3, xi=0.5, n_job=5, seed=1):

    _default_horizon_ = 24
    torch.manual_seed(seed)

    price = None
    if p_opt == 'TOU':
        price = bUtil.create_price(steps_perHr=1)
    elif p_opt == 'LMP':
        price = bUtil.create_LMP('../Data_LMP/8-18-2019_8days_Menlo_LMPs.csv', granular=24, steps_perHr=1)
    else:
        price = torch.rand((_default_horizon_, 1))  # price is a column vector

    # print(price)
    # raise NotImplementedError()

    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price)

    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h=h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None, n_job=n_job)

    clf = nets.Classifier(z_dim=24, y_dim=1)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=lr, betas=(0.6, 0.999))
    j = 0
    p_ = 0.5
    prior_pi = torch.Tensor([[p_],[1-p_]]) # shape : [2,1]
    with tqdm(total=iter_max) as pbar:
        # with tqdm(dataloader) as pbar:
        while True:
            correct_cnt = 0
            tot_cnt = 0
            label_cnt1 = 0
            label_cnt2 = 0

            # raise NotImplementedError(len(dataloader))
            for k, (D, Y) in enumerate(dataloader):
                j += 1
                # print(D, Y)
                # y_labels = (Y+1).squeeze().long()
                optimizer_g.zero_grad()
                optimizer_clf.zero_grad()

                y_labels = Y
                y_onehot = bUtil.convert_onehot_soft(y_labels, alphabet_size=2)
                D_tilde, z_noise = g.forward(D, y_onehot)
                # y_out = clf(D_tilde)
                clf_loss = - (F.logsigmoid(clf(D_tilde))).mean()

                print(clf_loss)

                loss_util_batch, util_grad, distort_ = g.util_loss(D, D_tilde, z_noise,
                                                                    y_onehot, prior=prior_pi)

                # print(loss_util_batch)
                print(util_grad.shape)
                # print(distort_ - xi)


                raise NotImplementedError
                if k > 0 :
                    break

            if j >= iter_max:
                return


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



run_train(dataloader_dict['train'], params=params, p_opt='TOU', iter_max=2, lr=1e-3, n_job=4, seed=1)

