import torch
from tqdm import tqdm
import processData
import nets
import numpy as np
import sys, os
sys.path.append("..")

import time
import cvxpy as cp

import matplotlib.pyplot as plt

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib
import basic_util as bUtil


desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=32)



def _form_QP_params(param_set, p=None):
    """

    :param param_set:
    :param p: price
    :return:
    """
    if not isinstance(param_set, dict):
        raise NotImplementedError("wrong type of param set: {}".format( param_set))

    c_i = param_set['c_i']
    c_o = param_set['c_o']
    eta_eff = param_set['eta_eff']
    beta1 = param_set['beta1']
    beta2 = param_set['beta2']
    beta3 = param_set['beta3']
    alpha = param_set['alpha']
    B = param_set['B']
    T = param_set['T']

    G = optMini_util.construct_G_batt_raw(T)
    h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    b = optMini_util.construct_b_batt_raw(T, batt_init=B / 2)
    Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, beta3=beta3)
    q, price = optMini_util.construct_q_batt_raw(T, price=p, batt_B=B, beta3=beta3, alpha=alpha)

    return [Q, q, G, h, A, b, T, price]

def _extract_filter_weight(x):
    return optMini_util.to_np(x.data)



def run_battery(dataloader, params=None):
    ## multiple iterations
    # init price
    _default_horizon_ = 48
    torch.manual_seed(2)
    price = torch.rand((_default_horizon_, 1))  # price is a column vector

    Q, q, G, h, A, b, T, price = _form_QP_params(params, p=price)
    # controller = OptPrivModel(Q, q, G, h, A, b, T=T)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h = h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None)
    # print(g)
    # raise NotImplementedError
    optimizer = torch.optim.Adam(g.filter.parameters(), lr=2*1e-3)
    # raise NotImplementedError(*g.filter.parameters())
    with tqdm(dataloader) as pbar:
        for k, (D, Y) in enumerate(pbar):
            # controller(D)
            optimizer.zero_grad()
            y_labels = bUtil.convert_binary_label(Y, 1500) # row vector
            y_onehot = bUtil.convert_onehot(y_labels.unsqueeze(1), alphabet_size=2)
            # print(D, y_labels, y_onehot)
            # D_tilde, z_noise = g.(D, y_onehot)
            loss = g.util_loss(D, y_onehot, xi=1)
            # print(loss)
            if k % 20 == 0:
                print(g.filter.fc.weight.data.shape)
                print(g.filter.fc.weight.data)
                print(torch.trace(torch.mm(g.filter.fc.weight.data, g.filter.fc.weight.data.t())))

            loss.backward()
            optimizer.step()

            # print(g.filter.fc.weight.shape) # 48 * 50
            # d = D_tilde[0]
            # eps = z_noise[0]
            # y_onehot_ = y_onehot[0]
            # # GAMMA = _extract_filter_weight(g.filter.fc.weight)
            #
            # GAMMA = g.filter.fc.weight
            #
            # # raise NotImplementedError("=========")
            # [price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b] = list(map(_extract_filter_weight,
            #                                                             [price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b]))
            # # ==== cvx ====
            # # x_ctrl = optMini_cvx._convex_formulation_w_GAMMA_d_cvx(price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b, T,
            # #                                                        sol_opt=cp.GUROBI, verbose=True)
            # #
            # # # print(x_ctrl[:T] - x_ctrl[T:(2 * T)])
            # # fig, ax =plt.subplots(2, 1, figsize=(6, 4))
            # # ax[0].bar(np.arange(1, T + 1), x_ctrl[:T] - x_ctrl[T:(2 * T)])
            # # ax[0].bar(np.arange(1, T + 1), price.flatten())
            # # plt.show()
            # # ==== conic ====
            # x_ctrl, db_ = optMini_cvx._convex_formulation_w_GAMMA_d_conic(price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b, T,
            #                                                          sol_opt=cp.GUROBI, verbose=True)
            # print(np.expand_dims(db_[T:(2*T)], 1), eps)
            #
            # print(np.tile(np.expand_dims(db_[T:(2*T)], 1), T))
            # grad_gamma = ((1/T) * (np.tile(np.expand_dims(db_[T:(2*T)], 1), T)) / eps )
            # print(grad_gamma.shape, grad_gamma)
            # # print(x_ctrl[:T] - x_ctrl[T:(2*T)])
            # # plt.figure(figsize=(6, 4))
            # # ax[1].bar(np.arange(1, T+1), x_ctrl[:T] - x_ctrl[T:(2*T)])
            # # ax[1].bar(np.arange(1, T+1), price.flatten())
            # # plt.show()



            # print(x_ctrl)
            if (k + 1) > 400:
                raise NotImplementedError("manual break!")


params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, beta3=0.5, alpha=0.2)
run_battery(dataloader_dict['train'], params=params)

