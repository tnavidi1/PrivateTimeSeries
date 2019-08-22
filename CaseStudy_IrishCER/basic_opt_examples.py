import torch
from tqdm import tqdm
import processData
import nets
import sys
sys.path.append("..")

import cvxpy as cp

import matplotlib.pyplot as plt

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx

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
    B = 1.5
    G = optMini_util.construct_G_batt_raw(T)
    h = optMini_util.construct_h_batt_raw(T, c_i=1, c_o=1, batt_B=B)
    A = optMini_util.construct_A_batt_raw(T, eta=0.95)
    b = optMini_util.construct_b_batt_raw(T, batt_init=B/2)
    Q = optMini_util.construct_Q_batt_raw(T, beta1=0.5, beta2=0.5, gamma=0.5)
    q, price = optMini_util.construct_q_batt_raw(T, price=None, batt_B=B, gamma=0.5, alpha=0.2)

    # print(price)
    # # print(G)
    # print("G shape ", G.shape)
    # # print(h)
    # print("h shape ", h.shape)
    # # print(A)
    # print(A.shape)
    # # print(b)
    # print(b.shape)
    # # print(Q)
    # print(Q.shape)
    # # print(q)
    # print(q.shape)
    Q = optMini_util.to_np(Q)
    q = optMini_util.to_np(q)
    G = optMini_util.to_np(G)
    h = optMini_util.to_np(h)
    A = optMini_util.to_np(A)
    b = optMini_util.to_np(b)
    price = optMini_util.to_np(price.squeeze(1))

    ################################
    # solving the optimization by cvx
    obj, x_sol, nu, lam, slacks = optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.GUROBI, verbose=True) # gurobi
    # optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=True)  # cvxopt
    # print(obj, x_sol, nu, lam, slacks)
    ################################
    # check battery status
    print(np.round(x_sol[:T]*0.95 - x_sol[T:(T+T)] + x_sol[2*T:], 3))
    print(np.round(x_sol[2*T:], 3))
    ################################
    # # plot figure
    # plt.figure(figsize=(6, 4))
    # plt.bar(np.arange(1, 25)-0.2, x_sol[:24] - x_sol[24:48], width=0.4, label='Net charging')
    # plt.bar(np.arange(1, 25)+0.2, price, width=0.4, label='Price')
    # plt.legend(fontsize=15)
    # plt.xlabel('Hour of Day', fontsize=18)
    # plt.tick_params(labelsize=16)
    # plt.tight_layout()
    # plt.savefig('../fig/basic_charging_plot.png')
    # # plt.show()
    ################################
    raise NotImplementedError
    with tqdm(dataloader) as pbar:
        for k, (X, Y) in enumerate(pbar):


            print(k, X, optMini_util.convert_binary_label(Y, 1500.0))

            if k > 8:
                raise NotImplementedError



check(dataloader_dict['train'])