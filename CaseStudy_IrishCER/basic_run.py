import torch
from tqdm import tqdm
import processData
import nets
import sys, os
sys.path.append("..")

import time
import cvxpy as cp

import matplotlib.pyplot as plt

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib

import numpy as np

import OptMiniModule.nn_models import models


desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=10)



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
    gamma = param_set['gamma']
    alpha = param_set['alpha']
    B = param_set['B']
    T = param_set['T']

    G = optMini_util.construct_G_batt_raw(T)
    h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    b = optMini_util.construct_b_batt_raw(T, batt_init=B / 2)
    Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, gamma=gamma)
    q, price = optMini_util.construct_q_batt_raw(T, price=p, batt_B=B, gamma=gamma, alpha=alpha)

    return [Q, q, G, h, A, b, T, price]


def run_battery(dataloader, params=None):
    ## multiple iterations
    # init price
    _default_horizon_ = 48
    torch.manual_seed(2)
    price = torch.rand((_default_horizon_, 1))  # price is a column vector
    Q, q, G, h, A, b, T, price = _form_QP_params(params, p=price)
    # controller = OptPrivModel(Q, q, G, h, A, b, T=T)
    g = models.Generator(z_dim=_default_horizon_, y_priv_dim=2, device=None)
    with tqdm(dataloader) as pbar:
        for k, (D, Y) in enumerate(pbar):
            print(D)
            # controller(D)

            if k > 0:
                raise NotImplementedError("manual break!")


params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, gamma=0.5, alpha=0.2)
run_battery(dataloader_dict['train'], params=params)

