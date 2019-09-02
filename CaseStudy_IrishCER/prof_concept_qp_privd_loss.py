import argparse
import sys
sys.path.append("..")

import numpy as np
import numpy.random as npr

# from qpth.qp import QPFunction
# import qpth.solvers.pdipm.single as pdipm_s
# import qpth.solvers.pdipm.batch as pdipm_b


import time

import torch

import OptMiniModule.qp_w_d as optMini_qpfunc
import OptMiniModule.util as optMini_util
import basic_util as bUtil




def run_func_test(params):
    _default_horizon_ = params['T'] if params is not None else 48
    torch.manual_seed(2)
    # price = _create_price(steps_perHr=2) + torch.rand((_default_horizon_, 1))* 0.05  # price is a column vector
    price = torch.rand((_default_horizon_, 1))
    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price)

    qpobjfunc = optMini_qpfunc.QP_privD(T)
    d = torch.rand(T)
    epsilon = torch.rand(T)
    y_onehot_ = torch.Tensor([0, 1])
    price_ = price
    # price_ = optMini_util.to_np(price)
    GAMMA_ = torch.randn((T,T+2))
    Q_ = Q
    x = qpobjfunc(price_, GAMMA_, d, epsilon, y_onehot_, Q_, G, h, A, b)
    qpobjfunc.backward(T)

params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=4, B=1.5, beta1=0.6, beta2=0.4, beta3=0.5, alpha=0.2)

run_func_test(params)


