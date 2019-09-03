import torch
import numpy as np
import sys
sys.path.append('..')

import OptMiniModule.util as optMini_util

def convert_onehot(y_label, alphabet_size=6):
    y_ = y_label.long()
    one_hot = torch.FloatTensor(y_.size(0), alphabet_size).zero_()
    y_oh = one_hot.scatter_(1, (y_.data), 1) # if y is from 1 to 6
    return y_oh

def convert_binary_label(y_label, median=4):
    y_ = y_label.squeeze()
    y_.apply_(lambda x: 1 if x >=median else 0)
    return y_.long()




def create_price(steps_perHr=2):
    HORIZON = 24
    T1 = 16
    T2 = T1 + 5
    T3 = HORIZON
    rate_offpeak = 0.202
    rate_onpeak = 0.463
    price_shape = np.hstack((rate_offpeak * np.ones((1, T1 * steps_perHr)),
                             rate_onpeak * np.ones((1, (T2-T1) * steps_perHr)),
                             rate_offpeak * np.ones((1, (T3-T2) * steps_perHr ))))
    p = torch.from_numpy(price_shape).to(torch.float).reshape(-1, 1)
    return p



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


def _convert_to_np_arr(X, j):
    """
    if X is 2d array
    :param X:
    :param j:
    :return:
    """
    xs = np.array([x_[j].tolist() for x_ in X])
    return xs

def _convert_to_np_scalars(X, j):
    """
    if X is a 1d array
    :param X:
    :param j:
    :return:
    """
    xs = np.array([x_[j] for x_ in X])
    return xs
