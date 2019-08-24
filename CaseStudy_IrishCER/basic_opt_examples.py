import torch
from tqdm import tqdm
import processData
import nets
import sys, os
sys.path.append("..")

import cvxpy as cp

import matplotlib.pyplot as plt

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib

import numpy as np
desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=24)


def _debug_check_verbose_sol_byCVX(x_sol, T):

    print("=" * 100)
    print(np.expand_dims(x_sol.round(4), 1))
    print("=" * 100)
    print(np.round(x_sol[:T]*0.95 - x_sol[T:(T+T)] + x_sol[2*T:], 3))
    print(np.round(x_sol[2*T:], 3))
    print("=" * 100)


def _debug_check_verbose_sol_byDIFFCP(x_sol, T):
    print("=" * 100)
    print(np.expand_dims(x_sol[:(3*T)].round(4), 1))
    print("=" * 100)



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



def check_basic(param_set=None, p=None, plotfig=False, debug=False, cp_solver=cp.CVXOPT):
    # if not isinstance(param_set, dict):
    #     raise NotImplementedError("wrong type of param set: {}".format( param_set))

    # c_i = param_set['c_i']
    # c_o = param_set['c_o']
    # eta_eff = param_set['eta_eff']
    # beta1 = param_set['beta1']
    # beta2 = param_set['beta2']
    # gamma = param_set['gamma']
    # alpha = param_set['alpha']
    # B = param_set['B']
    # T = param_set['T']
    #
    # G = optMini_util.construct_G_batt_raw(T)
    # h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    # A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    # b = optMini_util.construct_b_batt_raw(T, batt_init=B/2)
    # Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, gamma=gamma)
    # q, price = optMini_util.construct_q_batt_raw(T, price=None, batt_B=B, gamma=gamma, alpha=alpha)
    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)

    Q = optMini_util.to_np(Q)
    q = optMini_util.to_np(q)
    G = optMini_util.to_np(G)
    h = optMini_util.to_np(h)
    A = optMini_util.to_np(A)
    b = optMini_util.to_np(b)
    price = optMini_util.to_np(price.squeeze(1))

    ################################
    # solving the optimization by cvx
    obj, x_sol, lam, mu, slacks = optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp_solver, verbose=True) # gurobi
    # optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=True)  # cvxopt
    # print(obj, x_sol, nu, lam, slacks)
    ################################
    # check battery status
    if debug:
        _debug_check_verbose_sol_byCVX(x_sol, T)
    ################################
    # # plot figure
    if plotfig is True:
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(1, T+1)-0.2, x_sol[:T] - x_sol[T:2*T], width=0.4, label='Net charging')
        plt.bar(np.arange(1, T+1)+0.2, price, width=0.4, label='Price')
        plt.legend(fontsize=15)
        plt.title("Battery Control without Demand")
        plt.xlabel('Time steps (30min interval)', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.ylim([-1.1, 1.1])
        plt.tight_layout()
        plt.savefig('../fig/Batt_basic_charging_plot_cvx.png')
        plt.close('all')
    # # plt.show()


def construct_QP_battery_w_D_cvx(param_set=None, d=None, p=None, plotfig=False, cp_solver=cp.CVXOPT):
    """
    This method pass over cvx solving the canonical QP
    :param param_set:
    :param d: demand
    :return:
    """
    # c_i = param_set['c_i']
    # c_o = param_set['c_o']
    # eta_eff = param_set['eta_eff']
    # beta1 = param_set['beta1']
    # beta2 = param_set['beta2']
    # gamma = param_set['gamma']
    # alpha = param_set['alpha']
    # B = param_set['B']
    # T = param_set['T']
    #
    # G = optMini_util.construct_G_batt_raw(T)
    # h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    # A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    # b = optMini_util.construct_b_batt_raw(T, batt_init=B / 2)
    # Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, gamma=gamma)
    # q, price = optMini_util.construct_q_batt_raw(T, price=p, batt_B=B, gamma=gamma, alpha=alpha)

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)


    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)
    # print(h.shape, d.shape)

    h = torch.cat([h, d.view(T, 1)], dim=0)

    Q = optMini_util.to_np(Q)
    q = optMini_util.to_np(q)
    G = optMini_util.to_np(G)
    h = optMini_util.to_np(h)
    A = optMini_util.to_np(A)
    b = optMini_util.to_np(b)

    obj, x_sol, lam, mu, slacks = optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp_solver, verbose=True)  # gurobi
    # optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=True)  # cvxopt
    # print(obj, x_sol, nu, lam, slacks)
    ################################
    # check battery status
    # print(np.round(x_sol[:T] * 0.95 - x_sol[T:(T + T)] + x_sol[2 * T:], 3))
    # print(np.round(x_sol[2 * T:], 3))

    if plotfig is True:
        price = optMini_util.to_np(price.squeeze(1))  # price is already embedded in q, here we just convert it for plotting
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(1, T+1)-0.2, x_sol[:T] - x_sol[T:2*T], width=0.4, label='Net charging')
        plt.bar(np.arange(1, T+1)+0.2, price, width=0.4, label='Price')
        plt.legend(fontsize=15)
        plt.title("Battery Control with Demand")
        plt.xlabel('Time steps (30min interval)', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.ylim([-1.1, 1.1])
        plt.tight_layout()
        plt.savefig('../fig/Batt_with_demand_charging_plot_cvx.png')
        plt.close('all')


def check_basic_csc(param_set=None, p=None, plotfig=False, debug=False):
    # if not isinstance(param_set, dict):
    #     raise NotImplementedError("wrong type of param set: {}".format( param_set))

    # c_i = param_set['c_i']
    # c_o = param_set['c_o']
    # eta_eff = param_set['eta_eff']
    # beta1 = param_set['beta1']
    # beta2 = param_set['beta2']
    # gamma = param_set['gamma']
    # alpha = param_set['alpha']
    # B = param_set['B']
    # T = param_set['T']
    #
    # G = optMini_util.construct_G_batt_raw(T)
    # h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    # A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    # b = optMini_util.construct_b_batt_raw(T, batt_init=B/2)
    # Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, gamma=gamma)
    # q, price = optMini_util.construct_q_batt_raw(T, price=None, batt_B=B, gamma=gamma, alpha=alpha)

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)

    Q = optMini_util.to_np(Q)
    q = optMini_util.to_np(q)
    G = optMini_util.to_np(G)
    h = optMini_util.to_np(h)
    A = optMini_util.to_np(A)
    b = optMini_util.to_np(b)



    ################################
    # solving the optimization by cvx
    # obj, x_sol, nu, lam, slacks = optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.GUROBI, verbose=True) # gurobi
    # optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=True)  # cvxopt
    # print(obj, x_sol, nu, lam, slacks)
    ################################
    # check battery status
    # print(np.round(x_sol[:T]*0.95 - x_sol[T:(T+T)] + x_sol[2*T:], 3))
    # print(np.round(x_sol[2*T:], 3))
    # ################################
    #
    # ###### formulate & solve problem #######
    x_sol, y, s, D, DT, A_, b_, c_ = optMini_cvx.cvx_format_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=True)

    ##################################
    if debug:
        _debug_check_verbose_sol_byDIFFCP(x_sol, T)

    # dx, dy, ds = D(dA, db, dc)
    # print("size of y and z ", y.size, s.size)

    dA, db, dc = DT(c_, np.zeros(y.size), np.zeros(s.size), atol=1e-5, btol=1e-5)
    # print(dA, db, dc)

    # # plot figure
    if plotfig is True:
        price = optMini_util.to_np(price.squeeze(1))
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(1, T+1)-0.2, x_sol[:T] - x_sol[T:2*T], width=0.4, label='Net charging')
        plt.bar(np.arange(1, T+1)+0.2, price, width=0.4, label='Price')
        plt.legend(fontsize=15)
        plt.title("Battery Control without Demand")
        plt.xlabel('Time steps (30min interval)', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.ylim([-1.1, 1.1])
        plt.tight_layout()
        plt.savefig('../fig/Batt_basic_charging_plot_conic.png')
        plt.close('all')
    # # plt.show()
    ################################


def construct_QP_battery_w_D_conic(param_set=None, d=None, p=None, plotfig=False, debug=False):
    """
    The function serves to convert a quadratic programming problem into conic programming
    :param param_set:
    :param d:
    :param p:
    :param plotfig:
    :param debug:
    :return:
    """
    # if not isinstance(param_set, dict):
    #     raise NotImplementedError("wrong type of param set: {}".format( param_set))
    #
    # c_i = param_set['c_i']
    # c_o = param_set['c_o']
    # eta_eff = param_set['eta_eff']
    # beta1 = param_set['beta1']
    # beta2 = param_set['beta2']
    # gamma = param_set['gamma']
    # alpha = param_set['alpha']
    # B = param_set['B']
    # T = param_set['T']
    #
    # G = optMini_util.construct_G_batt_raw(T)
    # h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    # A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    # b = optMini_util.construct_b_batt_raw(T, batt_init=B / 2)
    # Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, gamma=gamma)
    # q, price = optMini_util.construct_q_batt_raw(T, price=p, batt_B=B, gamma=gamma, alpha=alpha)

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)

    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)
    # print(h.shape, d.shape)

    h = torch.cat([h, d.view(T, 1)], dim=0) # demand d is from data input

    Q = optMini_util.to_np(Q)
    q = optMini_util.to_np(q)
    G = optMini_util.to_np(G)
    h = optMini_util.to_np(h)
    A = optMini_util.to_np(A)
    b = optMini_util.to_np(b)

    # ###### formulate & solve problem #######
    x_sol, y, s, D, DT, A_, b_, c_ = optMini_cvx.cvx_format_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=True)

    if debug:
        _debug_check_verbose_sol_byDIFFCP(x_sol, T)

    # dx, dy, ds = D(dA, db, dc)
    # print("size of y and z ", y.size, s.size)
    # -------- calculate differential using adjoint operator ------
    # more details: https://en.wikipedia.org/wiki/Differential_operator#Adjoint_of_an_operator
    # -------------------------------------------------------------
    dA, db, dc = DT(c_, np.zeros(y.size), np.zeros(s.size), atol=1e-5, btol=1e-5)
    # print(dA, db, dc)

    # # plot figure
    if plotfig is True:
        price = optMini_util.to_np(price.squeeze(1)) # price is already embedded in q, here we just convert it for plotting
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(1, T + 1) - 0.2, x_sol[:T] - x_sol[T:2 * T], width=0.4, label='Net charging')
        plt.bar(np.arange(1, T + 1) + 0.2, price, width=0.4, label='Price')
        plt.legend(fontsize=15)
        plt.title("Battery Control without Demand")
        plt.xlabel('Time steps (30min interval)', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.ylim([-1.1, 1.1])
        plt.tight_layout()
        plt.savefig('../fig/Batt_with_demand_charging_plot_conic.png')
        plt.close('all')



def extract_xsols(Xs, T=48):
    xs = np.array([x.tolist() for x in Xs])
    x_sols = xs[:, 0:3*T]
    return x_sols

def construct_QP_battery_w_D_conic_batch(param_set=None, D=None, p=None, debug=False):
    bs = D.shape[0] # bs == batch size

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)

    Gs = [optMini_util.to_np(G) for i in range(bs)]
    hs = [optMini_util.to_np(torch.cat([h, d.view(T, 1)], dim=0)) for d in D] # demand d is from data input
    Qs = [optMini_util.to_np(Q) for i in range(bs)]
    qs = [optMini_util.to_np(q) for i in range(bs)]
    As = [optMini_util.to_np(A) for i in range(bs)]
    bs = [optMini_util.to_np(b) for i in range(bs)]

    x_sols_batch, y_sols_batch, s_sols_batch, Ds_batch, DTs_batch = optMini_cvx.conic_transform_batch(Qs, qs, Gs, hs, As, bs)
    # print(x_sols_batch.shape)
    xs_batch = extract_xsols(x_sols_batch, T=T)


    if debug:
        print("-" * 40)
        print(xs_batch.shape)
        print("-" * 40)
        print(xs_batch)
        print("==" * 40)
        print(hs)
        print("-" * 40)
        print(Qs[0].shape)
        # print(Qs[0])
        print("-" * 40)
        # print(qs)



def run_battery(dataloader, params=None):
    ## multiple iterations
    # init price
    _default_horizon_ = 48
    torch.manual_seed(2)
    price = torch.rand((_default_horizon_, 1))
    with tqdm(dataloader) as pbar:
        for k, (D, Y) in enumerate(pbar):
            # print(k, D, optMini_util.convert_binary_label(Y, 1500.0))
            # construct_QP_battery_w_D_cvx(param_set=params, d=D[0], p=price, plotfig=False)
            # construct_QP_battery_w_D_conic(param_set=params, d=D[0], p=price, plotfig=False)
            construct_QP_battery_w_D_conic_batch(param_set=params, D=D, p=price, debug=True)
            if k > 1:
                raise NotImplementedError




params = dict(c_i=1, c_o=1, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, gamma=0.5, alpha=0.2)
# check_basic(param_set=params, cp_solver=cp.CVXOPT)
# check_basic(param_set=params, cp_solver=cp.GUROBI)
# check_basic_csc(param_set=params)

run_battery(dataloader_dict['train'], params=params)