import torch
from tqdm import tqdm
import processData
import nets
import sys, os
sys.path.append("..")

import time
import cvxpy as cp

from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('paper', style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 2.5}, )

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib

import numpy as np
desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=40)


def _create_price(steps_perHr=2):
    HORIZON = 24
    T1 = 16
    T2 = T1 + 5
    T3 = HORIZON
    rate_offpeak = 0.202
    rate_onpeak = 0.463
    price_shape = np.hstack((rate_offpeak * np.ones((1, T1 * steps_perHr)),
                             rate_onpeak * np.ones((1, (T2-T1) * steps_perHr)),
                             rate_offpeak * np.ones((1, (T3-T2) * steps_perHr ))))
    p = torch.Tensor(price_shape).reshape(-1, 1)
    return p





def _debug_check_verbose_sol_byCVX(x_sol, T):

    print("=" * 100)
    print(np.expand_dims(x_sol.round(4), 1))
    print("=" * 100)
    print("one step backward:", np.round(x_sol[:T]*0.95 - x_sol[T:(T+T)] + x_sol[2*T:], 3))
    print("one step forward:",np.round(x_sol[2*T:], 3))
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



def _debug_compare_cvx_and_conic_solution(x_sol_cvx, x_sol_conic, batch=0):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].plot(x_sol_cvx.T)
    ax[0].set_title("cvx solution")
    ax[1].plot(x_sol_conic.T)
    ax[1].set_title("conic solution")
    plt.tight_layout()
    plt.savefig('../fig/batch_sol_comparison_batch_%d.png' % batch)
    plt.close(fig)


def check_basic(param_set=None, p=None, plotfig=False, debug=False, cp_solver=cp.CVXOPT):

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)

    [Q, q, G, h, A, b] = list(map(optMini_util.to_np, [Q, q, G, h, A, b]))
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

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)

    h = torch.cat([h, d.view(T, 1)], dim=0)

    [Q, q, G, h, A, b] = list(map(optMini_util.to_np, [Q, q, G, h, A, b]))

    obj, x_sol, lam, mu, slacks = optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp_solver, verbose=True)  # gurobi
    # optMini_cvx.forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=True)  # cvxopt
    # print(obj, x_sol, nu, lam, slacks)
    ################################
    # check battery status
    # print(np.round(x_sol[:T] * 0.95 - x_sol[T:(T + T)] + x_sol[2 * T:], 3))
    # print(np.round(x_sol[2 * T:], 3))

    if plotfig is True:
        print(h.shape, d.shape)
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



# TODO ================================
# @since 2019/08/27
# the scenario of privatized demand
def construct_QPSDP_battery_w_privD_cvx(param_set=None, d=None, p=None, plotfig=False, cp_solver=cp.CVXOPT, debug=False):

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    # FIXME now just comment out the positive demand constriant
    # G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    # G = torch.cat([G, G_append], dim=0)

    [Q, q, G, h, A, b] = list(map(optMini_util.to_np, [Q, q, G, h, A, b]))

    price = optMini_util.to_np(price.squeeze(1))
    d = optMini_util.to_np(d)  # convert torch tensor to numpy
    epsilon = np.random.rand(T)
    xi = 0.03

    obj, xhat, GAMMA_hat, lam, lam_sdp, mu, slacks = optMini_cvx.forward_single_d_cvx_Filter(Q, q, G, h, A, b, xi, d[:T], epsilon,
                                                                                              T=T, p=price, sol_opt=cp_solver,
                                                                                             verbose=debug)


    if debug:
        print("==== Obj value: {:.4f}".format(obj))
        print("==== [GAMMA]: ", GAMMA_hat.round(4))
        _debug_check_verbose_sol_byCVX(xhat, T)



    if plotfig:
        print(GAMMA_hat.shape)
        plt.figure(figsize=(6, 5))
        sns.heatmap(GAMMA_hat)
        plt.tight_layout()
        plt.savefig('../fig/linear_filter_w1_%s.png'% cp_solver)
    pass

#  ==== [cvx] end here for single demand input ====

#  ==== [conic] start here  ====
def construct_QPSDP_battery_w_privD_conic(param_set=None, d=None, p=None, plotfig=False, cp_solver=cp.CVXOPT, debug=False):

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    # FIXME now just comment out the positive demand constriant
    # G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    # G = torch.cat([G, G_append], dim=0)

    [Q, q, G, h, A, b] = list(map(optMini_util.to_np, [Q, q, G, h, A, b]))

    price = optMini_util.to_np(price.squeeze(1))
    d = optMini_util.to_np(d)  # convert torch tensor to numpy
    epsilon = np.random.rand(T)
    xi = 0.03
    delta =0.01
    x_sol, y, s, derivative, adjoint_derivative, A_, b_, c_ = optMini_cvx.forward_single_d_conic_solve_Filter(Q, q, G, h, A, b, xi, d[:T], epsilon, delta=delta,
                                                     T=T, p=price, sol_opt=cp_solver, verbose=debug)

    if debug:
        print("==== [x_sol]:", x_sol.shape)
        _debug_check_verbose_sol_byDIFFCP(x_sol, T)
        t_ = (3 * T)
        # print(x_sol.round(4))
        # print("potential GAMMA: {}".format(  x_sol[t_: (t_+T*T)].reshape(T, T).transpose()) )
        print("potential GAMMA:")
        pprint(x_sol[t_: (t_+T*T)].reshape(T, T).transpose())
        print("rand vector:", epsilon)
        print("demand: ", d[:T])
        print("GAMMA * eps:", x_sol[t_: (t_+T*T)].reshape(T, T).transpose().dot(epsilon.transpose()))


# TODO ==== end here for single Demand ====

def check_basic_csc(param_set=None, p=None, plotfig=False, debug=False):

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)

    [Q, q, G, h, A, b] = list(map(optMini_util.to_np, [Q, q, G, h, A, b]))
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
    x_sol, y, s, D, DT, A_, b_, c_ = optMini_cvx.forward_conic_format_solve_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=True)

    ##################################
    if debug:
        # print("===" * 20)
        print("=== [DEBUG] ===")
        _debug_check_verbose_sol_byDIFFCP(x_sol, T)
        # print("===" * 20)
        print("A:")
        print(A, A.shape)
        print("===" * 20)
        print("G:")
        print(G, G.shape)
        print("===" * 20)
        print("Q:")
        print(Q)
        print("L = LU(Q):")
        print(np.linalg.cholesky(Q), np.linalg.cholesky(Q)*2, np.linalg.cholesky(Q)*np.sqrt(2))
        print("===" * 20)
        print("b:")
        print(b)
        print("===" * 20)
        print("h:")
        print(h)
        print("===" * 20)
        print("q:")
        print(q)
        print("---" * 20)
        print(A_.todense(), A_.shape)
        print(b_, b_.shape)
        print("c:")
        print(c_, c_.shape)
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
    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)

    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)
    h = torch.cat([h, d.view(T, 1)], dim=0) # demand d is from data input

    [Q, q, G, h, A, b] = list(map(optMini_util.to_np, [Q, q, G, h, A, b]))

    # ###### formulate & solve problem #######
    x_sol, y, s, D, DT, A_, b_, c_ = optMini_cvx.forward_conic_format_solve_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=True)

    if debug:
        print(h.shape, d.shape)
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
        plt.bar(np.arange(1, T + 1) - 0.2, x_sol[:T] - x_sol[T:(2 * T)], width=0.4, label='Net charging')
        plt.bar(np.arange(1, T + 1) + 0.2, price, width=0.4, label='Price')
        plt.legend(fontsize=15)
        plt.title("Battery Control without Demand")
        plt.xlabel('Time steps (30min interval)', fontsize=16)
        plt.tick_params(labelsize=16)
        plt.ylim([-1.1, 1.1])
        plt.tight_layout()
        plt.savefig('../fig/Batt_with_demand_charging_plot_conic.png')
        plt.close('all')







################################################################################
# THE FOLLOWING function is developed for solving problems in a batched-manner #

##################################
### internal helper functions ####
##################################
# @since 2019/08/25
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

def _extract_xsols_from_conic_sol(Xs, T=48):
    xs = np.array([x.tolist() for x in Xs])
    x_sols = xs[:, 0:(3*T)]
    return x_sols



def construct_QP_battery_w_D_cvx_batch(param_set=None, D=None, p=None, cp_solver=cp.GUROBI, debug=False):
    batch_size = D.shape[0]
    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)

    Gs = [optMini_util.to_np(G) for i in range(batch_size)]
    hs = [optMini_util.to_np(torch.cat([h, d.view(T, 1)], dim=0)) for d in D]  # demand d is from data input
    Qs = [optMini_util.to_np(Q) for i in range(batch_size)]
    qs = [optMini_util.to_np(q) for i in range(batch_size)]
    As = [optMini_util.to_np(A) for i in range(batch_size)]
    bs = [optMini_util.to_np(b) for i in range(batch_size)]

    res = optMini_cvx.cvx_transform_solve_batch(Qs, qs, Gs, hs, As, bs, cp_sol =cp_solver, n_jobs = 10)
    objs_batch = _convert_to_np_scalars(res, 0)
    xs_batch = _convert_to_np_arr(res, 1)
    lams_batch = _convert_to_np_arr(res, 2)
    mus_batch = _convert_to_np_arr(res, 3)
    slacks_batch = _convert_to_np_arr(res, 4)

    if debug:
        print(xs_batch.shape)

    return xs_batch



def construct_QP_battery_w_D_conic_batch(param_set=None, D=None, p=None, debug=False):
    batch_size = D.shape[0] # bs == batch size

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    G_append = torch.cat([-torch.eye(T), torch.eye(T), torch.zeros((T, T))], dim=1)
    G = torch.cat([G, G_append], dim=0)

    Gs = [optMini_util.to_np(G) for i in range(batch_size)]
    hs = [optMini_util.to_np(torch.cat([h, d.view(T, 1)], dim=0)) for d in D] # demand d is from data input
    Qs = [optMini_util.to_np(Q) for i in range(batch_size)]
    qs = [optMini_util.to_np(q) for i in range(batch_size)]
    As = [optMini_util.to_np(A) for i in range(batch_size)]
    bs = [optMini_util.to_np(b) for i in range(batch_size)]

    # note : the following method solves the conic form of convex program
    x_sols_batch, y_sols_batch, s_sols_batch, Ds_batch, DTs_batch, As_batch, bs_batch, cs_batch = optMini_cvx.conic_transform_solve_batch(Qs, qs, Gs, hs, As, bs, n_jobs=10)
    xs_batch = _extract_xsols_from_conic_sol(x_sols_batch, T=T)



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
    return xs_batch


# TODO =================================

def construct_QPSDP_battery_w_privD_cvx_batch(param_set=None, D=None, p=None, plotfig=False, cp_solver=cp.CVXOPT, debug=False):

    batch_size = D.shape[0]

    Q, q, G, h, A, b, T, price = _form_QP_params(param_set, p)
    Gs = [optMini_util.to_np(G) for i in range(batch_size)]
    hs = [optMini_util.to_np(h) for i in range(batch_size)]  # demand d is from data input
    Qs = [optMini_util.to_np(Q) for i in range(batch_size)]
    qs = [optMini_util.to_np(q) for i in range(batch_size)]
    As = [optMini_util.to_np(A) for i in range(batch_size)]
    bs = [optMini_util.to_np(b) for i in range(batch_size)]

    optMini_cvx.cvx_transform_QPSDP_solve_batch()


    raise NotImplementedError

# TODO ========= end batched method =====



#########################

def run_battery(dataloader, params=None):
    ## multiple iterations
    # init price

    _default_horizon_ = params['T'] if params is not None else 48
    torch.manual_seed(2)
    # price = _create_price(steps_perHr=2) + torch.rand((_default_horizon_, 1))* 0.05  # price is a column vector
    price = torch.rand((_default_horizon_, 1))
    with tqdm(dataloader) as pbar:
        for k, (D, Y) in enumerate(pbar):
            # print(k, D, optMini_util.convert_binary_label(Y, 1500.0))
            # construct_QP_battery_w_D_cvx(param_set=params, d=D[0], p=price, plotfig=False)
            # construct_QP_battery_w_D_conic(param_set=params, d=D[0], p=price, plotfig=False)
            # FIXME hacking way to check private demand
            # @since 2019/08/27
            # start = time.perf_counter()
            # construct_QPSDP_battery_w_privD_cvx(param_set=params, d=D[0], p=price, cp_solver=cp.MOSEK, plotfig=True, debug=True)
            # end = time.perf_counter()
            # print("[CVX - %s] Compute solution : %.4f s." % (cp.MOSEK, end - start))
            start = time.perf_counter()
            construct_QPSDP_battery_w_privD_cvx(param_set=params, d=D[0], p=price, cp_solver=cp.SCS, plotfig=False, debug=True)
            end = time.perf_counter()
            print("[CVX - %s] Compute solution : %.4f s." % (cp.SCS, end - start))

            construct_QPSDP_battery_w_privD_conic(param_set=params, d=D[0], p=price, cp_solver=cp.SCS, debug=True)
            print("[DIFFCP - %s] Compute solution : %.4f s." % (cp.SCS, end - start))

            raise NotImplementedError("Mannul break!")

            ### ========== comparison with battery control with non private demand  ============= ###
            start = time.perf_counter()
            x_sol_cvx = construct_QP_battery_w_D_cvx_batch(param_set=params, D=D, p=price, cp_solver=cp.GUROBI, debug=False)
            end = time.perf_counter()
            print("[CVX - %s] Compute solution : %.4f s." % (cp.GUROBI, end - start))
            start = time.perf_counter()
            x_sol_conic = construct_QP_battery_w_D_conic_batch(param_set=params, D=D, p=price, debug=False)
            end = time.perf_counter()
            print("[DIFFCP] Compute solution and set up derivative: %.4f s." % (end - start))
            # _debug_compare_cvx_and_conic_solution(x_sol_cvx, x_sol_conic, batch=k)
            if (k + 1) > 0:
                raise NotImplementedError("---- Iter {:d} break manually! ----".format(k))




# params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, gamma=0.5, alpha=0.2)
params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=4, B=1.5, beta1=0.6, beta2=0.4, gamma=0.5, alpha=0.2)
# check_basic(param_set=params, cp_solver=cp.CVXOPT, plotfig=False)
# check_basic(param_set=params, cp_solver=cp.GUROBI, plotfig=False)
# check_basic_csc(param_set=params, plotfig=False, debug=True)

run_battery(dataloader_dict['train'], params=params)