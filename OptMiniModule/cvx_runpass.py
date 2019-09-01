import cvxpy as cp
import numpy as np
import time
from itertools import chain
import sys
sys.path.append('..')

try:
    import diffcp.cone_program as diffcp_cprog
except ModuleNotFoundError:
    import OptMiniModule.diffcp.cone_program as diffcp_cprog
except:
    FileNotFoundError("diffcp import error!")

try:
    import util as ut
except ModuleNotFoundError:
    import OptMiniModule.util as ut
except:
    FileNotFoundError("util import error!")


import multiprocessing as mp
from multiprocessing.pool import ThreadPool


"""
min 1/2 x^T Q x + q^T x \\
s.t  nu: Ax = b  \\
     lam : Gx <= h  

The corresponding Lagrangian is 
L(x, \lambda, \mu) =  1/2 x^T Q x + q^T x + \lambda^T (Ax-b) + \mu^T (Gx-h)

"""

def forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=False):
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    x_ = cp.Variable(nz)
    # print("x size {}, num of ineq {}".format(x_.size, nineq))
    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_)
    eqCon = A * x_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        ineqCon =  G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None

    cons = [constraint for constraint in [eqCon, ineqCon, slacksCon] if constraint is not None]
    prob = cp.Problem(obj, cons)
    # ------------------------------
    # calculate time
    # ------------------------------
    # start = time.perf_counter()
    prob.solve(solver=sol_opt, verbose=verbose)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    # end = time.perf_counter()
    # print("[CVX - %s] Compute solution : %.4f s." % (sol_opt, end - start))

    xhat = np.array(x_.value).ravel()
    lam = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        mu = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        mu = slacks = None

    return prob.value, xhat, lam, mu, slacks


def scs_data_from_cvxpy_problem(problem, cp_SCS=cp.SCS):
    """
    The SCS has default A, b, c three
    :param problem:
    :param cp_CSC:
    :return:
    """
    data = problem.get_problem_data(cp_SCS)
    cone_dims = data["dims"]
    return data['A'], data['b'], data['c'], cone_dims



def forward_conic_format_solve_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=False):
    """

    :param Q:
    :param q:
    :param G:
    :param h:
    :param A:
    :param b:
    :param sol_opt:
    :param verbose:
    :return:
    """
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    x_ = cp.Variable(nz)
    # print("x size {}, num of ineq {}".format(x_.size, nineq))
    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_)
    eqCon = A * x_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        ineqCon = G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None

    cons = [constraint for constraint in [eqCon, ineqCon, slacksCon] if constraint is not None]
    prob = cp.Problem(obj, cons)
    # The current form only accept cvx problem and scs solver option
    A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob, sol_opt)
    # @note: after converting into scs conic form
    # The A, b, c here represents general form of :
    #      min  c^T x
    #    s.t.   Ax + s = b
    #           s \in \mathcal{K}
    #  where K is a cone
    # ----------------------------
    # calculate time
    # ----------------------------
    # start = time.perf_counter()
    x, y, s, derivative, adjoint_derivative = diffcp_cprog.solve_and_derivative(
        A, b, c, cone_dims, eps=1e-5)
    # end = time.perf_counter()
    # print("[DIFFCP] Compute solution and set up derivative: %.4f s." % (end - start))

    return x, y, s, derivative, adjoint_derivative, A, b, c

# code up for batched cvx solver wrapper
def forward_single_np_cvx_wrapper(Q, q, G, h, A, b, cp_sol=cp.CVXOPT, verbose=False):
    return forward_single_np(Q, q, G, h, A, b, sol_opt=cp_sol, verbose=verbose)

# code up for batched cvx call -- using multiprocess
def cvx_transform_solve_batch(Qs, qs, Gs, hs, As, bs, cp_sol = cp.SCS, n_jobs = 1, verbose=False):

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    batch_size = len(As)
    pool = ThreadPool(processes=n_jobs)
    args = []
    for i in range(batch_size):
        args += [(Qs[i], qs[i], Gs[i], hs[i], As[i], bs[i], cp_sol, verbose)]
        # args += [(As[i], bs[i], cs[i], cone_dicts[i],
        #           None if warm_starts is None else warm_starts[i], kwargs)]
    return pool.starmap(forward_single_np_cvx_wrapper, args)


# code up for batched conic solver
def __single_cvxprob_formulate(Q, q, G, h, A, b, sol_opt=cp.SCS):
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]
    x_ = cp.Variable(nz)
    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_)
    eqCon = A * x_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        ineqCon = G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None

    cons = [constraint for constraint in [eqCon, ineqCon, slacksCon] if constraint is not None]
    prob = cp.Problem(obj, cons)
    A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob, sol_opt)
    return [A, b, c, cone_dims]


def conic_transform_solve_batch(Qs, qs, Gs, hs, As, bs, cp_sol=cp.SCS, n_jobs=4):

    results = np.array([__single_cvxprob_formulate(Q, q, G, h, A, b, sol_opt=cp_sol) \
                        for Q, q, G, h, A, b in zip(Qs, qs, Gs, hs, As, bs)])
    As_ = results[:, 0]
    bs_ = results[:, 1]
    cs_ = results[:, 2]
    cons_dims_list = results[:, 3]

    res = diffcp_cprog.solve_and_derivative_batch(As_, bs_, cs_, cons_dims_list, n_jobs=n_jobs, eps=1e-5)
    res = np.array(res)

    x_sols_batch =res[:, 0]
    y_sols_batch =res[:, 1]
    s_sols_batch =res[:, 2]
    derivative_batch =res[:, 3]
    adjoint_derivative = res[:, 4]

    return  x_sols_batch, y_sols_batch, s_sols_batch, derivative_batch, adjoint_derivative, As_, bs_, cs_



"""
Private demand 

1/2 x^T Q x + q^T x + p^T * pos(d+GAMMA \eps) 
s.t. Ax = b; 
     Gx <= h;
     prob( d + GAMMA\eps > 0) > 1-\delta  
"""

# this function injest in a single sequence demamnd
def forward_single_d_cvx_Filter(Q, q, G, h, A, b, d, epsilon, xi, delta, T, p=None,
                                sol_opt=cp.CVXOPT, verbose=False):
    """
    This function processes the SDP
    :param Q:
    :param q:
    :param G:
    :param h:
    :param A:
    :param b:
    :param xi:
    :param d:
    :param epsilon:
    :param delta:
    :param T:
    :param p:
    :param sol_opt:
    :param verbose:
    :return:
    """
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    if p.shape == (T,):
        p = np.expand_dims(p, 1) # convert the price into a column vector

    if d.shape == (T,):
        d = np.expand_dims(d, 1)

    if verbose:
        print("\n inside the cvx np filter :", T, nz )
        print([part.shape for part in [Q, q, G, h, A, b]])

    x_ = cp.Variable(nz)
    GAMMA = cp.Semidef(T)
    # assert T == nz / 3
    # print("x size {}, num of ineq {}".format(x_.size, nineq))
    term1 = GAMMA * epsilon + d

    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_ + p.T * cp.pos(term1) + cp.pos(cp.norm(GAMMA, "nuc") - xi ) )
    eqCon = A * x_ == b if neq > 0 else None
    soc_ineqCon = [cp.norm(GAMMA[:, i], 2) <= (d[i, 0] / abs(ut.function_normal_cdf_inv(delta))) for i in range(T)] # ut.function_normal_cdf_inv(delta)

    # eqCon_sdp = None  # convert the SDP constraint in the objective, # eqCon_sdp = cp.norm(GAMMA, "nuc") == xi
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        ineqCon = G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons_collected = [eqCon, ineqCon] + soc_ineqCon + [slacksCon]

    cons = [constraint for constraint in cons_collected if constraint is not None]
    prob = cp.Problem(obj, cons)
    # ------------------------------
    # calculate time
    # ------------------------------
    # start = time.perf_counter()
    if sol_opt == cp.MOSEK:
        # mosek params: https://docs.mosek.com/9.0/javafusion/parameters.html
        mosek_param_setting = {"MSK_DPAR_BASIS_TOL_X": 1e-4,
                               "MSK_DPAR_BASIS_TOL_S": 1e-5,
                               "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-5,
                               "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-5,
                               "MSK_DPAR_INTPNT_TOL_INFEAS": 1e-5,
                               "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-5,
                               "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-5}
        # The first 5 items might be not important, we discovered that
        # most relavent metrics is 'MSK_DPAR_INTPNT_CO_TOL_PFEAS' and 'MSK_DPAR_INTPNT_CO_TOL_REL_GAP'
        prob.solve(solver=sol_opt, verbose=verbose, mosek_params=mosek_param_setting)  # solver=cp.SCS, max_iters=5000, verbose=False)
    else:
        prob.solve(solver=sol_opt, verbose=verbose) # e.g. prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)

    assert ('optimal' in prob.status)
    # end = time.perf_counter()
    # print("[CVX - %s] Compute solution : %.4f s." % (sol_opt, end - start))

    xhat = np.array(x_.value).ravel()
    GAMMA_hat = np.array(GAMMA.value)
    lam = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    # lam_socp = np.array(soc_ineqCon.dual_value).ravel() if soc_ineqCon is not None else None
    if ineqCon is not None:
        mu = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        mu = slacks = None

    # return prob.value, xhat, GAMMA_hat, lam, lam_socp, mu, slacks
    return prob.value, xhat, GAMMA_hat, lam, mu, slacks


# the function contains random vector '\eps', '\xi', 'd' and '\delta'
def forward_single_d_conic_solve_Filter(Q, q, G, h, A, b, d, epsilon, xi, delta=0.01,
                                        T=48, p=None, sol_opt=cp.CVXOPT, verbose=False):
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    if p.shape == (T,):
        p = np.expand_dims(p, 1)  # convert the price into a column vector

    if d.shape == (T,):
        d = np.expand_dims(d, 1)

    if verbose:
        print("\n inside the cvx np filter :", T, nz)
        print([part.shape for part in [Q, q, G, h, A, b]])

    x_ = cp.Variable(nz)
    # GAMMA = cp.Semidef(T)
    GAMMA = cp.Variable(rows=T, cols=T)
    # assert T == nz / 3
    # print("x size {}, num of ineq {}".format(x_.size, nineq))
    term1 = GAMMA * epsilon + d

    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_ + p.T * cp.pos(term1) + cp.pos(cp.norm(GAMMA, "nuc") - xi))
    eqCon = A * x_ == b if neq > 0 else None
    prob_ineqCon = [cp.norm(GAMMA[:, i], 2) <= (d[i, 0] / abs(ut.function_normal_cdf_inv(delta))) for i in
                    range(T)]  # ut.function_normal_cdf_inv(delta)

    eqCon_sdp = None  # convert the SDP constraint in the objective, # eqCon_sdp = cp.norm(GAMMA, "nuc") == xi
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        ineqCon = G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons_collected = [eqCon, eqCon_sdp, ineqCon] + prob_ineqCon + [slacksCon]

    cons = [constraint for constraint in cons_collected if constraint is not None]
    prob = cp.Problem(obj, cons)

    A_, b_, c_, cone_dims = scs_data_from_cvxpy_problem(prob, cp_SCS=sol_opt)

    x, y, s, derivative, adjoint_derivative = diffcp_cprog.solve_and_derivative(
        A_, b_, c_, cone_dims, eps=1e-5)
    # end = time.perf_counter()
    # print("[DIFFCP] Compute solution and set up derivative: %.4f s." % (end - start))

    return x, y, s, derivative, adjoint_derivative, A_, b_, c_



# TODO ######## start to work on the batched manner ##########

def forward_cvx_single_d_filter_wrapper(Q, q, G, h, A, b, d, epsilon, xi, delta, T, p, cp_solver, verbose):
    return forward_single_d_cvx_Filter(Q, q, G, h, A, b, d, epsilon, xi, delta, T, p=p,
                                       sol_opt=cp_solver, verbose=verbose)

def cvx_transform_QPSDP_solve_batch(Qs, qs, Gs, hs, As, bs, D, eps, xi, delta, T, p=None,
                                    cp_sol = cp.SCS, n_jobs = 1, verbose=False):
    """

    :param Qs:
    :param qs:
    :param Gs:
    :param hs:
    :param As:
    :param bs:
    :param D:
    :param eps:
    :param xi:
    :param delta:
    :param cp_sol:
    :param n_jobs:
    :param verbose:
    :return:
    """

    # prob.value, xhat, GAMMA_hat, lam, lam_sdp, mu, slacks

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    batch_size = len(As)
    pool = ThreadPool(processes=n_jobs)
    args = []
    for i in range(batch_size):
        args += [(Qs[i], qs[i], Gs[i], hs[i], As[i], bs[i], D[i], eps[i], xi, delta, T, p, cp_sol, verbose)]

    return pool.starmap(forward_cvx_single_d_filter_wrapper, args)



# TODO ######## work on the batched manner with torch capatibility ##########

# start with simple case 1:
"""
\min_{x} p^T(x + \GAMMA * [\eps, y]^T)_+ 
 
"""

def _convex_formulation_w_GAMMA_d_cvx(p, GAMMA, d, epsilon, y_onehot, Q, G, h, A, b, T, sol_opt=cp.SCS, verbose=True):
    """

    :param p:
    :param GAMMA:
    :param d:
    :param epsilon:
    :param y_onehot:
    :return:
    """

    # print("price {}, GAMMA {}, demand {}, eps: {}, y_onehot {}".format(p, GAMMA, d, epsilon, y_onehot))

    cat_vec = None
    if epsilon.shape == (T, ) and y_onehot.shape == (2, ):
        cat_vec = np.concatenate([epsilon, y_onehot], axis=0)
        cat_vec = np.expand_dims(cat_vec, 1)
        # epsilon = np.expand_dims(epsilon, 1)

    if d.shape == (T, ):
        d = np.expand_dims(d, 1)

    x_ = cp.Variable(3*T)

    print(Q.shape)
    print(x_.size)

    Diff_coef_ = np.concatenate([np.eye(T), -np.eye(T) ], axis=1)
    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + p.T * cp.pos(Diff_coef_ * x_[0:(2*T)] + d + GAMMA.dot(cat_vec)))
    ineqCon = G * x_ <= h
    eqCon = A * x_ == b
    cons = [ineqCon, eqCon]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=sol_opt, verbose=verbose)
    xhat = np.array(x_.value).ravel()

    return xhat



def _convex_formulation_w_GAMMA_d_conic(p, GAMMA, d, epsilon, y_onehot, Q, G, h, A, b, T, sol_opt=cp.SCS, verbose=True):

    cat_vec = None
    if epsilon.shape == (T,) and y_onehot.shape == (2,):
        cat_vec = np.concatenate([epsilon, y_onehot], axis=0)
        cat_vec = np.expand_dims(cat_vec, 1)
        # epsilon = np.expand_dims(epsilon, 1)

    if d.shape == (T,):
        d = np.expand_dims(d, 1)

    x_ = cp.Variable(3 * T)

    Diff_coef_ = np.concatenate([np.eye(T), -np.eye(T)], axis=1)

    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + p.T * cp.pos(Diff_coef_ * x_[0:(2 * T), 0] + d + GAMMA.dot(cat_vec)))
    ineqCon = G * x_ <= h
    eqCon = A * x_ == b
    cons = [ineqCon, eqCon]
    prob = cp.Problem(obj, cons)

    A_, b_, c_, cone_dims = scs_data_from_cvxpy_problem(prob, cp_SCS=sol_opt)

    # print("A:", A_.shape, A_)
    # print("b:", b_.shape, np.expand_dims(b_, 1))
    # tau = 96
    # print("b[:{:d}] == {}".format(tau, b_[:tau]))
    # print("c:", c_.shape, np.expand_dims(c_, 1))
    # print("==="*20)
    # print("price:\n", p)
    # print("=="*20)
    # print("d_tilde:\n", d + GAMMA.dot(cat_vec))

    x, y, s, derivative, adjoint_derivative = diffcp_cprog.solve_and_derivative(
        A_, b_, c_, cone_dims, eps=1e-5)

    x_hat = x[:3*T]
    return x_hat