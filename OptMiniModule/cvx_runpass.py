import cvxpy as cp
import numpy as np
import time
import sys
sys.path.append('..')

try:
    import diffcp.cone_program as diffcp_cprog
except ModuleNotFoundError:
    import OptMiniModule.diffcp.cone_program as diffcp_cprog
except:
    FileNotFoundError


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



def conic_format_solve_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=False):
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
1/2 x^T Q x + q^T x + p^T * pos(d+GAMMA \eps) 
s.t. Ax = b; 
     Gx <= h;  
"""
# this function injest in a single sequence demamnd
def forward_single_cvx_np_Filter(Q, q, G, h, A, b, xi, d, epsilon, T=48, p=None, sol_opt=cp.CVXOPT, verbose=False):
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]
    if verbose:
        print("\n inside the cvx np filter :", T, nz )
        print([part.shape for part in [Q, q, G, h, A, b]])

    x_ = cp.Variable(nz)
    # assert T == nz / 3
    # GAMMA = cp.Semidef(T)
    p = np.expand_dims(p, 1) # convert the price into a column vector
    # gammas = [cp.Variable(T) for i in range(T)]
    GAMMA = cp.Semidef(T)

    if d.shape == (T,):
        d = np.expand_dims(d, 1)
    # print("x size {}, num of ineq {}".format(x_.size, nineq))
    # print(gammas.size, gammas[:,0], gammas[0,:])
    # raise NotImplementedError("===== break here =====")
    # term1 = np.array([ (epsilon.T * GAMMA[:,i] + d[i]) for i in range(T)])
    term1 = GAMMA * epsilon + d
    # print(eval(term1))
    # print(d.shape)
    # print(term1)
    # print(p.shape, p, np.expand_dims(p, 1))
    # print(p.T * cp.pos(term1))
    # raise NotImplementedError("===== break here =====")
    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_ + p.T * cp.pos(term1))
    # raise NotImplementedError("===== break here =====")
    eqCon = A * x_ == b if neq > 0 else None
    eqCon_sdp = cp.trace(GAMMA) == xi
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        ineqCon = G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None

    cons = [constraint for constraint in [eqCon, eqCon_sdp, ineqCon, slacksCon] if constraint is not None]
    prob = cp.Problem(obj, cons)
    # raise NotImplementedError("===== break here =====")
    # ------------------------------
    # calculate time
    # ------------------------------
    # start = time.perf_counter()
    prob.solve(solver=sol_opt, verbose=verbose)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert ('optimal' in prob.status)
    # end = time.perf_counter()
    # print("[CVX - %s] Compute solution : %.4f s." % (sol_opt, end - start))
    # raise NotImplementedError("===== break here =====")

    xhat = np.array(x_.value).ravel()
    GAMMA_hat = np.array(GAMMA.value).ravel()
    lam = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    lam_sdp = np.array(eqCon_sdp.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        mu = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        mu = slacks = None

    return prob.value, xhat, GAMMA_hat, lam, lam_sdp, mu, slacks