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
    start = time.perf_counter()
    prob.solve(solver=sol_opt, verbose=verbose)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    end = time.perf_counter()
    print("[CVX - %s] Compute solution : %.4f s." % (sol_opt, end - start))

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



def cvx_format_problem(Q, q, G, h, A, b, sol_opt=cp.SCS, verbose=False):
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
    # The current form only accept
    A, b, c, cone_dims = scs_data_from_cvxpy_problem(prob, sol_opt)
    # ----------------------------
    # calculate time
    # ----------------------------
    start = time.perf_counter()
    x, y, s, derivative, adjoint_derivative = diffcp_cprog.solve_and_derivative(
        A, b, c, cone_dims, eps=1e-5)
    # print(A.shape)
    end = time.perf_counter()
    print("[DIFFCP] Compute solution and set up derivative: %.4f s." % (end - start))

    return x, y, s, derivative, adjoint_derivative, A, b, c

