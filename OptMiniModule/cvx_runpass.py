import cvxpy as cp
import numpy as np


"""
min 1/2 x^T Q x + q^T x \\
s.t  nu: Ax = b  \\
     lam : Gx <= h  

"""

def forward_single_np(Q, q, G, h, A, b, sol_opt=cp.CVXOPT, verbose=False):
    nz, neq, nineq = q.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    x_ = cp.Variable(nz)
    # print("x size {}, num of ineq {}".format(x_.size, nineq))
    obj = cp.Minimize(0.5 * cp.quad_form(x_, Q) + q.T * x_)
    eqCon = A * x_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)  # define slack variables
        print(slacks.size)
        # print("G shape", G.shape)
        # print(G * x_)
        ineqCon =  G * x_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None

    cons = [constraint for constraint in [eqCon, ineqCon, slacksCon] if constraint is not None]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=sol_opt, verbose=verbose)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    xhat = np.array(x_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, xhat, nu, lam, slacks
