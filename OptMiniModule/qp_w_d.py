import torch
from torch.autograd import Function
import numpy as np

import cvxpy as cp

import sys
sys.path.append('..')

try:
    import diffcp.cone_program as diffcp_cprog
except ModuleNotFoundError:
    import OptMiniModule.diffcp.cone_program as diffcp_cprog
except:
    FileNotFoundError("diffcp import error")

try:
    import OptMiniModule.cvx_runpass as optMini_cvx
    import OptMiniModule.util as optMini_util
except ModuleNotFoundError:
    import cvx_runpass as optMini_cvx
    import util as optMini_util
except:
    FileNotFoundError("cvx_runpass import error")


def torch_bQuad(x, Q):
    # batch Quad_form
    # multiple   x^T Q x
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)

def torch_Quad_c_(x, Q):
    # Quad form cost, single sample
    #        x ^T Q x
    # torch.matmul(x.transpose(), Q)
    return (x.t().matmul(Q)).matmul(x)

def np_Quad_c_(x, Q):
    # numpy version, single sample
    #     x^T Q x
    return np.dot((x.transpose().dot(Q)), x)


def torch_Diff_coef(T):
    return torch.cat([torch.eye(T), -torch.eye(T)], dim=1)

def np_Diff_coef_(T):
    return np.concatenate([np.eye(T), -np.eye(T)], axis=1)




#class QPFunction(Function):
class QP_privD(Function):
    def __init__(self, T, verbose=False, solver_opt=cp.SCS):
        self.verbose = verbose
        self.solver_opt = solver_opt
        self.T = T
        self._store_tensors_ = None

    # def forward(self, price_, GAMMA_, d, epsilon, y_onehot, Q, G, h, A, b, T):
    # @staticmethod
    def forward(self, price_, GAMMA_, d, epsilon, y_onehot_, Q_, G, h, A, b, T=4, sol_opt=cp.SCS, verbose=0):
        # p = price_
        p = optMini_util.to_np(price_)

        # TODO ==== call the conic solver ====
        # x_ = conic solver
        [GAMMA, d, eps, y_onehot_, Q, G, h, A, b] = list(map(optMini_util.to_np, [GAMMA_, d, epsilon, y_onehot_, Q_, G, h, A, b]))
        # TODO the conic function should return:
        #   x,
        #  dA, db, dc by inputing dx, dy, ds
        """
        The standard conic form is  
           min c^Tx, 
        s.t.  Ax + s = b 
              s \in K 
        """
        x_, db_ = optMini_cvx._convex_formulation_w_GAMMA_d_conic(p, GAMMA, d, eps, y_onehot_, Q, G, h, A, b, T,
                                                                  sol_opt=sol_opt, verbose=verbose)
        print(x_, db_)
        """    
        # 0.5 * cp.quad_form(x_, Q) + p.T * cp.pos(Diff_coef_ * x_[0:(2 * T), 0] + d)
        # evaluate the loss based on true demand 
        """
        x_ = torch.Tensor(x_)
        db_ = torch.Tensor(db_)
        eps_ = torch.Tensor(eps)
        d = torch.Tensor(d)
        # print(torch_Diff_coef(self.T).matmul(x_[0:(2 * self.T)]))
        # print(torch_Quad_c_(x_, Q_))
        cost = 0.5 * torch_Quad_c_(x_, Q_) + torch.relu(torch_Diff_coef(self.T).matmul(x_[0:(2 * self.T)]) + d).matmul(price_)
        # print(cost)
        d_d_ = - db_[T:(2*T)] # derivative of demand
        # self.save_for_backward(x_, GAMMA_, d_d_, eps_)
        self._store_tensors_ = (x_, GAMMA_, d_d_, eps_)
        return cost

    # @staticmethod
    def backward(self, T):
        x, GAMMA, d_d, eps = self._store_tensors_
        d_d_ = (d_d.reshape(T, 1)).expand((T, T))
        dGAMMA = (1/T) * ( d_d_ / eps)
        # dGAMMA = ((1/T) * (np.tile(np.expand_dims(d_d, 1), T)) / eps)
        # print(dGAMMA)
        return dGAMMA


# x = torch.randn(3, 3)
# x = x.unsqueeze(0)
# print(x.repeat(2, 1, 1))

# c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
# print(torch.norm(c, p=1, dim=0))