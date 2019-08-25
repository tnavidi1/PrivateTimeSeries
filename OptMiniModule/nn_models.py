import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F


import sys
sys.path.append('..')

try:
    import diffcp.cone_program as diffcp_cprog
except ModuleNotFoundError:
    import OptMiniModule.diffcp.cone_program as diffcp_cprog
except:
    FileNotFoundError("diffcp import error")

try:
    import util as ut
except ModuleNotFoundError:
    import OptMiniModule.util as ut
except:
    FileNotFoundError("util import error")


class OptPrivModel(nn.Module):
    def __init__(self, Q, q, G, h, A, b, T=48):
        super().__init__()
        self.horizon = T
        self.Q = Q
        self.q = q
        self.G = G
        self.h = h
        self.A = A
        self.b = b

        self.GAMMA = Parameter(torch.rand((T, T)).double())

        # Set prior as fixed parameter attached to Module
        # TODO - fix it when prior is non-Gaussian
        self.z_dim = self.horizon
        self.z_prior_m = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        ##################################################
        # nx = (n ** 2) ** 3
        #
        # self.Q = Variable(Qpenalty * torch.eye(nx).double())
        # self.G = Variable(-torch.eye(nx).double())
        # self.h = Variable(torch.zeros(nx).double())
        # A_shape = (40, 64)  # Somewhat magic, it's from the true solution.
        # self.A = Parameter(torch.rand(A_shape).double())
        # self.b = Variable(torch.ones(A_shape[0]).double())
        ##################################################


    def forward(self, D):
        """
        input Demand
        :param D:
        :return:
        """
        batch_size = D.shape[0]
        z_noise = self.sample_z(batch=batch_size)



        pass



    def sample_z(self, batch):
        return ut.sample_gaussian( self.z_prior[0].expand(batch, self.z_dim),
                                   self.z_prior[1].expand(batch, self.z_dim))
