import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from torch._jit_internal import weak_script_method
import math
import torch.nn.init as init

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


# Controller model v1
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

        self.GAMMA = Parameter(torch.rand((T, T)))

        # Set prior as fixed parameter attached to Module
        # TODO - fix it when prior is non-Gaussian
        self.z_dim = self.horizon
        self.z_prior_m = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        self.reset_parameters()
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



    def reset_parameters(self):
        init.kaiming_uniform_(self.GAMMA, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    # @weak_script_method
    def forward(self, D):
        """
        input Demand
        :param D:
        :return:
        """
        batch_size = D.shape[0]
        z_noise = self.sample_z(batch=batch_size)
        print("z-noise:", z_noise.shape)
        print("GAMMA:", self.GAMMA.shape)
        print((self.GAMMA))
        # out = z_noise.matmul(self.GAMMA.t()) # print(out.shape) # batch_size, dimension
        D_tilde = D + z_noise.matmul(self.GAMMA.t())
        D_tilde = F.relu(D_tilde)
        print(D_tilde.shape, D_tilde)
        # input.matmul(weight.t())

        raise NotImplementedError



    def sample_z(self, batch):
        return ut.sample_gaussian( self.z_prior[0].expand(batch, self.z_dim),
                                   self.z_prior[1].expand(batch, self.z_dim))




# To align with previous model conventions in class
class LinearFilter(nn.Module):

    def __init__(self, input_dim = 10, y_dim=0, output_dim=10, bias=None):
        super(LinearFilter, self).__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim + self.y_dim, self.output_dim, bias=bias)

    def forward(self, x, y=None):
        """

        :param x: input the feature vectors
        :param y: the default y input would be one-hot encoding version of discrete labels
        :return:
        """
        xy = x if y is None else torch.cat([x, y], dim=1)
        o = self.fc(xy)
        return o


class Generator(nn.Module):
    def __init__(self, nn='v1', name='g_filter', z_dim=24, y_priv_dim=2, device=None):
        """

        :param nn: (str) version
        :param name: (str) model name
        :param z_dim: (int) 48 or 24 are the usual default settings
        :param y_priv_dim: one-hot vector dimensions
        :param device:
        """
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.y_priv_dim = y_priv_dim

        self.device = device

        # create a linear filter # @since 2019/08/29 modify zero bias
        self.filter = LinearFilter(self.z_dim, self.y_priv_dim, output_dim=self.z_dim, bias=None)  # setting noise dim is same as latent dim

        # Set prior as fixed parameter attached to module
        self.z_prior_m = Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def forward(self, x, y=None):
        batch_size = x.size()[0] # batch size is the first dimension

        z_noise = self.sample_z(batch=batch_size)
        z_noise = z_noise / z_noise.norm(2, dim=1).unsqueeze(1).repeat(1, self.z_dim)

        x_proc_noise = self.filter(z_noise, y)
        x_noise = x + x_proc_noise
        return x_noise

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                self.z_prior[1].expand(batch, self.z_dim))

    def display_filter_params(self):
        print(self.filter.parameters())
