import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import math
# try:
#     import util as ut
# except ModuleNotFoundError:
import sys
sys.path.append("..")

try:
    import basic_util as bUtil
    import losses as bLosses
except:
    raise ModuleNotFoundError("== basic_util file is not imported! ==")

try:
    import OptMiniModule.util as ut
    import OptMiniModule.cvx_runpass as OptMini_cvx
except:
    raise FileNotFoundError("== Opt module util import error! ==")




class Classifier(nn.Module):
    def __init__(self, z_dim=10, y_dim=0):
        super(Classifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 50),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(50, 24),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(24, y_dim)
            # nn.Linear(z_dim, y_dim),
        )

    def forward(self, x):
        o = self.net(x)
        return o



# @weak_module
class PosLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    # Examples::
    #
    #     >>> m = nn.Linear(20, 30)
    #     >>> input = torch.randn(128, 20)
    #     >>> output = m(input)
    #     >>> print(output.size())
    #     torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True, init_type="kaiming"):
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_type =init_type
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.init_type == "kaiming":
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif self.init_type == "uniform":
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.weight, 0, 2*bound)
        else:
            init.kaiming_normal_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, 0, 2*bound)

    # @weak_script_method
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )






# To align with previous model conventions in class
class LinearFilter(nn.Module):

    def __init__(self, input_dim = 10, y_dim=0, output_dim=10, bias=None, init_type="uniform"):
        super(LinearFilter, self).__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        # self.fc = nn.Linear(self.input_dim + self.y_dim, self.output_dim, bias=bias)
        self.fc = PosLinear(self.input_dim + self.y_dim, self.output_dim, bias=bias, init_type=init_type)

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
    def __init__(self, nn='v1', name='g_filter', z_dim=24, y_priv_dim=2,
                 Q=None, G=None, h = None, A=None, b=None, T=24, p=None,
                 device=None):
        """

        :param nn:
        :param name:
        :param z_dim:
        :param y_priv_dim:
        :param Q:
        :param G:
        :param h:
        :param A:
        :param b:
        :param T:
        :param p:
        :param device:
        """
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.y_priv_dim = y_priv_dim
        self._set_convex_prob_param(p, Q, G, h, A, b, T)
        self.device = device

        # create a linear filter # @since 2019/08/29 modify zero bias
        self.filter = LinearFilter(self.z_dim, self.y_priv_dim, output_dim=self.z_dim, bias=False, init_type="kaiming")  # setting noise dim is same as latent dim
        # Set prior as fixed parameter attached to module
        self.z_prior_m = Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def _set_convex_prob_param(self, p, Q, G, h, A, b, T):
        self.p = p # usually won't use self.p
        self.Q = Q
        self.G = G
        self.h = h
        self.A = A
        self.b = b
        self.T = T
        self.cached_noise = None
        self.cached_D_priv = None


    def forward(self, x, y=None):
        # x is the raw demand
        batch_size = x.shape[0] #x.size()[0] # batch size is the first dimension

        z_noise = self.sample_z(batch=batch_size)
        z_noise = z_noise / z_noise.norm(2, dim=1).unsqueeze(1).repeat(1, self.z_dim)

        x_proc_noise = self.filter(z_noise, y)
        x_priv = F.softplus(x + x_proc_noise)
        return x_priv, z_noise

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                self.z_prior[1].expand(batch, self.z_dim))

    def display_filter_params(self):
        params = list(self.filter.parameters())
        print(params[0].size())
        print(params[0])

    def solve_convex_forumation(self, p, D, Q, G, h, A, b, Y_onehot=None):
        batch_size = D.shape[0]
        D_ = D
        if isinstance(Y_onehot, torch.Tensor):
            D_, z_noise = self.forward(D, Y_onehot)
            self.cached_noise = z_noise
            self.cached_D_priv = D_
        elif Y_onehot is None:
            D_ = D
        else:
            raise NotImplementedError("Y_onehot:\n {} \n is not supportted".format(Y_onehot))

        def __expand_param_list(x):
            return [ut.to_np(x) for i in range(batch_size)]

        Qs, Gs, hs, As, bs = list(map(__expand_param_list, [Q, G, h, A, b])) # to numpy
        p = ut.to_np(p)
        D_detached = ut.to_np(D_.detach())
        # this call numpy data
        res = OptMini_cvx.forward_D_batch(Qs, Gs, hs, As, bs, D_detached, self.T, p=p)
        x_sols = bUtil._convert_to_np_arr(res, 1)
        objs = bUtil._convert_to_np_scalars(res, 0)

        return objs, x_sols

    def evaluate_cost_obj(self, x_sols, p=None):
        D_ = self.cached_D_priv
        Q = self.Q
        T = self.T
        return bLosses.objective_task_loss(p, x_sols, D_, Q, T)



    def util_loss(self, D, Y_onehot, p=None, xi=0.01):
        if p is None:
            p = self.p
        obj_raw, x_sol_raw = self.solve_convex_forumation(p, D, self.Q, self.G, self.h, self.A, self.b, Y_onehot=None)
        obj_priv, x_sol_priv = self.solve_convex_forumation(p, D, self.Q, self.G, self.h, self.A, self.b, Y_onehot=Y_onehot)

        # convert obj_raw, obj_priv as tensor
        [obj_raw, obj_priv, x_sol_raw, x_sol_priv] = [torch.from_numpy(x).to(torch.float) for x in \
                                                      [obj_raw, obj_priv, x_sol_raw, x_sol_priv]]

        # obj_priv = self.evaluate_cost_obj(x_sol_priv, D, Y_onehot, p=p)
        obj_priv = self.evaluate_cost_obj(x_sol_priv, p=p)
        # return MSELoss
        hinge_loss_mean = F.softplus(obj_priv - obj_raw).sum(0)
        # neg_tr_penalty = F.relu(-torch.symeig(self.filter.fc.weight)).sum(0)
        # eigvals, eig_vecs = torch.symeig(self.filter.fc.weight.data[:, :48])
        eigvals = torch.diag(self.filter.fc.weight.data[:, :48])
        neg_eig_vals =  torch.min(eigvals)
        neg_eig_penalty = F.relu(-neg_eig_vals)
        # print(self.filter.fc.weight.data[:, :48].shape)
        # raise NotImplementedError
        # + neg_tr_penalty
        print("eig values", eigvals)
        tr_penalty = F.relu(torch.trace(torch.mm(self.filter.fc.weight, self.filter.fc.weight.t())) - xi)
        hyper_lambda = 0.01 if hinge_loss_mean > 0 else 0
        hyper_nu = 100 if tr_penalty > 0 else 0
        hyper_sai = 200 if neg_eig_penalty > 0 else 0
        return hyper_lambda * F.mse_loss(obj_priv, obj_raw) + hinge_loss_mean + hyper_nu * tr_penalty + hyper_sai * neg_eig_penalty








