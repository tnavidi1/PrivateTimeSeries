# ============================= #
# @auther markcx'At'stanford.edu
# ============================= #

import torch
from torch import nn
import math

import sys
sys.path.append("..")

try:
    import OptMiniModule.util as ut
    import OptMiniModule.cvx_runpass as OptMini_cvx
except:
    raise FileNotFoundError("== Opt module util import error! ==")

try:
    import basic_util as bUtil
    import losses as bLosses
except:
    raise ModuleNotFoundError("== basic_util file is not imported! ==")

# ===========================
#  neural network models
# ===========================

class Classifier(nn.Module):
    def __init__(self, z_dim=10, y_dim=0):
        super(Classifier, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 15),
            nn.ELU(),
            nn.Linear(15, y_dim)
            # nn.Linear(z_dim, y_dim),
        )

    def forward(self, x):
        o = self.net(x)
        return o


class ClassifierLinear(nn.Module):
    def __init__(self, z_dim=10, y_dim=0):
        super(ClassifierLinear, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, y_dim),
            # nn.ELU(),
            # nn.Linear(15, y_dim)
            # nn.Linear(z_dim, y_dim),
        )

    def forward(self, x):
        o = self.net(x)
        return o



class LinearFilter(nn.Module):
    def __init__(self, input_dim = 10, y_dim=0, output_dim=10, bias=None, mask=None):
        super(LinearFilter, self).__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        if mask is None:
            self.fc = nn.Linear(self.input_dim + self.y_dim, self.output_dim,  bias=bias)
        elif isinstance(mask, torch.Tensor):
            self.fc = CustomizedLinear(mask, bias=bias)
        else:
            raise NotImplementedError("support linear layer with masked entries, "
                                      "not ready for {}".format(mask))

    def forward(self, x, y=None):
        """

        :param x: input the feature vectors
        :param y: the default y input would be one-hot encoding version of discrete labels
        :return:
        """
        xy = x if y is None else torch.cat([x, y], dim=1)
        o = self.fc(xy)
        return o


#################################
## A new diag module framework ##
#################################
# Define customized autograd function for masked connection.
# https://github.com/uchida-takumi/CustomizedLinear

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            # print("forward mask : {}".format(mask.shape))
            weight = weight * mask
        # output = input.mm(weight.t())
        output = input.mm(weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            # print(grad_output, grad_output.shape)
            # print("=="*20)
            # print(input, input.shape)
            # raise NotImplementedError
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                # raise NotImplementedError(grad_weight.shape, mask.shape)
                grad_weight = grad_weight.t() * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
            # self.mask = mask.type(torch.float)
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()
            # self.mask = torch.tensor(mask, dtype=torch.float)

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))
        # self.weight = nn.Parameter(torch.Tensor(self.input_features, self.output_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )






class Generator(nn.Module):
    def __init__(self, nn='v1', name='g_filter', z_dim=24, y_priv_dim=2,
                 Q=None, G=None, h=None, A=None, b=None, T=24, p=None, mask=None,
                 device=None, n_job=1):

        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.y_priv_dim = y_priv_dim
        self._set_convex_prob_param(p, Q, G, h, A, b, T)
        self.device = device
        self.n_job = n_job
        self.has_mask = 1 if isinstance(mask, torch.Tensor) else 0
        # create a linear filter
        self.filter = LinearFilter(self.z_dim, self.y_priv_dim, output_dim=self.z_dim, mask=mask, bias=None)  # setting noise dim is same as latent dim

        # Set prior as fixed parameter attached to module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
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
        batch_size = x.size()[0] # batch size is the first dimension

        z_noise = self.sample_z(batch=batch_size)
        z_noise = z_noise / z_noise.norm(2, dim=1).unsqueeze(1).repeat(1, self.z_dim)

        x_proc_noise = self.filter(z_noise, y)
        x_noise = x + x_proc_noise
        # x_noise = torch.clamp(x_noise, min=0)
        return x_noise, z_noise

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                self.z_prior[1].expand(batch, self.z_dim))

    def solve_convex_forumation(self, p, D, Q, G, h, A, b, Y_onehot=None, n_job=10):
        batch_size = D.shape[0]
        D_ = D

        def __expand_param_list(x):
            return [ut.to_np(x) for i in range(batch_size)]

        Qs, Gs, hs, As, bs = list(map(__expand_param_list, [Q, G, h, A, b]))  # to numpy
        p = ut.to_np(p)
        D_detached = ut.to_np(D_.detach())

        res = OptMini_cvx.forward_conic_D_batch(Qs, Gs, hs, As, bs, D_detached, self.T, p=p, n_jobs=n_job)

        x_sols = bUtil._convert_to_np_arr(res, 0)
        # diff_D = -bUtil._convert_to_np_arr(res, 1)  # it's minus d (neg demand )
        diff_D = bUtil._convert_to_np_arr(res, 1)  # it's minus d (neg demand )
        # raise NotImplementedError
        return diff_D, x_sols

    def evaluate_cost_obj(self, x_sols, D_, p=None):
        # Q = self.Q
        T = self.T
        # return bLosses.objective_task_loss(p, x_sols, D_, Q, T)
        return bLosses.objective_task_loss_linear(p, x_sols, D_, T)

    def evaluate_cost_grad(self, x_sol, D, p=None, dD=None, cat_noise=None):
        Q = self.Q
        T = self.T
        # bLosses.grad_dl_dx(p, x_sol, D, Q, T)
        # bLosses.grad_loss_dx_dD_eps(dD, cat_noise, T) # p, x_sol, D, Q, dD, cat_noise, T
        avg_grad = bLosses.grad_dldxdD(p, x_sol, D, Q, dD, cat_noise, T)
        return avg_grad

    def evaluate_cost_grad_diag(self, x_sol, D, p=None, dD=None, cat_noise=None):
        # print(cat_noise.shape)
        Q = self.Q
        T = self.T
        # raise NotImplementedError(bLosses.grad_dldxdD_diag(p, x_sol, D, Q, dD, cat_noise, T))
        return bLosses.grad_dldxdD_diag(p, x_sol, D, Q, dD, cat_noise, T)


    def _objective_vals_setter(self, obj_raw, obj_priv):
        self.obj_raw = obj_raw
        self.obj_priv = obj_priv

    def _objective_vals_getter(self):
        return self.obj_raw, self.obj_priv

    def _ctrl_decisions_setter(self, x_raw_ctrl, x_priv_ctrl):
        self.x_raw_ctrl = x_raw_ctrl
        self.x_priv_ctrl = x_priv_ctrl

    def _ctrl_decisions_getter(self):
        return self.x_raw_ctrl, self.x_priv_ctrl

    def util_loss(self, D, D_priv, z_noise, Y_onehot, p=None, prior=None):
        if p is None:
            p = self.p

        d_Xd, x_sol_raw = self.solve_convex_forumation(p, D, self.Q, self.G, self.h, self.A, self.b, Y_onehot=None,
                                                       n_job=self.n_job)
        d_Xd_priv, x_sol_priv = self.solve_convex_forumation(p, D_priv, self.Q, self.G, self.h, self.A, self.b, Y_onehot=None,
                                                             n_job=self.n_job)

        [d_Xd_priv, x_sol_raw, x_sol_priv] = [torch.from_numpy(x).to(torch.float) for x in \
                                              [d_Xd_priv, x_sol_raw, x_sol_priv]]

        cat_noise_ = torch.cat([z_noise, Y_onehot], dim=1)
        if self.has_mask == 0:
            grad = self.evaluate_cost_grad(x_sol_priv, D, p, d_Xd_priv, cat_noise_)
        elif self.has_mask == 1:
            grad = self.evaluate_cost_grad_diag(x_sol_priv, D, p, dD=d_Xd_priv, cat_noise=cat_noise_)
        else:
            raise NotImplementedError("NOT supported for the value {}".format(self.has_mask))


        obj_priv = self.evaluate_cost_obj(x_sol_priv, D_=D, p=p)
        obj_raw = self.evaluate_cost_obj(x_sol_raw, D_=D, p=p)

        self._objective_vals_setter(obj_raw, obj_priv)
        self._ctrl_decisions_setter(x_sol_raw, x_sol_priv)

        # size_of_tr = (torch.trace(torch.mm(self.filter.fc.weight, self.filter.fc.weight.t())) - xi).size()

        w_r, w_c = self.filter.fc.weight.shape  # decouple the weight matrix rows and columns
        GAMMA = self.filter.fc.weight[:, :self.T] if w_r == self.T else self.filter.fc.weight[:self.T, :]
        bias_vec = self.filter.fc.weight[:, self.T:].mm(prior) if w_r == self.T else (self.filter.fc.weight[self.T:, :].t()).mm(prior)
        distortion = torch.norm(GAMMA, p='fro')**2 + torch.norm(bias_vec, p=2)**2

        return obj_priv, grad, distortion
