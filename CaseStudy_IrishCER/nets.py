import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import math
import matplotlib.pyplot as plt
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
            nn.Linear(z_dim, 52),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(52, 24),
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

    def __init__(self, in_features, out_features, bias=True, init_type="kaiming", mask=False):
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # if mask:
        #     num_row_diff = out_features - in_features
        #     self.mask_mat = torch.cat([
        #                         torch.cat([torch.eye(in_features), torch.zeros((num_row_diff, in_features))], dim=0),
        #                         torch.ones((out_features, num_row_diff))], dim=1)
        #     self.weight = self.weights * self.mask_mat
        #     self.handle = self.register_backward_hook(zero_grad)

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





#################################

#################################
# Define custome autograd function for masked connection.

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


################################
# To align with previous model conventions in class
class LinearFilter(nn.Module):

    def __init__(self, input_dim = 10, y_dim=0, output_dim=10, bias=None, mask=None, init_type="uniform"):
        super(LinearFilter, self).__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.output_dim = output_dim
        # self.fc = nn.Linear(self.input_dim + self.y_dim, self.output_dim, bias=bias)
        if mask is None:
            self.fc = PosLinear(self.input_dim + self.y_dim, self.output_dim, bias=bias, init_type=init_type)
        elif isinstance(mask, torch.Tensor):
            self.fc = CustomizedLinear(mask, bias=bias)
        else:
            raise NotImplementedError("mask value:{}".format(mask))

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
                 device=None, mask=None, n_job=11):
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
        self.n_job = n_job

        # create a linear filter # @since 2019/08/29 modify zero bias
        self.has_mask = 1 if isinstance(mask, torch.Tensor) else 0
        self.filter = LinearFilter(self.z_dim, self.y_priv_dim, output_dim=self.z_dim, bias=None, mask=mask, init_type="kaiming")  # setting noise dim is same as latent dim
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

        # x_priv = F.softplus(x + x_proc_noise)
        x_priv = torch.clamp(x + x_proc_noise, min=0)
        ###########################
        # print(x_proc_noise.shape)
        # fig, ax = plt.subplots(1, 4)
        # ax[0].plot(x_proc_noise[:3].detach().cpu().numpy().transpose())
        # ax[0].plot((x + x_proc_noise)[:3].detach().cpu().numpy().transpose(), 'o-')
        #
        # ax[1].plot(x_priv[:3].detach().cpu().numpy().transpose(), '--')
        # ax[1].plot(x[:3].detach().cpu().numpy().transpose())
        #
        # ax[2].plot(x_priv[:3].detach().cpu().numpy().transpose(), '--')
        # ax[2].plot((x + x_proc_noise)[:3].detach().cpu().numpy().transpose())
        #
        # ax[3].plot(x_priv[:3].detach().cpu().numpy().transpose(), '--')
        # ax[3].plot((F.softplus(x + x_proc_noise))[:3].detach().cpu().numpy().transpose(), alpha=0.2)
        # ax[3].legend()
        # plt.show()

        return x_priv, z_noise

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                self.z_prior[1].expand(batch, self.z_dim))

    def display_filter_params(self):
        params = list(self.filter.parameters())
        print(params[0].size())
        print(params[0])

    def solve_convex_forumation(self, p, D, Q, G, h, A, b, Y_onehot=None, n_job=10):
        batch_size = D.shape[0]
        D_ = D
        # if isinstance(Y_onehot, torch.Tensor):
        #     # don't do this due to the randomness in forward function
        #     D_, z_noise = self.forward(D, Y_onehot)
        # elif Y_onehot is None:
        #     D_ = D  # assign input D to D_; for example D is privatized demand
        # else:
        #     raise NotImplementedError("Y_onehot:\n {} \n is not supportted".format(Y_onehot))

        def __expand_param_list(x):
            return [ut.to_np(x) for i in range(batch_size)]

        Qs, Gs, hs, As, bs = list(map(__expand_param_list, [Q, G, h, A, b])) # to numpy
        p = ut.to_np(p)
        D_detached = ut.to_np(D_.detach())
        ### this call numpy data,
        ### this call cvx solver
        # res = OptMini_cvx.forward_D_batch(Qs, Gs, hs, As, bs, D_detached, self.T, p=p, n_jobs=n_job)
        # x_sols = bUtil._convert_to_np_arr(res, 1)
        # objs = bUtil._convert_to_np_scalars(res, 0)

        ### this call conic auto projection
        res = OptMini_cvx.forward_conic_D_batch(Qs, Gs, hs, As, bs, D_detached, self.T, p=p, n_jobs=n_job)

        x_sols = bUtil._convert_to_np_arr(res, 0)
        # diff_D = -bUtil._convert_to_np_arr(res, 1)  # it's minus d (neg demand )
        diff_D = bUtil._convert_to_np_arr(res, 1)  # it's minus d (neg demand )
        return diff_D, x_sols

    def evaluate_cost_obj(self, x_sols, D_, p=None):
        # D_ = self.cached_D_priv
        # D_ is a privatized demand
        """
        inputs has torch.tensor format
        :param x_sols:
        :param D_:
        :param p:
        :return:
        """
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


    def _check_values(self, a, b):
        raise NotImplementedError(a, b)

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



    def util_loss(self, D, D_priv, z_noise, Y_onehot, p=None, xi=0.01, prior=None):
        if p is None:
            p = self.p

        # D_priv, z_noise = self.forward(D, Y_onehot)

        # obj_raw, x_sol_raw = self.solve_convex_forumation(p, D, self.Q, self.G, self.h, self.A, self.b, Y_onehot=None, n_job=self.n_job)
        # obj_priv, x_sol_priv = self.solve_convex_forumation(p, D_priv, self.Q, self.G, self.h, self.A, self.b, Y_onehot=None, n_job=self.n_job)
        #
        # convert obj_raw, obj_priv as tensor
        # [obj_raw, obj_priv, x_sol_raw, x_sol_priv] = [torch.from_numpy(x).to(torch.float) for x in \
        #                                               [obj_raw, obj_priv, x_sol_raw, x_sol_priv]]

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


        # raise NotImplementedError(x_sol_raw.shape, x_sol_priv.shape, d_Xd.shape, d_Xd_priv.shape)
        # obj_priv = self.evaluate_cost_obj(x_sol_priv, D, Y_onehot, p=p)
        # obj_priv = self.evaluate_cost_obj(x_sol_priv, D_=D_priv, p=p)
        obj_priv = self.evaluate_cost_obj(x_sol_priv, D_=D, p=p)
        obj_raw = self.evaluate_cost_obj(x_sol_raw, D_=D, p=p)

        # self._objective_vals_setter(obj_raw, obj_priv)

        self._objective_vals_setter(obj_raw, obj_priv)
        self._ctrl_decisions_setter(x_sol_raw, x_sol_priv)

        # size_of_tr = (torch.trace(torch.mm(self.filter.fc.weight, self.filter.fc.weight.t())) - xi).size()
        # raise NotImplementedError(size_of_tr, torch.zeros(size=size_of_tr))
        # ------ huber loss ------
        # tr_penalty = F.smooth_l1_loss(torch.trace(torch.mm(self.filter.fc.weight, self.filter.fc.weight.t())) - xi,
        #                               torch.zeros(size=size_of_tr))
        # tr_penalty = F.l1_loss(torch.trace(torch.mm(self.filter.fc.weight, self.filter.fc.weight.t())) - xi,
        #                               torch.zeros(size=size_of_tr))

        w_r, w_c = self.filter.fc.weight.shape  # decouple the weight matrix rows and columns
        GAMMA = self.filter.fc.weight[:, :self.T] if w_r == self.T else self.filter.fc.weight[:self.T, :]
        bias_vec = self.filter.fc.weight[:, self.T:].mm(prior) if w_r == self.T else (self.filter.fc.weight[self.T:, :].t()).mm(
            prior)
        distortion = torch.norm(GAMMA, p='fro') ** 2 + torch.norm(bias_vec, p=2) ** 2

        return obj_priv, grad, distortion
        ################################################
        # hinge_loss_mean = torch.clamp(obj_priv - obj_raw, min=0).mean()

        # neg_tr_penalty = F.relu(-torch.symeig(self.filter.fc.weight)).sum(0)
        # eigvals, eig_vecs = torch.symeig(self.filter.fc.weight.data[:, :48])

        # diagvals = torch.diag(self.filter.fc.weight.data[:, :48])

        # min_diag_vals =  torch.min(diagvals)
        # neg_diag_penalty = F.relu_(-min_diag_vals)
        # m = nn.utils.spectral_norm(self.filter.fc.weight.data[:, :48])


        ##########################################
        # tr_penalty = F.relu_(torch.trace(torch.mm(self.filter.fc.weight, self.filter.fc.weight.t())) - xi)
        # hyper_1 = 1 if hinge_loss_mean > 0.5 else 0.5
        # hyper_2 = 10 if tr_penalty > 1e-3 else 0
        # # hyper_3 = 1 if neg_diag_penalty > 1e-3 else 0
        # return hyper_1 * F.mse_loss(obj_priv, obj_raw) + hinge_loss_mean + hyper_2 * tr_penalty #+ 0.01* m #+ hyper_3 * neg_diag_penalty








