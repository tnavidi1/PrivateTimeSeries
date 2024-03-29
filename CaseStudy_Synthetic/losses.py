"""
@Description: This scripts contains loss functions
"""

import torch.nn.functional as F
import torch

# params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=2, B=1.5, beta1=0.6, beta2=0.4, gamma=0.5, alpha=0.2)

def objective_util_loss_no_demand(ctrl, price, params):
    T = params['T']
    x_in = ctrl[:, :T].float()
    x_out = ctrl[:, T:2*T].float()
    x_s = ctrl[:, 2*T:].float()
    costs = (
        (x_in - x_out) * price) + \
         (params['beta3'] * (x_s - params['B']*params['alpha'])**2) + \
         (params['beta1'] * x_in**2) + \
        (params['beta2'] * x_out**2)
    return costs.mean(dim=0)


def rmse_loss(Y_pred, Y_actual):
    return ((Y_pred - Y_actual)**2).mean(dim=0).sqrt().data


def objective_util_loss_w_demand(ctrl, p, D, params):
    T = params['T']
    x_in = ctrl[:, :T].float()
    x_out = ctrl[:, T:2 * T].float()
    x_s = ctrl[:, 2 * T:].float()
    cost = F.relu(x_in - x_out + D) * p + \
            (params['beta3'] * (x_s - params['B']*params['alpha'])**2) + \
            (params['beta1'] * x_in**2) + (params['beta2'] * x_out**2)
    return cost.mean(dim=0)


def torch_bQuad(x, Q):
    # batch Quad_form
    # multiple   x^T Q x
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)

def torch_bLin(x, D, p):
    return p.bmm(torch.clamp((x + D).unsqueeze(2), min=0)).squeeze(1).squeeze(1)

def objective_task_loss(price, x_ctrl, D, Q, T):
    # eval the objective
    # raise NotImplementedError(x_ctrl.shape)
    batch_size = D.shape[0]
    x_in = x_ctrl[:, :T]
    x_out = x_ctrl[:, T:(2 * T)]
    x_s = x_ctrl[:, (2 * T):]
    # raise NotImplementedError(x_in.shape, x_out.shape, x_s.shape) # batchsize x 48
    Q = Q.unsqueeze(0).repeat(batch_size, 1, 1)
    price = price.t().unsqueeze(0).repeat(batch_size, 1, 1)
    cost_quad_xQx = torch_bQuad(x_ctrl, Q)

    cost_lin_qx = torch_bLin(x_in-x_out, D, price)
    return 0.5 * cost_quad_xQx + cost_lin_qx


def objective_task_loss_linear(price, x_ctrl, D, T):
    batch_size = D.shape[0]
    price = price.t().unsqueeze(0).repeat(batch_size, 1, 1)
    x_in = x_ctrl[:, :T]
    x_out = x_ctrl[:, T:(2 * T)]
    cost_lin_qx = torch_bLin(x_in - x_out, D, price)
    return cost_lin_qx

def grad_dl_dx(price, x_ctrl, D, Q, T):
    """

    :param price:
    :param x_ctrl:
    :param Q:
    :param T:
    :return:
    """
    bsz = D.shape[0]
    x_in = x_ctrl[:, :T]
    x_out = x_ctrl[:, T:(2 * T)]
    # print(((x_in - x_out) + D).shape)  # bsz x T
    clipped_mat = ((x_in - x_out) + D).clamp(min=0)
    nonzero_indices = clipped_mat.nonzero()
    size_ones = nonzero_indices.size(0)
    ones_v = torch.ones(size_ones).float()
    mask_ = (torch.sparse.FloatTensor(nonzero_indices.t(), ones_v, clipped_mat.size()).to_dense())
    price = (price.squeeze(1)).expand(bsz, T) # batchsize x T
    dldx = mask_ * price
    # print(dldx.shape) # bs x T
    return dldx

def grad_dx_dD_eps(dD, concat_noise, T):
    bsz = dD.shape[0]
    cat_nz = concat_noise.shape[1]
    batched_mat_dD = (dD.reshape(bsz, T, 1)).expand((bsz, T, cat_nz))
    # print(concat_noise.shape)  # bs x 50
    # print(batched_mat_dD, batched_mat_dD.shape)  #bs x 48 x 50
    # ==> 48 x 50 matrix
    concat_noise = concat_noise.unsqueeze(1).expand((bsz, T, cat_nz))

    batch_grad_dx_dD_eps =  (batched_mat_dD / concat_noise) * 1/ ( cat_nz)
    return batch_grad_dx_dD_eps
    # avg_grad = batch_grad_dx_dD_eps.sum(0) / bsz
    # max_clip = 1
    # avg_grad.clamp_(-max_clip, max_clip) #  48 x 50
    # return avg_grad

def grad_dldxdD(p, x_sol, D, Q, dD, cat_noise, T):
    """
    This function
    :param p:
    :param x_sol:
    :param D:
    :param Q:
    :param dD:
    :param cat_noise:
    :param T:
    :return:
    """
    bsz = dD.shape[0]
    cat_nz_dim = cat_noise.shape[1]
    b_dldx = grad_dl_dx(p, x_sol, D, Q, T)
    b_dxdD = grad_dx_dD_eps(dD, cat_noise, T)

    b_grad = b_dldx.unsqueeze(2).expand((bsz, T, cat_nz_dim)) * (b_dxdD)
    # max_clip = 1.5
    # b_grad.clamp_(-max_clip, max_clip)
    # print(b_grad.shape)
    avg_grad = b_grad.sum(0) / bsz
    avg_grad = avg_grad / torch.max(torch.norm(avg_grad, p="fro"), torch.tensor(1e-8, dtype=torch.float))
    # avg_grad.clamp_(-max_clip, max_clip)
    # print(avg_grad)
    # raise NotImplementedError(avg_grad)
    return avg_grad


def grad_dx_dD_eps_diag(dD, concat_noise, T):
    bsz = dD.shape[0]
    cat_nz = concat_noise.shape[1]
    label_cat = cat_nz - T
    noise_ = concat_noise[:, :T]
    label_onehot = concat_noise[:, T:]
    # print("noise shape {}, dD shape{}".format(noise_.shape, dD.shape))
    d_diagGAMMA= dD / noise_
    # raise NotImplementedError(dGAMMA.shape, label_onehot.shape)
    batched_dD = dD.unsqueeze(2).expand((bsz, T, label_cat))
    batched_dD = batched_dD / label_onehot.unsqueeze(1)
    d_diagGAMMA = torch.diag_embed(d_diagGAMMA)
    batched_grad_dxdD_eps = torch.cat([d_diagGAMMA, batched_dD], dim=2)
    # raise NotImplementedError(batched_dD, d_diagGAMMA[0], d_diagGAMMA.shape, batched_dD.shape, label_onehot.shape)
    # raise NotImplementedError(batched_grad_dxdD_eps.shape)
    return batched_grad_dxdD_eps

def grad_dldxdD_diag(p, x_sol, D, Q, dD, cat_noise, T):
    bsz = dD.shape[0]
    cat_nz_dim = cat_noise.shape[1]
    b_dldx = grad_dl_dx(p, x_sol, D, Q, T)
    b_dxdD = grad_dx_dD_eps_diag(dD, cat_noise, T)

    b_grad = b_dldx.unsqueeze(2).expand((bsz, T, cat_nz_dim)) * (b_dxdD)
    # max_clip = 1.5
    # b_grad.clamp_(-max_clip, max_clip)
    # print(b_grad.shape)
    avg_grad = b_grad.sum(0) / bsz
    avg_grad = avg_grad / torch.max(torch.norm(avg_grad, p="fro"), torch.tensor(1e-8, dtype=torch.float))
    return avg_grad

