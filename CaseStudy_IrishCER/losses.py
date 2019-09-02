"""
@Description: This scripts contains loss functions
"""

import torch.nn.functional as F

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
    return p.bmm(F.relu_((x + D).unsqueeze(2))).squeeze(1).squeeze(1)

def objective_task_loss(price, x_ctrl, D, Q, T):
    # eval the objective
    # raise NotImplementedError(x_ctrl.shape)
    batch_size = D.shape[0]
    x_in = x_ctrl[:, :T]
    x_out = x_ctrl[:, T:2 * T]
    x_s = x_ctrl[:, 2 * T:]

    # raise NotImplementedError(x_in.shape, x_out.shape, x_s.shape) # batchsize x 48
    # print(x_ctrl.unsqueeze(1).shape)
    # print(Q.shape)
    # raise NotImplementedError
    Q = Q.unsqueeze(0).repeat(batch_size, 1, 1)
    # raise NotImplementedError(Q.shape)
    price = price.t().unsqueeze(0).repeat(batch_size, 1, 1)
    cost_quad_xQx = torch_bQuad(x_ctrl, Q)
    # print(price.shape, price)
    # print((x_in - x_out).shape)
    cost_lin_qx = torch_bLin(x_in-x_out, D, price)
    # print(price.shape, price)
    # print(cost_quad_xQx)
    # print(cost_lin_qx)
    # raise NotImplementedError
    return 0.5 * cost_quad_xQx + cost_lin_qx