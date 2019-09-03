import torch
import torch.nn.functional as F
from tqdm import tqdm
import processData
import nets
import numpy as np
import sys, os
sys.path.append("..")

import time
import cvxpy as cp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set('paper', style="whitegrid", font_scale=1.5, rc={"lines.linewidth": 2.5}, )


import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib
import basic_util as bUtil


desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=32)



def _form_QP_params(param_set, p=None):
    """

    :param param_set:
    :param p: price
    :return:
    """
    if not isinstance(param_set, dict):
        raise NotImplementedError("wrong type of param set: {}".format( param_set))

    c_i = param_set['c_i']
    c_o = param_set['c_o']
    eta_eff = param_set['eta_eff']
    beta1 = param_set['beta1']
    beta2 = param_set['beta2']
    beta3 = param_set['beta3']
    alpha = param_set['alpha']
    B = param_set['B']
    T = param_set['T']

    G = optMini_util.construct_G_batt_raw(T)
    h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    b = optMini_util.construct_b_batt_raw(T, batt_init=B / 2)
    Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, beta3=beta3)
    q, price = optMini_util.construct_q_batt_raw(T, price=p, batt_B=B, beta3=beta3, alpha=alpha)

    return [Q, q, G, h, A, b, T, price]

def _extract_filter_weight(x):
    return optMini_util.to_np(x.data)



def run_battery(dataloader, params=None, lr=1e-3):
    ## multiple iterations
    # init price

    _default_horizon_ = 48
    torch.manual_seed(2)
    # price = torch.rand((_default_horizon_, 1))  # price is a column vector
    price = bUtil.create_price()
    Q, q, G, h, A, b, T, price = _form_QP_params(params, p=price)
    # controller = OptPrivModel(Q, q, G, h, A, b, T=T)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h = h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None)

    # print(g)
    clf = nets.Classifier(z_dim=48, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))

    # raise NotImplementedError
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=2*lr)
    # raise NotImplementedError(*g.filter.parameters())
    with tqdm(dataloader) as pbar:
        correct_cnt = 0
        tot_cnt = 0
        label_cnt1 = 0
        label_cnt2 = 0

        for k, (D, Y) in enumerate(pbar):
            # controller(D)
            optimizer_g.zero_grad()
            optimizer_clf.zero_grad()
            y_labels = bUtil.convert_binary_label(Y, 1500) # row vector
            y_onehot = bUtil.convert_onehot(y_labels.unsqueeze(1), alphabet_size=2)
            # print(D, y_labels, y_onehot)
            D_tilde, z_noise = g.forward(D, y_onehot)

            y_out = clf(D_tilde)
            loss_priv = F.cross_entropy(y_out, y_labels, weight=None,
                                       ignore_index=-100, reduction='mean')
            loss_util = g.util_loss(D, y_onehot, xi=1)

            if k % 50 == 0:
            #     print(g.filter.fc.weight.data.shape)
                print(g.filter.fc.weight.data)
                plt.figure(figsize=(6, 5))
                sns.heatmap(g.filter.fc.weight.data.cpu().numpy())
                plt.title("iter==%d" % k)
                plt.tight_layout()
                plt.savefig('../fig/filter_visual/f_weight_%d.png' % k )
                plt.close()

                fig, ax=plt.subplots(3,1, figsize=(6.5, 10))
                i=np.random.randint(0, 32)
                ax[0].plot(D.cpu().numpy().transpose())
                ax[1].plot(D_tilde.detach().cpu().numpy().transpose())
                ax[2].plot(np.vstack((D[i].cpu().numpy(), D_tilde[i].detach().cpu().numpy())).transpose(), )
                ax[0].set_title("raw demand")
                ax[1].set_title("priv demand")
                ax[2].set_title("sampled demand plot")
                ax[2].legend(['raw', 'priv'])
                plt.tight_layout()
                plt.savefig('../fig/demand_visual/iter_%d.png'%k)
                plt.close()

            #     print(torch.trace(torch.mm(g.filter.fc.weight.data, g.filter.fc.weight.data.t())))

            loss_priv.backward(retain_graph=True)
            optimizer_clf.step()

            g_loss = loss_util - loss_priv
            g_loss.backward()
            optimizer_g.step()

            #
            _, y_max_idx = torch.max(y_out, dim=1)
            correct = y_max_idx == y_labels
            correct_cnt += correct.sum()
            tot_cnt += D.shape[0]
            label_cnt1 += (y_labels == 0).sum()
            label_cnt2 += (y_labels == 1).sum()

            trace_track = torch.trace(torch.mm(g.filter.fc.weight.data, g.filter.fc.weight.data.t()))
            pbar.set_postfix(iter='{:d}'.format(k), g_loss='{:.3e}'.format(g_loss),
                             util_loss = '{:.3e}'.format(loss_util),
                             priv_loss='{:.3e}'.format(loss_priv),
                             # cor_cnts='{:d}'.format(correct_cnt),
                             # tot_cnts='{:d}'.format(tot_cnt),
                             acc='{:.2e}'.format(float(correct_cnt) / tot_cnt),
                             prop1='{:.2e}'.format(float(label_cnt1) / tot_cnt),
                             prop2='{:.2e}'.format(float(label_cnt2) / tot_cnt),
                             tr='{:.3e}'.format(trace_track)
                             )
            pbar.update(10)

            ###############################################################
            # print(g.filter.fc.weight.shape) # 48 * 50
            # d = D_tilde[0]
            # eps = z_noise[0]
            # y_onehot_ = y_onehot[0]
            # # GAMMA = _extract_filter_weight(g.filter.fc.weight)
            #
            # GAMMA = g.filter.fc.weight
            #
            # # raise NotImplementedError("=========")
            # [price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b] = list(map(_extract_filter_weight,
            #                                                             [price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b]))
            # # ==== cvx ====
            # # x_ctrl = optMini_cvx._convex_formulation_w_GAMMA_d_cvx(price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b, T,
            # #                                                        sol_opt=cp.GUROBI, verbose=True)
            # #
            # # # print(x_ctrl[:T] - x_ctrl[T:(2 * T)])
            # # fig, ax =plt.subplots(2, 1, figsize=(6, 4))
            # # ax[0].bar(np.arange(1, T + 1), x_ctrl[:T] - x_ctrl[T:(2 * T)])
            # # ax[0].bar(np.arange(1, T + 1), price.flatten())
            # # plt.show()
            # # ==== conic ====
            # x_ctrl, db_ = optMini_cvx._convex_formulation_w_GAMMA_d_conic(price, GAMMA, d, eps, y_onehot_, Q, G, h, A, b, T,
            #                                                          sol_opt=cp.GUROBI, verbose=True)
            # print(np.expand_dims(db_[T:(2*T)], 1), eps)
            #
            # print(np.tile(np.expand_dims(db_[T:(2*T)], 1), T))
            # grad_gamma = ((1/T) * (np.tile(np.expand_dims(db_[T:(2*T)], 1), T)) / eps )
            # print(grad_gamma.shape, grad_gamma)
            # # print(x_ctrl[:T] - x_ctrl[T:(2*T)])
            # # plt.figure(figsize=(6, 4))
            # # ax[1].bar(np.arange(1, T+1), x_ctrl[:T] - x_ctrl[T:(2*T)])
            # # ax[1].bar(np.arange(1, T+1), price.flatten())
            # # plt.show()



            # print(x_ctrl)
            # if (k + 1) > 400:
            #     raise NotImplementedError("manual break!")


params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, beta3=0.5, alpha=0.2)
run_battery(dataloader_dict['train'], params=params)

