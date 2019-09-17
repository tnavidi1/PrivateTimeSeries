import torch
import torch.nn.functional as F
from tqdm import tqdm
import processData
import nets
import numpy as np
import logging
import argparse
import os
import sys
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


# ==== printing config ==== #
desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150
torch.set_printoptions(profile="full", linewidth=400)

dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=1, bsz=64, split=0.15)

def diagnose_filter(generator, D_tilde, D, y_onehot, noise=None, k_iter=0, folder=None):
    bsz = D.shape[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print('folder {} exsits already!'.format(folder))

    with torch.no_grad():
        # print(D.shape)
        batched_concat_noise = torch.cat([noise, y_onehot], dim=1)
        G = generator.filter.fc.weight.data
        batched_filterred_noise = batched_concat_noise.mm(G)
        # print(batched_filterred_noise.shape)
        calculated_D_tilde = D+batched_filterred_noise
        idx = np.random.randint(0, bsz, size=6)

        fig, ax = plt.subplots(2, 2, figsize=(9.5, 6))
        ax[0, 0].set_title('raw noise')
        ax[0, 0].plot(batched_concat_noise.cpu().numpy()[idx].transpose())
        ax[0, 1].set_title('out noise')
        ax[0, 1].plot(batched_filterred_noise.cpu().numpy()[idx].transpose())

        ax[1, 0].set_title('raw and priv load (forward pass)')
        ax[1, 0].plot(D.cpu().numpy()[idx].transpose(), alpha=0.5 )
        ax[1, 0].plot(D_tilde.detach().cpu().numpy()[idx].transpose(), '--')
        ax[1, 1].set_title('raw and priv load (calculated)')
        ax[1, 1].plot(D.cpu().numpy()[idx].transpose(), alpha=0.5)
        ax[1, 1].plot(calculated_D_tilde.cpu().numpy()[idx].transpose(), '--')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'diagnostic_a_iter_%d.png' %k_iter))
        plt.close('all')

        plt.figure(figsize=(8,5))
        sns.heatmap(G.t().cpu().numpy(), cmap="RdBu")
        plt.ylim(len(G.t().cpu().numpy())-0.5, -0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'diagnostic_b_filterG_iter_%d.png' %k_iter))
        plt.close('all')


def diagnose_sol(batch_obj_raw, batch_obj_priv, batch_x_raw, batch_x_priv, k_iter=0, folder=None):
    T = 24
    # print(batch_obj_raw.shape, batch_obj_priv.shape) # torch size [bsz]
    # print((batch_obj_raw-batch_obj_priv).cpu().numpy())
    # print(batch_x_raw.shape, batch_x_priv.shape)  # torch size [bsz x 72]
    bsz = batch_obj_raw.shape[0]
    idx = np.random.randint(0, bsz, size=10)
    net_x_raw = batch_x_raw[idx, :T] - batch_x_raw[idx, T:2*T]
    net_x_priv = batch_x_priv[idx, :T] - batch_x_priv[idx, T:2*T]


    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    ax[0, 0].set_title('raw control')
    ax[0, 0].plot(net_x_raw.cpu().numpy().transpose())
    ax[1, 0].set_title('priv control')
    ax[1, 0].plot(net_x_priv.cpu().numpy().transpose())
    bar_width = 1
    ax[0, 1].set_title('obj vals')
    ax[0, 1].bar(np.arange(1, bsz+1)- bar_width/2, batch_obj_raw.cpu().numpy(), 0.75, label='raw_obj')
    ax[0, 1].bar(np.arange(1, bsz+1)+ 0.1, batch_obj_priv.cpu().numpy(), 0.75, label='priv_obj')
    ax[0, 1].legend()
    ax[1, 1].set_title('$L(d) - L(\hat{d})$ histogram')
    ax[1, 1].hist((batch_obj_raw - batch_obj_priv).cpu().numpy())
    plt.tight_layout()
    if folder is not None:
        plt.savefig(os.path.join(folder, 'diagnostic_c_sol_iter_%d.png'% k_iter ))
    plt.close('all')

def run_train(dataloader, params, p_opt='TOU', iter_max=500, lr=1e-3, xi=0.5,
              tradeoff1_=1, tradeoff2_ = 1, tradeoff3_=1, n_job=5, seed=1,
              load_pretrain='debug_diagnose/iter_0800.pth.tar'):

    _default_horizon_ = 24
    torch.manual_seed(seed)

    price = None
    if p_opt == 'TOU':
        price = bUtil.create_price(steps_perHr=1)
    elif p_opt == 'LMP':
        price = bUtil.create_LMP('../Data_LMP/8-18-2019_8days_Menlo_LMPs.csv', granular=24, steps_perHr=1)
    else:
        price = torch.rand((_default_horizon_, 1))  # price is a column vector

    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price)

    mask_mat = torch.cat([torch.eye(_default_horizon_), torch.ones((_default_horizon_, 2))], dim=1)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h=h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None, mask=mask_mat, n_job=n_job)

    clf = nets.Classifier(z_dim=24, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))
    lr_g = lr*90
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=lr_g, betas=(0.6, 0.999))
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=200, gamma=0.5)

    if load_pretrain is not None:
        bUtil.load_checkopint_gan(load_pretrain, g, clf, optimizer_g=optimizer_g, optimizer_clf=optimizer_clf)

    outter_j = 0
    p_ = 0.5
    prior_pi = torch.Tensor([[p_], [1 - p_]])  # shape : [2,1]
    is_best = False
    losses_gen = []
    losses_adv = []
    with tqdm(total=iter_max) as pbar:
        # with tqdm(dataloader) as pbar:
        while outter_j < iter_max:
            correct_cnt = 0
            tot_cnt = 0
            label_cnt1 = 0
            label_cnt2 = 0

            # raise NotImplementedError(len(dataloader))
            for k, (D, Y) in enumerate(dataloader):
                outter_j += 1

                optimizer_g.zero_grad()
                optimizer_clf.zero_grad()

                y_labels = Y
                y_onehot_target = bUtil.convert_onehot_soft(y_labels, alphabet_size=2)
                D_tilde, z_noise = g.forward(D, y_onehot_target)
                y_pred = clf(D_tilde)
                clf_loss = F.binary_cross_entropy_with_logits(y_pred, y_onehot_target)
                if outter_j % 2 == 0 and clf_loss.item() > 0.2:
                    clf_loss.backward(retain_graph=True)
                    optimizer_clf.step()

                loss_util_batch, util_grad, distort_ = g.util_loss(D, D_tilde, z_noise,
                                                                   y_onehot_target, prior=prior_pi)

                g_y_target = (1 - y_onehot_target)
                g_priv_loss = tradeoff1_ * F.binary_cross_entropy_with_logits(y_pred, g_y_target)
                g_distort_loss = tradeoff2_ * ((distort_ - xi) ** 2) + tradeoff3_ * (torch.clamp(distort_ - xi, min=0))
                g_loss = g_priv_loss + g_distort_loss

                loss_util = loss_util_batch.mean()
                g_loss.backward()
                # optimizer_g.step()
                scheduler_g.step()
                curr_lr_g = [param_group['lr'] for param_group in optimizer_g.param_groups]
                g.filter.fc.weight.data -= lr * util_grad.t()

                losses_adv.append(clf_loss.item())
                losses_gen.append(g_loss.item())

                batch_j_obj_raw, batch_j_obj_priv = g._objective_vals_getter()
                batch_j_x_raw, batch_j_x_priv = g._ctrl_decisions_getter()

                if outter_j % 10 == 0:
                    diagnose_sol(batch_j_obj_raw, batch_j_obj_priv, batch_j_x_raw, batch_j_x_priv,
                                 outter_j, folder='debug_diagnose_diagmask_s%d'%seed)


                # curr_lr_g = [param_group['lr'] for param_group in optimizer_g.param_groups]

                pbar.set_postfix(out_iter='{:d}'.format(outter_j), in_iter='{:d}'.format(k),
                                 clf_loss='{:.3e}'.format(clf_loss.item()),
                                 g_loss='{:.3e}'.format(g_loss.item()),
                                 util_loss='{:.3e}'.format(loss_util),
                                 ds='{:.3e}'.format(distort_.item()),
                                 cur_lr = '{:.3e}'.format(curr_lr_g[0]))
                pbar.update(1)

                if outter_j % 25 == 0:
                    diagnose_filter(generator=g, D_tilde=D_tilde, D=D, y_onehot=y_onehot_target, noise=z_noise,
                                    k_iter=outter_j, folder='debug_diagnose_diagmask_s%d'%seed)

                if outter_j % 50 == 0:

                    bUtil.save_checkpoint({'epoch': outter_j + 1,
                                           'g_state_dict': g.state_dict(),
                                           'g_optim_dict': optimizer_g.state_dict(),
                                           'clf_state_dict': clf.state_dict(),
                                           'clf_optim_dict': optimizer_clf.state_dict(),
                                           'loss_g': losses_gen,
                                           'loss_a': losses_adv,
                                           'obj_raw': batch_j_obj_raw,
                                           'obj_priv': batch_j_obj_priv},
                                          is_best=is_best,
                                          checkpoint='debug_diagnose_diagmask_s%d'%seed, filename='iter_%04d.pth.tar' % outter_j)

                if outter_j >= iter_max:
                    print("terminate!!")
                    return



params = dict(learning_rate=1e-3, batch_size=64,
              iter_max=1002, iter_save=50, num_workers=10,
              tradeoff_beta1 = 3, tradeoff_beta2 = 2,
              c_i=150, c_o=150, eta_eff=0.99,
              T=24, B=160,
              beta1=0.05, beta2=0.04, beta3=0.02,
              alpha=0.2)


#  lr = 5*1e-3
run_train(dataloader_dict['train'], params=params, p_opt='TOU', iter_max=1201, xi=30000, lr=1e-2,
          n_job=10, seed=2, load_pretrain='debug_diagnose_diagmask_s2/iter_0750.pth.tar')
