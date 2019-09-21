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



def diagnoise_plot_demand(D1, D2, desc, fpath, iter=1):
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        print("----- create folder {}".format(fpath))
    # else:
    #     print('folder {} exsits already!'.format(fpath))


    plt.figure(figsize=(8, 6))
    plt.title(desc)
    plt.plot(D1.detach().t().cpu().numpy(), 'b-', alpha=0.4, label='+1')
    plt.plot(D2.detach().t().cpu().numpy(), 'g-.', alpha=0.5, label='-1')
    plt.tight_layout()
    plt.savefig(os.path.join(fpath, 'debug_%s_iter_%04d.png' % (desc, iter)))
    plt.close('all')


def diagnose_filter(generator, D_tilde, D, y_onehot, noise=None, k_iter=0, folder=None):
    bsz = D.shape[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("----- create folder {}".format(folder))
    # else:
        # print('folder {} exsits already!'.format(folder))

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


def diagnose_sol(batch_obj_raw, batch_obj_priv, batch_x_raw, batch_x_priv, D_raw, D_tilde, T = 24, k_iter=0, folder=None, sample_size=2):

    # print(batch_obj_raw.shape, batch_obj_priv.shape) # torch size [bsz]
    # print((batch_obj_raw-batch_obj_priv).cpu().numpy())
    # print(batch_x_raw.shape, batch_x_priv.shape)  # torch size [bsz x 72]
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("----- create folder {}".format(folder))
    # else:
    #     print('folder {} exsits already!'.format(folder))


    bsz = batch_obj_raw.shape[0]
    idx = np.random.randint(0, bsz, size=sample_size)
    net_x_raw = batch_x_raw[idx, :T] - batch_x_raw[idx, T:2*T]
    net_x_priv = batch_x_priv[idx, :T] - batch_x_priv[idx, T:2*T]
    sel_D_r = D_raw[idx, :]
    sel_D_p = D_tilde[idx, :].detach()

    net_L_r = net_x_raw + sel_D_r
    net_L_p = net_x_priv + sel_D_p

    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    ax[0, 0].set_title('raw control')
    ax[0, 0].plot(net_x_raw.cpu().numpy().transpose(), 'o-')
    ax[0, 0].plot(sel_D_r.cpu().numpy().transpose(), '*--')
    ax[0, 0].plot(net_L_r.cpu().numpy().transpose(),  '-.', linewidth=2.5,)
    ax[1, 0].set_title('priv control')
    ax[1, 0].plot(net_x_priv.cpu().numpy().transpose(), '^-')
    ax[1, 0].plot(sel_D_p.cpu().numpy().transpose(), 's--')
    ax[1, 0].plot(net_L_p.cpu().numpy().transpose(), '-.', linewidth=2.5, )
    bar_width = 1
    ax[0, 1].set_title('obj vals')
    ax[0, 1].bar(np.arange(1, bsz+1)- bar_width/2, batch_obj_raw.cpu().numpy(), 0.75, label='raw_obj')
    ax[0, 1].bar(np.arange(1, bsz+1)+ 0.1, batch_obj_priv.cpu().numpy(), 0.75, label='priv_obj')
    ax[0, 1].legend()
    ax[1, 1].set_title('$L(d) - L(\hat{d})$ histogram')
    ax[1, 1].hist((batch_obj_raw - batch_obj_priv).cpu().numpy())
    plt.tight_layout()
    if folder is not None:
        plt.savefig(os.path.join(folder, 'diagnostic_c_sol_iter_%d_s%d.png'% (k_iter, sample_size) ))
    plt.close('all')



def run_check(dataloader_train):

    for k, (D, Y) in enumerate(dataloader_train):
        max_val, max_idx = torch.max(D, dim=1)
        print(max_idx)
        plt.hist(max_idx.cpu().numpy())
        plt.show()
        raise NotImplementedError

def run_train(dataloader_train, dataloader_test, params, xi=1, iter_max=8000, iter_save=50, iter_dig=20,
              lr=1e-3, tradeoff_beta1=1, tradeoff_beta2=1, tradeoff_beta3=1,
              save_folder = None, reload_pretrain_folder=None, reload_step=0,
              seed=1, n_job=10):
    price = bUtil.create_simple_price_TOU(horizon=6, t1=3, t2=5, steps_perHr=1)
    # price = bUtil.create_price(steps_perHr=1)
    # raise NotImplementedError(price)
    _default_horizon_ = 6
    torch.manual_seed(seed)



    clf = nets.ClassifierLinear(z_dim=_default_horizon_, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr*10, betas=(0.6, 0.999))
    outter_j = 0
    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price, init_coef_B=0.1)
    mask_mat = torch.cat([torch.eye(_default_horizon_), torch.ones((_default_horizon_, 2))], dim=1)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h=h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None, mask=mask_mat, n_job=n_job)

    lr_g = lr * 200
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=lr_g, betas=(0.6, 0.999))
    # scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=200, gamma=0.5)
    dir_folder = '{:s}/{:s}_xi_{:04.0f}_tb1_{:04.0f}_tb2_{:04.0f}_run_{:d}'.format(save_folder,
                                                                                   args.param_file,
                                                                                   xi,
                                                                                   tradeoff_beta1,
                                                                                   tradeoff_beta2, seed)

    if (reload_pretrain_folder == save_folder) and (reload_step != 0) :
        print("reload steps iter=={:d}".format(reload_step))
        reload_pretrain_file = os.path.join(dir_folder, 'iter_%04d.pth.tar' % reload_step)
        bUtil.load_checkopint_gan(reload_pretrain_file, g, clf, optimizer_g=optimizer_g, optimizer_clf=optimizer_clf)



    p_ = 0.5
    prior_pi = torch.Tensor([[p_], [1 - p_]])  # shape : [2,1]

    is_best = False
    losses_gen = []
    losses_adv = []
    with tqdm(total=iter_max) as pbar:
        # with tqdm(dataloader) as pbar:
        while outter_j < (iter_max+1):
            correct_cnt = 0
            tot_cnt = 0
            label_cnt1 = 0
            label_cnt2 = 0

            for k, (D, Y) in enumerate(dataloader_train):
                # print(D.shape, Y.shape)
                g.train(True)
                outter_j += 1
                # unique, counts =np.unique(Y.cpu().numpy(), return_counts=True)
                # raise NotImplementedError(unique, counts)

                optimizer_clf.zero_grad()
                optimizer_g.zero_grad()
                # =========== for sanity check ============
                y_labels = Y.cpu().numpy()
                idx_0, _a1 = np.where(y_labels == 0)
                idx_1, _a2 = np.where(y_labels == 1)
                # # D0 = D[idx_0,:]
                # D1 = D[idx_1,:]
                # raise NotImplementedError(np.where(Y.cpu().numpy() == 0))
                # diagnoise_plot_demand(D0, D1)
                D_ = bUtil.simplify_load_demand(D, timestep_out=6) # D_ is bsz x 6

                # =========== for sanity check ============
                if outter_j % iter_dig == 0:
                    D0 = D_[idx_0, :]
                    D1 = D_[idx_1, :]
                    diagnoise_plot_demand(D0, D1, desc='raw', fpath=dir_folder, iter=outter_j)
                # ============ module end here ============
                y_labels = Y.long().squeeze()
                y_onehot_target = bUtil.convert_onehot_soft(Y.long(), alphabet_size=2)

                # ========== generator ==========
                D_tilde, z_noise = g.forward(D_, y_onehot_target)


                # y_pred = clf(D_)
                y_pred = clf(D_tilde)
                clf_loss = F.binary_cross_entropy_with_logits(y_pred, y_onehot_target)
                # if outter_j % 2 == 0:
                clf_loss.backward(retain_graph=True)
                optimizer_clf.step()

                g_y_target = (1 - y_onehot_target)
                g_priv_loss = tradeoff_beta1 * F.binary_cross_entropy_with_logits(y_pred, g_y_target)
                # g_distort_loss_mse = (torch.clamp((D_ - D_tilde).norm(2, dim=1) - xi, min=-6) ** 2).mean()
                # g_distort_loss_mse = (((D_ - D_tilde).norm(2, dim=1) - xi) ** 2).mean()
                # raise NotImplementedError((D_ - D_tilde).norm(2, dim=1).shape)
                tsize_ = (D_ - D_tilde).norm(2, dim=1).size()
                # print(tsize_)
                g_distort_loss_mse = F.mse_loss((D_ - D_tilde).norm(2, dim=1), torch.ones(tsize_, requires_grad=False)*xi)
                # g_distort_loss_hinge = torch.clamp((D_ - D_tilde).norm(2, dim=1) - xi, min=0).mean()

                r1_ = g_distort_loss_mse.item() / g_priv_loss.item() if g_distort_loss_mse.item() > 1 else 1e3
                r1 = np.clip(r1_, a_min=1e-3, a_max=10000)
                g_loss = tradeoff_beta1 *  g_priv_loss # + tradeoff_beta2 * (1/r1) * g_distort_loss_mse
                         # + tradeoff_beta3 * g_distort_loss_hinge

                loss_util_batch, util_grad, distort_ = g.util_loss(D_, D_tilde, z_noise,
                                                                   y_onehot_target, prior=prior_pi)

                # curr_lr_g = [param_group['lr'] for param_group in optimizer_g.param_groups]
                if outter_j % 2 == 0 and outter_j > 1 :
                    g_loss.backward()
                    with torch.no_grad():
                        g.filter.fc.weight.data -= tradeoff_beta1 * lr_g * g.filter.fc.weight.grad
                        g.filter.fc.weight.data -= lr_g * util_grad.t()

                if outter_j % 100 == 0:
                    lr_g = lr_g * 0.8
                    if outter_j % 1000 ==0:
                        lr_g = 0.1
                # scheduler_g.step()
                #########################################
                batch_j_obj_raw, batch_j_obj_priv = g._objective_vals_getter()
                batch_j_x_raw, batch_j_x_priv = g._ctrl_decisions_getter()

                if outter_j % iter_dig == 0:
                    D0_tilde = D_tilde[idx_0, :]
                    D1_tilde = D_tilde[idx_1, :]
                    diagnoise_plot_demand(D0_tilde, D1_tilde, desc='priv', fpath=dir_folder, iter=outter_j)



                    diagnose_sol(batch_j_obj_raw, batch_j_obj_priv, batch_j_x_raw, batch_j_x_priv, D_, D_tilde, T = _default_horizon_,
                                 k_iter=outter_j, folder=dir_folder, sample_size=2)

                    # print(g.filter.fc.weight.data)
                    # diagnose_filter(generator=g, D_tilde=D_tilde, D=D, y_onehot=y_onehot_target, noise=z_noise,
                    #                 k_iter=outter_j, folder='debug_diagnose_diagmask_s%d'%seed)
                    diagnose_filter(generator=g, D_tilde=D_tilde, D=D_, y_onehot=y_onehot_target, noise=z_noise,
                                    k_iter=outter_j, folder=dir_folder)

                _, y_max_idx = torch.max(y_pred, dim=1)
                correct = y_max_idx == y_labels
                correct_cnt += correct.sum()
                tot_cnt += D_.shape[0]
                label_cnt1 += (y_labels == 0).sum()
                label_cnt2 += (y_labels == 1).sum()

                pbar.set_postfix(iter='{:d}'.format(outter_j), loss='{:.3e}'.format(clf_loss),
                                 # cor_cnts='{:d}'.format(correct_cnt),
                                 # tot_cnts='{:d}'.format(tot_cnt),
                                 acc='{:.3e}'.format(float(correct_cnt) / tot_cnt),
                                 prop1='{:.2e}'.format(float(label_cnt1) / tot_cnt),
                                 prop2='{:.2e}'.format(float(label_cnt2) / tot_cnt),
                                 g_loss='{:.3e}'.format(g_loss.item()),
                                 dist_mse='{:.3e}'.format((D_ - D_tilde).norm(2, dim=1).mean()),
                                 # g_dist_hinge='{:.3e}'.format(g_distort_loss_hinge.item())
                                 )

                pbar.update(1)

                if outter_j % iter_save == 0:

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
                                           checkpoint=dir_folder, filename='iter_%04d.pth.tar' % outter_j)




if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    # Load the parameters from json file
    parser.add_argument('--model_dir', default='experiments/models', help="Directory containing params.json")
    parser.add_argument('--save_dir', default='experiments/models_logs_mask_debug', help="Directory of models logs")
    parser.add_argument('--param_file', default="param_set01_debug")
    parser.add_argument('--p_opt', default='TOU', help='price option (TOU or LMP)')
    parser.add_argument('--run', default=2, type=int)
    parser.add_argument('--load_pretrain_step', default=0, type=int)
    parser.add_argument('--load_pretrain_folder', default="experiments/models_logs_mask_debug_TOU", type=str)


    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, args.param_file+'.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = bUtil.Params(json_path)
    # print(*params.dict)
    save_folder = args.save_dir + '_' + args.p_opt
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        print("create a folder")

    dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=args.run, bsz=params.batch_size, split=0.15)

    if args.load_pretrain_folder != save_folder:
        raise NotImplementedError(" {} is not {}".format(args.load_pretrain_folder, save_folder))
    run_train(dataloader_dict['train'], dataloader_dict['test'], params.dict, xi=params.xi,
              iter_max=params.iter_max, iter_save=params.iter_save, iter_dig=5,
              save_folder=save_folder, reload_pretrain_folder=save_folder, seed=args.run, reload_step=args.load_pretrain_step)

    # run_check(dataloader_dict['train'])