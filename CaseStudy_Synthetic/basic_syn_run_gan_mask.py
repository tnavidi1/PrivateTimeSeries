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
sns.set('paper', style="whitegrid", font_scale=2.5, rc={"lines.linewidth": 2.5}, )
current_palette = sns.color_palette()
# sns.palplot(current_palette)

# raise NotImplementedError(current_palette)

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib
import basic_util as bUtil


# ==== printing config ==== #
desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150
torch.set_printoptions(profile="full", linewidth=400)

# dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=args.run, bsz=64, split=0.15)

def diagnoise_plot_demand(D1, D2, desc, fpath, iter=1):
    if not os.path.exists(fpath):
        os.mkdir(fpath)
        print("----- create folder {}".format(fpath))

    import pandas as pd
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(desc+' demand')
    # plt.plot(D1.detach().t().cpu().numpy(), 'b-', alpha=0.4, label='pos')
    # plt.plot(D2.detach().t().cpu().numpy(), 'g-.', alpha=0.5, label='neg')
    # print(D1.detach().t().cpu().numpy().shape)  # 24 32,
    # print(D2.detach().t().cpu().numpy().shape)  # 24 ,
    def reformatting(batch_D, label_input=1):
        data1_nparr = batch_D.detach().t().cpu().numpy()
        steps_ = data1_nparr.shape[0]
        hr = (np.arange(0, steps_))[:, np.newaxis]
        pos_label = np.repeat(label_input, steps_)[:, np.newaxis]
        data1 = np.concatenate([hr, pos_label, data1_nparr], axis=1)
        df1 = pd.DataFrame(data1, columns=['hr', 'label']+['day%d' % k for k in np.arange(data1_nparr.shape[1])]) # .melt()
        df1 = df1.melt(id_vars=['hr', 'label'])
        return df1
        # raise NotImplementedError(df1)

    df1 = reformatting(D1, label_input=1)
    df2 = reformatting(D2, label_input=0)
    df = pd.concat([df1, df2], axis=0)
    df['label'] = df.label.astype(int)
    # raise NotImplementedError(df)
    sns.lineplot(x='hr', y='value', hue='label', data=df, ax=ax);
    ax.plot(D1.detach().t().cpu().numpy(), color=current_palette[1], alpha=0.15)
    ax.plot(D2.detach().t().cpu().numpy(), color=current_palette[0], alpha=0.15)
    # sns.lineplot(data=D1.detach().t().cpu().numpy(), dashes=False,  label='pos')
    # sns.lineplot(data=D2.detach().t().cpu().numpy(), dashes=False,  label='neg')
    ax.set_ylabel('Power (W)')
    ax.set_xlabel('Time (hr)')
    plt.tight_layout()

    L = ax.legend()

    new_labels = ['attribute', 'high-income', 'low-income']
    for t, l in zip(L.get_texts(), new_labels):
        t.set_text(l)
    # plt.show()
    plt.savefig(os.path.join(fpath, 'debug_%s_iter_%04d.png' % (desc, iter)))
    plt.close('all')



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
        plt.show()
        # plt.savefig(os.path.join(folder, 'diagnostic_b_filterG_iter_%d.png' %k_iter))
        plt.close('all')


def diagnose_sol(batch_obj_raw, batch_obj_priv, batch_x_raw, batch_x_priv, k_iter=0, folder=None):
    T = 24
    # print(batch_obj_raw.shape, batch_obj_priv.shape) # torch size [bsz]
    # print((batch_obj_raw-batch_obj_priv).cpu().numpy())
    # print(batch_x_raw.shape, batch_x_priv.shape)  # torch size [bsz x 72]
    if not os.path.exists(folder):
        os.mkdir(folder)
        print("----- create folder {}".format(folder))
    else:
        print('folder {} exsits already!'.format(folder))


    bsz = batch_obj_raw.shape[0]
    idx = np.random.randint(0, bsz, size=10)
    net_x_raw = batch_x_raw[idx, :T] - batch_x_raw[idx, T:2*T]
    net_x_priv = batch_x_priv[idx, :T] - batch_x_priv[idx, T:2*T]


    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
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

def run_train(dataloader_train, dataloader_test, params, p_opt='TOU', iter_max=500, iter_save=20, iter_dig=10,
              lr=1e-3, xi=0.5,
              tradeoff_beta1=1, tradeoff_beta2 = 1, tradeoff_beta3=1,
              save_folder=None,
              reload_step=0,
              n_job=5,
              seed=1,
              reload_pretrain_folder=None, lambd_ =0.1, savefig=True, verbose=1 ):

    _default_horizon_ = 24
    torch.manual_seed(seed)

    price = None
    if p_opt == 'TOU':
        price = bUtil.create_price(steps_perHr=1)
    elif p_opt == 'LMP':
        price = bUtil.create_LMP('../Data_LMP/8-18-2019_8days_Menlo_LMPs.csv', granular=24, steps_perHr=1)
    else:
        price = torch.rand((_default_horizon_, 1))  # price is a column vector

    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price, init_coef_B=0.02)

    mask_mat = torch.cat([torch.eye(_default_horizon_), torch.ones((_default_horizon_, 2))], dim=1)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h=h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None, mask=mask_mat, n_job=n_job)

    clf = nets.Classifier(z_dim=24, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))
    lr_g = lr*90
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=lr_g, betas=(0.6, 0.999))
    # scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=400, gamma=0.5)

    dir_folder = '{:s}/{:s}_xi_{:04.0f}_tb1_{:04.0f}_tb2_{:04.0f}_run_{:d}'.format(save_folder,
                                                                                   args.param_file,
                                                                                   xi,
                                                                                   tradeoff_beta1,
                                                                                   tradeoff_beta2, args.run)

    if (reload_pretrain_folder is not None) and (reload_step != 0) :
        print("==== reload steps iter:{:d} ====".format(reload_step))
        reload_pretrain_file = os.path.join(reload_pretrain_folder, 'iter_%04d.pth.tar' % reload_step)
        bUtil.load_checkopint_gan(reload_pretrain_file, g, clf, optimizer_g=optimizer_g, optimizer_clf=optimizer_clf)

    outter_j = 0
    p_ = 0.5
    prior_pi = torch.Tensor([[p_], [1 - p_]])  # shape : [2,1]
    is_best = False
    losses_gen = []
    losses_adv = []
    acc_array = []
    dist_arr = []
    d_mag_arr = []
    with tqdm(total=iter_max) as pbar:
        # with tqdm(dataloader) as pbar:
        while outter_j < iter_max:
            correct_cnt = 0
            tot_cnt = 0
            label_cnt1 = 0
            label_cnt2 = 0

            # raise NotImplementedError(len(dataloader))
            for k, (D, Y) in enumerate(dataloader_train):
                outter_j += 1

                optimizer_g.zero_grad()
                optimizer_clf.zero_grad()

                y_labels = Y

                y_labels_1d = y_labels.cpu().numpy()
                idx_0, _a1 = np.where(y_labels_1d == 0)
                idx_1, _a2 = np.where(y_labels_1d == 1)

                y_onehot_target = bUtil.convert_onehot_soft(y_labels, alphabet_size=2)
                D_tilde, z_noise = g.forward(D, y_onehot_target)
                y_pred = clf(D_tilde)
                clf_loss = F.binary_cross_entropy_with_logits(y_pred, y_onehot_target)
                # if outter_j % 2 == 0 and clf_loss.item() > 0.2:
                clf_loss.backward(retain_graph=True)
                optimizer_clf.step()

                loss_util_batch, util_grad, distort_ = g.util_loss(D, D_tilde, z_noise,
                                                                   y_onehot_target, prior=prior_pi)

                g_y_target = (1 - y_onehot_target)
                g_priv_loss = F.binary_cross_entropy_with_logits(y_pred, g_y_target)

                # =============================
                ## g_distort_loss_mse = tradeoff_beta2 * ((distort_ - xi) ** 2) + tradeoff_beta3 * (torch.clamp(distort_ - xi, min=0))
                # g_distort_loss_mse = ((distort_ - xi) ** 2)
                # g_distort_loss_hinge = (torch.clamp(distort_ - xi, min=0))

                # ====== use diff demand ======
                D_diff_gap = (D - D_tilde).norm(2, dim=1).mean()
                D_mag = D.norm(2, dim=1).mean()
                g_distort_loss_mse=(torch.clamp(D_diff_gap - xi, min=0)**2).mean()
                g_distort_loss_hinge = torch.clamp(D_diff_gap - xi, min=0).mean()
                ## raise NotImplementedError(g_priv_loss.item(), g_distort_loss_mse.item(), g_distort_loss_hinge.item())
                r1_ = g_distort_loss_mse.item() / g_priv_loss.item() if g_distort_loss_mse.item() > (0.01 * xi) else 100
                r1 = np.clip(r1_, a_min=1e-3, a_max=10000)
                g_loss = tradeoff_beta1  * g_priv_loss # + tradeoff_beta2 * (1/r1)* g_distort_loss_mse + tradeoff_beta3*g_distort_loss_hinge

                loss_util = loss_util_batch.mean()

                # optimizer_g.step()
                # scheduler_g.step()
                # curr_lr_g = [param_group['lr'] for param_group in optimizer_g.param_groups]
                if outter_j % 2 == 0 and outter_j > 10 :
                    g_loss.backward()
                    with torch.no_grad():
                        g.filter.fc.weight.data -= lr_g * g.filter.fc.weight.grad
                        g.filter.fc.weight.data -= lr_g * util_grad.t()

                if outter_j % 100 == 0:
                    lr_g = max(lr_g * 0.8, 5*1e-4)
                    if outter_j % 1000 ==0:
                        lr_g = 0.1



                batch_j_obj_raw, batch_j_obj_priv = g._objective_vals_getter()
                batch_j_x_raw, batch_j_x_priv = g._ctrl_decisions_getter()

                # if outter_j % iter_dig == 0:
                #     diagnose_sol(batch_j_obj_raw, batch_j_obj_priv, batch_j_x_raw, batch_j_x_priv,
                #              outter_j, folder='debug_diagnose_diagmask_s%d' % seed)

                # curr_lr_g = [param_group['lr'] for param_group in optimizer_g.param_groups]

                y_labels_1d_long = Y.long().squeeze()
                _, y_max_idx = torch.max(y_pred, dim=1)
                correct = y_max_idx == y_labels_1d_long
                correct_cnt += correct.sum()
                tot_cnt += D.shape[0]
                label_cnt1 += (y_labels_1d_long == 0).sum()
                label_cnt2 += (y_labels_1d_long == 1).sum()

                losses_adv.append(clf_loss.item())
                losses_gen.append(g_loss.item())
                acc_array.append(float(correct_cnt) / tot_cnt)
                dist_arr.append(D_diff_gap.item())
                d_mag_arr.append(D_mag.item())

                pbar.set_postfix(o_iter='{:d}'.format(outter_j),
                                 # in_iter='{:d}'.format(k),
                                 acc = '{:.3e}'.format(float(correct_cnt) / tot_cnt),
                                 clf_loss='{:.3e}'.format(clf_loss.item()),
                                 # g_loss='{:.3e}'.format(g_loss.item()),
                                 util_loss='{:.3e}'.format(loss_util),
                                 ds1='{:.3e}'.format(distort_.item()),
                                 ds2='{:.3e}'.format(D_diff_gap.item()**2),
                                 g_p_loss ='{:.3e}'.format(g_priv_loss.item()),
                                 # g_dis_loss_mse = '{:.3e}'.format(g_distort_loss_mse.item()),
                                 # g_dis_loss_hinge = '{:.2e}'.format(g_distort_loss_hinge.item()),
                                 # cur_lr = '{:.3e}'.format(lr_g),
                                 # ratio_r = '{:.2e}'.format(r1_)
                                 )
                pbar.update(1)

                if outter_j % iter_dig == 0:
                    # diagnose_sol(batch_j_obj_raw, batch_j_obj_priv, batch_j_x_raw, batch_j_x_priv,
                    #              outter_j, folder='debug_diagnose_diagmask_s%d' % seed)

                    D0_tilde = D_tilde[idx_0, :]
                    D1_tilde = D_tilde[idx_1, :]
                    D0 = D[idx_0, :]
                    D1 = D[idx_1, :]
                    diagnoise_plot_demand(D0, D1, desc='raw', fpath=dir_folder, iter=outter_j)
                    diagnoise_plot_demand(D0_tilde, D1_tilde, desc='privatized', fpath=dir_folder, iter=outter_j)

                    diagnose_sol(batch_j_obj_raw, batch_j_obj_priv, batch_j_x_raw, batch_j_x_priv,
                                                     outter_j, folder=dir_folder)

                    # diagnose_filter(generator=g, D_tilde=D_tilde, D=D, y_onehot=y_onehot_target, noise=z_noise,
                    #                 k_iter=outter_j, folder='debug_diagnose_diagmask_s%d'%seed)
                    diagnose_filter(generator=g, D_tilde=D_tilde, D=D, y_onehot=y_onehot_target, noise=z_noise,
                                                        k_iter=outter_j, folder=dir_folder)


                # if outter_j % 5 == 0:
                #     raise NotImplementedError
                if outter_j % iter_save == 0 and outter_j > 99:

                    bUtil.save_checkpoint({'epoch': outter_j + 1,
                                           'g_state_dict': g.state_dict(),
                                           'g_optim_dict': optimizer_g.state_dict(),
                                           'clf_state_dict': clf.state_dict(),
                                           'clf_optim_dict': optimizer_clf.state_dict(),
                                           'loss_g': losses_gen,
                                           'loss_a': losses_adv,
                                           'acc': acc_array,
                                           'obj_raw': batch_j_obj_raw,
                                           'obj_priv': batch_j_obj_priv,
                                           'lambda': tradeoff_beta1,
                                           'distort': dist_arr,
                                           'd_magnitude': d_mag_arr},
                                           is_best=is_best,
                                           checkpoint=dir_folder, filename='iter_%04d.pth.tar' % outter_j)

                if outter_j >= iter_max:
                    print("terminate!!")
                    return



# params = dict(learning_rate=1e-3, batch_size=64,
#               iter_max=1002, iter_save=50, num_workers=10,
#               tradeoff_beta1 = 3, tradeoff_beta2 = 2,
#               c_i=150, c_o=150, eta_eff=0.99,
#               T=24, B=160,
#               beta1=0.05, beta2=0.04, beta3=0.02,
#               alpha=0.2)


#  lr = 5*1e-3
# run_train(dataloader_dict['train'], params=params, p_opt='TOU', iter_max=501, xi=30000, lr=1e-2,
#           n_job=10, seed=2, load_pretrain='debug_diagnose_diagmask_s2/iter_0700.pth.tar')


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    # Load the parameters from json file

    parser.add_argument('--model_dir', default='experiments/models', help="Directory containing params.json")
    parser.add_argument('--save_dir', default='experiments/models_logs_mask', help="Directory of models logs")
    parser.add_argument('--param_file', default="param_set_04")
    parser.add_argument('--p_opt', default='TOU', help='price option (TOU or LMP)')
    parser.add_argument('--run', default=3, type=int)
    parser.add_argument('--load_pretrain_step', default=1600, type=int)
    parser.add_argument('--load_pretrain_folder',
                        default="experiments/models_logs_mask_TOU/param_set_04_xi_0000_tb1_0128_tb2_0001_run_3", type=str)
    # parser.add_argument('--lambda')


    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, args.param_file+'.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = bUtil.Params(json_path)
    # print(*params.dict)
    save_folder = args.save_dir + '_' + args.p_opt
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        print("create a folder")
    # raise NotImplementedError(params.dict)
    # raise NotImplementedError(os.path.dirname(args.load_pretrain_folder))
    dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=args.run, bsz=params.batch_size, split=0.15)

    if args.load_pretrain_step > 0:
        if os.path.dirname(args.load_pretrain_folder) != save_folder:
            raise NotImplementedError(" {} is not {}".format(args.load_pretrain_folder, save_folder))

    run_train(dataloader_dict['train'], dataloader_dict['test'],
              params=params.dict, iter_max=params.iter_max, iter_save=params.iter_save, iter_dig=2,
              lr=params.learning_rate, xi=params.xi,
              tradeoff_beta1=params.tradeoff_beta1,
              tradeoff_beta2=params.tradeoff_beta2,
              tradeoff_beta3=params.tradeoff_beta3,
              p_opt=args.p_opt,
              save_folder=save_folder,
              reload_step= args.load_pretrain_step,
              reload_pretrain_folder=args.load_pretrain_folder,
              n_job=params.num_workers, seed=args.run, savefig=True, verbose=1 )
