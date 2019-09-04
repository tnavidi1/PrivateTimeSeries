import torch
import torch.nn.functional as F
from tqdm import tqdm
import processData
import nets
import numpy as np
import logging
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




def _extract_filter_weight(x):
    return optMini_util.to_np(x.data)



def run_battery(dataloader, params=None, iter_max=5001, iter_save=100, lr=1e-3, xi=0.5, tradeoff_beta=0.5, savefig=False, verbose=1):
    ## multiple iterations
    # init price

    _default_horizon_ = 48
    torch.manual_seed(2)
    # price = torch.rand((_default_horizon_, 1))  # price is a column vector
    price = bUtil.create_price()
    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price)
    # controller = OptPrivModel(Q, q, G, h, A, b, T=T)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h = h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None)

    clf = nets.Classifier(z_dim=48, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=lr)
    # raise NotImplementedError(*g.filter.parameters())
    batchs_length =len(dataloader)
    losses_gen = []
    losses_adv = []
    # for j in range(ep):
    j = 0
    best_val_acc = 0.0
    loss_avg_g = bUtil.RunningAverage()
    loss_avg_priv = bUtil.RunningAverage()
    with tqdm(total=iter_max) as pbar:
        # with tqdm(dataloader) as pbar:
        while True:
            correct_cnt = 0
            tot_cnt = 0
            label_cnt1 = 0
            label_cnt2 = 0

            # raise NotImplementedError(len(dataloader))
            for k, (D, Y) in enumerate(dataloader):
                #
                j += 1
                k = j
                # k = j * batchs_length + k
                optimizer_g.zero_grad()
                optimizer_clf.zero_grad()
                y_labels = bUtil.convert_binary_label(Y, 1500) # row vector
                y_onehot = bUtil.convert_onehot(y_labels.unsqueeze(1), alphabet_size=2)
                D_tilde, z_noise = g.forward(D, y_onehot)

                y_out = clf(D_tilde)
                loss_priv = F.cross_entropy(y_out, y_labels, weight=None,
                                           ignore_index=-100, reduction='mean')
                loss_util = g.util_loss(D, y_onehot, xi=xi)

                loss_priv.backward(retain_graph=True) # retain_graph=True
                optimizer_clf.step()

                g_loss = loss_util - tradeoff_beta * loss_priv #+ 0.1 * torch.norm(g.filter.fc.weight[:, 48:], p=1, dim=0).mean()
                g_loss.backward()
                optimizer_g.step()

                _, y_max_idx = torch.max(y_out, dim=1)
                correct = y_max_idx == y_labels
                correct_cnt += correct.sum()
                tot_cnt += D.shape[0]
                label_cnt1 += (y_labels == 0).sum()
                label_cnt2 += (y_labels == 1).sum()

                trace_track = (torch.trace(torch.mm(g.filter.fc.weight.data, g.filter.fc.weight.data.t()))).cpu()

                loss_avg_g.update(g_loss.item())
                loss_avg_priv.update(loss_priv.item())

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

                losses_adv.append(loss_avg_priv())
                losses_gen.append(loss_avg_g())

                val_metrics = {"acc": float(correct_cnt) / tot_cnt,
                               "prop1": float(label_cnt1) / tot_cnt,
                               "prop2": float(label_cnt2) /  tot_cnt,
                               "tr": trace_track.item()}

                dir_folder = '../fig/demand_visual_xi_{:04.0f}_tbeta_{:04.0f}'.format(xi * 10, tradeoff_beta * 10)
                if not os.path.exists(dir_folder):
                    os.mkdir(dir_folder)


                if j == iter_max:
                    return

                val_acc = float(correct_cnt) / tot_cnt
                is_best = val_acc >= best_val_acc


                if j % iter_save == 0:
                    bUtil.save_checkpoint({'epoch': k + 1,
                                           'g_state_dict': g.state_dict(),
                                           'g_optim_dict': optimizer_g.state_dict(),
                                           'clf_state_dict': clf.state_dict(),
                                           'clf_optim_dict': optimizer_clf.state_dict(),
                                           'loss_g': losses_gen,
                                           'loss_a': losses_adv},
                                            is_best=is_best,
                                            checkpoint=dir_folder)

                if is_best:
                    logging.info("- Found new best accuracy")
                    best_val_acc = val_acc

                    # Save best val metrics in a json file in the model directory
                    best_json_path = os.path.join(dir_folder, "metrics_val_best_weights.json")
                    bUtil.save_dict_to_json(val_metrics, best_json_path)

                # Save latest val metrics in a json file in the model directory

                if k % 50 == 0 and verbose == 1:
                    z_noise_gen = g.sample_z(batch=32)
                    z_noise_gen = z_noise_gen / z_noise_gen.norm(2, dim=1).unsqueeze(1).repeat(1, _default_horizon_)
                    concat_noise = torch.cat([z_noise_gen, y_onehot], dim=1).cpu().numpy()

                    out_purtbation = (concat_noise).dot(g.filter.fc.weight.data.cpu().numpy().transpose())
                    # print(out_purtbation)
                    ind_ = np.random.randint(low=0, high=32, size=4)
                    fig, ax =plt.subplots(2, 2, figsize=(9, 5))
                    max_axis_up = max(np.max(concat_noise[ind_]), np.max(out_purtbation[ind_].transpose()))
                    min_axis_up = min(np.min(concat_noise[ind_]), np.min(out_purtbation[ind_].transpose()))

                    ax[0, 0].plot(concat_noise[ind_].transpose())
                    ax[0, 0].set_title("noise")

                    ax[0, 1].plot(out_purtbation[ind_].transpose())
                    ax[0, 1].set_title("out noise")
                    ax[0, 0].set_ylim(min_axis_up, max_axis_up)
                    ax[0, 1].set_ylim(min_axis_up, max_axis_up)

                    max_axis_low = max(np.max(D[ind_].t().cpu().numpy())*1.05, np.max(D_tilde[ind_].detach().cpu().numpy()) * 1.05)
                    min_axis_low = min(np.min(D[ind_].t().cpu().numpy()), np.min(D_tilde[ind_].detach().cpu().numpy()))
                    ax[1, 0].plot(D[ind_].t().cpu().numpy() )
                    ax[1, 0].plot((out_purtbation[ind_]+D[ind_].cpu().numpy()).transpose(), '--')
                    ax[1, 0].set_title("raw and alter demand")

                    ax[1, 1].plot(D[ind_].t().cpu().numpy())
                    ax[1, 1].plot((D_tilde[ind_].detach().cpu().numpy()).transpose(), '--')
                    ax[1, 1].set_title("another view of alter demand")
                    ax[1, 0].set_ylim(min_axis_low, max_axis_low)
                    ax[1, 1].set_ylim(min_axis_low, max_axis_low)

                    plt.tight_layout()
                    plt.savefig('%s/diagnose_iter_%d.png'%(dir_folder, k))
                    plt.close('all')

                    #############################
                    plt.figure(figsize=(6.5,5))
                    s = k - 1001 if k > 1001 else 0
                    t = k
                    plt.plot(np.arange(len(losses_adv[s:t])), losses_adv[s:t], label='adv loss')
                    plt.plot(np.arange(len(losses_gen[s:t])), losses_gen[s:t], label='gen loss')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('%s/losses_iter_%d.png'%(dir_folder, k))
                    plt.close('all')

                # =======================================
                # plot out figures
                if k % 50 == 0 and savefig is True:
                    # print(g.filter.fc.weight.data)
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(g.filter.fc.weight.data.cpu().numpy())
                    plt.title("iter==%d" % k)
                    plt.tight_layout()
                    plt.savefig('../fig/filter_visual/f_weight_%d.png' % k )
                    plt.close('all')

                    fig, ax=plt.subplots(3,1, figsize=(6.5, 10))
                    i=np.random.randint(0, 32)
                    ax[0].plot(D.cpu().numpy().transpose())
                    ax[1].plot(D_tilde.detach().cpu().numpy().transpose())
                    ax[2].plot(np.vstack((D[i].cpu().numpy(), D_tilde[i].detach().cpu().numpy())).transpose() )
                    ax[0].set_title("raw demand")
                    ax[1].set_title("priv demand")
                    ax[2].set_title("sampled demand plot")
                    ax[2].legend(['raw', 'priv'])
                    plt.tight_layout()
                    plt.savefig('../fig/demand_visual/iter_%d.png'%k)
                    plt.close('all')

            # last_json_path = os.path.join(dir_folder, "metrics_val_last_weights.json")
            # bUtil.save_dict_to_json(val_metrics, last_json_path)


params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, beta3=0.5, alpha=0.2)

for xi in [10, 20]:
    run_battery(dataloader_dict['train'], params=params, iter_max=3001, lr=1e-3, xi=xi, tradeoff_beta=2)

