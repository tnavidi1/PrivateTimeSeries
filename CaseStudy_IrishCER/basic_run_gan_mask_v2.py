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


desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=64)



def _extract_filter_weight(x):
    return optMini_util.to_np(x.data)



def run_battery_train(dataloader, dataloader_test=None, params=None, iter_max=5001, iter_save=100, lr=1e-3, xi=0.5,
                tradeoff_beta1=0.5, tradeoff_beta2=1, p_opt = 'TOU', save_folder=None, reload_step=0,
                      savefig=False, verbose=1, n_job=5):
    ## multiple iterations
    # init price

    _default_horizon_ = 48
    torch.manual_seed(args.run)

    price=None
    if p_opt == 'TOU':
        price = bUtil.create_price()
    elif p_opt == 'LMP':
        price = bUtil.create_LMP('../Data_LMP/8-18-2019_8days_Menlo_LMPs.csv')
    else:
        price = torch.rand((_default_horizon_, 1))  # price is a column vector

    # print(price)
    # raise NotImplementedError()

    Q, q, G, h, A, b, T, price = bUtil._form_QP_params(params, p=price)

    mask_mat=torch.cat([torch.eye(_default_horizon_), torch.ones((_default_horizon_, 2))], dim=1)
    # print("creat mask {}".format(mask_mat.shape))
    # raise NotImplementedError(mask_mat)
    g = nets.Generator(z_dim=_default_horizon_, y_priv_dim=2, Q=Q, G=G, h = h, A=A, b=b,
                       T=_default_horizon_, p=price,
                       device=None, mask=mask_mat, n_job=n_job)

    # print(g)
    # raise NotImplementedError
    clf = nets.Classifier(z_dim=48, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))
    optimizer_g = torch.optim.Adam(g.filter.parameters(), lr=lr, betas=(0.6, 0.999))
    # raise NotImplementedError(*g.filter.parameters())
    # batchs_length =len(dataloader)


    #### define the folder ####
    dir_folder = '{:s}/{:s}_xi_{:04.0f}_tb1_{:04.0f}_tb2_{:04.0f}_run_{:d}'.format(save_folder,
                                                                                   args.param_file,
                                                                                   xi,
                                                                                   tradeoff_beta1,
                                                                                   tradeoff_beta2, args.run)


    if reload_step > 0:
        print("================= load pretrain step: {:d} =================".format(reload_step))
        ckpt = os.path.join(dir_folder, "iter_%04d.pth.tar" % reload_step)
        print(ckpt)
        bUtil.load_checkopint_gan(ckpt, g, clf, optimizer_g=optimizer_g, optimizer_clf=optimizer_clf)

    losses_gen = []
    losses_adv = []
    # for j in range(ep):
    j = 0
    best_val_acc = 1.0
    loss_avg_g = bUtil.RunningAverage()
    loss_avg_priv = bUtil.RunningAverage()
    acc_avg = bUtil.RunningAverage()
    acc_avg.update(best_val_acc)
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
                # D = F.normalize(D, p=1, dim=1)
                bsz = D.shape[0]
                j += 1
                k = j
                # k = j * batchs_length + k
                optimizer_g.zero_grad()
                optimizer_clf.zero_grad()
                y_labels = bUtil.convert_binary_label(Y, 1500) # row vector
                # y_onehot = bUtil.convert_onehot(y_labels.unsqueeze(1), alphabet_size=2)
                y_onehot = bUtil.convert_onehot_soft(y_labels.unsqueeze(1), alphabet_size=2)
                D_tilde, z_noise = g.forward(D, y_onehot)

                y_out = clf(D_tilde)
                loss_priv = F.cross_entropy(y_out, y_labels, weight=None,
                                           ignore_index=-100, reduction='mean')
                loss_util_batch, util_grad, loss_tr_p = g.util_loss(D, D_tilde, z_noise, y_onehot, xi=xi)

                loss_priv.backward(retain_graph=True) # retain_graph=True
                optimizer_clf.step()
                # print(loss_util)
                loss_util = loss_util_batch.mean()
                # hyper_2 = 10 if tr_penalty > 1e-3 else 0
                # tradeoff_beta1 = 0 if loss_tr_p.item() < 1e-3 else tradeoff_beta1
                # loss_tr_p
                batch_D_tilde_rowsum = torch.norm(D_tilde, p=1, dim=1)
                batch_D_rowsum = torch.ones(size=batch_D_tilde_rowsum.size())
                row_sum_penalty = F.l1_loss(batch_D_tilde_rowsum, batch_D_rowsum)

                tradeoff_beta1_ = tradeoff_beta1 if loss_tr_p > 0.01 else 0.01
                r1_ = np.clip(loss_tr_p.item()/loss_priv.item(), 1e-3, 1e4) if loss_tr_p > 0.01 else 1
                r2_ = np.clip(loss_tr_p.item()/row_sum_penalty.item(), 1e-3, 1e4) if row_sum_penalty.item() > 0.1 else 100

                tradeoff_beta2_ = tradeoff_beta2 if loss_priv.item() > 0.02 else 100
                print(r1_, r2_)

                # print(torch.norm(D_tilde, p=1, dim=1))
                # print(torch.norm(D, p=1, dim=1))
                # raise NotImplementedError(torch.norm(D_tilde, p=1, dim=1))
                g_loss = tradeoff_beta1_ * (1/r1_) * loss_tr_p - tradeoff_beta2_ * loss_priv + (1/r2_) * row_sum_penalty #+ loss_util #+ 0.1 * torch.norm(g.filter.fc.weight[:, 48:], p=1, dim=0).mean()
                # g_loss.backward(retain_graph=True)

                # print(g.filter.fc.weight.data, g.filter.fc.weight.data.shape)
                # raise NotImplementedError(util_grad.shape)
                g.filter.fc.weight.data -= lr * util_grad.t()
                g_loss.backward()
                optimizer_g.step()
                # raise NotImplementedError(g.filter.fc.weight.grad.shape)
                # g_loss.backward(gradient=(util_grad))

                # raise NotImplementedError

                _, y_max_idx = torch.max(y_out, dim=1)
                correct = y_max_idx == y_labels
                correct_cnt += correct.sum()
                tot_cnt += D.shape[0]
                label_cnt1 += (y_labels == 0).sum()
                label_cnt2 += (y_labels == 1).sum()

                trace_track = (torch.trace(torch.mm(g.filter.fc.weight.data, g.filter.fc.weight.data.t()))).cpu()

                loss_avg_g.update(g_loss.item())
                loss_avg_priv.update(loss_priv.item())

                batch_j_obj_raw, batch_j_obj_priv = g._objective_vals_getter()
                gap_obj = (batch_j_obj_priv.data.t() - batch_j_obj_raw.data.t()).mean()
                pbar.set_postfix(iter='{:d}'.format(k), g_loss='{:.3e}'.format(g_loss),
                                 util_loss = '{:.3e}'.format(loss_util),
                                 priv_loss='{:.3e}'.format(loss_priv),
                                 # cor_cnts='{:d}'.format(correct_cnt),
                                 # tot_cnts='{:d}'.format(tot_cnt),
                                 acc='{:.2e}'.format(float(correct_cnt) / tot_cnt),
                                 prop1='{:.2e}'.format(float(label_cnt1) / tot_cnt),
                                 prop2='{:.2e}'.format(float(label_cnt2) / tot_cnt),
                                 tr='{:.2e}'.format(trace_track),
                                 tr_gap='{:.2e}'.format(loss_tr_p),
                                 obj_gap = '{:.2e}'.format(gap_obj.item()),
                                 r_sum_p = '{:.2e}'.format(row_sum_penalty.item())
                                 )
                pbar.update(1)

                losses_adv.append(loss_avg_priv())
                losses_gen.append(loss_avg_g())

                val_metrics = {"acc": float(correct_cnt) / tot_cnt,
                               "prop1": float(label_cnt1) / tot_cnt,
                               "prop2": float(label_cnt2) /  tot_cnt,
                               "tr": trace_track.item()}


                ## create a dir folder ##
                if not os.path.exists(dir_folder):
                    os.mkdir(dir_folder)

                if j == iter_max:
                    return

                val_acc = float(correct_cnt) / tot_cnt
                acc_avg.update(val_acc)
                is_best = acc_avg() <= best_val_acc
                if j % 100 == 1:
                    train_clf(dataloader_test, model_clf=clf, model_g=g, optimizer_clf=optimizer_clf)

                if j % iter_save == 1 :

                    bUtil.save_checkpoint({'epoch': k + 1,
                                           'g_state_dict': g.state_dict(),
                                           'g_optim_dict': optimizer_g.state_dict(),
                                           'clf_state_dict': clf.state_dict(),
                                           'clf_optim_dict': optimizer_clf.state_dict(),
                                           'loss_g': losses_gen,
                                           'loss_a': losses_adv,
                                           'obj_raw': batch_j_obj_raw,
                                           'obj_priv': batch_j_obj_priv},
                                            is_best=is_best,
                                            checkpoint=dir_folder, filename='iter_%04d.pth.tar' % j)

                    print(batch_j_obj_raw.data, batch_j_obj_priv.data)
                    # print()
                    res_json_path = os.path.join(dir_folder, "metrics_val_best_weights_%04d.json"%j)
                    # val_acc = float(correct_cnt) / tot_cnt
                    bUtil.save_dict_to_json(val_metrics, res_json_path)

                    if is_best:
                        logging.info("- Found new best accuracy")
                        best_val_acc = acc_avg()

                        # Save best val metrics in a json file in the model directory
                        best_json_path = os.path.join(dir_folder, "metrics_val_best_weights.json")
                        bUtil.save_dict_to_json(val_metrics, best_json_path)

                # Save latest val metrics in a json file in the model directory

                if k % iter_save == 1 and verbose == 1:
                    # # figure
                    # z_noise_gen = g.sample_z(batch=bsz)
                    # z_noise_gen = z_noise_gen / z_noise_gen.norm(2, dim=1).unsqueeze(1).repeat(1, _default_horizon_)
                    # concat_noise = torch.cat([z_noise_gen, y_onehot], dim=1).cpu().numpy()
                    # # @FIXME this line is different from orignal line
                    # out_purtbation = (concat_noise).dot(g.filter.fc.weight.data.cpu().numpy())
                    # # print(out_purtbation)
                    # ind_ = np.random.randint(low=0, high=bsz, size=4)
                    # fig, ax =plt.subplots(2, 2, figsize=(9, 5))
                    # max_axis_up_l = max(np.max(concat_noise[ind_]), np.max(out_purtbation[ind_].transpose()))
                    # min_axis_up_l = min(np.min(concat_noise[ind_]), np.min(out_purtbation[ind_].transpose()))
                    # max_axis_up_r = np.max(out_purtbation[ind_].transpose())
                    # min_axis_up_r = np.min(out_purtbation[ind_].transpose())
                    # # print(max_axis_up_r, min_axis_up_r)
                    # # raise NotImplementedError
                    #
                    # ax[0, 0].plot(concat_noise[ind_].transpose())
                    # ax[0, 0].set_title("noise")
                    #
                    # ax[0, 1].plot(out_purtbation[ind_].transpose())
                    # ax[0, 1].set_title("out noise")
                    # ax[0, 0].set_ylim(min_axis_up_l, max_axis_up_l)
                    # ax[0, 1].set_ylim(min_axis_up_r, max_axis_up_r)
                    #
                    # max_axis_low = max(np.max(D[ind_].t().cpu().numpy())*1.05, np.max(D_tilde[ind_].detach().cpu().numpy()) * 1.05)
                    # min_axis_low = min(np.min(D[ind_].t().cpu().numpy()), np.min(D_tilde[ind_].detach().cpu().numpy()))
                    # ax[1, 0].plot(D[ind_].t().cpu().numpy() )
                    # ax[1, 0].plot((out_purtbation[ind_]+D[ind_].cpu().numpy()).transpose(), '--')
                    # ax[1, 0].set_title("raw and alter demand")
                    #
                    # ax[1, 1].plot(D[ind_].t().cpu().numpy())
                    # ax[1, 1].plot((D_tilde[ind_].detach().cpu().numpy()).transpose(), '--')
                    # ax[1, 1].set_title("another view of alter demand")
                    # ax[1, 0].set_ylim(min_axis_low, max_axis_low)
                    # ax[1, 1].set_ylim(min_axis_low, max_axis_low)
                    #
                    # plt.tight_layout()
                    # plt.savefig('%s/diagnose_iter_%d.png'%(dir_folder, k))
                    # plt.close('all')

                    #############################
                    # plt.figure(figsize=(6.5,5))
                    # s = k - 1500 if k > 1500 else 0
                    # # s = 50
                    # t = k
                    # plt.plot(np.arange(len(losses_adv[s:t])), losses_adv[s:t], label='adv loss')
                    # plt.plot(np.arange(len(losses_gen[s:t])), losses_gen[s:t], label='gen loss')
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.savefig('%s/losses_iter_%d.png'%(dir_folder, k))
                    # plt.close('all')

                # =======================================
                # plot out figures
                # if k % iter_save == 1 and savefig is True:
                #     # print(g.filter.fc.weight.data)
                #     plt.figure(figsize=(6, 5))
                #     sns.heatmap(g.filter.fc.weight.data.cpu().numpy().transpose(), cmap="RdBu")
                #     plt.title("iter==%d" % k)
                #     plt.tight_layout()
                #     plt.savefig('%s/filter_visual_weight_%d.png' %( dir_folder, k))
                #     plt.close('all')
                #
                #     fig, ax=plt.subplots(3,1, figsize=(6.5, 10))
                #     i=np.random.randint(0, 32)
                #     ax[0].plot(D.cpu().numpy().transpose())
                #     ax[1].plot(D_tilde.detach().cpu().numpy().transpose())
                #     ax[2].plot(np.vstack((D[i].cpu().numpy(), D_tilde[i].detach().cpu().numpy())).transpose() )
                #     ax[0].set_title("raw demand")
                #     ax[1].set_title("priv demand")
                #     ax[2].set_title("sampled demand plot")
                #     ax[2].legend(['raw', 'priv'])
                #     plt.tight_layout()
                #     plt.savefig('%s/demand_visual_iter_%d.png'%(dir_folder,k))
                #     plt.close('all')


            # last_json_path = os.path.join(dir_folder, "metrics_val_last_weights.json")
            # bUtil.save_dict_to_json(val_metrics, last_json_path)


# params = dict(c_i=0.99, c_o=0.98, eta_eff=0.95, T=48, B=1.5, beta1=0.6, beta2=0.4, beta3=0.5, alpha=0.2)
# for xi in [10, 50]:
#     run_battery(dataloader_dict['train'], params=params, iter_max=4001, iter_save=200,
#                 lr=1e-3, xi=xi, tradeoff_beta1=1, tradeoff_beta2=1, savefig=True, verbose=1)


def train_clf(dataloader, model_clf, model_g, optimizer_clf):
    for k, (D, Y) in enumerate(dataloader):
        # print(D, Y)
        model_g.train(False)
        # D = F.normalize(D, p=1, dim=1)
        optimizer_clf.zero_grad()
        y_labels = bUtil.convert_binary_label(Y, 1500)  # row vector
        y_onehot = bUtil.convert_onehot_soft(y_labels.unsqueeze(1), alphabet_size=2)
        D_tilde, z_noise = model_g.forward(D, y_onehot)

        y_out = model_clf(D_tilde)
        loss_priv = F.cross_entropy(y_out, y_labels, weight=None,
                                    ignore_index=-100, reduction='mean')

        loss_priv.backward()
        optimizer_clf.step()
        if k % 50 == 0:
            print("retrian clf loss: {:.4f}".format(loss_priv))

    model_g.train(True)

    # raise NotImplementedError





if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    # Load the parameters from json file

    parser.add_argument('--model_dir', default='experiments/models', help="Directory containing params.json")
    parser.add_argument('--save_dir', default='experiments/models_logs_mask', help="Directory of models logs")
    parser.add_argument('--param_file', default="param_set_01")
    parser.add_argument('--p_opt', default='LMP', help='price option (TOU or LMP)')
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--load_pretrain_step', default=701, type=int)

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, args.param_file+'.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = bUtil.Params(json_path)
    # print(*params.dict)
    save_folder = args.save_dir + '_' + args.p_opt
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        print("create a folder")

    run_battery_train(dataloader_dict['train'], dataloader_dict['test'], params=params.dict, iter_max=params.iter_max, iter_save=params.iter_save,
                lr=params.learning_rate, xi=params.xi,
                tradeoff_beta1=params.tradeoff_beta1,
                tradeoff_beta2=params.tradeoff_beta2,
                p_opt=args.p_opt,
                save_folder=save_folder,
                reload_step= args.load_pretrain_step,
                savefig=True, verbose=1, n_job=params.num_workers)
