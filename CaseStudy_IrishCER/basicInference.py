import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import processData
import nets

# print(torch.__version__)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=64)


def convert_onehot(y_label, alphabet_size=6):
    y_ = y_label.long()
    one_hot = torch.FloatTensor(y_.size(0), alphabet_size).zero_()
    y_oh = one_hot.scatter_(1, (y_.data), 1) # if y is from 1 to 6
    return y_oh

def convert_binary_label(y_label, median=4):
    y_ = y_label.squeeze()
    # print(y_)
    y_.apply_(lambda x: 1 if x >=median else 0)
    # print(y_)
    # raise NotImplementedError
    return y_.long()


def run_raw_classification(dataloader, lr=1e-3, iter_max=10):

    clf = nets.Classifier(z_dim=48, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))

    # criterion = nn.CrossEntropyLoss()
    for i_ in range(iter_max):
        correct_cnt = 0
        tot_cnt = 0
        label_cnt1 = 0
        label_cnt2 = 0
        with tqdm(dataloader) as pbar:
            for X, Y in pbar:
                # print(X, Y)
                optimizer_clf.zero_grad()
                y_out= clf(X)
                # y_labels = convert_binary_label(Y-1)
                y_labels = convert_binary_label(Y, 1500)
                # y_labels = (Y-1).long().squeeze()
                loss = F.cross_entropy(y_out, y_labels, weight=None,
                                       ignore_index=-100, reduction='mean')
                loss.backward()
                optimizer_clf.step()
                _, y_max_idx = torch.max(y_out, dim=1)
                # print(y_max_idx, y_labels)
                correct = y_max_idx == y_labels
                # print(correct)
                correct_cnt += correct.sum()
                tot_cnt += X.shape[0]
                label_cnt1 += (y_labels == 0).sum()
                label_cnt2 += (y_labels == 1).sum()

                pbar.set_postfix(iter='{:d}'.format(i_), loss='{:.3e}'.format(loss),
                                 cor_cnts='{:d}'.format(correct_cnt),
                                 tot_cnts='{:d}'.format(tot_cnt),
                                 acc='{:.3e}'.format(float(correct_cnt)/tot_cnt),
                                 prop1='{:.2e}'.format(float(label_cnt1)/tot_cnt),
                                 prop2='{:.2e}'.format(float(label_cnt2)/tot_cnt)
                                 )
                                 # acc='{:.3e}'.format(correct.sum() * 1.0 / float(X.size(0)) ))
                pbar.update(10)
                # break
                # print("iter={:d}, loss={:3e}, acc={:3e}".format(_, loss.data, correct.sum()/X.size(0)))

                # pbar.set_postfix(correct_counts='{:d}'.format(correct_cnt),
                #                  total_counts='{:d}'.format(tot_cnt),
                #                  acc='{:.3e}'.format(float(correct_cnt)/tot_cnt))

run_raw_classification(dataloader_dict['train'], lr=2e-4, iter_max=400)
