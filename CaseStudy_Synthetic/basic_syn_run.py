import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import processData
import nets
from tqdm import tqdm



dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=1, bsz=128, split=0.15)

def run(dataloader, lr=1e-3, iter_max=10):

    clf = nets.Classifier(z_dim=24, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))

    for i_ in range(iter_max):
        correct_cnt = 0
        tot_cnt = 0
        label_cnt1 = 0
        label_cnt2 = 0
        with tqdm(dataloader) as pbar:
            for X, Y in pbar:
                # print(X, Y)

                X = F.normalize(X, p=1, dim=1)
                y_labels = Y.long().squeeze()
                optimizer_clf.zero_grad()
                y_out = clf(X)
                # y_labels = convert_binary_label(Y-1)
                # y_labels = bUtil.convert_binary_label(Y, name_median[1])
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
                                 acc='{:.3e}'.format(float(correct_cnt) / tot_cnt),
                                 prop1='{:.2e}'.format(float(label_cnt1) / tot_cnt),
                                 prop2='{:.2e}'.format(float(label_cnt2) / tot_cnt)
                                 )
                # acc='{:.3e}'.format(correct.sum() * 1.0 / float(X.size(0)) ))
                pbar.update(10)

        # break

run(dataloader_dict['train'], lr=1e-4, iter_max=1000)