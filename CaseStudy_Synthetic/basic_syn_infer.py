import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import processData
import nets
from tqdm import tqdm
import basic_util as bUtil


dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=1, bsz=128, split=0.15)

def run(dataloader, lr=1e-3, iter_max=10):

    clf = nets.Classifier(z_dim=24, y_dim=2)
    optimizer_clf = torch.optim.Adam(clf.parameters(), lr=lr, betas=(0.6, 0.999))
    j=0
    for i_ in range(iter_max):
        correct_cnt = 0
        tot_cnt = 0
        label_cnt1 = 0
        label_cnt2 = 0
        with tqdm(dataloader) as pbar:
            for X, Y in pbar:
                j += 1
                # X = F.normalize(X, p=1, dim=1)
                y_labels = Y.long().squeeze()
                y_onehot_target = bUtil.convert_onehot(Y.long(), 2)
                optimizer_clf.zero_grad()
                y_out = clf(X)
                # loss = F.cross_entropy(y_out, y_labels, weight=None,
                #                        ignore_index=-100, reduction='mean')
                loss = F.binary_cross_entropy_with_logits(y_out, y_onehot_target)
                # loss = - (F.logsigmoid(clf(X))).mean()
                loss.backward()
                optimizer_clf.step()

                # diff=(y_labels.float() - F.sigmoid(y_out)).mean()
                if j % 100 == 0:
                    # print(0.5, F.logsigmoid(y_out))
                    print(F.binary_cross_entropy_with_logits(y_out, y_onehot_target).item())
                    print(y_onehot_target.sum(dim=0)/y_onehot_target.shape[0])


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
                # pbar.set_postfix(iter='{:d}'.format(i_), loss='{:.3e}'.format(loss)) #,
                #                 diff='{:.3e}'.format(diff))
                # acc='{:.3e}'.format(correct.sum() * 1.0 / float(X.size(0)) ))
                pbar.update(10)


###############################
# iter==500 roughly is fine, unnormalized data (0.94) beats normed data (0.68)
run(dataloader_dict['train'], lr=1e-4, iter_max=800)