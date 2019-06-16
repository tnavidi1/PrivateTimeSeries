import numpy as np
import matplotlib.pyplot as plt

import optimizer
import neural_models

import argparse
import torch
from torch.nn import functional as F
import torchvision
from pprint import pprint
from torchvision import datasets, transforms
import os

import tqdm

def generate_training_data():
    netDemandFull = 1000 * np.loadtxt('netDemandFull.csv')  # convert to KW Data is 50% solar
    netDemand = netDemandFull[76, :]

    N = int(np.floor(len(netDemand) / T))
    x_dat = np.zeros((N * int(netDemandFull.shape[0]), T))
    y_dat = np.zeros((N * int(netDemandFull.shape[0]), 1))
    labels = np.max(netDemandFull, axis=1)
    labels = labels > 145  # if maximum demand is greater than 50 then set income label to 1
    print('number of high income nodes out of 123', np.sum(labels))

    idx = 0
    # place data into training samples
    for i in range(N):
        for j in range(netDemandFull.shape[0]):
            if np.sum(netDemandFull[j, i * T:T * (i + 1)]) < 1:
                # print('skipping zero load sample')
                continue
            x_dat[idx, :] = netDemandFull[j, i * T:T * (i + 1)]  # split data into 24 hour segments
            y_dat[idx, :] = labels[j]  # assign the label of the node
            # y_dat[idx, :] = np.mean(netDemandFull[j, i * T:T * (i + 1)]) > 30
            idx += 1
    x_dat = x_dat[0:idx, :]
    y_dat = y_dat[0:idx, :]

    print('percentage of high income examples', np.sum(y_dat)/idx)

    print('shape of training data', x_dat.shape)

    np.savez('training_data.npz', x_dat=x_dat, y_dat=y_dat)

    return True

def load_training_data():
    data = np.load('training_data.npz')
    x_dat = data['x_dat']
    y_dat = data['y_dat']

    return x_dat, y_dat

def normalize_data(X):
    data_mean = np.mean(X)
    data_std = np.std(X)
    X = (X - data_mean)/data_std
    return X, data_mean, data_std

def unnormalize_data(X, data_mean, data_std):
    X = X*data_std + data_mean
    return X


def gaussian_mechanism(d, u_opt, T):
    filters = (np.arange(20)+1.)*2.
    costs = []
    for filt in filters:
        d_hat = d + filt * np.random.normal(0, 1, (T, 1))

        # find optimal u and Q using d
        u_star, Q_star = u_opt.u_star_of_d_np(d_hat)

        cost = u_opt.costU_np(u_star, Q_star, d)
        # print(cost.shape)
        costs.append(cost.flatten())

    return costs

def convert_onehot(yu, alphabet_size=2):
    #yun = yu.unsqueeze(1)  # make 2-d if not already
    yun = yu

    one_hot = torch.FloatTensor(yu.size(0), alphabet_size).zero_()
    y_oh = one_hot.scatter_(1, yun.data, 1)

    return y_oh

def getBatch(training_data, batch_size):
    x = training_data[0]
    y = training_data[1]

    n, T = x.shape

    ids = np.random.choice(n, size=batch_size, replace=False)

    x_train = x[ids, :]
    y_train = y[ids]

    return x_train, y_train


def GanLoss(g, d_adv, u_opt, x_real, y_target, eps_=1.0, BETA=1.0, LAMBDA=10.0):

    y_onehot = convert_onehot(y_target, alphabet_size=2)

    x_priv = g(x_real, y_onehot)

    y_out = d_adv(x_priv)

    u_out, Q_out = u_opt.u_star_of_d(x_priv)

    # print('shapes of ys', y_out.shape)
    # print(y_target.squeeze().shape)

    adv_loss = F.cross_entropy(y_out, y_target.squeeze(), weight=None,
                               ignore_index=-100, reduction='mean')

    util_loss = u_opt.costU(u_out, Q_out, x_real)

    distortion = (torch.clamp((x_real - x_priv).norm(2, dim=1) - eps_, min=0) ** 2).mean()  # hinge squared loss

    ALPHA = 1.
    # ALPHA = 1./10000

    g_loss = - BETA * adv_loss + ALPHA * util_loss + LAMBDA * distortion
    # pass
    return g_loss, adv_loss, util_loss, distortion

def evaluate_opt(g, u_opt, data_set):
    X, y = data_set

    y = torch.LongTensor(y)
    X = torch.FloatTensor(X)

    y_onehot = convert_onehot(y, alphabet_size=2)
    x_priv = g(X, y_onehot)

    # print(x_priv.shape)

    u_out, Q_out = u_opt.u_star_of_d(x_priv)

    # print(u_out.detach().numpy()[:, 10])

    """
    sample = 422
    plt.figure()
    plt.plot(X.detach().numpy()[sample, :])
    plt.plot(X.detach().numpy()[sample, :] + u_out.detach().numpy()[:, sample])
    plt.figure()
    plt.plot(X.detach().numpy()[sample, :])
    plt.plot(x_priv.detach().numpy()[sample, :])
    plt.show()
    """

    util_loss = u_opt.costU(u_out, Q_out, X)

    print('utility loss', util_loss)

    return util_loss

def evaluate_opt_pre(u_opt, data_set):
    X, y = data_set

    X = torch.FloatTensor(X)

    u_out, Q_out = u_opt.u_star_of_d(X)

    # print(u_out.detach().numpy()[:, 10])

    """
    sample = 422
    plt.figure()
    plt.plot(X.detach().numpy()[sample, :])
    plt.plot(X.detach().numpy()[sample, :] + u_out.detach().numpy()[:, sample])
    """

    util_loss = u_opt.costU(u_out, Q_out, X)

    print('pre noise utility loss', util_loss)

    return util_loss

def evaluate_classifier(g, d_priv, data_set):
    X, y = data_set

    y = torch.LongTensor(y)
    X = torch.FloatTensor(X)

    y_onehot = convert_onehot(y, alphabet_size=2)
    x_priv = g(X, y_onehot)

    output_priv = d_priv(x_priv)

    accuracy_priv = (output_priv.argmax(1) == y.squeeze()).float().mean()

    print('classification accuracy', accuracy_priv)

    return accuracy_priv

def evaluate_classifier_pre(d_priv, data_set):
    X, y = data_set

    y = torch.LongTensor(y)
    X = torch.FloatTensor(X)

    output_priv = d_priv(X)

    # print(output_priv.argmax(1))
    # print(y.shape)

    accuracy_priv = (output_priv.argmax(1) == y.squeeze()).float().mean()

    print('pre noise classification accuracy', accuracy_priv)

    return accuracy_priv


def training(g, d_priv, u_opt, training_data, args, sub_folder, lr=1e-3, eps_=1.0, beta_=1.0, lambd_=10.0):

    optimizer_d_priv = torch.optim.Adam(d_priv.parameters(), lr=lr, betas=(0.6, 0.999))

    optimizer_g = torch.optim.Adam([{'params': g.filter.parameters()}], lr=lr*100, betas=(0.6, 0.999))

    steps = 0

    losses_u = []
    losses_d = []
    losses_g = []

    while steps < args.iter_max:
        steps += 1  # num of gradient steps taken by end of loop iteration

        optimizer_d_priv.zero_grad()
        optimizer_g.zero_grad()

        x_real, y_target = getBatch(training_data, args.batch_size)
        y_target = torch.LongTensor(y_target)
        x_real = torch.FloatTensor(x_real)

        g_loss, adv_loss, util_loss, distortion = GanLoss(g, d_priv, u_opt, x_real, y_target, eps_, beta_, lambd_)

        adv_loss.backward(retain_graph=True)
        optimizer_d_priv.step()

        g_loss.backward()
        optimizer_g.step()

        if steps % args.iter_save == 0:
            print('Saving model checkpoint at steps:', steps)
            print('loss_g={:.2e}'.format(g_loss), 'loss_adv={:.2e}'.format(beta_*adv_loss),
                    'loss_util={:.2e}'.format(util_loss), 'loss_dist={:.2e}'.format(lambd_ * distortion))
            neural_models.save_model_by_desc(g, sub_folder, "generator", steps, root='s_cost')
            neural_models.save_model_by_desc(d_priv, sub_folder, "d_priv", steps, root='s_cost')

            print('distortion loss', lambd_ * distortion)

        losses_u.append(util_loss)
        losses_d.append(beta_ * adv_loss)
        losses_g.append(g_loss)

    return losses_d, losses_g, losses_u


def training_classifier(d_priv, training_data, args, sub_folder, g=None, lr=1e-2, eps_=1.0, beta_=1.0, lambd_=10.0):

    optimizer_d_priv = torch.optim.Adam(d_priv.parameters(), lr=lr, betas=(0.6, 0.999))

    steps = 0

    losses_d = []

    while steps < args.iter_max:
        steps += 1  # num of gradient steps taken by end of loop iteration

        optimizer_d_priv.zero_grad()

        x_real, y_target = getBatch(training_data, args.batch_size)
        y_target = torch.LongTensor(y_target)
        x_real = torch.FloatTensor(x_real)

        #print('x shape', x_real)

        if g is not None:
            y_onehot = convert_onehot(y_target, alphabet_size=2)
            x_real = g(x_real, y_onehot)

        y_out = d_priv(x_real)

        #print('y shape', y_out.shape)
        #print('y target', y_target)

        adv_loss = F.cross_entropy(y_out, y_target.squeeze(), weight=None,
                                   ignore_index=-100, reduction='mean')

        # print(adv_loss)

        adv_loss.backward()
        optimizer_d_priv.step()

        if steps % args.iter_save == 0:
            #print(y_out)
            #print(y_target)
            print('Saving model checkpoint at steps:', steps)
            print('loss_adv={:.2e}'.format(adv_loss))
            neural_models.save_model_by_desc(d_priv, sub_folder, "d_priv", steps, root='s_cost')

        losses_d.append(adv_loss)

    return losses_d


def main(args):
    # define constants of problem
    price_e = np.hstack((.202 * np.ones((1, 12)), .463 * np.ones((1, 6)), .202 * np.ones((1, 6))))
    T = 24  # number of hours in optimization horizon
    # print('price', price_e.shape)
    u_min = -4 * 1  # multiply by 20 to represent 50% of homes with storage
    Q_min = 0.1
    Q_max = 12 * 1
    u_max = -u_min
    Q0 = 0  # about 50% capacity)

    # generate training data if necessary or just load it
    # generate_training_data()
    x_dat, y_dat = load_training_data()
    # normalize data to have 0 mean 1 std
    # x_dat, data_mean, data_std = normalize_data(x_dat)
    training_data = (x_dat, y_dat)

    # initialize models
    d_priv = neural_models.ClassifierLatent(z_dim=T, y_dim=2)
    g = neural_models.Generator(z_dim=T, y_priv_dim=2, device=None)
    # initialize optimizer
    opt = optimizer.Optimizer(T, price_e, u_max, Q_max, Q0)
    # Get lagrangian matrix inverse
    F_inv, gamma, mu = opt.getFinv()

    # for gaussian mechanism
    if args.gauss:
        print('runnning gaussian mechanism')
        fcosts = []
        n, T = x_dat.shape
        #print(x_dat.shape)
        for i in range(n):
            d = x_dat[i, :].reshape((T,1))
            #print(d.shape)
            costs = gaussian_mechanism(d, opt, T)
            fcosts.append(costs)
        np.savez('gaussMech', fcosts=fcosts)
        print('saved gauss mech')
        return

    opt_name = 's_cost'
    label_name = 'income'
    train_setting = [
        ('opt={:s}=', opt_name),
        ('label={:s}=', label_name),
        ('eps={:.2f}', args.eps),
        ('beta={:.2f}', args.beta),
        ('lambd={:.2f}', args.lambd)
    ]

    sub_folder = '_'.join([t.format(v) for (t, v) in train_setting])
    print(' checkpoint location', sub_folder)

    if args.train:
        # run training

        # just classifier with no noise
        # losses_d = training_classifier(d_priv, training_data, args, sub_folder, lr=1e-3, eps_=args.eps,
        #                    beta_=args.beta, lambd_=args.lambd)
        # evaluate_classifier_pre(d_priv, training_data)

        # gan
        evaluate_opt_pre(opt, training_data)
        losses_d, losses_g, losses_u = training(g, d_priv, opt, training_data, args, sub_folder, lr=1e-3, eps_=args.eps,
                                                beta_=args.beta, lambd_=args.lambd)
        acc_priv = evaluate_classifier(g, d_priv, training_data)
        acc_util = evaluate_opt(g, opt, training_data)

        """
        plt.figure()
        plt.plot(losses_d)
        plt.figure()
        plt.plot(losses_g)
        plt.figure()
        plt.plot(losses_u)
        plt.show()
        """

        return acc_priv.detach().numpy(), acc_util.detach().numpy(), acc_priv_retrain.detach().numpy()


    else:
        testing()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', type=int, default=0, help="Train or test?")
    parser.add_argument('--gauss', type=int, default=0, help="dp or g filt")
    parser.add_argument('--iter_max', type=int, default=20, help="Number of training iterations")
    parser.add_argument('--iter_save', type=int, default=10, help="Number of iterations before save")
    parser.add_argument('--batch_size', type=int, default=10, help="size of training batch")
    parser.add_argument('--eps', default=1, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--lambd', default=10, type=float)
    args = parser.parse_args()

    # for one arg
    main(args)


    """
    # for running many args many times
    reps = 10
    faccs_p = []
    faccs_p_r = []
    faccs_u = []
    for i in range(reps):

        epses = (np.arange(20)+1) * 10

        accs_p = []
        accs_p_r = []
        accs_u = []

        for ep in epses:
            args.eps = ep
            acc_priv, acc_util, acc_priv_retrain = main(args)
            accs_p.append(acc_priv)
            accs_p_r.append(acc_priv_retrain)
            accs_u.append(acc_util)

        faccs_p.append(accs_p)
        faccs_p_r.append(accs_p_r)
        faccs_u.append(accs_u)

        np.savez('cost_vs_distortion.npz', faccs_u=faccs_u, faccs_p=faccs_p, faccs_p_r=faccs_p_r)
        print('accs u', accs_u)
        print('accs p', accs_p)
        print('accs p', accs_p_r)

    np.savez('cost_vs_distortion.npz', faccs_u=faccs_u, faccs_p=faccs_p, faccs_p_r=faccs_p_r)
    print('accs u', faccs_u)
    print('accs p', faccs_p)
    print('accs p', faccs_p_r)
    """


