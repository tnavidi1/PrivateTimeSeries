import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_training_data(T=24):
    """

    :param T: what is the T ? @Thomas
    :return:
    """
    netDemandFull = 1e3 * np.loadtxt('netDemandFull.csv')  # convert to KW
    netDemand = netDemandFull[76, :] # Pick a node with a lot of solar generation

    N = int(np.floor(len(netDemand) / T))
    x_dat = np.zeros((N * int(netDemandFull.shape[0]), T))
    y_dat = np.zeros((N * int(netDemandFull.shape[0]), 1))
    labels = np.max(netDemandFull, axis=1)
    labels = labels > 145  # if maximum demand is greater than 145 then set income label to 1
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



def load_training_data(filepath='training_data.npz'):
    data = np.load(filepath)
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


def get_loaders_tt(arrays_dict, bsz):
    train_loader = DataLoader(TensorDataset(
        arrays_dict['X_train'], arrays_dict['Y_train']), shuffle=False, batch_size=bsz)
    test_loader  = DataLoader(TensorDataset(
        arrays_dict['X_test'], arrays_dict['Y_test']), shuffle=False, batch_size=bsz)
    return {'train': train_loader, 'test': test_loader}


def get_loaders_tth(filepath='training_data.npz', seed=1, split=0.15, bsz=32):
    X, Y = load_training_data(filepath)
    import sklearn.model_selection as skm
    X_train, X_test, Y_train, Y_test = skm.train_test_split(X, Y, test_size=split, random_state=seed)
    arrays = {'X_train': torch.Tensor(X_train), 'Y_train': torch.Tensor(Y_train),
              'X_test' : torch.Tensor(X_test), 'Y_test': torch.Tensor(Y_test)}

    return get_loaders_tt(arrays, bsz)




