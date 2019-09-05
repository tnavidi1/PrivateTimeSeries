import torch
import numpy as np
import sys
sys.path.append('..')

import OptMiniModule.util as optMini_util

def convert_onehot(y_label, alphabet_size=6):
    y_ = y_label.long()
    one_hot = torch.FloatTensor(y_.size(0), alphabet_size).zero_()
    y_oh = one_hot.scatter_(1, (y_.data), 1) # if y is from 1 to 6
    return y_oh

def convert_onehot_soft(y_label, alphabet_size=6):
    y_ = y_label.long()
    one_hot = torch.FloatTensor(y_.size(0), alphabet_size).zero_()
    y_oh = one_hot.scatter_(1, (y_.data), 1) # if y is from 1 to 6
    min_v = 0.01
    y_oh = y_oh+min_v
    y_oh_soft = y_oh.clamp(max= 1 - min_v*(alphabet_size-1))
    return y_oh_soft



def convert_binary_label(y_label, median=4):
    y_ = y_label.squeeze()
    y_.apply_(lambda x: 1 if x >=median else 0)
    return y_.long()




def create_price(steps_perHr=2):
    HORIZON = 24
    T1 = 16
    T2 = T1 + 5
    T3 = HORIZON
    rate_offpeak = 0.202
    rate_onpeak = 0.463
    price_shape = np.hstack((rate_offpeak * np.ones((1, T1 * steps_perHr)),
                             rate_onpeak * np.ones((1, (T2-T1) * steps_perHr)),
                             rate_offpeak * np.ones((1, (T3-T2) * steps_perHr ))))
    p = torch.from_numpy(price_shape).to(torch.float).reshape(-1, 1)
    return p



def _form_QP_params(param_set, p=None):
    """

    :param param_set:
    :param p: price
    :return:
    """
    if not isinstance(param_set, dict):
        raise NotImplementedError("wrong type of param set: {}".format( param_set))

    c_i = param_set['c_i']
    c_o = param_set['c_o']
    eta_eff = param_set['eta_eff']
    beta1 = param_set['beta1']
    beta2 = param_set['beta2']
    beta3 = param_set['beta3']
    alpha = param_set['alpha']
    B = param_set['B']
    T = param_set['T']

    G = optMini_util.construct_G_batt_raw(T)
    h = optMini_util.construct_h_batt_raw(T, c_i=c_i, c_o=c_o, batt_B=B)
    A = optMini_util.construct_A_batt_raw(T, eta=eta_eff)
    b = optMini_util.construct_b_batt_raw(T, batt_init=B / 2)
    Q = optMini_util.construct_Q_batt_raw(T, beta1=beta1, beta2=beta2, beta3=beta3)
    q, price = optMini_util.construct_q_batt_raw(T, price=p, batt_B=B, beta3=beta3, alpha=alpha)

    return [Q, q, G, h, A, b, T, price]


def _convert_to_np_arr(X, j):
    """
    if X is 2d array
    :param X:
    :param j:
    :return:
    """
    xs = np.array([x_[j].tolist() for x_ in X])
    return xs

def _convert_to_np_scalars(X, j):
    """
    if X is a 1d array
    :param X:
    :param j:
    :return:
    """
    xs = np.array([x_[j] for x_ in X])
    return xs


def fixed_softplus(x):
    return torch.log1p(torch.exp(x))



########################################
########################################
########################################

import json
import logging
import os
import shutil

import torch


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, filename='last.pth.tar'):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

