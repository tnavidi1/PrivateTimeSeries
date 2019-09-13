import os
import torch
import numpy as np
import json
import logging
import shutil


import sys
sys.path.append('..')

import OptMiniModule.util as optMini_util



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

# @create & load the LMP
def create_LMP(filename, granular=24, steps_perHr=2):
    lmp = np.genfromtxt(filename)
    # print(lmp.shape) # 192
    n = int(len(lmp) / granular)
    lmp = lmp.reshape(n, granular)
    # print(lmp)
    # lmp_ = np.concatenate((lmp[0], lmp[1]), axis=1).reshape(-1)
    T = int(steps_perHr * granular)
    if T == 48:
        lmp = np.concatenate((lmp[0][:, np.newaxis], lmp[1][:,np.newaxis]), axis=1).reshape(-1)
    elif T == 24:
        lmp = lmp[0]
    else:
        raise NotImplementedError("---- time steps T={:d}----".format(T))
    # print(lmp_)
    p = torch.from_numpy(lmp).to(torch.float).reshape(-1, 1)
    return p




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

def load_checkopint_gan(checkpoint, model_g, model_clf, optimizer_g=None, optimizer_clf=None):
    """
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint

    :param checkpoint:
    :param model_g: (nn.Model) generator
    :param model_clf: (nn.Model) classifier
    :param optimizer_g:
    :param optimizer_clf:
    :return:
    """

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_clf.load_state_dict(checkpoint['clf_state_dict'])

    if not optimizer_g or not optimizer_clf:
        optimizer_g.load_state_dict(checkpoint['g_optim_dict'])
        optimizer_clf.load_state_dict(checkpoint['clf_optim_dict'])

    return checkpoint

