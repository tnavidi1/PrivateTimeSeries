import torch
from tqdm import tqdm
import processData
import nets
import sys, os
sys.path.append("..")

import time
import cvxpy as cp

import matplotlib.pyplot as plt

import OptMiniModule.util as optMini_util
import OptMiniModule.cvx_runpass as optMini_cvx
import OptMiniModule.diffcp.cones as cone_lib

import numpy as np
desired_width = 300
np.set_printoptions(precision=4, linewidth=desired_width, threshold=5000) # threshold=150*150

torch.set_printoptions(profile="full", linewidth=400)

data_tt_dict = processData.get_train_test_split(dir_root='../Data_IrishCER', attr='floor')
data_tth_dict = processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/floor')
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=40)