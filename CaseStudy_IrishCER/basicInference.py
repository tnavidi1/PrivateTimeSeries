import numpy as np
import matplotlib.pyplot as plt
import processData


data_tt_dict = (processData.get_train_test_split(dir_root='../Data_IrishCER'))
data_tth_dict = (processData.get_train_hold_split(data_tt_dict, 0.9, '../Data_IrishCER/income'))
dataloader_dict = processData.get_loaders_tth(data_tth_dict, bsz=64)

for X, Y in dataloader_dict['train']:
    print(X, Y)
    break
