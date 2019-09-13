import numpy as np
import processData
import matplotlib.pyplot as plt




dataloader_dict = processData.get_loaders_tth('../training_data.npz', seed=1, bsz=12, split=0.15)

def run(dataloader):
    for k, (X, y) in enumerate(dataloader):
        print(X, y)
        break

run(dataloader_dict['train'])