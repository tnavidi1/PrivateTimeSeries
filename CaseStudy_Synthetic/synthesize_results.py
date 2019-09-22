"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json
import os
# import torch
import numpy as np
import basic_util as bUtil

import seaborn
import matplotlib.pyplot as plt
from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.

    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_val_best_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

################################
def load_stats_tar(ckpt_path):
    ckpt = bUtil.load_checkopint_gan(ckpt_path, None, None)

    # print(ckpt)
    fig, ax=plt.subplots(1, 3, figsize=(8,4))
    ax[0].plot(ckpt['loss_g'], label='loss_g')
    ax[0].plot(ckpt['loss_a'], label='loss_a')
    ax[1].plot(ckpt['acc'])
    ratios_ =np.array(ckpt['obj_priv']-ckpt['obj_raw'])/np.array(ckpt['obj_raw'])
    ax[2].hist(ratios_, bins=np.arange(12)*0.05 )
    print(ckpt['acc'])
    print(np.mean(ratios_), np.median(ratios_))
    plt.legend()
    plt.show()


#################################
# if __name__ == "__main__":
#     args = parser.parse_args()
#
#     # Aggregate metrics from args.parent_dir directory
#     metrics = dict()
#     aggregate_metrics(args.parent_dir, metrics)  # pass in a empty dict
#     table = metrics_to_table(metrics)
#
#     # Display the table to terminal
#     print(table)
#
#     # Save results in parent_dir/results.md
#     save_file = os.path.join(args.parent_dir, "results.md")
#     with open(save_file, 'w') as f:
#         f.write(table)
#################################



if __name__ == "__main__":
    args = parser.parse_args()
    # Aggregate metrics
    # metrics = dict()
    folder = os.path.join(args.parent_dir, 'models_logs_mask_TOU')
    ckpt_path = os.path.join(folder, 'param_set_08_xi_0000_tb1_0008_tb2_0001_run_1/iter_1800.pth.tar')
    load_stats_tar(ckpt_path)
    ckpt_path = os.path.join(folder, 'param_set_07_xi_0000_tb1_0016_tb2_0001_run_1/iter_1800.pth.tar')
    load_stats_tar(ckpt_path)
    ckpt_path = os.path.join(folder, 'param_set_05_xi_0000_tb1_0032_tb2_0001_run_1/iter_1800.pth.tar')
    load_stats_tar(ckpt_path)

    ckpt_path = os.path.join(folder, 'param_set_06_xi_0000_tb1_0064_tb2_0001_run_1/iter_1800.pth.tar')
    load_stats_tar(ckpt_path)
    # ckpt_path = os.path.join(folder, 'param_set_02_xi_0800_tb1_0000_tb2_0001_run_2/iter_1800.pth.tar')
    # ckpt_path = os.path.join(folder, 'param_set_03_xi_1600_tb1_0010_tb2_0001_run_2/iter_1800.pth.tar')
    # load_stats_tar(ckpt_path)
    ckpt_path = os.path.join(folder, 'param_set_04_xi_0000_tb1_0128_tb2_0001_run_3/iter_0800.pth.tar')
    load_stats_tar(ckpt_path)