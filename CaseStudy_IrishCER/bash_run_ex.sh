#!/usr/bin/env bash


#python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_01" --run=1
#python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_01" --run=2
#python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_02" --run=1
#python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_03" --p_opt="TOU" --run=1 --load_pretrain_step=0
#python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_03" --p_opt="LMP" --run=1 --load_pretrain_step=351
python basic_run_gan_mask.py --model_dir="experiments/models" --save_dir="experiments/models_logs_mask" --param_file="param_set_03" --p_opt="TOU" --run=1 --load_pretrain_step=0
python basic_run_gan_mask.py --model_dir="experiments/models" --save_dir="experiments/models_logs_mask" --param_file="param_set_03" --p_opt="LMP" --run=1 --load_pretrain_step=0
python basic_run_gan_mask.py --model_dir="experiments/models" --save_dir="experiments/models_logs_mask" --param_file="param_set_01" --p_opt="TOU" --run=1 --load_pretrain_step=0