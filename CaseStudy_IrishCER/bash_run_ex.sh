#!/usr/bin/env bash


python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_01" --run=1
python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_01" --run=2
#python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_02"
python basic_run_gan.py --model_dir="experiments/models" --save_dir="experiments/models_logs" --param_file="param_set_03"