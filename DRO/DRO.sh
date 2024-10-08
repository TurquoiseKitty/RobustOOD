#!/bin/bash

task_name="knapsack"
num_train_samples=500
deg=4
f_model_name="OLS"

python main_only_f.py --task_name $task_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name

h_model_name="homo"
python main_hetRes.py --task_name $task_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name
h_model_name="example1-2"
python main_hetRes.py --task_name $task_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name

kernel_name="gaussian"
param=1
python main_localDRO.py --task_name $task_name --num_train_samples $num_train_samples --deg $deg --f_model_name $kernel_name --param $param
#E_dro=1
#python main_PTC.py --task_name $task_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --kernel_name $kernel_name --param $param --E_dro $E_dro

E_dro=0
python main_PTC.py --task_name $task_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --kernel_name $kernel_name --param $param --E_dro $E_dro

cd ../data/
python evaluate_DRO.py --task_name $task_name --deg $deg

cd ../DRO/