#!/bin/bash
dataset_name="shortest_path"
n_cluster=10
num_train_samples=5000
deg=5
smooth_param=1
k_param=1
dim_covs=5

cd CRO/code/train_nn/
python main_AE.py --dataset_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg
python main_deep_kmeans.py --dataset_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg
python main_deep_kmeans.py --dataset_name $dataset_name --dim_covs $dim_covs --n_cluster 1 --num_train_samples $num_train_samples --deg $deg

cd ../solver/
alpha=0.8
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster 1 --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"

alpha=0.9
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster 1 --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"

alpha=0.95
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"
python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster 1 --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"

cd ../../../kNN/
python main_kNN.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha 0.8 --k_param $k_param --smooth_param $smooth_param
python main_kNN.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha 0.9 --k_param $k_param --smooth_param $smooth_param
python main_kNN.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha 0.95 --k_param $k_param --smooth_param $smooth_param

python main_ellipsoid.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha 0.8
python main_ellipsoid.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha 0.9
python main_ellipsoid.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha 0.95

cd ../LUQ/
f_model_name="KernelRidge-poly"
python train_f.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
h_model_name_box="MLP"
python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha 0.8
python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha 0.9
python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha 0.95
h_model_name_ellipsoid="MLP"
python train_2norm_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha 0.8
python train_2norm_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha 0.9
python train_2norm_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha 0.95

cd ../data/
python evaluate_RO.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --k_param $k_param --smooth_param $smooth_param --n_cluster $n_cluster --f_model_name $f_model_name --h_model_name_box $h_model_name_box --h_model_name_ellipsoid $h_model_name_ellipsoid

