#!/bin/bash
dataset_name="toy"
n_cluster=10
num_train_samples=500
deg=1
alpha=0.8
smooth_param=1
k_param=1
dim_covs=1

cd kNN/
python main_kNN.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha --k_param $k_param --smooth_param $smooth_param
python main_ellipsoid.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha
if [ $deg -ge 100 ]; then
    cd ../CRO/code/train_nn/
    python main_deep_kmeans.py --dataset_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --deep_main 0
    cd ../../../kNN/
    python main_DCC_notDeep.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha --n_cluster $n_cluster
else
    python main_cluster.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha --n_cluster $n_cluster
fi

cd ../LUQ/
f_model_name="Lasso"
python train_f.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
h_model_name_box="grb"
python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha $alpha
h_model_name_ellipsoid="grb"
python train_2norm_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha $alpha

cd ../data/
python plot_LB_UB.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --k_param $k_param --smooth_param $smooth_param --n_cluster $n_cluster --h_model_name_box $h_model_name_box --h_model_name_ellipsoid $h_model_name_ellipsoid
