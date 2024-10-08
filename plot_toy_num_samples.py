import subprocess

import os
import ipdb

num_train_samples_list = [100,200,1000]
dim_covs = 1
dataset_name="toy"
n_cluster=10

deg=1
alpha=0.8
smooth_param=1
k_param=1
f_model_name="MLP"
h_model_name_box="MLP"
h_model_name_ellipsoid="MLP"

cur_dir = os.getcwd()
# kNN dir is kNN/
kNN_dir = cur_dir+"/kNN/"
CRO_dir = cur_dir+"/CRO/code/train_nn/"
PTC_dir = cur_dir+"/LUQ/"
data_dir = cur_dir+"/data/"

for num_train_samples in num_train_samples_list:
    # go to kNN dir
    os.chdir(kNN_dir)
    # python main_kNN.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha --k_param $k_param --smooth_param $smooth_param
    subprocess.call(["python","main_kNN.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--k_param",str(k_param),"--smooth_param",str(smooth_param)])
    # python main_ellipsoid.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha
    subprocess.call(["python","main_ellipsoid.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha)])
    # python main_cluster.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha --n_cluster $n_cluster
    subprocess.call(["python","main_cluster.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--n_cluster",str(n_cluster)])

    # go to PTC dir
    os.chdir(PTC_dir)

    # python train_f.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
    subprocess.call(["python","train_f.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--model_name",f_model_name])
    # python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name --alpha $alpha
    subprocess.call(["python","train_quantile_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])
    # python train_2norm_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name --alpha $alpha
    subprocess.call(["python","train_2norm_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_ellipsoid,"--alpha",str(alpha)])

    # go to data dir
    os.chdir(data_dir)
    # python plot_LB_UB.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --k_param $k_param --smooth_param $smooth_param --n_cluster $n_cluster --h_model_name_box $h_model_name_box --h_model_name_ellipsoid $h_model_name_ellipsoid
    subprocess.call(["python","plot_LB_UB.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--k_param",str(k_param),"--smooth_param",str(smooth_param),"--n_cluster",str(n_cluster),"--f_model_name",f_model_name,"--h_model_name_box",h_model_name_box,"--h_model_name_ellipsoid",h_model_name_ellipsoid])

# python toy_samples_group_figures.py
subprocess.call(["python","toy_samples_group_figures.py"])