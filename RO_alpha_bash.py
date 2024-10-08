import os
import subprocess
import ipdb


########################## Note: if change here, change in evaluate_RO.py as well ##############################
alpha_list = [0.6,0.7,0.8,0.85,0.9,0.95]

dataset_name="knapsack" 
n_cluster=10
num_train_samples=5000
# for knapsack, deg=2, for shortest_path, deg=5
if dataset_name=="knapsack":
    deg=2
elif dataset_name=="shortest_path":
    deg=5

smooth_param=1
k_param=1
dim_covs=10 # for both knapsack and shortest_path

f_model_name="KernelRidge-rbf"
h_model_name_box="MLP"
h_model_name_ellipsoid="MLP"

cur_dir = os.getcwd()

# CRO train dir CRO/code/train_nn/
CRO_train_dir = cur_dir+"/CRO/code/train_nn/"
# CRO solver dir CRO/code/solver/
CRO_solver_dir = cur_dir+"/CRO/code/solver/"

# kNN dir
kNN_dir = cur_dir+"/kNN/"

# PTC dir
PTC_dir = cur_dir+"/LUQ/"

data_dir = cur_dir+"/data/"


# got to CRO train dir
os.chdir(CRO_train_dir)
# train IDCC
# python main_AE.py --dataset_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg
subprocess.call(["python", "main_AE.py", "--dataset_name", dataset_name, "--dim_covs", str(dim_covs), "--n_cluster", str(n_cluster), "--num_train_samples", str(num_train_samples), "--deg", str(deg)])
# train DCC
# python main_deep_kmeans.py --dataset_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg
subprocess.call(["python", "main_deep_kmeans.py", "--dataset_name", dataset_name, "--dim_covs", str(dim_covs), "--n_cluster", str(n_cluster), "--num_train_samples", str(num_train_samples), "--deg", str(deg)])

# go to PTC dir
os.chdir(PTC_dir)
# train PTC
# python train_f.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
subprocess.call(["python","train_f.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--model_name",f_model_name])

for alpha in alpha_list:
    # go to CRO solver dir
    os.chdir(CRO_solver_dir)
    # python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"
    subprocess.call(["python","start-PTC.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--net_name","DCC"])
    # python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
    subprocess.call(["python","start-PTC.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--net_name","IDCC"])


    # go to kNN dir
    os.chdir(kNN_dir)
    # python main_kNN.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha --k_param $k_param --smooth_param $smooth_param
    subprocess.call(["python","main_kNN.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--k_param",str(k_param),"--smooth_param",str(smooth_param)])
    # python main_ellipsoid.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --alpha $alpha
    subprocess.call(["python","main_ellipsoid.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha)])

    # go to PTC dir
    os.chdir(PTC_dir)
    # python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha 0.8
    subprocess.call(["python","train_quantile_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])
    # python train_2norm_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha 0.8
    subprocess.call(["python","train_2norm_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_ellipsoid,"--alpha",str(alpha)])

# go to data dir
os.chdir(data_dir)
# python evaluate_RO.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --k_param $k_param --smooth_param $smooth_param --n_cluster $n_cluster --f_model_name $f_model_name --h_model_name_box $h_model_name_box --h_model_name_ellipsoid $h_model_name_ellipsoid
subprocess.call(["python","evaluate_RO.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--k_param",str(k_param),"--smooth_param",str(smooth_param),"--n_cluster",str(n_cluster),"--f_model_name",f_model_name,"--h_model_name_box",h_model_name_box,"--h_model_name_ellipsoid",h_model_name_ellipsoid])
