import os
import subprocess
import numpy as np



dim_covs_list = [2,4,10,20,40]

num_train_samples = 5000

dataset_name="shortest_path"
n_cluster=10

deg=5
smooth_param=1
k_param=1
f_model_name="KernelRidge-rbf"
h_model_name_box="MLP"
h_model_name_ellipsoid="grb"
alpha = 0.8

cur_dir = os.getcwd()



# LUQ dir is cur_dir/LUQ/
LUQ_dir = cur_dir+"/LUQ/"
# kNN dir is cur_dir/kNN/
kNN_dir = cur_dir+"/kNN/"
# CRO train dir is cur_dir/CRO/code/train_nn/
CRO_train_dir = cur_dir+"/CRO/code/train_nn/"
# CRO solver dir is cur_dir/CRO/code/solver/
CRO_solver_dir = cur_dir+"/CRO/code/solver/"



PTC_box_VaR_list = []
PTC_ellipsoid_VaR_list = []
kNN_VaR_list = []
ellipsoid_VaR_list = []
IDCC_VaR_list = []
DCC_VaR_list = []

VaR_filename_suffix = "VaR.csv"

# solve the RO problem for each num_train_samples
for dim_covs in dim_covs_list:
    # test dir is cur_dir/dataset_name/dim_covs/deg/test/
    test_dir = cur_dir+"/data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

    # load c.npy and covs.csv from this directory
    c_test = np.load(test_dir+"c.npy")
    dim_c = c_test.shape[1]

    # train dir is cur_dir/dataset_name/01/deg/train/num_train_samples/
    train_dir = cur_dir+"/data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"

    # go to LUQ dir
    os.chdir(LUQ_dir)
    # python train_f.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
    subprocess.call(["python","train_f.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--model_name",f_model_name])
    # python train_quantile_h.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha $alpha
    subprocess.call(["python","train_quantile_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])
    
    temp_dir = train_dir+"/LUQ/"+f_model_name+"/quantile/"+h_model_name_box+"/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    PTC_box_VaR_list.append(VaR)
    
    # python train_2norm_h.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha $alpha
    subprocess.call(["python","train_2norm_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_ellipsoid,"--alpha",str(alpha)])

    temp_dir = train_dir+"/LUQ/"+f_model_name+"/norm/"+h_model_name_ellipsoid+"/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    PTC_ellipsoid_VaR_list.append(VaR)

    # go to kNN dir
    os.chdir(kNN_dir)
    #python main_kNN.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --alpha $alpha --k_param $k_param --smooth_param $smooth_param
    subprocess.call(["python","main_kNN.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--k_param",str(k_param),"--smooth_param",str(smooth_param)])
    
    k = max(int(np.ceil(k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*dim_c)

    temp_dir = train_dir+"kNN/"+str(k)+"/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    kNN_VaR_list.append(VaR)

    # python main_ellipsoid.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --alpha $alpha
    subprocess.call(["python","main_ellipsoid.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha)])

    temp_dir = train_dir+"ellipsoid/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    ellipsoid_VaR_list.append(VaR)

    # go to CRO train dir
    os.chdir(CRO_train_dir)
    # python main_AE.py --dataset_name $dataset_name --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg
    subprocess.call(["python","main_AE.py","--dataset_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg)])
    # python main_deep_kmeans.py --dataset_name $dataset_name --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg
    subprocess.call(["python","main_deep_kmeans.py","--dataset_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg)])

    # go to CRO solver dir
    os.chdir(CRO_solver_dir)
    # python start-PTC.py --task_name $dataset_name --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "DCC"
    subprocess.call(["python","start-PTC.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--net_name","DCC"])
    
    temp_dir = train_dir+"DCC/"+str(n_cluster)+"/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    DCC_VaR_list.append(VaR)
    
    # python start-PTC.py --task_name $dataset_name --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
    subprocess.call(["python","start-PTC.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--net_name","IDCC"])

    temp_dir = train_dir+"IDCC/"+str(n_cluster)+"/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    IDCC_VaR_list.append(VaR)

#%% 

# go to cur dir
os.chdir(cur_dir)

# plot x-dim_covs y-np.mean(VaR)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot(dim_covs_list,PTC_box_VaR_list,label="PTC_box",marker="o")
plt.plot(dim_covs_list,PTC_ellipsoid_VaR_list,label="PTC_ellipsoid",marker="v")
plt.plot(dim_covs_list,kNN_VaR_list,label="kNN",marker="<")               
plt.plot(dim_covs_list,ellipsoid_VaR_list,label="ellipsoid",marker=">")
plt.plot(dim_covs_list,DCC_VaR_list,label="DCC",marker="s")
plt.plot(dim_covs_list,IDCC_VaR_list,label="IDCC",marker="p")

plt.xlabel("the number of covariates")
plt.ylabel("VaR")

# set x-axis in log
plt.xscale("log")

plt.xticks(dim_covs_list,dim_covs_list)

plt.legend()
plt.show()

# save the figure
plt.savefig(cur_dir+"/data/"+dataset_name+"/"+str(alpha)+"_dim_covs_VaR_"+f_model_name+"_"+h_model_name_box+"_"+h_model_name_ellipsoid+".png")
plt.savefig(cur_dir+"/data/"+dataset_name+"/"+str(alpha)+"_dim_covs_VaR_"+f_model_name+"_"+h_model_name_box+"_"+h_model_name_ellipsoid+".pdf")
