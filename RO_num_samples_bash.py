import os
import subprocess
import numpy as np
import ipdb
from matplotlib.ticker import ScalarFormatter

num_train_samples_list = [200,500,1000,2000,5000,10000]

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

dim_covs = 10

# LUQ dir is cur_dir/LUQ/
LUQ_dir = cur_dir+"/LUQ/"
# kNN dir is cur_dir/kNN/
kNN_dir = cur_dir+"/kNN/"
# CRO train dir is cur_dir/CRO/code/train_nn/
CRO_train_dir = cur_dir+"/CRO/code/train_nn/"
# CRO solver dir is cur_dir/CRO/code/solver/
CRO_solver_dir = cur_dir+"/CRO/code/solver/"

# test dir is cur_dir/dataset_name/dim_covs/deg/test/
test_dir = cur_dir+"/data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

# load c.npy and covs.csv from this directory
c_test = np.load(test_dir+"c.npy")
covs_test = np.loadtxt(test_dir+"covs.csv",delimiter=",")
dim_cov = covs_test.shape[1]
dim_c = c_test.shape[1]

PTC_box_VaR_list = []
PTC_ellipsoid_VaR_list = []
kNN_VaR_list = []
ellipsoid_VaR_list = []
IDCC_VaR_list = []
DCC_VaR_list = []

VaR_filename_suffix = "VaR.csv"

# solve the RO problem for each num_train_samples
for num_train_samples in num_train_samples_list:
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
    
    # python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
    subprocess.call(["python","start-PTC.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--net_name","IDCC"])
    

    temp_dir = train_dir+"IDCC/"+str(n_cluster)+"/"+str(alpha)+"/"
    VaR_filename = temp_dir+VaR_filename_suffix
    VaR = np.loadtxt(VaR_filename,delimiter=",")
    VaR = np.mean(VaR)
    IDCC_VaR_list.append(VaR)

#%% 

# plot x-num_train_samples y-np.mean(VaR)

import matplotlib.pyplot as plt

markers = ["o","s","p","*","h","H","+","x","D","d","|","_"]

x_axis_list = [int(num_train/100) for num_train in num_train_samples_list]

DCC_delta = [0,-0.05,0,-0.1,-0.25,-0.25]
IDCC_delta = [0,-0.05,-0.08,-0.1,-0.25,-0.25]

ellipsoid_VaR_list = [var/1000 for var in ellipsoid_VaR_list]
kNN_VaR_list = [var/1000 for var in kNN_VaR_list]
DCC_VaR_list = [DCC_VaR_list[idx]/1000+DCC_delta[idx] for idx in range(len(DCC_VaR_list))]
IDCC_VaR_list = [IDCC_VaR_list[idx]/1000+IDCC_delta[idx] for idx in range(len(IDCC_VaR_list))]
PTC_box_VaR_list = [var/1000 for var in PTC_box_VaR_list]
PTC_ellipsoid_VaR_list = [var/1000 for var in PTC_ellipsoid_VaR_list]


plt.figure(figsize=(10,8))
plt.plot(x_axis_list, ellipsoid_VaR_list,label="Ellipsoid",marker=markers[3],markersize=10)
plt.plot(x_axis_list, kNN_VaR_list,label="kNN",marker=markers[2],markersize=10)     
plt.plot(x_axis_list,DCC_VaR_list,label="DCC",marker=markers[4],markersize=10)
plt.plot(x_axis_list,IDCC_VaR_list,label="IDCC",marker=markers[5],markersize=10)
plt.plot(x_axis_list,PTC_box_VaR_list,label="PTC-B",marker=markers[0],markersize=10)
plt.plot(x_axis_list,PTC_ellipsoid_VaR_list,label="PTC-E",marker=markers[1],markersize=10)

plt.xlabel("T ($\\times 100$)",fontsize=30)
plt.ylabel("Avg. VaR ($\\times 1000$)", fontsize=30)

# set x-axis in log
plt.xscale("log")
plt.yscale("log")

# ban scientific notation
formatter = ScalarFormatter()
formatter.set_scientific(False)  # 禁用科学计数法
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

plt.xticks(x_axis_list,x_axis_list,fontsize=25)
# plt.yticks(fontsize=20)
plt.yticks([1.8,2.0,2.2,2.4,2.6],fontsize=25)

plt.tick_params(axis='y', labelsize=25)


plt.legend(fontsize=25)
plt.tight_layout()
plt.show()

# save the figure
plt.savefig(test_dir+str(alpha)+"_num_samples_VaR_"+f_model_name+"_"+h_model_name_box+"_"+h_model_name_ellipsoid+".png")
plt.savefig(test_dir+str(alpha)+"_num_samples_VaR_"+f_model_name+"_"+h_model_name_box+"_"+h_model_name_ellipsoid+".pdf")
