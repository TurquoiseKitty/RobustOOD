import os
import subprocess
import numpy as np
import ipdb


dim_covs_list = [40,80]
num_train_samples_list = [500,1000,2000,5000,10000,20000]

kNN_dim_list = [40,80]


dataset_name="shortest_path"
n_cluster=10

deg=5
smooth_param=1
k_param=1
f_model_name="KernelRidge-rbf"
h_model_name_box="MLP"

alpha = 0.8

cur_dir = os.getcwd()



# LUQ dir is cur_dir/LUQ/
LUQ_dir = cur_dir+"/LUQ/"

# kNN dir is cur_dir/kNN/
kNN_dir = cur_dir+"/kNN/"


PTC_box_VaR_list = [[] for i in range(len(dim_covs_list))]



VaR_filename_suffix = "VaR.csv"

# solve the RO problem for each num_train_samples
for curve_idx,dim_covs in enumerate(dim_covs_list):
    # test dir is cur_dir/dataset_name/dim_covs/deg/test/
    test_dir = cur_dir+"/data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

    # load c.npy and covs.csv from this directory
    c_test = np.load(test_dir+"c.npy")
    dim_c = c_test.shape[1]

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
        PTC_box_VaR_list[curve_idx].append(VaR)

kNN_VaR_list = [[] for i in range(len(kNN_dim_list))]
for curve_idx,dim_covs in enumerate(kNN_dim_list):
    # go to kNN dir
    os.chdir(kNN_dir)
    for num_train_samples in num_train_samples_list:

        # python train_kNN.py --task_name $dataset_name --deg $deg --knn_dim $knn_dim
        subprocess.call(["python","main_kNN.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--k_param",str(k_param),"--smooth_param",str(smooth_param)])
            
        k = max(int(np.ceil(k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*dim_c)

        train_dir = cur_dir+"/data/"+dataset_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
        temp_dir = train_dir+"kNN/"+str(k)+"/"+str(alpha)+"/"
        VaR_filename = temp_dir+VaR_filename_suffix
        VaR = np.loadtxt(VaR_filename,delimiter=",")
        VaR = np.mean(VaR)
        kNN_VaR_list[curve_idx].append(VaR)

#%% 

# go to cur dir
os.chdir(cur_dir)

# plot x-dim_covs y-np.mean(VaR)

import matplotlib.pyplot as plt

markers = ["o","s","p","P","*","h","H","+","x","X","D","d","|","_"]

marker_idx = 0
plt.figure(figsize=(10,10))
for dim_covs in range(len(dim_covs_list)):
    plt.plot(num_train_samples_list,PTC_box_VaR_list[dim_covs],marker=markers[dim_covs],markersize=10,label="BUQ d="+str(dim_covs_list[dim_covs]))
    marker_idx += 1

for knn_dim_idx in range(len(kNN_dim_list)):
    plt.plot(num_train_samples_list,kNN_VaR_list[knn_dim_idx],marker=markers[marker_idx],markersize=10,label="kNN d="+str(kNN_dim_list[knn_dim_idx]),linestyle=":")
    marker_idx += 1


plt.xlabel("T",fontsize=30)
plt.ylabel("Avg. VaR",fontsize=30)

# set x-axis in log
plt.xscale("log")
plt.yscale("log")

plt.xticks(num_train_samples_list,num_train_samples_list,fontsize=25)
plt.yticks([2000,2200,2400,2600,2800],fontsize=25)

plt.legend(fontsize=25)

plt.tight_layout()
plt.show()

# save the figure
plt.savefig(cur_dir+"/data/"+dataset_name+"/PTC-box_"+str(alpha)+"_dim_covs_VaR_"+f_model_name+"_"+h_model_name_box+".png")
plt.savefig(cur_dir+"/data/"+dataset_name+"/PTC-box"+str(alpha)+"_dim_covs_VaR_"+f_model_name+"_"+h_model_name_box+".pdf")
