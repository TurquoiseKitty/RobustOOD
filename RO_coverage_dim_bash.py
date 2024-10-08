import os
import subprocess
import numpy as np



dim_covs_list = [10,20,40,80]
num_train_samples_list = [50,100,200,500,1000,2000,5000,10000]



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




PTC_box_Coverage_list = [[] for i in range(len(dim_covs_list))]


Coverage_filename_suffix = "coverage.csv"

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
        Coverage_filename = temp_dir+Coverage_filename_suffix
        Coverage = np.loadtxt(Coverage_filename,delimiter=",")
        Coverage = np.mean(Coverage-alpha)
        PTC_box_Coverage_list[curve_idx].append(Coverage)
    
    

#%% 

# go to cur dir
os.chdir(cur_dir)

# plot x-dim_covs y-np.mean(Coverage)

import matplotlib.pyplot as plt

marker = ["o","v","<",">","s","p"]

plt.figure(figsize=(10,10))
for dim_covs in range(len(dim_covs_list)):
    plt.plot(num_train_samples_list,PTC_box_Coverage_list[dim_covs],marker=marker[dim_covs],label="d="+str(dim_covs_list[dim_covs]))


plt.xlabel("T")
plt.ylabel("Coverage Error")

# set x-axis in log
plt.xscale("log")
#plt.yscale("log")

plt.xticks(num_train_samples_list,num_train_samples_list)

plt.legend()
plt.show()

# save the figure
plt.savefig(cur_dir+"/data/"+dataset_name+"/PTC-box_coverage_"+str(alpha)+"_dim_covs_"+f_model_name+"_"+h_model_name_box+".png")
plt.savefig(cur_dir+"/data/"+dataset_name+"/PTC-box_coverage"+str(alpha)+"_dim_covs_"+f_model_name+"_"+h_model_name_box+".pdf")
