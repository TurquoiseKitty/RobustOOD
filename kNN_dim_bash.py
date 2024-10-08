import os
import subprocess
import numpy as np



dim_covs_list = [2,4,10,20,40,80]
num_train_samples_list = [200,500,1000,2000,5000,10000]


dataset_name="shortest_path"
n_cluster=10

deg=5
smooth_param=1
k_param=1

alpha = 0.8

cur_dir = os.getcwd()



# kNN dir is cur_dir/kNN/
kNN_dir = cur_dir+"/kNN/"



kNN_VaR_list = [[] for i in range(len(dim_covs_list))]


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
        os.chdir(kNN_dir)
        #python main_kNN.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --alpha $alpha --k_param $k_param --smooth_param $smooth_param
        subprocess.call(["python","main_kNN.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--k_param",str(k_param),"--smooth_param",str(smooth_param)])
        
        k = max(int(np.ceil(k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*dim_c)

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

marker = ["o","v","<",">","s","p"]

plt.figure(figsize=(10,10))
for dim_covs in range(len(dim_covs_list)):
    plt.plot(num_train_samples_list,kNN_VaR_list[dim_covs],marker=marker[dim_covs],label="d="+str(dim_covs_list[dim_covs]))


plt.xlabel("T")
plt.ylabel("VaR")

# set x-axis in log
plt.xscale("log")
plt.yscale("log")

plt.xticks(num_train_samples_list,num_train_samples_list)

plt.legend()
plt.show()

# save the figure
plt.savefig(cur_dir+"/data/"+dataset_name+"/kNN_"+str(alpha)+"_dim_covs_VaR.png")
plt.savefig(cur_dir+"/data/"+dataset_name+"/kNN_"+str(alpha)+"_dim_covs_VaR.pdf")
