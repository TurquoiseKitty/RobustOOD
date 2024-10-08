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

PTC_box_coverage_list = []
PTC_ellipsoid_coverage_list = []
kNN_coverage_list = []
ellipsoid_coverage_list = []
IDCC_coverage_list = []
DCC_coverage_list = []

coverage_filename_suffix = "coverage.csv"

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
    coverage_filename = temp_dir+coverage_filename_suffix
    coverage = np.loadtxt(coverage_filename,delimiter=",")
    # coverage = np.mean(coverage)
    PTC_box_coverage_list.append(coverage)
    
    # python train_2norm_h.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_ellipsoid --alpha $alpha
    subprocess.call(["python","train_2norm_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_ellipsoid,"--alpha",str(alpha)])

    temp_dir = train_dir+"/LUQ/"+f_model_name+"/norm/"+h_model_name_ellipsoid+"/"+str(alpha)+"/"
    coverage_filename = temp_dir+coverage_filename_suffix
    coverage = np.loadtxt(coverage_filename,delimiter=",")
    # coverage = np.mean(coverage)
    PTC_ellipsoid_coverage_list.append(coverage)

    # go to kNN dir
    os.chdir(kNN_dir)
    #python main_kNN.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --alpha $alpha --k_param $k_param --smooth_param $smooth_param
    subprocess.call(["python","main_kNN.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--k_param",str(k_param),"--smooth_param",str(smooth_param)])
    
    k = max(int(np.ceil(k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*dim_c)

    temp_dir = train_dir+"kNN/"+str(k)+"/"+str(alpha)+"/"
    coverage_filename = temp_dir+coverage_filename_suffix
    coverage = np.loadtxt(coverage_filename,delimiter=",")
    # coverage = np.mean(coverage)
    kNN_coverage_list.append(coverage)

    # python main_ellipsoid.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --alpha $alpha
    subprocess.call(["python","main_ellipsoid.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha)])

    temp_dir = train_dir+"ellipsoid/"+str(alpha)+"/"
    coverage_filename = temp_dir+coverage_filename_suffix
    coverage = np.loadtxt(coverage_filename,delimiter=",")
    # coverage = np.mean(coverage)
    ellipsoid_coverage_list.append(coverage)

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
    coverage_filename = temp_dir+coverage_filename_suffix
    coverage = np.loadtxt(coverage_filename,delimiter=",")
    # coverage = np.mean(coverage)
    DCC_coverage_list.append(coverage)
    
    # python start-PTC.py --task_name $dataset_name --dim_covs $dim_covs --n_cluster $n_cluster --num_train_samples $num_train_samples --deg $deg --alpha $alpha --net_name "IDCC"
    subprocess.call(["python","start-PTC.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--n_cluster",str(n_cluster),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--alpha",str(alpha),"--net_name","IDCC"])
    

    temp_dir = train_dir+"IDCC/"+str(n_cluster)+"/"+str(alpha)+"/"
    coverage_filename = temp_dir+coverage_filename_suffix
    coverage = np.loadtxt(coverage_filename,delimiter=",")
    # coverage = np.mean(coverage)
    IDCC_coverage_list.append(coverage)

#%% 

# plot x-num_train_samples y-np.mean(coverage)

import matplotlib.pyplot as plt

markers = ["o","s","p","*","h","H","+","x","D","d","|","_"]

x_axis_list = [int(num_train) for num_train in num_train_samples_list]

# use boxplot to show the distribution of coverage, where the x-axis is the number of training samples, the y-axis is the coverage

# gather the coverage of each method to a dataframe
import pandas as pd

# create a dataframe, where the colomns correspond to the method name, the coverage value, and the number of training samples
coverage_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
for sample_idx in range(len(num_train_samples_list)):
    num_sample = num_train_samples_list[sample_idx]

    # construct coverage temp dataframe to be concated to coverage_df
    # ellipsoid
    temp_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
    coverage = ellipsoid_coverage_list[sample_idx]
    coverage_arr = np.array(coverage)
    temp_df["coverage"] = coverage_arr.reshape(-1)
    temp_df["num_sample"] = num_sample*np.ones(coverage_arr.shape[0])
    # repeat the str alg_name for coverage_arr.shape[0] times
    temp_df["alg_name"] = ["Ellipsoid"]*coverage_arr.shape[0]
    coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)

    

    # kNN
    temp_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
    coverage = kNN_coverage_list[sample_idx]
    coverage_arr = np.array(coverage)
    temp_df["coverage"] = coverage_arr.reshape(-1)
    temp_df["num_sample"] = num_sample*np.ones(coverage_arr.shape[0])
    # repeat the str alg_name for coverage_arr.shape[0] times
    temp_df["alg_name"] = ["kNN"]*coverage_arr.shape[0]
    coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)

    
    # DCC
    temp_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
    coverage = DCC_coverage_list[sample_idx]
    coverage_arr = np.array(coverage)
    temp_df["coverage"] = coverage_arr.reshape(-1)
    temp_df["num_sample"] = num_sample*np.ones(coverage_arr.shape[0])
    # repeat the str alg_name for coverage_arr.shape[0] times
    temp_df["alg_name"] = ["DCC"]*coverage_arr.shape[0]
    coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)

    # IDCC
    temp_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
    coverage = IDCC_coverage_list[sample_idx]
    coverage_arr = np.array(coverage)
    temp_df["coverage"] = coverage_arr.reshape(-1)
    temp_df["num_sample"] = num_sample*np.ones(coverage_arr.shape[0])
    # repeat the str alg_name for coverage_arr.shape[0] times
    temp_df["alg_name"] = ["IDCC"]*coverage_arr.shape[0]
    coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)

    # PTC-B
    temp_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
    coverage = PTC_box_coverage_list[sample_idx]
    coverage_arr = np.array(coverage)
    temp_df["coverage"] = coverage_arr.reshape(-1)
    temp_df["num_sample"] = num_sample*np.ones(coverage_arr.shape[0])
    # repeat the str alg_name for coverage_arr.shape[0] times
    temp_df["alg_name"] = ["PTC-B"]*coverage_arr.shape[0]
    coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)

    # PTC-E
    temp_df = pd.DataFrame(columns=["num_sample","alg_name","coverage"])
    coverage = PTC_ellipsoid_coverage_list[sample_idx]
    coverage_arr = np.array(coverage)
    temp_df["coverage"] = coverage_arr.reshape(-1)
    temp_df["num_sample"] = num_sample*np.ones(coverage_arr.shape[0])
    # repeat the str alg_name for coverage_arr.shape[0] times
    temp_df["alg_name"] = ["PTC-E"]*coverage_arr.shape[0]
    coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)


import seaborn as sns
plt.figure(figsize=(10,8))

ax = sns.boxplot(x="num_sample", y="coverage", hue="alg_name", data=coverage_df,palette="Set3",showfliers=False)

# plot a dashed horizontal line at y=0.8
plt.axhline(y=0.8, color='black', linestyle='--')

plt.xlabel("T",fontsize=30)
plt.ylabel("Coverage", fontsize=30)


plt.tick_params(axis='y', labelsize=25)
plt.tick_params(axis='x', labelsize=25)


plt.legend(fontsize=20)
plt.tight_layout()
plt.show()

# save the figure
plt.savefig(test_dir+str(alpha)+"_num_samples_coverage"+".png",bbox_inches='tight')
plt.savefig(test_dir+str(alpha)+"_num_samples_coverage"+".pdf",bbox_inches='tight')

# plot the boxplot for absolute error
# for the coverage column, minus the alpha and get the absolute error
coverage_df["coverage"] = coverage_df["coverage"] - alpha
coverage_df["coverage"] = coverage_df["coverage"].abs()
plt.figure(figsize=(10,8))

ax = sns.boxplot(x="num_sample", y="coverage", hue="alg_name", data=coverage_df,palette="Set3",showfliers=False)

plt.xlabel("T",fontsize=30)
plt.ylabel("|Coverage-$\\alpha$|", fontsize=30)

"""
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
"""

plt.tick_params(axis='y', labelsize=25)
plt.tick_params(axis='x', labelsize=25)


plt.legend(fontsize=20)
plt.tight_layout()
plt.show()

# save the figure
plt.savefig(test_dir+str(alpha)+"_num_samples_coverage_abserror"+".png",bbox_inches='tight')
plt.savefig(test_dir+str(alpha)+"_num_samples_coverage_abserror"+".pdf",bbox_inches='tight')
