import subprocess
import os
from read_mse_Var import read_mse_Var
import ipdb

dataset_name="shortest_path"

num_train_samples=5000
deg= 5
dim_covs = 10
h_model_name_box="MLP"
alpha = 0.8

f_model_name_list = ["Lasso","random_forest","MLP","OLS","KernelRidge-rbf","KernelRidge-poly"] #

markers = ["o","v","^","<",">","s","p","P","*","h","H","+","x","X","D","d","|","_"]

cur_dir = os.getcwd()

# the dir of f_model is in ../LUQ/
LUQ_dir = os.path.join(cur_dir,"..","LUQ")

# set the workspace to LUQ
os.chdir(LUQ_dir)

# run the python script train_f.py under LUQ/
for f_model_name in f_model_name_list:

    # in shell: python train_f.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
    subprocess.call(["python","train_f.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--model_name",f_model_name])
    # in shell: python train_quantile_h.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha 0.8
    subprocess.call(["python","train_quantile_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])


os.chdir(cur_dir)

train_dir = os.path.join(cur_dir,dataset_name,str(dim_covs),str(deg),"train",str(num_train_samples),"LUQ")
test_dir = os.path.join(cur_dir,dataset_name,str(dim_covs),str(deg),"test")

mse_list, VaR_list = read_mse_Var(train_dir,test_dir,alpha,f_model_name_list,h_model_name_box,dataset_name)

# scatter the point (mse, np.mean(VaR)) for each f_model_name, and use f_model_name as the legend label
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,8))

# set x axis use log scale
# plt.xscale("log")



for i in range(len(f_model_name_list)):
    f_model_name = f_model_name_list[i]
    if f_model_name == "random_forest":
        f_model_name="Random Forest"  
    elif f_model_name == "MLP":
        f_model_name = "NN"
    plt.scatter(mse_list[i]/100,np.mean(VaR_list[i])/1000,label=f_model_name,marker=markers[i],s=150)

# set the legend fontsize
plt.legend(fontsize=25)
plt.rcParams['text.usetex'] = True
plt.xlabel('MSE ($\\times 100$)',fontsize=30)

plt.ylabel('Avg. VaR ($\\times 1000$)',fontsize=30)
# plt.tick_params(axis='x', labelsize=23)

# set the fontsize of the tick label
plt.xticks([5,6,7,8,9,10],fontsize=25)
plt.yticks(fontsize=25)

# set tight layout
plt.tight_layout()

# save the figure to png and pdf to test_dir
plt.savefig(os.path.join(test_dir,"mse_VaR.png"),bbox_inches='tight')
plt.savefig(os.path.join(test_dir,"mse_VaR.pdf"),bbox_inches='tight')
