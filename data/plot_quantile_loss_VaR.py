import subprocess
import os
from read_quantile_loss_VaR import read_quantile_loss_Var
import ipdb

dataset_name="shortest_path"

num_train_samples=5000
deg= 5
dim_covs = 10
f_model_name="KernelRidge-rbf"
alpha = 0.8
box_or_ellipsoid = "box"

h_model_name_list = ["Linear","MLP","GBR"] 
markers = ["o","v","^","<",">","s","p","P","*","h","H","+","x","X","D","d","|","_"]

cur_dir = os.getcwd()

# the dir of f_model is in ../LUQ/
LUQ_dir = os.path.join(cur_dir,"..","LUQ")

# set the workspace to LUQ
os.chdir(LUQ_dir)

# run the python script train_f.py under LUQ/
for h_model_name_box in h_model_name_list:

    # in shell: python train_f.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
    subprocess.call(["python","train_f.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--model_name",f_model_name])
    # in shell: python train_quantile_h.py --task_name $dataset_name --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name_box --alpha 0.8
    
    if box_or_ellipsoid=="box":
        subprocess.call(["python","train_quantile_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])
    elif box_or_ellipsoid=="ellipsoid":
        subprocess.call(["python","train_2norm_h.py","--task_name",dataset_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])

os.chdir(cur_dir)

train_dir = os.path.join(cur_dir,dataset_name,str(dim_covs),str(deg),"train",str(num_train_samples),"LUQ")
test_dir = os.path.join(cur_dir,dataset_name,str(dim_covs),str(deg),"test")

loss_list, VaR_list = read_quantile_loss_Var(train_dir,test_dir,alpha,f_model_name,h_model_name_list,dataset_name,box_or_ellipsoid)

# scatter the point (mse, np.mean(VaR)) for each f_model_name, and use f_model_name as the legend label
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

# set x axis use log scale
plt.xscale("log")

for i in range(len(h_model_name_list)):
    h_model_name = h_model_name_list[i]
    if h_model_name=="MLP":
        h_model_name = "NN"
    plt.scatter(loss_list[i],np.mean(loss_list[i]),label=h_model_name,marker=markers[i])
plt.legend()

plt.xlabel("Pinball Loss")
plt.ylabel("VaR")

# save the figure to png and pdf to test_dir
plt.savefig(os.path.join(test_dir,box_or_ellipsoid+"_quanile_loss_VaR.png"))
plt.savefig(os.path.join(test_dir,box_or_ellipsoid+"_quantile_loss_VaR.pdf"))
