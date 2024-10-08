import argparse
import numpy as np
import os
import pandas as pd
import ipdb
from coverage_solver import in_box,in_DNN_ellipsoid,in_kNN_ellipsoid,in_ellipsoid
import matplotlib.pyplot as plt
# import pickle
import pickle5 as pickle

# import time to measure the time cost
import time


alpha_list = [0.8]
task_name = "toy"
deg = 1
num_train_samples = 200
k_param = 2
smooth_param = 1
n_cluster = 10
dim_covs = 1


f_model_name = "Lasso"
h_model_name_box = "MLP"
h_model_name_ellipsoid = "MLP"
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--k_param', type=float, default=k_param)
parser.add_argument('--smooth_param', type=float, default=smooth_param)
parser.add_argument('--n_cluster', type=int, default=n_cluster)
parser.add_argument('--dim_covs', type=int, default=dim_covs)
parser.add_argument('--alpha_list', type=list, default=alpha_list)
parser.add_argument('--f_model_name', type=str, default=f_model_name)
parser.add_argument('--h_model_name_box', type=str, default=h_model_name_box)
parser.add_argument('--h_model_name_ellipsoid', type=str, default=h_model_name_ellipsoid)

args = parser.parse_args()
task_name = args.task_name
deg = args.deg
num_train_samples = args.num_train_samples
k_param = args.k_param
smooth_param = args.smooth_param
n_cluster = args.n_cluster
dim_covs = args.dim_covs
alpha_list = args.alpha_list
f_model_name = args.f_model_name
h_model_name_box = args.h_model_name_box
h_model_name_ellipsoid = args.h_model_name_ellipsoid

train_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

# load c.npy from this directory
c_test = np.load(test_dir+"/c.npy")

alg_name_lists = []



LB_lists = []
UB_lists = []


def get_LB_UB_from_coverage(coverage,c_test):
    LB = np.zeros(coverage.shape[0])
    UB = np.zeros(coverage.shape[0])
    # for each row, get the max column index whose value is 1
    for row_idx in range(coverage.shape[0]):
        for column_idx in range(coverage.shape[1]):
            if coverage[row_idx,column_idx]==1:
                LB[row_idx] = c_test[row_idx,0,column_idx]
                break
        for column_idx in range(coverage.shape[1]-1,-1,-1):
            if coverage[row_idx,column_idx]==1:
                UB[row_idx] = c_test[row_idx,0,column_idx]
                break
    return LB,UB


time_load_start = time.time()
for alpha in alpha_list:
    alg_name_lists.append([])
    LB_lists.append([])
    UB_lists.append([])

    # PTC quantile algorithm
    prefix = "LUQ/"+f_model_name+"/quantile/"+h_model_name_box+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        #alg_name_lists[-1].append("PTC"+"-box-"+f_model_name+"-"+h_model_name_box)
        alg_name_lists[-1].append("PTC"+"-box")
        coverage = in_box(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)
    
    # PTC norm algorithm
    
    prefix = "LUQ/"+f_model_name+"/norm/"+h_model_name_ellipsoid+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        #alg_name_lists[-1].append("PTC"+"-ellipsoid-"+f_model_name+"-"+h_model_name_ellipsoid)
        alg_name_lists[-1].append("PTC"+"-ellipsoid")
        coverage = in_ellipsoid(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)

    # kNN algorithm
    k = max(int(np.ceil(args.k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*1)
    prefix = "kNN/"+str(k)+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        alg_name_lists[-1].append("kNN")
        #alg_name_lists[-1].append("kNN"+"-"+str(k))
        coverage = in_kNN_ellipsoid(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)
    
    
    # ellipsoid algorithm
    prefix = "ellipsoid/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        alg_name_lists[-1].append("Ellipsoid")
        coverage = in_kNN_ellipsoid(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)
    
    # CRO simple cluster algorithm
    prefix = "cluster/"+str(n_cluster)+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        #alg_name_lists[-1].append("cluster-"+str(n_cluster))
        alg_name_lists[-1].append("K-Means")
        coverage = in_kNN_ellipsoid(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)

    """
    # deep cluster algorithm
    prefix = "DCC_notDeep/"+str(n_cluster)+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        alg_name_lists[-1].append("DCC-"+str(n_cluster))
        coverage = in_kNN_ellipsoid(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)
    """
    # deep cluster algorithm
    n_cluster = 20
    prefix = "DCC_notDeep/"+str(n_cluster)+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        alg_name_lists[-1].append("DCC-"+str(n_cluster))
        coverage = in_kNN_ellipsoid(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_lists[-1].append(LB)
        UB_lists[-1].append(UB)
    
    
    

############################################### plot LB and UB curves of different algorithms ###############################################
# load covs.csv, c.csv from train_dir
# Enable LaTeX rendering in Matplotlib
#plt.rcParams['text.usetex'] = True

# Set the font family to Computer Modern
#plt.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'


# fill in the dataframe


covs = np.loadtxt(train_dir+"covs.csv",delimiter=",")
c = np.loadtxt(train_dir+"c.csv",delimiter=",")
if len(covs.shape)==1:
    covs = np.reshape(covs,(-1,1))

covs_test = np.loadtxt(test_dir+"covs.csv",delimiter=",")

# create numpy for each algorithm to be saved in dataframe
alg_np_list = []
# get unique alg names
uniq_alg_name_list = []
for alg_name_list in alg_name_lists:
    for alg_name in alg_name_list:
        if alg_name not in uniq_alg_name_list:
            uniq_alg_name_list.append(alg_name)
            
            alg_np_list.append(np.zeros((len(alpha_list)*covs_test.shape[0],5)))

true_np = np.zeros((len(alpha_list)*covs_test.shape[0],5))

if len(covs_test.shape)==1:
    covs_test = np.reshape(covs_test,(-1,1))

for alpha_idx,alpha in enumerate(alpha_list):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # plot the LB and UB curves of different algorithms
    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('c',fontsize=20)
    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)

    start_idx = alpha_idx*covs_test.shape[0]
    end_idx = (alpha_idx+1)*covs_test.shape[0]

    for alg_idx,alg_name in enumerate(alg_name_lists[alpha_idx]):
        color = plt.cm.tab10(alg_idx/len(alg_name_lists[alpha_idx]))
        ax.plot(covs_test[:,0],UB_lists[alpha_idx][alg_idx],label=alg_name,color=color,linewidth=3)
        ax.plot(covs_test[:,0],LB_lists[alpha_idx][alg_idx],color=color,linewidth=3)
        # find the index of algrithm in uniq_alg_name_list
        name_idx = uniq_alg_name_list.index(alg_name)
        temp_np = alg_np_list[name_idx]
        
        temp_np[start_idx:end_idx,0] = covs_test[:,0]
        temp_np[start_idx:end_idx,1] = UB_lists[alpha_idx][alg_idx]
        temp_np[start_idx:end_idx,2] = LB_lists[alpha_idx][alg_idx]

    true_np[start_idx:end_idx,0] = covs_test[:,0]

    # scatter the points with x=covs[:,0], y=c, with color in black
    ax.scatter(covs[:,0],c,color="black",s=3,zorder=10)
    ax.legend(fontsize=20)
    
    # save the figure
    fig.savefig(test_dir+str(dim_covs)+"_"+str(num_train_samples)+"_uncertainty_set.pdf")
    fig.savefig(test_dir+str(dim_covs)+"_"+str(num_train_samples)+"_uncertainty_set.png")
    # save such ax
    



#%% 
################################### plot the optimal solution of different algorithms ###################################
x_sol_list = []
x_opt_list = []
for alpha_idx,alpha in enumerate(alpha_list):
    x_sol_list.append([])
    # get the optimal solution of min Var of covs_test
    test_true_UB = 2/(1+np.exp(covs_test[:,0]+(1-alpha)*0.2))-0.7
    test_true_LB = 2/(1+np.exp(covs_test[:,0]+alpha*0.2))-0.7

    x_opt = -np.ones(len(covs_test))*(test_true_LB>0)+np.ones(len(covs_test))*(test_true_UB<0)
    x_opt_list.append(x_opt)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('x',fontsize=20)

    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)

    start_idx = alpha_idx*covs_test.shape[0]
    end_idx = (alpha_idx+1)*covs_test.shape[0]

    for alg_idx,alg_name in enumerate(alg_name_lists[alpha_idx]):
        color = plt.cm.tab10(alg_idx/len(alg_name_lists[alpha_idx]))
        LB = LB_lists[alpha_idx][alg_idx]
        UB = UB_lists[alpha_idx][alg_idx]
        x = -np.ones(len(covs_test))*(LB>0)+np.ones(len(covs_test))*(UB<0)
        if alg_name[:3]=="PTC":
            ax.plot(covs_test[:,0],x,label=alg_name,color=color,linewidth=3)
        else:
            ax.plot(covs_test[:,0],x,label=alg_name,color=color,linewidth=3,linestyle="--")
        
        x_sol_list[-1].append(x)
        # find the index of algrithm in uniq_alg_name_list
        name_idx = uniq_alg_name_list.index(alg_name)
        temp_np = alg_np_list[name_idx]
        
        temp_np[start_idx:end_idx,3] = x

    true_np[start_idx:end_idx,3] = x_opt

    # plot x_opt curve with dot line
    ax.plot(covs_test[:,0],x_opt,label="optimal",color="black",linewidth=3,linestyle=":")

    ax.legend(fontsize=20)

    # save the figure
    fig.savefig(test_dir+str(dim_covs)+"_"+str(num_train_samples)+"_optimal_solution.pdf")
    fig.savefig(test_dir+str(dim_covs)+"_"+str(num_train_samples)+"_optimal_solution.png")
    
#%%
################################### plot the Var of x from different algorithms ###################################
for alpha_idx,alpha in enumerate(alpha_list):
    x_opt = x_opt_list[alpha_idx]
    test_true_UB = 2/(1+np.exp(covs_test[:,0]+(1-alpha)*0.2))-0.7
    test_true_LB = 2/(1+np.exp(covs_test[:,0]+alpha*0.2))-0.7

    # true_var = max{x_opt*test_true_UB,x_opt*test_true_LB}
    true_var = np.maximum(x_opt*test_true_UB,x_opt*test_true_LB)

    # plot the true_var curve
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # set the ylim in [-0.05,0.05]
    #ax.set_ylim([-0.05,0.05])
    # set the xlim in [0.3,0.7]
    #ax.set_xlim([0.3,0.7])
    
    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('VaR',fontsize=20)

    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)

    start_idx = alpha_idx*covs_test.shape[0]
    end_idx = (alpha_idx+1)*covs_test.shape[0]

    for alg_idx,alg_name in enumerate(alg_name_lists[alpha_idx]):
        color = plt.cm.tab10(alg_idx/len(alg_name_lists[alpha_idx]))
        x = x_sol_list[alpha_idx][alg_idx]
        x_var = np.maximum(x*test_true_UB,x*test_true_LB)
        if alg_name[:3]=="PTC":
            ax.plot(covs_test[:,0],x_var,label=alg_name,color=color,linewidth=3)
        else:
            ax.plot(covs_test[:,0],x_var,label=alg_name,color=color,linewidth=3,linestyle="--")
        # find the index of algrithm in uniq_alg_name_list
        name_idx = uniq_alg_name_list.index(alg_name)
        temp_np = alg_np_list[name_idx]
        
        temp_np[start_idx:end_idx,4] = x_var

    true_np[start_idx:end_idx,4] = true_var

    ax.plot(covs_test[:,0],true_var,label="optimal",color="black",linewidth=3,linestyle=":")
    ax.legend(fontsize=20)

    # save the figure
    fig.savefig(test_dir+str(dim_covs)+"_"+str(num_train_samples)+"_VaR.pdf")
    fig.savefig(test_dir+str(dim_covs)+"_"+str(num_train_samples)+"_VaR.png")


#%%
####### save the numpy array of all algorithms #######
for alg_idx,alg_name in enumerate(uniq_alg_name_list):
    np.savetxt(test_dir+alg_name+"_"+str(num_train_samples)+".csv",alg_np_list[alg_idx],delimiter=",")
np.savetxt(test_dir+"true.csv",true_np,delimiter=",")