from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from get_weights import get_weights
import os
import sys
from solver import get_kp_Ab,get_spp_Ab,solve_wasserstein1_LP
from tqdm import tqdm

import argparse
import ipdb

sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_HetRes

task_name = "knapsack"
num_train_samples = 500

deg = 2
f_model_name = "gaussian" #option: kNN
param = 1

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--f_model_name',type=str, default=f_model_name)
parser.add_argument('--param',type=float, default=param)

args = parser.parse_args()

task_name = args.task_name
num_train_samples = args.num_train_samples

deg = deg = args.deg
model_name = args.f_model_name
param = args.param


train_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

LUQ_dir = train_dir+"localDRO/"
# split data in this dir
split_data_for_HetRes(train_dir,LUQ_dir,train_ratio=0.9,max_UQ_num = 100)



if task_name=="knapsack":
    constraint_dir = "../data/"+task_name+"/"
    # load prices and budgets from cosntraint_dir
    prices = pd.read_csv(constraint_dir+"prices.csv",header=None).to_numpy()
    budgets = pd.read_csv(constraint_dir+"budgets.csv",header=None).to_numpy()

# load covs_fit, covs_UQ, covs_test, c_fit, c_UQ
# load the fit data
fit_dir = LUQ_dir+"fit/"
covs_fit = pd.read_csv(fit_dir+"covs_fit.csv",header=None).to_numpy()
c_fit = pd.read_csv(fit_dir+"c_fit.csv",header=None).to_numpy()

# load the UQ data
UQ_dir = LUQ_dir+"UQ/"
covs_UQ = pd.read_csv(UQ_dir+"covs_UQ.csv",header=None).to_numpy()
c_UQ = pd.read_csv(UQ_dir+"c_UQ.csv",header=None).to_numpy()


# load the test data
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()

#preprocess data
scaler = StandardScaler()
covs_fit = scaler.fit_transform(covs_fit)
covs_UQ = scaler.transform(covs_UQ)
covs_test = scaler.transform(covs_test)


num_test_samples = covs_test.shape[0]

save_dir = LUQ_dir+model_name+"_"+str(param)+"/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dim_covs = covs_fit.shape[1]
dim_c = c_fit.shape[1]

num_fit_samples = covs_fit.shape[0]
# calculate the weights of train points for each test point
weights_UQ = np.zeros((covs_UQ.shape[0],num_fit_samples))
weights_test = np.zeros((num_test_samples,num_fit_samples))


if model_name == "gaussian":
    for i in range(covs_UQ.shape[0]):
        weights_UQ[i,:] = get_weights(covs_fit,covs_UQ[i,:],"gaussian",param)
    for i in range(num_test_samples):
        weights_test[i,:] = get_weights(covs_fit,covs_test[i,:],"gaussian",param)
elif model_name == "kNN":
    for i in range(covs_UQ.shape[0]):
        weights_UQ[i,:] = get_weights(covs_fit,covs_UQ[i,:],"kNN",param)
    for i in range(num_test_samples):
        weights_test[i,:] = get_weights(covs_fit,covs_test[i,:],"kNN",param)
    

eps_set = [0,0.05,0.1,0.15,0.2]
T = None
phi = None

f_UQ = np.zeros(c_UQ.shape)
f_test = np.zeros((num_test_samples,dim_c))

if task_name == "shortest_path":
    # set the eps set
    A,b = get_spp_Ab()
    
    # objective under different eps
    ERMs = np.zeros(len(eps_set))
    for eps_idx,eps in enumerate(eps_set):
        if not (os.path.exists(save_dir+"eps-"+str(eps)+".txt")):
            x_sol_UQ,_ = solve_wasserstein1_LP(eps,A,b,T,phi,weights_UQ,f_UQ,c_fit,task_name)
            ERM = np.sum(x_sol_UQ*c_UQ,axis=1)
            ERMs[eps_idx] = np.mean(ERM)
            np.savetxt(save_dir+"eps-"+str(eps)+".txt",np.array([ERMs[eps_idx]]))
        else:
            ERMs[eps_idx] = np.loadtxt(save_dir+"eps-"+str(eps)+".txt")
    # find the best eps
    best_eps = eps_set[np.argmin(ERMs)]
    #save best eps
    np.savetxt(save_dir+"best_eps.csv",np.array([best_eps]),delimiter=",")

    # solve the problem
    if not (os.path.exists(save_dir+"x_sol.csv")):
        x_sol,objs = solve_wasserstein1_LP(best_eps,A,b,T,phi,weights_test,f_test,c_fit,task_name)
        # save the x solution
        np.savetxt(save_dir+"x_sol.csv",x_sol,delimiter=",")
        # save the objective values
        np.savetxt(save_dir+"objs.csv",objs,delimiter=",")


elif task_name=="knapsack":
    # load prices and budgets
    prices = pd.read_csv("../data/knapsack/prices.csv",header=None).to_numpy()
    budgets = pd.read_csv("../data/knapsack/budgets.csv",header=None).to_numpy()
    #check if there is a npy file record the x solution

    
    num_constraints = budgets.shape[0]
    num_tests = covs_test.shape[0]
    ERMs_cons = np.zeros((len(eps_set),num_constraints))
    ERMs = np.zeros(len(eps_set))
    for eps_idx,eps in enumerate(eps_set):
        if not (os.path.exists(save_dir+"eps-"+str(eps)+".txt")):
            for i in tqdm(range(num_constraints)):
                A,b = get_kp_Ab(prices[i,:],budgets[i])
                x_sol_UQ,objs_UQ = solve_wasserstein1_LP(eps,A,b,T,phi,weights_UQ,f_UQ,c_fit,task_name)
                ERM = np.sum(x_sol_UQ*c_UQ,axis=1)
                ERMs_cons[eps_idx,i] = np.mean(ERM)
            np.savetxt(save_dir+"eps-"+str(eps)+".txt",ERMs_cons[eps_idx,:])
        else:
            ERMs_cons[eps_idx,:] = np.loadtxt(save_dir+"eps-"+str(eps)+".txt")
        ERMs[eps_idx] = np.mean(ERMs_cons[eps_idx,:])
    # find the best eps
    # note this is max
    best_eps = eps_set[np.argmax(ERMs)]     
    #save best eps
    np.savetxt(save_dir+"best_eps.csv",np.array([best_eps]),delimiter=",")

    if not os.path.exists(save_dir+"x_sol.npy"):
        # solve the problem
        x_sols = np.zeros((num_constraints,num_tests,prices.shape[1]))
        objss = np.zeros((num_constraints,num_tests))
        for i in tqdm(range(num_constraints)):
            A,b = get_kp_Ab(prices[i,:],budgets[i])
            x_sol,objs = solve_wasserstein1_LP(best_eps,A,b,T,phi,weights_test,f_test,c_fit,task_name)
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the x solution
        np.save(save_dir+"x_sol.npy",x_sols)
        # save the objective values
        np.save(save_dir+"objs.npy",objss)