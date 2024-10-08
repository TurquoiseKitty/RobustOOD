from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from get_weights import get_weights
import os
import sys
from julia import Julia
from solver import get_kp_Ab,get_spp_Ab

ju = Julia()
ju.include("NW_KL.jl")

task_name = "knapsack"
num_train_samples = 5000

dim_covs = 5
deg = 2

train_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"
alpha = 0.8 # options: 0.8, 0.85, 0.9, 0.95, if fit_goal is "f", then alpha is not used
r = np.log(1/(1-alpha))/num_train_samples # distribution distance

model_name = "NW" #option: kNN
# if model_name is "NW", param is the bandwidth of the Gaussian kernel; if model_name is "kNN", param is the number of nearest neighbors
if model_name == "NW":
    param = 1.0
elif model_name == "kNN":
    param = 50

if task_name=="knapsack":
    constraint_dir = "../data/"+task_name+"/"
    # load prices and budgets from cosntraint_dir
    prices = pd.read_csv(constraint_dir+"prices.csv",header=None).to_numpy()
    budgets = pd.read_csv(constraint_dir+"budgets.csv",header=None).to_numpy()

# load covs_train, c_train, covs_test, c_test from train_dir and test_dir
covs_train = pd.read_csv(train_dir+"covs.csv",header=None).to_numpy()
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()
c_train = pd.read_csv(train_dir+"c.csv",header=None).to_numpy()
c_test = pd.read_csv(test_dir+"c.csv",header=None).to_numpy()
num_test_samples = c_test.shape[0]

save_dir = train_dir+"KL/"+model_name+"/"+str(alpha)+"/"
dim_covs = covs_train.shape[1]
dim_c = c_train.shape[1]

#preprocess data
scaler = StandardScaler()
covs_train = scaler.fit_transform(covs_train)
covs_test = scaler.transform(covs_test)

# calculate the weights of train points for each test point
weights = np.zeros((num_test_samples,num_train_samples))
if model_name == "NW":
    for i in range(num_test_samples):
        weights[i,:] = get_weights(covs_train,covs_test[i,:],"gaussian",param)
elif model_name == "kNN":
    for i in range(num_test_samples):
        weights[i,:] = get_weights(covs_train,covs_test[i,:],"kNN",param)

if task_name == "shortest_path":
    pass

elif task_name=="knapsack":
    # check if the solution file x_sol.npy is exist
    if not os.path.exists(save_dir+"x_sol.npy"):
        num_constraints = budgets.shape[0]
        num_test = covs_test.shape[0]
        # solve the optimization problem
        x_sols = np.zeros((num_constraints,num_test,prices.shape[1]))
        objss = np.zeros((num_constraints,num_test))
        for i in range(1):#range(num_constraints):
            A, b = get_kp_Ab(prices[i,:],budgets[i])
            for test_ind in range(1):
                x_sol,obj = ju.NW(c_train,weights[test_ind,:].T,r,A,b,task_name)
                print(x_sol)
        """
            x_sol,objs = solve_ellipsoid(c_test_pred,res_test_2norm_pred,cov,r,A,b,task_name)
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the solution
        np.save(save_dir+"x_sol.npy",x_sols)
        np.save(save_dir+"objs.npy",objss)
        """