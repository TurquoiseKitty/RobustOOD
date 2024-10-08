from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from get_weights import get_weights
import os
import sys
from tqdm import tqdm

from solver import get_kp_Ab,get_spp_Ab,solve_DRCME


task_name = "knapsack"
num_train_samples = 5000

train_dir = "../data/"+task_name+"/01/1/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/01/1/test/"
alpha = 0.8 # options: 0.8, 0.85, 0.9, 0.95, if fit_goal is "f", then alpha is not used


k_approx = 10 # number of nearest neighbors used to get the radius in covs space
C = 0.1 # constant used to get the radius in c space



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

save_dir = train_dir+"DRCME/"+str(k_approx)+"/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dim_covs = covs_train.shape[1]
dim_c = c_train.shape[1]

num_train_samples = covs_train.shape[0]
rho = C*np.power(num_train_samples,-1/(dim_c+dim_covs))*(1/(dim_c+dim_covs))*np.log(num_train_samples)

#preprocess data
scaler = StandardScaler()
covs_train = scaler.fit_transform(covs_train)
covs_test = scaler.transform(covs_test)

# calculate the distances between covs_test and covs_train using Mahalanobis distance
covs_train_cov = np.cov(covs_train.T)
covs_train_cov_inv = np.linalg.inv(covs_train_cov)
distances = np.zeros((num_test_samples,num_train_samples))
gammas = np.zeros(num_test_samples) # the value of the k_approx-th lowest distance
DX = np.zeros((num_test_samples,num_train_samples)) # >= distances-gammas, >=0
rho_DX = np.zeros((num_test_samples,num_train_samples)) # rho - DX
for i in range(num_test_samples):
    distances[i,:] = np.sqrt(np.sum((covs_test[i,:]-covs_train)@covs_train_cov_inv*(covs_test[i,:]-covs_train),axis=1))
    gammas[i] = np.sort(distances[i,:])[k_approx-1]
    DX[i,:] = distances[i,:]-gammas[i]
    DX[i,:] = np.maximum(DX[i,:],0)
    rho_DX[i,:] = rho-DX[i,:]

# get the indexes of training samples that are within rho+gamma of covs_test
Imat = np.zeros((num_test_samples,num_train_samples))
# indexes with distance in gamma-rho 
I1mat = np.zeros((num_test_samples,num_train_samples))
# I\I1
I2mat = np.zeros((num_test_samples,num_train_samples))
for i in range(num_test_samples):
    Imat[i,:] = distances[i,:]<= (rho+gammas[i])
    I1mat[i,:] = distances[i,:]<= (gammas[i]-rho)
    I2mat[i,:] = Imat[i,:]-I1mat[i,:]




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
        for i in tqdm(range(num_constraints)):
            A, b = get_kp_Ab(prices[i,:],budgets[i])
            x_sol,objs = solve_DRCME(c_train,rho_DX,Imat,I1mat,I2mat,A,b,task_name)
        
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the solution
        np.save(save_dir+"x_sol.npy",x_sols)
        np.save(save_dir+"objs.npy",objss)
