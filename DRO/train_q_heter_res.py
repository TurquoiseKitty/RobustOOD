'''
find the best model (parameter) to fit the data
'''

#import random forest regressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
# import keras to build MLP
from solver import get_spp_Ab,get_kp_Ab

from HetRes_Example import train_q
from solver import solve_wasserstein_res
import ipdb
from tqdm import tqdm


task_name = "knapsack"
num_train_samples = 5000

train_dir = "../data/"+task_name+"/01/1/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/01/1/test/"
alpha = 0.8 # options: 0.8, 0.85, 0.9, 0.95, if fit_goal is "f", then alpha is not used
f_model_name = "random_forest"
h_model_name = "example1-2"
eps = 0.1

# support set: Tx<=phi
T = None
phi = None


# turn to LUQ dir
LUQ_dir = train_dir+"HetRes/"

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


# load the prediction data
pred_dir = LUQ_dir+"random_forest/"
c_UQ_pred = pd.read_csv(pred_dir+"c_UQ_pred.csv",header=None).to_numpy()
c_test_pred = pd.read_csv(pred_dir+"c_test_pred.csv",header=None).to_numpy()
c_fit_pred = pd.read_csv(pred_dir+"c_fit_pred.csv",header=None).to_numpy()

# calculate the residual
res_UQ = c_UQ - c_UQ_pred
res_fit = c_fit - c_fit_pred

save_dir = LUQ_dir+f_model_name+"/HetRes/"+h_model_name+"/"+str(eps)+"/"
# create the dir if not exist
# judge if the q is already calculated
if not os.path.exists(save_dir+"q_test.csv"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if h_model_name=="example1-2":

        # train the heteroscedastic model
        q2_fit,q2_UQ,q2_test = train_q(covs_UQ,res_UQ,covs_fit,covs_test)
        # get the square root of q2
        q_fit = np.sqrt(q2_fit)
        q_UQ = np.sqrt(q2_UQ)
        q_test = np.sqrt(q2_test)
        # save the q
        np.savetxt(save_dir+"q_fit.csv",q_fit,delimiter=",")
        np.savetxt(save_dir+"q_UQ.csv",q_UQ,delimiter=",")
        np.savetxt(save_dir+"q_test.csv",q_test,delimiter=",")

        # get the eps residuals
        eps_UQ = res_UQ/q_UQ
        eps_fit = res_fit/q_fit


        # save the eps residuals
        np.savetxt(save_dir+"eps_UQ.csv",eps_UQ,delimiter=",")
        np.savetxt(save_dir+"eps_fit.csv",eps_fit,delimiter=",")

        # concatenate the eps_UQ and eps_fit
        eps_UQ_fit = np.concatenate((eps_UQ,eps_fit),axis=0)
        # save it
        np.savetxt(save_dir+"eps_UQ_fit.csv",eps_UQ_fit,delimiter=",")

else:
    # load data
    q_test = pd.read_csv(save_dir+"q_test.csv",header=None).to_numpy()
    eps_UQ_fit = pd.read_csv(save_dir+"eps_UQ_fit.csv",header=None).to_numpy()
    



if task_name == "shortest_path":
    A, b = get_spp_Ab()
    #check if there is a csv file record the x solution
    if not (os.path.exists(save_dir+"x_sol.csv")):
        # solve the problem
        x_sol,objs = solve_wasserstein_res(eps,A,b,T,phi,c_test_pred,q_test,eps_UQ_fit,task_name)
        # save the x solution
        np.savetxt(save_dir+"x_sol.csv",x_sol,delimiter=",")
        # save the objective values
        np.savetxt(save_dir+"objs.csv",objs,delimiter=",")

elif task_name == "knapsack":
    # load prices and budgets
    prices = pd.read_csv("../data/knapsack/prices.csv",header=None).to_numpy()
    budgets = pd.read_csv("../data/knapsack/budgets.csv",header=None).to_numpy()
    #check if there is a npy file record the x solution
    if not os.path.exists(save_dir+"x_sol.npy"):
        # solve the problem
        num_constraints = budgets.shape[0]
        num_tests = c_test_pred.shape[0]
        x_sols = np.zeros((num_constraints,num_tests,prices.shape[1]))
        objss = np.zeros((num_constraints,num_tests))
        for i in tqdm(range(num_constraints)):
            A,b = get_kp_Ab(prices[i,:],budgets[i])
            x_sol,objs = solve_wasserstein_res(eps,A,b,T,phi,c_test_pred,q_test,eps_UQ_fit,task_name)
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the x solution
        np.save(save_dir+"x_sol.npy",x_sols)
        # save the objective values
        np.save(save_dir+"objs.npy",objss)
        

    

