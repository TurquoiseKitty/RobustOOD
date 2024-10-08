'''
find the best model (parameter) to fit the data
'''

#import random forest regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#import linear regression method and Lasso method
from sklearn.linear_model import LinearRegression, Lasso

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

from HetRes_Example import train_q
from solver import solve_wasserstein1_LP,get_spp_Ab,get_kp_Ab
from get_weights import get_weights
from tqdm import tqdm

import argparse

sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_HetRes,split_data_for_PTC_DRO

task_name = "knapsack"
num_train_samples = 500
deg = 2
dim_covs = 5
f_model_name = "OLS"
kernel_name = "gaussian" # options: gaussian, uniform, kNN
param = 1 # kernal parameter

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--f_model_name',type=str, default=f_model_name)
parser.add_argument('--kernel_name',type=str, default=kernel_name)
parser.add_argument('--param',type=float, default=param)
parser.add_argument('--E_dro',type=bool,default=True)
parser.add_argument('--dim_covs',type=int,default=dim_covs)


args = parser.parse_args()

task_name = args.task_name
num_train_samples = args.num_train_samples

deg = args.deg


f_model_name = args.f_model_name
kernel_name = args.kernel_name
param = args.param
E_dro = args.E_dro
dim_covs = args.dim_covs

train_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"




# support set: Tx<=phi
T = None
phi = None

# split data in this dir
split_data_for_PTC_DRO(train_dir,train_ratio=0.7,UQ_ratio=0.2,max_cal_num = 100)

# turn to HetRes dir
LUQ_dir = train_dir+"PTC_DRO/"

# load the fit data
fit_dir = LUQ_dir+"fit/"
covs_fit = pd.read_csv(fit_dir+"covs_fit.csv",header=None).to_numpy()
c_fit = pd.read_csv(fit_dir+"c_fit.csv",header=None).to_numpy()

# load the UQ data
UQ_dir = LUQ_dir+"UQ/"
covs_UQ = pd.read_csv(UQ_dir+"covs_UQ.csv",header=None).to_numpy()
c_UQ = pd.read_csv(UQ_dir+"c_UQ.csv",header=None).to_numpy()

# load the cal data
cal_dir = LUQ_dir+"cal/"
covs_cal = pd.read_csv(cal_dir+"covs_cal.csv",header=None).to_numpy()
c_cal = pd.read_csv(cal_dir+"c_cal.csv",header=None).to_numpy()

# load the test data
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()

#preprocess data
scaler = StandardScaler()
covs_fit = scaler.fit_transform(covs_fit)
covs_UQ = scaler.transform(covs_UQ)
covs_cal = scaler.transform(covs_cal)
covs_test = scaler.transform(covs_test)

save_dir = LUQ_dir+f_model_name+"/"
if f_model_name == "random_forest":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        random_state = 0
        best_est = 0
        best_depth = 0
        best_mse = 1000000000

        rf = RandomForestRegressor(random_state=random_state)

        # define param search space
        param_grid = {'n_estimators': [100,200,500,800,1000], 'max_depth': [10,20,30,50,80]}

        #use grid search to find the best parameters
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(covs_fit,c_fit)

        #save best parameters
        
        #save best_est to a txt file
        np.savetxt(save_dir+"f_best_est.txt",np.array([grid_search.best_params_['n_estimators']]))
        #save best_depth to a txt file
        np.savetxt(save_dir+"f_best_depth.txt",np.array([grid_search.best_params_['max_depth']]))

        

        # predict the c_UQ, c_test
        rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],max_depth=grid_search.best_params_['max_depth'],random_state=random_state)
        rf.fit(covs_fit,c_fit)
        c_UQ_pred = rf.predict(covs_UQ)
        c_fit_pred = rf.predict(covs_fit)
        c_cal_pred = rf.predict(covs_cal)
        c_test_pred = rf.predict(covs_test)
        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,rf.predict(covs_fit))
        np.savetxt(save_dir+"rf_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
        np.savetxt(save_dir+"c_cal_pred.csv",c_cal_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_cal_pred = np.loadtxt(save_dir+"c_cal_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")

elif f_model_name=="OLS":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # using linear regression model to fit
        reg = LinearRegression().fit(covs_fit, c_fit)
        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = reg.predict(covs_UQ)
        c_test_pred = reg.predict(covs_test)
        c_fit_pred = reg.predict(covs_fit)
        c_cal_pred = reg.predict(covs_cal)

        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,reg.predict(covs_fit))
        np.savetxt(save_dir+"OLS_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
        np.savetxt(save_dir+"c_cal_pred.csv",c_cal_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")
        c_cal_pred = np.loadtxt(save_dir+"c_cal_pred.csv",delimiter=",")
elif f_model_name=="Lasso":
    if not os.path.exists(save_dir):
        # using Lasso model to fit
        reg = Lasso().fit(covs_fit, c_fit)
        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = reg.predict(covs_UQ)
        c_test_pred = reg.predict(covs_test)
        c_fit_pred = reg.predict(covs_fit)
        c_cal_pred = reg.predict(covs_cal)

        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,reg.predict(covs_fit))
        np.savetxt(save_dir+"Lasso_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
        np.savetxt(save_dir+"c_cal_pred.csv",c_cal_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")
        c_cal_pred = np.loadtxt(save_dir+"c_cal_pred.csv",delimiter=",")


elif f_model_name=="MLP":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # Create a function that builds the MLP model
        def build_model(optimizer='adam', activation='relu', neurons=16):
            model = Sequential()
            model.add(Dense(neurons, input_dim=covs_fit.shape[1], activation=activation))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return model
        
        # Create a KerasClassifier object with the build_model function and its default parameters
        model = KerasClassifier(build_fn=build_model, verbose=0)

        # Define the hyperparameters grid
        param_grid = {'optimizer': ['adam', 'sgd'],
                    'activation': ['relu', 'sigmoid'],
                    'neurons': [8, 16, 32]}
        
        # Use GridSearchCV to search for the best hyperparameters
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        grid_result = grid.fit(covs_fit, c_fit)

        # save the best parameters
        np.savetxt(save_dir+"f_best_opt.txt",np.array([grid_result.best_params_['optimizer']]))
        np.savetxt(save_dir+"f_best_act.txt",np.array([grid_result.best_params_['activation']]))
        np.savetxt(save_dir+"f_best_neu.txt",np.array([grid_result.best_params_['neurons']]))

        # predict the c_UQ, c_cal, c_test
        model = Sequential()
        model.add(Dense(grid_result.best_params_['neurons'], input_dim=covs_fit.shape[1], activation=grid_result.best_params_['activation']))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=grid_result.best_params_['optimizer'], metrics=['accuracy'])
        model.fit(covs_fit,c_fit,epochs=500,verbose=0)

        c_UQ_pred = model.predict(covs_UQ)
        c_test_pred = model.predict(covs_test)
        c_fit_pred = model.predict(covs_fit)
        c_cal_pred = model.predict(covs_cal)
        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,model.predict(covs_fit))
        np.savetxt(save_dir+"MLP_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
        np.savetxt(save_dir+"c_cal_pred.csv",c_cal_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")
        c_cal_pred = np.loadtxt(save_dir+"c_cal_pred.csv",delimiter=",")

#%%
############### get the weights of cal and test data according to UQ data#################
# calculate the residual
save_dir = save_dir+kernel_name+"_"+str(param)+"//"
res_UQ = c_UQ - c_UQ_pred
# judge if the q is already calculated
if not os.path.exists(save_dir+"weights_cal.csv") or not os.path.exists(save_dir+"weights_test.csv"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # calculate the weights of cal and test data
    weights_cal = np.zeros((covs_cal.shape[0],covs_UQ.shape[0]))
    for cal_idx in range(covs_cal.shape[0]):
        weights_cal[cal_idx,:] = get_weights(covs_UQ,covs_cal[cal_idx,:],kernel_name,param)

    weights_test = np.zeros((covs_test.shape[0],covs_UQ.shape[0]))
    for test_idx in range(covs_test.shape[0]):
        weights_test[test_idx,:] = get_weights(covs_UQ,covs_test[test_idx,:],kernel_name,param)
    
    # save the weights to csv file
    np.savetxt(save_dir+"weights_cal.csv",weights_cal,delimiter=",")
    np.savetxt(save_dir+"weights_test.csv",weights_test,delimiter=",")
  
else:
    # load data
    weights_cal = np.loadtxt(save_dir+"weights_cal.csv",delimiter=",")
    weights_test = np.loadtxt(save_dir+"weights_test.csv",delimiter=",")


#%%
############## select radius ################
eps_set = [0,0.001,0.01,0.02,0.05]

save_dir = save_dir+str(E_dro)+"/"

num_UQ = covs_UQ.shape[0]
num_cal = covs_cal.shape[0]
num_test = covs_test.shape[0]
if task_name == "shortest_path":
    A, b = get_spp_Ab()
    #check if there is a csv file record the x solution
    
    # objective under different eps
    ERMs = np.zeros(len(eps_set))
    for eps_idx,eps in enumerate(eps_set):
        if not (os.path.exists(save_dir+"eps-"+str(eps)+".txt")):
            if E_dro==False:
                x_sol_cal,objs_cal = solve_wasserstein1_LP(eps,A,b,T,phi,weights_cal,c_cal_pred,res_UQ,task_name)
            else:
                Er = weights_cal@res_UQ
                Er = np.diag(Er)
                x_sol_cal,objs_cal = solve_wasserstein1_LP(eps,A,b,T,phi,np.diag(np.ones(num_cal)),c_cal_pred,Er,task_name)
            ERM = np.sum(x_sol_cal*c_cal,axis=1)
            ERMs[eps_idx] = np.mean(ERM)
            np.savetxt(save_dir+"eps-"+str(eps)+".txt",np.array([ERMs[eps_idx]]))
        else:
            ERMs[eps_idx] = np.loadtxt(save_dir+"eps-"+str(eps)+".txt")
    # find the best eps
    best_eps = eps_set[np.argmin(ERMs)]
    #save best eps
    np.savetxt(save_dir+"best_eps.csv",np.array([best_eps]),delimiter=",")
    if not (os.path.exists(save_dir+"x_sol.csv")):
        # solve the problem
        if E_dro==False:

            x_sol,objs = solve_wasserstein1_LP(best_eps,A,b,T,phi,weights_test,c_test_pred,res_UQ,task_name)
        else:
            Er = weights_test@res_UQ
            Er = np.diag(Er)
            x_sol,objs = solve_wasserstein1_LP(best_eps,A,b,T,phi,np.diag(np.ones(num_test)),c_test_pred,Er,task_name)
        # save the x solution
        np.savetxt(save_dir+"x_sol.csv",x_sol,delimiter=",")
        # save the objective values
        np.savetxt(save_dir+"objs.csv",objs,delimiter=",")

elif task_name == "knapsack":
    # load prices and budgets
    prices = pd.read_csv("../data/knapsack/prices.csv",header=None).to_numpy()
    budgets = pd.read_csv("../data/knapsack/budgets.csv",header=None).to_numpy()
    #check if there is a npy file record the x solution
    
    num_constraints = budgets.shape[0]
    num_tests = c_test_pred.shape[0]
    ERMs_cons = np.zeros((len(eps_set),num_constraints))
    ERMs = np.zeros(len(eps_set))
    for eps_idx,eps in enumerate(eps_set):
        if not (os.path.exists(save_dir+"eps-"+str(eps)+".txt")):
            for i in tqdm(range(num_constraints)):
                A,b = get_kp_Ab(prices[i,:],budgets[i])
                if E_dro==False:
                    x_sol_cal,objs_cal = solve_wasserstein1_LP(eps,A,b,T,phi,weights_cal,c_cal_pred,res_UQ,task_name)
                else:
                    Er = weights_cal@res_UQ
                    Er = np.diag(Er)
                    x_sol_cal,objs_cal = solve_wasserstein1_LP(eps,A,b,T,phi,np.diag(np.ones(num_cal)),c_cal_pred,Er,task_name)
                ERM = np.sum(x_sol_cal*c_cal,axis=1)
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
            if E_dro==False:
                x_sol,objs = solve_wasserstein1_LP(best_eps,A,b,T,phi,weights_test,c_test_pred,res_UQ,task_name)
            else:
                Er = weights_test@res_UQ
                Er = np.diag(Er)

                x_sol,objs = solve_wasserstein1_LP(best_eps,A,b,T,phi,np.diag(np.ones(num_test)),c_test_pred,Er,task_name)
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the x solution
        np.save(save_dir+"x_sol.npy",x_sols)
        # save the objective values
        np.save(save_dir+"objs.npy",objss)




