'''train a predictor with train samples, and use it to predict the test samples, and plug it into the linear program to get the optimal solution'''


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
from solver import solve_wasserstein1_LP,get_spp_Ab,get_kp_Ab,solve_f_LP
from get_weights import get_weights
from tqdm import tqdm
import argparse


task_name = "knapsack"
num_train_samples = 500

deg = 2
f_model_name = "OLS"

parser = argparse.ArgumentParser()

parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--f_model_name',type=str, default=f_model_name)


args = parser.parse_args()

task_name = args.task_name
num_train_samples = args.num_train_samples

deg = deg = args.deg
f_model_name = args.f_model_name


train_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

# load train data
covs_train = pd.read_csv(train_dir+"covs.csv",header=None).to_numpy()
c_train = pd.read_csv(train_dir+"c.csv",header=None).to_numpy()

# load test data
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()

# save dir
save_dir = train_dir+"only_f/"


#preprocess data
scaler = StandardScaler()
covs_train = scaler.fit_transform(covs_train)
covs_test = scaler.transform(covs_test)

save_dir = save_dir+f_model_name+"/"
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
        grid_search.fit(covs_train,c_train)

        #save best parameters
        
        #save best_est to a txt file
        np.savetxt(save_dir+"f_best_est.txt",np.array([grid_search.best_params_['n_estimators']]))
        #save best_depth to a txt file
        np.savetxt(save_dir+"f_best_depth.txt",np.array([grid_search.best_params_['max_depth']]))

        

        # predict the c_UQ, c_test
        rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],max_depth=grid_search.best_params_['max_depth'],random_state=random_state)
        rf.fit(covs_train,c_train)

        c_test_pred = rf.predict(covs_test)
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")

    else:
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        
elif f_model_name=="OLS":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # using linear regression model to fit
        reg = LinearRegression().fit(covs_train, c_train)
        # predict the c_UQ, c_cal, c_test
        c_test_pred = reg.predict(covs_test)

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")

    else:    
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")

elif f_model_name=="Lasso":
    if not os.path.exists(save_dir):
        # using Lasso model to fit
        reg = Lasso().fit(covs_train, c_train)
        # predict the c_UQ, c_cal, c_test
        c_test_pred = reg.predict(covs_test)

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")

    else:
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
    

elif f_model_name=="MLP":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # Create a function that builds the MLP model
        def build_model(optimizer='adam', activation='relu', neurons=16):
            model = Sequential()
            model.add(Dense(neurons, input_dim=covs_train.shape[1], activation=activation))
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
        grid_result = grid.fit(covs_train, c_train)

        # save the best parameters
        np.savetxt(save_dir+"f_best_opt.txt",np.array([grid_result.best_params_['optimizer']]))
        np.savetxt(save_dir+"f_best_act.txt",np.array([grid_result.best_params_['activation']]))
        np.savetxt(save_dir+"f_best_neu.txt",np.array([grid_result.best_params_['neurons']]))

        # predict the c_UQ, c_cal, c_test
        model = Sequential()
        model.add(Dense(grid_result.best_params_['neurons'], input_dim=covs_train.shape[1], activation=grid_result.best_params_['activation']))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=grid_result.best_params_['optimizer'], metrics=['accuracy'])
        model.fit(covs_train,c_train,epochs=500,verbose=0)

        c_test_pred = model.predict(covs_test)


        #save the prediction to a csv file
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")

    else:
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")

if task_name == "shortest_path":
    A, b = get_spp_Ab()
    #check if there is a csv file record the x solution
    if not os.path.exists(save_dir+"x_sol.csv"):
        # solve the shortest path problem
        x_sol,objs = solve_f_LP(c_test_pred,A,b,task_name)
        #save the solution to a csv file
        np.savetxt(save_dir+"x_sol.csv",x_sol,delimiter=",")
        np.savetxt(save_dir+"objs.csv",objs,delimiter=",")

elif task_name == "knapsack":
    # load prices and budgets
    prices = pd.read_csv("../data/knapsack/prices.csv",header=None).to_numpy()
    budgets = pd.read_csv("../data/knapsack/budgets.csv",header=None).to_numpy()
    #check if there is a npy file record the x solution
    
    

    num_constraints = budgets.shape[0]
    num_tests = c_test_pred.shape[0]

    x_sols = np.zeros((num_constraints,num_tests,prices.shape[1]))
    objss = np.zeros((num_constraints,num_tests))
    if not os.path.exists(save_dir+"x_sol.npy"):
        for cons_idx in range(num_constraints):
            A,b = get_kp_Ab(prices[cons_idx,:],budgets[cons_idx,:])
            # solve the knapsack problem
            x_sol,objs = solve_f_LP(c_test_pred,A,b,task_name)
            # save solution and obj to a npy file
            x_sols[cons_idx,:,:] = x_sol
            objss[cons_idx,:] = objs
        np.save(save_dir+"x_sol.npy",x_sols)
        np.save(save_dir+"objs.npy",objss)

            
    