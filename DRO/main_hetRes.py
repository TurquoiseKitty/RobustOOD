'''
find the best model (parameter) to fit the data
'''

#import random forest regressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
#from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# import linear regression method and Lasso method
from sklearn.linear_model import LinearRegression, Lasso

import argparse

#from keras.models import Sequential
#from keras.layers import Dense


from solver import solve_wasserstein_res,get_spp_Ab,get_kp_Ab
from tqdm import tqdm


sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_HetRes

task_name = "knapsack"
num_train_samples = 500

deg = 2


f_model_name = "OLS"
h_model_name = "homo" # options: example1-2, homo

parser = argparse.ArgumentParser(
                    prog='hetRes',
                    description='hetRes or homoRes')

parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--f_model_name',type=str, default=f_model_name)
parser.add_argument('--h_model_name',type=str, default=h_model_name)

args = parser.parse_args()

task_name = args.task_name
num_train_samples = args.num_train_samples

deg = args.deg


f_model_name = args.f_model_name
h_model_name = args.h_model_name

train_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = "../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

# support set: Tx<=phi
T = None
phi = None

# turn to HetRes dir
LUQ_dir = train_dir+"HetRes/"
# split data in this dir
split_data_for_HetRes(train_dir,LUQ_dir,train_ratio=0.9,max_UQ_num = 100)



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
        c_test_pred = rf.predict(covs_test)
        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,rf.predict(covs_fit))
        np.savetxt(save_dir+"rf_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
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

        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,reg.predict(covs_fit))
        np.savetxt(save_dir+"OLS_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")
elif f_model_name=="Lasso":
    if not os.path.exists(save_dir):
        # using Lasso model to fit
        reg = Lasso().fit(covs_fit, c_fit)
        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = reg.predict(covs_UQ)
        c_test_pred = reg.predict(covs_test)
        c_fit_pred = reg.predict(covs_fit)

        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,reg.predict(covs_fit))
        np.savetxt(save_dir+"Lasso_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")

"""
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
        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,model.predict(covs_fit))
        np.savetxt(save_dir+"MLP_mse.txt",np.array([mse]))

        #save the prediction to a csv file
        np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
        np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")
        np.savetxt(save_dir+"c_fit_pred.csv",c_fit_pred,delimiter=",")
    else:
        c_UQ_pred = np.loadtxt(save_dir+"c_UQ_pred.csv",delimiter=",")
        c_test_pred = np.loadtxt(save_dir+"c_test_pred.csv",delimiter=",")
        c_fit_pred = np.loadtxt(save_dir+"c_fit_pred.csv",delimiter=",")
"""
#%%
############### fit Q(x) #################
from HetRes_Example import train_q
# calculate the residual
save_dir = save_dir+h_model_name+"/"
res_fit = c_fit - c_fit_pred
# judge if the q is already calculated
if not os.path.exists(save_dir+"q_test.csv"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if h_model_name=="example1-2":

        # train the heteroscedastic model
        q2_UQ,q2_fit,q2_test = train_q(covs_fit,res_fit,covs_UQ,covs_test)
        # get the square root of q2
        q_fit = np.sqrt(q2_fit)
        q_UQ = np.sqrt(q2_UQ)
        q_test = np.sqrt(q2_test)
        # save the q
        np.savetxt(save_dir+"q_fit.csv",q_fit,delimiter=",")
        np.savetxt(save_dir+"q_UQ.csv",q_UQ,delimiter=",")
        np.savetxt(save_dir+"q_test.csv",q_test,delimiter=",")

        # get the eps residuals
        eps_UQ = (c_UQ - c_UQ_pred)/q_UQ
        eps_fit = res_fit/q_fit


        # save the eps residuals
        np.savetxt(save_dir+"eps_fit.csv",eps_fit,delimiter=",")
    elif h_model_name=="homo":
        eps_fit = res_fit
        q_UQ = np.ones(c_UQ.shape)
        q_test = np.ones((covs_test.shape[0],c_UQ.shape[1]))
        np.savetxt(save_dir+"q_UQ.csv",q_UQ,delimiter=",")
        np.savetxt(save_dir+"q_test.csv",q_test,delimiter=",")
        np.savetxt(save_dir+"eps_fit.csv",eps_fit,delimiter=",")

else:
    # load data
    q_UQ = pd.read_csv(save_dir+"q_UQ.csv",header=None).to_numpy()
    q_test = pd.read_csv(save_dir+"q_test.csv",header=None).to_numpy()
    eps_fit = pd.read_csv(save_dir+"eps_fit.csv",header=None).to_numpy()


#%%
############## select radius ################
eps_set = [0,0.05,0.1,0.15,0.2]

if task_name == "shortest_path":
    A, b = get_spp_Ab()
    
    # objective under different eps
    ERMs = np.zeros(len(eps_set))
    for eps_idx,eps in enumerate(eps_set):
        if not (os.path.exists(save_dir+"eps-"+str(eps)+".txt")):
            x_sol_UQ,objs_UQ = solve_wasserstein_res(eps,A,b,T,phi,c_UQ_pred,q_UQ,eps_fit,task_name)
            ERM = np.sum(x_sol_UQ*c_UQ,axis=1)
            ERMs[eps_idx] = np.mean(ERM)
            np.savetxt(save_dir+"eps-"+str(eps)+".txt",np.array([ERMs[eps_idx]]))
        else:
            ERMs[eps_idx] = np.loadtxt(save_dir+"eps-"+str(eps)+".txt")
    # find the best eps
    best_eps = eps_set[np.argmin(ERMs)]
    #save best eps
    np.savetxt(save_dir+"best_eps.csv",np.array([best_eps]),delimiter=",")

    #check if there is a csv file record the x solution
    if not (os.path.exists(save_dir+"x_sol.csv")):
        # solve the problem
        x_sol,objs = solve_wasserstein_res(best_eps,A,b,T,phi,c_test_pred,q_test,eps_fit,task_name)
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
                x_sol_UQ,objs_UQ = solve_wasserstein_res(eps,A,b,T,phi,c_UQ_pred,q_UQ,eps_fit,task_name)
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
            x_sol,objs = solve_wasserstein_res(best_eps,A,b,T,phi,c_test_pred,q_test,eps_fit,task_name)
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the x solution
        np.save(save_dir+"x_sol.npy",x_sols)
        # save the objective values
        np.save(save_dir+"objs.npy",objss)




