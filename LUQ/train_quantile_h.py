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
from keras import backend as K

from CP_alg import get_q, get_q_with_betas
from get_uncertainty_set import get_box_US,get_box_US_with_betas

from UQ_alg import UQ_train_quantile,UQ_test
from solver import solve_box,solve_true_model,get_spp_Ab,get_kp_Ab
import argparse
import ipdb


sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_LUQ
# import get_Var from ../data/rea_mse_Var.py
from data.read_mse_Var import get_Var
from data.coverage_solver import in_box

task_name = "shortest_path"
num_train_samples = 5000
deg = 1
plot_cov_dim = 1

alpha = 0.8 # options: 0.8, 0.85, 0.9, 0.95, if fit_goal is "f", then alpha is not used
f_model_name = "random_forest"
h_model_name = "MLP"

dim_covs = 5

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--plot_cov_dim', type=int, default=plot_cov_dim )
parser.add_argument('--alpha', type=float, default=alpha)
parser.add_argument('--f_model_name', type=str, default=f_model_name)
parser.add_argument('--h_model_name', type=str, default=h_model_name)
parser.add_argument('--dim_covs', type=int, default=dim_covs)
args = parser.parse_args()

task_name = args.task_name
num_train_samples = args.num_train_samples
deg = args.deg
plot_cov_dim = args.plot_cov_dim
alpha = args.alpha
f_model_name = args.f_model_name
h_model_name = args.h_model_name
dim_covs = args.dim_covs

train_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//train//"+str(num_train_samples)+"//"
test_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//test//"
#plot_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//plot//"+str(plot_cov_dim)+"//"
plot_dir = None


#define quantile loss
def quantile_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def quantile_loss_sklearn(alpha,y_true, y_pred):
    error = y_true - y_pred
    return np.mean(np.where(error >= 0, alpha * error, (1-alpha) * (1 - error)),axis=-1)

# turn to LUQ dir
LUQ_dir = train_dir+"LUQ/"

# load the fit data
fit_dir = LUQ_dir+"fit/"
covs_fit = pd.read_csv(fit_dir+"covs_fit.csv",header=None).to_numpy()
c_fit = pd.read_csv(fit_dir+"c_fit.csv",header=None).to_numpy()

# load the UQ data
UQ_dir = LUQ_dir+"UQ/"
covs_UQ = pd.read_csv(UQ_dir+"covs_UQ.csv",header=None).to_numpy()
c_UQ = pd.read_csv(UQ_dir+"c_UQ.csv",header=None).to_numpy()

# load the calibration data
cal_dir = LUQ_dir+"cal/"
covs_cal = pd.read_csv(cal_dir+"covs_cal.csv",header=None).to_numpy()
c_cal = pd.read_csv(cal_dir+"c_cal.csv",header=None).to_numpy()

# load the test data
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()
# load the c.npy data from test_dir
c_test = np.load(test_dir+"c.npy")

# load the plot data
if plot_dir is not None:
    covs_plot = pd.read_csv(plot_dir+"covs.csv",header=None).to_numpy()

#preprocess data
scaler = StandardScaler()
covs_fit = scaler.fit_transform(covs_fit)
covs_UQ = scaler.transform(covs_UQ)
covs_cal = scaler.transform(covs_cal)
covs_test = scaler.transform(covs_test)
if plot_dir is not None:
    covs_plot = scaler.transform(covs_plot)

# load the prediction data
pred_dir = LUQ_dir+f_model_name+"/"
c_UQ_pred = pd.read_csv(pred_dir+"c_UQ_pred.csv",header=None).to_numpy()
c_cal_pred = pd.read_csv(pred_dir+"c_cal_pred.csv",header=None).to_numpy()
c_test_pred = pd.read_csv(pred_dir+"c_test_pred.csv",header=None).to_numpy()
if plot_dir is not None:
    c_plot_pred = pd.read_csv(plot_dir+"c_plot_pred.csv",header=None).to_numpy()

# calculate the residual
res_UQ = c_UQ - c_UQ_pred
res_cal = c_cal - c_cal_pred

save_dir = LUQ_dir+f_model_name+"/quantile/"+h_model_name+"/"+str(alpha)+"/"

# if the result already exists, then skip
if not os.path.exists(save_dir+"x_sol.csv") and not os.path.exists(save_dir+"x_sol.npy"):
    
    if not os.path.exists(save_dir+"resq_cal_pred.csv"):
        # create the dir if not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if h_model_name == "gbr" or h_model_name == "GBR":
            random_state = 0

            # for each coordinate of the residual, fit a gradient boosting regressor
            resq_cal_pred = np.zeros((res_cal.shape[0],res_cal.shape[1]))
            resq_test_pred = np.zeros((covs_test.shape[0],res_cal.shape[1]))
            resq_UQ_pred = np.zeros((res_UQ.shape[0],res_UQ.shape[1]))

            # define the parameters
            for i in range(res_UQ.shape[1]):
                #create GradientBoostingRegressor
                gbr = GradientBoostingRegressor(loss='quantile',alpha=alpha,random_state=0)

                gbr.fit(covs_UQ,np.abs(res_UQ[:,i])) # use the default parameter

                """
                # learn the best parameters
                # define parameter grid
                param_grid = {'learning_rate': [0.01,0.05,0.1,0.5,1],
                                'n_estimators': [100,200,500,800],
                                'max_depth': [10,20,30,50,80]}
                
                #use grid search to find the best parameters
                grid_search = GridSearchCV(estimator = gbr, param_grid = param_grid,
                                            cv = 5, n_jobs = -1, verbose = 2)
                
                grid_search.fit(covs_UQ,np.abs(res_UQ[:,i]))

                # predict the residuals quantile of calibration and test data
                gbr = grid_search.best_estimator_
                """
                
                resq_cal_pred[:,i] = gbr.predict(covs_cal)
                resq_test_pred[:,i] = gbr.predict(covs_test)
                
                resq_UQ_pred[:,i] = gbr.predict(covs_UQ)
            
            #calculate the loss on UQ data and save it to a txt file
            loss = np.sum(np.sum(quantile_loss(alpha,np.abs(res_UQ),resq_UQ_pred)))

            np.savetxt(save_dir+"loss.txt",np.array([loss]))
            
            if plot_dir is not None:
                # deal plot data
                resq_plot_pred = gbr.predict(covs_plot)
            
        elif h_model_name=="DCT":
            from sklearn.tree import DecisionTreeRegressor
            # fit the quantile of residuals
            quantile = alpha
            # create the DecisionTreeRegressor and use grid search to find the best parameters
            random_state = 0

            resq_cal_pred = np.zeros((res_cal.shape[0],res_cal.shape[1]))
            resq_test_pred = np.zeros((covs_test.shape[0],res_cal.shape[1]))
            resq_UQ_pred = np.zeros((res_UQ.shape[0],res_UQ.shape[1]))
            
            for i in range(res_UQ.shape[1]):
                dtr = DecisionTreeRegressor(criterion=quantile_loss_sklearn,random_state=random_state)
                dtr.fit(covs_UQ,np.abs(res_UQ[:,i]))
                resq_cal_pred[:,i] = dtr.predict(covs_cal)
                resq_test_pred[:,i] = dtr.predict(covs_test)
                resq_UQ_pred[:,i] = dtr.predict(covs_UQ)

            

        elif h_model_name=="MLP":
            # import keras to build MLP
            from keras.models import Sequential
            from keras.layers import Dense
            

            # create a sequantial model
            model = Sequential()

            # add a hidden layer
            model.add(Dense(16, input_dim=covs_UQ.shape[1], activation='relu'))
            # add output layer
            model.add(Dense(units=c_UQ.shape[1], activation='linear'))

            y_true = np.abs(res_UQ)
            #compile the model
            model.compile(loss=lambda y_true,y_pred:quantile_loss(alpha,y_true,y_pred), optimizer='adam')
            #train the model
            model.fit(covs_UQ, y_true, epochs=500, batch_size=32, verbose=0)

            # predict the residuals quantile of calibration and test data
            resq_cal_pred = model.predict(covs_cal)
            resq_test_pred = model.predict(covs_test)
            #calculate the loss on UQ data and save it to a txt file
            resq_UQ_pred = model.predict(covs_UQ)
            if plot_dir is not None:
                # deal plot data
                resq_plot_pred = model.predict(covs_plot)

            loss = np.sum(np.sum(quantile_loss(alpha,np.abs(res_UQ),resq_UQ_pred)))
            np.savetxt(save_dir+"loss.txt",np.array([loss]))


        elif h_model_name=="Linear":
            abs_c = np.abs(res_UQ)
            # use linear regression to fit the quantile of abs(res)
            betas = UQ_train_quantile(covs_UQ,abs_c,alpha)

            # predict the residuals quantile of calibration and test data
            resq_cal_pred = UQ_test(covs_cal,betas)
            resq_test_pred = UQ_test(covs_test,betas)
            #calculate the loss on UQ data and save it to a txt file
            resq_UQ_pred = UQ_test(covs_UQ,betas)
            

        #save the resq predictions
        np.savetxt(save_dir+"resq_cal_pred.csv",resq_cal_pred,delimiter=",")
        np.savetxt(save_dir+"resq_test_pred.csv",resq_test_pred,delimiter=",")
        if plot_dir is not None:
            # save the plot data
            # deal plot data
            resq_plot_pred = UQ_test(covs_plot,betas)
            np.savetxt(plot_dir+"resq_plot_pred.csv",resq_plot_pred,delimiter=",")
    else:
        # load the resq predictions
        resq_cal_pred = pd.read_csv(save_dir+"resq_cal_pred.csv",header=None).to_numpy()
        resq_test_pred = pd.read_csv(save_dir+"resq_test_pred.csv",header=None).to_numpy()
        if plot_dir is not None:
            # load the plot data
            resq_plot_pred = pd.read_csv(plot_dir+"resq_plot_pred.csv",header=None).to_numpy()

    
    num_covs,dim_covs,num_c = c_test.shape
    c_test_pred_rep = np.repeat(c_test_pred.reshape((num_covs,dim_covs,1)),num_c,axis=2)
    res_test = c_test_pred_rep - c_test
    resq_test_pred_rep = np.repeat(resq_test_pred.reshape((num_covs,dim_covs,1)),num_c,axis=2)
    quantile_loss_test = np.mean(np.mean(quantile_loss(alpha,np.abs(res_test),resq_test_pred_rep)))
    np.savetxt(save_dir+"quantile_loss_test.txt",np.array([quantile_loss_test]))

    
    # calibrate the resq predictions
    q = get_q(alpha,c_cal,c_cal_pred,resq_cal_pred)
    # use the calibrated parameter to get uncertainty set
    test_LB,test_UB = get_box_US(c_test_pred,resq_test_pred,q)
    # save test_LB and test_UB to csv
    np.savetxt(save_dir+"test_LB.csv",test_LB,delimiter=",")
    np.savetxt(save_dir+"test_UB.csv",test_UB,delimiter=",")

    if plot_dir is not None:
        # use the calibrated parameter to get uncertainty set for plot data
        plot_LB,plot_UB = get_box_US(c_plot_pred,resq_plot_pred,q)
        # save plot_LB and plot_UB to csv
        np.savetxt(plot_dir+"PTC_quantile_"+str(num_train_samples)+"-"+str(alpha)+"_LB.csv",plot_LB,delimiter=",")
        np.savetxt(plot_dir+"PTC_quantile_"+str(num_train_samples)+"-"+str(alpha)+"_UB.csv",plot_UB,delimiter=",")

    if f_model_name=="NW":
        ipdb.set_trace()

    if task_name == "shortest_path":
        A, b = get_spp_Ab()
        #check if there is a csv file record the x solution
        if not (os.path.exists(save_dir+"x_sol.csv")):
            # solve the problem
            x_sol,objs = solve_box(test_LB,test_UB,A,b,task_name)
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
            num_tests = test_LB.shape[0]
            x_sols = np.zeros((num_constraints,num_tests,prices.shape[1]))
            objss = np.zeros((num_constraints,num_tests))
            for i in range(num_constraints):
                A,b = get_kp_Ab(prices[i,:],budgets[i])
                x_sol,objs = solve_box(test_LB,test_UB,A,b,task_name)
                x_sols[i,:,:] = x_sol
                objss[i,:] = objs
            # save the x solution
            np.save(save_dir+"x_sol.npy",x_sols)
            # save the objective values
            np.save(save_dir+"objs.npy",objss)




if task_name!="toy":        
    if not os.path.exists(save_dir+"VaR.csv") or not os.path.exists(save_dir+"obj_pos.csv") or not os.path.exists(save_dir+"coverage.csv"):
        # load c.npy from test_dir
        c_test = np.load(test_dir+"c.npy")
        # load x_sol and objs
        if task_name == "shortest_path":
            x_sol = pd.read_csv(save_dir+"x_sol.csv",header=None).to_numpy()
            objs = pd.read_csv(save_dir+"objs.csv",header=None).to_numpy()
            (num_test_covs,dim_x) = x_sol.shape
            num_test_c = c_test.shape[-1]

            obj_positions = np.zeros((num_test_covs))
            for cov_idx in range(num_test_covs):
                true_objs = x_sol[cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                obj_positions[cov_idx] = np.sum(true_objs<=objs[cov_idx])/num_test_c
            
        elif task_name == "knapsack":
            x_sol = np.load(save_dir+"x_sol.npy")
            objs = np.load(save_dir+"objs.npy")
            (num_constraints,num_test_covs,dim_x) = x_sol.shape
            num_test_c = c_test.shape[-1]
            
            obj_positions = np.zeros((num_constraints,num_test_covs))
            for cons_idx in range(num_constraints):
                for cov_idx in range(num_test_covs):
                    true_objs = x_sol[cons_idx,cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                    obj_positions[cons_idx,cov_idx] = np.sum(true_objs>objs[cons_idx,cov_idx])/num_test_c

        # save the obj_positions
        np.savetxt(save_dir+"obj_pos.csv",obj_positions,delimiter=",")    
        
        
        # calculate the VaR
        VaR = get_Var(x_sol,c_test,alpha,task_name)
        # save the VaR
        np.savetxt(save_dir+"VaR.csv",VaR,delimiter=",")


        coverage = in_box(save_dir,c_test)
        coverage = np.mean(coverage,axis=1)
        np.savetxt(save_dir+"coverage.csv",coverage,delimiter=",")

print("PTC-box, done!")

