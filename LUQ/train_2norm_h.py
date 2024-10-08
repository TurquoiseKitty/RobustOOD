'''
find the best model (parameter) to fit the data
'''

# import dicision tree regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import sys
# import keras to build MLP

from CP_alg import get_r
from get_uncertainty_set import get_box_US,get_box_US_with_betas

from solver import solve_ellipsoid,get_spp_Ab,get_kp_Ab,get_LB_UB_of_ellipsoid
import argparse
import ipdb


sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_LUQ
from data.read_mse_Var import get_Var
from data.coverage_solver import in_ellipsoid

task_name = "toy"
num_train_samples = 200

deg = 1
plot_cov_dim = 1

alpha = 0.8 # options: 0.8, 0.85, 0.9, 0.95, if fit_goal is "f", then alpha is not used
f_model_name = "SVR"
h_model_name = "kernel_ridge"
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

# calculate the 2-norm of the residual
res_UQ_2norm = np.sqrt(np.sum(res_UQ**2,axis=1)).reshape(-1,1)
res_cal_2norm = np.sqrt(np.sum(res_cal**2,axis=1)).reshape(-1,1)

save_dir = LUQ_dir+f_model_name+"/norm/"+h_model_name+"/"+str(alpha)+"/"

def quantile_loss_sklearn(y_true, y_pred):
    error = y_true - y_pred
    return np.mean(np.where(error >= 0, alpha * error, (1-alpha) * (1 - error)),axis=-1)

if not os.path.exists(save_dir+"res_cal_2norm_pred.csv"):
    # create the dir if not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if h_model_name=="DCT":
        from sklearn.tree import DecisionTreeRegressor
        # fit the quantile of residuals
        quantile = alpha
        # create the DecisionTreeRegressor and use grid search to find the best parameters
        random_state = 0
        #greate DecisionTreeRegressor
        DT = DecisionTreeRegressor(criterion=quantile_loss_sklearn,random_state=random_state)
        # define parameter grid
        param_grid = {'max_depth': [10,20,30,50,80]}

        #use grid search to find the best parameters
        grid_search = GridSearchCV(estimator = DT, param_grid = param_grid,
                                    cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(covs_UQ, res_UQ_2norm)
        # get the best parameters
        best_grid = grid_search.best_estimator_

        # fit the model
        best_grid.fit(covs_UQ, res_UQ_2norm)
        # predict the residual
        res_cal_2norm_pred = best_grid.predict(covs_cal)
        res_test_2norm_pred = best_grid.predict(covs_test)
        if plot_dir is not None:
            res_plot_2norm_pred = best_grid.predict(covs_plot)


    elif h_model_name == "grb" or h_model_name=="GRB" or h_model_name=="GBR":
        from sklearn.ensemble import GradientBoostingRegressor
        # fit the quantile of residuals
        quantile = alpha
        # create the GradientBoostingRegressor and use grid search to find the best parameters
        random_state = 0
        #greate GradientBoostingRegressor
        grb = GradientBoostingRegressor(loss='quantile',alpha=alpha,random_state=random_state)
        # define parameter grid
        # define parameter grid
        param_grid = {'learning_rate': [0.01,0.05,0.1,0.5,1],
                        'n_estimators': [100,200,500,800,1000],
                        'max_depth': [10,20,30,50,80]}
    
        
        #use grid search to find the best parameters
        grid_search = GridSearchCV(estimator = grb, param_grid = param_grid,
                                    cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(covs_UQ,res_UQ_2norm.ravel())
        # predict the residuals quantile of calibration and test data
        gbr = grid_search.best_estimator_

        res_UQ_2norm_pred = gbr.predict(covs_UQ)
        res_cal_2norm_pred = gbr.predict(covs_cal)
        res_test_2norm_pred = gbr.predict(covs_test)
        if plot_dir is not None:
            res_plot_2norm_pred = gbr.predict(covs_plot)
        
        
        


    elif h_model_name == "random_forest":
        random_state = 0

        #greate GradientBoostingRegressor
        rf = RandomForestRegressor(random_state=random_state)
        # define parameter grid
        param_grid = {'n_estimators': [100,200,500,800,1000],
                        'max_depth': [10,20,30,50,80]}
        
        #use grid search to find the best parameters
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                                    cv = 5, n_jobs = -1, verbose = 2)
        
        

        grid_search.fit(covs_UQ,res_UQ_2norm.ravel())

        #save best_est to a txt file
        np.savetxt(save_dir+"rf_best_est.txt",np.array([grid_search.best_params_['n_estimators']]))
        #save best_depth to a txt file
        np.savetxt(save_dir+"rf_best_depth.txt",np.array([grid_search.best_params_['max_depth']]))
        
        rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],max_depth=grid_search.best_params_['max_depth'],random_state=random_state)
        rf.fit(covs_UQ,res_UQ_2norm)
        res_cal_2norm_pred = rf.predict(covs_cal)
        res_test_2norm_pred = rf.predict(covs_test)
        
        #calculate the loss on UQ data and save it to a txt file
        res_UQ_2norm_pred = rf.predict(covs_UQ)
        np.savetxt(save_dir+"quantile_loss_test.txt",np.array([np.mean((res_UQ_2norm_pred-res_UQ_2norm)**2)]))

        if plot_dir is not None:
            res_plot_2norm_pred = rf.predict(covs_plot)
        

    elif h_model_name=="MLP":
        from keras.models import Sequential
        from keras.layers import Dense
        from keras import backend as K
        #ipdb.set_trace()
        from train_quantile_h import quantile_loss
        # create a sequantial model
        model = Sequential()

        # add a hidden layer
        model.add(Dense(16, input_dim=covs_UQ.shape[1], activation='relu'))
        # add output layer
        model.add(Dense(units=1, activation='linear'))

        #compile the model
        model.compile(loss=lambda y_true,y_pred:quantile_loss(alpha,y_true,y_pred), optimizer='adam')
        #train the model
        model.fit(covs_UQ, res_UQ_2norm, epochs=200, batch_size=32, verbose=0)

        # predict the residuals quantile of calibration and test data
        res_cal_2norm_pred = model.predict(covs_cal)
        res_test_2norm_pred = model.predict(covs_test)
        #calculate the loss on UQ data and save it to a txt file
        res_UQ_2norm_pred = model.predict(covs_UQ)
        np.savetxt(save_dir+"quantile_loss_test.txt",np.array([np.mean((res_UQ_2norm_pred-res_UQ_2norm)**2)]))

        if plot_dir is not None:
            res_plot_2norm_pred = model.predict(covs_plot)

    elif h_model_name=="Linear":
        from UQ_alg import UQ_train_quantile,UQ_test

        # use linear regression to fit the quantile of abs(res)
        betas = UQ_train_quantile(covs_UQ,res_UQ_2norm,alpha)

        # predict the residuals quantile of calibration and test data, take maximum between 0 and prediction
        res_cal_2norm_pred = np.maximum(UQ_test(covs_cal,betas),0)
        res_test_2norm_pred = np.maximum(UQ_test(covs_test,betas),0)
        #calculate the loss on UQ data and save it to a txt file
        res_UQ_2norm_pred = np.maximum(UQ_test(covs_UQ,betas),0)


        """
        # use linear regression to fit 2-norm of residual
        lr = LinearRegression(criterion=quantile_loss_sklearn)
        lr.fit(covs_UQ,res_UQ_2norm)
        # predict the residuals quantile of calibration and test data
        res_cal_2norm_pred = lr.predict(covs_cal)
        res_test_2norm_pred = lr.predict(covs_test)
        #calculate the loss on UQ data and save it to a txt file
        res_UQ_2norm_pred = lr.predict(covs_UQ)
        

        if plot_dir is not None:
            res_plot_2norm_pred = lr.predict(covs_plot)
        """
        np.savetxt(save_dir+"quantile_loss_test.txt",np.array([np.mean((res_UQ_2norm_pred-res_UQ_2norm)**2)]))

    elif h_model_name=="kernel_ridge":
        from sklearn.kernel_ridge import KernelRidge
        # use linear regression to fit 2-norm of residual
        kr = KernelRidge(kernel='poly')
        kr.fit(covs_UQ,res_UQ_2norm)
        # predict the residuals quantile of calibration and test data
        res_cal_2norm_pred = kr.predict(covs_cal)
        res_test_2norm_pred = kr.predict(covs_test)
        #calculate the loss on UQ data and save it to a txt file
        res_UQ_2norm_pred = kr.predict(covs_UQ)
        np.savetxt(save_dir+"quantile_loss_test.txt",np.array([np.mean((res_UQ_2norm_pred-res_UQ_2norm)**2)]))

        if plot_dir is not None:
            res_plot_2norm_pred = kr.predict(covs_plot)

    #calculate the covariance of the scaled residuals
    res_UQ_scaled = res_UQ/res_UQ_2norm_pred.reshape(-1,1)
    # fit the distribution of res_UQ_scaled with a zero mean multivariate normal distribution
    cov = res_UQ_scaled.T@res_UQ_scaled/(res_UQ_scaled.shape[0]-1)
    #cov = np.cov(res_UQ_scaled.T)
    #save the covariance matrix
    np.savetxt(save_dir+"cov.txt",cov)
    #save the 2norm predictions
    np.savetxt(save_dir+"res_cal_2norm_pred.csv",res_cal_2norm_pred,delimiter=",")
    np.savetxt(save_dir+"res_test_2norm_pred.csv",res_test_2norm_pred,delimiter=",")

    if plot_dir is not None:
        # save the plot data
        np.savetxt(plot_dir+"res_plot_2norm_pred.csv",res_plot_2norm_pred,delimiter=",")
else:
    # load the covariance matrix
    cov = np.loadtxt(save_dir+"cov.txt")

    # if cov is a scalar, then covert it to a 1*1 matrix
    cov = np.array(cov)
    if cov.shape==():
        cov = np.array([[cov]])
    

    # load the 2-norm predictions
    res_cal_2norm_pred = pd.read_csv(save_dir+"res_cal_2norm_pred.csv",header=None).to_numpy()
    res_test_2norm_pred = pd.read_csv(save_dir+"res_test_2norm_pred.csv",header=None).to_numpy()
    if plot_dir is not None:
        # load the plot data
        res_plot_2norm_pred = pd.read_csv(plot_dir+"res_plot_2norm_pred.csv",header=None).to_numpy()


# load c.npy from test_dir
c_test = np.load(test_dir+"c.npy")

#ipdb.set_trace()
num_covs,dim_c,num_c = c_test.shape
c_test_pred_rep = np.repeat(c_test_pred.reshape((num_covs,dim_c,1)),num_c,axis=2)
res_test_2norm = np.sum((c_test_pred_rep - c_test)**2,axis=1)**0.5
res_test_2norm_pred_rep = np.repeat(res_test_2norm_pred.reshape((num_covs,1)),num_c,axis=1)
quantile_loss_test = np.mean(quantile_loss_sklearn(res_test_2norm,res_test_2norm_pred_rep))
np.savetxt(save_dir+"quantile_loss_test.txt",np.array([quantile_loss_test]))


#save_dir = save_dir+str(alpha)+"/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# check if the solution file x_sol.csv is exist
if not os.path.exists(save_dir+"r.txt"):
    
    # calibrate the radius of the uncertainty ellipsoid
    r = get_r(alpha,res_cal,res_cal_2norm_pred,cov)
    # save the radius
    np.savetxt(save_dir+"r.txt",np.array([r]))
    # solve the optimization problem

else:
    # load the radius
    r = np.loadtxt(save_dir+"r.txt")

#################### get the LB and UB of each c dimension of the plot data ####################

if plot_dir is not None:
    plot_LB, plot_UB = get_LB_UB_of_ellipsoid(c_plot_pred,res_plot_2norm_pred,cov,r)

    # save the LB and UB
    # save plot_LB and plot_UB to csv
    np.savetxt(plot_dir+"PTC_2norm_"+str(num_train_samples)+"-"+str(alpha)+"_LB.csv",plot_LB,delimiter=",")
    np.savetxt(plot_dir+"PTC_2norm_"+str(num_train_samples)+"-"+str(alpha)+"_UB.csv",plot_UB,delimiter=",")


#%%
#################### solve the optimization problem ####################


if task_name=="shortest_path":
    A,b = get_spp_Ab()
    if not os.path.exists(save_dir+"x_sol.csv"):
        x_sol,objs = solve_ellipsoid(c_test_pred,res_test_2norm_pred,cov,r,A,b,task_name)
        # save the solution
        np.savetxt(save_dir+"x_sol.csv",x_sol,delimiter=",")
        np.savetxt(save_dir+"objs.csv",objs,delimiter=",")

elif task_name=="knapsack":
    # check if the solution file x_sol.npy is exist
    if not os.path.exists(save_dir+"x_sol.npy"):
        # load prices and budgets
        prices = np.loadtxt("../data/knapsack/prices.csv",delimiter=",")
        budgets = np.loadtxt("../data/knapsack/budgets.csv",delimiter=",")
        num_constraints = budgets.shape[0]
        num_test = c_test_pred.shape[0]
        # solve the optimization problem
        x_sols = np.zeros((num_constraints,num_test,prices.shape[1]))
        objss = np.zeros((num_constraints,num_test))
        for i in range(num_constraints):
            A, b = get_kp_Ab(prices[i,:],budgets[i])
            x_sol,objs = solve_ellipsoid(c_test_pred,res_test_2norm_pred,cov,r,A,b,task_name)
            x_sols[i,:,:] = x_sol
            objss[i,:] = objs
        # save the solution
        np.save(save_dir+"x_sol.npy",x_sols)
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

        # calculate the coverage
        coverage = in_ellipsoid(save_dir,c_test)
        coverage = np.mean(coverage,axis=1)
        # save the coverage
        np.savetxt(save_dir+"coverage.csv",coverage,delimiter=",")

print("PTC-ellipsoid, done!")