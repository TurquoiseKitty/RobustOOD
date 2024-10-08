'''
find the best model (parameter) to fit the data
'''

#import random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
# import linear regression
from sklearn.linear_model import LinearRegression


import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from CP_alg import get_q, get_r
from get_uncertainty_set import get_box_US,get_ellipsoid_US
import ipdb

import argparse

sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_LUQ

task_name = "knapsack"
num_train_samples = 1000

deg = 2
plot_cov_dim = 1
model_name = "random_forest"

dim_covs = 5

param = 1 # parameter for bandwidth of kernel

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--plot_cov_dim', type=int, default=plot_cov_dim)
parser.add_argument('--model_name', type=str, default=model_name)
parser.add_argument('--param', type=float, default=param)
parser.add_argument('--dim_covs', type=int, default=dim_covs)
args = parser.parse_args()

task_name = args.task_name
num_train_samples = args.num_train_samples
deg = args.deg
plot_cov_dim = args.plot_cov_dim
model_name = args.model_name
param = args.param

dim_covs = args.dim_covs

train_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//train//"+str(num_train_samples)+"//"
test_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//test//"

#plot_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//plot//"+str(plot_cov_dim)+"//"
plot_dir = None


# split data in this dir
split_data_for_LUQ(train_dir)

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

if plot_dir is not None:
    # load the plot covs data
    covs_plot = pd.read_csv(plot_dir+"covs.csv",header=None).to_numpy()


# load the test data
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()
# load c.npy from test_dir as c_test
c_test = np.load(test_dir+"c.npy")

#preprocess data
scaler = StandardScaler()
covs_fit = scaler.fit_transform(covs_fit)
covs_UQ = scaler.transform(covs_UQ)
covs_cal = scaler.transform(covs_cal)
covs_test = scaler.transform(covs_test)
if plot_dir is not None:
    covs_plot = scaler.transform(covs_plot)

save_dir = LUQ_dir+model_name+"/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if not os.path.exists(save_dir+"c_test_pred.csv"):
    if model_name == "random_forest":
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

        

        # predict the c_UQ, c_cal, c_test
        rf = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'],max_depth=grid_search.best_params_['max_depth'],random_state=random_state)
        rf.fit(covs_fit,c_fit)
        c_UQ_pred = rf.predict(covs_UQ)
        c_cal_pred = rf.predict(covs_cal)
        c_test_pred = rf.predict(covs_test)
        
        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,rf.predict(covs_fit))
        # print the loss
        print("mse on fit data: ",mse)

        


        if plot_dir is not None:
            # save the plot data
            c_plot_pred = rf.predict(covs_plot)
            np.savetxt(plot_dir+"c_plot_pred.csv",c_plot_pred,delimiter=",")

    elif model_name=="MLP":

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_squared_error

        # Create a class for MLP model
        class MLP(nn.Module):
            def __init__(self, input_size, output_size, activation, neurons):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, neurons)
                self.fc2 = nn.Linear(neurons, 16)
                self.fc3 = nn.Linear(16, output_size)
                self.activation = activation
            
            def forward(self, x):
                x = self.activation(self.fc1(x))
                x = self.activation(self.fc2(x))
                x = self.fc3(x)
                return x

        # Create a function that builds the MLP model
        def build_model(activation='relu', neurons=16):
            model = MLP(covs_fit.shape[1], c_fit.shape[1], activation=nn.ReLU(), neurons=neurons)
            # use mse loss function
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(),lr=0.01)

            return model, loss_fn, optimizer

        # Create a PyTorch object with the build_model function and its default parameters
        model, loss_fn, optimizer = build_model()


        # predict the c_UQ, c_cal, c_test
        model.train()

        # define dataloader
        trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(covs_fit).float(),torch.tensor(c_fit).float()), batch_size=32, shuffle=True)


        # train the model

        for t in range(500):
            for covs, c in trainloader:
                c_pred = model(covs)
                loss = loss_fn(c_pred, c)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # save the model
        torch.save(model.state_dict(), save_dir+"MLP_model.pt")

        #calculate the loss on fit data and print it
        mse = mean_squared_error(c_fit,model(torch.tensor(covs_fit).float()).detach().numpy())
        print("MSE on fit data: ",mse)

        # predict the c_UQ, c_cal, c_test
        model.load_state_dict(torch.load(save_dir+"MLP_model.pt"))
        model.eval()
        c_UQ_pred = model(torch.tensor(covs_UQ).float()).detach().numpy()
        c_cal_pred = model(torch.tensor(covs_cal).float()).detach().numpy()
        c_test_pred = model(torch.tensor(covs_test).float()).detach().numpy()

        

        # save the plot data
        if plot_dir is not None:
            c_plot_pred = model.predict(covs_plot)
            np.savetxt(plot_dir+"c_plot_pred.csv",c_plot_pred,delimiter=",")



    elif model_name=="OLS":   

        from sklearn.linear_model import LinearRegression

        # predict the c_UQ, c_cal, c_test
        lr = LinearRegression()
        lr.fit(covs_fit,c_fit)
        c_UQ_pred = lr.predict(covs_UQ)
        c_cal_pred = lr.predict(covs_cal)
        c_test_pred = lr.predict(covs_test)

        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,lr.predict(covs_fit))
        # print the loss
        print("mse on fit data: ",mse)

        
        
    elif model_name=="NW":
        # use Nadaraya-Watson method with rbf kernel
        from sklearn.neighbors import KNeighborsRegressor

        ipdb.set_trace()

        # find the best parameter for the number of neighbors
        grid_search = GridSearchCV(KNeighborsRegressor(weights='distance',kernel='rbf'),{'n_neighbors':np.arange(1,20)},cv=5,scoring='neg_mean_squared_error')
        grid_search.fit(covs_fit,c_fit)
        param = grid_search.best_params_['n_neighbors']

        # predict the c_UQ, c_cal, c_test
        nw_model = KNeighborsRegressor(weights='distance',n_neighbors=param,kernel='rbf')
        nw_model.fit(covs_fit,c_fit)
        c_UQ_pred = nw_model.predict(covs_UQ)
        c_cal_pred = nw_model.predict(covs_cal)
        c_test_pred = nw_model.predict(covs_test)

        #calculate the loss on fit data and save it to a txt file
        mse = mean_squared_error(c_fit,nw_model.predict(covs_fit))
        # print the loss
        print("mse on fit data: ",mse)

        

    elif model_name=="SVR":
        from sklearn.svm import SVR

        svr_model = SVR(kernel='rbf', C=param, gamma='auto')

        # if the second dimension of covs_fit is 1, call ravels to convert it to a 1d array
        if covs_fit.shape[1]==1:
            c_fit = c_fit.ravel()
            c_UQ = c_UQ.ravel()
            c_cal = c_cal.ravel()

        svr_model.fit(covs_fit,c_fit)

        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = svr_model.predict(covs_UQ)
        c_cal_pred = svr_model.predict(covs_cal)
        c_test_pred = svr_model.predict(covs_test)

        # calculate and save the mse on fit data
        c_fit_pred = svr_model.predict(covs_fit)
        mse = mean_squared_error(c_fit,c_fit_pred)


    elif model_name=="Lasso":
        from sklearn.linear_model import Lasso

        # search the best param
        param_grid = {'alpha': np.logspace(-4, 4, 20)}
        lasso_model = Lasso()
        lasso_model_cv = GridSearchCV(lasso_model, param_grid, cv=5)
        lasso_model_cv.fit(covs_fit,c_fit)
        param = lasso_model_cv.best_params_['alpha']

        lasso_model = Lasso(alpha=param)
        lasso_model.fit(covs_fit,c_fit)

        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = lasso_model.predict(covs_UQ)
        c_cal_pred = lasso_model.predict(covs_cal)
        c_test_pred = lasso_model.predict(covs_test)

        # calculate and save the mse on fit data
        c_fit_pred = lasso_model.predict(covs_fit)
        mse = mean_squared_error(c_fit,c_fit_pred)


    elif model_name=="KernelRidge-rbf":
        from sklearn.kernel_ridge import KernelRidge

        # search the best param
        param_grid = {'alpha': np.logspace(-4, 4, 20)}
        kr_model = KernelRidge(kernel='rbf', gamma=0.1)
        kr_model_cv = GridSearchCV(kr_model, param_grid, cv=5)
        kr_model_cv.fit(covs_fit,c_fit)
        param = kr_model_cv.best_params_['alpha']

        kr_model = KernelRidge(kernel='rbf', alpha=param, gamma=0.1)
        kr_model.fit(covs_fit,c_fit)

        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = kr_model.predict(covs_UQ)
        c_cal_pred = kr_model.predict(covs_cal)
        c_test_pred = kr_model.predict(covs_test)

        # calculate and save the mse on fit data
        c_fit_pred = kr_model.predict(covs_fit)
        mse = mean_squared_error(c_fit,c_fit_pred)


    elif model_name=="KernelRidge-linear":
        from sklearn.kernel_ridge import KernelRidge

        # search the best param
        param_grid = {'alpha': np.logspace(-4, 4, 20)}
        kr_model = KernelRidge(kernel='linear')
        kr_model_cv = GridSearchCV(kr_model, param_grid, cv=5)
        kr_model_cv.fit(covs_fit,c_fit)

        param = kr_model_cv.best_params_['alpha']
        
        kr_model = KernelRidge(kernel='linear', alpha=param)
        kr_model.fit(covs_fit,c_fit)

        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = kr_model.predict(covs_UQ)
        c_cal_pred = kr_model.predict(covs_cal)
        c_test_pred = kr_model.predict(covs_test)

        # calculate and save the mse on fit data
        c_fit_pred = kr_model.predict(covs_fit)
        mse = mean_squared_error(c_fit,c_fit_pred)

    
    elif model_name=="KernelRidge-poly":
        from sklearn.kernel_ridge import KernelRidge

        # search the best param
        param_grid = {'alpha': np.logspace(-4, 4, 20)}
        kr_model = KernelRidge(kernel='poly', degree=2)
        kr_model_cv = GridSearchCV(kr_model, param_grid, cv=5)
        kr_model_cv.fit(covs_fit,c_fit)

        param = kr_model_cv.best_params_['alpha']

        kr_model = KernelRidge(kernel='poly', degree=2, alpha=param)
        kr_model.fit(covs_fit,c_fit)

        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = kr_model.predict(covs_UQ)
        c_cal_pred = kr_model.predict(covs_cal)
        c_test_pred = kr_model.predict(covs_test)

        # calculate and save the mse on fit data
        c_fit_pred = kr_model.predict(covs_fit)
        mse = mean_squared_error(c_fit,c_fit_pred)


    elif model_name=="KernelRidge-cosine":
        from sklearn.kernel_ridge import KernelRidge

        # search the best param
        param_grid = {'alpha': np.logspace(-4, 4, 20)}
        kr_model = KernelRidge(kernel='cosine')
        kr_model_cv = GridSearchCV(kr_model, param_grid, cv=5)
        kr_model_cv.fit(covs_fit,c_fit)

        param = kr_model_cv.best_params_['alpha']

        kr_model = KernelRidge(kernel='cosine', alpha=param)
        kr_model.fit(covs_fit,c_fit)

        # predict the c_UQ, c_cal, c_test
        c_UQ_pred = kr_model.predict(covs_UQ)
        c_cal_pred = kr_model.predict(covs_cal)
        c_test_pred = kr_model.predict(covs_test)

        # calculate and save the mse on fit data
        c_fit_pred = kr_model.predict(covs_fit)
        mse = mean_squared_error(c_fit,c_fit_pred)

    # c_test has (num_covs,dim_covs,num_c) shape, but c_test_pred has (num_covs,dim_covs) shape, so we need to repeat c_test_pred num_c times
    num_covs,dim_covs,num_c = c_test.shape
    c_test_pred_rep = np.repeat(c_test_pred.reshape((num_covs,dim_covs,1)),num_c,axis=2)
    #calculate the loss on test data and save it to a txt file
    mse_test = np.mean(np.sqrt(np.sum((c_test_pred_rep-c_test)**2,axis=1)))
    # print the loss
    print("mse on test data: ",mse_test)

    # save mse
    np.savetxt(save_dir+"mse.txt",np.array([mse]))
    # save mse_test
    np.savetxt(save_dir+"mse_test.txt",np.array([mse_test]))

    # save the result
    np.savetxt(save_dir+"c_UQ_pred.csv",c_UQ_pred,delimiter=",")
    np.savetxt(save_dir+"c_cal_pred.csv",c_cal_pred,delimiter=",")
    np.savetxt(save_dir+"c_test_pred.csv",c_test_pred,delimiter=",")

    """
    # calculate the distances between each pair of points
    c_UQ_weights = np.zeros((covs_UQ.shape[0],covs_fit.shape[0]))
    c_cal_weights = np.zeros((covs_cal.shape[0],covs_fit.shape[0]))
    c_test_weights = np.zeros((covs_test.shape[0],covs_fit.shape[0]))
    c_fit_weights = np.zeros((covs_fit.shape[0],covs_fit.shape[0]))

    dist_UQ = np.zeros((covs_UQ.shape[0],covs_fit.shape[0]))
    dist_cal = np.zeros((covs_cal.shape[0],covs_fit.shape[0]))
    dist_test = np.zeros((covs_test.shape[0],covs_fit.shape[0]))
    dist_fit = np.zeros((covs_fit.shape[0],covs_fit.shape[0]))
    bandwidth = param*np.power(covs_fit.shape[0],-1/(covs_fit.shape[1]+4))
    # calculate the weights: if distance is larger than bandwidth, weight is 0
    for i in range(covs_fit.shape[0]):
        for j in range(covs_UQ.shape[0]):
            dist_UQ[j,i] = np.linalg.norm(covs_fit[i,:]-covs_UQ[j,:])
            if dist_UQ[j,i]<bandwidth:
                c_UQ_weights[j,i] = 1
        for j in range(covs_cal.shape[0]):
            dist_cal[j,i] = np.linalg.norm(covs_fit[i,:]-covs_cal[j,:])
            if dist_cal[j,i]<bandwidth:
                c_cal_weights[j,i] = 1
        for j in range(covs_test.shape[0]):
            dist_test[j,i] = np.linalg.norm(covs_fit[i,:]-covs_test[j,:])
            if dist_test[j,i]<bandwidth:
                c_test_weights[j,i] = 1
        for j in range(covs_fit.shape[0]):
            dist_fit[j,i] = np.linalg.norm(covs_fit[i,:]-covs_fit[j,:])
            if dist_fit[j,i]<bandwidth:
                c_fit_weights[j,i] = 1
    


    # normalize the weights
    c_UQ_weights = c_UQ_weights/np.sum(c_UQ_weights,axis=1).reshape(-1,1)
    c_cal_weights = c_cal_weights/np.sum(c_cal_weights,axis=1).reshape(-1,1)
    c_test_weights = c_test_weights/np.sum(c_test_weights,axis=1).reshape(-1,1)

    # calculate and save the mse on fit data
    c_fit_pred = np.dot(c_fit_weights,c_fit)
    mse = mean_squared_error(c_fit,c_fit_pred)


    # predict the c_UQ, c_cal, c_test
    c_UQ_pred = np.dot(c_UQ_weights,c_fit)
    c_cal_pred = np.dot(c_cal_weights,c_fit)
    c_test_pred = np.dot(c_test_weights,c_fit)

    """

    """
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.models import Sequential
    from keras.layers import Dense

    # Create a function that builds the MLP model
    def build_model(optimizer='adam', activation='relu', neurons=16):
        model = Sequential()
        model.add(Dense(neurons, input_dim=covs_fit.shape[1], activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
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
    #np.savetxt(save_dir+"f_best_opt.txt",np.array([grid_result.best_params_['optimizer']]))
    #np.savetxt(save_dir+"f_best_act.txt",np.array([grid_result.best_params_['activation']]))
    #np.savetxt(save_dir+"f_best_neu.txt",np.array([grid_result.best_params_['neurons']]))

    # predict the c_UQ, c_cal, c_test
    model = Sequential()
    model.add(Dense(grid_result.best_params_['neurons'], input_dim=covs_fit.shape[1], activation=grid_result.best_params_['activation']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=grid_result.best_params_['optimizer'], metrics=['accuracy'])
    model.fit(covs_fit,c_fit,epochs=500,verbose=0)

    # print the loss on fit data
    print("loss on fit data: ",model.evaluate(covs_fit,c_fit)[0])

    c_UQ_pred = model.predict(covs_UQ)
    c_cal_pred = model.predict(covs_cal)
    c_test_pred = model.predict(covs_test)
    """

        
        
