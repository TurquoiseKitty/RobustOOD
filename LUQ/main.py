# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:34:04 2023

@author: Linyu
"""
import pandas as pd
import numpy as np
import GP_model
from UQ_alg import UQ_train
from CP_alg import get_q

from sklearn import datasets, linear_model
from solver import solve,solve_true_model
from VaRs import compute_VaR,Obj2alpha

model_name = "perfect" #GP,linear_regression,perfect

#%% load data

#Warning: should make sure using the same data when in comparison
"""
#portfolio data
data=pd.read_csv(r"..\Data-Driven-Conditional-Robust-Optimization-main\path\data\finance\final_port.csv")

data.DATE=pd.DatetimeIndex(data.DATE)


train=data[(data.DATE.dt.year<=2015)]
UQ = data[(data.DATE.dt.year==2016)]
calibrate_data=data[(data.DATE.dt.year==2017)]
test = data[(data.DATE.dt.year==2018)]

#drop 'DATE' column
train.drop('DATE', axis=1, inplace=True)
UQ.drop('DATE',axis=1,inplace=True)
calibrate_data.drop('DATE',axis=1,inplace=True)
test.drop('DATE', axis=1, inplace=True)

#dataframe to numpy format
train = np.array(train)
UQ = np.array(UQ)
calibrate_data = np.array(calibrate_data)
test = np.array(test)


#seperate covariates and labels
X_dims = range(15,train.shape[1])
Y_dims = range(15)

train_X = train[:,X_dims]
UQ_X = UQ[:,X_dims]
calibrate_X = calibrate_data[:,X_dims]
test_X = test[:,X_dims]

train_Y = train[:,Y_dims]
UQ_Y = UQ[:,Y_dims]
calibrate_Y = calibrate_data[:,Y_dims]
test_Y = test[:,Y_dims]
"""

#synthetic data
d = 10 #dims of covariate
n = 3 #dims of random parameters
num_train = 100000
num_UQ = 1000
num_calibrate = 2000
num_test = 2000
num_total = num_train+num_UQ+num_calibrate+num_test

X = np.random.uniform(-1,1,[num_total,d])
V1 = np.random.uniform(-1,1,[d,n])
V2 = np.random.uniform(-1,1,[d,n])
v3 = np.random.uniform(-1,1,[1,d])
W = np.random.uniform(-0.5,0.5,[d,n])
phi1 = np.random.uniform(-1,1,1)
phi2 = np.random.uniform(-1,1,1)
theta = np.random.uniform(0.5,5,1)
gX = phi1*np.sin(2*np.pi*np.matmul(X,V1))+phi2*np.matmul((X-v3.repeat(num_total,axis=0))**2,V2)
WX = np.matmul(X,W)**2


Y = np.zeros((num_total,n))
for i in range(num_total):
    Y[i,:] = np.random.multivariate_normal(gX[i,:],cov=np.diag(WX[i,:]))
    

if n<=1:
    A=0
    b=0
elif n==2:
    A = np.array([1,1])
    b = np.array([1])
else:
    m = np.random.randint(1,n-1)
    A = np.random.uniform(0,1,[m,n])
    b = np.random.uniform(0,1,[m,1])

train_X = X[:num_train,:]
UQ_X = X[num_train:num_train+num_UQ,:]
calibrate_X = X[num_train+num_UQ:num_train+num_UQ+num_calibrate,:]
test_X = X[num_train+num_UQ+num_calibrate:,:]
    
train_Y = Y[:num_train,:]
UQ_Y = Y[num_train:num_train+num_UQ,:]
calibrate_Y = Y[num_train+num_UQ:num_train+num_UQ+num_calibrate,:]
test_Y = Y[num_train+num_UQ+num_calibrate:,:]


#%% data preprocess
#normalie X data to [0,1] using train data
max_X = train_X.max(axis=0)
min_X = train_X.min(axis=0)
train_X = (train_X-min_X)/(max_X-min_X)
UQ_X = (UQ_X-min_X)/(max_X-min_X)
calibrate_X = (calibrate_X-min_X)/(max_X-min_X)
test_X = (test_X-min_X)/(max_X-min_X)


#norm Y to 1.0 variance
"""
mean_y = np.mean(train_Y,axis=0)
std_y = np.std(train_Y,axis=0)
train_Y = (train_Y)/std_y
UQ_Y = (UQ_Y)/std_y
calibrate_Y = (calibrate_Y)/std_y
test_Y = (test_Y)/std_y
"""

"""
y_mean = np.mean(train_Y)
train_Y = train_Y-y_mean
calibrate_Y = calibrate_Y-y_mean
test_Y = test_Y-y_mean
"""


#%% training fitted model f

if model_name == "GP":
    GPlist = GP_model.groupGP_train(train_X, train_Y)
    pred_mean,_ = GP_model.groupGP_test(GPlist, train_X)

if model_name == "linear_regression":
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_Y)
    pred_mean = regr.predict(train_X)
if model_name == "perfect":
    pred_mean = gX[:num_train,:]

mse = sum(sum((pred_mean-train_Y)**2)/train_X.shape[0])


#%% uncertainty quantification
tau = 0.5	
print("tau=",tau)
if model_name == "GP":
    UQ_pred_y,_ = GP_model.groupGP_test(GPlist, UQ_X)
if model_name == "linear_regression":
    UQ_pred_y = regr.predict(UQ_X)    
if model_name == "perfect":
    UQ_pred_y = gX[num_train:num_train+num_UQ,:]

uqmse = sum(sum((UQ_pred_y-UQ_Y)**2)/UQ_X.shape[0])

betas = UQ_train(UQ_X,UQ_Y,UQ_pred_y,tau)

#%% conformal prediction
alpha = 0.5
print("alpha=",alpha)
if model_name == "GP":
    calibrate_pred_Y,_ = GP_model.groupGP_test(GPlist, calibrate_X)
if model_name =="linear_regression":
    calibrate_pred_Y = regr.predict(calibrate_X)
if model_name == "perfect":
    calibrate_pred_Y = gX[num_train+num_UQ:num_train+num_UQ+num_calibrate,:]
eta = get_q(alpha,calibrate_X,calibrate_Y,calibrate_pred_Y,betas)
print("eta=",eta)
#%% construct uncertainty set
if model_name == "GP":
    test_pred_Y,_ = GP_model.groupGP_test(GPlist,test_X)
if model_name == "linear_regression":
    test_pred_Y = regr.predict(test_X)
if model_name == "perfect":
    test_pred_Y = gX[num_train+num_UQ+num_calibrate:,:]
res = eta*np.matmul(test_X,betas)
Uset_LB = (test_pred_Y-res)
Uset_UB = (test_pred_Y+res)

#%% solve RobustLP
x_sol,RLP_objs = solve(Uset_LB,Uset_UB,A,b)

if model_name == "perfect":
    gX_sub = gX[num_train+num_UQ+num_calibrate:,:]
    WX_sub = WX[num_train+num_UQ+num_calibrate:,:]
    true_x, true_opt = solve_true_model(gX_sub,WX_sub,alpha,A,b)

"""
#optimal solution and value for portfolio optimiztion
x_inds = np.argmax(Uset_LB,axis=1)
x_sol = np.zeros(test_pred_Y.shape)
for i in range(x_sol.shape[0]):
    x_sol[i,x_inds[i]]=1
    
RLP_objs = np.sum(Uset_LB*x_sol,axis=1)
"""    
#%% evaluate the solution for minimization problem
if model_name=="perfect":
    gX_sub = gX[num_train+num_UQ+num_calibrate:,:]
    WX_sub = WX[num_train+num_UQ+num_calibrate:,:]
    true_vars = compute_VaR(gX_sub,WX_sub,x_sol,alpha)
    
    
    print("ratio of upper bounding true VaR:",np.sum(true_vars<=RLP_objs)/RLP_objs.size)
    gaps = (true_vars-true_opt)/np.abs(true_opt)*100
    print("mean of gaps=",np.mean(gaps),"%")
    emp_alphas = Obj2alpha(gX_sub,WX_sub,x_sol,RLP_objs)
    print("mean of empirical alphas=",np.mean(emp_alphas))
    
else:
    #actual_objs = np.sum(std_y*test_Y*x_sol,axis=1)
    actual_objs = np.sum(test_Y*x_sol,axis=1)
    empirical_VaR = np.sum(actual_objs<RLP_objs)/test_Y.shape[0]

actual_objs = np.sum(test_Y*x_sol,axis=1)
print("ratio of upper bounding actual obj:",np.sum(actual_objs<=RLP_objs)/RLP_objs.size)