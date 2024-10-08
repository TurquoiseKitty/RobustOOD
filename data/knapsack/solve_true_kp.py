
import numpy as np
import gurobipy
from gurobipy import GRB
from scipy.stats import norm
import os
import sys
from joblib import Parallel, delayed
import multiprocessing
import ipdb

sys.path.append(os.getcwd())
sys.path.append('..')
sys.path.append('../..')
from LUQ.solver import get_spp_Ab,get_kp_Ab


def solve_true_kp(true_f,true_var,alpha,prices,budgets):
    """Note: this is a max-min problem, not a min-max"""
    num_samples = true_f.shape[0]
    num_constraints = prices.shape[0]
    n = true_f.shape[1]

    x_sols = np.zeros((num_constraints,num_samples,n))
    objss = np.zeros((num_constraints,num_samples))
    
    f_inv = norm.ppf((1-alpha),loc=0,scale=1)
    
    for n_inst in range(num_constraints):
        A,b = get_kp_Ab(prices[n_inst],budgets[n_inst])
        for t in range(num_samples):
            model =  gurobipy.Model("RobustLP"+str(t))
            
            
            x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.CONTINUOUS,name="x")
            v = model.addVar(lb=0,vtype=GRB.CONTINUOUS,name="v")

            model.addConstr(gurobipy.quicksum(A[0,j]*x[j] for j in range(n))<=b[0])
            
            #add SOCP constraint
            quadExpr = gurobipy.QuadExpr()
            for j in range(n):
                quadExpr.addTerms(true_var[t,j]**2, x[j],x[j])
            quadExpr.addTerms(-1,v,v)
            model.addQConstr(quadExpr>=0)

            #set objective    
            mu = gurobipy.quicksum(true_f[t,j]*x[j] for j in range(n))
            model.setObjective(mu+f_inv*v,GRB.MAXIMIZE)
            model.Params.LogToConsole = 0 #not display, otherwise set to 1
            
            model.optimize()
            
            for j in range(n):
                x_sols[n_inst,t,j] = x[j].x
            objss[n_inst,t] = model.objVal
            
    return x_sols,objss

def solve_expected_kp(true_f,prices,budgets):
    """Note: this is a max-min problem, not a min-max"""
    num_samples = true_f.shape[0]
    num_constraints = prices.shape[0]
    n = true_f.shape[1]

    x_sols = np.zeros((num_constraints,num_samples,n))
    objss = np.zeros((num_constraints,num_samples))
    
    
    for n_inst in range(num_constraints):
        A,b = get_kp_Ab(prices[n_inst],budgets[n_inst])

        x_sol_list, obj_list = zip(*Parallel(n_jobs=20)(delayed(solve_expected_single)(true_f[t,:],A,b) for t in range(num_samples)))

        x_sols[n_inst,:,:] = np.array(x_sol_list)
        objss[n_inst,:] = np.array(obj_list)

    return x_sols,objss

def solve_expected_single(true_f_single,A,b):
    n = len(A)
    model =  gurobipy.Model("expectedLP")
            
            
    x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.CONTINUOUS,name="x")


    model.addConstr(gurobipy.quicksum(A[0,j]*x[j] for j in range(n))<=b[0])
    


    #set objective    
    mu = gurobipy.quicksum(true_f_single[j]*x[j] for j in range(n))
    model.setObjective(mu,GRB.MAXIMIZE)
    model.Params.LogToConsole = 0 #not display, otherwise set to 1
    
    model.optimize()

    x_val = np.zeros(n)
    for j in range(n):
        x_val[j] = x[j].x

    return x_val,model.objVal



def get_VaR_of_x(true_f,true_var,x,alpha):
    """Note: this is a max-min problem, not min-max"""
    f_inv = norm.ppf((1-alpha),loc=0,scale=1)
    n = true_f.shape[1]

    num_instances = x.shape[0]
    num_samples = true_f.shape[0]
    x_VaR = np.zeros((num_instances,num_samples))
    for n_inst in range(num_instances):
        x_cur = x[n_inst,:].reshape((num_samples,n))
        x_VaR[n_inst,:] = np.sum(true_f*x_cur,axis=1)+f_inv*np.sqrt(np.sum((true_var*x_cur)**2,axis=1))

    return x_VaR


def in_box(test_LB,test_UB,c_test):
    # judge if c_test satisfies test_LB<=c_test<=test_UB
    result = np.zeros(c_test.shape[0])
    for t in range(c_test.shape[0]):
        if (c_test[t,:]>=test_LB[t,:]).all() and (c_test[t,:]<=test_UB[t,:]).all():
            result[t] = 1
    return np.mean(result)

def in_ellipsoid(true_y,pred_y,pred_norm,cov,r):
    # judge if true_y lies in the ellisoid of shope cov and radius r

    cov_inv = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov_inv)
    LT = L.transpose()

    reg = np.matmul((true_y-pred_y)/pred_norm,L)
    distances = np.sqrt(np.sum(reg**2,axis=1))

    result = distances<=r

    return np.mean(result)

def in_kNN_ellipsoid(true_y,mus,sigmas,Rs):
    # judge if true_y lies in the ellipsoid of kNN
    result = np.zeros(mus.shape[0])
    for i in range(mus.shape[0]):
        cov_inv = np.linalg.inv(sigmas[i,:,:])
        L = np.linalg.cholesky(cov_inv)
        LT = L.transpose()

        reg = np.matmul((true_y[i,:]-mus[i])/Rs[i],L)
        distances = np.sqrt(np.sum(reg**2))

        result[i] = distances<=1

    return np.mean(result)


def in_DNN_ellipsoid(n_cluster,W_list,test_assignment,R_list,c_list,cov_list,X):
    result = np.zeros(X.shape[0])
    LP_list = []
    for i in range(n_cluster):
        cov_inv = np.linalg.inv(cov_list[i])
        # decompose cov_inv
        cov_inv_L = np.linalg.cholesky(cov_inv)
        LP_list.append(cov_inv_L.transpose())

    for j in range(0,X.shape[0],1):
        outLayer = X[j,:]
        label = np.argmax(test_assignment[j,:])
        listW = W_list[label]
        for i in range(0,len(listW)-1,1):
            outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
            # print(i,outLayer)

        outLayer = np.dot(listW[len(listW)-1],outLayer)
        # print(X[i],outLayer,np.linalg.norm(outLayer-c0))

        result[j] = np.linalg.norm(LP_list[label]@(outLayer-c_list[label]))<=R_list[label]
    
    return np.mean(result)
