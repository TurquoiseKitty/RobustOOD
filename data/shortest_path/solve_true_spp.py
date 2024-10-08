
import numpy as np
import gurobipy
from gurobipy import GRB
from scipy.stats import norm
import os
import sys
from joblib import Parallel, delayed
import multiprocessing

sys.path.append(os.getcwd())
sys.path.append('..')
sys.path.append('../..')
from LUQ.solver import get_spp_Ab

def solve_true_spp(true_f,true_var,alpha):
    num_samples = true_f.shape[0]
    n = true_f.shape[1]
    x_sol = np.zeros((num_samples,n))
    objs = np.zeros(num_samples)
    f_inv = norm.ppf(alpha,loc=0,scale=1)
    A,b = get_spp_Ab()

    for t in range(num_samples):
        model =  gurobipy.Model("RobustLP"+str(t))
        
        x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.BINARY,name="x")
        v = model.addVar(lb=0,vtype=GRB.CONTINUOUS,name="v")
        if A.size!=1:
            m = A.shape[0]
            model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))==b[i] for i in range(m))
        
        #add SOCP constraint
        quadExpr = gurobipy.QuadExpr()
        for j in range(n):
            quadExpr.addTerms(true_var[t,j]**2, x[j],x[j])
        quadExpr.addTerms(-1,v,v)
        model.addQConstr(quadExpr<=0)

        #set objective    
        mu = gurobipy.quicksum(true_f[t,j]*x[j] for j in range(n))
        model.setObjective(mu+f_inv*v,GRB.MINIMIZE)
        model.Params.LogToConsole = 0 #not display, otherwise set to 1
        
        model.optimize()
        
        for j in range(n):
            x_sol[t,j] = x[j].x
        objs[t] = model.objVal
            
    return x_sol,objs

def solve_expected_spp(true_f):
    num_samples = true_f.shape[0]
    n = true_f.shape[1]
    x_sol = np.zeros((num_samples,n))

    A,b = get_spp_Ab()

    x_sol,objs = zip(*Parallel(n_jobs=20)(delayed(solve_expected_spp_single)(true_f[t,:],A,b) for t in range(num_samples)))

    x_sol = np.array(x_sol)
    objs = np.array(objs)
            
    return x_sol,objs

def solve_expected_spp_single(true_f_single,A,b):
    n = len(true_f_single)
    model =  gurobipy.Model("RobustLP")
        
    x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.BINARY,name="x")
    
    if A.size!=1:
        m = A.shape[0]
        model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))==b[i] for i in range(m))
    
    

    #set objective    
    mu = gurobipy.quicksum(true_f_single[j]*x[j] for j in range(n))
    model.setObjective(mu,GRB.MINIMIZE)
    model.Params.LogToConsole = 0 #not display, otherwise set to 1
    
    model.optimize()

    x_sol = np.zeros(n)
    for j in range(n):
        x_sol[j] = x[j].x

    return x_sol,model.objVal


def get_VaR_of_x(true_f,true_var,x,alpha):
    f_inv = norm.ppf(alpha,loc=0,scale=1)
    n = true_f.shape[1]

    x_VaR = np.sum(true_f*x,axis=1)+f_inv*np.sqrt(np.sum((true_var*x)**2,axis=1))

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