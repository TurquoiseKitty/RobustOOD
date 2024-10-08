# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:44:23 2023

@author: Admin
"""

import gurobipy
from gurobipy import GRB
import numpy as np
from scipy.stats import norm
import ipdb
from joblib import Parallel, delayed
import multiprocessing


def get_spp_Ab():
    '''
    Construct the constraint matrix A and vec b for the shortest path problem starting from [0,0] to [4,4] on a 5*5 grid network.
    '''
    n = 40
    m = 25
    A = np.zeros((m,n))
    b = np.zeros(m)
    for i in range(5):
        for j in range(5):
            v_idx = i*5+j
            if j!=4:
                #edge that point from v_idx to its right neighbor
                edge_to_right_idx = 9*i+j
                A[v_idx,edge_to_right_idx] = 1
            if i!=4:
                #edge that point from v_idx to its bottom neighbor
                edge_to_bottom_idx = 9*i+4+j
                A[v_idx,edge_to_bottom_idx] = 1
            if j!=0:
                #edge that point from the left neighbor to v_idx
                edge_from_left_idx = 9*i+j-1
                A[v_idx,edge_from_left_idx] = -1
            if i!=0:
                #edge that point from the top neighbor to v_idx
                edge_from_top_idx = 9*(i-1)+4+j
                A[v_idx,edge_from_top_idx] = -1
            if i==0 and j==0:
                b[v_idx] = 1
            elif i==4 and j==4:
                b[v_idx] = -1
    return A,b

def get_kp_Ab(price,budget):
    '''
    Construct the constraint matrix A and vec b for the 0-1 knapsack problem with 10 items and 10 knapsacks.
    '''
    n = len(price)
    m = 1
    A = np.ones((m,n))
    b = np.zeros(m)
    for j in range(n):
        A[0,j] = price[j]
    b[0] = budget
    return A,b


def solve_box(Uset_LB,Uset_UB,A,b,task_name):
    num_samples = Uset_LB.shape[0]
    n = Uset_LB.shape[1]
    x_sol = np.zeros((num_samples,n))
    objs = np.zeros(num_samples)
    
    for t in range(num_samples):
        model =  gurobipy.Model("RobustLP"+str(t))
        
        
        x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.CONTINUOUS,name="x")
        mu_p = model.addVars(range(n),lb=0,vtype=GRB.CONTINUOUS,name="mu_p")
        mu_n = model.addVars(range(n),lb=0,vtype=GRB.CONTINUOUS,name="mu_n")

        model.addConstrs(x[j]<=1 for j in range(n))
        #set objective    
        if task_name=="shortest_path":
            if A.size!=1:
                m = A.shape[0]
                model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))==b[i] for i in range(m))
            model.addConstrs(x[j]+mu_n[j]-mu_p[j]==0 for j in range(n))
            model.setObjective(gurobipy.quicksum(mu_p[j]*Uset_UB[t,j]-mu_n[j]*Uset_LB[t,j] for j in range(n)),GRB.MINIMIZE)
        elif task_name=="knapsack":
            model.addConstr(gurobipy.quicksum(A[0,j]*x[j] for j in range(n))<=b[0])
            model.addConstrs(-x[j]+mu_n[j]-mu_p[j]==0 for j in range(n))
            model.setObjective(gurobipy.quicksum(-mu_p[j]*Uset_UB[t,j]+mu_n[j]*Uset_LB[t,j] for j in range(n)),GRB.MAXIMIZE)
        model.Params.LogToConsole = 0 #not display, otherwise set to 1
        
        model.optimize()
        
        if model.status != GRB.Status.OPTIMAL:
            ipdb.set_trace()
            model.computeIIS()
            model.write("model1.ilp")
            
        else:
            for j in range(n):
                x_sol[t,j] = x[j].x
            objs[t] = model.objVal
            
    return x_sol,objs







def solve_true_model(gX,WX,alpha,A,b):
    num_samples = gX.shape[0]
    n = gX.shape[1]
    x_sol = np.zeros((num_samples,n))
    objs = np.zeros(num_samples)
    
    f_inv = norm.ppf(alpha,loc=0,scale=1)
    
    for t in range(num_samples):
        model =  gurobipy.Model("RobustLP"+str(t))
        
        
        x = model.addVars(range(n),lb=0,vtype=GRB.CONTINUOUS,name="x")
        v = model.addVar(lb=0,vtype=GRB.CONTINUOUS,name="v")
        if A.size!=1:
            m = A.shape[0]
            model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))==b[i] for i in range(m))
        
        #add SOCP constraint
        quadExpr = gurobipy.QuadExpr()
        for j in range(n):
            quadExpr.addTerms(WX[t,j], x[j],x[j])
        quadExpr.addTerms(-1,v,v)
        model.addQConstr(quadExpr<=0)

        #set objective    
        mu = gurobipy.quicksum(gX[t,j]*x[j] for j in range(n))
        model.setObjective(mu+f_inv*v,GRB.MINIMIZE)
        model.Params.LogToConsole = 0 #not display, otherwise set to 1
        
        model.optimize()
        
        for j in range(n):
            x_sol[t,j] = x[j].x
        objs[t] = model.objVal
            
    return x_sol,objs

def solve_ellipsoid(mean_pred,res_2norm_pred,cov,r,A,b,task_name):
    num_samples = mean_pred.shape[0]
    n = mean_pred.shape[1]
    x_sol = np.zeros((num_samples,n))
    objs = np.zeros(num_samples)

    Rs = (r*res_2norm_pred)

    # convert Rs to a 1d array
    Rs = Rs.reshape((Rs.shape[0],))


    x_sols, objs = zip(*Parallel(n_jobs=20)(delayed(solve_ellipsoid_single)(mean_pred[t,:].transpose(),cov,Rs[t],A,b,task_name) for t in range(num_samples)))
    x_sols = np.array(x_sols)
    objs = np.array(objs)

    return x_sols,objs
    


def solve_ellipsoid_single(mu,cov,R,A,b,task_name):
    n = A.shape[1]
    x_sol = np.zeros(n)
    
    model =  gurobipy.Model("RobustLP")
    
    v = model.addVar(lb=0,vtype=GRB.CONTINUOUS,name="v")
    if task_name=="shortest_path":
        x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.BINARY,name="x")
        # Add constraint: A*x==b
        m = A.shape[0]
        model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))==b[i] for i in range(m))
        
    elif task_name=="knapsack":
        x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.CONTINUOUS,name="x")
        # Add constraint: A*x<=b
        model.addConstr(gurobipy.quicksum(A[0,j]*x[j] for j in range(n))<=b[0])
    
    # add SOCP constraint
    quadExpr = gurobipy.QuadExpr()
    for i in range(n):
        for j in range(n):
            quadExpr.addTerms(cov[i,j], x[i],x[j])
    quadExpr.addTerms(-1,v,v)
    model.addQConstr(quadExpr<=0)

    #set objective
    if task_name=="shortest_path":
        model.setObjective(gurobipy.quicksum(mu[i]*x[i] for i in range(n))+R*v,GRB.MINIMIZE)
    elif task_name=="knapsack":
        model.setObjective(gurobipy.quicksum(mu[i]*x[i] for i in range(n))-R*v,GRB.MAXIMIZE)
    model.Params.LogToConsole = 0 #not display, otherwise set to 1
    model.Params.NonConvex = 2 #allow non-convex quadratic constraints
    model.optimize()

    if model.status != GRB.Status.OPTIMAL:
        ipdb.set_trace()
        model.computeIIS()
        model.write("model1.ilp")
    else:
        for j in range(n):
            x_sol[j] = x[j].x

    return x_sol,model.objVal
        
def get_LB_UB_of_ellipsoid(mean_pred,res_2norm_pred,cov,r):
    num_samples = mean_pred.shape[0]
    n = mean_pred.shape[1]

    LB = np.zeros((num_samples,n))
    UB = np.zeros((num_samples,n))

    Rs = (r*res_2norm_pred)

    cov_inv = np.linalg.inv(cov)

    # convert Rs to a 1d array
    Rs = Rs.reshape((Rs.shape[0],))

    for t in range(num_samples):
        model =  gurobipy.Model("RobustLP"+str(t))

        c = model.addVars(range(n),vtype=GRB.CONTINUOUS,name="w")
        
        # add SOCP constraint
        """
        quadExpr = gurobipy.QuadExpr()
        for i in range(n):
            for j in range(n):
                quadExpr.addTerms(cov_inv[i,j], c[i],c[j])
                quadExpr.addTerms(-cov_inv[i,j]*mean_pred[t,j], c[i])
                quadExpr.addTerms(-cov_inv[i,j]*mean_pred[t,i], c[j])
        model.addQConstr(quadExpr<=Rs[t]*Rs[t]-mean_pred[t,:]@cov_inv@(mean_pred[t,:].transpose()))
        """
        quadExpr = ""
        for i in range(n):
            for j in range(n):
                quadExpr += cov_inv[i,j]*(c[i]-mean_pred[t,i])*(c[j]-mean_pred[t,j])
        
        model.addConstr(quadExpr<=Rs[t]*Rs[t])

        
        model.Params.LogToConsole = 0 #not display, otherwise set to 1
        for c_dim in range(n):
            #set objective
            model.setObjective(c[c_dim],GRB.MINIMIZE)

            model.optimize()
            if model.status != GRB.Status.OPTIMAL and model.status != GRB.Status.SUBOPTIMAL:
                ipdb.set_trace()
            else:
                LB[t,c_dim] = model.objVal
            
            model.setObjective(c[c_dim],GRB.MAXIMIZE)
            model.optimize()
            if model.status != GRB.Status.OPTIMAL and model.status != GRB.Status.SUBOPTIMAL:
                ipdb.set_trace()
            else:
                UB[t,c_dim] = model.objVal

    return LB,UB


"""
def solve_ellipsoid(mean_pred,res_2norm_pred,cov,r,A,b,task_name):
    # get the ellipsoid shape
    L = np.linalg.cholesky(cov)
    LT = L.transpose()

    # define R
    R = r*res_2norm_pred
    
    num_samples = mean_pred.shape[0]
    n = mean_pred.shape[1]
    x_sol = np.zeros((num_samples,n))
    objs = np.zeros(num_samples)

    for t in range(num_samples):
        model = ro.Model('solveRobustPortfolio_Ellipsoidal')
        mu = mean_pred[t,:].transpose()
        x=model.dvar(n)
        z=model.rvar(n)
        EllipsoidSet=(rso.norm(z,2)<=1)
        P = R[t]*LT
        if task_name=="shortest_path":
            model.minmax((mu+P@z)@x,EllipsoidSet)
        elif task_name=="knapsack":
            model.minmax(-(mu+P@z)@x,EllipsoidSet)
        
        model.st(x<=1)
        model.st(x>=0)
        if task_name=="shortest_path":
            model.st(A@x==b)
        elif task_name=="knapsack":
            model.st(A[0,:]@x<=b[0])

        model.solve(grb)

        x_sol[t,:] = x.get()
        if task_name=="shortest_path":
            objs[t] = model.get()
        elif task_name == "knapsack":
            objs[t] = -model.get()


    return x_sol,objs
"""    