# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:29:41 2023

@author: Admin
"""
import gurobipy
from gurobipy import GRB
import numpy as np

def UQ_test(X,betas):
    return np.matmul(X,betas[:X.shape[1],:])+np.repeat(betas[X.shape[1],:][np.newaxis,:],X.shape[0],axis=0)



def UQ_train_quantile(UQ_X,res_Y,tau):
    X_dims = UQ_X.shape[1]
    Y_dims = res_Y.shape[1]
    num_samples = UQ_X.shape[0]
    
    abs_c = np.abs(res_Y)
    
    betas = np.zeros([X_dims+1,Y_dims])
    
    #construct LP models and solve them
    for j in range(Y_dims):
        model = gurobipy.Model(str(j))
        
        #add decision variables
        rho_p = model.addVars(range(num_samples),lb=0,vtype=GRB.CONTINUOUS,name="rho_p")
        rho_n = model.addVars(range(num_samples),lb=0,vtype=GRB.CONTINUOUS,name="rho_n")
        rho = model.addVars(range(num_samples),lb=0,vtype=GRB.CONTINUOUS,name="rho")
        beta = model.addVars(range(X_dims+1),vtype=GRB.CONTINUOUS,name="beta")
        
        #add constraints
        model.addConstrs(rho_p[t]>=tau*(abs_c[t,j]-(beta[X_dims]+gurobipy.quicksum([beta[i]*UQ_X[t,i] for i in range(X_dims)]))) for t in range(num_samples))
        model.addConstrs(rho_n[t]>=(1-tau)*((beta[X_dims]+gurobipy.quicksum([beta[i]*UQ_X[t,i] for i in range(X_dims)]))-abs_c[t,j]) for t in range(num_samples))
        model.addConstrs(rho[t]>=rho_p[t]+rho_n[t] for t in range(num_samples))
        
        #set objective
        model.setObjective(gurobipy.quicksum(rho),GRB.MINIMIZE)
        
        #setting the solver
        model.Params.LogToConsole = 0 #not display, otherwise set to 1
        
        model.optimize()
        
        if model.status == GRB.Status.OPTIMAL:
            #get beta_value
            for i in range(X_dims+1):
                betas[i,j] = beta[i].x
        else:
            raise Exception("beta not solved")
            
    return betas
        
        
        