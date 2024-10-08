

import numpy as np
import gurobipy
from gurobipy import GRB
from joblib import Parallel, delayed
import multiprocessing

import ipdb

def solve_ellipsoid(mus,covs,Rs,A,b,task_name):
    num_samples = mus.shape[0]
    n = mus.shape[1]


    x_sols, objs = zip(*Parallel(n_jobs=20)(delayed(solve_ellipsoid_single)(mus[t,:].transpose(),covs[t,:,:],Rs[t],A,b,task_name) for t in range(num_samples)))

    """
    x_sols = []
    objs = []
    for t in range(num_samples):
        x_sol,obj = solve_ellipsoid_single(mus[t,:].transpose(),covs[t,:,:],Rs[t],A,b,task_name)
        x_sols.append(x_sol)
        objs.append(obj)
    """
        
    x_sol = np.array(x_sols)
    objs = np.array(objs)
    return x_sol,objs

def solve_ellipsoid_single(mu,cov,R,A,b,task_name):

    n = A.shape[1]
    model = gurobipy.Model("RobustLP")

    if task_name=="shortest_path":
        x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.BINARY,name="x")
        # constraints Ax = b
        model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))==b[i] for i in range(A.shape[0]))
    elif task_name=="knapsack":
        x = model.addVars(range(n),lb=0,ub=1,vtype=GRB.CONTINUOUS,name="x")
        # constraints Ax<=b
        model.addConstrs(gurobipy.quicksum(A[i,j]*x[j] for j in range(n))<=b[i] for i in range(A.shape[0]))

    
    v = model.addVar(lb=0,vtype=GRB.CONTINUOUS,name="v")

    # check if covs[t,:,:] is positive semi-definite
    if not np.all(np.linalg.eigvals(cov) >= 0):
        ipdb.set_trace()
    # check if covs[t,:,:] is symmetric
    if not np.allclose(cov, cov.T, atol=1e-12):
        ipdb.set_trace()

    #add SOCP constraint
    quadExpr = gurobipy.QuadExpr()
    for i in range(n):
        for j in range(n):
            quadExpr.addTerms(cov[i,j], x[i],x[j])
    quadExpr.addTerms(-1,v,v)
    model.addQConstr(quadExpr==0)
    """
    
    L = np.linalg.cholesky(covs[t,:,:])

    Lx = model.addVars(range(n),lb=-gurobipy.GRB.INFINITY,vtype=GRB.CONTINUOUS,name="Lx")
    model.addConstrs(Lx[i]==gurobipy.quicksum(L[i,j]*x[j] for j in range(n)) for i in range(n))
    """

    #set objective
    if task_name=="shortest_path":
        model.setObjective(gurobipy.quicksum(mu[i]*x[i] for i in range(n))+R*v,GRB.MINIMIZE)
    elif task_name=="knapsack":
        model.setObjective(gurobipy.quicksum(mu[i]*x[i] for i in range(n))-R*v,GRB.MAXIMIZE)

    model.Params.LogToConsole = 0 #not display, otherwise set to 1
    model.Params.NonConvex = 2 #2: allow non-convex quadratic constraints

    # set time limit
    model.Params.TimeLimit = 20

    model.optimize()

    x_sol = np.zeros(n)
    if model.status == GRB.Status.OPTIMAL or model.status== GRB.Status.SUBOPTIMAL:
        for i in range(n):
            x_sol[i] = x[i].x
        
    else:
        if task_name=="shortest_path":
            model.setObjective(gurobipy.quicksum(mu[i]*x[i] for i in range(n)),GRB.MINIMIZE)
        elif task_name=="knapsack":
            model.setObjective(gurobipy.quicksum(mu[i]*x[i] for i in range(n)),GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.Status.OPTIMAL or model.status== GRB.Status.SUBOPTIMAL:
            for i in range(n):
                x_sol[i] = x[i].x
        else:
            ipdb.set_trace()
            print("Optimization was stopped with status %d" % model.status)

        #ipdb.set_trace()
        #print("Optimization was stopped with status %d" % model.status)
    return x_sol,model.objVal

def get_LB_UB(mus,covss,Rs):
    num_samples = mus.shape[0]
    n = mus.shape[1]

    LB = np.zeros((num_samples,n))
    UB = np.zeros((num_samples,n))

    # convert Rs to a 1d array
    Rs = Rs.reshape((Rs.shape[0],))

    for t in range(num_samples):
        model =  gurobipy.Model("kNN"+str(t))
        cov_inv = np.linalg.inv(covss[t,:,:])

        c = model.addVars(range(n),vtype=GRB.CONTINUOUS,name="w")
        
        # add SOCP constraint
        quadExpr = gurobipy.QuadExpr()
        for i in range(n):
            for j in range(n):

                quadExpr.addTerms(cov_inv[i,j], c[i],c[j])
                quadExpr.addTerms(cov_inv[i,j]*(-mus[t,j]), c[i])
                quadExpr.addTerms(cov_inv[i,j]*(-mus[t,i]), c[j])


        model.addQConstr(quadExpr<=Rs[t]*Rs[t]-mus[t,:]@cov_inv@(mus[t,:].transpose()))

        
        model.Params.LogToConsole = 0 #not display, otherwise set to 1
        for c_dim in range(n):
            #set objective
            model.setObjective(c[c_dim],GRB.MINIMIZE)

            model.optimize()
            """
            if model.status != GRB.Status.OPTIMAL and model.status!= GRB.Status.SUBOPTIMAL:
                ipdb.set_trace()
            else:
                LB[t,c_dim] = model.objVal
            """
            LB[t,c_dim] = model.objVal
            
            model.setObjective(c[c_dim],GRB.MAXIMIZE)
            model.optimize()
            """
            if model.status != GRB.Status.OPTIMAL and model.status!= GRB.Status.SUBOPTIMAL:
                ipdb.set_trace()
            else:
                UB[t,c_dim] = model.objVal
            """
            UB[t,c_dim] = model.objVal


    return LB,UB

"""
def solve_ellipsoid(mus,covss,Rs,A,b,task_name):

    
    num_samples = mus.shape[0]
    n = mus.shape[1]
    x_sol = np.zeros((num_samples,n))
    objs = np.zeros(num_samples)

    for t in range(num_samples):
        # get the ellipsoid shape
        L = np.linalg.cholesky(covss[t,:,:])
        LT = L.transpose()

        R = Rs[t]

        model = ro.Model('solveRobustPortfolio_Ellipsoidal')
        mu = mus[t,:].transpose()
        x=model.dvar(n)
        z=model.rvar(n)
        EllipsoidSet=(rso.norm(z,2)<=1)
        P = R*LT
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