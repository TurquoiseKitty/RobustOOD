'''get the LB and UB of uncertainty set'''
import numpy as np
import os
import pandas as pd

import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed
import ipdb


import sys
sys.path.append(os.getcwd())
sys.path.append('..')
import solver.RO_DNN as RO

data_name = "shortest_path"

deg = 1
num_train_samples = 5000
plot_cov_dim = 1
net_name  = "DCC"
n_cluster = 1

plot_path = "..\\..\\..\\data\\"+data_name+"\\01\\"+str(deg)+"\\plot\\"+str(plot_cov_dim)+"\\"
train_path =  "..\\..\\..\\data\\"+data_name+"\\01\\"+str(deg)+"\\train\\"+str(num_train_samples)+"\\"

file_name = plot_path+net_name+"_"+str(num_train_samples)+"_"+str(n_cluster)+"_"+"plot_assignments.npy"



resultdir = "..\\..\\..\\data\\"+data_name+"\\01\\"+str(deg)+"\\train\\"+str(num_train_samples)+"\\"+net_name+"\\"+str(n_cluster)+"\\"

alpha = 0.8


def get_worst_LB_UB_sigma(N,L,dimLayers,c0,cov, R, W, lb, ub, sigma):
    cov_inv = np.linalg.inv(cov)

    timeLimit = 5
    gap=0.01 


    ip = gp.Model("AdversarialProblem")
    ip.setParam("OutputFlag", 0)

    ip.setParam("TimeLimit", timeLimit)
    ip.setParam('MIPGap', gap)
    ip.setParam("FeasibilityTol", 0.000000001)

    # Create variables
    c = []

    c.append(ip.addVars(N, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="c[1]"))

    for i in range(1,L,1):
        c.append(ip.addVars(dimLayers[i-1], lb=0, vtype=GRB.CONTINUOUS, name="c["+ str(i+1) + "]"))
    c.append(ip.addVars(dimLayers[L-1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c["+ str(L) + "]"))
    
    xi = ip.addVars(dimLayers[L-1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="xi")

    
    for i in [L-1]:
        for l in range(0,dimLayers[i],1):
            dimL = dimLayers[i-1]
            lhs=""
            lhs = lhs + 1 * c[i+1][l]
            for j in range(0,dimL,1):
                lhs = lhs - W[i][l,j] * c[i][j]
                            
            ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))


    for i in range(0,L-1,1):
        for l in range(0,dimLayers[i],1):
            if i==0:
                dimL = N
            else:
                dimL = dimLayers[i-1]

            lhs=""
            for j in range(0,dimL,1):
                if sigma[i][l] > 0.5:
                    lhs = lhs + W[i][l,j] * c[i][j]
                else:
                    lhs = lhs - W[i][l,j] * c[i][j]
            
            ip.addConstr(lhs, sense=GRB.GREATER_EQUAL, rhs=0.0, name="W"+str(i+1)+"_L_" + str(l))
            
            # add auxiliary constraints to make the equilaties hold
            if sigma[i][l] > 0.5:
                lhs=""
                lhs = lhs + 1*c[i+1][l]
                for j in range(0,dimL,1):
                    lhs = lhs - W[i][l,j] * c[i][j]
                                
                ip.addConstr(lhs, sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))
                
            else:
                ip.addConstr(c[i+1][l], sense=GRB.EQUAL, rhs=0.0, name="c"+str(i+1)+"_" + str(l))

    #Add Ball-Constraint    
    lhs = ""
    for i in range(0,dimLayers[L-1],1):
        for j in range(0,dimLayers[L-1],1):
            lhs = lhs + xi[i] *cov_inv[i][j]* xi[j]
    
    ip.addQConstr(lhs, sense=GRB.LESS_EQUAL, rhs=R*R, name="Ball")
    
    #Add xi = c-c0 constraints
    for j in range(0,dimLayers[L-1],1):
        lhs = ""
        lhs = 1 * c[L][j] - 1 * xi[j] 
        ip.addConstr(lhs, sense=GRB.EQUAL, rhs=c0[j], name="xi" + str(j))


    LB = np.zeros(N)
    UB = np.zeros(N)    
    #Set objective
    for i in range(0,N,1):
        # get lower bound
        ip.setObjective(c[0][i],GRB.MINIMIZE)
        ip.optimize()
        if ip.status == GRB.Status.OPTIMAL:
            LB[i] = ip.objVal
            ipdb.set_trace()
        else:
            LB[i] = ub[i]

        # get upper bound
        ip.setObjective(c[0][i],GRB.MAXIMIZE)
        ip.optimize()
        if ip.status == GRB.Status.OPTIMAL:

            UB[i] = ip.objVal
            ipdb.set_trace()
        else:
            UB[i] = lb[i]
    
        
    return LB,UB

W_list = []
c_list = []
cov_list = []
L=3
for k in range(n_cluster):
    # load c and cov from dir's precedent dir, delimit by ","
    c = np.loadtxt(resultdir+"\\c_"+str(k)+".txt",delimiter=",")
    cov = np.loadtxt(resultdir+"\\cov_"+str(k)+".txt",delimiter=",")
    
    c_list.append(c)
    cov_list.append(cov)
    W = []
    for Li in range(L):
        W.append(np.loadtxt(resultdir+"\\W_"+str(k)+"_"+str(Li)+".txt",delimiter=","))
    W_list.append(W)

# load csv from train_path
X_train_main = pd.read_csv(train_path+"c.csv",header=None).to_numpy()

# get train labels
train_assignments = np.load(resultdir+"train_assignments.npy")

train_labels = np.argmax(train_assignments,axis=1)

zero_train_samples_cluster = []
for k in range(n_cluster):
    if np.sum(train_labels==k)==0:
        zero_train_samples_cluster.append(k)



dim_Y = X_train_main.shape[1]

plot_assignments = np.load(file_name)
num_plot_samples = plot_assignments.shape[0]
# get labels of plot data
plot_labels = np.argmax(plot_assignments,axis=1)


zero_plot_samples_cluster = []
for k in range(n_cluster):
    if np.sum(plot_labels==k)==0:
        zero_plot_samples_cluster.append(k)



LB_list = []
UB_list = []
for k in range(n_cluster):
    if k in zero_train_samples_cluster:
        LB_list.append(np.zeros(dim_Y))
        UB_list.append(np.zeros(dim_Y))
        continue

    ck = np.genfromtxt(resultdir+"c_"+str(k)+".txt", delimiter=',')
    cov = np.genfromtxt(resultdir+"cov_"+str(k)+".txt", delimiter=',')
    ipdb.set_trace()

    listW = []
    dimLayers = []

    for F in range(0,L,1):
        fileName = resultdir+"W_"+str(k)+"_"+str(F)+".txt"
        listW.append(np.genfromtxt(fileName, delimiter=','))
        dimLayers.append(listW[F].shape[0])
        
    N=listW[0].shape[1]


    # get activate layers -- sigmas
    _, R,sigmas,lb,ub = RO.getRadiiDataPoints(L,ck,cov,X_train_main[train_labels==k,:],listW, alpha)
    """
    # construct the constraints of the optimization problem
    LBs, UBs = zip(*Parallel(n_jobs=8)(delayed(get_worst_LB_UB_sigma)(N,L,dimLayers,ck,cov, R, W, lb, ub, sigma) for sigma in sigmas))
    LBs = np.array(LBs)
    UBs = np.array(UBs)

    plot_LB_k = np.zeros(dim_Y)
    plot_UB_k = np.zeros(dim_Y)
    for dim_idx in range(dim_Y):
        plot_LB_k[dim_idx] = np.min(LBs[:,dim_idx])
        plot_UB_k[dim_idx] = np.max(UBs[:,dim_idx])
    """
    LB_list.append(lb)
    UB_list.append(ub)



test_assignments = np.load(resultdir+"test_assignments.npy")


LB = np.zeros((num_plot_samples,dim_Y))
UB = np.zeros((num_plot_samples,dim_Y))
for i in range(num_plot_samples):
    LB[i,:] = LB_list[plot_labels[i]]
    UB[i,:] = UB_list[plot_labels[i]]

# save LB and UB to csv files
np.savetxt(plot_path+net_name+"_"+str(num_train_samples)+"_"+str(n_cluster)+"_"+str(alpha)+"_LB.csv",LB,delimiter=",")
np.savetxt(plot_path+net_name+"_"+str(num_train_samples)+"_"+str(n_cluster)+"_"+str(alpha)+"_UB.csv",UB,delimiter=",")




    




