import RO_DNN as RO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random as r
import gurobipy as gp
from gurobipy import GRB
import csv
import torch
import sys
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os
import ipdb
from tqdm import tqdm
import argparse

#import one-hot encoding method
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.getcwd())
sys.path.append('../../../')
from data.read_mse_Var import get_Var
from data.coverage_solver import in_DNN_ellipsoid

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

def testPolicyVaR(returns,delta):
    tmp = sorted(returns)
    VaR=tmp[int(np.floor(delta*len(returns)))]
    return VaR


def backtest_policy(val,VaR):
    x=[1 for i in val if i>VaR]
    return np.sum(x)/len(val)

task_name = "shortest_path"
deg = 2
alpha = 0.8
net_name = "IDCC"
n_cluster = 10
num_train_samples = 5000
dim_covs = 5

parse = argparse.ArgumentParser()
parse.add_argument('--task_name', type=str, default=task_name)
parse.add_argument('--deg', type=int, default=deg)
parse.add_argument('--alpha', type=float, default=alpha)
parse.add_argument('--net_name', type=str, default=net_name)
parse.add_argument('--n_cluster', type=int, default=n_cluster)
parse.add_argument('--num_train_samples', type=int, default=num_train_samples)
parse.add_argument('--dim_covs', type=int, default=dim_covs)

args = parse.parse_args()


task_name = args.task_name
deg = args.deg
alpha = args.alpha
net_name = args.net_name
n_cluster = args.n_cluster
num_train_samples = args.num_train_samples
dim_covs = args.dim_covs


if task_name=="knapsack":
    prices = np.loadtxt("../../../data/knapsack/prices.csv",delimiter=",")
    budgets = np.loadtxt("../../../data/knapsack/budgets.csv",delimiter=",")
    num_constraints = prices.shape[0]

train_path = "../../../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_path = "../../../data/"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"


analysisdir = train_path+net_name+"/"+str(n_cluster)+"/"


resultdir=train_path+net_name+"/"+str(n_cluster)+"/"

save_dir = analysisdir+str(alpha)+"/"

if not os.path.exists(save_dir+"x_sol.csv") and not os.path.exists(save_dir+"x_sol.npy"):

    # load train_alignments.npy and test_alignments.npy
    train_assignments = np.load(resultdir+"train_assignments.npy")
    test_assignments = np.load(resultdir+"test_assignments.npy")



    num_tests = test_assignments.shape[0]

    train_labels = np.argmax(train_assignments,axis=1)
    # check if there are clusters with zero samples
    num_train_sample_in_each_cluster = []
    zero_sample_clusters = []
    for k in range(n_cluster):
        num_sample = np.sum(train_labels==k)
        num_train_sample_in_each_cluster.append(num_sample)
        if num_sample==0:
            zero_sample_clusters.append(k)

    test_labels = np.argmax(test_assignments,axis=1)
    # check if test_labels is in zero_sample_clusters, if so, assign it to the index of test_assignments with second highest probability
    for i in tqdm(range(test_labels.shape[0])):
        non_assigned = False
        while test_labels[i] in zero_sample_clusters:
            test_assignments[i,test_labels[i]] = 0
            test_labels[i] = np.argmax(test_assignments[i,:])
            if np.sum(test_assignments[i,:])==0:
                non_assigned = True
                break
        if non_assigned:
            # assign a cluster with maximum samples
            test_labels[i] = np.argmax(num_train_sample_in_each_cluster)
            test_assignments[i,test_labels[i]] = 1



    #ipdb.set_trace()

    # #read data
    # load csv file as a numpy array


    X_train_main = pd.read_csv(train_path+"c.csv",header=None).to_numpy()

    fileName = resultdir+"W_"+str(0)+"_"+str(0)+".txt"
    W = np.genfromtxt(fileName, delimiter=',')
    # if W is one dimensional, reshape it to 2D
    if len(W.shape)==1:
        W = W.reshape((W.shape[0],1))

    N = W.shape[1]


    if task_name=="knapsack":
        x_sols = np.zeros((num_constraints,num_tests,prices.shape[1]))
        objss = np.zeros((num_constraints,num_tests))
    elif task_name=="shortest_path":
        x_sols = np.zeros((num_tests,N))
        objss = np.zeros((num_tests))


    # #solve neural network
    L = 3 #number of layers


    for k in tqdm(range(n_cluster)):
        print("k=",k)
        fileName = resultdir+"c_"+str(k)+".txt"
        ck = np.genfromtxt(fileName, delimiter=',')
        cov = np.genfromtxt(resultdir+"cov_"+str(k)+".txt", delimiter=',')

        listW = []
        dimLayers = []

        for F in range(0,L,1):
            fileName = resultdir+"W_"+str(k)+"_"+str(F)+".txt"
            W = np.genfromtxt(fileName, delimiter=',')
            if len(W.shape)==1:
                W = W.reshape((W.shape[0],1))
            listW.append(W)
            dimLayers.append(listW[F].shape[0])



        if not (k in zero_sample_clusters):
            maxEntry = np.amax(X_train_main[train_labels==k,:])
            M=[]
            for i in range(0,L,1):
                rowSums = np.sum(np.absolute(listW[i]),axis=1)
                M.append(maxEntry*np.amax(rowSums))
                maxEntry = maxEntry*np.amax(rowSums)
        
        # for each confidence level

        save_dir = analysisdir+str(alpha)+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if k in zero_sample_clusters:
            R = -1
            np.savetxt(save_dir+"R_"+str(k)+".txt",[R],delimiter=",")
            continue

        R_all, R,sigmas,lb,ub = RO.getRadiiDataPoints(L,ck,cov,X_train_main[train_labels==k,:],listW, alpha)
        # save R
        
        
        # save R to a txt file
        np.savetxt(save_dir+"R_"+str(k)+".txt",[R],delimiter=",")

        start = time.time()


        
        if task_name=="knapsack":
            x_sol_k = np.zeros((num_constraints,prices.shape[1]))
            x_obj_k = np.zeros((num_constraints))   
            for i in tqdm(range(num_constraints)):

                A, b = get_kp_Ab(prices[i,:],budgets[i])
                obj, x = RO.solveRobustSelection(N,L,dimLayers,ck, cov, R, listW, M, lb, ub,sigmas,task_name,A,b)
                x_sols[i,test_labels==k,:] = x
                objss[i,test_labels==k] = obj

                x_sol_k[i,:] = x
                x_obj_k[i] = obj
            # save x_sol_k to a npy file
            np.save(save_dir+"x_sol_"+str(k)+".npy",x_sol_k)
            # save x_obj_k to a npy file
            np.save(save_dir+"x_obj_"+str(k)+".npy",x_obj_k)
        elif task_name == "shortest_path":
            A, b = get_spp_Ab()
            obj, x = RO.solveRobustSelection(N,L,dimLayers,ck, cov, R, listW, M, lb, ub,sigmas,task_name,A,b)
            x_sols[test_labels==k,:] = x
            objss[test_labels==k] = obj 

            end = time.time()
            t = end-start
            

# check if save_dir exists, if not, create it

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if task_name=="knapsack":
    if not os.path.exists(save_dir+"x_sol.npy"):
        np.save(save_dir+"x_sol.npy",x_sols)
        np.save(save_dir+"objs.npy",objss)
elif task_name == "shortest_path":
    # save in  csv format
    if not os.path.exists(save_dir+"x_sol.csv"):
        # save x_sols to a csv file
        np.savetxt(save_dir+"x_sol.csv",x_sols,delimiter=",")
        # save objss to a csv file
        np.savetxt(save_dir+"objs.csv",objss,delimiter=",")

if task_name!="toy":
    if not os.path.exists(save_dir+"VaR.csv") or not os.path.exists(save_dir+"obj_pos.csv") or not os.path.exists(save_dir+"coverage.csv"):
        test_dir = test_path
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
        coverage = in_DNN_ellipsoid(save_dir,n_cluster,c_test)
        coverage = np.mean(coverage,axis=1)
        # save the coverage
        np.savetxt(save_dir+"coverage.csv",coverage,delimiter=",")

print(net_name+", done!")

