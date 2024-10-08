import numpy as np
import gurobipy
from gurobipy import GRB
from scipy.stats import norm
import os
import sys
import pandas as pd
import ipdb


def in_box(dir,c_test):
    test_LB = pd.read_csv(dir+"test_LB.csv",header=None).to_numpy()
    test_UB = pd.read_csv(dir+"test_UB.csv",header=None).to_numpy()
    # judge if c_test satisfies test_LB<=c_test<=test_UB
    num_covs = test_LB.shape[0]
    num_test_c = c_test.shape[-1]
    results = np.zeros((num_covs,num_test_c))
    for test_idx in range(num_test_c):
        result = np.zeros(num_covs)
        for t in range(num_covs):
            if (c_test[t,:,test_idx]>=test_LB[t,:]).all() and (c_test[t,:,test_idx]<=test_UB[t,:]).all():
                result[t] = 1
        results[:,test_idx] = result
    return results

def in_ellipsoid(dir,true_y):
    # judge if true_y lies in the ellisoid of shope cov and radius r
    # load cov.txt and res_test_2norm_pred.csv from dir's precedent dir to numpy array
    cov= np.loadtxt(dir+"cov.txt")

    # if cov is a scalar, then covert it to a 1*1 matrix
    cov = np.array(cov)
    if cov.shape==():
        cov = np.array([[cov]])

    # split twice because the dir is end with "/"
    parent_dir,_ = os.path.split(dir)
    parent_dir,_ = os.path.split(parent_dir)

    #pred_norm = pd.read_csv(parent_dir+"/res_test_2norm_pred.csv",header=None).to_numpy()
    pred_norm = pd.read_csv(dir+"res_test_2norm_pred.csv",header=None).to_numpy()
    #res_test_2norm_pred = pd.read_csv(dir+"../res_test_2norm_pred.csv",header=None).to_numpy()
    # read r from dir's r.txt
    r = float(open(dir+"r.txt").read())
    grandparent_dir,_ = os.path.split(parent_dir)
    grandparent_dir,_ = os.path.split(grandparent_dir)
    # read c_test_pred from dir's last two precedent dirs
    pred_y = pd.read_csv(grandparent_dir+"/c_test_pred.csv",header=None).to_numpy()



    cov_inv = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov_inv)

    num_covs = true_y.shape[0]
    num_test_c = true_y.shape[-1]

    results = np.zeros((num_covs,num_test_c))

    for test_idx in range(num_test_c):
        reg = np.matmul((true_y[:,:,test_idx]-pred_y)/pred_norm,L)
        distances = np.sqrt(np.sum(reg**2,axis=1))
        result = distances<=r
        results[:,test_idx] = result
    
    return results


def in_kNN_ellipsoid(dir,true_y):
    covss = np.load(dir+"covss.npy")
    mus = np.load(dir+"mus.npy")
    Rs = np.load(dir+"Rs.npy")

    num_covs = true_y.shape[0]
    num_test_c = true_y.shape[-1]
    if len(mus.shape)==1:
        mus = np.tile(mus,(num_covs,1))
        covss = np.tile(covss,(num_covs,1,1))
        Rs = np.tile(Rs,(num_covs,1))
    
    results = np.zeros((num_covs,num_test_c))
    for test_idx in range(num_test_c):
        # judge if true_y lies in the ellipsoid of kNN
        result = np.zeros(mus.shape[0])
        for i in range(mus.shape[0]):
            cov_inv = np.linalg.inv(covss[i,:,:])
            L = np.linalg.cholesky(cov_inv)


            reg = np.matmul((true_y[i,:,test_idx]-mus[i])/Rs[i],L)
            distances = np.sqrt(np.sum(reg**2))

            result[i] = distances<=1
        results[:,test_idx] = result
    
    return results



def in_DNN_ellipsoid(dir,n_cluster,X):
    # load R from dir
    R_list = []
    for k in range(n_cluster):
        R = np.loadtxt(dir+"R_"+str(k)+".txt")
        R_list.append(R)
    parent_dir,_ = os.path.split(dir)
    parent_dir,_ = os.path.split(parent_dir)
    W_list = []
    c_list = []
    cov_list = []
    L=3
    for k in range(n_cluster):
        # load c and cov from dir's precedent dir, delimit by ","
        c = np.loadtxt(parent_dir+"/c_"+str(k)+".txt",delimiter=",")
        cov = np.loadtxt(parent_dir+"/cov_"+str(k)+".txt",delimiter=",")
        
        c_list.append(c)
        cov_list.append(cov)
        W = []
        for Li in range(L):
            W_temp = np.loadtxt(parent_dir+"/W_"+str(k)+"_"+str(Li)+".txt",delimiter=",")
            if len(W_temp.shape)==1:
                W_temp = W_temp.reshape((W_temp.shape[0],1))
            W.append(W_temp)
        W_list.append(W)
    # load test_assignments.npy from dir
    test_assignment = np.load(parent_dir+"/test_assignments.npy")

    (num_covs,dim_c,num_test_c)=X.shape

    
    LP_list = []
    for i in range(n_cluster):
        cov = cov_list[i]
        # check if cov is positive definite, if not, add a small number to the diagonal
        if not np.all(np.linalg.eigvals(cov) > 0):
            # get the smallest none-zero eigenvalue
            min_eig = np.min(np.linalg.eigvals(cov))
            # add a small number to the diagonal
            if min_eig<0:
                cov = cov + abs(min_eig)*np.eye(cov.shape[0])
            
            cov = cov + 0.01*abs(min_eig)*np.eye(cov.shape[0])


        cov_inv = np.linalg.inv(cov)
        # decompose cov_inv
        cov_inv_L = np.linalg.cholesky(cov_inv)
        LP_list.append(cov_inv_L.transpose())

    results = np.zeros((num_covs,num_test_c))
    for j in range(0,num_covs,1):

        label = np.argmax(test_assignment[j,:])
        listW = W_list[label]
        result = np.zeros(num_test_c)
        for test_idx in range(num_test_c):
            outLayer = X[j,:,test_idx]
            
            for i in range(0,len(listW)-1,1):
                outLayer = np.maximum(np.zeros(listW[i].shape[0]),np.dot(listW[i],outLayer))
                # print(i,outLayer)

            outLayer = np.dot(listW[len(listW)-1],outLayer)
            # print(X[i],outLayer,np.linalg.norm(outLayer-c0))

            result[test_idx] = np.linalg.norm(LP_list[label]@(outLayer-c_list[label]))<=R_list[label]
        results[j,:] = result
    
    return results