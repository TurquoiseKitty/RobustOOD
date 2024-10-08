'''
Generate shortest path data with two degree parameters and different sample sizes
Note: in this dataset, we use deg to indicate the index of dataset, not the degree of the polynomial
'''

import numpy as np
import pandas as pd
import os
import ipdb


np.random.seed(0)


def generate_covs(num_train_samples):
    # Note: if this function is changed, the variable cov_test in function generate_muldim_covs() should also be changed
    covs = np.random.uniform(low = -0.5, high = 0.5, size=(num_train_samples,1))
    return covs

def generate_random_res(num_train_samples):
    res = np.random.uniform(low = -0.5, high = 0.5, size=(num_train_samples,1))
    return res

def generate_c(covs,eps,const):
    abs_value = np.sqrt(np.abs(covs[:,0]))
    c = np.sign(covs[:,0])*abs_value + (abs_value)*eps[:,0]

    return c

def get_true_LB_UB(covs,alpha):
    abs_value = np.sqrt(np.abs(covs[:,0]))
    LB = np.sign(covs[:,0])*abs_value - abs_value*(alpha-0.5)
    UB = np.sign(covs[:,0])*abs_value + abs_value*(alpha-0.5)
    return LB,UB

################# generate 1 dim covariate #################
def generate_one_dim_samples(deg,num_train_samples,const,num_test_samples,random_seed=0):
    np.random.seed(random_seed)
    set_idx = 1
    train_dir = str(set_idx)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
    test_dir = str(set_idx)+"/"+str(deg)+"/test/"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # generate covariates
    covs = generate_covs(num_train_samples)

    eps = generate_random_res(num_train_samples)

    c = generate_c(covs,eps,const)

    # save covs to train_dir
    covs_path = train_dir+"covs.csv"
    pd.DataFrame(covs).to_csv(covs_path,header=False,index=False)

    # save c to train_dir
    c_path = train_dir+"c.csv"
    pd.DataFrame(c).to_csv(c_path,header=False,index=False)

    # generate test data
    cov_test = np.linspace(-0.5,0.5,num_test_samples).reshape(-1,1)
    
    cov_test_path = test_dir+"covs.csv"

    pd.DataFrame(cov_test).to_csv(cov_test_path,header=False,index=False)


    # c_test has dim (101,1,1000), where the first dim is the number of test samples, the second dim is the number of covariates, the third dim is the split number on y-axis
    lb = np.min(c)
    ub = np.max(c)
    interval = ub-lb
    lb = lb-0.05*interval
    ub = ub+0.05*interval
    c_test = np.linspace(lb,ub,1000).reshape(1,1,-1)
    # repeat the c_test to (num_test_samples,1,1000)
    c_test = np.repeat(c_test,num_test_samples,axis=0)
    # save c_test to test_dir
    c_test_path = test_dir+"c.npy"
    np.save(c_test_path,c_test)

#################### generate multiple-dims covariates, where the second dim is inrelavant ####################
def generate_muldim_covs(deg,num_train_samples,num_dim):

    set_idx = num_dim

    train_dir = str(1)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
    test_dir = str(1)+"/"+str(deg)+"/test/"

    train_dir2 = str(num_dim)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
    test_dir2 = str(num_dim)+"/"+str(deg)+"/test/"

    if not os.path.exists(train_dir2):
        os.makedirs(train_dir2)

    if not os.path.exists(test_dir2):
        os.makedirs(test_dir2)

    # load the covs, true_f, hf, c from the train_dir of 1-dim covariates
    covs = pd.read_csv(train_dir+"covs.csv",header=None).values
    c = pd.read_csv(train_dir+"c.csv",header=None).values

    # generate the second dim
    for dim_idx in range(1,num_dim):
        covs_2 = generate_covs(num_train_samples)
        covs = np.concatenate((covs,covs_2),axis=1)



    # save to csv without headers and index
    pd.DataFrame(covs).to_csv(train_dir2+"covs.csv",header=False,index=False)
    pd.DataFrame(c).to_csv(train_dir2+"c.csv",header=False,index=False)

    # copy the test data from 1-dim covariates
    cov_test = pd.read_csv(test_dir+"covs.csv",header=None).values

    # generate the second dim
    for dim_idx in range(1,num_dim):
        cov_test_2 = 0.5* np.ones((cov_test.shape[0],1))
        cov_test = np.concatenate((cov_test,cov_test_2),axis=1)


    # save to csv without headers and index
    pd.DataFrame(cov_test).to_csv(test_dir2+"covs.csv",header=False,index=False)

    # load c_test from 1-dim covariates
    c_test = np.load(test_dir+"c.npy")

    # save to test_dir2
    np.save(test_dir2+"c.npy",c_test)

if __name__=="__main__":
    num_train_samples_list = [100,200,500,1000]
    deg = 1
    dim_covs_list = [16]
    for num_train_samples in num_train_samples_list:
        if 1 in dim_covs_list:
            generate_one_dim_samples(deg,num_train_samples=num_train_samples,const=0.7,num_test_samples=1000)
        for num_dim in dim_covs_list:
            generate_muldim_covs(deg,num_train_samples=num_train_samples,num_dim=num_dim)