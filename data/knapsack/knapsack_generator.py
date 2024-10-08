'''Generate shortest path data with two degree parameters and different sample sizes'''

import numpy as np
import pandas as pd
import os
import shutil

def generate_B(dim_c,dim_cov,noise_dim_proportion,dir_name, random_seed=0):
    np.random.seed(random_seed)

    # generate B as SPO

    B = np.random.binomial(1,0.5,size=(dim_c,dim_cov))
    
    dim_noise = int(dim_cov*noise_dim_proportion)
    B[:,dim_cov-dim_noise:] = 0

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    file_path = dir_name+"/B.csv"
    df = pd.DataFrame(B)
    # save to csv without headers and index
    df.to_csv(file_path,header=False,index=False)

def generate_kp_data(deg,num_samples,eps,B_dir,train_or_test,random_seed):
    np.random.seed(random_seed)
    B_path = B_dir+"/B.csv"
    # save to csv without the first column and the first row
    B = pd.read_csv(B_path,header=None)
    # convert to numpy array without the first column and the first row
    B = B.to_numpy()
    
    dim_cov = B.shape[1]
    
    # generate covariates
    covs = np.random.uniform(low = 0, high = 4, size=(num_samples,dim_cov))

    true_f = np.power(np.matmul(covs,B.transpose()),deg)

    res_half_width = eps*true_f
    
    
    save_dir = B_dir+"/"+str(deg)+"/"
    
    if train_or_test=="train":
        save_dir+="train/"+str(num_samples)+"/"
    else:
        save_dir+="test/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    covs_path=save_dir+"covs.csv"
    true_f_path = save_dir+"true_f.csv"
    hf_path = save_dir+"half_width.csv"
    

    pd.DataFrame(covs).to_csv(covs_path,header=False,index=False)
    pd.DataFrame(true_f).to_csv(true_f_path,header=False,index=False)
    pd.DataFrame(res_half_width).to_csv(hf_path,header=False,index=False)

    if train_or_test=="train":
        unif_epss = np.random.uniform(low = 1-eps, high = 1+eps, size=true_f.shape)
        samples_c = true_f*unif_epss
        c_path = save_dir+"c.csv"
        pd.DataFrame(samples_c).to_csv(c_path,header=False,index=False)
    else:
        # only use to evaluate RO algorithms
        num_test_group = 1000
        unif_epss_group = np.random.uniform(low = 1-eps, high = 1+eps, size=(true_f.shape[0],true_f.shape[1],num_test_group))
        samples_c = np.zeros((true_f.shape[0],true_f.shape[1],num_test_group))
        for group_idx in range(num_test_group):
            samples_c[:,:,group_idx] = true_f * unif_epss_group[:,:,group_idx]
        c_path = save_dir+"c.npy"
        np.save(c_path,samples_c)
        

if __name__=="__main__":
    eps = 0.2
    dim_covs = 10
    dim_c = 20


    dir_name = str(dim_covs)

    generate_B(dim_c=dim_c,dim_cov=dim_covs,noise_dim_proportion=0.2,dir_name=dir_name,random_seed = 0)

    for (deg) in [2]:
        for n in [200,500,1000,5000]:
            generate_kp_data(deg,n,eps,dir_name,"train",random_seed = n)
            
        generate_kp_data(deg,100,eps,dir_name,"test",random_seed = deg)