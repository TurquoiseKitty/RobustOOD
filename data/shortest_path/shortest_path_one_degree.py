'''Generate shortest path data with two degree parameters and different sample sizes'''

import numpy as np
import pandas as pd
import os
import itertools

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

    
    return B

def generate_B_fast(dim_c,dim_cov,noise_dim_proportion,dir_name, random_seed=0):
    np.random.seed(random_seed)

    # generate B as SPO
    """
    B = np.random.binomial(1,0.5,size=(dim_c,dim_cov))
    
    dim_noise = int(dim_cov*noise_dim_proportion)
    B[:,dim_cov-dim_noise:] = 0
    """

    # generate B as <<fast convergence rate>>
    eff_dim = int(dim_cov*(1-noise_dim_proportion))
    # generate uniform B
    B = np.random.uniform(low = 0, high = 1, size=(dim_c,2**eff_dim-1))
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    file_path = dir_name+"/B.csv"
    df = pd.DataFrame(B)
    # save to csv without headers and index
    df.to_csv(file_path,header=False,index=False)

    
    return B

def generate_sp_data(deg,num_samples,eps,B_dir,train_or_test,random_seed=0):
    np.random.seed(random_seed)
    B_path = B_dir+"/B.csv"

    # save to csv without the first column and the first row
    B = pd.read_csv(B_path,header=None)
    # convert to numpy array without the first column and the first row
    B = B.to_numpy()
    
    dim_cov = B.shape[1]
    
    covs = np.random.multivariate_normal(np.zeros(dim_cov), np.diag(np.ones(dim_cov)),size=num_samples)
    true_f = np.power(1/np.sqrt(dim_cov)*np.matmul(covs,B.transpose())+3,deg)+1
    #res_half_width = eps*(1/np.abs(true_f))
    res_half_width = eps*true_f
    
    
    save_dir = B_dir+"/"+str(deg)+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    if train_or_test=="train":
        save_dir+="train/"+str(num_samples)+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        c_path = save_dir+"c.csv"
        #samples_c = true_f+res_half_width*np.random.normal(size=true_f.shape)
        unif_epss = np.random.uniform(low = 1-eps, high = 1+eps, size=true_f.shape)
        samples_c = true_f*unif_epss
        pd.DataFrame(samples_c).to_csv(c_path,header=False,index=False)
    else:
        save_dir+="test/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # only use to evaluate RO algorithms
        num_test_group = 1000
        unif_epss_group = np.random.uniform(low = 1-eps, high = 1+eps, size=(true_f.shape[0],true_f.shape[1],num_test_group))
        samples_c = np.zeros((true_f.shape[0],true_f.shape[1],num_test_group))
        for group_idx in range(num_test_group):
            samples_c[:,:,group_idx] = true_f * unif_epss_group[:,:,group_idx]
        c_path = save_dir+"c.npy"
        np.save(c_path,samples_c)

    covs_path=save_dir+"covs.csv"
    true_f_path = save_dir+"true_f.csv"
    hf_path = save_dir+"half_width.csv"
    pd.DataFrame(covs).to_csv(covs_path,header=False,index=False)
    pd.DataFrame(true_f).to_csv(true_f_path,header=False,index=False)
    pd.DataFrame(res_half_width).to_csv(hf_path,header=False,index=False)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def get_covariate_products(covariates):
    n, p = covariates.shape
    num_combinations = sum([len(list(itertools.combinations(range(p), r))) for r in range(1, p + 1)])

    covariate_products = np.zeros((n, num_combinations))
    col_index = 0

    # Generate all combinations of distinct covariates
    for r in range(1, p + 1):
        combinations = itertools.combinations(range(p), r)

        # Calculate the product of each combination for each sample
        for combo in combinations:
            product = np.prod(covariates[:, combo], axis=1)
            covariate_products[:, col_index] = product
            col_index += 1

    return covariate_products

def generate_sp_data_fast(deg,dim_covs,noise_dim_proportion,num_samples,eps,B_dir,train_or_test,random_seed=0):
    np.random.seed(random_seed)
    B_path = B_dir+"/B.csv"

    # save to csv without the first column and the first row
    B = pd.read_csv(B_path,header=None)
    # convert to numpy array without the first column and the first row
    B = B.to_numpy()
    
    eff_dim = int(dim_covs*(1-noise_dim_proportion))

    covs = np.random.multivariate_normal(np.zeros(eff_dim), np.diag(np.ones(eff_dim)),size=num_samples)
    
    covs_product = get_covariate_products(covs)

    true_f = np.matmul(covs_product,B.transpose())+3
    
    res_half_width = eps*np.abs(true_f)

    
    save_dir = B_dir+"/"+str(deg)+"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if train_or_test=="train":
        save_dir+="train/"+str(num_samples)+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        c_path = save_dir+"c.csv"
        #samples_c = true_f+res_half_width*np.random.normal(size=true_f.shape)
        unif_epss = np.random.uniform(low = 1-eps, high = 1+eps, size=true_f.shape)
        samples_c = true_f*unif_epss
        pd.DataFrame(samples_c).to_csv(c_path,header=False,index=False)
    else:
        save_dir+="test/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # only use to evaluate RO algorithms
        num_test_group = 1000
        unif_epss_group = np.random.uniform(low = 1-eps, high = 1+eps, size=(true_f.shape[0],true_f.shape[1],num_test_group))
        samples_c = np.zeros((true_f.shape[0],true_f.shape[1],num_test_group))
        for group_idx in range(num_test_group):
            samples_c[:,:,group_idx] = true_f * unif_epss_group[:,:,group_idx]
        c_path = save_dir+"c.npy"
        np.save(c_path,samples_c)

    covs_path=save_dir+"covs.csv"
    true_f_path = save_dir+"true_f.csv"
    hf_path = save_dir+"half_width.csv"
    pd.DataFrame(covs).to_csv(covs_path,header=False,index=False)
    pd.DataFrame(true_f).to_csv(true_f_path,header=False,index=False)
    pd.DataFrame(res_half_width).to_csv(hf_path,header=False,index=False)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    

    
    

if __name__=="__main__":
    random_seed = 0
    
    deg = 5
    eps = 0.25
    # num_sample_list = [20000]
    num_sample_list = [50,100,200,500,1000,2000,5000,10000] # investigating the impact of sample size
    #num_sample_list_4 = [200,500,1000,2000,5000] # investigating the impact of sample size at 4 dimension
    #num_sample_list_not4 = [5000]
    num_dim_list_5000 = [5,10,20,40,60,80] # investigating the impact of dimension at 5000 sample size
    num_dim_list_not5000 = [5]
    #generate B
    for dim_covs in num_dim_list_5000: 
        dir_name = str(dim_covs)
        noise_dim_proportion = 0.2
        generate_B(dim_c=40,dim_cov=dim_covs,noise_dim_proportion = noise_dim_proportion,dir_name=dir_name,random_seed = random_seed)

        """
        if dim_covs==4:
            num_sample_list = num_sample_list_4
        else:
            num_sample_list = num_sample_list_not4
        """
        
        for n in num_sample_list:
            #generate_sp_data_fast(deg,dim_covs,noise_dim_proportion,n,eps,dir_name,"train",random_seed=n)
            generate_sp_data(deg,n,eps,dir_name,"train",random_seed=n)
            
        #generate_sp_data_fast(deg,dim_covs,noise_dim_proportion,500,eps,dir_name,"test",random_seed=dim_covs)
        generate_sp_data(deg,500,eps,dir_name,"test",random_seed=dim_covs)
        