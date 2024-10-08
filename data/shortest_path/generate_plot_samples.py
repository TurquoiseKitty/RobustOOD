
import numpy as np
import pandas as pd
import os

dim_covs = 5
data_path = str(dim_covs)+"\\"                                                                              
observe_cov_dim = 1
eps = 0.1 # keep the same as in "shortest_path_one_degree.py"

# load B.csv 
B_path = data_path+"B.csv"
B = pd.read_csv(B_path,header=None).to_numpy()

dim_c = B.shape[0]
dim_cov = B.shape[1]

for deg in [1,2,4]:
    x = np.linspace(-1,1,20)
    covs = np.zeros((20,dim_cov))
    covs[:,observe_cov_dim] = x
    true_f = np.power(1/np.sqrt(dim_cov)*np.matmul(covs,B.transpose())+3,deg)+1
    res_half_width = eps*(1/np.abs(true_f))
    samples_c = true_f+res_half_width*np.random.normal(size=true_f.shape)

    save_dir = data_path+str(deg)+"\\plot\\"+str(observe_cov_dim)+"\\"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    covs_path=save_dir+"covs.csv"
    true_f_path = save_dir+"true_f.csv"
    hf_path = save_dir+"half_width.csv"
    c_path = save_dir+"c.csv"

    pd.DataFrame(covs).to_csv(covs_path,header=False,index=False)
    pd.DataFrame(true_f).to_csv(true_f_path,header=False,index=False)
    pd.DataFrame(res_half_width).to_csv(hf_path,header=False,index=False)
    pd.DataFrame(samples_c).to_csv(c_path,header=False,index=False)