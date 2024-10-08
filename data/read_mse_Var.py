import numpy as np
import pandas as pd

def get_Var(x_sol,c_test,alpha,task_name):
    if task_name == "knapsack":
        (num_constraints,num_test_covs,dim_x) = x_sol.shape
        Qs = np.zeros((num_constraints,num_test_covs))
        for cons_idx in range(num_constraints):
            for cov_idx in range(num_test_covs):
                objs = x_sol[cons_idx,cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                Qs[cons_idx,cov_idx] = np.quantile(objs,1-alpha)

        # reshape Qs to one dimension
        Qs = Qs.reshape(-1)


    elif task_name == "shortest_path":
        (num_test_covs,dim_x) = x_sol.shape
        Qs = np.zeros((num_test_covs))
        for cov_idx in range(num_test_covs):
            objs = x_sol[cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
            Qs[cov_idx] = np.quantile(objs,alpha)
            
    return Qs



def read_mse_Var(train_dir,test_dir,alpha,f_model_name_list,h_model_name_box,task_name):
    mse_list = []
    Var_list = []
    
    # read c_test from c.npy in test_dir
    c_test = np.load(test_dir+"/c.npy")

    for f_model_name in f_model_name_list:
        # read mse from mse.txt in train_dir+f_model_name
        mse_file = train_dir+"/"+f_model_name+"/mse_test.txt"    
        mse = np.loadtxt(mse_file)

        mse_list.append(mse)

        # load x_sol from train_dir/f_model_name/quantile/h_model_name_box/x_sol.csv or x_sol.npy
        if task_name=="shortest_path":
            # load csv
            x_sol = pd.read_csv(train_dir+"/"+f_model_name+"/quantile/"+h_model_name_box+"/"+str(alpha)+"/x_sol.csv",header=None).to_numpy()
        elif task_name=="knapsack":
            # load npy
            x_sol = np.load(train_dir+"/"+f_model_name+"/quantile/"+h_model_name_box+"/"+str(alpha)+"/x_sol.npy")

        VaR = get_Var(x_sol,c_test,alpha,task_name)
        Var_list.append(VaR)

    return mse_list,Var_list