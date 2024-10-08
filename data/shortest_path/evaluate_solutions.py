
import os
from solve_true_spp import solve_true_spp,get_VaR_of_x,in_box,in_ellipsoid
import pandas as pd
import numpy as np

cur_dir = os.getcwd()
alpha = 0.8
dim_covs = 5
deg = 2

dataset_dir = cur_dir+"\\"+str(dim_covs)+"\\"+str(deg)+"\\"

num_train_samples = 5000


train_result_dir = dataset_dir+"train\\"+str(num_train_samples)+"\\"

test_dir = dataset_dir+"test\\"
# load true_f.csv and half_width.csv
true_f = pd.read_csv(test_dir+"true_f.csv",header=None).to_numpy()
half_width = pd.read_csv(test_dir+"half_width.csv",header=None).to_numpy()
c_test = pd.read_csv(test_dir+"c.csv",header=None).to_numpy()

test_dir = dataset_dir+"test\\"+str(alpha)+"\\"

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# check is there are a opt_sol.csv and a opt_obj.csv under test_dir
if not os.path.exists(test_dir+"opt_sol.csv") or not os.path.exists(test_dir+"opt_obj.csv"):
    
    # solve the true shortest path problem
    opt_sol, opt_obj = solve_true_spp(true_f,half_width,alpha)
    # save the solution and objective
    pd.DataFrame(opt_sol).to_csv(test_dir+"opt_sol.csv",header=None,index=None)
    pd.DataFrame(opt_obj).to_csv(test_dir+"opt_obj.csv",header=None,index=None)
else:
    # load the solution and objective
    opt_sol = pd.read_csv(test_dir+"opt_sol.csv",header=None).to_numpy()
    opt_obj = pd.read_csv(test_dir+"opt_obj.csv",header=None).to_numpy()


#search all the x_sol.csv files under train_result_dir where the last dir should be named as str(alpha), and get their corresponding paths
x_sol_paths = []
alg_names_tuple = []
alg_names_lists = []
for root, dirs, files in os.walk(train_result_dir):
    for file in files:
        if file == 'x_sol.csv':
            if root.split("\\")[-1] == str(alpha) or root.split("\\")[-2] == "kNN":
                x_sol_paths.append(root+"\\")
                # split the path's last 4, last 3, and lst 2 dirs, and combine them to a tuple
                dirs_list = root.split("\\")[-4:-1]
                dirs_tuple = tuple(dirs_list)
                alg_names_tuple.append(dirs_tuple)
                alg_name = dirs_list[0]+"-"+dirs_list[1]+"-"+dirs_list[2]
                alg_names_lists.append(alg_name)


# two metrics to be compared

result = pd.DataFrame(columns=["alg_name","mean_relative_gap","var_relative_gap","in_set_rate"])
for idx,dir in enumerate(x_sol_paths):

    # load x_sol.csv from this fir
    x_sol = pd.read_csv(dir+"x_sol.csv",header=None).to_numpy()
    x_VaRs = get_VaR_of_x(true_f,half_width,x_sol,alpha)
    #reshape the x_VaRs to opt_obj's shape
    x_VaRs = x_VaRs.reshape(opt_obj.shape)
    # calculate relative gaps
    rg = (x_VaRs-opt_obj)/opt_obj

    
    # calculate if the real c_test is in the uncertainty set
    if alg_names_tuple[idx][1]=="quantile":
        # load test_LB and test_UB
        test_LB = pd.read_csv(dir+"test_LB.csv",header=None).to_numpy()
        test_UB = pd.read_csv(dir+"test_UB.csv",header=None).to_numpy()
        freq = in_box(test_LB,test_UB,c_test)
        
    elif alg_names_tuple[idx][1]=="norm":
        # load cov.txt and res_test_2norm_pred.csv from dir's precedent dir to numpy array
        cov= np.loadtxt(dir+"..\\cov.txt")
        # split twice because the dir is end with "\\"
        parent_dir,_ = os.path.split(dir)
        parent_dir,_ = os.path.split(parent_dir)

        res_test_2norm_pred = pd.read_csv(parent_dir+"\\res_test_2norm_pred.csv",header=None).to_numpy()
        #res_test_2norm_pred = pd.read_csv(dir+"../res_test_2norm_pred.csv",header=None).to_numpy()
        # read r from dir's r.txt
        r = float(open(dir+"r.txt").read())
        grandparent_dir,_ = os.path.split(parent_dir)
        grandparent_dir,_ = os.path.split(grandparent_dir)
        # read c_test_pred from dir's last two precedent dirs
        c_test_pred = pd.read_csv(grandparent_dir+"\\c_test_pred.csv",header=None).to_numpy()
        
        freq = in_ellipsoid(c_test,c_test_pred,res_test_2norm_pred,cov,r)

    # save the freq to result's idx row and the third column
    result.loc[idx] = [alg_names_lists[idx],np.mean(rg),np.var(rg),freq]
        
# save the result dataframe to a csv file
result.to_csv(test_dir+"result.csv")