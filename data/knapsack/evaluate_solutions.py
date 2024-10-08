
import os
from solve_true_kp import solve_true_kp,get_VaR_of_x,in_box,in_ellipsoid,in_kNN_ellipsoid,in_DNN_ellipsoid
import pandas as pd
import numpy as np
import ipdb

cur_dir = os.getcwd()

#load prices and budgets
prices = pd.read_csv(cur_dir+"\\prices.csv",header=None).to_numpy()
budgets = pd.read_csv(cur_dir+"\\budgets.csv",header=None).to_numpy()
alpha = 0.8
dataset_dir = cur_dir+"\\01\\1\\"

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
if not os.path.exists(test_dir+"opt_sol.npy") or not os.path.exists(test_dir+"opt_obj.npy"):
    
    # solve the true shortest path problem
    opt_sol, opt_obj = solve_true_kp(true_f,half_width,alpha,prices,budgets)
    # save to npy files
    np.save(test_dir+"opt_sol.npy",opt_sol)
    np.save(test_dir+"opt_obj.npy",opt_obj)
    
else:
    # load the solution and objective from npy files
    opt_sol = np.load(test_dir+"opt_sol.npy")
    opt_obj = np.load(test_dir+"opt_obj.npy")


#search all the x_sol.npy files under train_result_dir where the last dir should be named as str(alpha), and get their corresponding paths
x_sol_paths = []
alg_names_tuple = []
alg_names_lists = []
for root, dirs, files in os.walk(train_result_dir):
    for file in files:
        #print(file)
        if file == 'x_sol.npy':
            if root.split("\\")[-1] == str(alpha) or root.split("\\")[1] == "kNN":
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

    # load x_sol.npy from this dir
    x_sol = np.load(dir+"x_sol.npy")
    x_VaRs = get_VaR_of_x(true_f,half_width,x_sol,alpha)
    #ipdb.set_trace()
    #reshape the x_VaRs to opt_obj's shape
    x_VaRs = x_VaRs.reshape(opt_obj.shape)
    # calculate relative gaps
    rg = (x_VaRs-opt_obj)/opt_obj

    
    # calculate if the real c_test is in the uncertainty set
    if alg_names_tuple[idx][1]=="quantile":
        # load test_LB and test_UB
        test_LB = pd.read_csv(dir+"test_LB.csv",header=None).to_numpy()
        test_UB = pd.read_csv(dir+"test_UB.csv",header=None).to_numpy()
        if np.mean(rg)<-0.1:
            print("LB \n",test_LB[0,:],"\n UB\n",test_UB[0,:],"\n true_f\n",true_f[0,:],
              "\n sigma \n",half_width[0,:],"\n RO_sol\n",x_sol[0,0,:],"\n opt_sol\n",opt_sol[0,0,:],"\n prices\n",prices[0,:],"\n budgets\n",budgets[0],"\n RO_VaR\n",x_VaRs[0,0],"\n opt_VaR\n",opt_obj[0,0])
            ipdb.set_trace()
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
    elif alg_names_tuple[idx][1]=="kNN":
        # load covss.npy, mus.npy and Rs.npy from dir's precedent dir to numpy array
        # split twice because the dir is end with "\\"
        parent_dir,_ = os.path.split(dir)
        parent_dir,_ = os.path.split(parent_dir)
        covss = np.load(parent_dir+"\\covss.npy")
        mus = np.load(parent_dir+"\\mus.npy")
        Rs = np.load(parent_dir+"\\Rs.npy")

        freq = in_kNN_ellipsoid(c_test,mus,covss,Rs)
    elif alg_names_tuple[idx][-2]=="IDCC" or alg_names_tuple[idx][-2]=="DCC":
        n_cluster = int(alg_names_tuple[idx][-1])
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
            c = np.loadtxt(parent_dir+"\\c_"+str(k)+".txt",delimiter=",")
            cov = np.loadtxt(parent_dir+"\\cov_"+str(k)+".txt",delimiter=",")
            
            c_list.append(c)
            cov_list.append(cov)
            W = []
            for Li in range(L):
                W.append(np.loadtxt(parent_dir+"\\W_"+str(k)+"_"+str(Li)+".txt",delimiter=","))
            W_list.append(W)
        # load test_assignments.npy from dir
        test_assignment = np.load(parent_dir+"\\test_assignments.npy")
        

        freq = in_DNN_ellipsoid(n_cluster,W_list,test_assignment,R_list,c_list,cov_list,c_test)

    # save the freq to result's idx row and the third column
    result.loc[idx] = [alg_names_lists[idx],np.mean(rg),np.var(rg),freq]
        
# save the result dataframe to a csv file
result.to_csv(test_dir+"result.csv")