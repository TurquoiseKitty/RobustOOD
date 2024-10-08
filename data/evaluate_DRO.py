
import numpy as np
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
import ipdb
import argparse

sys.path.append(os.getcwd())
sys.path.append('..')
from DRO.solver import solve_f_LP,get_spp_Ab,get_kp_Ab

task_name = "knapsack"

deg = 4
dim_covs = 5

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--dim_covs', type=int, default=dim_covs)
args = parser.parse_args()

task_name = args.task_name
deg = args.deg
dim_covs = args.dim_covs

# create a dataframe where column indexed by num_train_samples, algorithm_name, gap
df = pd.DataFrame(columns=['num_train_samples', 'algorithm_name', 'gap'])


x_sol_list = []
alg_name_list = []
quantile_5_list = []
quantile_25_list = []
quantile_50_list = []
quantile_75_list = []
quantile_95_list = []

ns = [500]
for num_train_samples in ns:
    train_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//train//"+str(num_train_samples)+"//"
    test_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//test//"

    ############################# solve optimization problem of true model #############################
    # load true_f.csv from test dir
    true_f = pd.read_csv(test_dir+"true_f.csv",header=None).to_numpy()

    save_dir = test_dir

    if task_name == "shortest_path":
        A, b = get_spp_Ab()
        #check if there is a csv file record the x solution
        if not os.path.exists(save_dir+"x_opt.csv"):
            # solve the shortest path problem
            x_sol,obj_opt = solve_f_LP(true_f,A,b,task_name)
            #save the solution to a csv file
            pd.DataFrame(x_sol).to_csv(save_dir+"x_opt.csv",header=False,index=False)
            pd.DataFrame(obj_opt).to_csv(save_dir+"obj_opt.csv",header=False,index=False)
        else:
            x_sol = pd.read_csv(save_dir+"x_opt.csv",header=None).to_numpy()
            obj_opt = pd.read_csv(save_dir+"obj_opt.csv",header=None).to_numpy()


    elif task_name == "knapsack":
        # load prices and budgets
        prices = pd.read_csv("knapsack/prices.csv",header=None).to_numpy()
        budgets = pd.read_csv("knapsack/budgets.csv",header=None).to_numpy()
        #check if there is a npy file record the x solution
        
        
        num_constraints = budgets.shape[0]
        num_tests = true_f.shape[0]

        x_opt = np.zeros((num_constraints,num_tests,prices.shape[1]))
        obj_opt = np.zeros((num_constraints,num_tests))
        if not os.path.exists(save_dir+"x_sol.npy"):
            for cons_idx in range(num_constraints):
                A,b = get_kp_Ab(prices[cons_idx,:],budgets[cons_idx,:])
                # solve the knapsack problem
                x,obj = solve_f_LP(true_f,A,b,task_name)
                # save solution and obj to a npy file
                x_opt[cons_idx,:,:] = x
                obj_opt[cons_idx,:] = obj

            np.save(save_dir+"x_sol.npy",x_opt)
            np.save(save_dir+"obj_opt.npy",obj_opt)
        else:
            x_opt = np.load(save_dir+"x_sol.npy")
            obj_opt = np.load(save_dir+"obj_opt.npy")

    #%%
    ############################# load the solution from different algorithms #############################
    suffix = ""
    if task_name == "shortest_path":
        suffix = "x_sol.csv"
    elif task_name == "knapsack":
        suffix = "x_sol.npy"

    # only_f use OLS solution
    filename = train_dir+"only_f//OLS//"+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        alg_name_list.append("f-OLS")
    """
    # HetRes use OLS+example1-2 solution
    prefix = train_dir+"HetRes//OLS//example1-2//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("Heter-OLS-exp-"+str(best_eps))
    
    # HetRes use random_forest+example1-2 solution
    prefix = train_dir+"HetRes//random_forest//example1-2//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("Heter-OLS-exp-"+str(best_eps))
    """
    # HomoRes use OLS solution
    prefix = train_dir+"HetRes//OLS//homo//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("Homo-OLS-"+str(best_eps))
    
    
    # HomoRes use rf solution
    prefix = train_dir+"HetRes//random_forest//homo//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("Homo-rf-"+str(best_eps))
    

    # PTC_DRO use OSL+gaussian with param 1 solution
    prefix = train_dir+"PTC_DRO//OLS//gaussian_1.0//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("PTC_DRO-OLS-gaussian1-"+str(best_eps))

    # PTC_DRO use random_forest+gaussian with param 1 solution
    prefix = train_dir+"PTC_DRO//random_forest//gaussian_1.0//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("PTC_DRO-rf-gaussian1-"+str(best_eps))

    # local DRO use gaussian with param 1 solution
    prefix = train_dir+"localDRO//gaussian_1.0//"
    filename = prefix+suffix
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).to_numpy()
        else:
            x_sol = np.load(filename)
        x_sol_list.append(x_sol)
        best_eps = pd.read_csv(prefix+"best_eps.csv",header=None).to_numpy()[0,0]
        alg_name_list.append("NW-DRO-gaussian1-"+str(best_eps))


    #%%
    ########################### for each solution, calculate the optimality gaps ##############################
    

    for alg_idx,x_sol in enumerate(x_sol_list):
        if task_name == "shortest_path":
            obj = np.sum(x_sol*true_f,axis=1)
            gap  = abs(obj-obj_opt)/abs(obj_opt)
            # add to dataframe
            for sample_idx in range(len(gap)):
                # concatenate to the dataframe
                df = pd.concat([df,pd.DataFrame({'num_train_samples':num_train_samples,'algorithm_name':alg_name_list[alg_idx],'gap':gap[sample_idx]})],ignore_index=True)
            
        elif task_name == "knapsack":
            obj = np.zeros((num_constraints,num_tests))
            for cons_ind in range(num_constraints):
                obj[cons_ind,:] = np.sum(x_sol[cons_ind,:,:]*true_f,axis=1)
            gap = abs(obj-obj_opt)/abs(obj_opt)

            # reshape gap to 1 dim
            gap = gap.reshape((-1,1))
            # add to dataframe
            for sample_idx in range(len(gap)):
                # concatenate to the dataframe
                df = pd.concat([df,pd.DataFrame({'num_train_samples':num_train_samples,'algorithm_name':alg_name_list[alg_idx],'gap':gap[sample_idx]})],ignore_index=True)

    # save the dataframe
    df.to_csv(save_dir+"gap.csv",index=False)

#%%
################################# draw box plot of these data #################################
import seaborn as sns

ax = sns.boxplot(x="num_train_samples", y="gap", hue="algorithm_name", data=df,palette="Set3",showfliers=False)

# plot the result
plt.show()
# save the result
fig = ax.get_figure()
fig.savefig(save_dir+"boxplot.png")


