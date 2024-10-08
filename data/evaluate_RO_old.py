import argparse
import numpy as np
import os
import pandas as pd
import ipdb
from coverage_solver import in_box,in_DNN_ellipsoid,in_kNN_ellipsoid,in_ellipsoid
import matplotlib.pyplot as plt

# import time to measure the time cost
import time

plot_covergae = False
plot_obj_position = True

alpha_list = [0.8,0.9,0.95]
task_name = "shortest_path"
deg = 2
num_train_samples = 5000

f_model_name = "Lasso"
h_model_name_box = "MLP"
h_model_name_ellipsoid = "grb"

n_cluster = 10

# paramter for kNN
k_param = 2
smooth_param = 1

dim_covs = 5

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--f_model_name', type=str, default=f_model_name)
parser.add_argument('--h_model_name_box', type=str, default=h_model_name_box)
parser.add_argument('--h_model_name_ellipsoid', type=str, default=h_model_name_ellipsoid)
parser.add_argument('--n_cluster', type=int, default=n_cluster)
parser.add_argument('--k_param', type=float, default=k_param)
parser.add_argument('--smooth_param', type=float, default=smooth_param)
parser.add_argument('--dim_covs', type=int, default=dim_covs)

args = parser.parse_args()
task_name = args.task_name
deg = args.deg
num_train_samples = args.num_train_samples
f_model_name = args.f_model_name
h_model_name_box = args.h_model_name_box
h_model_name_ellipsoid = args.h_model_name_ellipsoid
n_cluster = args.n_cluster
k_param = args.k_param
smooth_param = args.smooth_param
dim_covs = args.dim_covs

train_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

# load c.npy from this directory
c_test = np.load(test_dir+"/c.npy")
dim_c = c_test.shape[1]

alg_name_lists = []
x_sol_lists = []
objs_lists = []
dir_lists = []
coverage_name_lists = []

time_load_start = time.time()
for alpha in alpha_list:
    alg_name_lists.append([])
    x_sol_lists.append([])
    objs_lists.append([])
    dir_lists.append([])
    coverage_name_lists.append([])

    if task_name == "shortest_path":
        suffix = str(alpha)+"/x_sol.csv"
        suffix_obj = str(alpha)+"/objs.csv"
    elif task_name == "knapsack":
        suffix = str(alpha)+"/x_sol.npy"
        suffix_obj = str(alpha)+"/objs.npy"

    # IDCC algorithm
    prefix = "IDCC/"+str(n_cluster)+"/"
    filename = train_dir+prefix+suffix
    obj_filename = train_dir+prefix+suffix_obj
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).values
            objs = pd.read_csv(obj_filename,header=None).values
        elif task_name=="knapsack":
            x_sol = np.load(filename)
            objs = np.load(obj_filename)

        x_sol_lists[-1].append(x_sol)
        objs_lists[-1].append(objs)
        alg_name_lists[-1].append("IDCC"+"-"+str(n_cluster))
        dir_lists[-1].append(train_dir+prefix+str(alpha)+"/")
        coverage_name_lists[-1].append("DCC-"+str(n_cluster))


    # DCC algorithm
    prefix = "DCC/"+str(n_cluster)+"/"
    filename = train_dir+prefix+suffix
    obj_filename = train_dir+prefix+suffix_obj
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).values
            objs = pd.read_csv(obj_filename,header=None).values
        elif task_name=="knapsack":
            x_sol = np.load(filename)
            objs = np.load(obj_filename)
        x_sol_lists[-1].append(x_sol)
        objs_lists[-1].append(objs)
        alg_name_lists[-1].append("DCC"+"-"+str(n_cluster))
        dir_lists[-1].append(train_dir+prefix+str(alpha)+"/")
        coverage_name_lists[-1].append("DCC-"+str(n_cluster))


    # kNN algorithm
    k = max(int(np.ceil(args.k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*dim_c)
    prefix = "kNN/"+str(k)+"/"
    filename = train_dir+prefix+suffix
    obj_filename = train_dir+prefix+suffix_obj
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).values
            objs = pd.read_csv(obj_filename,header=None).values
        elif task_name=="knapsack":
            x_sol = np.load(filename)
            objs = np.load(obj_filename)
        x_sol_lists[-1].append(x_sol)
        objs_lists[-1].append(objs)
        alg_name_lists[-1].append("kNN"+"-"+str(k))
        dir_lists[-1].append(train_dir+prefix+str(alpha)+"/")
        coverage_name_lists[-1].append("kNN")
    
    # ellipsoid algorithm
    prefix = "ellipsoid/"
    filename = train_dir+prefix+suffix
    obj_filename = train_dir+prefix+suffix_obj
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).values
            objs = pd.read_csv(obj_filename,header=None).values
        elif task_name=="knapsack":
            x_sol = np.load(filename)
            objs = np.load(obj_filename)
        x_sol_lists[-1].append(x_sol)
        objs_lists[-1].append(objs)
        alg_name_lists[-1].append("ellipsoid")
        dir_lists[-1].append(train_dir+prefix+str(alpha)+"/")
        coverage_name_lists[-1].append("kNN")
    
    # PTC quantile algorithm
    prefix = "LUQ/"+f_model_name+"/quantile/"+h_model_name_box+"/"
    filename = train_dir+prefix+suffix
    obj_filename = train_dir+prefix+suffix_obj
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).values
            objs = pd.read_csv(obj_filename,header=None).values
        elif task_name=="knapsack":
            x_sol = np.load(filename)
            objs = np.load(obj_filename)
        x_sol_lists[-1].append(x_sol)
        objs_lists[-1].append(objs)
        alg_name_lists[-1].append("LUQ"+"-quantile-"+f_model_name+"-"+h_model_name_box)
        dir_lists[-1].append(train_dir+prefix+str(alpha)+"/")
        coverage_name_lists[-1].append("quantile")
    
    # PTC norm algorithm
    prefix = "LUQ/"+f_model_name+"/norm/"+h_model_name_ellipsoid+"/"
    filename = train_dir+prefix+suffix
    obj_filename = train_dir+prefix+suffix_obj
    if os.path.exists(filename):
        if task_name=="shortest_path":
            x_sol = pd.read_csv(filename,header=None).values
            objs = pd.read_csv(obj_filename,header=None).values
        elif task_name=="knapsack":
            x_sol = np.load(filename)
            objs = np.load(obj_filename)
        x_sol_lists[-1].append(x_sol)
        objs_lists[-1].append(objs)
        alg_name_lists[-1].append("LUQ"+"-norm-"+f_model_name+"-"+h_model_name_ellipsoid)
        dir_lists[-1].append(train_dir+prefix+str(alpha)+"/")
        coverage_name_lists[-1].append("norm")
    
Qs_list = []
meanQ_list = []
coverage_list = []
mean_coverage_list = []
obj_position_list = []

coverage_df = pd.DataFrame(columns=["alpha","alg_name","coverage"])
obj_position_df = pd.DataFrame(columns=["alpha","alg_name","obj_position"])
for alpha_idx in range(len(alpha_list)):
    Qs_list.append([])
    meanQ_list.append([])
    coverage_list.append([])
    mean_coverage_list.append([])
    obj_position_list.append([])

    alpha = alpha_list[alpha_idx]
    
    for alg_idx,x_sol in enumerate(x_sol_lists[alpha_idx]):
        if coverage_name_lists[alpha_idx][alg_idx][:3]=="DCC":
            n_cluster = int(coverage_name_lists[alpha_idx][alg_idx][4:])
            coverage = in_DNN_ellipsoid(dir_lists[alpha_idx][alg_idx],n_cluster,c_test)
        elif coverage_name_lists[alpha_idx][alg_idx]=="kNN":
            coverage = in_kNN_ellipsoid(dir_lists[alpha_idx][alg_idx],c_test)
        elif coverage_name_lists[alpha_idx][alg_idx]=="norm":
            coverage = in_ellipsoid(dir_lists[alpha_idx][alg_idx],c_test)
        elif coverage_name_lists[alpha_idx][alg_idx]=="quantile":
            coverage = in_box(dir_lists[alpha_idx][alg_idx],c_test)
            
        coverage = np.mean(coverage,axis=1)

        coverage_list[-1].append(coverage)
        mean_coverage_list[-1].append(np.mean(coverage))

        if plot_covergae:
            # construct coverage temp dataframe to be concated to coverage_df
            temp_df = pd.DataFrame(columns=["alpha","alg_name","coverage"])
            coverage_arr = np.array(coverage)
            temp_df["coverage"] = coverage_arr.reshape(-1)
            temp_df["alpha"] = alpha*np.ones(coverage_arr.shape[0])
            # repeat the str alg_name for coverage_arr.shape[0] times

            temp_df["alg_name"] = [alg_name_lists[alpha_idx][alg_idx]]*coverage_arr.shape[0]
            coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)
        

        if task_name == "knapsack":
            (num_constraints,num_test_covs,dim_x) = x_sol.shape
            num_test_c = c_test.shape[-1]
            Qs = np.zeros((num_constraints,num_test_covs))
            obj_positions = np.zeros((num_constraints,num_test_covs))
            for cons_idx in range(num_constraints):
                for cov_idx in range(num_test_covs):
                    objs = x_sol[cons_idx,cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                    Qs[cons_idx,cov_idx] = np.quantile(objs,1-alpha)
                    obj_ro = objs_lists[alpha_idx][alg_idx][cons_idx,cov_idx]
                    obj_positions[cons_idx,cov_idx] = np.sum(objs>obj_ro)/num_test_c-alpha
            # reshape Qs to one dimension
            Qs = Qs.reshape(-1)
            Qs_list[-1].append(Qs)
            meanQ_list[-1].append(np.mean(Qs))

            obj_positions = obj_positions.reshape(-1)
            obj_position_list[-1].append(obj_positions)


        elif task_name == "shortest_path":
            (num_test_covs,dim_x) = x_sol.shape
            num_test_c = c_test.shape[-1]
            Qs = np.zeros((num_test_covs))
            obj_positions = np.zeros((num_test_covs))
            for cov_idx in range(num_test_covs):
                objs = x_sol[cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                Qs[cov_idx] = np.quantile(objs,alpha)
                obj_ro = objs_lists[alpha_idx][alg_idx][cov_idx]
                obj_positions[cov_idx] = np.sum(objs<=obj_ro)/num_test_c-alpha
            Qs_list[-1].append(Qs)

            meanQ_list[-1].append(np.mean(Qs))

            obj_position_list[-1].append(obj_positions)

        if plot_obj_position:
            # construct obj_position temp dataframe to be concated to obj_position_df
            temp_df = pd.DataFrame(columns=["alpha","alg_name","obj_position"])
            obj_position_arr = np.array(obj_positions)
            temp_df["obj_position"] = obj_position_arr.reshape(-1)
            temp_df["alpha"] = alpha*np.ones(obj_position_arr.shape[0])
            # repeat the str alg_name for obj_position_arr.shape[0] times
            temp_df["alg_name"] = [alg_name_lists[alpha_idx][alg_idx]]*obj_position_arr.shape[0]
            obj_position_df = pd.concat([obj_position_df,temp_df],ignore_index=True)

        


# save the results to a csv file, where column name is alpha, algrithm name, and meanQ
df = pd.DataFrame(columns=["alpha","alg_name","meanQ","mean_coverage"])

for alpha_idx in range(len(alpha_list)):
    alpha = alpha_list[alpha_idx]
    for alg_idx in range(len(alg_name_lists[alpha_idx])):
        alg_name = alg_name_lists[alpha_idx][alg_idx]
        meanQ = meanQ_list[alpha_idx][alg_idx]
        mean_coverage = mean_coverage_list[alpha_idx][alg_idx]
        # concatenate the data to the dataframe
        df = pd.concat([df,pd.DataFrame([[alpha,alg_name,meanQ,mean_coverage]],columns=["alpha","alg_name","meanQ","mean_coverage"])],ignore_index=True)

df.to_csv(test_dir+"/RO_result_"+str(num_train_samples)+".csv",index=False)

#%%

# use box plot to show the coverage_df's result
if plot_covergae:
    import seaborn as sns

    ax = sns.boxplot(x="alpha", y="coverage", hue="alg_name", data=coverage_df,palette="Set3",showfliers=False)

    # plot the result
    plt.show()
    # save the result
    fig = ax.get_figure()
    fig.savefig(test_dir+"RO_coverage_plot.png")
    fig.savefig(test_dir+"RO_coverage_plot.pdf")

#%%
# use box plot to show the obj_position_df's result
if plot_obj_position:
    import seaborn as sns

    ax = sns.barplot(x="alpha", y="obj_position", hue="alg_name", data=obj_position_df,palette="Set3")

    # plot the result
    plt.show()
    # save the result
    fig = ax.get_figure()
    fig.savefig(test_dir+"RO_obj_position_plot.png")
    fig.savefig(test_dir+"RO_obj_position_plot.pdf")
