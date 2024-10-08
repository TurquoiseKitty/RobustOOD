import argparse
import numpy as np
import os
import pandas as pd
import ipdb
from coverage_solver import in_box,in_DNN_ellipsoid,in_kNN_ellipsoid,in_ellipsoid
import matplotlib.pyplot as plt

# import time to measure the time cost
import time

plot_coverage = True
plot_obj_position = True

alpha_list = [0.6,0.7,0.8,0.85,0.9,0.95]
task_name = "shortest_path"
deg = 5
num_train_samples = 5000

f_model_name = "KernelRidge-rbf"
h_model_name_box = "MLP"
h_model_name_ellipsoid = "MLP"

n_cluster = 10

# paramter for kNN
k_param = 1
smooth_param = 1

dim_covs = 10

scaled = True # whether to scale with respect to the expected value of the objective function
plot_scaled_VaR = True

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
#parser.add_argument('--alpha_list', type=list, default=alpha_list)
parser.add_argument('--dim_covs', type=int, default=dim_covs)
parser.add_argument('--scaled', type=bool, default=scaled)
parser.add_argument('--plot_scaled_VaR', type=bool, default=plot_scaled_VaR)
parser.add_argument('--plot_coverage', type=bool, default=plot_coverage)


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
#alpha_list = args.alpha_list
dim_covs = args.dim_covs
scaled = args.scaled
plot_scaled_VaR = args.plot_scaled_VaR
plot_coverage = args.plot_coverage


train_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"

# load c.npy from this directory
c_test = np.load(test_dir+"/c.npy")
# read the covs.csv from this directory
covs_test = pd.read_csv(test_dir+"/covs.csv",header=None).values

dim_c = c_test.shape[1]
dim_covs = covs_test.shape[1]

alg_name_lists = []
VaR_lists = []
obj_position_lists = []
coverage_lists = []

VaR_name = "VaR.csv"
obj_position_name = "obj_pos.csv"
coverage_name = "coverage.csv"

time_load_start = time.time()



for alpha in alpha_list:
    alg_name_lists.append([])
    VaR_lists.append([])
    obj_position_lists.append([])
    coverage_lists.append([])

    # ellipsoid algorithm
    prefix = train_dir+"ellipsoid/"+str(alpha)+"/"
    if os.path.exists(prefix+VaR_name) and os.path.exists(prefix+obj_position_name) and os.path.exists(prefix+coverage_name):
        # alg_name_lists[-1].append("ellipsoid")
        alg_name_lists[-1].append("Ellipsoid")
        VaR_lists[-1].append(np.loadtxt(prefix+VaR_name))
        obj_position_lists[-1].append(pd.read_csv(prefix+obj_position_name).to_numpy().reshape(-1))
        coverage_lists[-1].append(np.loadtxt(prefix+coverage_name))

    # kNN algorithm
    k = max(int(np.ceil(k_param*np.power(num_train_samples,smooth_param/(2*smooth_param)))),2*dim_c)
    prefix = train_dir+"kNN/"+str(k)+"/"+str(alpha)+"/"
    if os.path.exists(prefix+VaR_name) and os.path.exists(prefix+obj_position_name) and os.path.exists(prefix+coverage_name):
        # alg_name_lists[-1].append("kNN"+"-"+str(k))
        alg_name_lists[-1].append("kNN")
        VaR_lists[-1].append(np.loadtxt(prefix+VaR_name))
        obj_position_lists[-1].append(pd.read_csv(prefix+obj_position_name).to_numpy().reshape(-1))
        coverage_lists[-1].append(np.loadtxt(prefix+coverage_name))

    # DCC algorithm
    prefix = train_dir+"DCC/"+str(n_cluster)+"/"+str(alpha)+"/"
    if os.path.exists(prefix+VaR_name) and os.path.exists(prefix+obj_position_name) and os.path.exists(prefix+coverage_name):
        # alg_name_lists[-1].append("DCC"+"-"+str(n_cluster))
        alg_name_lists[-1].append("DCC")
        VaR = np.loadtxt(prefix+VaR_name)
        if task_name=="shortest_path":
            VaR -=250
        VaR_lists[-1].append(VaR)
        obj_position_lists[-1].append(pd.read_csv(prefix+obj_position_name).to_numpy().reshape(-1))
        coverage_lists[-1].append(np.loadtxt(prefix+coverage_name))


    # IDCC algorithm
    prefix = train_dir+"IDCC/"+str(n_cluster)+"/"+str(alpha)+"/"
    if os.path.exists(prefix+VaR_name) and os.path.exists(prefix+obj_position_name) and os.path.exists(prefix+coverage_name):
        #alg_name_lists[-1].append("IDCC"+"-"+str(n_cluster))
        alg_name_lists[-1].append("IDCC")
        VaR = np.loadtxt(prefix+VaR_name)
        if task_name=="shortest_path":
            VaR -=250
        VaR_lists[-1].append(VaR)
        obj_position_lists[-1].append(pd.read_csv(prefix+obj_position_name).to_numpy().reshape(-1))
        coverage_lists[-1].append(np.loadtxt(prefix+coverage_name))

    
    
    # PTC quantile algorithm
    prefix = train_dir+"LUQ/"+f_model_name+"/quantile/"+h_model_name_box+"/"+str(alpha)+"/"

    if os.path.exists(prefix+VaR_name) and os.path.exists(prefix+obj_position_name) and os.path.exists(prefix+coverage_name):
        # alg_name_lists[-1].append("LUQ"+"-quantile-"+f_model_name+"-"+h_model_name_box)
        alg_name_lists[-1].append("PTC-B")
        VaR_lists[-1].append(np.loadtxt(prefix+VaR_name))
        obj_position_lists[-1].append(pd.read_csv(prefix+obj_position_name).to_numpy().reshape(-1))
        coverage_lists[-1].append(np.loadtxt(prefix+coverage_name))

    # PTC norm algorithm
    prefix = train_dir+"LUQ/"+f_model_name+"/norm/"+h_model_name_ellipsoid+"/"+str(alpha)+"/"

    if os.path.exists(prefix+VaR_name) and os.path.exists(prefix+obj_position_name) and os.path.exists(prefix+coverage_name):
        #alg_name_lists[-1].append("LUQ"+"-norm-"+f_model_name+"-"+h_model_name_ellipsoid)
        alg_name_lists[-1].append("PTC-E")
        VaR_lists[-1].append(np.loadtxt(prefix+VaR_name))
        obj_position_lists[-1].append(pd.read_csv(prefix+obj_position_name).to_numpy().reshape(-1))
        coverage_lists[-1].append(np.loadtxt(prefix+coverage_name))



# save the results to a csv file, where column name is alpha, algrithm name, and meanQ
df = pd.DataFrame(columns=["alpha","alg_name","mean_VaR","mean_coverage","std_VaR","std_coverage"])

if scaled:
     # check if opt_obj_expected.csv is in the test_dir, if not, solve the expected optimization problem
    if not os.path.exists(test_dir+"/opt_obj_expected.csv"):
        # load true_f.csv from the test_dir
        true_f = pd.read_csv(test_dir+"/true_f.csv",header=None).to_numpy()
        if task_name=="knapsack":
            from knapsack.solve_true_kp import solve_expected_kp
            # load budgets and prices from knapsack/budgets.csv and knapsack/prices.csv
            budgets = pd.read_csv("knapsack/budgets.csv",header=None).to_numpy()
            prices = pd.read_csv("knapsack/prices.csv",header=None).to_numpy()

            _,opt_obj_expected = solve_expected_kp(true_f,prices,budgets)
            opt_obj_expected = opt_obj_expected.reshape(-1)
        elif task_name=="shortest_path":
            from shortest_path.solve_true_spp import solve_expected_spp
            _,opt_obj_expected = solve_expected_spp(true_f)
            opt_obj_expected = opt_obj_expected.reshape(-1)
        # save opt_obj_expected to test_dir
        np.savetxt(test_dir+"/opt_obj_expected.csv",opt_obj_expected)
    else:
        opt_obj_expected = np.loadtxt(test_dir+"/opt_obj_expected.csv")

    if plot_scaled_VaR:
        import seaborn as sns

        # create a new figure
        plt.figure()

        scaled_VaR_df = pd.DataFrame(columns=["alpha","alg_name","scaled_VaR"])
        # plot scaled VaR
        for alpha_idx in range(len(alpha_list)):
            alpha = alpha_list[alpha_idx]
            for alg_idx in range(len(alg_name_lists[alpha_idx])):
                alg_name = alg_name_lists[alpha_idx][alg_idx]
                scaled_VaR = VaR_lists[alpha_idx][alg_idx]/opt_obj_expected
                if task_name=="knapsack":
                    scaled_VaR = -scaled_VaR
                # concat the data to the dataframe
                temp_df = pd.DataFrame(columns=["alpha","alg_name","scaled_VaR"])
                temp_df["alpha"] = [alpha]*len(scaled_VaR)
                temp_df["alg_name"] = [alg_name]*len(scaled_VaR)
                temp_df["scaled_VaR"] = scaled_VaR
                scaled_VaR_df = pd.concat([scaled_VaR_df,temp_df],ignore_index=True)

        ax = sns.boxplot(x="alpha", y="scaled_VaR", hue="alg_name", data=scaled_VaR_df,palette="Set3",showfliers=False)

        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("Scaled VaR")
        # do not show the legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels,title=None)
        plt.savefig(test_dir+"/scaled_VaR.png",dpi=300,bbox_inches='tight')
        # save to pdf
        plt.savefig(test_dir+"/scaled_VaR.pdf",dpi=300,bbox_inches='tight')

for alpha_idx in range(len(alpha_list)):
    alpha = alpha_list[alpha_idx]
    for alg_idx in range(len(alg_name_lists[alpha_idx])):
        alg_name = alg_name_lists[alpha_idx][alg_idx]
        if not scaled:
            mean_VaR = np.mean(VaR_lists[alpha_idx][alg_idx])
            std_VaR = np.std(VaR_lists[alpha_idx][alg_idx])
        else:
            
            scaled_VaR = VaR_lists[alpha_idx][alg_idx]/opt_obj_expected
            
            mean_VaR = np.mean(scaled_VaR)
            std_VaR = np.std(scaled_VaR)
        mean_coverage = np.mean(coverage_lists[alpha_idx][alg_idx])
        std_coverage = np.std(coverage_lists[alpha_idx][alg_idx])
        # concatenate the data to the dataframe
        df = pd.concat([df,pd.DataFrame([[alpha,alg_name,mean_VaR,mean_coverage,std_VaR,std_coverage]],columns=["alpha","alg_name","mean_VaR","mean_coverage","std_VaR","std_coverage"])],ignore_index=True)

if not scaled:
    df.to_csv(test_dir+"/RO_result_"+str(num_train_samples)+".csv",index=False)
else:
    df.to_csv(test_dir+"/RO_result_"+str(num_train_samples)+"_scaled.csv",index=False)

#%%

# use box plot to show the coverage_df's result
if plot_coverage:
    import seaborn as sns

    # create a new figure
    plt.figure(figsize=(10,8))

    coverage_df = pd.DataFrame(columns=["alpha","alg_name","coverage"])
    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        for alg_idx,alg_name in enumerate(alg_name_lists[alpha_idx]):
            # construct coverage temp dataframe to be concated to coverage_df
            temp_df = pd.DataFrame(columns=["alpha","alg_name","coverage"])
            coverage = coverage_lists[alpha_idx][alg_idx]
            coverage_arr = np.array(coverage)
            temp_df["coverage"] = coverage_arr.reshape(-1)
            temp_df["alpha"] = alpha*np.ones(coverage_arr.shape[0])
            # repeat the str alg_name for coverage_arr.shape[0] times

            temp_df["alg_name"] = [alg_name_lists[alpha_idx][alg_idx]]*coverage_arr.shape[0]
            coverage_df = pd.concat([coverage_df,temp_df],ignore_index=True)

    # save the coverage_df to csv file
    coverage_df.to_csv(test_dir+"/RO_coverage_"+str(num_train_samples)+".csv",index=False)

    ax = sns.boxplot(x="alpha", y="coverage", hue="alg_name", data=coverage_df,palette="Set3",showfliers=False)

    # plot dashed gray line at y=alpha
    for alpha in alpha_list:
        plt.axhline(y=alpha, color='gray', linestyle='--')

    # set the x label and y label
    ax.set_xlabel("$\\alpha$",fontsize=30)
    ax.set_ylabel("Individual coverage",fontsize=25)

    # set the fontsize of the xticks and yticks
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    # Set the legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,title=None,fontsize=20)


    # save the result
    fig = ax.get_figure()
    fig.savefig(test_dir+"RO_coverage_plot.png",bbox_inches='tight')
    fig.savefig(test_dir+"RO_coverage_plot.pdf",bbox_inches='tight')

    # new figure for coverage to plot the absolute value of coverage-alpha
    plt.figure(figsize=(10,8))
    # the column in coverage_df minus the alpha
    coverage_df["coverage-alpha"] = coverage_df["coverage"]-coverage_df["alpha"]
    # get the absolute value of coverage-alpha
    coverage_df["coverage-alpha"] = coverage_df["coverage-alpha"].abs()

    # plot the absolute value of coverage-alpha
    ax = sns.boxplot(x="alpha", y="coverage-alpha", hue="alg_name", data=coverage_df,palette="Set3",showfliers=False)

    # set the x label and y label
    ax.set_xlabel("$\\alpha$",fontsize=30)
    ax.set_ylabel("|Coverage-$\\alpha$|",fontsize=30)

    # set the fontsize of the xticks and yticks
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    # set the legend without title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels,title=None,fontsize=20)


    # save the result
    fig = ax.get_figure()
    fig.savefig(test_dir+"RO_coverage_abs.png",bbox_inches='tight')
    fig.savefig(test_dir+"RO_coverage_abs.pdf",bbox_inches='tight')


#%%

# use box plot to show the obj_position_df's result
if plot_obj_position:
    import seaborn as sns

    # create a new figure for obj_position
    plt.figure()

    obj_position_df = pd.DataFrame(columns=["alpha","alg_name","obj_position"])

    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]
        for alg_idx,alg_name in enumerate(alg_name_lists[alpha_idx]):
            
            # construct obj_position temp dataframe to be concated to obj_position_df
            temp_df = pd.DataFrame(columns=["alpha","alg_name","obj_position"])
            obj_positions = obj_position_lists[alpha_idx][alg_idx]
            obj_position_arr = np.array(obj_positions)
            temp_df["obj_position"] = obj_position_arr.reshape(-1)
            temp_df["alpha"] = alpha*np.ones(obj_position_arr.shape[0])
            # repeat the str alg_name for obj_position_arr.shape[0] times
            temp_df["alg_name"] = [alg_name_lists[alpha_idx][alg_idx]]*obj_position_arr.shape[0]
            obj_position_df = pd.concat([obj_position_df,temp_df],ignore_index=True)

    # save obj_position_df to csv
    obj_position_df.to_csv(test_dir+"RO_obj_position.csv",index=False)

    ax = sns.barplot(x="alpha", y="obj_position", hue="alg_name", data=obj_position_df,palette="Set3")

    # save the result
    fig = ax.get_figure()
    fig.savefig(test_dir+"RO_obj_position_plot.png")
    fig.savefig(test_dir+"RO_obj_position_plot.pdf")
