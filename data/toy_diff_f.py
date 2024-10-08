import argparse
import numpy as np
import os
import pandas as pd
import ipdb
from coverage_solver import in_box,in_DNN_ellipsoid,in_kNN_ellipsoid,in_ellipsoid
import matplotlib.pyplot as plt
# import pickle
import pickle5 as pickle
import subprocess

# import time to measure the time cost
import time
from toy.generate_samples import get_true_LB_UB


alpha = 0.8
task_name = "toy"
deg = 1
num_train_samples = 1000
k_param = 2
smooth_param = 1
n_cluster = 10
dim_covs = 1

f_model_name_list = ["Lasso","KernelRidge-rbf","MLP"]#,"KernelRidge-poly"]

h_model_name_box = "MLP"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default=task_name)
parser.add_argument('--deg', type=int, default=deg)
parser.add_argument('--num_train_samples', type=int, default=num_train_samples)
parser.add_argument('--k_param', type=float, default=k_param)
parser.add_argument('--smooth_param', type=float, default=smooth_param)
parser.add_argument('--n_cluster', type=int, default=n_cluster)
parser.add_argument('--dim_covs', type=int, default=dim_covs)

parser.add_argument('--h_model_name_box', type=str, default=h_model_name_box)


args = parser.parse_args()
task_name = args.task_name
deg = args.deg
num_train_samples = args.num_train_samples
k_param = args.k_param
smooth_param = args.smooth_param
n_cluster = args.n_cluster
dim_covs = args.dim_covs

h_model_name_box = args.h_model_name_box


train_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/train/"+str(num_train_samples)+"/"
test_dir = task_name+"/"+str(dim_covs)+"/"+str(deg)+"/test/"


######################## call and run algorithms #############################
cur_dir = os.getcwd()
#PTC_dir = cur_dir+"../LUQ/"
PTC_dir = cur_dir+"/../LUQ/"

for f_model_name in f_model_name_list:
    # go to PTC dir
    os.chdir(PTC_dir)

    # python train_f.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --model_name $f_model_name
    subprocess.call(["python","train_f.py","--task_name",task_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--model_name",f_model_name])
    # python train_quantile_h.py --task_name $dataset_name --dim_covs $dim_covs --num_train_samples $num_train_samples --deg $deg --f_model_name $f_model_name --h_model_name $h_model_name --alpha $alpha
    subprocess.call(["python","train_quantile_h.py","--task_name",task_name,"--dim_covs",str(dim_covs),"--num_train_samples",str(num_train_samples),"--deg",str(deg),"--f_model_name",f_model_name,"--h_model_name",h_model_name_box,"--alpha",str(alpha)])
    
# go back to cur_dir
os.chdir(cur_dir)


######################## process results #####################################

# load c.npy from this directory
c_test = np.load(test_dir+"/c.npy")

alg_name_list = []
LB_list = []
UB_list = []


def get_LB_UB_from_coverage(coverage,c_test):
    LB = np.zeros(coverage.shape[0])
    UB = np.zeros(coverage.shape[0])
    # for each row, get the max column index whose value is 1
    for row_idx in range(coverage.shape[0]):
        for column_idx in range(coverage.shape[1]):
            if coverage[row_idx,column_idx]==1:
                LB[row_idx] = c_test[row_idx,0,column_idx]
                break
        for column_idx in range(coverage.shape[1]-1,-1,-1):
            if coverage[row_idx,column_idx]==1:
                UB[row_idx] = c_test[row_idx,0,column_idx]
                break
    return LB,UB

c_test_pred_list = []
time_load_start = time.time()
for f_model_name in f_model_name_list:
    # PTC quantile algorithm
    prefix = "LUQ/"+f_model_name+"/quantile/"+h_model_name_box+"/"+str(alpha)+"/"
    filename = train_dir+prefix
    if os.path.exists(filename):
        #alg_name_lists[-1].append("PTC"+"-box-"+f_model_name+"-"+h_model_name_box)
        if f_model_name=="KernelRidge-rbf":
            alg_name_list.append("KernelRidge")
        elif f_model_name=="MLP":
            alg_name_list.append("NN")
        else:
            alg_name_list.append(f_model_name)
        coverage = in_box(train_dir+prefix,c_test)
        LB,UB = get_LB_UB_from_coverage(coverage,c_test)
        LB_list.append(LB)
        UB_list.append(UB)
        # load c_test_pred.csv from this directory
        fit_dir = train_dir+"LUQ/"+f_model_name+"/"
        c_test_pred_list.append(pd.read_csv(fit_dir+"c_test_pred.csv",header=None).values)
    
    
    

############################################### solve to x and save it to numpy ###############################################


covs_train = np.loadtxt(train_dir+"covs.csv",delimiter=",")
c_train = np.loadtxt(train_dir+"c.csv",delimiter=",")
if len(covs_train.shape)==1:
    covs_train = np.reshape(covs_train,(-1,1))

covs_test = np.loadtxt(test_dir+"covs.csv",delimiter=",")


if len(covs_test.shape)==1:
    covs_test = np.reshape(covs_test,(-1,1))

alg_np_list = [np.zeros((covs_test.shape[0],5)) for _ in range(len(alg_name_list))]
true_np = np.zeros((covs_test.shape[0],5))

start_idx = 0
end_idx = covs_test.shape[0]


for alg_idx,alg_name in enumerate(alg_name_list):
    # find the index of algrithm in uniq_alg_name_list
    name_idx = alg_name_list.index(alg_name)
    temp_np = alg_np_list[name_idx]
    
    temp_np[start_idx:end_idx,0] = covs_test[:,0]
    temp_np[start_idx:end_idx,1] = UB_list[alg_idx]
    temp_np[start_idx:end_idx,2] = LB_list[alg_idx]

true_np[start_idx:end_idx,0] = covs_test[:,0]


#%% 
################################### plot the optimal solution of different algorithms ###################################
x_sol_list = []
x_opt_list = []

# get the optimal solution of min Var of covs_test
# test_true_UB = 2/(1+np.exp(covs_test[:,0]+(1-alpha)*0.2))-0.7
# test_true_LB = 2/(1+np.exp(covs_test[:,0]+alpha*0.2))-0.7
test_true_LB, test_true_UB = get_true_LB_UB(covs_test,alpha)

x_opt = -np.ones(len(covs_test))*(test_true_LB>0)+np.ones(len(covs_test))*(test_true_UB<0)
x_opt_list.append(x_opt)


start_idx = 0
end_idx = covs_test.shape[0]

for alg_idx,alg_name in enumerate(alg_name_list):
    LB = LB_list[alg_idx]
    UB = UB_list[alg_idx]
    x = -np.ones(len(covs_test))*(LB>0)+np.ones(len(covs_test))*(UB<0)
    x_sol_list.append(x)
    # find the index of algrithm in uniq_alg_name_list
    name_idx = alg_name_list.index(alg_name)
    temp_np = alg_np_list[name_idx]
    
    temp_np[start_idx:end_idx,3] = x

true_np[start_idx:end_idx,3] = x_opt


    
#%%
################################### calculate the Var of x from different algorithms ###################################

# test_true_UB = 2/(1+np.exp(covs_test[:,0]+(1-alpha)*0.2))-0.7
# test_true_LB = 2/(1+np.exp(covs_test[:,0]+alpha*0.2))-0.7
test_true_LB, test_true_UB = get_true_LB_UB(covs_test,alpha)

# true_var = max{x_opt*test_true_UB,x_opt*test_true_LB}
true_var = np.maximum(x_opt*test_true_UB,x_opt*test_true_LB)

start_idx = 0
end_idx = covs_test.shape[0]

for alg_idx,alg_name in enumerate(alg_name_list):
    x = x_sol_list[alg_idx]
    x_var = np.maximum(x*test_true_UB,x*test_true_LB)
    # find the index of algrithm in uniq_alg_name_list
    name_idx = alg_name_list.index(alg_name)
    temp_np = alg_np_list[name_idx]
    
    temp_np[start_idx:end_idx,4] = x_var

true_np[start_idx:end_idx,4] = true_var



#%%
####### save the numpy array of all algorithms #######
for alg_idx,alg_name in enumerate(alg_name_list):
    np.savetxt(test_dir+alg_name+"_"+str(num_train_samples)+".csv",alg_np_list[alg_idx],delimiter=",")
np.savetxt(test_dir+"true.csv",true_np,delimiter=",")


#%%

# plot the uncertainty set
train_dir = "toy/"+str(dim_covs)+"/1/train/"+str(num_train_samples)+"/"
covs_train = np.loadtxt(train_dir+"covs.csv", delimiter=',')
covs_train = covs_train.reshape(num_train_samples,-1)
c_train = np.loadtxt(train_dir+"c.csv", delimiter=',')


# Group features
colors = ["blue","green","red",'#ffc300',"orange","purple"]
linestyles = ["dotted","dashdot","dashed"]

dim_covs = 1

# create 3*3 subplots figure
fig, axs = plt.subplots(2, 1, figsize=(10,8),sharex='col',sharey='row',gridspec_kw={'height_ratios': [4, 1]})

legend_handles_row1 = []  
legend_handles_row2 = []  

# Map colors to values
color_map = { -1: '#FA7F6F', 1: '#82B0D2', 0: '#eddd86' }

true_np = np.loadtxt("toy/"+str(dim_covs)+"/1/test/true.csv", delimiter=',')

x_opt = true_np[:,3]
flip_points = []
for i in range(len(x_opt)-1):
    if x_opt[i]!=x_opt[i+1]:
        flip_points.append(true_np[i+1,0])


#################### plot uncertainty set ######################
for alg_idx in range(len(alg_name_list)):
    if alg_name_list[alg_idx]=="PTC-box":
        alg_name_list[alg_idx] = "PTC-BUQ"
    elif alg_name_list[alg_idx]=="PTC-ellipsoid":
        alg_name_list[alg_idx] = "PTC-EUQ"
    elif alg_name_list[alg_idx] == "KernelRidge":
        alg_name_list[alg_idx] = "KR"

ax = axs[0]
ax.set_title('Uncertainty set',fontsize=20)
# set ax size
ax.tick_params(axis='x', labelsize=20)
ax.set_xlabel('$z_1$',fontsize=20)
ax.set_ylabel('$c$',fontsize=30)

for alg_idx,alg_name in enumerate(alg_name_list):
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # plot the LB and UB curves of different algorithms
    
    # ax.plot(alg_np_list[alg_idx][:,0],c_test_pred_list[alg_idx],color=colors[alg_idx],linestyle="solid")
    line, =ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,1],color=colors[alg_idx],label=alg_name,linestyle = linestyles[alg_idx])
    ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,2],color=colors[alg_idx],linestyle = linestyles[alg_idx])

    legend_handles_row1.append(line)
    ax.scatter(covs_train[:,0],c_train,color="black",s=1)

ax.legend(handles=legend_handles_row1, loc='best',fontsize=20)

# plot the dashed vertical lines indicating the flip points
for flip_point in flip_points:
    ax.axvline(x=flip_point,color="black",linestyle="--",linewidth=1)

############################### plot optimal solution ###############################
ax = axs[1]
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.set_xlabel('$z_1$',fontsize=20)
#ax.set_ylabel('Solution x',fontsize=20)

# set title
ax.set_title("Robust solution $x$",fontsize=20)

gant_df = pd.DataFrame(columns=["x","algorithm_name","start","length"])

for alg_idx,alg_name in enumerate(alg_name_list):
    #ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)
    # create a dataframe used to draw the gant graph
    

    # plot the gant graph
    start = alg_np_list[alg_idx][0,0]
    cur_x = alg_np_list[alg_idx][0,3]
    for j in range(1,true_np.shape[0]):
        if alg_np_list[alg_idx][j,3]!=cur_x: # if x has changed the phase
            length = alg_np_list[alg_idx][j,0]-start
            #gant_df = gant_df.append({"x":cur_x,"Algorithm_name":alg_name,"start":start,"length":alg_np_list[alg_idx][j,0]-start},ignore_index=True)
            gant_df = pd.concat([gant_df,pd.DataFrame({"x":[cur_x],"algorithm_name":[alg_name],"start":[start],"length":[length]})])
            start = alg_np_list[alg_idx][j,0] # update start z
            cur_x = alg_np_list[alg_idx][j,3]
    # add the last phase
    length = alg_np_list[alg_idx][-1,0]-start
    gant_df = pd.concat([gant_df,pd.DataFrame({"x":[cur_x],"algorithm_name":[alg_name],"start":[start],"length":[length]})])



# inverse the order of the dataframe by row
gant_df = gant_df.iloc[::-1]


# do the same for the optimal solution
start = true_np[0,0]
cur_x = true_np[0,3]
for j in range(1,true_np.shape[0]):
    if true_np[j,3]!=cur_x:
        length = true_np[j,0]-start
        gant_df = pd.concat([gant_df,pd.DataFrame({"x":[cur_x],"algorithm_name":["Optimal"],"start":[start],"length":[length]})])
        start = true_np[j,0]
        cur_x = true_np[j,3]
length = true_np[-1,0]-start
gant_df = pd.concat([gant_df,pd.DataFrame({"x":[cur_x],"algorithm_name":["Optimal"],"start":[start],"length":[length]})])

fig.tight_layout()
# draw the gant graph
bar_containers = ax.barh(gant_df.algorithm_name,gant_df.length,left=gant_df.start,color=gant_df.x.map(color_map))
# ax.legend(bar_containers, gant_df.x.unique(),loc='best',ncol=3,fontsize=20)
plt.subplots_adjust(bottom=0.2)
fig.legend(bar_containers,["$x=-1$","$x=0$","$x=1$"],loc='lower center', bbox_to_anchor=(0.5,0.03), ncol=3,fontsize=20)

#legend_handles_row2 = [plt.Rectangle((0, 0), 1, 1, color=color_map[x]) for x in gant_df.x.unique()]

# plot the dashed vertical lines indicating the flip points
for flip_point in flip_points:
    ax.axvline(x=flip_point,color="black",linestyle="--",linewidth=1)




#lines, labels = fig.axes[-1].get_legend_handles_labels()

#fig.legend(lines, labels, loc='upper center',ncol=6,fontsize=20)

#legend1 = fig.legend(legend_handles_row1,alg_name_list,loc='upper center',ncol=6,fontsize=20)

#legend2 = fig.legend(legend_handles_row2,["x=-1","x=0","x=1"],loc='lower center',ncol=3,fontsize=20)



fig.savefig("toy/diff_f.png",bbox_inches='tight')
fig.savefig("toy/diff_f.pdf",bbox_inches='tight')
