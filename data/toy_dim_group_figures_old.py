import matplotlib.pyplot as plt
import numpy as np
import ipdb

dim_covs_list = [1,2,4]

alg_names = ["PTC-box","PTC-ellipsoid","kNN","K-Means","Ellipsoid"]

colors = ["red","blue","green","orange","purple"]

num_train_samples = 1000

# create 3*3 subplots figure
fig, axs = plt.subplots(3, 3, figsize=(18,18))

for i in range(len(dim_covs_list)):
    dim_covs = dim_covs_list[i]
    # load data under this dim_covs
    alg_np_list = []
    for alg_name in alg_names:
        file_name = "toy/"+str(dim_covs)+"/1/test/"+alg_name+"_"+str(num_train_samples)+".csv"
        alg_np_list.append(np.loadtxt(file_name, delimiter=','))

    true_np = np.loadtxt("toy/"+str(dim_covs)+"/1/test/true.csv", delimiter=',')

    # plot the uncertainty set
    train_dir = "toy/"+str(dim_covs)+"/1/train/"+str(num_train_samples)+"/"
    covs_train = np.loadtxt(train_dir+"covs.csv", delimiter=',')
    covs_train = covs_train.reshape(num_train_samples,-1)
    c_train = np.loadtxt(train_dir+"c.csv", delimiter=',')

    ax = axs[0,i]
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # plot the LB and UB curves of different algorithms
    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('c',fontsize=20)
    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)
    for alg_idx,alg_name in enumerate(alg_names):
        ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,1],color=colors[alg_idx],label=alg_name)
        ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,2],color=colors[alg_idx])
    
    ax.scatter(covs_train[:,0],c_train,color="black",s=3,zorder=10)


    # plot the optimal solution
    ax = axs[1,i]
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('x',fontsize=20)

    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)
    for alg_idx,alg_name in enumerate(alg_names):
        ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,3],color=colors[alg_idx],label=alg_name)
    ax.plot(true_np[:,0],true_np[:,3],color="black",label="optimal",linestyle=":")
    # not show the legend in the ax
    #ax.legend(fontsize=20)

    # plot the VaR
    ax = axs[2,i]
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    
    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('VaR',fontsize=20)

    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)

    for alg_idx,alg_name in enumerate(alg_names):
        ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,4],color=colors[alg_idx],label=alg_name)
    ax.plot(true_np[:,0],true_np[:,4],color="black",label="optimal",linestyle=":")

lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc='upper center',ncol=6,fontsize=20)
#fig.legend( lines, labels, bbox_to_anchor=(0.5, -0.3),ncol=5, framealpha=1)

fig.tight_layout()
fig.savefig("toy/dim.png")
fig.savefig("toy/dim.pdf")

