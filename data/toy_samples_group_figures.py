import matplotlib.pyplot as plt
import numpy as np
import ipdb
import pandas as pd



num_train_samples_list = [100,200,1000]

alg_names = ["Ellipsoid","kNN","K-Means","PTC-box","PTC-ellipsoid"]
legend_alg_names = ["Ellipsoid","kNN","K-Means","PTC-B","PTC-E"]

colors = ["purple","green","orange","red","blue"]

dim_covs = 1

# create 3*3 subplots figure
fig, axs = plt.subplots(2, 3, figsize=(18,10),sharex='col',sharey='row', gridspec_kw={'height_ratios': [3, 1]})

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

for i in range(len(num_train_samples_list)):
    num_train_samples = num_train_samples_list[i]
    # load data under this dim_covs
    alg_np_list = []
    for alg_name in alg_names:
        file_name = "toy/"+str(dim_covs)+"/1/test/"+alg_name+"_"+str(num_train_samples)+".csv"
        alg_np_list.append(np.loadtxt(file_name, delimiter=','))

    

    # plot the uncertainty set
    train_dir = "toy/"+str(dim_covs)+"/1/train/"+str(num_train_samples)+"/"
    covs_train = np.loadtxt(train_dir+"covs.csv", delimiter=',')
    covs_train = covs_train.reshape(num_train_samples,-1)
    c_train = np.loadtxt(train_dir+"c.csv", delimiter=',')

    ax = axs[0,i]
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    # plot the LB and UB curves of different algorithms
    if i==0:
        #ax.set_xlabel('z',fontsize=20)
        ax.set_ylabel('$c$',fontsize=30)
    ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)
    for alg_idx,alg_name in enumerate(alg_names):
        line, = ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,1],color=colors[alg_idx],label=legend_alg_names[alg_idx])
        ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,2],color=colors[alg_idx])
        legend_handles_row1.append(line)
    
    ax.scatter(covs_train[:,0],c_train,color="black",s=3,zorder=10)

    # plot the dashed vertical lines indicating the flip points
    for flip_point in flip_points:
        ax.axvline(x=flip_point,color="black",linestyle="--",linewidth=1)


    # plot the optimal solution
    ax = axs[1,i]
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    ax.set_xlabel('$z_1$',fontsize=20)
    #if i==0:
    #    ax.set_ylabel('Algorithm Name',fontsize=20)

    #ax.set_title("T="+str(num_train_samples)+", d="+str(dim_covs),fontsize=20)
    # create a dataframe used to draw the gant graph
    gant_df = pd.DataFrame(columns=["x","algorithm_name","start","length"])


    for alg_idx,alg_name in enumerate(alg_names):
        if alg_name=="PTC-box":
            alg_name="PTC-B"
        elif alg_name=="PTC-ellipsoid":
            alg_name="PTC-E"
        start = alg_np_list[alg_idx][0,0]
        cur_x = alg_np_list[alg_idx][0,3]
        for j in range(1,true_np.shape[0]):
            if alg_np_list[alg_idx][j,3]!=cur_x: # if x has change the phase
                length = alg_np_list[alg_idx][j,0]-start
                #gant_df = gant_df.append({"x":cur_x,"Algorithm_name":alg_name,"start":start,"length":alg_np_list[alg_idx][j,0]-start},ignore_index=True)
                gant_df = pd.concat([gant_df,pd.DataFrame({"x":[cur_x],"algorithm_name":[legend_alg_names[alg_idx]],"start":[start],"length":[length]})])
                start = alg_np_list[alg_idx][j,0] # update start z
                cur_x = alg_np_list[alg_idx][j,3]
        # add the last phase
        length = alg_np_list[alg_idx][-1,0]-start
        gant_df = pd.concat([gant_df,pd.DataFrame({"x":[cur_x],"algorithm_name":[alg_name],"start":[start],"length":[length]})])

    # inverse the row of gant_df
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

        #ax.plot(alg_np_list[alg_idx][:,0],alg_np_list[alg_idx][:,3],color=colors[alg_idx],label=alg_name)
    #ax.plot(true_np[:,0],true_np[:,3],color="black",label="optimal",linestyle=":")
    # draw the gant graph
    bar_containers = ax.barh(gant_df.algorithm_name,gant_df.length,left=gant_df.start,color=gant_df.x.map(color_map))

    legend_handles_row2 = [plt.Rectangle((0, 0), 1, 1, color=color_map[x]) for x in gant_df.x.unique()]

    # plot the dashed vertical lines indicating the flip points
    for flip_point in flip_points:
        ax.axvline(x=flip_point,color="black",linestyle="--",linewidth=1)



    """
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
    """


fig.tight_layout()
plt.subplots_adjust(top=0.8, bottom=0.2)

legend1 = fig.legend(legend_handles_row1,legend_alg_names,loc='upper center',ncol=6,fontsize=20, bbox_to_anchor=(0.5, 0.9))
#lt.subplots_adjust(top=0.8) 
# legend1.set_bbox_to_anchor((0.5, 1.1))
legend2 = fig.legend(legend_handles_row2,["$x=-1$","$x=0$","$x=1$"],loc='lower center', bbox_to_anchor=(0.5,0.04), ncol=3,fontsize=20)
#plt.subplots_adjust(bottom=0.2)  
# legend2.set_bbox_to_anchor((0.5, -0.15))
# fig.tight_layout()

fig.savefig("toy/samples.png", dpi=300, bbox_inches='tight')
fig.savefig("toy/samples.pdf", bbox_inches='tight')

