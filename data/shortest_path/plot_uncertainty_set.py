
import pandas as pd
from matplotlib import pyplot as plt

deg = 1
plot_cov_dim = 1
plot_c_dim = 1

num_train_samples = 5000

plot_path = "01//"+str(deg)+"//plot//"+str(plot_cov_dim)+"//"

plot_covs = pd.read_csv(plot_path+"covs.csv",header=None).to_numpy()
plot_x = plot_covs[:,plot_cov_dim]

# LB and UB from different algorithms
LBs = []
UBs = []
legends = []
colors = [] # color of the plot line for each algorithm

# append IDCC algorithms' result
LB = pd.read_csv(plot_path+"IDCC"+"_"+str(num_train_samples)+"_"+"10"+"_"+"0.8"+"_LB.csv",header=None).to_numpy()
LB = LB[:,plot_c_dim]
UB = pd.read_csv(plot_path+"IDCC"+"_"+str(num_train_samples)+"_"+"10"+"_"+"0.8"+"_UB.csv",header=None).to_numpy()
UB = UB[:,plot_c_dim]
LBs.append(LB)
UBs.append(UB)
legends.append("IDCC")
colors.append("red")

# append kNN's result
LB = pd.read_csv(plot_path+"kNN"+"_"+str(num_train_samples)+"_"+"50"+"_LB.csv",header=None).to_numpy()
LB = LB[:,plot_c_dim]
UB = pd.read_csv(plot_path+"kNN"+"_"+str(num_train_samples)+"_"+"50"+"_UB.csv",header=None).to_numpy()
UB = UB[:,plot_c_dim]
LBs.append(LB)
UBs.append(UB)
legends.append("kNN")
colors.append("blue")

# append PTC_quantile's result
LB = pd.read_csv(plot_path+"PTC_quantile"+"_"+str(num_train_samples)+"-"+"0.8"+"_LB.csv",header=None).to_numpy()
LB = LB[:,plot_c_dim]
UB = pd.read_csv(plot_path+"PTC_quantile"+"_"+str(num_train_samples)+"-"+"0.8"+"_UB.csv",header=None).to_numpy()
UB = UB[:,plot_c_dim]
LBs.append(LB)
UBs.append(UB)
legends.append("PTC_quantile")
colors.append("green")


# plot the results
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for i in range(len(LBs)):
    ax.plot(plot_x,LBs[i],color=colors[i],label=legends[i])
    ax.plot(plot_x,UBs[i],color=colors[i])
ax.legend()
plt.show()