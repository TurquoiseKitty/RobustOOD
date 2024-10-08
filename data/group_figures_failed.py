import matplotlib.pyplot as plt
import argparse
# import pickle
import pickle5 as pickle


plot_toy_dim = True

parser = argparse.ArgumentParser()
parser.add_argument('--plot_toy_dim', type=bool, default=plot_toy_dim)
args = parser.parse_args()

plot_toy_dim = args.plot_toy_dim

if plot_toy_dim:
    ############ for toy dim ################
    # List of .fig file names
    fig_files = []
    toy_dim_list = [1,2,4]
    num_train_samples = 1000
    for dim_covs in toy_dim_list:
        file_name = "toy/"+str(dim_covs)+"/1/test/"+str(dim_covs)+"_"+str(num_train_samples)+"_uncertainty_set_ax.pkl"
        fig_files.append(file_name)
        file_name = "toy/"+str(dim_covs)+"/1/test/"+str(dim_covs)+"_"+str(num_train_samples)+"_optimal_solution_ax.pkl"
        fig_files.append(file_name)
        file_name = "toy/"+str(dim_covs)+"/1/test/"+str(dim_covs)+"_"+str(num_train_samples)+"_VaR_ax.pkl"
        fig_files.append(file_name)
else:
    ############# for toy num_train_samples ###########
    # List of .fig file names
    fig_files = []
    toy_num_train_samples_list = [100,200,1000]
    dim_covs = 1
    for num_train_samples in toy_num_train_samples_list:
        file_name = "toy/"+str(dim_covs)+"/1/test/"+str(dim_covs)+"_"+str(num_train_samples)+"_uncertainty_set_ax.pkl"
        fig_files.append(file_name)
        file_name = "toy/"+str(dim_covs)+"/1/test/"+str(dim_covs)+"_"+str(num_train_samples)+"_optimal_solution_ax.pkl"
        fig_files.append(file_name)
        file_name = "toy/"+str(dim_covs)+"/1/test/"+str(dim_covs)+"_"+str(num_train_samples)+"_VaR_ax.pkl"
        fig_files.append(file_name)


legend_labels = ["PTC-box","PTC-ellipsoid","KNN","Ellipsoid","K-Means"]

# Create a new figure with a 3x3 grid layout
plt.figure(figsize=(18,18))

indices = [(3,3,1),(3,3,4),(3,3,7),(3,3,2),(3,3,5),(3,3,8),(3,3,3),(3,3,6),(3,3,9)]

# Iterate over the subplots and load the .fig files
for subplot_idx, fig_file in zip(indices, fig_files):
    
    ax = plt.subplot(subplot_idx)
    # Load the .fig file into the subplot
    loaded_ax = pickle.load(open(fig_file, 'rb'))
    ax.__dict__.update(loaded_ax.__dict__)



# Display the big figure
plt.show()

# save the figures
if plot_toy_dim:
    plt.savefig("toy/toy_dim.pdf", bbox_inches='tight')
    plt.savefig("toy/toy_dim.png", bbox_inches='tight')
else:
    plt.savefig("toy/toy_num_train_samples.pdf", bbox_inches='tight')
    plt.savefig("toy/toy_num_train_samples.png", bbox_inches='tight')