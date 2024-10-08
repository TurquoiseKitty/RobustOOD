
'''Use kNN to solve the contextual robust optimization problem'''

# import kmeans clustering
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import sys
# import the mahanalobis distance
from scipy.spatial.distance import mahalanobis

import ipdb

from scipy.optimize import minimize
import scipy

from solver import solve_ellipsoid,get_LB_UB
import argparse



sys.path.append(os.getcwd())
sys.path.append('..')
from data.split_data_for_LUQ import split_data_for_LUQ
from data.read_mse_Var import get_Var
from data.coverage_solver import in_kNN_ellipsoid
from LUQ.solver import get_spp_Ab,get_kp_Ab

n_cluster = 10

task_name = "shortest_path"
num_train_samples = 5000

deg = 1
plot_cov_dim = 1
alpha = 0.8 # options: 0.8, 0.85, 0.9, 0.95, if fit_goal is "f", then alpha is not used
dim_covs = 5

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='shortest_path')
parser.add_argument('--num_train_samples', type=int, default=5000)
parser.add_argument('--deg', type=int, default=1)
parser.add_argument('--plot_cov_dim', type=int, default=1)
parser.add_argument('--n_cluster', type=int, default=n_cluster)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--dim_covs', type=int, default=5)

arg = parser.parse_args()

task_name = arg.task_name
num_train_samples = arg.num_train_samples
deg = arg.deg
plot_cov_dim = arg.plot_cov_dim

n_cluster = arg.n_cluster
alpha = arg.alpha
dim_covs = arg.dim_covs

train_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//train//"+str(num_train_samples)+"//"
test_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//test//"
#plot_dir = "..//data//"+task_name+"/"+str(dim_covs)+"/"+str(deg)+"//plot//"+str(plot_cov_dim)+"//"

plot_dir = None



if task_name=="knapsack":
    constraint_dir = "../data/"+task_name+"/"
    # load prices and budgets from cosntraint_dir
    prices = pd.read_csv(constraint_dir+"prices.csv",header=None).to_numpy()
    budgets = pd.read_csv(constraint_dir+"budgets.csv",header=None).to_numpy()

# load covs_train, c_train, covs_test from train_dir and test_dir
covs_train = pd.read_csv(train_dir+"covs.csv",header=None).to_numpy()
covs_test = pd.read_csv(test_dir+"covs.csv",header=None).to_numpy()
c_train = pd.read_csv(train_dir+"c.csv",header=None).to_numpy()

dim_c = c_train.shape[1]

# load plot samples
if plot_dir is not None:
    covs_plot = pd.read_csv(plot_dir+"covs.csv",header=None).to_numpy()


num_test_samples = covs_test.shape[0]

save_dir = train_dir+"cluster/"+str(n_cluster)+"/"+str(alpha)+"/"

if not os.path.exists(save_dir+"x_sol.npy") and not os.path.exists(save_dir+"x_sol.csv"):
    
    dim_covs = covs_train.shape[1]
    dim_c = c_train.shape[1]

    #preprocess data
    scaler = StandardScaler()
    covs_train = scaler.fit_transform(covs_train)
    covs_test = scaler.transform(covs_test)

    if plot_dir is not None:
        covs_plot = scaler.transform(covs_plot)


    covss = np.zeros((num_test_samples,dim_c,dim_c))
    mus = np.zeros((num_test_samples,dim_c))
    Rs = np.zeros(num_test_samples)
    # check if covss and mus have been saved, if so, load them
    if os.path.exists(save_dir+"covss.npy") and os.path.exists(save_dir+"mus.npy") and os.path.exists(save_dir+"Rs.npy"):
        covss = np.load(save_dir+"covss.npy")
        mus = np.load(save_dir+"mus.npy")
        Rs = np.load(save_dir+"Rs.npy")
    else:
        covs_clusters = np.zeros((n_cluster,dim_c,dim_c))
        mus_clusters = np.zeros((n_cluster,dim_c))
        Rs_clusters = np.zeros(n_cluster)
        # cluster the covs_train into n_cluster clusters
        # for each cluster, calculate the c_train's covariance matrix and the mean vector
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(covs_train)
        labels = kmeans.labels_
        labels_test = kmeans.predict(covs_test)

        for k in range(n_cluster):
            if np.sum(labels==k)==0:
                covs_clusters[k,:,:] = np.cov(covs_train.transpose())
                mus_clusters[k,:] = np.mean(covs_train,axis=0)
                c_points = covs_train
            else:
                covs_clusters[k,:,:] = np.cov(c_train[labels==k,:].transpose())
                mus_clusters[k,:] = np.mean(c_train[labels==k,:],axis=0)
                # calibrate to get Rs
                c_points = c_train[labels==k,:]
                
            # calculate the mahalanobis distance of each point to the mean
            dists = np.zeros(c_points.shape[0])
            for i in range(c_points.shape[0]):
                dists[i] = mahalanobis(c_points[i,:],mus_clusters[k,:],np.linalg.inv(covs_clusters[k,:,:]))
            # get the alpha percentile of the distances
            Rs_clusters[k] = np.percentile(dists,alpha*100)


        for i in range(covs_test.shape[0]):
            covss[i,:,:] = covs_clusters[labels_test[i],:,:] # covs_test[i,:,:]
            mus[i,:] = mus_clusters[labels_test[i],:] # c_test[i,:]
            Rs[i] = Rs_clusters[labels_test[i]]
            
        # save covss and mus and Rs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir+"covss.npy",covss)
        np.save(save_dir+"mus.npy",mus)
        np.save(save_dir+"Rs.npy",Rs)

    if plot_dir is not None:
        # get cov, mu, R of the plot samples
        covss_plot = np.zeros((covs_plot.shape[0],dim_c,dim_c))
        mus_plot = np.zeros((covs_plot.shape[0],dim_c))
        Rs_plot = np.zeros(covs_plot.shape[0])
        for i in range(covs_plot.shape[0]):
            dists = np.sum(np.square(covs_train-covs_plot[i,:]),axis=1)
            sorted_dists = np.sort(dists)
            k_nearest_neighbors = np.where(dists<=sorted_dists[k-1])[0]
            points = c_train[k_nearest_neighbors,:]
            mu = np.mean(points,axis=0)
            cov_mat = np.cov(points.transpose())
            covss_plot[i,:,:] = cov_mat
            mus_plot[i,:] = mu
            cov_inv = np.linalg.inv(cov_mat)
            L = np.linalg.cholesky(cov_inv)
            distances = np.linalg.norm((points-mu.reshape(1,-1))@L,axis=1)
            R = np.max(distances)
            Rs_plot[i] = R
        plot_LB, plot_UB = get_LB_UB(mus_plot,covss_plot,Rs_plot)
        # save the plot_LB and plot_UB
        np.savetxt(plot_dir+"kNN_"+str(num_train_samples)+"_"+str(k)+"_LB.csv",plot_LB,delimiter=",")
        np.savetxt(plot_dir+"kNN_"+str(num_train_samples)+"_"+str(k)+"_UB.csv",plot_UB,delimiter=",")

    # solve the robust optimization problem

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if task_name=="shortest_path":
        if not os.path.exists(save_dir+"x_sol.csv"):
            A,b = get_spp_Ab()
            x_sol,objs = solve_ellipsoid(mus,covss,Rs,A,b,task_name)
            # save the solution
            np.savetxt(save_dir+"x_sol.csv",x_sol,delimiter=",")
            np.savetxt(save_dir+"objs.csv",objs,delimiter=",")

    elif task_name=="knapsack":
        # check if the solution file x_sol.npy is exist
        if not os.path.exists(save_dir+"x_sol.npy"):
            # load prices and budgets
            prices = np.loadtxt("../data/knapsack/prices.csv",delimiter=",")
            budgets = np.loadtxt("../data/knapsack/budgets.csv",delimiter=",")
            num_constraints = budgets.shape[0]
            num_test = mus.shape[0]
            # solve the optimization problem
            x_sols = np.zeros((num_constraints,num_test,prices.shape[1]))
            objss = np.zeros((num_constraints,num_test))
            for i in range(num_constraints):
                A, b = get_kp_Ab(prices[i,:],budgets[i])
                x_sol,objs = solve_ellipsoid(mus,covss,Rs,A,b,task_name)
                x_sols[i,:,:] = x_sol
                objss[i,:] = objs
            # save the solution
            np.save(save_dir+"x_sol.npy",x_sols)
            np.save(save_dir+"objs.npy",objss)

if task_name!="toy":
    if not os.path.exists(save_dir+"VaR.csv") or not os.path.exists(save_dir+"obj_pos.csv") or not os.path.exists(save_dir+"coverage.csv"):
        # load c.npy from test_dir
        c_test = np.load(test_dir+"c.npy")
        # load x_sol and objs
        if task_name == "shortest_path":
            x_sol = pd.read_csv(save_dir+"x_sol.csv",header=None).to_numpy()
            objs = pd.read_csv(save_dir+"objs.csv",header=None).to_numpy()
            (num_test_covs,dim_x) = x_sol.shape
            num_test_c = c_test.shape[-1]

            obj_positions = np.zeros((num_test_covs))
            for cov_idx in range(num_test_covs):
                true_objs = x_sol[cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                obj_positions[cov_idx] = np.sum(true_objs<=objs[cov_idx])/num_test_c
            
        elif task_name == "knapsack":
            x_sol = np.load(save_dir+"x_sol.npy")
            objs = np.load(save_dir+"objs.npy")
            (num_constraints,num_test_covs,dim_x) = x_sol.shape
            num_test_c = c_test.shape[-1]
            
            obj_positions = np.zeros((num_constraints,num_test_covs))
            for cons_idx in range(num_constraints):
                for cov_idx in range(num_test_covs):
                    true_objs = x_sol[cons_idx,cov_idx,:].reshape(1,dim_x)@c_test[cov_idx,:,:]
                    obj_positions[cons_idx,cov_idx] = np.sum(true_objs>objs[cons_idx,cov_idx])/num_test_c

        # save the obj_positions
        np.savetxt(save_dir+"obj_pos.csv",obj_positions,delimiter=",")    
        
        
        # calculate the VaR
        VaR = get_Var(x_sol,c_test,alpha,task_name)
        # save the VaR
        np.savetxt(save_dir+"VaR.csv",VaR,delimiter=",")

        # calculate the coverage
        coverage = in_kNN_ellipsoid(save_dir,c_test)
        coverage = np.mean(coverage,axis=1)
        # save the coverage
        np.savetxt(save_dir+"coverage.csv",coverage,delimiter=",")

print("cluster, done!")

"""
        def fun(P_rho):
            # P_rho is a 1D array, convert it to a 2D array
            P_rho = P_rho.reshape(-1,dim_c+1)

            dim = P_rho.shape[0]
            P = P_rho[:,:dim]
            (sign, logdet) = np.linalg.slogdet(scipy.linalg.inv(P))
            return logdet
        def cons_fun(P_rho):
            # P_rho is a 1D array, convert it to a 2D array
            P_rho = P_rho.reshape(-1,dim_c+1)
            dim = P_rho.shape[0]
            P = P_rho[:,:dim]
            rho = P_rho[:,dim]
            ppoints = points.transpose()
            # calculate the 2 norm of (P @ point +rho) for each point with scipy package
            norms = np.zeros(ppoints.shape[1])
            for point_idx in range(ppoints.shape[1]):
                norms[point_idx] = np.linalg.norm(P @ ppoints[:,point_idx] + rho)
            return norms
        cons = scipy.optimize.NonlinearConstraint(cons_fun, np.zeros(points.shape[0]), ub = np.ones(points.shape[0]))
        init_P = np.cov(points.transpose())
        init_rho = np.mean(points,axis=0)
        init_P_rho = np.concatenate((init_P,init_rho.reshape(-1,1)),axis=1)
        # convert init_P_rho to a 1D array
        init_P_rho = init_P_rho.reshape(-1)

        res = minimize(fun, init_P_rho, method='trust-constr', constraints=cons, options={'verbose': 1})
        sol = res.x
        # convert sol to a 2D array
        P_rho = sol.reshape(-1,dim_c+1)
        dim = P_rho.shape[0]
        P = P_rho[:,:dim]
        rho = P_rho[:,dim]
"""