'''Calculate the weights of samples by using kernel methods'''

import numpy as np
import scipy
import scipy.stats

def get_weights(X,x,kernel_name,param):
    '''
    Calculate the weights of samples by using kernel methods
    '''
    num_samples = X.shape[0]
    dim_covs = X.shape[1]
    # calculate the distance between x and each sample in X using Mahalanobis distance metric
    dist = np.zeros(num_samples)
    # calculate the inverse of covariance matrix
    X_inv = np.linalg.inv(np.cov(X.T))
    for i in range(num_samples):
        dist[i] = np.sqrt(np.matmul(np.matmul((x-X[i,:]),X_inv),(x-X[i,:]).T))
    

    weights = np.zeros(num_samples)
    if kernel_name == 'gaussian':
        bandwidth = param*np.power(num_samples,-1/(4+dim_covs))
        for i in range(num_samples):
            weights[i] = scipy.stats.norm.pdf(dist[i],0,bandwidth)

    elif kernel_name == 'uniform':
        bandwidth = param*np.power(num_samples,-1/(4+dim_covs))
        for i in range(num_samples):
            if dist[i] <= bandwidth:
                weights[i] = 1
            else:
                weights[i] = 0
    elif kernel_name == 'kNN':
        k = param
        # find the k nearest neighbors
        idx = np.argsort(dist)
        for i in range(k):
            weights[idx[i]] = 1
    else:
        raise Exception('Invalid kernel name!')
    
    # normalize the weights
    weights = weights/np.sum(weights)


    return weights