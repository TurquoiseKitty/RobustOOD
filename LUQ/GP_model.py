# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 21:12:52 2022

@author: Admin
"""
import numpy as np
import torch
import gpytorch
import GPy


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1],has_lengthscale=True))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.shape[1]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

def GPygp(cov_samples,cov,demand_samples,alpha):
    train_x = cov_samples.transpose()
    train_y = np.asmatrix(demand_samples).T
    num_dims = train_x.shape[1]
    
    kernel = GPy.kern.RBF(input_dim=num_dims,ARD=True) + GPy.kern.Bias(input_dim=num_dims)
    model = GPy.models.GPRegression(train_x,train_y,kernel)
    model.optimize()
    
    test_x = cov.reshape(-1,num_dims)
    model.predict(test_x)
    solutions = model.predict_quantiles(test_x,alpha)
    return solutions

def GPmulti(train_X,train_Y):
    training_iterations = 200
    train_x = torch.Tensor(train_X)
    train_y = torch.Tensor(train_Y)
    num_dims = train_x.shape[1]
    num_tasks = train_y.shape[1]
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        
    model.eval()
    likelihood.eval()
    return model,likelihood

def GP(cov_samples,demand_samples):
    train_x = torch.Tensor(cov_samples) #transpose to n*d
    train_y = torch.Tensor(demand_samples).view(-1)
    num_dims = train_x.shape[1]


    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    training_iter=100

    cur_loss = 1000
    loop_count = 0
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.sum().backward()
        """
        print(
            i + 1, training_iter, loss.mean().item()
        )
        """
        last_loss = cur_loss
        cur_loss = loss.mean()
        optimizer.step()
        """
        if loop_count>1 and (last_loss-cur_loss)/cur_loss<0.01:
            print("iter_num=",loop_count+1,"length_scale=",model.covar_module.base_kernel.lengthscale.data)
            break
        loop_count+=1
        """
        
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    """
    kf = model.covar_module(test_x,train_x).to_dense()
    #print(kf)
    Kff = model.covar_module(train_x,train_x).to_dense()
    #print(Kff)
    
    f_preds = model(test_x)
    y_preds = likelihood(model(test_x))
    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    
    weights = torch.mm(kf,torch.linalg.inv(Kff))
    weights = weights.detach().numpy()
    return weights,f_mean.detach().numpy(),f_var.detach().numpy()
    """
    return model
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    #with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #    observed_pred = likelihood(model(test_x))

def groupGP_train(train_X,train_Y):
    GPlist = []
    num_tasks = train_Y.shape[1]
    for i in range(num_tasks):
        model = GP(train_X,train_Y[:,i])
        GPlist.append(model)
    return GPlist

def groupGP_test(GPlist,test_X):
    test_means = np.zeros([test_X.shape[0],len(GPlist)])
    test_vars = np.zeros(test_means.shape)
    
    for i in range(len(GPlist)):
        y_preds = GPlist[i](torch.Tensor(test_X))
        test_means[:,i] = y_preds.mean.detach().numpy()
        test_vars[:,i] = y_preds.variance.detach().numpy()
        
    return test_means,test_vars

def NW(ker,bandwidth,cov_samples,cov):
    num_samples = cov_samples.shape[1]
    cov = cov.reshape([-1,1])

    if ker=='exp':
        theta = ((cov_samples-cov.reshape([-1,1]).repeat(axis=1,repeats=num_samples))**2/bandwidth)**2
        theta = np.sqrt(np.sum(theta,axis=0))
        weights = np.exp(-theta)
        
    normalized_weights = weights/np.sum(weights)
    return normalized_weights