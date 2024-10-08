# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:18:23 2023

@author: Admin
"""
import numpy as np
from scipy.stats import norm

def compute_VaR(gX,WX,x_sol,alpha):
    num_samples = x_sol.shape[0]
    mu = np.sum(gX*x_sol,axis=1)
    sigma = np.sqrt(np.sum(WX*x_sol**2,axis=1))
    
    f_inv = norm.ppf(alpha,loc=0,scale=1)
    Vs = f_inv*sigma+mu
    
    return Vs

def Obj2alpha(gX,WX,x_sol,RLP_objs):
    num_samples = x_sol.shape[0]
    mu = np.sum(gX*x_sol,axis=1)
    sigma = np.sqrt(np.sum(WX*x_sol**2,axis=1))
    
    f_inv = (RLP_objs-mu)/sigma
    
    alphas = norm.cdf(f_inv,loc=0,scale=1)
    
    return alphas

    