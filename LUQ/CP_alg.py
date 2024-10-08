# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:19:58 2023

@author: Admin
"""
import numpy as np
import ipdb

def get_q_with_betas(alpha,calibrate_X,calibrate_Y,calibrate_pred_Y,betas):
    n = calibrate_X.shape[0]
    X_dims = calibrate_X.shape[1]
    etass = np.abs(calibrate_pred_Y-calibrate_Y)/(np.matmul(calibrate_X,betas[:X_dims,:])+np.repeat(betas[X_dims,:][np.newaxis,:],calibrate_X.shape[0],axis=0))
    #need to check the sign of s (should be positive)
    
    etas = np.max(etass,axis=1)
    etas = np.sort(etas)
    pos = np.ceil(alpha*(n+1))-1
    
    
    eta = etas[int(pos)]
    
    return eta

def get_q(alpha,calibrate_Y,calibrate_pred_Y,res_pred):

    n = calibrate_Y.shape[0]
    
    etass = np.abs(calibrate_pred_Y-calibrate_Y)/res_pred
    #need to check the sign of s (should be positive)
    
    etas = np.max(np.abs(etass),axis=1)
    etas = np.sort(etas)
    pos = np.ceil(alpha*(n+1))-1
    
    
    eta = etas[int(pos)]
    
    return eta

def get_r(alpha,res_cal,res_cal_2norm_pred,cov):
    n = res_cal.shape[0]

    # scale the residuals
    res_cal_scaled = res_cal/res_cal_2norm_pred.reshape(-1,1)


    # if cov has only one dimension, then convert it to two dimensions
    if len(cov.shape)==0:
        cov = np.array([[cov[0]]])

    #inverse cov
    cov_inv = np.linalg.inv(cov)

    # decompose cov_inv to inv_L*inv_L^T
    inv_L = np.linalg.cholesky(cov_inv)

    
    # get the distance
    distances = np.linalg.norm(res_cal_scaled @ inv_L, axis=1)

    # get the alpha percentile of distances as r
    distances = np.sort(distances)
    pos = np.ceil(alpha*(n+1))-1
    r = distances[int(pos)]


    return r

