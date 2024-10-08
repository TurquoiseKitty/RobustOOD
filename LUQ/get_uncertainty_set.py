
import numpy as np


def get_box_US_with_betas(test_X,test_pred_Y,betas,q):
    # Get the residuals for each dimensions, which are test_X*betas
    X_dims = test_X.shape[1]
    residuals = np.matmul(test_X,betas[:X_dims,:])+np.repeat(betas[X_dims,:][np.newaxis,:],test_X.shape[0],axis=0)
    # Get the uncertainty set
    LB = test_pred_Y - q*residuals
    UB = test_pred_Y + q*residuals

    # if LB > UB, then LB and UB just take test_pred_Y as the prediction
    LB = np.where(LB>UB,test_pred_Y,LB)
    UB = np.where(LB>UB,test_pred_Y,UB)

    return LB,UB

def get_box_US(test_pred_Y,test_resq,q):
    # Get the uncertainty set
    LB = test_pred_Y - q*test_resq
    UB = test_pred_Y + q*test_resq

    # if LB > UB, then LB and UB just take test_pred_Y as the prediction
    LB = np.where(LB>UB,test_pred_Y,LB)
    UB = np.where(LB>UB,test_pred_Y,UB)

    return LB,UB

def get_ellipsoid_US(test_pred_Y,res_test_2norm_pred,cov,r):
    # get the ellipsoid shape
    L = np.linalg.cholesky(cov)
    
    P = r*res_test_2norm_pred*L

    return test_pred_Y,P