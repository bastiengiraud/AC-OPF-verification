# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:41:41 2021

@author: rnelli
"""
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch

def cvxpy_layer(n,m,k,n_h, Data_stat, Z_min,Z_max):
    
    Gen_delta=torch.tensor(Data_stat['Gen_delta'].reshape(-1,))
    Dem_min=torch.tensor(Data_stat['Dem_min'].reshape(-1,))
    Dem_delta=torch.tensor(Data_stat['Dem_delta'].reshape(-1,))

    # Z_min = cp.Parameter((m,n_h),value=Z_min)
    # Z_max = cp.Parameter((m,n_h),value=Z_max)
    
    # n, m ,k= 21, 20,10
    
    x = cp.Variable(n)
    Z = cp.Variable((m,n_h),nonneg=True)
    Z_hat = cp.Variable((m,n_h))
    ReLU_stat = cp.Variable((m,n_h),nonneg=True)
    pg_pre = cp.Variable(k)
    
    W1 = cp.Parameter((m, n))
    b1 = cp.Parameter(m)
    W2 = cp.Parameter((m, m))
    b2 = cp.Parameter(m)
    W3 = cp.Parameter((m, m))
    b3 = cp.Parameter(m)
    W4 = cp.Parameter((k, m))
    b4 = cp.Parameter(k)   
    
    
    constraints = [ x >= 0 , x <= 1]
    constraints += [ ReLU_stat <= 1]
    
    
    # Hidden Layers
    constraints += [Z_hat[:,0] - W1 @ x -b1 == 0]    
    constraints += [Z_hat[:,1] - W2 @ Z[:,0] -b2 == 0]
    constraints += [Z_hat[:,2] - W3 @ Z[:,1] -b3 == 0]

    
    constraints += [pg_pre-W4 @ Z[:,2] + b4 == 0] 
    
    # # Hidden layers
    # for i in range(1,N_hid_l):
    #     constraints += [Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0]
    
    constraints += [Z - Z_hat - cp.multiply(ReLU_stat,Z_min) + Z_min <= 0]
    constraints += [Z - Z_hat>= 0]
    constraints += [Z - cp.multiply(ReLU_stat,Z_max) <= 0]
      
    objective_p = cp.Maximize(sum(cp.multiply(Gen_delta,pg_pre))- sum(Dem_min + cp.multiply(Dem_delta,x)))
    objective_n = cp.Maximize(-sum(cp.multiply(Gen_delta,pg_pre)) + sum(Dem_min + cp.multiply(Dem_delta,x)))

    problem_1 = cp.Problem(objective_p, constraints)  
    problem_2 = cp.Problem(objective_n, constraints)  

    cvxpylayer_1 = CvxpyLayer(problem_1, parameters=[W1,W2,W3,W4,b1,b2,b3,b4], variables=[x])
    cvxpylayer_2 = CvxpyLayer(problem_2, parameters=[W1,W2,W3,W4,b1,b2,b3,b4], variables=[x])
    return cvxpylayer_1,cvxpylayer_2  