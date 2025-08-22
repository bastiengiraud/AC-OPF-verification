# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:41:41 2021

@author: rnelli
"""
import cvxpy as cp
import numpy as np

def cvxpy(n,m,k,n_h, Data_stat, Z_min,Z_max,W1,W2,W3,W4,b1,b2,b3,b4):
    
    Gen_delta=(Data_stat['Gen_delta'].reshape(-1,))
    Dem_min=(Data_stat['Dem_min'].reshape(-1,))
    Dem_delta=(Data_stat['Dem_delta'].reshape(-1,))

    # Z_min = cp.Parameter((m,n_h),value=Z_min)
    # Z_max = cp.Parameter((m,n_h),value=Z_max)
    
    # n, m ,k= 21, 20,10
    x = cp.Variable(n)
    Z = cp.Variable((m,n_h),nonneg=True)
    Z_hat = cp.Variable((m,n_h))
    ReLU_stat = cp.Variable((m,n_h),nonneg=True)
    pg_pre = cp.Variable(k)
    
    # W1 = cp.Parameter(W1)
    # b1 = cp.Parameter(b1)
    # W2 = cp.Parameter(W2)
    # b2 = cp.Parameter(b2)
    # W3 = cp.Parameter(W3)
    # b3 = cp.Parameter(b3)
    # W4 = cp.Parameter(W4)
    # b4 = cp.Parameter(b4)   
    
    
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


    obj=cp.Variable(1)
    PF_p = sum(cp.multiply(Gen_delta,pg_pre)) - sum(Dem_min + cp.multiply(Dem_delta,x))
    PF_n = -sum(cp.multiply(Gen_delta,pg_pre)) + sum(Dem_min + cp.multiply(Dem_delta,x))
    # constraints += [obj - PF_p >= 0]
    # constraints += [obj - PF_n >= 0]    

    # w = cp.Variable(1)
    # constraints = [ w >= 0 , w <= 1]
    # constraints += [obj - PF_p - 10**6*w <= 0 ]
    # constraints += [obj - PF_n - 10**6*(1-w) <= 0 ]

    objective = cp.Maximize(sum(cp.multiply(Gen_delta,pg_pre))- sum(Dem_min + cp.multiply(Dem_delta,x)))
    
    objective = cp.Maximize(PF_n)


    problem = cp.Problem(objective, constraints)  
    
    print(problem.solve())
    return objective  