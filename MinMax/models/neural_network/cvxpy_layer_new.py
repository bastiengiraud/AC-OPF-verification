# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 22:41:41 2021

@author: rnelli
"""
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch

def cvxpy_layer(n_lod,n_h,n_g,n_hid_l, Data_stat, Z_min,Z_max,Pg_min,Pg_max):
    
    Gen_delta=torch.tensor(Data_stat['Gen_delta'].reshape(-1,))
    Dem_min=torch.tensor(Data_stat['Dem_min'].reshape(-1,))
    Dem_delta=torch.tensor(Data_stat['Dem_delta'].reshape(-1,))
    Pg_min = torch.tensor((Pg_min).reshape(-1,))
    Pg_max = torch.tensor((Pg_max).reshape(-1,))


    Pg_min = torch.tensor((Pg_min).reshape(-1,))
    Pg_max = torch.tensor((Pg_max).reshape(-1,))

    N_nd_p_lyr = [n_lod] + [n_h] * n_hid_l + [n_g] + [n_g] + [n_g]
    N_layers = len(N_nd_p_lyr)-1
    N_ReLU_layers = len(N_nd_p_lyr)-2

    W={}
    b={}

    for i in range(n_hid_l+1):
        W[i] = cp.Parameter((N_nd_p_lyr[i+1],N_nd_p_lyr[i]))
        b[i] = cp.Parameter(N_nd_p_lyr[i+1]) 

    # W[0] = W1
    # b[0] = b1

    # W[1] = W2
    # b[1] = b2
    
    # W[2] = W3
    # b[2] = b3

    # W[3] = W4
    # b[3] = b4   

    W[n_hid_l+1] = torch.tensor(np.eye(n_g, dtype=float)*(-1))
    b[n_hid_l+1] = torch.tensor(np.ones(n_g))

    W[n_hid_l+2] = torch.tensor(np.eye(n_g, dtype=float)*(-1))
    b[n_hid_l+2] = torch.tensor(np.ones(n_g))

    x_min = {}
    x_max = {}
    y_min = {}
    y_max = {}

    x_min[0] = 0
    x_max[0] = 1

    for i in range(n_hid_l):
        y_min[i] = Z_min[:,i]
        y_max[i] = Z_max[:,i]
        
        x_min[i+1] = 0
        x_max[i+1] = Z_max[:,i]
    
    y_min[n_hid_l] = Pg_min
    y_max[n_hid_l] = Pg_max

    x_min[n_hid_l+1] = 0
    x_max[n_hid_l+1] = Pg_max

    y_min[n_hid_l+1] = 1 - Pg_max
    y_max[n_hid_l+1] = np.ones(n_g)

    x_min[n_hid_l+2] = 0
    x_max[n_hid_l+2] = 1
    
    y_min[n_hid_l+2] = 0
    y_max[n_hid_l+2] = 1

    x={}
    y_hat={}
    ReLU_state={}

    for i in range(N_layers):
        x[i] = cp.Variable(N_nd_p_lyr[i])
        y_hat[i] = cp.Variable(N_nd_p_lyr[i+1])

    for i in range(N_ReLU_layers):
        ReLU_state[i] = cp.Variable(N_nd_p_lyr[i+1],nonneg=True)
    
    constraints = [x[0]>=0]
    for i in range(N_layers):
        constraints += [ x[i] >= x_min[i] , x[i] <= x_max[i]]
        constraints += [ y_hat[i]>= y_min[i] , y_hat[i] <= y_max[i]]
    
    for i in range(N_ReLU_layers):
        constraints += [ ReLU_state[i] >= 0 , ReLU_state[i] <= 1]


    for i in range(N_layers):
        constraints += [y_hat[i] - W[i] @ x[i] -b[i] == 0]

        # Hidden layers

    for i in range(N_ReLU_layers):
        constraints += [x[i+1] - y_hat[i] - cp.multiply(ReLU_state[i],y_min[i]) + y_min[i]<= 0]
        constraints += [x[i+1] - y_hat[i]>= 0]
        constraints += [x[i+1] - cp.multiply(ReLU_state[i],y_max[i])<= 0]

    i = n_hid_l

    #constraints += [ReLU_state[i] + ReLU_state[i+1] >= 1]

    i_out = N_layers-1

    objective_p = cp.Maximize(sum(cp.multiply(Gen_delta,y_hat[i_out]))- sum(Dem_min + cp.multiply(Dem_delta,x[0])))
    objective_n = cp.Maximize(-sum(cp.multiply(Gen_delta,y_hat[i_out])) + sum(Dem_min + cp.multiply(Dem_delta,x[0])))

    problem_1 = cp.Problem(objective_p, constraints)  
    problem_2 = cp.Problem(objective_n, constraints)  

    cvxpylayer_1 = CvxpyLayer(problem_1, parameters=[W[0],W[1],W[2],W[3],b[0],b[1],b[2],b[3]], variables=[x[0]])
    cvxpylayer_2 = CvxpyLayer(problem_2, parameters=[W[0],W[1],W[2],W[3],b[0],b[1],b[2],b[3]], variables=[x[0]])

    return cvxpylayer_1,cvxpylayer_2  