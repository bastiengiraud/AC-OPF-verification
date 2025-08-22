
import numpy as np
import pandas as pd
import torch 
import time


from Support_Files.CreateDataset import create_data, create_test_data
from Support_Files.create_example_parameters import create_example_parameters

n_buses=118
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_iter=7

N_batch=1
simulation_parameters = create_example_parameters(n_buses)

MaskB1 = simulation_parameters['true_system']['B1'].copy()
MaskB1[MaskB1 == 0] = False
MaskB1[MaskB1 != 0] = True

MaskB2 = simulation_parameters['true_system']['B2'].copy()
MaskB2[MaskB2 == 0] = False
MaskB2[MaskB2 != 0] = True

MaskB1 = torch.tensor(MaskB1).to(device)
MaskB2 = torch.tensor(MaskB2).to(device)

B1 = torch.tensor(simulation_parameters['true_system']['B1'].astype(float),requires_grad=True).to(device)
B2 = torch.tensor(simulation_parameters['true_system']['B2'].astype(float),requires_grad=True).to(device)


B1.retain_grad()
B2.retain_grad()

B1_init = torch.tensor(simulation_parameters['true_system']['B1'].astype(float)).to(device)
B2_init = torch.tensor(simulation_parameters['true_system']['B2'].astype(float)).to(device)

    
pv_bus = np.transpose(simulation_parameters['true_system']['pv_bus'])
pq_bus = np.transpose(simulation_parameters['true_system']['pq_bus'])

pv_pq_bus= np.append(pv_bus,pq_bus)

sl_bus = np.transpose(simulation_parameters['true_system']['sl_bus'])[0][0]

Ybus = torch.tensor(simulation_parameters['true_system']['Ybus']).to(device)

Psp_train, Qsp_train = create_data(simulation_parameters=simulation_parameters)
Psp_test, Qsp_test = create_test_data(simulation_parameters=simulation_parameters)

Psp_train = torch.tensor(np.transpose(Psp_train)).to(device)
Qsp_train = torch.tensor(np.transpose(Qsp_train)).to(device)

B1_inv = torch.linalg.inv(torch.mul(B1.to_dense(),MaskB1))
B2_inv = torch.linalg.inv(torch.mul(B2.to_dense(),MaskB2))


P={}
Q={}
V={}
delta = {}
V[0] = torch.tensor(np.ones((n_buses,N_batch))).to(device)
delta[0] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)

Psp_train_batch = Psp_train 
Qsp_train_batch = Qsp_train 


for i in range(0,n_iter):
    
    P[i] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
    Q[i] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
    start_time = time.time()
    
    V_cpmlx = V[i]*torch.exp(1j*delta[i])
    
    Scal = V_cpmlx*torch.conj(Ybus@V_cpmlx)
    P[i] = Scal.real
    Q[i] = Scal.imag
    

    delp=P[i]-Psp_train_batch;
    delq=Q[i]-Qsp_train_batch;
    dmp=delp[pv_pq_bus-1,:]
    dmq=delq[pq_bus[0]-1,:]

    dP=torch.divide(delp,V[i]);
    dQ=torch.divide(delq,V[i]);
    dP=dP[pv_pq_bus-1,:]
    dQ=dQ[pq_bus[0]-1,:]                

    
    x =-torch.mm(B1_inv,dP)
    delv = -torch.mm(B2_inv,dQ)

    
    # print('time take for solving the equation =' +str(time.time()-start_time))
    V[i+1] = torch.tensor(np.ones((n_buses,N_batch))).to(device)
    delta[i+1] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
    
    delta[i+1][pv_pq_bus-1,:] = delta[i][pv_pq_bus-1,:]+x
    V[i+1][pq_bus-1,:] = V[i][pq_bus-1,:] +delv

V_cpmlx = V[n_iter]*torch.exp(1j*delta[i])