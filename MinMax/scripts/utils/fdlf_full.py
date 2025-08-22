
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 22:28:48 2023

@author: rnelli

State estimation to learn B1 and B2....


"""
import numpy as np
import pandas as pd
import torch 
import time

from Support_Files.CreateDataset import create_data, create_test_data
from EarlyStopping import EarlyStopping
import random


from Support_Files.create_example_parameters import create_example_parameters
import string
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def delete(arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
    skip = [i for i in range(arr.size(dim)) if np.all(i != ind)]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)

def mult_list_array(list_, arr,length) -> torch.Tensor:
    res=0
    for i in range(length):
        res+= arr[i]*list_[i]
    return res
n_buses=118
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_iter=7

Sweep_ID = get_random_string(8)

N_train_iter_list = [2000,3000]
batch_size_list  = [1000]
lr_list = [0.001,0.002,0.005,0.01,0.05,0.1]
gamma_list = [0.995,0.99,0.985,0.98]

num_hyper_tunning = 10
Hyper_parameters = pd.DataFrame(columns = ['N_train_iter','batch_size','lr','gamma','BestTestError'],
                  index = range(num_hyper_tunning))

for hyper_ID in range(num_hyper_tunning):
    
    
    N_train_iter = random.choice(N_train_iter_list)
    batch_size = random.choice(batch_size_list)
    lr  = random.choice(lr_list)
    gamma  = random.choice(gamma_list)
    
    Hyper_parameters.loc[hyper_ID] = [N_train_iter,batch_size,lr,gamma,100]
      
    
    Hyper_parameters.to_csv('Results/Hyper_parameters_'+str(n_buses)+Sweep_ID
                +'.csv', index=False)
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
    
    
    Ntraining = Psp_train.shape[0]
    Psp_train = torch.tensor(np.transpose(Psp_train)).to(device)
    Qsp_train = torch.tensor(np.transpose(Qsp_train)).to(device)
    
    
    Ntesting = Psp_test.shape[0]
    Psp_test = torch.tensor(np.transpose(Psp_test)).to(device)
    Qsp_test = torch.tensor(np.transpose(Qsp_test)).to(device)
    
    optimizer = torch.optim.Adam([B1,B2],lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # Starting the iteration
    
    TrainingError = np.ones(N_train_iter)
    TestError = np.ones(N_train_iter)
    TimeStanmbs = np.ones((N_train_iter,7))
    
    
    
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    pathB1='Results/B1_trained_'+str(n_buses)+Sweep_ID\
        + '_hyperID_' + str(hyper_ID) \
        +'.csv'
        
    pathB2='Results/B2_trained_'+str(n_buses)+Sweep_ID\
        + '_hyperID_' + str(hyper_ID) \
        +'.csv'
    
    early_stopping = EarlyStopping(patience=500, verbose=False, pathB1 = pathB1, pathB2= pathB2)
    
    Best_test_loss = [100]
    Loss_weights= torch.tensor(np.logspace(-4, 0, num=n_iter, base=5).astype(float))
    for train_iter in range(N_train_iter):
        iteration_start=time.time()
        TimeStanmbs[train_iter,0]=iteration_start
        
        
        B1_inv = torch.linalg.inv(torch.mul(B1.to_dense(),MaskB1))
        B2_inv = torch.linalg.inv(torch.mul(B2.to_dense(),MaskB2))
        
        TimeStanmbs[train_iter,1] = time.time()-iteration_start
        
        num_batches_train = Ntraining//batch_size
        
        Tol_train = 0
        for i in range(num_batches_train):
        
            slce = get_slice(i, batch_size)
            
            Psp_train_batch = Psp_train[:,slce] 
            Qsp_train_batch = Qsp_train[:,slce] 
        
            N_batch = Psp_train_batch.shape[1]
            
           # Training  
            optimizer.zero_grad
            j=0
            
            P={}
            Q={}
            V={}
            delta = {}
            V[0] = torch.tensor(np.ones((n_buses,N_batch))).to(device)
            delta[0] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
            
            Tol_iter = {}
            for i in range(0,n_iter):
                
                P[i] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
                Q[i] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
                start_time = time.time()
                
                V_cpmlx = V[i]*torch.exp(1j*delta[i])
                
                Scal = V_cpmlx*torch.conj(Ybus@V_cpmlx)
                P[i] = Scal.real
                Q[i] = Scal.imag
                
      
                TimeStanmbs[train_iter,2] = time.time()-start_time         
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
                TimeStanmbs[train_iter,3] = time.time()-start_time
                
                # print('time take for solving the equation =' +str(time.time()-start_time))
                V[i+1] = torch.tensor(np.ones((n_buses,N_batch))).to(device)
                delta[i+1] = torch.tensor(np.zeros((n_buses,N_batch))).to(device)
                
                delta[i+1][pv_pq_bus-1,:] = delta[i][pv_pq_bus-1,:]+x
                V[i+1][pq_bus-1,:] = V[i][pq_bus-1,:] +delv
                Tol_iter[i] = torch.mean(torch.max(torch.abs(torch.concatenate((dmp,dmq),axis=0)),axis=0).values)/num_batches_train


            Tol_train =+ mult_list_array(Tol_iter,Loss_weights,n_iter)            

        Tol_change = torch.mean(torch.abs(B1 -B1_init)) + torch.mean(torch.abs(B2 -B2_init))
        
        Loss = Tol_train + Tol_change*0.001   
        TimeStanmbs[train_iter,4] = time.time() - iteration_start
        Loss.backward()
        TimeStanmbs[train_iter,5] = time.time() - iteration_start
        optimizer.step()
        TimeStanmbs[train_iter,6] = time.time() - iteration_start
        
        
        # Testing
        
        P={}
        Q={}
        V={}
        delta = {}
        V[0] = torch.tensor(np.ones((n_buses,Ntesting))).to(device)
        delta[0] = torch.tensor(np.zeros((n_buses,Ntesting))).to(device)
        
        for i in range(0,n_iter):
            P[i] = torch.tensor(np.zeros((n_buses,Ntesting))).to(device)
            Q[i] = torch.tensor(np.zeros((n_buses,Ntesting))).to(device)
            
            
            V_cpmlx = V[i]*torch.exp(1j*delta[i])
            
    
            Scal = V_cpmlx*torch.conj(Ybus@V_cpmlx)
            P[i] = Scal.real
            Q[i] = Scal.imag
            
           
            delp=P[i]-Psp_test;
            delq=Q[i]-Qsp_test;
            dmp=delp[pv_pq_bus-1,:]
            dmq=delq[pq_bus[0]-1,:]

            dP=torch.divide(delp,V[i]);
            dQ=torch.divide(delq,V[i]);
            dP=dP[pv_pq_bus-1,:]
            dQ=dQ[pq_bus[0]-1,:]   
            start_time = time.time()
            
            x =-B1_inv@dP
            delv = -B2_inv@dQ
            
            V[i+1] = torch.tensor(np.ones((n_buses,Ntesting))).to(device)
            delta[i+1] = torch.tensor(np.zeros((n_buses,Ntesting))).to(device)
            
            delta[i+1][pv_pq_bus-1,:] = delta[i][pv_pq_bus-1,:]+x
            V[i+1][pq_bus-1,:] = V[i][pq_bus-1,:] +delv

        
        Tol_test=torch.mean(torch.max(torch.abs(torch.concatenate((dmp,dmq),axis=0)),axis=0).values)
       
        TrainingError[train_iter] = Tol_train.detach().cpu().numpy()
        TestError[train_iter] = Tol_test.detach().cpu().numpy()
        
        if Best_test_loss[0] > TestError[train_iter]:
            Best_test_loss[0] = TestError[train_iter]
  
            Hyper_parameters.loc[hyper_ID] = [N_train_iter,batch_size,lr,gamma,Best_test_loss[0]]
            # Hyper_parameters.to_csv('/work3/rnelli/FDLF/Results/Hyper_parameters_'+str(n_buses)+Sweep_ID
            #            +'.csv', index=False)
            
        print(Tol_test.detach().cpu().numpy())
        # print(Tol_train.detach().cpu().numpy())

        np.savetxt('Results/TestError_'+str(n_buses)+Sweep_ID
                    + '_hyperID_' + str(hyper_ID)
                    +'.csv', TestError, delimiter=',')
        np.savetxt('Results/TrainError_'+str(n_buses) +Sweep_ID
                    + '_hyperID_' + str(hyper_ID)
                    + '.csv', TrainingError, delimiter=',')

        
        scheduler.step()
    
        early_stopping(Tol_test, [B1,B2])
        
        if early_stopping.early_stop:
            np.savetxt('Results/TimeStanmbs'+str(n_buses)+'.csv', TimeStanmbs, delimiter=',')
            print("Early stopping")
            break 
