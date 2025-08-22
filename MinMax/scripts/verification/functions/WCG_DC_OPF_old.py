
import gurobipy as gp
import numpy as np   
import pandas as pd
from multiprocessing import Pool, Array
import os
import itertools
import ctypes
from numpy import savetxt
import torch
from torch import nn
import time

from Neural_Network.create_example_parameters import create_example_parameters

def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)


def init_worker(shared_array_max,shared_array_min, shape):
    '''
    Initialize worker for processing:
    Create the numpy array from the shared memory Array for each process in the pool.
    '''
    global bound_max
    global bound_min

    bound_max = to_numpy_array(shared_array_max, shape)
    bound_min = to_numpy_array(shared_array_min, shape)

def MILP_WCG(py_init_sed,n_buses, W,b, Gen_delta,Gen_max,config):
    
    N_hid_l = len(W)-1
    
    N_lod= np.size(W[0],1)
    N_dns= np.size(W[0],0)
    N_gen= np.size(W[N_hid_l],0)

    simulation_parameters = create_example_parameters(n_buses)
    Dem_min=simulation_parameters['true_system']['Pd_min']
    Dem_delta=simulation_parameters['true_system']['Pd_delta']
    
    PTDF = simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float32)

    # Maping generators to bus
    Map_g = simulation_parameters['true_system']['Map_g'].astype(np.float32)
    # maping loads to buses
    Map_L = simulation_parameters['true_system']['Map_L'].astype(np.float32) 
    
    # Line limits
    Pl_max = simulation_parameters['true_system']['Pl_max'].astype(np.float32)


    
    # Parameters 

    Z_min_IB =np.ones((N_dns,N_hid_l))*(-10000)
    Z_max_IB =np.ones((N_dns,N_hid_l))*(10000)
    
    u_init = np.ones((N_lod,1))
    l_ini = 0* np.ones((N_lod,1))
       
    Z_max_IB[:,0] = (np.maximum(W[0], 0)@u_init 
                              + np.minimum(W[0], 0)@l_ini ).reshape((N_dns,)) + b[0]
    Z_min_IB[:,0]= (np.minimum(W[0], 0)@u_init 
                              + np.maximum(W[0], 0)@l_ini).reshape((N_dns,)) + b[0]
    for k in range(1,N_hid_l):
        Z_max_IB[:,k] = ((np.maximum(W[k], 0))@np.maximum(Z_max_IB[:,k-1], 0) + 
                              (np.minimum(W[k], 0))@np.maximum(Z_min_IB[:,k-1], 0)).reshape((N_dns,)) + b[k].reshape((N_dns,)) 
        Z_min_IB[:,k] = ((np.minimum(W[k], 0))@np.maximum(Z_max_IB[:,k-1], 0) 
                              + (np.maximum(W[k], 0))@np.maximum(Z_min_IB[:,k-1], 0)).reshape((N_dns,)) + b[k].reshape((N_dns,))
    
    Pg_hat_max_IB = ((np.maximum(W[N_hid_l], 0))@np.maximum(Z_max_IB[:,N_hid_l-1], 0) + 
                          (np.minimum(W[N_hid_l], 0))@np.maximum(Z_min_IB[:,N_hid_l-1], 0)).reshape((N_gen,)) + b[N_hid_l].reshape((N_gen,))
    
    Pg_hat_min_IB = ((np.minimum(W[N_hid_l], 0))@np.maximum(Z_max_IB[:,N_hid_l-1], 0) 
                          + (np.maximum(W[N_hid_l], 0))@np.maximum(Z_min_IB[:,N_hid_l-1], 0)).reshape((N_gen,)) + b[N_hid_l].reshape((N_gen,))

    
    
    Z_min = Z_min_IB
    Z_max= np.maximum(Z_max_IB,0)

    # ReLU BigM Bounds

    # for i in range(1,N_hid_l+1):
        
    #     paramlist = range(N_dns)
    #     bound_max_shared = Array(ctypes.c_float, N_dns , lock=False)
    #     bound_min_shared = Array(ctypes.c_float, N_dns , lock=False)

    #     pool = Pool(os.cpu_count(),initializer=init_worker, initargs=(bound_max_shared,bound_min_shared, (N_dns,1))) 
        
    #     bound_fun = Bound_pool(n_buses, W,b,i,ReLU_stability_inactive,ReLU_stability_active,Z_min,Z_max)
    #     pool.map(bound_fun, paramlist)
    #     Z_min[:,i-1:i] = to_numpy_array(bound_min_shared, (N_dns,1))
    #     Z_max[:,i-1:i] = to_numpy_array(bound_min_shared, (N_dns,1))
        

    BigM_bound = config.BigM_bound # 'MILP'
    
    if BigM_bound != 'None':
        
        Z_min,Z_max = interval_bound(n_buses, W,b, Gen_delta,Gen_max)
        print("ReLU Bounds -- Done")
        Z_max= np.maximum(Z_max,0)
        
    
    print("ReLU Bounds -- Done")
    
    ReLU_Stability_check = False
    if ReLU_Stability_check == True:
        # ReLU stability from Data
        ReLU_stability_active, ReLU_stability_inactive = ReLU_stability(n_buses,W,b)
    else:
        ReLU_stability_inactive = (Z_max_IB<=0)
        ReLU_stability_active = (Z_min_IB>0)
        
        

    
    if BigM_bound == 'MILP':
        already_done = True
        if already_done == False:

            Pg_hat_max, Pg_hat_min = Gen_bound(n_buses, W,b, Gen_delta,Gen_max,ReLU_stability_inactive,ReLU_stability_active,Z_min,Z_max)
            savetxt('Gen_max_MILP'+str(n_buses)+'.csv', Pg_hat_max, delimiter=',')
            savetxt('Gen_min_MILP'+str(n_buses)+'.csv', Pg_hat_min, delimiter=',')
        else: 
            Pg_hat_min = arr = np.loadtxt('Gen_min_MILP'+str(n_buses)+'.csv',delimiter=",")
            Pg_hat_max = arr = np.loadtxt('Gen_max_MILP'+str(n_buses)+'.csv',delimiter=",")
    elif BigM_bound == 'ABC':      
        Pg_hat_min = np.loadtxt("LowerBound_"+str(n_buses)+".csv",delimiter=",")
        Pg_hat_max = np.loadtxt("UpperBound_"+str(n_buses)+".csv",delimiter=",")
        
        Pg_hat_min[Pg_hat_min>0] = 0
        Pg_hat_max[Pg_hat_max<1] = 1
        
    else:
        Pg_hat_max = Pg_hat_max_IB
        Pg_hat_min = Pg_hat_min_IB   

    # paramlist = range(0,N_gen)
    
    # Pg_hat_max = Array(ctypes.c_double, Pg_hat_max_IB , lock=False)
    # Pg_hat_min = Array(ctypes.c_double, Pg_hat_min_IB , lock=False)

    # pool = Pool(os.cpu_count(),initializer=init_worker, initargs=(Pg_hat_max,Pg_hat_min, Pg_hat_max_IB.shape)) 
    # gen_bound = Gen_bound_pool(n_buses, W,b,ReLU_stability_inactive,ReLU_stability_active,Z_min,Z_max)
    # pool.map(gen_bound, paramlist)
    # Pg_hat_max = to_numpy_array(Pg_hat_max, Pg_hat_max_IB.shape)
    # Pg_hat_min = to_numpy_array(Pg_hat_min, Pg_hat_max_IB.shape)

    print("Generation Bounds -- Done")


    # making sure Pg_max < Pg_hat_max from interval bound
    Pg_hat_max[Pg_hat_max_IB<Pg_hat_max] = Pg_hat_max_IB[Pg_hat_max_IB<Pg_hat_max]
    Pg_hat_min[Pg_hat_min_IB>Pg_hat_min] = Pg_hat_min_IB[Pg_hat_min_IB>Pg_hat_min]

    ReLU_stability_non_negative= np.transpose(Pg_hat_min>=0)
    ReLU_stability_non_violating = np.transpose(Pg_hat_max<=1)


    Max_gcp_iter = config.Max_gcp_iter 
    
    for n_gcp_iter in range(0,Max_gcp_iter):
        if n_gcp_iter == 0:
            Ver_type = 'simple'
        else:
            Ver_type = 'GCP'

        W[N_hid_l+1] = np.eye(N_gen, dtype=float)*(-1)
        b[N_hid_l+1] = np.ones(N_gen)

        W[N_hid_l+2] = np.eye(N_gen, dtype=float)*(-1)
        b[N_hid_l+2] = np.ones(N_gen)

        N_node_per_layer = [N_lod] + [N_dns] * N_hid_l + [N_gen] + [N_gen] + [N_gen] 
        N_layers = len(N_node_per_layer)-1
        N_ReLU_layers = len(N_node_per_layer)-2
        # Dict to store min and max of x(results after ReLU)
        # and y ( result before ReLU )
        x_min = {}
        x_max = {}
        y_min = {}
        y_max = {}

        x_min[0] = 0
        x_max[0] = 1

        for i in range(N_hid_l):
            y_min[i] = Z_min[:,i]
            y_max[i] = Z_max[:,i]
            
            x_min[i+1] = 0
            x_max[i+1] = Z_max[:,i]
        
        y_min[N_hid_l] = Pg_hat_min
        y_max[N_hid_l] = Pg_hat_max       

        x_min[N_hid_l+1] = 0
        x_max[N_hid_l+1] = Pg_hat_max

        y_min[N_hid_l+1] = 1 - Pg_hat_max
        y_max[N_hid_l+1] = np.ones(N_gen)

        x_min[N_hid_l+2] = 0
        x_max[N_hid_l+2] = 1
        
        y_min[N_hid_l+2] = 0
        y_max[N_hid_l+2] = 1

        m = gp.Model()
        
        # Dict to store gurobi variable
        x={}
        y_hat={}
        ReLU_state={}

        x_start=m.addMVar(1,lb=0,ub=1,name="x_start")
        del_start = m.addMVar(N_node_per_layer[0],lb=-0.1,ub=0.1,name="del_start")
        
        for i in range(N_layers):
            x[i] = m.addMVar(N_node_per_layer[i],lb=x_min[i],ub=x_max[i],name="x")
            y_hat[i] = m.addMVar(N_node_per_layer[i+1],lb=y_min[i],ub=y_max[i],name="y_hat")
        
        for i in range(N_ReLU_layers):
            ReLU_state[i] = m.addMVar(N_node_per_layer[i+1],lb=0,ub=1,name="ReLU_state")
         
        
        P_line = m.addMVar((1,Pl_max.shape[1]),lb=-np.inf,name="Pl")
         

        for i in range(N_dns):
            for j in range(N_hid_l):
                if ReLU_stability_inactive[i][j] == True:
                    m.addConstr(ReLU_state[j][i]== False)
                if ReLU_stability_active[i][j] == True:
                    m.addConstr(ReLU_state[j][i]== True)

        for i in range(N_gen):
                if ReLU_stability_non_negative[i] == True:
                    m.addConstr(ReLU_state[N_hid_l][i]== True)
                if ReLU_stability_non_violating[i] == True:
                    m.addConstr(ReLU_state[N_hid_l+1][i]== True)


        corelated_input = config.corelated_input # 'correlated'
        
        if corelated_input == 'correlated':
            #input domain constrainint 
            m.addConstr(x[0] == x_start + del_start)

        # First layer
        for i in range(N_layers):
            m.addConstr(y_hat[i] - W[i] @ x[i] -b[i] == 0)

        # Hidden layers

        for i in range(N_ReLU_layers):
            m.addConstr(x[i+1] - y_hat[i] - ReLU_state[i]*y_min[i] + y_min[i]<= 0)
            m.addConstr(x[i+1] - y_hat[i]>= 0)
            m.addConstr(x[i+1] >= 0)
            m.addConstr(x[i+1] - ReLU_state[i]*y_max[i]<= 0)

        i = N_hid_l

        m.addConstr(ReLU_state[i] + ReLU_state[i+1] >= 1)

        i_out = len(N_node_per_layer)-2

        m.addConstr(((Gen_delta.reshape(1,N_gen)*y_hat[i_out].reshape(1,N_gen))@Map_g + (Dem_min.reshape(1,N_lod) + Dem_delta.reshape(1,N_lod)*x[0].reshape(1,N_lod))@Map_L )@PTDF == P_line)
        
        line_ID_max = config.Line_ID_max
        
        for line_ID in range(line_ID_max):
            
            m.setObjective(Pl_max[0,line_ID] - P_line[0,line_ID:line_ID+1])
    
            # # Relu MILP
            LP_obj = np.zeros(n_gcp_iter)
            ReLU_tighting_dict = {}
            if Ver_type == 'GCP':
                iteration = 0
    
                while iteration < n_gcp_iter :
                    # # Solving the LP
                    
                    m.optimize()
                    LP_obj[iteration] = -m.ObjVal
                    m_solution={}
                    savetxt('data_compresed.csv', LP_obj, delimiter=',')
                    m_solution['x']={}
                    m_solution['y_hat']={}
                    m_solution['ReLU_state']={}
    
                    for i in range(N_layers):
                        m_solution['x'][i] = x[i].x
                        m_solution['y_hat'][i] = y_hat[i].x
                    
                    for i in range(N_ReLU_layers):
                        m_solution['ReLU_state'][i] = ReLU_state[i].x
    
                    ReLU_tighting,I_dict,new_Z_min,new_Z_max = Get_GCP(n_buses, W,b,m_solution,ReLU_stability_inactive,ReLU_stability_active,x_min,x_max)
                    
                    ReLU_tighting_dict[iteration]=ReLU_tighting
                    
                    # ReLU tightening
    
                    for h in range(len(W)-1):
                        I=I_dict[h]
                        w=W[h]
                        L=new_Z_min[h]
                        U=new_Z_max[h]
                        bh = b[h]
    
                        xh = x[h]
                        xhp1 = x[h+1]
                        ReLU_state_h = ReLU_state[h]
                        NPL_h=N_node_per_layer[h]
                        start_inner = time.time()
                        
                        for d in range(N_node_per_layer[h+1]):
                            wd = w[d]
                            Id = I[d]
                            Ld = L[d]
                            Ud = U[d]
    
                            if ReLU_tighting[h][d] == 1:
                                XLZ = wd*(xh - Ld*(1-ReLU_state_h[d]))
                                UZ = wd*(Ud*ReLU_state_h[d])
                                m.addConstr(xhp1[d]-bh[d]*ReLU_state_h[d] <= gp.quicksum(XLZ[i] for i in range(NPL_h) if Id[i] == 1) + gp.quicksum(UZ[i] for i in range(NPL_h) if Id[i] != 1))
    
                        
                        time_taken = time.time() - start_inner 
                        print(time_taken)
                    iteration += 1
            
            for i in range(N_ReLU_layers):
                ReLU_state[i].vtype = gp.GRB.BINARY
                
            m.Params.MIPGap=0.01
            m.Params.TimeLimit = config.TimeLimit
            
            Adverse_attack = config.Adverse_attack # True
            
            if Adverse_attack == 'gd_attack_true' or Adverse_attack == 'Adverse_data_true':
                n_adver= 50
                if corelated_input == 'correlated':            
                    if Adverse_attack == 'gd_attack_true' :
                        ReLU_state_start = Adverse_data_points_correlated(n_buses,W,b,n_adver,Z_min,Z_max,Pg_hat_min,Pg_hat_max,line_ID,gradient_attak=True)
                    else:
                        ReLU_state_start = Adverse_data_points_correlated(n_buses,W,b,n_adver,Z_min,Z_max,Pg_hat_min,Pg_hat_max,line_ID,gradient_attak=False)        
                else:
                    if Adverse_attack == 'gd_attack_true' :
                        ReLU_state_start = Adverse_data_points(n_buses,W,b,n_adver,line_ID,gradient_attak=True)
                    else:
                        ReLU_state_start = Adverse_data_points(n_buses,W,b,n_adver,line_ID,gradient_attak=False)
               
                m.NumStart = n_adver
                m.update()
        
                # iterate over all MIP starts
                for s in range(m.NumStart):
                
                    # set StartNumber
                    m.params.StartNumber = s
        
                    # now set MIP start values using the Start attribute, e.g.:
                    for i in range(N_ReLU_layers):
                        ReLU_state[i].Start = ReLU_state_start[i][s,:]
    
            
    
            def mycallback(model, where):
                if where == gp.GRB.Callback.MIPNODE:
                  status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
                  if status == gp.GRB.OPTIMAL:
                      
                      rel = model.cbGetNodeRel(model._vars)
                      
                      m_vars={}
                      m_vars['x'] = {}
                      m_vars['y_hat'] = {}
                      m_vars['ReLU_state'] = {}
                      
                      m_solution={}
                      m_solution['x']={}
                      m_solution['y_hat']={}
                      m_solution['ReLU_state']={}
                      
                      start = N_node_per_layer[0] +1
                      
                      for i in range(N_layers):
                          x_sol_i = {}
                          x_var_i = {}
                          for j in range(N_node_per_layer[i]):
                              x_sol_i[j]=rel[start+j].x
                              x_var_i[j]=rel[start+j]
                              
                          m_solution['x'][i] = x_sol_i
                          m_vars['x'][i] = x_var_i
                          start += N_node_per_layer[i]
                          
                          y_hat_sol_i = {}
                          y_hat_var_i = {}
                          for j in range(N_node_per_layer[i+1]):
                              y_hat_sol_i[j] = rel[start+j].x
                              y_hat_var_i[j] = rel[start+j]
                              
                          m_solution['y_hat'][i] = y_hat_sol_i
                          m_vars['y_hat'][i] = y_hat_var_i
                          
                          start += N_node_per_layer[i+1]
                      
                      for i in range(N_ReLU_layers):
                          ReLU_state_sol_i={}
                          ReLU_state_var_i={}
                          for j in range(N_node_per_layer[i+1]):
                              ReLU_state_sol_i[j] = rel[start+j].x
                              ReLU_state_var_i[j] = rel[start+j]
                              
                          start += N_node_per_layer[i+1]
                          m_solution['ReLU_state'][i] = ReLU_state_sol_i
                          m_vars['ReLU_state'][i] = ReLU_state_sol_i
                    
                      ReLU_tighting,I_dict,new_Z_min,new_Z_max = Get_GCP(n_buses, W,b,m_solution,ReLU_stability_inactive,ReLU_stability_active,x_min,x_max)
               
                
                        # ReLU tightening
        
                      for h in range(len(W)-1):
                            I=I_dict[h]
                            w=W[h]
                            L=new_Z_min[h]
                            U=new_Z_max[h]
                            bh = b[h]
        
                            xh = m_vars['x'][h]
                            xhp1 =  m_vars['x'][h+1]
                            ReLU_state_h = m_vars['ReLU_state'][h]
                            NPL_h=N_node_per_layer[h]
                            start_inner = time.time()
                            
                            for d in range(N_node_per_layer[h+1]):
                                wd = w[d]
                                Id = I[d]
                                Ld = L[d]
                                Ud = U[d]
        
                                if ReLU_tighting[h][d] == 1:
                                    XLZ = wd*(xh - Ld*(1-ReLU_state_h[d]))
                                    UZ = wd*(Ud*ReLU_state_h[d])
                                    model.cbCut(xhp1[d]-bh[d]*ReLU_state_h[d] <= gp.quicksum(XLZ[i] for i in range(NPL_h) if Id[i] == 1) + gp.quicksum(UZ[i] for i in range(NPL_h) if Id[i] != 1))
    
    
            
            logfile = open('/work3/rnelli/GP_Log/Log_' + str(n_buses) + 'callback_'+str(config.callback)+ '_lineID'+ str(line_ID) +'_'+ str(py_init_sed) + '_' + corelated_input + '_' + BigM_bound + '_' + Ver_type + '_' + Adverse_attack + '_' +str(n_gcp_iter)+'.log', 'w')
            m.Params.LogFile = logfile.name
            
            if config.callback == True:
                m._vars = m.getVars()
                m.optimize(mycallback)        
            else:
                m.optimize()
    
            nSolutions = m.SolCount
            X_vals= np.zeros((nSolutions,N_lod))
            for sol in range(nSolutions):
                m.setParam(gp.GRB.Param.SolutionNumber, sol)
                X_vals[sol,:] = m.Xn[0:N_lod]
        
    return max(-m.ObjVal,0)


def interval_bound(n_buses, W,b, Gen_delta,Gen_max):
    
    N_hid_l = len(W)-1
    
    N_lod= np.size(W[0],1)
    N_dns= np.size(W[0],0)

    Z_min_IB =np.ones((N_dns,N_hid_l))*(-10000)
    Z_max_IB =np.ones((N_dns,N_hid_l))*(10000)
    
    u_init = np.ones((N_lod,1))
    l_ini = 0* np.ones((N_lod,1))
       
    Z_max_IB[:,0] = (np.maximum(W[0], 0)@u_init 
                              + np.minimum(W[0], 0)@l_ini ).reshape((N_dns,)) + b[0]
    Z_min_IB[:,0]= (np.minimum(W[0], 0)@u_init 
                              + np.maximum(W[0], 0)@l_ini).reshape((N_dns,)) + b[0]
    for k in range(1,N_hid_l):
        Z_max_IB[:,k] = ((np.maximum(W[k], 0))@np.maximum(Z_max_IB[:,k-1], 0) + 
                              (np.minimum(W[k], 0))@np.maximum(Z_min_IB[:,k-1], 0)).reshape((N_dns,)) + b[k].reshape((N_dns,)) 
        Z_min_IB[:,k] = ((np.minimum(W[k], 0))@np.maximum(Z_max_IB[:,k-1], 0) 
                              + (np.maximum(W[k], 0))@np.maximum(Z_min_IB[:,k-1], 0)).reshape((N_dns,)) + b[k].reshape((N_dns,))

    Z_max_IB = np.maximum(Z_max_IB,0)
    
    m = gp.Model()
    
    x = m.addMVar(N_lod,lb=0,ub=1,name="x")
    Z = m.addMVar((N_dns,N_hid_l),lb=0,ub=Z_max_IB,name="Z")
    Z_hat = m.addMVar((N_dns,N_hid_l),lb=Z_min_IB,ub=Z_max_IB,name="Z_hat")
    
    
    ReLU_stat = m.addMVar((N_dns,N_hid_l),vtype=gp.GRB.BINARY,name="ReLU_stat")
    
    # for i in range(N_dns):
    #     for j in range(N_hid_l):
    #         if ReLU_stability_inactive[i][j] == True:
    #             m.addConstr(ReLU_stat[i][j]== False)
    #         if ReLU_stability_active[i][j] == True:
    #             m.addConstr(ReLU_stat[i][j]== True)

    
    # First layer
    m.addConstr(Z_hat[:,0] - W[0] @ x -b[0] == 0)
    # Hidden layers
    for i in range(1,N_hid_l):
        m.addConstr(Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0)
    
    # # Relu MILP     
    m.addConstr(Z - Z_hat - ReLU_stat*Z_min_IB + Z_min_IB <= 0)
    m.addConstr(Z - Z_hat>= 0)
    m.addConstr(Z >= 0)
    m.addConstr(Z - ReLU_stat*Z_max_IB <= 0)
    
    Z_max_val=np.copy(Z_max_IB)
    Z_min_val=np.copy(Z_min_IB)
    for i in range(N_dns):
        for j in range(1,2):
            m.setObjective(Z_hat[i][j], gp.GRB.MINIMIZE)
            m.Params.TimeLimit = 5
            m.optimize()
            if m.Status != gp.GRB.INFEASIBLE or m.Status != gp.GRB.INF_OR_UNBD:
                Z_min_val[i][j] = max(m.ObjBound,Z_min_IB[i][j])
                Z_min_val[i][j] = min(Z_min_val[i][j],Z_max_IB[i][j])

            m.setObjective(Z_hat[i][j], gp.GRB.MAXIMIZE)
            m.Params.TimeLimit = 5
            m.optimize()
            if m.Status != gp.GRB.INFEASIBLE or m.Status != gp.GRB.INF_OR_UNBD:
                Z_max_val[i][j] = min(m.ObjBound,Z_max_IB[i][j])
                Z_max_val[i][j] = max(Z_max_val[i][j],Z_min_IB[i][j])
    
    return Z_min_val, Z_max_val

def Gen_bound(n_buses, W,b, Gen_delta,Gen_max,ReLU_stability_inactive,ReLU_stability_active,Z_min_IB,Z_max_IB):
    
    N_hid_l = len(W)-1
    
    N_lod= np.size(W[0],1)
    N_dns= np.size(W[0],0)
    N_gen= np.size(W[N_hid_l],0)
    
    m = gp.Model()
    
    x = m.addMVar(N_lod,lb=0,ub=1,name="x")
    Z = m.addMVar((N_dns,N_hid_l),lb=0,ub=Z_max_IB,name="Z")
    Z_hat = m.addMVar((N_dns,N_hid_l),lb=Z_min_IB,ub=Z_max_IB,name="Z_hat")
    pg_hat = m.addMVar(N_gen,lb=-np.inf, name="pg_pre")
    ReLU_stat = m.addMVar((N_dns,N_hid_l),vtype=gp.GRB.BINARY,name="ReLU_stat")
    
    for i in range(N_dns):
        for j in range(N_hid_l):
            if ReLU_stability_inactive[i][j] == True:
                m.addConstr(ReLU_stat[i][j]== False)
            if ReLU_stability_active[i][j] == True:
                m.addConstr(ReLU_stat[i][j]== True)

    
    # First layer
    m.addConstr(Z_hat[:,0] - W[0] @ x -b[0] == 0)
    # Hidden layers
    for i in range(1,N_hid_l):
        m.addConstr(Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0)
    
    # # Relu MILP     
    m.addConstr(Z - Z_hat - ReLU_stat*Z_min_IB + Z_min_IB <= 0)
    m.addConstr(Z - Z_hat>= 0)
    m.addConstr(Z >= 0)
    m.addConstr(Z - ReLU_stat*Z_max_IB <= 0)
    m.addConstr(pg_hat - W[N_hid_l] @ Z[:,N_hid_l-1] -b[N_hid_l] == 0)
    
    Pg_hat_max=np.ones(N_gen)*1000
    Pg_hat_min=np.zeros(N_gen)*(-1000)
    for i in range(N_gen):
        m.setObjective(pg_hat[i], gp.GRB.MINIMIZE)
        m.Params.TimeLimit = 10
        m.optimize()
        if m.Status != gp.GRB.INFEASIBLE or m.Status != gp.GRB.INF_OR_UNBD:
            Pg_hat_min[i] = min(m.ObjBound,Pg_hat_max[i])
        else:
            Pg_hat_min[i] = -np.inf

        m.setObjective(pg_hat[i], gp.GRB.MAXIMIZE)
        m.Params.TimeLimit = 10
        m.optimize()
        if m.Status != gp.GRB.INFEASIBLE or m.Status != gp.GRB.INF_OR_UNBD:
            Pg_hat_max[i] = max(m.ObjBound,Pg_hat_min[i])
        else:
            Pg_hat_max[i] =np.inf
    return Pg_hat_max, Pg_hat_min

def ReLU_stability(n_buses,W,b):    
    # Input_NN = pd.read_csv('./Data_File/'+str(n_buses)+'/NN_input.csv', header=None)   
    Input_NN = pd.read_csv('/work3/rnelli/Dataset/'+str(n_buses)+'/NN_input.csv', header=None) 
    # Input_NN = pd.read_csv('/home/rnellikkath3/TestCode/DC/Data_File/'+str(n_buses)+'/NN_input.csv', header=None)
    
    N_hid_l = len(W)-1
    
    N_sample = np.size(Input_NN,0)
    N_dns= np.size(W[0],0)
    ReLU_stat={}
    
    Zhat = Input_NN@ np.transpose(W[0]) + np.array(b[0]).reshape(1,-1)

    ReLU_stat[0] = np.ones(Zhat.shape) 
    ReLU_stat[0][Zhat<0] = 0
    Z = np.maximum(Zhat,0)
    for k in range(1,N_hid_l):
        Zhat = Z@ np.transpose(W[k]) + np.array(b[k]).reshape(1,-1)
        ReLU_stat[k] = np.ones(Zhat.shape) 
        ReLU_stat[k][Zhat<0] = 0
        Z = np.maximum(Zhat,0)


    ReLU_stat =dict_to_numpy(ReLU_stat)         
    ReLU_stability_active=np.transpose(np.sum(ReLU_stat,1)==N_sample);
    ReLU_stability_inactive=np.transpose(np.sum(ReLU_stat,1)==0);

    return ReLU_stability_active, ReLU_stability_inactive

def dict_to_numpy(dict):
    Len = len(dict)
    if Len > 0:
        array = np.ones((Len,dict[0].shape[0],dict[0].shape[1]))
    
    for l in range(Len):
        array[l,:,:] = dict[l]

    return array


def Adverse_data_points(n_buses,W,b,n_adver,line_ID,gradient_attak =True):    
    # Input_NN = np.array(pd.read_csv('./Data_File/'+str(n_buses)+'/NN_input.csv', header=None))   
    # Input_NN = np.array(pd.read_csv('/home/rnellikkath3/TestCode/DC/Data_File/'+str(n_buses)+'/NN_input.csv', header=None))
    
    Input_NN = np.array(pd.read_csv('/work3/rnelli/Dataset/'+str(n_buses)+'/NN_input.csv', header=None)) 
    
    N_hid_l = len(W)-1
    
    simulation_parameters = create_example_parameters(n_buses)

    Gen_delta=simulation_parameters['true_system']['Pg_delta'] 
    Dem_min=simulation_parameters['true_system']['Pd_min']
    Dem_delta=simulation_parameters['true_system']['Pd_delta']

    PTDF = simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float32)

    # Maping generators to bus
    Map_g = simulation_parameters['true_system']['Map_g'].astype(np.float32)
    # maping loads to buses
    Map_L = simulation_parameters['true_system']['Map_L'].astype(np.float32) 
    
    # Line limits
    Pl_max = simulation_parameters['true_system']['Pl_max'].astype(np.float32)

    N_sample = np.size(Input_NN,0)
    N_dns= np.size(W[0],0)
    Relu=np.zeros((N_sample,N_hid_l,N_dns))

    Zhat = Input_NN@ np.transpose(W[0]) + np.array(b[0]).reshape(1,-1)
    Z = np.maximum(Zhat,0)
    for k in range(1,N_hid_l):
        Zhat = Z@ np.transpose(W[k]) + np.array(b[k]).reshape(1,-1)
        Z = np.maximum(Zhat,0)

    G_hat = Z@ np.transpose(W[N_hid_l]) + np.array(b[N_hid_l]).reshape(1,-1)
    G_hat = np.maximum(G_hat,0)
    G_hat = np.minimum(G_hat,1)

    # ((Gen_delta[:,0].reshape(1,N_gen)*G_hat[i_out].reshape(1,N_gen))@Map_g + (Dem_min.reshape(1,N_lod) + Dem_delta.reshape(1,N_lod)*x[0].reshape(1,N_lod))@Map_L )@PTDF == P_line)
    Pline=((Gen_delta.reshape(1,-1)*G_hat)@Map_g+(Dem_min.reshape(1,-1) + Dem_delta.reshape(1,-1)*Input_NN)@Map_L)@PTDF
    
    # PB = np.sum(G_hat*Gen_delta,1) - np.sum(Dem_min.reshape(1,-1)  + Input_NN*Dem_delta.reshape(1,-1),1)
    
    ind=np.argpartition(-Pline[:,line_ID], -4)[-n_adver:]

    Adverse_example = Input_NN[ind][:]

    if gradient_attak == True:
        Adverse_example = gradient_attaks(n_buses,W,b,Adverse_example)

    ReLU_stat={}
    Zhat = Adverse_example@ np.transpose(W[0]) + np.array(b[0]).reshape(1,-1)
    
    ReLU_stat[0] = np.ones(Zhat.shape) 
    ReLU_stat[0][Zhat<0] = 0
    Z = np.maximum(Zhat,0)
    for k in range(1,N_hid_l):
        Zhat = Z@ np.transpose(W[k]) + np.array(b[k]).reshape(1,-1)
        ReLU_stat[k] = np.ones(Zhat.shape) 
        ReLU_stat[k][Zhat<0] = 0
        Z = np.maximum(Zhat,0)
        
    G_hat = Z@ np.transpose(W[N_hid_l]) + np.array(b[N_hid_l]).reshape(1,-1)
    ReLU_stat[N_hid_l] = np.ones(G_hat.shape)
    ReLU_stat[N_hid_l][G_hat<0] = 0

    ReLU_stat[N_hid_l+1] = np.ones(G_hat.shape)
    ReLU_stat[N_hid_l+1][G_hat>1] = 0

    G_pred = np.maximum(np.minimum(G_hat,1),0)

    PB = np.sum(G_pred*Gen_delta,1) - np.sum(Dem_min.reshape(1,-1)  + Adverse_example*Dem_delta.reshape(1,-1),1)

    Adv_sol=ReLU_stat

    # Adv_sol['ReLU_stat'] = ReLU_stat
    # Adv_sol['x'] = Adverse_example
    # Adv_sol['Z'] = Z
    # Adv_sol['Z_hat'] = Zhat
    # Adv_sol['G_hat'] = G_hat
    # Adv_sol['PB'] = PB


    return Adv_sol

def gradient_attaks(n_buses,W,b,X_input):
    
    N_hid_l = len(W)-1
    N_dns= np.size(W[0],0)

    N_gen = b[N_hid_l].shape[0]
    simulation_parameters = create_example_parameters(n_buses)

    Gen_delta=torch.tensor(( simulation_parameters['true_system']['Pg_delta']).astype(np.float32))
    Dem_min=torch.tensor((simulation_parameters['true_system']['Pd_min']).reshape(1,-1).astype(np.float32))
    Dem_delta=torch.tensor((simulation_parameters['true_system']['Pd_delta']).reshape(1,-1).astype(np.float32))


    x = torch.tensor(X_input.astype(np.float32), requires_grad=True)
    W_tensor ={}
    b_tensor ={}
    for i in range(N_hid_l+1):
        W_tensor[i]  = torch.tensor(np.transpose(W[i]).astype(np.float32))
        b_tensor[i]  = torch.tensor(np.array(b[i]).reshape(1,-1).astype(np.float32))
    Z={}
    Zhat = {}
    for i in range(N_hid_l):
        Z[i]  = torch.tensor(np.zeros((X_input.shape[0],N_dns)), requires_grad=True)
        Zhat[i]  = torch.tensor(np.zeros((X_input.shape[0],N_dns)), requires_grad=True)

    # Z = torch.tensor(np.zeros((N_hid_l,N_dns)).astype(np.float32), requires_grad=True)
    # Zhat = torch.tensor(np.zeros((N_hid_l,N_dns)).astype(np.float32), requires_grad=True)

    P_g_hat = torch.tensor(np.zeros((X_input.shape[0],N_gen)), requires_grad=True)
    P_g = torch.tensor(np.zeros((X_input.shape[0],N_gen)), requires_grad=True)

    optimizer = torch.optim.Adam([x],lr=0.001)

    for i in range(1000):
        optimizer.zero_grad()
        x=torch.tensor(x.detach(), requires_grad=True)
        
        
        Zhat[0] = x@ W_tensor[0] + b_tensor[0]
        ReLU = torch.nn.ReLU()
        Z[0] = ReLU(Zhat[0])
        
        for k in range(1,N_hid_l):
            Zhat[k] = Z[k-1]@W_tensor[k] + b_tensor[k]
            Z[k]  = ReLU(Zhat[k])
        
        clamp_function=Clamp(N_gen)

        P_g_hat = Z[N_hid_l-1]@W_tensor[N_hid_l] + b_tensor[N_hid_l]
        P_g = clamp_function(P_g_hat)

        # loss= torch.sum(P_g@Gen_delta,1) - (torch.sum(Dem_min)+x@Dem_delta)
        
        loss = torch.sum(torch.sum(P_g*Gen_delta,1) -  torch.sum(Dem_min + x*Dem_delta,1))
        loss.backward()
        # optimizer.step()
        

        x = x.clone().detach() - torch.tensor(0.001)*x.grad
        x = torch.maximum(x.clone().detach(),torch.tensor(0))
        x = torch.minimum(x.clone().detach(),torch.tensor(1))

    return x.detach().numpy()


def Adverse_data_points_correlated(n_buses,W,b,n_adver,Z_min,Z_max,Pg_hat_min,Pg_hat_max,gradient_attak =True):
    # Input_NN = np.array(pd.read_csv('./Data_File/'+str(n_buses)+'/NN_input.csv', header=None))   
 
    Input_NN = np.array(pd.read_csv('/work3/rnelli/Dataset/'+str(n_buses)+'/NN_input.csv', header=None)) 

    N_hid_l = len(W)-1

    simulation_parameters = create_example_parameters(n_buses)

    Gen_delta=simulation_parameters['true_system']['Pg_delta'] 
    Dem_min=simulation_parameters['true_system']['Pd_min']
    Dem_delta=simulation_parameters['true_system']['Pd_delta']    
    
    N_sample = np.size(Input_NN,0)

    np.random.seed(0)
    mean = 0.1 + 0.8*np.random.rand(N_sample,1)
    np.random.seed(1)
    alphs = -0.1 + 0.2*np.random.rand(N_sample,Input_NN.shape[1])
    
    Input_NN = mean + alphs
    
    Zhat = Input_NN@ np.transpose(W[0]) + np.array(b[0]).reshape(1,-1)
    Z = np.maximum(Zhat,0)
    for k in range(1,N_hid_l):
        Zhat = Z@ np.transpose(W[k]) + np.array(b[k]).reshape(1,-1)
        Z = np.maximum(Zhat,0)

    G_hat = Z@ np.transpose(W[N_hid_l]) + np.array(b[N_hid_l]).reshape(1,-1)
    G_hat = np.maximum(G_hat,0)
    G_hat = np.minimum(G_hat,1)

    PB = np.sum(G_hat*Gen_delta,1) - np.sum(Dem_min.reshape(1,-1)  + Input_NN*Dem_delta.reshape(1,-1),1)
    
    ind=np.argpartition(-PB, -4)[-n_adver:]

    Adverse_example = Input_NN[ind][:]
    
    Adverse_mean = mean[ind][:]
    Adverse_alpha = alphs[ind][:]
    
    if gradient_attak == True:
        Adverse_example = gradient_attaks_correlated(n_buses,W,b,Adverse_mean,Adverse_alpha,Z_min,Z_max,Pg_hat_min,Pg_hat_max)

    
    
    ReLU_stat={}
    Zhat = Adverse_example@ np.transpose(W[0]) + np.array(b[0]).reshape(1,-1)
    
    ReLU_stat[0] = np.ones(Zhat.shape) 
    ReLU_stat[0][Zhat<0] = 0
    Z = np.maximum(Zhat,0)
    for k in range(1,N_hid_l):
        Zhat = Z@ np.transpose(W[k]) + np.array(b[k]).reshape(1,-1)
        ReLU_stat[k] = np.ones(Zhat.shape) 
        ReLU_stat[k][Zhat<0] = 0
        Z = np.maximum(Zhat,0)
        
    G_hat = Z@ np.transpose(W[N_hid_l]) + np.array(b[N_hid_l]).reshape(1,-1)
    ReLU_stat[N_hid_l] = np.ones(G_hat.shape)
    ReLU_stat[N_hid_l][G_hat<0] = 0

    ReLU_stat[N_hid_l+1] = np.ones(G_hat.shape)
    ReLU_stat[N_hid_l+1][G_hat>1] = 0

    G_pred = np.maximum(np.minimum(G_hat,1),0)

    PB = np.sum(G_pred*Gen_delta,1) - np.sum(Dem_min.reshape(1,-1)  + Adverse_example*Dem_delta.reshape(1,-1),1)

    Adv_sol=ReLU_stat

    # Adv_sol['ReLU_stat'] = ReLU_stat
    # Adv_sol['x'] = Adverse_example
    # Adv_sol['Z'] = Z
    # Adv_sol['Z_hat'] = Zhat
    # Adv_sol['G_hat'] = G_hat
    # Adv_sol['PB'] = PB


    return Adv_sol    

    
    
def gradient_attaks_correlated(n_buses,W,b,mean_input,alpha_input,Z_min,Z_max,Pg_hat_min,Pg_hat_max):
    
    N_hid_l = len(W)-1
    N_dns= np.size(W[0],0)

    N_gen = b[N_hid_l].shape[0]
    simulation_parameters = create_example_parameters(n_buses)

    Gen_delta=torch.tensor(( simulation_parameters['true_system']['Pg_delta']).astype(np.float32))
    Dem_min=torch.tensor((simulation_parameters['true_system']['Pd_min']).reshape(1,-1).astype(np.float32))
    Dem_delta=torch.tensor((simulation_parameters['true_system']['Pd_delta']).reshape(1,-1).astype(np.float32))

    alpha = torch.tensor(alpha_input.astype(np.float32), requires_grad=True)    
    mean = torch.tensor(mean_input.astype(np.float32), requires_grad=True)
    
    x = alpha + mean
    
    
    W_tensor ={}
    b_tensor ={}
    for i in range(N_hid_l+1):
        W_tensor[i]  = torch.tensor(np.transpose(W[i]).astype(np.float32))
        b_tensor[i]  = torch.tensor(np.array(b[i]).reshape(1,-1).astype(np.float32))
    Z={}
    Zhat = {}
    for i in range(N_hid_l):
        Z[i]  = torch.tensor(np.zeros((alpha.shape[0],N_dns)), requires_grad=True)
        Zhat[i]  = torch.tensor(np.zeros((alpha.shape[0],N_dns)), requires_grad=True)

    P_g_hat = torch.tensor(np.zeros((alpha.shape[0],N_gen)), requires_grad=True)
    P_g = torch.tensor(np.zeros((alpha.shape[0],N_gen)), requires_grad=True)

    optimizer = torch.optim.Adam([alpha,mean],lr=0.001)

    for i in range(1000):
        optimizer.zero_grad()
        
        alpha = torch.tensor(alpha.clone().detach(), requires_grad=True)    
        mean = torch.tensor(mean.clone().detach(), requires_grad=True)
        
        x = alpha + mean        
        
        Zhat[0] = x@ W_tensor[0] + b_tensor[0]
        ReLU = torch.nn.ReLU()
        Z[0] = ReLU(Zhat[0])
        
        for k in range(1,N_hid_l):
            Zhat[k] = Z[k-1]@W_tensor[k] + b_tensor[k]
            Z[k]  = ReLU(Zhat[k])
        
        clamp_function=Clamp(N_gen)

        P_g_hat = Z[N_hid_l-1]@W_tensor[N_hid_l] + b_tensor[N_hid_l]
        P_g = clamp_function(P_g_hat)

        # loss= torch.sum(P_g@Gen_delta,1) - (torch.sum(Dem_min)+x@Dem_delta)
        
        loss = torch.sum(torch.sum(P_g*Gen_delta,1) -  torch.sum(Dem_min + x*Dem_delta,1))
        loss.backward()
        # optimizer.step()
        

        alpha = alpha.clone().detach() - torch.tensor(0.001)*alpha.grad
        
        mean = mean.clone().detach() - torch.tensor(0.001)*mean.grad
        
        mean = torch.maximum(mean.clone().detach(),torch.tensor(0.1))
        mean = torch.minimum(mean.clone().detach(),torch.tensor(0.9))

        alpha = torch.maximum(alpha.clone().detach(),torch.tensor(-0.1))
        alpha = torch.minimum(alpha.clone().detach(),torch.tensor(0.1))
        
        
        
    return (alpha + mean).detach().numpy()

class Clamp(nn.Module):
    def __init__(self, n_neurons):
        super(Clamp, self).__init__()
        self.lower_bound = nn.Parameter(data=torch.zeros(1), requires_grad=False)
        self.upper_bound = nn.Parameter(data=torch.ones(1), requires_grad=False)
        
    def forward(self,input):
        return input.clamp( self.lower_bound, self.upper_bound)
    

def Get_GCP(n_buses, W,b,m_solution,ReLU_stability_inactive,ReLU_stability_active,Z_min,Z_max):
    
    # unpack solution dict
    x = {}
    y_hat = {}
    ReLU_state = {}

    for i in range(len(W)):
        x[i] = m_solution['x'][i]
        y_hat[i]  = m_solution['y_hat'][i]
    
    for i in range(len(W)-1):
        ReLU_state[i] = m_solution['ReLU_state'][i]
 
    N_hid_l = len(W)-1
    
    new_Z_min = {}
    new_Z_max = {}

    I_dict ={}
    ReLU_tighting = {}

    for i in range(0,N_hid_l):
        new_Z_min[i]=np.ones((W[i].shape))*np.transpose(Z_min[i])
        new_Z_max[i]=np.ones((W[i].shape))*np.transpose(Z_max[i])
        Z_min_tep=np.ones((W[i].shape))*np.transpose(Z_min[i])
        
        new_Z_min[i][W[i]<0] = new_Z_max[i][W[i]<0]
        new_Z_max[i][W[i]<0] = Z_min_tep[W[i]<0]
    

    for i in range(0,N_hid_l):      
        # Calculating I matrix I_i = 1 means x_1 in the set else not in the set
        LHS=W[i]*x[i]
        RHS=W[i]*(new_Z_min[i]*(1-ReLU_state[i].reshape(-1,1))+ new_Z_max[i]*ReLU_state[i].reshape(-1,1))
        I = LHS < RHS
        I_dict[i]=I
        
        # Calculating ReLU tightining

        LHS=x[i+1]-b[i]*ReLU_state[i]
        XLZ=W[i]*(x[i] - new_Z_min[i]*(1-ReLU_state[i].reshape(-1,1)))
        UZ=W[i]*(new_Z_max[i]*ReLU_state[i].reshape(-1,1))
        
        RHS =np.sum(I*XLZ,axis=1) + np.sum((1-I)*UZ,axis=1)

        ReLU_tighting[i]=LHS>RHS
    
    for i in range(ReLU_stability_active.shape[1]):
        ReLU_tighting[i] = (1-(ReLU_stability_inactive[:,i] | ReLU_stability_active[:,i])) * ReLU_tighting[i]

    return ReLU_tighting, I_dict,new_Z_min,new_Z_max

class Bound_pool(object):
    def __init__(self,n_buses, W,b,N_layer,ReLU_stability_inactive,ReLU_stability_active,Z_min_IB,Z_max_IB):
        # self.n_buses = n_buses
        self.W = W
        self.b = b
        self.N_layer = N_layer
        self.ReLU_stability_inactive = ReLU_stability_inactive
        self.ReLU_stability_active = ReLU_stability_active
        self.Z_min_IB = Z_min_IB
        self.Z_max_IB = Z_max_IB

        self.Z_max_val=np.copy(Z_max_IB)
        self.Z_min_val=np.copy(Z_min_IB)

    def  __call__(self,para):
        
        ## Initializing Variables
        W = self.W
        b = self.b
        N_layer = self.N_layer
        ReLU_stability_inactive = self.ReLU_stability_inactive
        ReLU_stability_active = self.ReLU_stability_active
        Z_min_IB = self.Z_min_IB
        Z_max_IB = self.Z_max_IB

        N_hid_l = len(W)-1
        
        N_lod= np.size(W[0],1)
        N_dns= np.size(W[0],0)

        m = gp.Model()
        
        x = m.addMVar(N_lod,lb=0,ub=1,name="x")
        Z = m.addMVar((N_dns,N_hid_l),lb=0,ub=Z_max_IB,name="Z")
        Z_hat = m.addMVar((N_dns,N_hid_l),lb=Z_min_IB,ub=Z_max_IB,name="Z_hat")
        
        ReLU_stat = m.addMVar((N_dns,N_hid_l),vtype=gp.GRB.BINARY,name="ReLU_stat")
        
        for i in range(N_dns):
            for j in range(N_hid_l):
                if ReLU_stability_inactive[i][j] == True:
                    m.addConstr(ReLU_stat[i][j]== False)
                if ReLU_stability_active[i][j] == True:
                    m.addConstr(ReLU_stat[i][j]== True)
        # First layer
        m.addConstr(Z_hat[:,0] - W[0] @ x -b[0] == 0)
        # Hidden layers
        for i in range(1,N_layer):
            m.addConstr(Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0)
        

        # # Relu MILP     
        m.addConstr(Z - Z_hat - ReLU_stat*Z_min_IB + Z_min_IB <= 0)
        m.addConstr(Z - Z_hat>= 0)
        m.addConstr(Z >= 0)
        m.addConstr(Z - ReLU_stat*Z_max_IB <= 0)

        i=para
        j=N_layer-1

        m.setObjective(Z_hat[i][j], gp.GRB.MINIMIZE)
        m.Params.TimeLimit = 5*N_layer
        m.optimize()
        bound_min[i] = max(m.ObjBound,Z_min_IB[i][j])
        bound_min[i] = min(bound_min[i],Z_max_IB[i][j])

        m.setObjective(Z_hat[i][j], gp.GRB.MAXIMIZE)
        m.Params.TimeLimit = 5*N_layer
        m.optimize()
        bound_max[i] = min(m.ObjBound,Z_max_IB[i][j])
        bound_max[i] = max(bound_max[i],Z_min_IB[i][j])
        
class Gen_bound_pool(object):
    def __init__(self,n_buses, W,b,ReLU_stability_inactive,ReLU_stability_active,Z_min_IB,Z_max_IB):
        # self.n_buses = n_buses
        self.W = W
        self.b = b
        self.ReLU_stability_inactive = ReLU_stability_inactive
        self.ReLU_stability_active = ReLU_stability_active
        self.Z_min_IB = Z_min_IB
        self.Z_max_IB = Z_max_IB
        
        N_gen= np.size(W[len(W)-1],0)

    def  __call__(self, g):

        # n_buses = self.n_buses
        W = self.W
        b = self.b
        ReLU_stability_inactive = self.ReLU_stability_inactive
        ReLU_stability_active = self.ReLU_stability_active
        Z_min_IB = self.Z_min_IB
        Z_max_IB = self.Z_max_IB
        
        N_hid_l = len(W)-1
        
        N_lod= np.size(W[0],1)
        N_dns= np.size(W[0],0)
        N_gen= np.size(W[N_hid_l],0)
        
        m = gp.Model()
        
        x = m.addMVar(N_lod,lb=0,ub=1,name="x")
        Z = m.addMVar((N_dns,N_hid_l),lb=0,ub=Z_max_IB,name="Z")
        Z_hat = m.addMVar((N_dns,N_hid_l),lb=Z_min_IB,ub=Z_max_IB,name="Z_hat")
        pg_hat = m.addMVar(N_gen,lb=-np.inf, name="pg_pre")
        ReLU_stat = m.addMVar((N_dns,N_hid_l),vtype=gp.GRB.BINARY,name="ReLU_stat")
        
        for i in range(N_dns):
            for j in range(N_hid_l):
                if ReLU_stability_inactive[i][j] == True:
                    m.addConstr(ReLU_stat[i][j]== False)
                if ReLU_stability_active[i][j] == True:
                    m.addConstr(ReLU_stat[i][j]== True)
    
        
        # First layer
        m.addConstr(Z_hat[:,0] - W[0] @ x -b[0] == 0)
        # Hidden layers
        for i in range(1,N_hid_l):
            m.addConstr(Z_hat[:,i] - W[i] @ Z[:,i-1] -b[i] == 0)
        
        # # Relu MILP     
        m.addConstr(Z - Z_hat - ReLU_stat*Z_min_IB + Z_min_IB <= 0)
        m.addConstr(Z - Z_hat>= 0)
        m.addConstr(Z >= 0)
        m.addConstr(Z - ReLU_stat*Z_max_IB <= 0)
        m.addConstr(pg_hat - W[N_hid_l] @ Z[:,N_hid_l-1] -b[N_hid_l] == 0)
        
        m.setObjective(pg_hat[g], gp.GRB.MINIMIZE)
        m.Params.TimeLimit = 20
        m.optimize()
        if m.Status != gp.GRB.INFEASIBLE:
            bound_min[g] = m.ObjBound
        else:
            bound_min[g] = -np.inf
        m.setObjective(pg_hat[g], gp.GRB.MAXIMIZE)
        m.Params.TimeLimit = 20
        m.optimize()
        if m.Status != gp.GRB.INFEASIBLE:
            # bound[g] = max(m.ObjBound,Pg_hat_min[i])
            bound_max[g] = m.ObjBound
        else:
            bound_max[g] = np.inf
