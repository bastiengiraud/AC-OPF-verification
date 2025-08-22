import pandas as pd
import numpy as np
import os

def create_example_parameters(n_buses: int):
    """
    creates a basic set of parameters that are used in the following processes:
    * data creation if measurements are to be simulated
    * setting up the neural network model
    * training procedure

    :param n_buses: integer number of buses in the system
    :return: simulation_parameters: dictonary that holds all parameters
    """

    # -----------------------------------------------------------------------------------------------
    # underlying parameters of the power system
    # primarily for data creation when no measurements are provided
    # -----------------------------------------------------------------------------------------------
    
    # Get the absolute path to the folder containing this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the data directory path
    data_dir = os.path.join(base_dir, f'dc_opf_data/{n_buses}')

    DC_OPF = True    
    if DC_OPF == True:
        Gen = pd.read_csv(os.path.join(data_dir, 'Gen.csv'), index_col=0)
        g_bus=Gen.index[Gen['Pg_delta']!=0].to_numpy()
        n_gbus=len(g_bus)
        # Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_delta=Gen['Pg_delta'].to_numpy().reshape((1, n_gbus))
        Map_g = np.zeros((n_gbus,n_buses))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            gen_no+=1
            
        Bus = pd.read_csv(os.path.join(data_dir, 'Bus.csv'))
        PTDF = pd.read_csv(os.path.join(data_dir, 'PTDF.csv'), header=None)
        
        # PTDF = pd.read_csv('/work3/rnelli/Dataset/'+str(n_buses)+'/PTDF.csv', header=None)
        l_bus=Bus['ID'].to_numpy()
        Pd_min = Bus['Pd_min'].to_numpy()
        Pd_delta = Bus['Pd_delta'].to_numpy()
        n_lbus=len(l_bus)
        Map_L = np.zeros((n_lbus,n_buses))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            l_no+=1
        
        branches = pd.read_csv(os.path.join(data_dir, 'branches.csv'))
        n_line=branches.shape[0]      
        Pl_max=branches['branch_flowlimit'].to_numpy().reshape((1,n_line))
    # -----------------------------------------------------------------------------------------------
    # True system parameters of the power system that are assumed to be known in the identification process
    # -----------------------------------------------------------------------------------------------
    
        true_system_parameters = {'Pg_delta' : Pg_delta,
                                  'Pd_min':Pd_min,
                                  'Pd_delta':Pd_delta,
                                  'g_bus':g_bus,
                                  'Map_g':Map_g,
                                  'Map_L':Map_L,
                                  'PTDF':PTDF,
                                  'Pl_max':Pl_max
                                  }
        
    AC_OPF = False    
    if AC_OPF == True:
        Gen = pd.read_csv(os.path.join(data_dir, 'Gen.csv'), index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        # C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        Qg_max=Gen['Qg_max'].to_numpy().reshape((1, n_gbus))
        Qg_min=Gen['Qg_min'].to_numpy().reshape((1, n_gbus))
        Gen_delta=np.concatenate((Pg_max-Pg_min,Qg_max-Qg_min),axis=1).reshape((2*n_gbus,1))
        Gen_max=np.concatenate((Pg_max,Qg_max),axis=1).reshape((2*n_gbus,1))
        Volt_max = 1.06
        Volt_min = 0.94
        Map_g = np.zeros((2*n_gbus,2*n_buses))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            Map_g[n_gbus + gen_no][n_buses + g-1]=1
            gen_no+=1   
            
        Bus = pd.read_csv(os.path.join(data_dir, 'Bus.csv'))
        l_bus=Bus['Node'].to_numpy()
        Pd_max = Bus['Pdmax'].to_numpy()
        Qd_max = Bus['Qdmax'].to_numpy()
        n_lbus=len(l_bus)
        Dem_max=np.concatenate((Pd_max,Qd_max),axis=0).reshape((2*n_lbus,1))
        
        Map_L = np.zeros((2*n_lbus,2*n_buses))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            Map_L[n_lbus+l_no][n_buses+l-1]=1
            l_no+=1
        Y= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/Y.csv', header=None).to_numpy()
        Yconj= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/Yconj.csv', header=None).to_numpy()
        Ybr= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/Ybr.csv', header=None).to_numpy()
        IM= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/IM.csv', header=None).to_numpy()
        n_line=int(np.size(IM,0)/2)
        L_limit= pd.read_csv('verify-powerflow/DC_OPF_Data/'+str(n_buses)+'/L_limit.csv', header=None).to_numpy().reshape(1, n_line)
        
        
    # -----------------------------------------------------------------------------------------------
    # True system parameters of the power system that are assumed to be known in the identification process
    # -----------------------------------------------------------------------------------------------
    
        true_system_parameters = {'Gen_delta': Gen_delta,
                                  'Gen_max': Gen_max,
                                  'Dem_max': Dem_max,
                                  'Pd_max':Pd_max,
                                  'Volt_max': Volt_max,
                                  'Volt_min':Volt_min,
                                  'Y':Y,
                                  'Yconj':Yconj,
                                  'Ybr':Ybr,
                                  'Map_g':Map_g,
                                  'Map_L':Map_L,
                                  'IM':IM,
                                  'g_bus':g_bus,
                                  'n_lbus':n_lbus,
                                  'n_line' : n_line,
                                  'L_limit':L_limit
                                  }
    # -----------------------------------------------------------------------------------------------
    # general parameters of the power system that are assumed to be known in the identification process
    # n_buses: integer number of buses in the system
    # -----------------------------------------------------------------------------------------------
    general_parameters = {'n_buses': n_buses,
                          'g_bus': g_bus,
                          'n_gbus':n_gbus
                          }
    # -----------------------------------------------------------------------------------------------
    # parameters for the training data creation 
    # n_data_points: number of data points where measurements are present
    # n_test_data_points: number of test data points where measurements are present
    # s_point: starting point in the dataset from which the data sets will be collected from
    # -----------------------------------------------------------------------------------------------
    n_data_points = 4000
    n_test_data_points=1000

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_test_data_points': n_test_data_points,
                                's_point': 0}

    # -----------------------------------------------------------------------------------------------
    # combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters}

    return simulation_parameters
