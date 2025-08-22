import pandas as pd
import numpy as np


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

    Psp = pd.read_csv('Data_File/case'+str(n_buses)+'/Psp.csv', header=None).to_numpy()
    Qsp = pd.read_csv('Data_File/case'+str(n_buses)+'/Qsp.csv', header=None).to_numpy()
    pv_bus = pd.read_csv('Data_File/case'+str(n_buses)+'/pv_bus.csv', header=None).to_numpy()
    pq_bus = pd.read_csv('Data_File/case'+str(n_buses)+'/pq_bus.csv', header=None).to_numpy()
    sl_bus = pd.read_csv('Data_File/case'+str(n_buses)+'/sl_bus.csv', header=None).to_numpy()
    B = pd.read_csv('Data_File/case'+str(n_buses)+'/B.csv', header=None).to_numpy()
    G = pd.read_csv('Data_File/case'+str(n_buses)+'/G.csv', header=None).to_numpy()
    B1 = pd.read_csv('Data_File/case'+str(n_buses)+'/B1.csv', header=None).to_numpy()
    B2 = pd.read_csv('Data_File/case'+str(n_buses)+'/B2.csv', header=None).to_numpy()
    # Y= pd.read_csv('Data_File/case'+str(n_buses)+'/Y.csv', header=None).to_numpy()
    Ybus_str= pd.read_csv('Data_File/case'+str(n_buses)+'/Ybus.csv', header=None)
    Ybus = Ybus_str.applymap(lambda s: np.complex_(s.replace('i', 'j'))).values

   
    # -----------------------------------------------------------------------------------------------
    # True system parameters of the power system that are assumed to be known in the identification process
    # -----------------------------------------------------------------------------------------------
    
    true_system_parameters = {'Psp': Psp,
                              'Qsp': Qsp,
                              'pv_bus': pv_bus,
                              'pq_bus':pq_bus,
                              'sl_bus':sl_bus,
                              'B':B,
                              'G':G,
                              'B1':B1,
                              'B2':B2,
                              'Ybus':Ybus
                              }
    # -----------------------------------------------------------------------------------------------
    # general parameters of the power system that are assumed to be known in the identification process
    # n_buses: integer number of buses in the system
    # -----------------------------------------------------------------------------------------------
    general_parameters = {'n_buses': n_buses
                          }
    # -----------------------------------------------------------------------------------------------
    # parameters for the training data creation 
    # n_data_points: number of data points where measurements are present
    # -----------------------------------------------------------------------------------------------
    n_data_points = 7000
    n_test_data_points=1000

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_test_data_points': n_test_data_points}

    # -----------------------------------------------------------------------------------------------
    # combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters}

    return simulation_parameters
