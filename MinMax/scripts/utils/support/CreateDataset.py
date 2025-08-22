# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 07:25:14 2023

@author: rnelli
"""

import numpy as np
import pandas as pd

def create_data(simulation_parameters):
    
    n_buses=simulation_parameters['general']['n_buses'] 
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = 0
    Psp = pd.read_csv('Data_File/case'+str(n_buses)+'/Psp.csv', header=None).to_numpy()[s_point:s_point+n_data_points][:]
    Qsp = pd.read_csv('Data_File/case'+str(n_buses)+'/Qsp.csv', header=None).to_numpy()[s_point:s_point+n_data_points][:]
    
    return Psp, Qsp


def create_test_data(simulation_parameters):

    n_buses=simulation_parameters['general']['n_buses'] 
    n_test_data_points = simulation_parameters['data_creation']['n_test_data_points']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = 0
    n_total = n_data_points + n_test_data_points

    
    # L_Val=pd.read_csv('/work3/rnelli/Dataset/'+str(n_buses)+'/NN_input_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    # Gen_out = pd.read_csv('/work3/rnelli/Dataset/'+str(n_buses)+'/NN_output_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    
    Psp = pd.read_csv('Data_File/case'+str(n_buses)+'/Psp.csv', header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]
    Qsp = pd.read_csv('Data_File/case'+str(n_buses)+'/Qsp.csv', header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]

    return Psp, Qsp


