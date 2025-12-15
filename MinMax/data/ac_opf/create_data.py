import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

import pandapower.converter as pc
import pandapower as pp
from pypower.api import runopf, loadcase
from pypower.idx_bus import PD, QD, BUS_I, VM, VA, BUS_TYPE
from pypower.idx_gen import PG, QG, GEN_BUS
from pypower.idx_brch import F_BUS, T_BUS
from pypower.ppoption import ppoption
from pypower.makeYbus import makeYbus
import torch
import copy

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['config']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

import loadsampling as ls

def create_data(simulation_parameters):
    
    n_buses=simulation_parameters['general']['n_buses'] 
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = simulation_parameters['data_creation']['s_point']
    nn_config = simulation_parameters['nn_output']
    
    # Get the absolute path to the folder containing this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the data directory path
    data_dir = os.path.join(base_dir, f'ac_opf_data/{n_buses}')
    output_dir = os.path.join(data_dir, f'Dataset')
    
    if n_buses == 300:
        L_Val = pd.read_csv(os.path.join(output_dir, f'NN_input_{nn_config}.csv'), skiprows=1,low_memory=False, header=None).to_numpy()[s_point:s_point+n_data_points][:] 
        Gen_out = pd.read_csv(os.path.join(output_dir, f'NN_output_{nn_config}.csv'), skiprows=1,low_memory=False, header=None).to_numpy()[s_point:s_point+n_data_points][:]
    else:
        L_Val = pd.read_csv(os.path.join(output_dir, f'NN_input_{nn_config}.csv'), header=None).to_numpy()[s_point:s_point+n_data_points][:] 
        Gen_out = pd.read_csv(os.path.join(output_dir, f'NN_output_{nn_config}.csv'), header=None).to_numpy()[s_point:s_point+n_data_points][:]
    
    #L_Val=pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_input_actual.csv').to_numpy()[s_point:s_point+n_data_points][:] 
    #Gen_out = pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_output_actual.csv').to_numpy()[s_point:s_point+n_data_points][:]

    x_training = L_Val
    return x_training, Gen_out


def create_test_data(simulation_parameters):

    n_buses=simulation_parameters['general']['n_buses'] 
    n_test_data_points = simulation_parameters['data_creation']['n_test_data_points']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = simulation_parameters['data_creation']['s_point']
    n_total = n_data_points + n_test_data_points
    nn_config = simulation_parameters['nn_output']
    
    # Get the absolute path to the folder containing this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the data directory path
    data_dir = os.path.join(base_dir, f'ac_opf_data/{n_buses}')
    output_dir = os.path.join(data_dir, f'Dataset')
    
    if n_buses == 300:
        L_Val = pd.read_csv(os.path.join(output_dir, f'NN_input_{nn_config}.csv'), skiprows=1,low_memory=False, header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]
        Gen_out = pd.read_csv(os.path.join(output_dir, f'NN_output_{nn_config}.csv'), skiprows=1,low_memory=False, header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]
    else:
        L_Val = pd.read_csv(os.path.join(output_dir, f'NN_input_{nn_config}.csv'), header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]
        Gen_out = pd.read_csv(os.path.join(output_dir, f'NN_output_{nn_config}.csv'), header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]   
        
    #L_Val=pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_input_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    #Gen_out = pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_output_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
        
    x_test = np.concatenate([L_Val], axis=0)
    return x_test, Gen_out


import cvxpy as cp

# def generate_power_system_data(simulation_parameters, save_csv=True):
    
#     """ 
#     Generates training data for AC OPF by simulating various load profiles,
#     solving the AC OPF, and storing inputs (scaled loads) and outputs
#     (generator active power and bus voltage magnitudes) to CSV files.
    
#     """
    
#     # =============== Extract Parameters from simulation_parameters ==============
#     true_system_params = simulation_parameters['true_system']
#     general_params = simulation_parameters['general']
#     data_creation_params = simulation_parameters['data_creation']

#     n_buses = general_params['n_buses']
#     n_gbus = general_params['n_gbus'] # Number of generators
#     n_data_points = 10_000 # data_creation_params['n_data_points']
       
    
#     # ============= specify pglib-opf case based on n_buses ==================
#     base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
    
#     combination = simulation_parameters['nn_output']
    
#     if n_buses == 118:
#         case_name = 'pglib_opf_case118_ieee.m'
#     elif n_buses == 300:
#         case_name = 'pglib_opf_case300_ieee.m'
#     elif n_buses == 793:
#         case_name = 'pglib_opf_case793_goc.m'
#     elif n_buses == 1354:
#         case_name = 'pglib_opf_case1354_pegase.m'
#     elif n_buses == 2869:
#         case_name = 'pglib_opf_case2869_pegase.m'
        
#     case_path = os.path.join(base_dir, 'pglib-opf', case_name)
    
#     # Load the MATPOWER case from a local .m file
#     net = pc.from_mpc(case_path, casename_mpc_file=True)
#     base_ppc = pc.to_ppc(net, init = 'flat')
    
#     # Run OPF
#     ppopt = ppoption(OUT_ALL=0)
#     results = runopf(base_ppc, ppopt)
#     Ybus, _, _ = makeYbus(base_ppc['baseMVA'], base_ppc['bus'], base_ppc['branch'])

#     # Net data
#     Sbase = base_ppc['baseMVA'] # Sbase from PYPOWER case
#     n_bus = base_ppc['bus'].shape[0]
#     n_gens = base_ppc['gen'].shape[0]
#     n_loads = len(net.load) # Number of distinct load elements in pandapower's view
    
#     # Extract min/max Pg per generator:
#     pg_min_values = base_ppc['gen'][:, 9].reshape((n_gens, 1))  # PMIN (column 9)
#     pg_max_values = base_ppc['gen'][:, 8].reshape((n_gens, 1))  # PMAX (column 8)
    
#     qg_min_values = base_ppc['gen'][:, 4].reshape((n_gens, 1))  # QMIN (column 4)
#     qg_max_values = base_ppc['gen'][:, 3].reshape((n_gens, 1))  # QMAX (column 3)

#     # Extract min/max Vm per bus:
#     vm_min_values = base_ppc['bus'][:, 11].reshape((n_bus, 1))  # VMIN
#     vm_max_values = base_ppc['bus'][:, 12].reshape((n_bus, 1))  # VMAX

#     print(f"System details: {n_bus} buses, {n_gens} generators, {n_loads} loads.")

#     # Obtain nominal loads (from the loaded case's ppc)
#     pd_nom = np.array(net.load['p_mw']).reshape(-1, 1) # Active power for each load element
#     qd_nom = np.array(net.load['q_mvar']).reshape(-1, 1) # Reactive power for each load element
#     loads_nominal = np.vstack([pd_nom, qd_nom]) # Combined P and Q nominal loads for each load element

#     # Define load perturbation bounds (relative to nominal loads)
#     lb_factor = 0.6 * np.ones(loads_nominal.shape[0])
#     ub_factor = 1.0 * np.ones(loads_nominal.shape[0])

#     # # Generate load scaling factors
#     # X_factors = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, n_data_points)

#     # # Calculate actual load values for each data point (MW/Mvar)
#     # X_unscaled_loads_mw = loads_nominal * X_factors

#     # # Separate active and reactive power components for adjustment
#     # pd_tot_mw = X_unscaled_loads_mw[:n_loads, :]
#     # qd_tot_mvar = X_unscaled_loads_mw[n_loads:, :]

#     # # --- Input Scaling (for NN input) ---
#     # # Convert to per-unit for NN input, as it's common practice
#     # X_loads_pu = X_unscaled_loads_mw / Sbase

#     # Min-max scaling for loads (per unit)
#     load_min = np.zeros_like(loads_nominal)  # zero load vector
#     load_max = loads_nominal / Sbase          # nominal load in per-unit

#     # To safely broadcast, reshape min and max:
#     load_min = load_min.reshape((-1, 1))  
#     load_max = load_max.reshape((-1, 1))

#     # # Min-max scale: (x - min) / (max - min), here max-min = max since min=0
#     # denominator = np.where(load_max == 0, 1, load_max)  # avoid division by zero
#     # X_scaled = (X_loads_pu - load_min) / denominator  # shape: (2*n_loads, n_data_points)

#     # # Transpose to (n_data_points, 2*n_loads) for NN input format
#     # X_nn_input = X_scaled.T

#     # --- Output Collection (for NN labels) ---
#     pg_tot = torch.zeros(n_gens, int(n_data_points), dtype=torch.float32)
#     qg_tot = torch.zeros(n_gens, int(n_data_points), dtype=torch.float32)
#     vm_tot = torch.zeros(n_bus, int(n_data_points), dtype=torch.float32)
    
#     pinj_tot = torch.zeros(n_bus, int(n_data_points), dtype=torch.float32)
#     qinj_tot = torch.zeros(n_bus, int(n_data_points), dtype=torch.float32)
#     vr_tot = torch.zeros(n_bus, int(n_data_points), dtype=torch.float32)
#     vi_tot = torch.zeros(n_bus, int(n_data_points), dtype=torch.float32)
    
#     # -------------------------------------------------------------
    
#     # Get the internal PYPOWER case from the pandapower network
#     base_ppc = net._ppc
    
#     # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
#     external_bus_numbers = net.bus.index.values # indices from .m file from pandapower
#     internal_indices = base_ppc['bus'][:, 0].astype(int) #  internal inddices from pypower

#     external_to_internal = dict(zip(external_bus_numbers, internal_indices))
#     internal_to_external = dict(zip(internal_indices, external_bus_numbers))


#     # ===================== Data Generation Loop (Corrected) =====================
#     print(f"Solving {n_data_points} AC-OPF problems with PYPOWER...")

#     i = 0
#     pbar = tqdm(total=int(n_data_points), position=0, leave=True)

#     # CORRECTED: Initialize a list to store all successful inputs
#     all_X_nn_inputs = []

#     while i < n_data_points:
#         # Generate a single load profile sample
#         X_factors_sample = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, 1)
#         X_unscaled_loads_mw = loads_nominal * X_factors_sample
        
#         pd_tot_mw = X_unscaled_loads_mw[:n_loads, :]
#         qd_tot_mvar = X_unscaled_loads_mw[n_loads:, :]
        
#         X_loads_pu = X_unscaled_loads_mw / Sbase
#         load_min = np.zeros_like(loads_nominal).reshape((-1, 1))
#         load_max = (loads_nominal / Sbase).reshape((-1, 1))
#         denominator = np.where(load_max == 0, 1, load_max)
#         X_scaled = (X_loads_pu - load_min) / denominator
#         X_nn_input = X_scaled.T

#         try:
#             current_ppc = copy.deepcopy(base_ppc)
            
#             for load_idx_pp, bus_idx_pp_internal in net.load['bus'].items():
#                 original_matpower_bus_id = external_to_internal.get(bus_idx_pp_internal, None)
#                 if original_matpower_bus_id is None:
#                     continue
                
#                 ppc_bus_row_idx = np.where(current_ppc['bus'][:, BUS_I] == original_matpower_bus_id)[0]
#                 if len(ppc_bus_row_idx) == 0:
#                     continue
#                 ppc_bus_row_idx = ppc_bus_row_idx[0]

#                 current_ppc['bus'][ppc_bus_row_idx, PD] = pd_tot_mw[load_idx_pp, 0]
#                 current_ppc['bus'][ppc_bus_row_idx, QD] = qd_tot_mvar[load_idx_pp, 0]
            
#             results = runopf(current_ppc, ppopt)
#             success = (results['success'] == 1)
            
#             if success:
#                 # Store results
#                 pg_tot[:, i] = torch.tensor(results['gen'][:, PG] / 100, dtype=torch.float32)
#                 qg_tot[:, i] = torch.tensor(results['gen'][:, QG] / 100, dtype=torch.float32)
#                 vm_tot[:, i] = torch.tensor(results['bus'][:, VM], dtype=torch.float32)
                
#                 vm = results['bus'][:, VM]
#                 va_rad = np.deg2rad(results['bus'][:, VA])
#                 vr = vm * np.cos(va_rad)
#                 vi = vm * np.sin(va_rad)
                
#                 V = results['bus'][:, VM] * np.exp(1j * np.deg2rad(results['bus'][:, VA]))
#                 I = Ybus @ V
#                 S = V * np.conj(I)
                
#                 pinj_tot[:, i] = torch.tensor(np.real(S), dtype=torch.float32)
#                 qinj_tot[:, i] = torch.tensor(np.imag(S), dtype=torch.float32)
#                 vr_tot[:, i] = torch.tensor(vr, dtype=torch.float32)
#                 vi_tot[:, i] = torch.tensor(vi, dtype=torch.float32)
                
#                 # CORRECTED: Append the successful input to our list
#                 all_X_nn_inputs.append(X_nn_input)
                
#                 i += 1
#                 pbar.update(1)
#             else:
#                 print(f"Warning: OPF did not converge for sample {i}. Generating new load profile and retrying.")
                
#         except Exception as e:
#             print(f"Warning: PYPOWER OPF failed for sample {i} with error: {e}. Generating new load profile and retrying.")

#     pbar.close()

#     # ===================== Save Data (Corrected) =====================
#     # Now, we process the collected input data.
#     # CORRECTED: Combine the list of successful inputs into a single NumPy array
#     X_loads_all = np.vstack(all_X_nn_inputs)


#     # Obtain labels (NN output)
#     pg_np = pg_tot.numpy()
#     qg_np = qg_tot.numpy()
#     vm_np = vm_tot.numpy()
    
#     vr_tot = vr_tot.numpy()
#     vi_tot = vi_tot.numpy()
    
#     # ============= remove entries with slack or pg_max = 0 ================
#     pg_max_zero_mask = pg_max_values.flatten() < 1e-9  # shape: (n_gens,)
#     qg_max_zero_mask = qg_max_values.flatten() < 1e-9  # shape: (n_gens,)
#     slack_bus_indices = np.where(base_ppc['bus'][:, 1] == 3)[0]  # BUS_TYPE == 3 (slack)
#     slack_gen_mask = np.isin(base_ppc['gen'][:, 0], slack_bus_indices)  # shape: (n_gens,)

#     # remove generators where pg_max = 0, and the slack bus
#     gen_mask_to_remove = np.logical_or(pg_max_zero_mask, slack_gen_mask)  # shape: (n_gens,)
#     gen_mask_to_keep = ~gen_mask_to_remove  # invert mask to keep desired generators
#     pg_np = pg_np[gen_mask_to_keep, :]
    
#     # remove generator where qg_max = 0 and slack bus
#     qgen_mask_to_remove = np.logical_or(qg_max_zero_mask, slack_gen_mask)  # shape: (n_gens,)
#     qgen_mask_to_keep = ~qgen_mask_to_remove  # invert mask to keep desired generators
#     qg_np = qg_np[qgen_mask_to_keep, :]
    
#     # # remove from min max values as well
#     # pg_min_values = pg_min_values[gen_mask_to_keep, :]
#     # pg_max_values = pg_max_values[gen_mask_to_keep, :]
    
#     # qg_min_values = qg_min_values[qgen_mask_to_keep, :]
#     # qg_max_values = qg_max_values[qgen_mask_to_keep, :]
    
#     # ================= only keep voltages at generators: =======================
#     gen_bus_indices = base_ppc['gen'][:, 0].astype(int)  # buses with generators
#     mask = ~np.isin(gen_bus_indices, slack_bus_indices)
#     gen_bus_indices_no_slack = gen_bus_indices[mask]
#     vm_np = vm_np[gen_bus_indices_no_slack, :]  # shape (n_gens, n_samples)
    
#     # # remove from min max values as well
#     # vm_min_values = vm_min_values[gen_bus_indices_no_slack, :]
#     # vm_max_values = vm_max_values[gen_bus_indices_no_slack, :]
    
#     # =================== remove slack entries for vr and vi =====================
#     # bus_indices = base_ppc['bus'][:, 0].astype(int)
#     # mask = ~np.isin(bus_indices, slack_bus_indices)
    
#     # # remove slack bus
#     # vr_tot = vr_tot[mask, :]
#     # vi_tot = vi_tot[mask, :]
    
#     # # ============== Min-max scaling for generators (across entries, axis=1 is gen dimension, axis=0 is sample dimension) =========
#     # pg_max_values += 1e-6 # avoid numerical instability for very small min and max values
#     # pg_min_values -= 1e-6
#     # pg_denominator = np.where(pg_max_values - pg_min_values == 0, 1, pg_max_values - pg_min_values)
#     # pg_scaled = (pg_np - pg_min_values) / pg_denominator
    
#     # qg_max_values += 1e-6 # avoid numerical instability for very small min and max values
#     # qg_min_values -= 1e-6
#     # qg_denominator = np.where(qg_max_values - qg_min_values == 0, 1, qg_max_values - qg_min_values)
#     # qg_scaled = (qg_np - qg_min_values) / qg_denominator

#     # vm_max_values += 1e-6
#     # vm_min_values -= 1e-6
#     # vm_denominator = np.where(vm_max_values - vm_min_values == 0, 1, vm_max_values - vm_min_values)
#     # vm_scaled = (vm_np - vm_min_values) / vm_denominator
    
#     # === Save all desired combinations ===
#     combinations_to_save = ["pg_vm", "pg_qg", "vr_vi"]  # or add "surrogate" if needed

#     output_data_dir = os.path.join(
#         base_dir,
#         'data/ac_opf_data',
#         str(n_buses),
#         'Dataset'
#     )
#     os.makedirs(output_data_dir, exist_ok=True)

#     for combination in combinations_to_save:
#         if combination == "pg_vm":
#             y_scaled = np.vstack([pg_np, vm_np])
#             Y_nn_output = y_scaled.T
#             X_nn_input = X_loads_all # X_loads_pu.T
#         elif combination == "pg_qg":
#             y_scaled = np.vstack([pg_np, qg_np])
#             Y_nn_output = y_scaled.T
#             X_nn_input = X_loads_all # X_loads_pu.T
#         elif combination == "vr_vi":
#             y_scaled = np.vstack([vr_tot, vi_tot])
#             Y_nn_output = y_scaled.T
#             X_nn_input = X_loads_all #X_loads_pu.T
#         elif combination == "surrogate":
#             y = np.vstack([vr_tot, vi_tot])
#             x = np.vstack([pinj_tot, qinj_tot])
#             Y_nn_output = y.T
#             X_nn_input = x.T
#         else:
#             print(f"Combination '{combination}' not recognized. Skipping.")
#             continue

#         # Save to CSV
#         input_filename = os.path.join(output_data_dir, f"NN_input_{combination}.csv")
#         output_filename = os.path.join(output_data_dir, f"NN_output_{combination}.csv")

#         if save_csv:
#             pd.DataFrame(X_nn_input).to_csv(input_filename, index=False, header=False)
#             pd.DataFrame(Y_nn_output).to_csv(output_filename, index=False, header=False)

#         print(f"Saved combination '{combination}' with shapes: input {X_nn_input.shape}, output {Y_nn_output.shape}")


#     return X_nn_input, Y_nn_output




def generate_power_system_data(simulation_parameters, save_csv=True, append_data=False):
    """
    Generates training data for AC OPF by simulating various load profiles,
    solving the AC OPF, and storing inputs (scaled loads) and outputs
    (generator active power and bus voltage magnitudes) to CSV files.
    
    Args:
        simulation_parameters (dict): Dictionary containing simulation parameters.
        save_csv (bool): If True, saves the generated data to CSV files.
        append_data (bool): If True, appends to existing CSV files instead of overwriting.
    """
    
    # =============== Extract Parameters from simulation_parameters ==============
    true_system_params = simulation_parameters['true_system']
    general_params = simulation_parameters['general']
    data_creation_params = simulation_parameters['data_creation']

    n_buses = general_params['n_buses']
    n_gbus = general_params['n_gbus'] # Number of generators
    n_data_points = 5_000 # data_creation_params['n_data_points']
    
    # ============= specify pglib-opf case based on n_buses ==================
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    
    combination = simulation_parameters['nn_output']
    
    if n_buses == 14:
        case_name = 'pglib_opf_case14_ieee.m'
    elif n_buses == 57:
        case_name = 'pglib_opf_case57_ieee.m'
    elif n_buses == 118:
        case_name = 'pglib_opf_case118_ieee.m'
    elif n_buses == 300:
        case_name = 'pglib_opf_case300_ieee.m'
    elif n_buses == 793:
        case_name = 'pglib_opf_case793_goc_cleaned.m'
    elif n_buses == 1354:
        case_name = 'pglib_opf_case1354_pegase.m'
    elif n_buses == 2869:
        case_name = 'pglib_opf_case2869_pegase.m'
    
    case_path = os.path.join(base_dir, 'pglib-opf', case_name)
    
    # Load the MATPOWER case from a local .m file
    net = pc.from_mpc(case_path, casename_mpc_file=True)
    base_ppc = pc.to_ppc(net, init = 'flat')

    # Run OPF
    ppopt = ppoption(OUT_ALL=0)
    results = runopf(base_ppc, ppopt)
    Ybus, _, _ = makeYbus(base_ppc['baseMVA'], base_ppc['bus'], base_ppc['branch'])

    # Net data
    Sbase = base_ppc['baseMVA'] # Sbase from PYPOWER case
    n_bus = base_ppc['bus'].shape[0]
    n_gens = base_ppc['gen'].shape[0]
    n_loads = len(net.load) # Number of distinct load elements in pandapower's view
    
    # Extract min/max Pg per generator:
    pg_min_values = base_ppc['gen'][:, 9].reshape((n_gens, 1)) # PMIN (column 9)
    pg_max_values = base_ppc['gen'][:, 8].reshape((n_gens, 1)) # PMAX (column 8)
    
    qg_min_values = base_ppc['gen'][:, 4].reshape((n_gens, 1)) # QMIN (column 4)
    qg_max_values = base_ppc['gen'][:, 3].reshape((n_gens, 1)) # QMAX (column 3)

    # Extract min/max Vm per bus:
    vm_min_values = base_ppc['bus'][:, 11].reshape((n_bus, 1)) # VMIN
    vm_max_values = base_ppc['bus'][:, 12].reshape((n_bus, 1)) # VMAX

    print(f"System details: {n_bus} buses, {n_gens} generators, {n_loads} loads.")

    # Obtain nominal loads (from the loaded case's ppc)
    pd_nom = np.array(net.load['p_mw']).reshape(-1, 1) # [MW]
    qd_nom = np.array(net.load['q_mvar']).reshape(-1, 1) # [Mvar]
    loads_nominal = np.vstack([pd_nom, qd_nom]) # Combined P and Q nominal loads for each load element

    # Define load perturbation bounds (relative to nominal loads)
    lb_factor = 0.6 * np.ones(loads_nominal.shape[0])
    ub_factor = 1.0 * np.ones(loads_nominal.shape[0])

    # --- Output Collection (for NN labels) ---
    pg_tot = []
    qg_tot = []
    vm_tot = []
    pinj_tot = []
    qinj_tot = []
    vr_tot = []
    vi_tot = []
    
    # -------------------------------------------------------------
    
    # Get the internal PYPOWER case from the pandapower network
    base_ppc = net._ppc
    
    # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
    external_bus_numbers = net.bus.index.values # indices from .m file from pandapower
    internal_indices = base_ppc['bus'][:, 0].astype(int) #  internal inddices from pypower

    external_to_internal = dict(zip(external_bus_numbers, internal_indices))
    internal_to_external = dict(zip(internal_indices, external_bus_numbers))

    # ===================== Data Generation Loop (Corrected) =====================
    print(f"Solving {n_data_points} AC-OPF problems with PYPOWER...")

    i = 0
    pbar = tqdm(total=int(n_data_points), position=0, leave=True)

    # CORRECTED: Initialize a list to store all successful inputs
    all_X_nn_inputs = []

    while i < n_data_points:
        # Generate a single load profile sample
        X_factors_sample = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, 1)
        X_unscaled_loads_mw = loads_nominal * X_factors_sample # 60% - 100% [MW, MVar]
        
        pd_tot_mw = X_unscaled_loads_mw[:n_loads, :]
        qd_tot_mvar = X_unscaled_loads_mw[n_loads:, :]
        
        X_loads_pu = X_unscaled_loads_mw / Sbase # 60% - 100% [p.u., p.u.]
        x = X_loads_pu
        X_nn_input = x.T
        

        try:
            current_ppc = copy.deepcopy(base_ppc)
            
            for load_idx_pp, bus_idx_pp_internal in net.load['bus'].items():
                original_matpower_bus_id = external_to_internal.get(bus_idx_pp_internal, None)
                if original_matpower_bus_id is None:
                    continue
                
                ppc_bus_row_idx = np.where(current_ppc['bus'][:, BUS_I] == original_matpower_bus_id)[0]
                if len(ppc_bus_row_idx) == 0:
                    continue
                ppc_bus_row_idx = ppc_bus_row_idx[0]

                current_ppc['bus'][ppc_bus_row_idx, PD] = pd_tot_mw[load_idx_pp, 0]
                current_ppc['bus'][ppc_bus_row_idx, QD] = qd_tot_mvar[load_idx_pp, 0]
            
            results = runopf(current_ppc, ppopt)
            success = (results['success'] == 1)
            
            if success:
                # Store results for this single successful sample
                pg_tot.append(torch.tensor(results['gen'][:, PG] / Sbase, dtype=torch.float32).unsqueeze(1))
                qg_tot.append(torch.tensor(results['gen'][:, QG] / Sbase, dtype=torch.float32).unsqueeze(1))
                vm_tot.append(torch.tensor(results['bus'][:, VM], dtype=torch.float32).unsqueeze(1))
                
                vm = results['bus'][:, VM]
                va_rad = np.deg2rad(results['bus'][:, VA])
                vr = vm * np.cos(va_rad)
                vi = vm * np.sin(va_rad)
                
                V = results['bus'][:, VM] * np.exp(1j * np.deg2rad(results['bus'][:, VA]))
                I = Ybus @ V
                S = V * np.conj(I)
                
                pinj_tot.append(torch.tensor(np.real(S), dtype=torch.float32).unsqueeze(1))
                qinj_tot.append(torch.tensor(np.imag(S), dtype=torch.float32).unsqueeze(1))
                vr_tot.append(torch.tensor(vr, dtype=torch.float32).unsqueeze(1))
                vi_tot.append(torch.tensor(vi, dtype=torch.float32).unsqueeze(1))
                
                # CORRECTED: Append the successful input to our list
                all_X_nn_inputs.append(X_nn_input)
                
                i += 1
                pbar.update(1)

                # Incremental saving logic
                if (i % 1_000 == 0) and save_csv:
                    print(f"\nSaving checkpoint at {i} samples...")
                    
                    # Convert collected lists to NumPy arrays
                    pg_np = torch.cat(pg_tot, dim=1).numpy()
                    qg_np = torch.cat(qg_tot, dim=1).numpy()
                    vm_np = torch.cat(vm_tot, dim=1).numpy()
                    vr_np = torch.cat(vr_tot, dim=1).numpy()
                    vi_np = torch.cat(vi_tot, dim=1).numpy()
                    
                    # Process and save the data for the collected chunk
                    _process_and_save_data(
                        X_loads_all=np.vstack(all_X_nn_inputs),
                        pg_np=pg_np,
                        qg_np=qg_np,
                        vm_np=vm_np,
                        vr_tot=vr_np,
                        vi_tot=vi_np,
                        simulation_parameters=simulation_parameters,
                        append_data=append_data
                    )
                    
                    # Clear the lists to free memory
                    all_X_nn_inputs = []
                    pg_tot, qg_tot, vm_tot = [], [], []
                    pinj_tot, qinj_tot, vr_tot, vi_tot = [], [], [], []

            else:
                print(f"Warning: OPF did not converge for sample {i}. Generating new load profile and retrying.")
                
        except Exception as e:
            print(f"Warning: PYPOWER OPF failed for sample {i} with error: {e}. Generating new load profile and retrying.")

    pbar.close()

    # Save any remaining data that wasn't part of a full 1000-sample chunk
    if i % 1_000 != 0 and save_csv:
        print("\nSaving final remaining samples...")
        pg_np = torch.cat(pg_tot, dim=1).numpy()
        qg_np = torch.cat(qg_tot, dim=1).numpy()
        vm_np = torch.cat(vm_tot, dim=1).numpy()
        vr_np = torch.cat(vr_tot, dim=1).numpy()
        vi_np = torch.cat(vi_tot, dim=1).numpy()

        _process_and_save_data(
            X_loads_all=np.vstack(all_X_nn_inputs),
            pg_np=pg_np,
            qg_np=qg_np,
            vm_np=vm_np,
            vr_tot=vr_np,
            vi_tot=vi_np,
            simulation_parameters=simulation_parameters,
            append_data=append_data
        )

    print("Data generation complete.")
    return None, None # Returning None since data is saved to file

def _process_and_save_data(X_loads_all, pg_np, qg_np, vm_np, vr_tot, vi_tot, simulation_parameters, append_data):
    """
    Helper function to process and save data.
    """
    n_buses = simulation_parameters['general']['n_buses']
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if n_buses == 14:
        case_name = 'pglib_opf_case14_ieee.m'
    elif n_buses == 57:
        case_name = 'pglib_opf_case57_ieee.m'
    elif n_buses == 118:
        case_name = 'pglib_opf_case118_ieee.m'
    elif n_buses == 300:
        case_name = 'pglib_opf_case300_ieee.m'
    elif n_buses == 793:
        case_name = 'pglib_opf_case793_goc_cleaned.m'
    elif n_buses == 1354:
        case_name = 'pglib_opf_case1354_pegase.m'
    elif n_buses == 2869:
        case_name = 'pglib_opf_case2869_pegase.m'
    
    case_path = os.path.join(base_dir, 'pglib-opf', case_name)
    net = pc.from_mpc(case_path, casename_mpc_file=True)
    base_ppc = pc.to_ppc(net, init = 'flat')
    
    # Get masks
    pg_max_values = base_ppc['gen'][:, 8].reshape((base_ppc['gen'].shape[0], 1))
    qg_max_values = base_ppc['gen'][:, 3].reshape((base_ppc['gen'].shape[0], 1))
    slack_bus_indices = np.where(base_ppc['bus'][:, 1] == 3)[0]  # BUS_TYPE == 3 (slack)
    slack_gen_mask = np.isin(base_ppc['gen'][:, BUS_I], slack_bus_indices)
    pg_max_zero_mask = pg_max_values.flatten() < 1e-9
    qg_max_zero_mask = qg_max_values.flatten() < 1e-9

    gen_mask_to_remove = np.logical_or(pg_max_zero_mask, slack_gen_mask)
    gen_mask_to_keep = ~gen_mask_to_remove
    pg_np = pg_np[gen_mask_to_keep, :]
    
    qgen_mask_to_remove = np.logical_or(qg_max_zero_mask, slack_gen_mask)
    qgen_mask_to_keep = ~qgen_mask_to_remove
    qg_np = qg_np[qgen_mask_to_keep, :]
    
    gen_bus_indices = base_ppc['gen'][:, BUS_I].astype(int)
    mask = ~np.isin(gen_bus_indices, slack_bus_indices)
    gen_bus_indices_no_slack = gen_bus_indices[mask]
    vm_np = vm_np[gen_bus_indices_no_slack, :]
    
    combinations_to_save = ["pg_vm", "pg_qg", "vr_vi"]
    output_data_dir = os.path.join(
        base_dir,
        'data/ac_opf_data',
        str(n_buses),
        'Dataset'
    )
    os.makedirs(output_data_dir, exist_ok=True)

    for combination in combinations_to_save:
        if combination == "pg_vm":
            y_data = np.vstack([pg_np, vm_np]).T
            x_data = X_loads_all
        elif combination == "pg_qg":
            y_data = np.vstack([pg_np, qg_np]).T
            x_data = X_loads_all
        elif combination == "vr_vi":
            y_data = np.vstack([vr_tot, vi_tot]).T
            x_data = X_loads_all
        else:
            continue
        
        input_filename = os.path.join(output_data_dir, f"NN_input_{combination}.csv")
        output_filename = os.path.join(output_data_dir, f"NN_output_{combination}.csv")

        # Determine if header should be written
        write_header = not os.path.exists(input_filename) or not append_data

        # Save to CSV
        pd.DataFrame(x_data).to_csv(input_filename, mode='a' if append_data else 'w', index=False, header=write_header)
        pd.DataFrame(y_data).to_csv(output_filename, mode='a' if append_data else 'w', index=False, header=write_header)

        print(f"Saved checkpoint for combination '{combination}' with shapes: input {x_data.shape}, output {y_data.shape}")



# --- Example Usage (How to call this function) ---
if __name__ == "__main__":
    
    from data.ac_opf.create_example_parameters import create_example_parameters

    test_n_buses = 793 
    
    # Check if data_dir for parameters exists before attempting to load
    current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_data_dir = os.path.join(current_script_dir, f'ac_opf_data/{test_n_buses}')
    
    # Create directory if it does not exist
    if not os.path.exists(param_data_dir):
        os.makedirs(param_data_dir, exist_ok=True)
        print(f"Created missing directory: {param_data_dir}")
    else:
        print(f"Directory already exists: {param_data_dir}")
        
    # 1. Create/Load simulation parameters (from your provided function)
    simulation_params = create_example_parameters(test_n_buses)

    # 2. Generate training data using the new function
    generate_power_system_data(simulation_params, save_csv=True, append_data=True)
    # X_train_data, Y_train_data = generate_power_system_data(simulation_params)

    # print("\nGenerated Training Data Shapes:")
    # print(f"Scaled Load Profiles (X_train_data): {X_train_data.shape}")
    # print(f"Generator Dispatches (Y_train_data): {Y_train_data.shape}")

