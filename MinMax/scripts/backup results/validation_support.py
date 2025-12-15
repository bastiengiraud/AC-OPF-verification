import os
import numpy as np
import copy
import torch
from tqdm import tqdm
import pandas as pd
import json
import time
import torch.nn as nn

import pandapower.networks as pn
from pandapower.converter.pandamodels.to_pm import convert_to_pm_structure, dump_pm_json
from pandapower.converter.pandamodels.from_pm import read_pm_results_to_net

from pandapower.optimal_powerflow import OPFNotConverged

import logging
logger = logging.getLogger(__name__)

# Import pypower functions
# Make sure your pypower installation is accessible
try:
    from pypower.api import ppoption, runopf, runpf, makeYbus
    from pypower.idx_bus import BUS_I, PD, QD, VM, VMAX, VMIN, VA, BUS_TYPE, REF, GS, BS
    from pypower.idx_gen import PG, QG, VG, PMAX, PMIN, QMAX, QMIN, GEN_BUS
    from pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT, RATE_A
    from pypower.idx_cost import MODEL, NCOST, COST, POLYNOMIAL, PW_LINEAR
    from pypower.ext2int import ext2int
except ImportError:
    raise ImportError("PYPOWER is not installed or not in your Python path. "
                      "Please install it (e.g., pip install pypower).")

# Import pandapower for case loading (assuming you're using it for .m files)
try:
    import pandapower.converter as pc
    import pandapower as pp # You might need this for net.load etc.
except ImportError:
    raise ImportError("pandapower is not installed. Please install it (e.g., pip install pandapower).")

# Assuming load_scaling is a custom module you have
try:
    import loadsampling as ls
except ImportError:
    print("Warning: 'load_scaling' module not found. Please ensure it's in your path if needed for load generation.")



from neural_network.lightning_nn_crown import NeuralNetwork, OutputWrapper

import os
import sys

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

for subdir in ['data', 'models', 'scripts/utils', 'config']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))
    


def _initialize_result_tensors(n_bus: int, n_gens: int, n_branches: int, n_data_points: int) -> dict:
    """Initializes all torch.Tensor containers for collected results."""
    # This function is directly reusable. Added default n_data_points=1 for single run.
    return {
        'pg_tot': torch.zeros(n_gens, n_data_points, dtype=torch.float32),
        'qg_tot': torch.zeros(n_gens, n_data_points, dtype=torch.float32),
        'vm_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'pinj_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'qinj_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'vr_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'vi_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'Iinj_r_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'Iinj_i_tot': torch.zeros(n_bus, n_data_points, dtype=torch.float32),
        'Ibr_from_r_tot': torch.zeros(n_branches, n_data_points, dtype=torch.float32),
        'Ibr_from_i_tot': torch.zeros(n_branches, n_data_points, dtype=torch.float32),
        'Ibr_to_r_tot': torch.zeros(n_branches, n_data_points, dtype=torch.float32),
        'Ibr_to_i_tot': torch.zeros(n_branches, n_data_points, dtype=torch.float32),
        'cost_tot': torch.zeros(1, n_data_points, dtype=torch.float32),  
        'slack_tot': torch.zeros(1, n_data_points, dtype=torch.float32),  
        'sample_num': torch.zeros(1, n_data_points, dtype=torch.int32),  # To track sample indices
    }
    
def _limit_tensor(simulation_parameters):
    
    n_gens = simulation_parameters['general']['n_gbus']
    n_bus = simulation_parameters['general']['n_buses']
    n_branches = simulation_parameters['true_system']['n_line']
    
    pg_max = simulation_parameters['true_system']['Sg_max'][:n_gens] / 100
    qg_max = simulation_parameters['true_system']['Sg_max'][n_gens:] / 100
    qg_min = simulation_parameters['true_system']['qg_min'] / 100
    vm_max = simulation_parameters['true_system']['Volt_max']
    vn_min = simulation_parameters['true_system']['Volt_min']
    I_max = simulation_parameters['true_system']['I_max_pu']
    
    return {
        'pg_max': torch.tensor(pg_max, dtype=torch.float32),
        'qg_max': torch.tensor(qg_max, dtype=torch.float32),
        'qg_min': torch.tensor(qg_min.T, dtype=torch.float32),
        'vm_max': torch.tensor(vm_max, dtype=torch.float32),
        'vm_min': torch.tensor(vn_min, dtype=torch.float32),
        'I_max': torch.tensor(I_max, dtype=torch.float32),
    }
    
# def calculate_violations(solution_data, simulation_parameters):
#     """
#     Calculates the average and max violation for various quantities, including
#     nodal current balance. Converts any numpy.ndarray inputs to torch.Tensor internally.

#     Args:
#         solution_data (dict): Dictionary containing simulation results. Can have
#                               values as either numpy.ndarray or torch.Tensor.
#         simulation_parameters (dict): Dictionary containing system parameters.
#                                       Assumes 'Ybus_r' and 'Ybus_i' are present.

#     Returns:
#         dict: A dictionary with both average and maximum violations for 'pg', 'qg',
#               'vm', 'Ibr', and the new 'Ibal' (current balance).
#     """
#     # --- INTERNAL MODIFICATION: Convert all numpy arrays to tensors ---
#     for key, value in solution_data.items():
#         if isinstance(value, np.ndarray):
#             solution_data[key] = torch.from_numpy(value).float()
            
#     limits = _limit_tensor(simulation_parameters)
    
#     # Initialize the results dictionary
#     violations = {
#         'pg_up_avg_violation': torch.tensor(0.0),
#         'pg_up_max_violation': torch.tensor(0.0),
#         'pg_down_avg_violation': torch.tensor(0.0),
#         'pg_down_max_violation': torch.tensor(0.0),
#         'qg_up_avg_violation': torch.tensor(0.0),
#         'qg_up_max_violation': torch.tensor(0.0),
#         'qg_down_avg_violation': torch.tensor(0.0),
#         'qg_down_max_violation': torch.tensor(0.0),
#         'vm_up_avg_violation': torch.tensor(0.0),
#         'vm_up_max_violation': torch.tensor(0.0),
#         'vm_down_avg_violation': torch.tensor(0.0),
#         'vm_down_max_violation': torch.tensor(0.0),
#         'Ibr_avg_violation': torch.tensor(0.0),
#         'Ibr_max_violation': torch.tensor(0.0),
#         'Ibal_avg_violation': torch.tensor(0.0), # New key for current balance
#         'Ibal_max_violation': torch.tensor(0.0), # New key for current balance
#     }
    
#     # --- 1. Generator Active Power (Pg) Violation ---
#     pg_actual = solution_data['pg_tot']
#     pg_max_expanded = limits['pg_max']
#     pg_upper_violation = torch.relu(pg_actual - pg_max_expanded)
#     pg_lower_violation = torch.relu(-pg_actual)
    
#     violations['pg_up_avg_violation'] = torch.mean(pg_upper_violation)
#     violations['pg_up_max_violation'] = torch.max(pg_upper_violation)
    
#     violations['pg_down_avg_violation'] = torch.mean(pg_lower_violation)
#     violations['pg_down_max_violation'] = torch.max(pg_lower_violation)
    
#     # --- 2. Generator Reactive Power (Qg) Violation ---
#     qg_actual = solution_data['qg_tot']
#     qg_max_expanded = limits['qg_max']
#     qg_min_expanded = limits['qg_min']
#     qg_upper_violation = torch.relu(qg_actual - qg_max_expanded)
#     qg_lower_violation = torch.relu(qg_min_expanded - qg_actual)
    
#     violations['qg_up_avg_violation'] = torch.mean(qg_upper_violation)
#     violations['qg_up_max_violation'] = torch.max(qg_upper_violation)
    
#     violations['qg_down_avg_violation'] = torch.mean(qg_lower_violation)
#     violations['qg_down_max_violation'] = torch.max(qg_lower_violation)
    
#     # --- 3. Voltage Magnitude (Vm) Violation ---
#     vm_actual = solution_data['vm_tot']
#     vm_max_expanded = limits['vm_max'].unsqueeze(1)
#     vm_min_expanded = limits['vm_min'].unsqueeze(1)
    
#     vm_upper_violation = torch.relu(vm_actual - vm_max_expanded)
#     vm_lower_violation = torch.relu(vm_min_expanded - vm_actual)
    
#     vm_up_violation = vm_upper_violation 
#     vm_down_violation = vm_lower_violation
    
#     violations['vm_up_avg_violation'] = torch.mean(vm_up_violation)
#     violations['vm_up_max_violation'] = torch.max(vm_up_violation)
    
#     violations['vm_down_avg_violation'] = torch.mean(vm_down_violation)
#     violations['vm_down_max_violation'] = torch.max(vm_down_violation)
    
#     # --- 4. Branch Current Magnitude (Ibr) Violation ---
#     I_from_r = solution_data['Ibr_from_r_tot']
#     I_from_i = solution_data['Ibr_from_i_tot']
#     I_to_r = solution_data['Ibr_to_r_tot']
#     I_to_i = solution_data['Ibr_to_i_tot']
    
#     I_from_mag = torch.sqrt(I_from_r**2 + I_from_i**2)
#     I_to_mag = torch.sqrt(I_to_r**2 + I_to_i**2)
    
#     I_max_expanded = limits['I_max'].unsqueeze(1)
    
#     I_from_violation = torch.relu(I_from_mag - I_max_expanded)
#     I_to_violation = torch.relu(I_to_mag - I_max_expanded)
    
#     all_Ibr_violations = torch.cat((I_from_violation, I_to_violation), dim=0)
    
#     violations['Ibr_avg_violation'] = torch.mean(all_Ibr_violations)
#     violations['Ibr_max_violation'] = torch.max(all_Ibr_violations)

#     # --- 5. Nodal Current Balance (Ibal) Violation ---
#     # Retrieve Ybus from simulation parameters and convert to tensor
#     kcl_im = torch.tensor(simulation_parameters['true_system']['kcl_im'], dtype=torch.float32)
#     bs_values = torch.tensor(simulation_parameters['true_system']['bus_bs'], dtype=torch.float64).unsqueeze(1)
#     kcl_from_im = torch.relu(kcl_im) # +1 at from-bus, 0 elsewhere
#     kcl_to_im = -torch.relu(-kcl_im) # +1 at to-bus, 0 elsewhere

#     # branch_injections_r = torch.matmul(kcl_im, Ibr_from_r_tot[:, i]) #+ torch.matmul(kcl_im, Ibr_to_r_tot[:, i])
#     # branch_injections_i = torch.matmul(kcl_im, Ibr_from_i_tot[:, i]) #+ torch.matmul(kcl_im, Ibr_to_i_tot[:, i])

#     branch_injections_r = torch.matmul(kcl_from_im, I_from_r) - torch.matmul(kcl_to_im, I_to_r) 
#     branch_injections_i = torch.matmul(kcl_from_im, I_from_i) - torch.matmul(kcl_to_im, I_to_i)

#     # Get bus voltages and injected currents
#     vr_tot = solution_data['vr_tot']
#     vi_tot = solution_data['vi_tot']
#     Iinj_r_tot = solution_data['Iinj_r_tot']
#     Iinj_i_tot = solution_data['Iinj_i_tot']
    
#     I_shunt_r = -bs_values * vi_tot 
#     I_shunt_i = bs_values * vr_tot 
    
#     # Calculate the current imbalance (I_injected - Ybus * V)
#     delta_I_r = Iinj_r_tot - branch_injections_r - I_shunt_r
#     delta_I_i = Iinj_i_tot - branch_injections_i - I_shunt_i
    
#     # The violation is the magnitude of the imbalance
#     Ibal_violation = torch.sqrt(delta_I_r**2 + delta_I_i**2)

#     violations['Ibal_avg_violation'] = torch.mean(Ibal_violation)
#     violations['Ibal_max_violation'] = torch.max(Ibal_violation)
    
#     return violations


def calculate_violations(solution_data, simulation_parameters):
    """
    Calculates the average and max violation for various quantities, including
    nodal current balance. Only considers samples where pf_convergence == 1.
    """
    # Convert numpy arrays to tensors if needed
    for key, value in solution_data.items():
        if isinstance(value, np.ndarray):
            solution_data[key] = torch.from_numpy(value).float()

    limits = _limit_tensor(simulation_parameters)

    # Create a mask for only converged samples
    pf_conv_mask = solution_data['pf_convergence'].flatten() == 1  # boolean tensor

    # If no sample converged, return zeros
    if pf_conv_mask.sum() == 0:
        return {key: torch.tensor(1000.0) for key in [
            'pg_up_avg_violation', 'pg_up_max_violation',
            'pg_down_avg_violation', 'pg_down_max_violation',
            'qg_up_avg_violation', 'qg_up_max_violation',
            'qg_down_avg_violation', 'qg_down_max_violation',
            'vm_up_avg_violation', 'vm_up_max_violation',
            'vm_down_avg_violation', 'vm_down_max_violation',
            'Ibr_avg_violation', 'Ibr_max_violation',
            'Ibal_avg_violation', 'Ibal_max_violation'
        ]}

    # --- 1. Generator Active Power (Pg) Violation ---
    pg_actual = solution_data['pg_tot'][:, pf_conv_mask]
    pg_max_expanded = limits['pg_max']
    pg_upper_violation = torch.relu(pg_actual - pg_max_expanded)
    pg_lower_violation = torch.relu(-pg_actual)

    # --- 2. Generator Reactive Power (Qg) Violation ---
    qg_actual = solution_data['qg_tot'][:, pf_conv_mask]
    qg_max_expanded = limits['qg_max']
    qg_min_expanded = limits['qg_min']
    qg_upper_violation = torch.relu(qg_actual - qg_max_expanded)
    qg_lower_violation = torch.relu(qg_min_expanded - qg_actual)

    # --- 3. Voltage Magnitude (Vm) Violation ---
    vm_actual = solution_data['vm_tot'][:, pf_conv_mask]
    vm_max_expanded = limits['vm_max'].unsqueeze(1)
    vm_min_expanded = limits['vm_min'].unsqueeze(1)
    vm_upper_violation = torch.relu(vm_actual - vm_max_expanded)
    vm_lower_violation = torch.relu(vm_min_expanded - vm_actual)

    # --- 4. Branch Current Magnitude (Ibr) Violation ---
    I_from_r = solution_data['Ibr_from_r_tot'][:, pf_conv_mask]
    I_from_i = solution_data['Ibr_from_i_tot'][:, pf_conv_mask]
    I_to_r = solution_data['Ibr_to_r_tot'][:, pf_conv_mask]
    I_to_i = solution_data['Ibr_to_i_tot'][:, pf_conv_mask]
    I_from_mag = torch.sqrt(I_from_r**2 + I_from_i**2)
    I_to_mag = torch.sqrt(I_to_r**2 + I_to_i**2)
    I_max_expanded = limits['I_max'].unsqueeze(1)
    I_from_violation = torch.relu(I_from_mag - I_max_expanded)
    I_to_violation = torch.relu(I_to_mag - I_max_expanded)
    all_Ibr_violations = torch.cat((I_from_violation, I_to_violation), dim=0)

    # --- 5. Nodal Current Balance (Ibal) Violation ---
    kcl_im = torch.tensor(simulation_parameters['true_system']['kcl_im'], dtype=torch.float32)
    bs_values = torch.tensor(simulation_parameters['true_system']['bus_bs'], dtype=torch.float32).unsqueeze(1)
    kcl_from_im = torch.relu(kcl_im)
    kcl_to_im = -torch.relu(-kcl_im)

    vr_tot = solution_data['vr_tot'][:, pf_conv_mask]
    vi_tot = solution_data['vi_tot'][:, pf_conv_mask]
    Iinj_r_tot = solution_data['Iinj_r_tot'][:, pf_conv_mask]
    Iinj_i_tot = solution_data['Iinj_i_tot'][:, pf_conv_mask]

    I_shunt_r = -bs_values * vi_tot
    I_shunt_i = bs_values * vr_tot
    branch_injections_r = torch.matmul(kcl_from_im, I_from_r) - torch.matmul(kcl_to_im, I_to_r)
    branch_injections_i = torch.matmul(kcl_from_im, I_from_i) - torch.matmul(kcl_to_im, I_to_i)
    delta_I_r = Iinj_r_tot - branch_injections_r - I_shunt_r
    delta_I_i = Iinj_i_tot - branch_injections_i - I_shunt_i
    Ibal_violation = torch.sqrt(delta_I_r**2 + delta_I_i**2)

    # --- Aggregate results ---
    violations = {
        'pg_up_avg_violation': torch.mean(pg_upper_violation),
        'pg_up_max_violation': torch.max(pg_upper_violation),
        'pg_down_avg_violation': torch.mean(pg_lower_violation),
        'pg_down_max_violation': torch.max(pg_lower_violation),
        'qg_up_avg_violation': torch.mean(qg_upper_violation),
        'qg_up_max_violation': torch.max(qg_upper_violation),
        'qg_down_avg_violation': torch.mean(qg_lower_violation),
        'qg_down_max_violation': torch.max(qg_lower_violation),
        'vm_up_avg_violation': torch.mean(vm_upper_violation),
        'vm_up_max_violation': torch.max(vm_upper_violation),
        'vm_down_avg_violation': torch.mean(vm_lower_violation),
        'vm_down_max_violation': torch.max(vm_lower_violation),
        'Ibr_avg_violation': torch.mean(all_Ibr_violations),
        'Ibr_max_violation': torch.max(all_Ibr_violations),
        'Ibal_avg_violation': torch.mean(Ibal_violation),
        'Ibal_max_violation': torch.max(Ibal_violation),
    }

    return violations
    
    
def _process_and_store_opf_results(results: dict, Sbase: float, Ybus_dense: np.ndarray, result_tensors: dict, entry_idx: int, external_to_ppc_row_idx: dict):
    """
    Extracts and stores relevant data from a successful OPF solution.
    This function can be reused directly, as long as 'results' dict
    is formatted like PYPOWER's runopf output.
    """
    # Unpack for direct assignment within this helper
    pg_tot, qg_tot, vm_tot, pinj_tot, qinj_tot, vr_tot, vi_tot, \
    Iinj_r_tot, Iinj_i_tot, Ibr_from_r_tot, Ibr_from_i_tot, Ibr_to_r_tot, Ibr_to_i_tot, cost_tot, slack_tot, sample_num = \
        (result_tensors[key] for key in result_tensors.keys()) # Dynamically unpack
        
    cost_tot[:, entry_idx] = results['f']  # Store the cost for this entry
    
    # Store generator and bus results
    pg_tot[:, entry_idx] = torch.tensor(results['gen'][:, PG], dtype=torch.float32) / Sbase
    qg_tot[:, entry_idx] = torch.tensor(results['gen'][:, QG], dtype=torch.float32) / Sbase
    vm_tot[:, entry_idx] = torch.tensor(results['bus'][:, VM], dtype=torch.float32)
    
    # Calculate complex voltages and their real/imaginary parts
    vm = results['bus'][:, VM]
    va_rad = np.deg2rad(results['bus'][:, VA])
    vr = vm * np.cos(va_rad)
    vi = vm * np.sin(va_rad)
    V_complex = vm * np.exp(1j * va_rad)
    
    # Calculate complex current injections I = Ybus * V
    I_complex = Ybus_dense @ V_complex
    Iinj_r_tot[:, entry_idx] = torch.tensor(np.real(I_complex), dtype=torch.float32)
    Iinj_i_tot[:, entry_idx] = torch.tensor(np.imag(I_complex), dtype=torch.float32)
    
    # Complex power injection S = V * conj(I)
    S_complex_injections = V_complex * np.conj(I_complex)
    pinj_tot[:, entry_idx] = torch.tensor(np.real(S_complex_injections), dtype=torch.float32)
    qinj_tot[:, entry_idx] = torch.tensor(np.imag(S_complex_injections), dtype=torch.float32)
    
    # Store rectangular voltage components
    vr_tot[:, entry_idx] = torch.tensor(vr, dtype=torch.float32)
    vi_tot[:, entry_idx] = torch.tensor(vi, dtype=torch.float32)
    
    # Branch current calculations
    branch_data = results['branch'] # Assuming results['branch'] is a numpy array similar to ppc['branch']
    fbus_external = branch_data[:, F_BUS]
    tbus_external = branch_data[:, T_BUS]

    # Convert external bus IDs to internal PyPower array indices
    # This part relies on external_to_ppc_row_idx
    fbus_internal_idx = np.array([external_to_ppc_row_idx[int(bus_id)] for bus_id in fbus_external])
    tbus_internal_idx = np.array([external_to_ppc_row_idx[int(bus_id)] for bus_id in tbus_external])

    V_f_complex = V_complex[fbus_internal_idx]
    V_t_complex = V_complex[tbus_internal_idx]

    # These power flows are typically in MVA base from runopf results
    S_f_complex_mva = branch_data[:, PF] + 1j * branch_data[:, QF]
    S_t_complex_mva = branch_data[:, PT] + 1j * branch_data[:, QT]

    S_f_complex_pu = S_f_complex_mva / Sbase
    S_t_complex_pu = S_t_complex_mva / Sbase

    # Calculate complex currents using I = S* / V*
    I_br_from_complex = np.divide(np.conj(S_f_complex_pu), np.conj(V_f_complex),
                                  out=np.zeros_like(S_f_complex_pu, dtype=complex),
                                  where=np.abs(V_f_complex) > 1e-9)
    I_br_to_complex = np.divide(np.conj(S_t_complex_pu), np.conj(V_t_complex),
                                out=np.zeros_like(S_t_complex_pu, dtype=complex),
                                where=np.abs(V_t_complex) > 1e-9)

    Ibr_from_r_tot[:, entry_idx] = torch.tensor(np.real(I_br_from_complex), dtype=torch.float32)
    Ibr_from_i_tot[:, entry_idx] = torch.tensor(np.imag(I_br_from_complex), dtype=torch.float32)
    Ibr_to_r_tot[:, entry_idx] = torch.tensor(np.real(I_br_to_complex), dtype=torch.float32)
    Ibr_to_i_tot[:, entry_idx] = torch.tensor(np.imag(I_br_to_complex), dtype=torch.float32)
    
    # get active power on slack bus
    bus_data = results['bus']
    gen_data = results['gen']

    
    slack_bus_idx = np.where(bus_data[:, BUS_TYPE] == REF)[0][0] # The slack bus has a BUS_TYPE of REF (which is 3)
    slack_bus_number = int(bus_data[slack_bus_idx, 0]) # Get the actual bus number from the `BUS_I` column
    gen_on_slack_idx = np.where(gen_data[:, GEN_BUS] == slack_bus_number)[0][0] # The GEN_BUS column in the gen matrix stores the bus number

    # Get the active power output (PG) for that generator ---
    slack_pg = gen_data[gen_on_slack_idx, PG]
    slack_tot[:, entry_idx] = slack_pg / Sbase  
    
    # sample number
    sample_num[:, entry_idx] = entry_idx
    
    
def _filter_generator_data(gen_data_tensor: torch.Tensor, ppc_bus_data: np.ndarray, base_ppc) -> torch.Tensor:
    """Removes data for slack buses from generator tensors."""
    slack_bus_indices = np.where(ppc_bus_data[:, BUS_TYPE] == 3)[0] # BUS_TYPE == 3 (slack)
    
    gen_bus_ids_in_ppc = ppc_bus_data[base_ppc['gen'][:, GEN_BUS].astype(int), BUS_I]
    
    # Create a mask for generators connected to slack buses
    gen_mask_to_remove = np.isin(base_ppc['gen'][:, GEN_BUS], slack_bus_indices)
    gen_mask_to_keep = ~gen_mask_to_remove
    
    return gen_data_tensor[gen_mask_to_keep, :]



def solve_ac_opf_and_collect_data(seed, case_num: int, num_opf_solves: int) -> dict:
    """
    Solves multiple AC Optimal Power Flow (OPF) problems for a given case
    with varied load conditions and collects relevant solution data.

    Args:
        case_num (int): The number of buses for the case (e.g., 118, 300, 793).
        num_opf_solves (int): The number of AC OPF problems to solve (data points).

    Returns:
        dict: A dictionary containing the collected solution data:
              'pg_tot', 'qg_tot', 'vm_tot', 'pinj_tot', 'qinj_tot', 'vr_tot', 'vi_tot'.
              Each value is a torch.Tensor of shape (num_quantity, num_opf_solves).
              Returns zeros for failed OPF solves.
    """

    # --- Configuration and Setup ---
    n_buses = case_num
    n_data_points = num_opf_solves

    # Determine case name based on the number of buses
    case_name_map = {
        14: 'pglib_opf_case14_ieee.m',
        57: 'pglib_opf_case57_ieee.m',
        118: 'pglib_opf_case118_ieee.m',
        300: 'pglib_opf_case300_ieee.m',
        793: 'pglib_opf_case793_goc_cleaned.m',
        1354: 'pglib_opf_case1354_pegase.m',
        2869: 'pglib_opf_case2869_pegase.m'
    }

    case_name = case_name_map.get(n_buses)
    if case_name is None:
        raise ValueError(f"Unsupported case number: {n_buses}. "
                         f"Supported cases are: {list(case_name_map.keys())}")

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(current_script_dir)) # Go up two levels from current script dir
    base_dir = os.path.join(project_root_dir, 'pglib-opf') # Assuming pglib-opf is here

    case_path = os.path.join(base_dir, case_name)

    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Case file not found at: {case_path}. "
                                "Please ensure 'pglib-opf' repository is cloned and correctly located.")

    # Load the MATPOWER case
    net = pc.from_mpc(case_path, casename_mpc_file=True)
    base_ppc = pc.to_ppc(net, init='flat') # Get initial PPC for OPF solves
    
    
    # Extract constants from the base case
    Sbase = base_ppc['baseMVA']
    n_bus = base_ppc['bus'].shape[0]
    n_gens = base_ppc['gen'].shape[0]
    n_loads = len(net.load)
    n_branches = base_ppc['branch'].shape[0]
    print(f"System details: {n_bus} buses, {n_gens} generators, {n_loads} loads.")

    # Obtain nominal loads (from the loaded case's ppc)
    pd_nom = np.array(net.load['p_mw']).reshape(-1, 1)              # [MW]
    qd_nom = np.array(net.load['q_mvar']).reshape(-1, 1)            # [MVar]
    loads_nominal = np.vstack([pd_nom, qd_nom]) 

    # Define load perturbation bounds (relative to nominal loads)
    lb_factor = 0.6 * np.ones(loads_nominal.shape[0])
    ub_factor = 1.0 * np.ones(loads_nominal.shape[0])
    
    # --- Initialize Output Tensors ---
    result_tensors = _initialize_result_tensors(n_bus, n_gens, n_branches, n_data_points)
    timing_tensor = torch.zeros(n_data_points, dtype=torch.float32)

    # --- PYPOWER OPF Setup ---
    ppopt = ppoption(OUT_ALL=0, TOL=1e-5, MAX_IT=200) # Suppress verbose output from PYPOWER
    
    # Re-calculate Ybus here, as it depends on baseMVA, bus, branch from base_ppc
    Ybus, _, _ = makeYbus(base_ppc['baseMVA'], base_ppc['bus'], base_ppc['branch'])
    Ybus_dense = Ybus.toarray()

    # Get the internal PYPOWER case from the pandapower network
    initial_ppc = base_ppc.copy()  # Make a copy of the initial PPC for modification
    initial_net = copy.deepcopy(net)

    # 1. Create a map from external bus ID (from .m file) to its row index in base_ppc['bus']
    external_to_ppc_row_idx = {
        int(bus_data[BUS_I]): idx for idx, bus_data in enumerate(initial_ppc['bus'])
    }
    
    # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
    external_bus_numbers = net.bus.index.values # indices from .m file from pandapower
    internal_indices = base_ppc['bus'][:, 0].astype(int) #  internal inddices from pypower

    external_to_internal = dict(zip(external_bus_numbers, internal_indices))
    internal_to_external = dict(zip(internal_indices, external_bus_numbers))
    
    ppc_bus_number_to_row = {int(b): i for i, b in enumerate(base_ppc['bus'][:, BUS_I])}
    
    # ============ WARM-UP RUN ================
    # This first run initializes the Julia environment and compiles the code,
    # removing overhead from the timed loop below.
    print("Running a warm-up OPF solve to initialize Julia/PyCall...")
    try:
        # Use a generic initial network for the warm-up
        _, _ = runpm_opf(initial_net, pm_solver='ipopt')
        print("Warm-up complete. Starting timed data generation loop.")
    except Exception as e:
        print(f"Warning: Warm-up OPF failed. This may indicate a problem with the initial case. Error: {e}")
    
    # initialize counters
    count = 0
    entry = 0

    print(f"Solving {n_data_points} AC-OPF problems with PYPOWER (resampling if OPF fails)...")
    X_factor_old = None
    np.random.seed(seed)  # Set seed for reproducibility

    while count < n_data_points:
        current_ppc = copy.deepcopy(initial_ppc)  # Deep copy fresh case
        current_net = copy.deepcopy(initial_net)
        
        # === Sample a new load profile ===
        # Draw new factors (1 sample at a time)
        X_factor = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, 1).flatten() 
        
        if entry > 0 and np.array_equal(X_factor, X_factor_old):
            print(f"Resampled identical load factors at entry {entry}. Resampling again...")
            continue  # Skip to next iteration to resample
        
        X_factor_old = X_factor.copy()
        
        pd_sample = pd_nom.flatten() * X_factor[:n_loads]
        qd_sample = qd_nom.flatten() * X_factor[n_loads:]

        # Adjust loads in ppc
        for load_idx_pp, bus_idx_pp_external in net.load['bus'].items():
            # ppc_bus_row_idx = external_to_ppc_row_idx.get(bus_idx_pp_external)
            
            ppc_bus_row_idx = ppc_bus_number_to_row.get(external_to_internal.get(bus_idx_pp_external))
            if ppc_bus_row_idx is None:
                continue
            
            # with pypower
            current_ppc['bus'][ppc_bus_row_idx, PD] = float(pd_sample[load_idx_pp])
            current_ppc['bus'][ppc_bus_row_idx, QD] = float(qd_sample[load_idx_pp])

            
        # with powermodels
        current_net.load['p_mw'] = pd_sample
        current_net.load['q_mvar'] = qd_sample
                
        solved_net, results_pm = None, None
        success_pypower = False
        success_pandamodels = False
        
        # Try PyPower first
        try:
            results = runopf(current_ppc, ppopt)
            success_pypower = (results['success'] == 1)
        except Exception as e:
            print(f"PyPower OPF failed at attempt {count}. Error: {e}. Resampling...")

        # Only run pandamodels if PyPower succeeded
        if success_pypower:
            try:
                # pp.runpp(current_net, algorithm="nr", init="flat", enforce_q_lims=True, max_iteration=200)
                solved_net, results_pm = runpm_opf(current_net, pm_solver='ipopt')
                success_pandamodels = True
                print("Both PYPOWER and Pandamodels OPF succeeded.")
            except Exception as e:
                print(f"Pandamodels OPF failed at attempt {count}. Error: {e}. Resampling...")
    
        # # Try solving OPF pandamodels
        # try:           
        #     # ------------- with pandamodels ------------
        #     solved_net, results_pm = runpm_opf(current_net, pm_solver='ipopt')
        #     success_pandamodels = True
            
        # except Exception as e:
        #     print(f"OPF failed at attempt {entry}. Error: {e}. Resampling...")
        #     success_pandamodels = False
            
        # # Try solving OPF pandamodels
        # try:
        #     results = runopf(current_ppc, ppopt)
        #     success_pypower = (results['success'] == 1)
            
        # except Exception as e:
        #     print(f"OPF failed at attempt {entry}. Error: {e}. Resampling...")
        #    success_pypower = False
            
        
        if success_pandamodels and success_pypower:
            # Store results
            _process_and_store_opf_results(results, Sbase, Ybus_dense, result_tensors, count, external_to_ppc_row_idx)
            timing_tensor[count] = results_pm['solve_time']
            # Also store loads (so they match nn_input!)
            if count == 0:
                X_nn_input = []
                pd_tot_list = []
                qd_tot_list = []
            X_nn_input.append(np.hstack([pd_sample, qd_sample]) / Sbase)
            pd_tot_list.append(pd_sample / Sbase)
            qd_tot_list.append(qd_sample / Sbase)

            count += 1
        else:
            # Don’t increment counter → resample in next loop
            continue

        

    # --- Post-processing: Remove slack bus generators ---
    result_tensors['pg_tot'] = _filter_generator_data(result_tensors['pg_tot'], base_ppc['bus'], base_ppc)
    result_tensors['qg_tot'] = _filter_generator_data(result_tensors['qg_tot'], base_ppc['bus'], base_ppc)
    
    # After the OPF loop, convert lists → arrays
    X_nn_input = np.vstack(X_nn_input)              # shape: (num_samples, 2*n_loads)
    pd_tot_arr = np.vstack(pd_tot_list)             # shape: (num_samples, n_loads)
    qd_tot_arr = np.vstack(qd_tot_list)             # shape: (num_samples, n_loads)

    # Then convert to torch tensors
    solution_data = {
        'nn_input': torch.tensor(X_nn_input, dtype=torch.float32),
        'pd_tot': torch.tensor(pd_tot_arr.T, dtype=torch.float32),
        'qd_tot': torch.tensor(qd_tot_arr.T, dtype=torch.float32),
        'solve_time': timing_tensor
    }
    solution_data.update(result_tensors) # Add all collected tensors    

    print(f"Finished solving {n_data_points} AC-OPF problems for case {n_buses}.")
    return solution_data


# # ============ stupid fix, for now doing OPFs first for time ================
#     print(f"Solving {n_data_points} AC-OPF problems with PYPOWER...")
#     for entry in tqdm(range(n_data_points), position=0, leave=True):
        
#         #----------------- for pandamodels------------
#         current_net = copy.deepcopy(initial_net) # Deep copy the net for each iteration
        
#         # Adjust loads in the current_net using pandapower syntax
#         # for load_idx_pp, bus_idx_pp_external in current_net.load['bus'].items():
#         #     current_net.load.loc[load_idx_pp, 'p_mw'] = pd_tot_mw_data[load_idx_pp, entry]
#         #     current_net.load.loc[load_idx_pp, 'q_mvar'] = qd_tot_mvar_data[load_idx_pp, entry]
        
#         current_net.load['p_mw'] = pd_tot_mw_data[:, entry]
#         current_net.load['q_mvar'] = qd_tot_mvar_data[:, entry]

            
#         solved_net, results_pm = None, None
        
#         # Run the OPF for timing
#         try:           
#             # ------------- with pandamodels ------------
#             solved_net, results_pm = runpm_opf(current_net, pm_solver='ipopt')
            
#         except Exception as e:
#             print(f"Warning: PYPOWER OPF failed for entry {entry}. Error: {e}. Storing zeros.")
#             success = False # Explicitly set to False if an exception occurs
            
#         timing_tensor[entry] = results_pm['solve_time']
    
#     # ============ Data Generation Loop (using PYPOWER) ================
#     print(f"Solving {n_data_points} AC-OPF problems with PYPOWER...")
#     for entry in tqdm(range(n_data_points), position=0, leave=True):
#         current_ppc = copy.deepcopy(initial_ppc) # Make a deep copy for each iteration
            
#         # ---------------- for pypower ------------------
#         # Adjust loads in the current_ppc
#         for load_idx_pp, bus_idx_pp_external in net.load['bus'].items():
#             # bus_idx_pp_external is the original bus ID from the .m file
#             ppc_bus_row_idx = external_to_ppc_row_idx.get(bus_idx_pp_external)
            
#             if ppc_bus_row_idx is None:
#                 # This warning might indicate a problem with your bus ID handling
#                 # or if some loads are connected to buses not in the ppc (unlikely for valid cases).
#                 print(f"Warning: Bus {bus_idx_pp_external} (from pandapower load {load_idx_pp}) "
#                       f"not found in PYPOWER bus matrix. Skipping load adjustment for this bus.")
#                 continue
            
#             # Adjust loads
#             current_ppc['bus'][ppc_bus_row_idx, PD] = pd_tot_mw_data[load_idx_pp, entry] # / Sbase (if you want pu)
#             current_ppc['bus'][ppc_bus_row_idx, QD] = qd_tot_mvar_data[load_idx_pp, entry] # / Sbase (if you want pu)
        
#         # --- Time the OPF solve ---
#         start_time = time.perf_counter()
        
#         # Run the OPF with PYPOWER
#         try:
#             # -------------- with pypower -------------
#             results = runopf(current_ppc, ppopt)
#             success = (results['success'] == 1)
            
#         except Exception as e:
#             print(f"Warning: PYPOWER OPF failed for entry {entry}. Error: {e}. Storing zeros.")
#             success = False # Explicitly set to False if an exception occurs
            
#         end_time = time.perf_counter()
        
#         # Store results or zeros if OPF did not converge/failed
#         if success:
#             _process_and_store_opf_results(results, Sbase, Ybus_dense, result_tensors, entry, external_to_ppc_row_idx)
            
#         else:
#             pass 

    
    
    
def load_and_prepare_voltage_nn_for_inference(nn_file_name: str, case_num: int, config, simulation_parameters, solution_data) -> (nn.Module, dict):
    """
    Loads a pre-trained neural network model and prepares it for inference.
    It handles model loading, surrogate model loading, and normalization.

    """
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    n_buses = case_num
    
    # Access parameters from the provided simulation_parameters
    true_system_params = simulation_parameters['true_system']
    general_params = simulation_parameters['general']
    data_creation_params = simulation_parameters['data_creation']
    
    # solution data
    sd_test = solution_data['nn_input']
    num_classes = n_buses * 2

    # --- Data Statistics for Normalization ---
    sd_min = torch.tensor(true_system_params['Sd_min']).float()#.to(device)
    sd_delta = torch.tensor(true_system_params['Sd_delta']).float()#.to(device)

    vrvi_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float()
    vrvi_min = -vrvi_max
    vrvi_delta = vrvi_max - vrvi_min
    vrvi_delta[vrvi_delta <= 1e-12] = 1.0 # Avoid division by zero

    data_stat = {
        'sd_min': sd_min,
        'sd_delta': sd_delta,
        'vrvi_min': vrvi_min,
        'vrvi_delta': vrvi_delta,
    }   

    # --- Build and Normalize Network ---
    network_gen = build_network('vr_vi', sd_test.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed,
                                simulation_parameters) # Pass full simulation_parameters
    
    # Step 2: Construct the path
    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = nn_file_name # f"checkpoint_{n_buses}_{config.hidden_layer_size}_False_vr_vi.pt"
    path_dir = os.path.join(model_save_directory, path)

    # Step 3: Load the saved weights
    network_gen = torch.load(path_dir, map_location=torch.device('cpu'))
    network_gen.eval()  # set to evaluation mode if you're doing inference
    
    print(f"Neural network for case {n_buses} loaded and prepared for inference.")
    return network_gen


def load_and_prepare_power_nn_for_inference(nn_file_name: str, case_num: int, config, simulation_parameters, solution_data) -> (nn.Module, dict):
    """
    Loads a pre-trained neural network model and prepares it for inference.
    It handles model loading, surrogate model loading, and normalization.

    """

    n_buses = case_num
    n_gens = simulation_parameters['general']['n_gbus']
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # system
    base_ppc = simulation_parameters['net_object']
    slack_bus_indices = np.where(base_ppc['bus'][:, 1] == 3)[0]  # BUS_TYPE == 3 (slack)
    gen_bus_indices = base_ppc['gen'][:, 0].astype(int)  # buses with generators
    mask = ~np.isin(gen_bus_indices, slack_bus_indices)
    gen_bus_indices_no_slack = gen_bus_indices[mask]
    

    # solution data
    sd_test = solution_data['nn_input']
    
    # generator min max
    pg_max_zero_mask = simulation_parameters['true_system']['Sg_max'][:n_gens] < 1e-9
    gen_mask_to_keep = ~pg_max_zero_mask  # invert mask to keep desired generators
    gen_delta = torch.tensor(simulation_parameters['true_system']['Sg_delta'][:n_gens][gen_mask_to_keep]).float().unsqueeze(1) / 100
    gen_min = torch.zeros_like(gen_delta)
    
    # voltage min max   
    volt_min = torch.tensor(simulation_parameters['true_system']['Volt_min']).float().unsqueeze(1)[gen_bus_indices_no_slack]
    volt_max = torch.tensor(simulation_parameters['true_system']['Volt_max']).float().unsqueeze(1)[gen_bus_indices_no_slack]
    volt_delta = volt_max - volt_min
    
    output_min = torch.vstack((gen_min, volt_min))
    output_delta = torch.vstack((gen_delta, volt_delta))
    
    num_classes = len(output_min)
    
    dem_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float()
    dem_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float()

    data_stat = {
        'gen_min': output_min,
        'gen_delta': output_delta,
        'dem_min': dem_min,
        'dem_delta': dem_delta,
    }
    
    # --- Build and Normalize Network ---
    network_gen = build_network('pg_vm', sd_test.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed,
                                simulation_parameters) # Pass full simulation_parameters
    
    # Step 2: Construct the path
    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = nn_file_name # f"checkpoint_{n_buses}_{config.hidden_layer_size}_False_pg_vm.pt"
    path_dir = os.path.join(model_save_directory, path)

    # Step 3: Load the saved weights
    network_gen = torch.load(path_dir, map_location=torch.device('cpu'))
    network_gen.eval()  # set to evaluation mode if you're doing inference
    
    print(f"Neural network for case {n_buses} loaded and prepared for inference.")
    return network_gen




def voltage_nn_inference(net, case_num, model, solution_data, simulation_parameters):
    
    n_buses = case_num
    
    # Access parameters from the provided simulation_parameters
    true_system_params = simulation_parameters['true_system']
    pv_buses = simulation_parameters['true_system']['pv_buses']
    
    # Add the cost to the solution_data dictionary
    gen_costs_table = net.poly_cost.loc[net.poly_cost['et'] == 'gen']
    generator_costs = np.array(gen_costs_table['cp1_eur_per_mw'].values)
    
    extgrid_costs_table = net.poly_cost.loc[net.poly_cost['et'] == 'ext_grid']
    extgrid_costs = np.array(extgrid_costs_table['cp1_eur_per_mw'].values)
    
    # solution data
    sd_test = torch.tensor(solution_data['nn_input']).float()# .to(device)
    n_loads = sd_test.shape[1] // 2
    n_samples = sd_test.shape[0]
    pd_tot = sd_test[:, :n_loads]
    qd_tot = sd_test[:, n_loads:]

    # --- Data Statistics for Normalization ---
    sd_min = torch.tensor(true_system_params['Sd_min']).float()#.to(device)
    sd_delta = torch.tensor(true_system_params['Sd_delta']).float()#.to(device)

    vrvi_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float()
    vrvi_min = -vrvi_max
    vrvi_delta = vrvi_max - vrvi_min
    vrvi_delta[vrvi_delta <= 1e-12] = 1.0 # Avoid division by zero

    data_stat = {
        'sd_min': sd_min,
        'sd_delta': sd_delta,
        'vrvi_min': vrvi_min,
        'vrvi_delta': vrvi_delta,
    }   
    
    # inference
    n_loads = sd_test.shape[1] // 2
    num_samples = sd_test.shape[0]
    n_bus = simulation_parameters['general']['n_buses']
    n_lines = simulation_parameters['true_system']['n_line']
    f_bus = torch.tensor(simulation_parameters['true_system']['fbus'], dtype=torch.float32)
    t_bus = torch.tensor(simulation_parameters['true_system']['tbus'], dtype=torch.float32)
    
    # Y bus
    Ybr_rect = torch.tensor(simulation_parameters['true_system']['Ybr_rect'], dtype=torch.float32)
    Ybus = torch.tensor(simulation_parameters['true_system']['Ybus'], dtype=torch.complex64)
    Ybus_real = Ybus.real
    Ybus_imag = Ybus.imag
    map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float32)
    
    Gen_output = model.forward_train(sd_test)
    
    # Compute magnitude from surrogate 
    Ibr =  Gen_output @ Ybr_rect.T 
    
    # first half is 'from' currents
    I_f = Ibr[:, :2 * n_lines]
    Ir_f = I_f[:, :n_lines]
    Ii_f = I_f[:, n_lines:2*n_lines]
    I_mag_f = torch.sqrt(Ir_f**2 + Ii_f**2)

    # second half is 'to' currents
    I_t = Ibr[:, 2 * n_lines:]
    Ir_t = I_t[:, :n_lines]
    Ii_t = I_t[:, n_lines:]
    I_mag_t = torch.sqrt(Ir_t**2 + Ii_t**2)
    
    I_mag_check = torch.cat((I_mag_f, I_mag_t), dim=1)
        
    vr = Gen_output[:, :n_bus]
    vi = Gen_output[:, n_bus:]
    
    Ir = torch.matmul(vr, Ybus_real.T) - torch.matmul(vi, Ybus_imag.T)
    Ii = torch.matmul(vr, Ybus_imag.T) + torch.matmul(vi, Ybus_real.T)
    
    pinj = vr * Ir + vi * Ii
    qinj = vi * Ir - vr * Ii
    
    pd = sd_test[:, :n_loads]
    qd = sd_test[:, n_loads:]
    
    pg = pinj + torch.einsum('bi,ij->bj', pd, map_l[:n_loads, :n_bus])
    qg = qinj + torch.einsum('bi,ij->bj', qd, map_l[n_loads:, n_bus:])
        
    # Penalize line flow violations
    V_complex = vr + 1j * vi
    vm_tot = torch.abs(V_complex)
    
    # Compute branch currents
    I_complex_f = Ir_f + 1j * Ii_f  # (batch_size, n_lines)
    I_complex_t = Ir_t + 1j * Ii_t  # (batch_size, n_lines)

    # Voltages at from and to buses
    V_from = V_complex[:, f_bus.long()]  # shape: (batch_size, n_bus)
    V_to = V_complex[:, t_bus.long()]    # shape: (batch_size, n_bus)

    # Complex power flows S = V * conj(I)
    S_from = V_from * torch.conj(I_complex_f)  # shape: (batch_size, n_lines)
    S_to = V_to * torch.conj(I_complex_t)      # shape: (batch_size, n_lines)
    
    S_mag_f = torch.abs(S_from)  # (batch_size, n_lines)
    S_mag_t = torch.abs(S_to)
  
    
    # Combine results into a dictionary
    solution_data = {
        'pd_tot': pd_tot.T, 
        'qd_tot': qd_tot.T, 
        'nn_input': sd_test.detach().cpu().numpy(), 
        'pg_tot': pg[:, pv_buses].T.detach().cpu().numpy(),
        'qg_tot': qg[:, pv_buses].T.detach().cpu().numpy(),
        'vm_tot': vm_tot.T.detach().cpu().numpy(),
        'pinj_tot': pinj.T.detach().cpu().numpy(),
        'qinj_tot': qinj.T.detach().cpu().numpy(),
        'vr_tot': vr.T.detach().cpu().numpy(),
        'vi_tot': vi.T.detach().cpu().numpy(),
        'Iinj_r_tot': Ir.T.detach().cpu().numpy(), # Add new tensors to the dictionary
        'Iinj_i_tot': Ii.T.detach().cpu().numpy(),
        'Ibr_from_r_tot': Ir_f.T.detach().cpu().numpy(),
        'Ibr_from_i_tot': Ii_f.T.detach().cpu().numpy(),
        'Ibr_to_r_tot': Ir_t.T.detach().cpu().numpy(),
        'Ibr_to_i_tot': Ii_t.T.detach().cpu().numpy(),
        'pf_convergence': np.ones(num_samples).reshape(1, -1),
        'sample_num': np.arange(num_samples).reshape(1, -1)

        
    }
    
    extgrid_mw = pd_tot[:,:].sum(dim=1) - pg[:,:].sum(dim=1)
    extgrid = extgrid_costs * extgrid_mw.unsqueeze(1).detach().numpy() * 100

    tot_cost = []
    for i in range(n_samples):
        sample_cost = calculate_total_cost(generator_costs, solution_data['pg_tot'][:, i].T * 100, extgrid[i])
        tot_cost.append(sample_cost)

    solution_data['cost_tot'] = np.vstack(tot_cost)
    solution_data['slack_tot'] = extgrid_mw.unsqueeze(1).detach().numpy()
 
    return solution_data




def power_nn_inference(net, case_num, model, solution_data, simulation_parameters):
    
    n_buses = case_num
    n_gens = simulation_parameters['general']['n_gbus']
    
    # Access parameters from the provided simulation_parameters
    true_system_params = simulation_parameters['true_system']
    Ybus = torch.tensor(simulation_parameters['true_system']['Ybus'], dtype=torch.complex128)
    bs_values = torch.tensor(simulation_parameters['true_system']['bus_bs'], dtype=torch.float64)
    
    # solution data
    sd_test = torch.tensor(solution_data['nn_input']).float()# .to(device)
    n_loads = sd_test.shape[1] // 2
    pd_tot = sd_test[:, :n_loads]
    qd_tot = sd_test[:, n_loads:]
    num_samples = sd_test.shape[0]
    Sbase = 100

    # --- Data Statistics for Normalization ---
    # system
    base_ppc = simulation_parameters['net_object']
    slack_bus_indices = np.where(base_ppc['bus'][:, 1] == 3)[0]  # BUS_TYPE == 3 (slack)
    gen_bus_indices = base_ppc['gen'][:, 0].astype(int)  # buses with generators
    mask = ~np.isin(gen_bus_indices, slack_bus_indices)
    gen_bus_indices_no_slack = gen_bus_indices[mask]
    
    # Add the cost to the solution_data dictionary
    gen_costs_table = net.poly_cost.loc[net.poly_cost['et'] == 'gen']
    generator_costs = np.array(gen_costs_table['cp1_eur_per_mw'].values)
    
    extgrid_costs_table = net.poly_cost.loc[net.poly_cost['et'] == 'ext_grid']
    extgrid_costs = np.array(extgrid_costs_table['cp1_eur_per_mw'].values)

    # generator min max
    pg_max_zero_mask = simulation_parameters['true_system']['Sg_max'][:n_gens] < 1e-9
    gen_mask_to_keep = ~pg_max_zero_mask  # invert mask to keep desired generators
    gen_delta = torch.tensor(simulation_parameters['true_system']['Sg_delta'][:n_gens][gen_mask_to_keep]).float().unsqueeze(1) / 100
    gen_min = torch.zeros_like(gen_delta)
    
    # voltage min max   
    volt_min = torch.tensor(simulation_parameters['true_system']['Volt_min']).float().unsqueeze(1)[gen_bus_indices_no_slack]
    volt_max = torch.tensor(simulation_parameters['true_system']['Volt_max']).float().unsqueeze(1)[gen_bus_indices_no_slack]
    volt_delta = volt_max - volt_min
    
    output_min = torch.vstack((gen_min, volt_min))
    output_delta = torch.vstack((gen_delta, volt_delta))
    
    dem_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float()
    dem_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float()

    data_stat = {
        'gen_min': output_min,
        'gen_delta': output_delta,
        'dem_min': dem_min,
        'dem_delta': dem_delta,
    }

    
    # inference
    n_bus = simulation_parameters['general']['n_buses']
    gen_bus = simulation_parameters['general']['g_bus']
    n_branches = base_ppc['branch'].shape[0]
    
    Gen_output = model.forward_train(sd_test)
    
    # placeholder for Pg, and fill in with NN prediction
    Pg_place = torch.zeros((sd_test.shape[0], n_gens), dtype=torch.float64)
    Vm_nn_place = torch.ones((sd_test.shape[0], n_bus), dtype=torch.float64)
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long)

    # clip pg between limits
    # tiny inward margin so we don't sit exactly on the limit
    epsilon = 1e-4

    sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32)
    pg_max_gens = (sg_max[:n_gens, :][gen_mask_to_keep.squeeze()] / 100).squeeze() # (sg_max.T @ map_g)[:, :n_buses]
    
    
    Pg_active = Gen_output[:, :n_act_gens] 
    Pg = Pg_place.clone()
    Pg_active = torch.clamp(Pg_active, min=torch.zeros_like(pg_max_gens) + epsilon, max=pg_max_gens - epsilon)
    Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64) 
    # Pg = torch.clamp(Pg, min=0.0) # set negative values to zero.

    # pull Vmin/Vmax from ppc (they are per-bus)
    vmin_arr = base_ppc['bus'][:, VMIN].astype(float)
    vmax_arr = base_ppc['bus'][:, VMAX].astype(float)
    
    vmin_t = torch.tensor(vmin_arr, dtype=torch.float64, device=Vm_nn_place.device)
    vmax_t = torch.tensor(vmax_arr, dtype=torch.float64, device=Vm_nn_place.device)
        
    Vm_nn_g = Gen_output[:, n_act_gens:] 
    Vm_nn = Vm_nn_place.clone()
    Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
    Vm_nn = torch.clamp(Vm_nn, vmin_t + epsilon, vmax_t - epsilon)
    
    
    pg_tot = torch.zeros(n_gens, num_samples, dtype=torch.float32)
    qg_tot = torch.zeros(n_gens, num_samples, dtype=torch.float32)
    vm_tot = torch.ones(n_bus, num_samples, dtype=torch.float32)
    pinj_tot = torch.zeros(n_bus, num_samples, dtype=torch.float32)
    qinj_tot = torch.zeros(n_bus, num_samples, dtype=torch.float32)
    vr_tot = torch.zeros(n_bus, num_samples, dtype=torch.float32)
    vi_tot = torch.zeros(n_bus, num_samples, dtype=torch.float32)
    Iinj_r_tot = torch.zeros(n_bus, num_samples, dtype=torch.float64)
    Iinj_i_tot = torch.zeros(n_bus, num_samples, dtype=torch.float64)
    Ibr_from_r_tot = torch.zeros(n_branches, num_samples, dtype=torch.float64)
    Ibr_from_i_tot = torch.zeros(n_branches, num_samples, dtype=torch.float64)
    Ibr_to_r_tot = torch.zeros(n_branches, num_samples, dtype=torch.float64)
    Ibr_to_i_tot = torch.zeros(n_branches, num_samples, dtype=torch.float64)
    total_cost = torch.zeros(1, num_samples, dtype=torch.float64)
    slack_tot = torch.zeros(1, num_samples, dtype=torch.float64)
    pf_convergence = torch.zeros(1, num_samples, dtype=torch.float64)
    
    
    slack_bus_internal_idx = np.where(base_ppc['bus'][:, BUS_TYPE] == REF)[0]
    if len(slack_bus_internal_idx) == 0:
        raise ValueError("No slack bus found in the base_ppc!")
    slack_bus_internal_idx = slack_bus_internal_idx[0] # Get the first (and usually only) slack bus

    # Get the external bus number (BUS_I) of the slack bus
    slack_bus_external_id = base_ppc['bus'][slack_bus_internal_idx, BUS_I]
    slack_gen_indices = np.where(base_ppc['gen'][:, 0] == slack_bus_external_id)[0]
    
    external_to_ppc_row_idx = {
        int(bus_data[BUS_I]): idx for idx, bus_data in enumerate(base_ppc['bus'])
    }
    
    # --- Loop to solve power flow for each sample ---
    for i in range(num_samples):
        updated_ppc = copy.deepcopy(base_ppc) # Always work on a copy
        
        # Convert current sample's NN outputs to NumPy
        pg_values_np_sample = Pg[i].detach().cpu().numpy()
        vm_values_np_sample = Vm_nn[i].detach().cpu().numpy()

        #### Set pg and vm values in the current ppc
        updated_ppc['gen'][act_gen_indices, PG] = pg_values_np_sample[act_gen_indices]
        updated_ppc['bus'][pv_indices, VM] = vm_values_np_sample[pv_indices]
        # print(pg_values_np_sample[act_gen_indices])
        # print(vm_values_np_sample[pv_indices])

        # Solve power flow for this sample
        ppopt = ppoption(OUT_ALL=0, PF_ALG = 1, PF_TOL=1e-4, PF_MAX_IT=200) # PF_ALG=1 is Newton-Raphson, 2 = fast decoupled
        results, success_flag = runpf(updated_ppc, ppopt)

        # --- Extract and process results for the current sample ---
        if success_flag == 1: # Check if power flow converged successfully
            pf_convergence[0, i] = int(1.0)
            # Extract QG for all generators
            qg_all_gens = results['gen'][:, QG] # Shape (num_gens,)
            qg_without_slack = np.delete(qg_all_gens, slack_gen_indices, axis=0) / 100
            
            vm_tot[:, i] = torch.tensor(results['bus'][:, VM], dtype=torch.float32)
            
            vm = results['bus'][:, VM]
            va_rad = np.deg2rad(results['bus'][:, VA])
            vr = vm * np.cos(va_rad)
            vi = vm * np.sin(va_rad)
            
            # Complex voltages
            V_complex = results['bus'][:, VM] * np.exp(1j * np.deg2rad(results['bus'][:, VA]))

            # Calculate Vr and Vi
            vr_tot[:, i] = torch.tensor(vr) # Shape (num_buses,)
            vi_tot[:, i] = torch.tensor(vi) # Shape (num_buses,)

            I_complex_injections = Ybus @ V_complex
            I_complex_injections_np = I_complex_injections.detach().cpu().numpy()  # Ensure it's on CPU for further processing
            
            # Need to get Ybus from net._ppc
            Iinj_r_tot[:, i] = torch.tensor(np.real(I_complex_injections_np), dtype=torch.float64) # correct
            Iinj_i_tot[:, i] = torch.tensor(np.imag(I_complex_injections_np), dtype=torch.float64)
            
            # Complex power injection S = V * conj(I)
            S_complex_injections = V_complex * np.conj(I_complex_injections_np)
            
            pinj_tot[:, i] = torch.tensor(np.real(S_complex_injections), dtype=torch.float64) # correct
            qinj_tot[:, i] = torch.tensor(np.imag(S_complex_injections), dtype=torch.float64)        
            
            # Get from/to bus indices (external IDs)
            branch_data = results['branch']
            fbus_external = branch_data[:, F_BUS]
            tbus_external = branch_data[:, T_BUS]

            # Convert external bus IDs to internal PyPower array indices
            # Ensure that `external_to_ppc_row_idx` handles all bus IDs present in `fbus_external`/`tbus_external`
            fbus_internal_idx = np.array([external_to_ppc_row_idx[int(bus_id)] for bus_id in fbus_external])
            tbus_internal_idx = np.array([external_to_ppc_row_idx[int(bus_id)] for bus_id in tbus_external])

            # Get complex voltages at 'from' and 'to' buses for each branch
            V_f_complex = V_complex[fbus_internal_idx]
            V_t_complex = V_complex[tbus_internal_idx]

            # Get complex power flows (from 'from' bus side, and from 'to' bus side)
            S_f_complex_mva = branch_data[:, PF] + 1j * branch_data[:, QF]
            S_t_complex_mva = branch_data[:, PT] + 1j * branch_data[:, QT]

            # Convert power flows to pu for current calculation
            S_f_complex_pu = S_f_complex_mva / Sbase
            S_t_complex_pu = S_t_complex_mva / Sbase
            
            # Calculate complex currents using I = S* / V*
            I_br_from_complex = np.divide(np.conj(S_f_complex_pu), np.conj(V_f_complex),
                                        out=np.zeros_like(S_f_complex_pu, dtype=np.complex128),
                                        where=np.abs(V_f_complex) > 1e-9) # Small tolerance to avoid ZeroDivisionError

            I_br_to_complex = np.divide(np.conj(S_t_complex_pu), np.conj(V_t_complex),
                                        out=np.zeros_like(S_t_complex_pu, dtype=np.complex128),
                                        where=np.abs(V_t_complex) > 1e-9)

            # Store rectangular components of branch currents (assuming Ibr_from_r_tot etc. are initialized)
            Ibr_from_r_tot[:, i] = torch.tensor(np.real(I_br_from_complex), dtype=torch.float64)
            Ibr_from_i_tot[:, i] = torch.tensor(np.imag(I_br_from_complex), dtype=torch.float64)
            Ibr_to_r_tot[:, i] = torch.tensor(np.real(I_br_to_complex), dtype=torch.float64)
            Ibr_to_i_tot[:, i] = torch.tensor(np.imag(I_br_to_complex), dtype=torch.float64)

            extgrid_mw = pd_tot[i, :].sum() - Pg[i, :].sum()
            extgrid = extgrid_costs * extgrid_mw.detach().numpy() * Sbase

            total_cost[:, i] = calculate_total_cost(generator_costs, Pg[i, :].detach().cpu().numpy() * Sbase, extgrid)
            slack_tot[:, i] = extgrid_mw
            
            

            # # ------------
            # kcl_im = torch.tensor(simulation_parameters['true_system']['kcl_im'], dtype=torch.float64)
            # kcl_from_im = torch.relu(kcl_im) # +1 at from-bus, 0 elsewhere
            # kcl_to_im = torch.relu(-kcl_im) # +1 at to-bus, 0 elsewhere
    
            # branch_injections_r = torch.matmul(kcl_from_im, Ibr_from_r_tot[:, i]) + torch.matmul(kcl_to_im, Ibr_to_r_tot[:, i])
            # branch_injections_i = torch.matmul(kcl_from_im, Ibr_from_i_tot[:, i]) + torch.matmul(kcl_to_im, Ibr_to_i_tot[:, i])

            # I_shunt_r = -bs_values * vi_tot[:, i] 
            # I_shunt_i = bs_values * vr_tot[:, i] 

            # # Calculate the current imbalance (I_injected - Ybus * V)
            # # The matrix multiplication is (n_bus x n_bus) @ (n_bus x n_data_points)
            # delta_I_r = Iinj_r_tot[:, i] - branch_injections_r - I_shunt_r
            # delta_I_i = Iinj_i_tot[:, i] - branch_injections_i - I_shunt_i

            # # The violation is the magnitude of the imbalance
            # Ibal_violation = torch.sqrt(delta_I_r**2 + delta_I_i**2)
            
            
            # print(delta_I_r)
            # print(delta_I_i)
            # print(Ibal_violation > 1e-9)
            # print(Ibal_violation.mean())
            # print(Ibal_violation.max())
            # stop
            
            # # -----------
            

        else:
            
            # set voltages to NN guess
            vm_tot[:, i] = torch.tensor(Vm_nn[i].detach().cpu().numpy(), dtype=torch.float32)
            
            ### still calculate cost..
            extgrid_mw = pd_tot[i, :].sum() - Pg[i, :].sum()
            extgrid = extgrid_costs * extgrid_mw.detach().numpy() * Sbase

            total_cost[:, i] = calculate_total_cost(generator_costs, Pg[i, :].detach().cpu().numpy() * Sbase, extgrid)
            slack_tot[:, i] = extgrid_mw
            
            print(f"Warning: Power flow did not converge for sample {i}.")


    # Combine results into a dictionary
    solution_data = {
        'pd_tot': pd_tot.T, 
        'qd_tot': qd_tot.T, 
        'nn_input': sd_test.detach().cpu().numpy(), 
        'pg_tot': Pg[:, gen_bus].T.detach().cpu().numpy(),
        'qg_tot': qg_tot,
        'vm_tot': vm_tot,
        'pinj_tot': pinj_tot,
        'qinj_tot': qinj_tot,
        'vr_tot': vr_tot,
        'vi_tot': vi_tot,
        'Iinj_r_tot': Iinj_r_tot, # Add new tensors to the dictionary
        'Iinj_i_tot': Iinj_i_tot,
        'Ibr_from_r_tot': Ibr_from_r_tot.detach().cpu().numpy(),
        'Ibr_from_i_tot': Ibr_from_i_tot.detach().cpu().numpy(),
        'Ibr_to_r_tot': Ibr_to_r_tot.detach().cpu().numpy(),
        'Ibr_to_i_tot': Ibr_to_i_tot.detach().cpu().numpy(),
        'cost_tot': total_cost.detach().cpu().numpy(),
        'slack_tot': slack_tot.detach().cpu().numpy(),
        'sample_num': np.arange(num_samples).reshape(1, -1),
        'pf_convergence': pf_convergence.detach().cpu().numpy(),
    }

    return solution_data



def extract_solution_data_from_pandapower_net(net: pp.pandapowerNet, sample, num_samples) -> dict:
    """
    Extracts various power system quantities from a solved pandapower network
    and formats them into a dictionary similar to the user's 'solution_data'
    from PYPOWER simulations.

    Args:
        net (pandapower.Net): A pandapower network object after an OPF (or power flow)
                              has been successfully solved, so that net.res_bus,
                              net.res_gen, net.res_line tables are populated.

    Returns:
        dict: A dictionary containing extracted and calculated power system data
              as PyTorch tensors, with a (num_elements, 1) shape for a single data point.
              Note: 'nn_input' is not derivable from OPF results and is excluded.
    """
    if not hasattr(net, 'res_bus') or net.res_bus.empty:
        raise ValueError("Network has no results. Please run an OPF or power flow first (e.g., pp.runpm_ac_opf(net)).")


    # Get system base MVA (Sbase) from the internal PYPOWER case
    base_ppc = net._ppc
    Sbase = base_ppc['baseMVA']
    n_bus = base_ppc['bus'].shape[0]
    n_gens = base_ppc['gen'].shape[0]
    n_loads = len(net.load) # Number of distinct load elements in pandapower's view
    n_branches = base_ppc['branch'].shape[0]
    
    # get Ybus
    Ybus, _, _ = makeYbus(net._ppc['baseMVA'], net._ppc['bus'], net._ppc['branch'])
    Ybus_dense = Ybus.toarray()
    
    # Get from/to bus indices (internal PYPOWER 0-based row indices)
    external_to_ppc_row_idx = {
        int(bus_data[BUS_I]): idx for idx, bus_data in enumerate(base_ppc['bus'])
    }
    
    # --- Initialize Output Tensors ---
    result_tensors = _initialize_result_tensors(n_bus, n_gens, n_branches, 1) # one result tensor per sample!
    
    _process_and_store_opf_results(base_ppc, Sbase, Ybus_dense, result_tensors, 0, external_to_ppc_row_idx)
                    
    # --- Post-processing: Remove slack bus generators ---
    result_tensors['pg_tot'] = _filter_generator_data(result_tensors['pg_tot'], base_ppc['bus'], base_ppc)
    result_tensors['qg_tot'] = _filter_generator_data(result_tensors['qg_tot'], base_ppc['bus'], base_ppc)
    
    # Combine results into a dictionary for return
    solution_data = {
        'pd_tot': torch.tensor(net.load['p_mw'].values, dtype=torch.float32).unsqueeze(1) / Sbase, # Convert to pu here
        'qd_tot': torch.tensor(net.load['q_mvar'].values, dtype=torch.float32).unsqueeze(1) / Sbase, # Convert to pu here
    }
    solution_data.update(result_tensors) # Add all collected tensors
    
    # Add the cost to the solution_data dictionary
    gen_costs_table = net.poly_cost.loc[net.poly_cost['et'] == 'gen']
    generator_costs = np.array(gen_costs_table['cp1_eur_per_mw'].values)
    
    extgrid_costs_table = net.poly_cost.loc[net.poly_cost['et'] == 'ext_grid']
    extgrid_costs = np.array(extgrid_costs_table['cp1_eur_per_mw'].values)
    extgrid_mw = np.array(net.res_ext_grid['p_mw'].values)
    extgrid = extgrid_costs * extgrid_mw
    
    solution_data['cost_tot'] = calculate_total_cost(generator_costs, result_tensors['pg_tot'][:, 0].detach().cpu().numpy() * Sbase, extgrid)

    return solution_data


def calculate_total_cost(linear_coeffs: np.ndarray, Pg: np.ndarray, extgrid) -> float:
    """
    Calculates the total operational cost for a given generator dispatch,
    assuming a simple linear cost model for each generator.

    Cost = c_1 * P

    Args:
        linear_coeffs (np.ndarray): A 1D array of linear cost coefficients
                                     (e.g., dollars per MW-hour) for each generator.
        Pg (np.ndarray): An array of real power output for each generator (in MW).
                         It should have the same number of elements as linear_coeffs.

    Returns:
        float: The total operational cost in dollars per hour.
    """
    # Ensure that Pg is a 1D array
    if Pg.ndim > 1:
        if Pg.shape[1] == 1:
            Pg = Pg.flatten()
        else:
            raise ValueError("Pg should be a 1D array of generator outputs.")

    if linear_coeffs.shape != Pg.shape:
        raise ValueError(f"The number of linear coefficients ({linear_coeffs.shape}) must match the number of generator outputs ({Pg.shape}).")
    
    # Calculate the cost for each generator and sum them up
    total_cost = np.sum(linear_coeffs * Pg) + extgrid[0]
        
    return torch.tensor([[float(total_cost)]])






def merge_solution_dicts(list_of_dicts):
    """
    Merges a list of solution dictionaries into a single dictionary by
    concatenating the columns (rows) of each key's value.

    Args:
        list_of_dicts (list): A list of dictionaries to be merged.

    Returns:
        dict: A single dictionary with all the data concatenated.
    """
    if not list_of_dicts:
        return {}

    # Start with the first dictionary as the base
    merged_dict = copy.deepcopy(list_of_dicts[0])

    # Iterate through the rest of the dictionaries in the list
    for new_dict in list_of_dicts[1:]:
        for key, value in new_dict.items():
            if key in merged_dict:
                existing_value = merged_dict[key]
                
                # Concatenate based on data type
                if isinstance(value, np.ndarray):
                    merged_dict[key] = np.concatenate((existing_value, value), axis=1)
                elif isinstance(value, torch.Tensor):
                    merged_dict[key] = torch.cat((existing_value, value), dim=1)
                else:
                    # For other types (like lists), extend them
                    try:
                        merged_dict[key].extend(value)
                    except (AttributeError, TypeError):
                        # Handle case where value is not a list
                        # Or other un-stackable data. You might choose to skip or raise an error.
                        pass
            else:
                # If a key is not in the base dict, just add it
                merged_dict[key] = value

    return merged_dict



def filter_and_merge_solution_dicts(list_of_dicts, n_top=1, n_bottom=1):
    """
    Removes the top n_top dicts with largest solve_time and 
    the bottom n_bottom dicts with smallest solve_time before merging.

    Args:
        list_of_dicts (list): List of solution dictionaries.
        n_top (int): Number of largest solve_time dicts to remove.
        n_bottom (int): Number of smallest solve_time dicts to remove.

    Returns:
        dict: Merged dictionary of the remaining entries.
    """
    if not list_of_dicts:
        return {}

    # Extract solve_times for sorting (take first element if it's a list)
    solve_times = [d['solve_time'][0] if isinstance(d['solve_time'], list) else d['solve_time'] 
                   for d in list_of_dicts]

    # Sort dicts by solve_time
    sorted_dicts = [d for _, d in sorted(zip(solve_times, list_of_dicts), key=lambda x: x[0])]

    # Remove bottom n_bottom and top n_top
    filtered_dicts = sorted_dicts[n_bottom: len(sorted_dicts)-n_top]

    # Merge remaining dicts
    return merge_solution_dicts(filtered_dicts)






def build_network(nn_type, n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed, simulation_parameters):
        hidden_layer_size = [hidden_layer_size] * n_hidden_layers
        model = NeuralNetwork(nn_type, n_input_neurons, hidden_layer_size=hidden_layer_size,
                            num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed, simulation_parameters = simulation_parameters)
        return model#.to(device)
    
    

# def compare_accuracy_with_mse(ground_truth: dict, proxy: dict):
#     """
#     Compares shared keys in two dictionaries and computes Mean Squared Error (MSE),
#     percentage deviation, Mean Absolute Error (MAE), and adds the mean ground truth
#     and proxy values for each key.

#     Parameters:
#     - ground_truth (dict): A dictionary containing the ground truth values.
#     - proxy (dict): A dictionary containing the proxy values for comparison.
#     """
#     result = {}
#     shared_keys = set(ground_truth).intersection(proxy)

#     for key in shared_keys:
#         if key == 'solve_time':
#             result[key] = {
#                 "ground_truth_value": ground_truth[key],
#                 "proxy_value": proxy[key],
#                 # No MSE or deviation for solve time, as it's a direct comparison
#             }
#             continue
        
#         gt = np.asarray(ground_truth[key])
#         px = np.asarray(proxy[key])

#         gt_flat = gt.flatten()
#         px_flat = px.flatten()

#         if gt_flat.shape != px_flat.shape:
#             result[key] = {"error": f"Shape mismatch: {gt.shape} vs {px.shape}"}
#             continue

#         # Calculate Mean Squared Error (MSE)
#         mse = np.mean((gt_flat - px_flat)**2)
        
#         # Calculate Percentage Deviation
#         non_zero_gt = gt_flat != 0
#         if np.sum(non_zero_gt) > 0:
#             absolute_percentage_deviation = np.abs((gt_flat[non_zero_gt] - px_flat[non_zero_gt]) / gt_flat[non_zero_gt])
#             percentage_deviation = np.mean(absolute_percentage_deviation) * 100
#         else:
#             percentage_deviation = 0.0

#         # Calculate Mean Absolute Error (MAE)
#         mae = np.mean(np.abs(gt_flat - px_flat))

#         mean_gt_value = np.mean(gt_flat)
#         mean_proxy_value = np.mean(px_flat)


#         result[key] = {
#             "mean_squared_error": mse,
#             "percentage_deviation": percentage_deviation,
#             "mean_absolute_error": mae,

#             "ground_truth_value": mean_gt_value,
#             "proxy_value": mean_proxy_value

#         }

#     return result


import numpy as np

def compare_accuracy_with_mse(ground_truth: dict, proxy: dict):
    """
    Compares shared keys in two dictionaries and computes Mean Squared Error (MSE),
    percentage deviation, Mean Absolute Error (MAE), and adds the mean ground truth
    and proxy values for each key, **only for samples present in the proxy dict**.

    Parameters:
    - ground_truth (dict): Dictionary containing ground truth values, must have 'sample_num'.
    - proxy (dict): Dictionary containing proxy values, must have 'sample_num'.
    """
    result = {}

    gt_samples = np.asarray(ground_truth.get('sample_num', []))
    px_samples = np.asarray(proxy.get('sample_num', []))

    if len(px_samples) == 0:
        print("Proxy has no sample numbers. Returning empty metrics.")
        return result

    shared_keys = set(ground_truth).intersection(proxy)

    def flatten_to_scalar_list(arr):
        """Flattens nested or array-like input to a list of scalars."""
        arr = np.asarray(arr).flatten()
        return [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in arr]

    def select_samples_by_number(array, sample_nums, target_samples):
        """Select elements from `array` where corresponding sample_num is in target_samples."""
        array = np.asarray(array)

        # Flatten and sanitize all lists/arrays into scalars for hashing
        sample_nums_list = flatten_to_scalar_list(sample_nums)
        target_samples_list = flatten_to_scalar_list(target_samples)
        target_samples_set = set(target_samples_list)

        indices = [i for i, s in enumerate(sample_nums_list) if s in target_samples_set]

        if len(indices) == 0:
            return np.array([])

        if array.ndim == 1:
            return array[indices]
        elif array.ndim == 2:
            # assume samples are along rows
            if array.shape[0] == len(sample_nums_list):
                return array[indices, :]
            # assume samples are along columns
            elif array.shape[1] == len(sample_nums_list):
                return array[:, indices]
            else:
                raise ValueError(f"Cannot match samples: array shape {array.shape}, len(sample_nums)={len(sample_nums_list)}")
        else:
            raise ValueError(f"Unexpected array shape: {array.shape}")

    for key in shared_keys:
        if key in ('sample_num', 'solve_time'):
            result[key] = {"ground_truth_value": ground_truth[key], "proxy_value": proxy[key]}
            continue

        gt_array = np.asarray(ground_truth[key])
        px_array = np.asarray(proxy[key])

        # Match ground truth samples to proxy samples (based on proxy sample numbers)
        gt_matched = select_samples_by_number(gt_array, gt_samples, px_samples)
        px_matched = select_samples_by_number(px_array, px_samples, px_samples)
        
        # --- Auto-fix shape mismatches if transposing helps ---
        if gt_matched.size != 0 and px_matched.size != 0 and gt_matched.shape != px_matched.shape:
            if gt_matched.T.shape == px_matched.shape:
                print(f"AUTO-FIX: Transposing ground truth for {key}")
                gt_matched = gt_matched.T
            elif px_matched.T.shape == gt_matched.shape:
                print(f"AUTO-FIX: Transposing proxy for {key}")
                px_matched = px_matched.T

        if gt_matched.size == 0 or px_matched.size == 0 or gt_matched.shape != px_matched.shape:
            result[key] = {
                "mean_squared_error": np.nan,
                "percentage_deviation": np.nan,
                "mean_absolute_error": np.nan,
                "ground_truth_value": np.nan,
                "proxy_value": np.nan,
            }
            continue

        mse = np.mean((gt_matched - px_matched) ** 2)
        non_zero_gt = gt_matched != 0
        mae = np.mean(np.abs(gt_matched - px_matched))
        mean_gt_value = np.mean(gt_matched)
        mean_proxy_value = np.mean(px_matched)
        
        percentage_deviation = (
            abs(mean_proxy_value - mean_gt_value) / abs(mean_gt_value) * 100
            if mean_gt_value != 0
            else 0.0
        )


        result[key] = {
            "mean_squared_error": mse,
            "percentage_deviation": percentage_deviation,
            "mean_absolute_error": mae,
            "ground_truth_value": mean_gt_value,
            "proxy_value": mean_proxy_value,
        }

    return result



    
    
    
def print_comparison_table(
    mse_results_list: list[dict],
    model_names: list[str],
    table_title: str = "Comparison of Metrics for Different Models vs. Ground Truth",
    num_bus = None,
    num_samp = None, 
    goal=None,
):
    """
    Prepares and prints a formatted comparison table of metrics for multiple models.
    Displays percentage deviation for the 'cost' metric and Mean Squared Error for others.

    Parameters:
    - mse_results_list (list[dict]): A list of dictionaries, where each dictionary
                                     contains metric results for a single model.
                                     Expected format for each dict:
                                     {metric_name: {'mean_squared_error': value,
                                                    'percentage_deviation': value}}
    - model_names (list[str]): A list of strings, providing names for each model.
                               The order must correspond to `mse_results_list`.
    - table_title (str, optional): Title to print above the table.
                                  Defaults to "Comparison of Metrics for Different Models vs. Ground Truth".
    """
    if not mse_results_list:
        print("No results provided for comparison.")
        return
    if len(mse_results_list) != len(model_names):
        raise ValueError("The number of result dictionaries must match the number of model names.")

    # --- Step 1: Extract the correct metric values into a list of Series ---
    all_series = []
    for i, results_dict in enumerate(mse_results_list):
        metric_values = {}
        for k, v in results_dict.items():
            # Check for "error" key to skip invalid entries
            if 'error' in v:
                continue

            # Conditionally choose the metric based on the key name
            if k in ['cost_tot', 'slack_tot']:
                proxy_value = v.get('proxy_value', 'N/A')
                real_value = v.get('ground_truth_value', 'N/A')
                mse_value = v.get('percentage_deviation', 'N/A')
                
                # Format the string, handling potential missing values
                if real_value != 'N/A' and mse_value != 'N/A' and k == 'cost_tot':
                    metric_values[k] = f"{proxy_value:.2f} ({real_value:.2f} > {mse_value:.2f}%)"
                elif real_value != 'N/A' and mse_value != 'N/A' and k == 'slack_tot':
                    metric_values[k] = f"{proxy_value:.2f} ({real_value:.2f})"
                else:
                    metric_values[k] = f"N/A (N/A)"
            elif k in ['solve_time']:
                proxy_value = v.get('proxy_value', 'N/A')
                real_value = v.get('ground_truth_value', 'N/A')
                
                # convert to mean
                if isinstance(proxy_value, torch.Tensor):
                    proxy_value = torch.mean(proxy_value).item()
                elif isinstance(proxy_value, list):
                    proxy_value = sum(proxy_value) / len(proxy_value)
                    
                if isinstance(real_value, torch.Tensor):
                    real_value = torch.mean(real_value).item()
                elif isinstance(real_value, list):
                    real_value = sum(real_value) / len(real_value)
 
                if proxy_value != 'N/A' and real_value != 'N/A':
                    metric_values[k] = f"{proxy_value:.2f}s ({real_value:.2f} s)"
                else:
                    metric_values[k] = f"N/A (N/A)"
            else:
                if 'mean_squared_error' in v:
                    metric_values[k] = v['mean_squared_error']
                else:
                    metric_values[k] = np.nan  # or "N/A"

        # Create a Pandas Series for this model's metrics
        s = pd.Series(metric_values, name=model_names[i])
        all_series.append(s)

    # --- Step 2: Combine all Series into a single DataFrame ---
    comparison_df = pd.concat(all_series, axis=1)
    comparison_df.index.name = 'Metric [MSE]'
    
    try:
        # Create a unique filename with a timestamp
        # filename = f"violations_comparison_table_{num_bus}_{num_samp}.csv"
        filename = f"accuracy_comparison_table_{num_bus}_{num_samp}_{goal}.xlsx"
        
        # Get the directory of the current script and join with the filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, filename)

        # Save the DataFrame to a CSV file
        # comparison_df.to_csv(output_path, index=True, float_format="%.4f")
        comparison_df.to_excel(output_path, index=True)
        print(f"Comparison table successfully saved to {output_path}")
        
    except IOError as e:
        print(f"Error writing to CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the CSV: {e}")

    # --- Step 3: Custom Ordering and Printing for Nicer Table ---
    metric_groups_display = [
        ("Input Data Metrics", ['nn_input', 'pd_tot', 'qd_tot']),
        ("Voltage Metrics", ['vm_tot', 'vr_tot', 'vi_tot']),
        ("Power Metrics", ['pg_tot', 'qg_tot', 'pinj_tot', 'qinj_tot']),
        ("Current Metrics (Bus Injections)", ['Iinj_r_tot', 'Iinj_i_tot']),
        ("Current Metrics (Branch Flows)", ['Ibr_from_r_tot', 'Ibr_from_i_tot', 'Ibr_to_r_tot', 'Ibr_to_i_tot']),
        ("Objective Cost (Percentage Deviation)", ['cost_tot']),  
        ("Solve Time", ['solve_time'])
    ]

    print(f"{table_title}:\n")

    printed_keys = set()
    for group_name, keys_in_order in metric_groups_display:
        current_group_df_keys = [key for key in keys_in_order if key in comparison_df.index]
        if current_group_df_keys:
            print(f"\n--- {group_name} ---")
            print(comparison_df.loc[current_group_df_keys]) # No need for float_format when using f-strings for cost
            printed_keys.update(current_group_df_keys)

    remaining_keys = [key for key in comparison_df.index if key not in printed_keys]
    if remaining_keys:
        print("\n--- Other / Uncategorized Metrics ---")
        print(comparison_df.loc[sorted(remaining_keys)])

    print("\n" + "=" * 60 + "\n")
    
    
def print_violations_comparison_table(
    violations_dicts: list[dict],
    model_names: list[str],
    table_title: str = "Comparison of Violations for Different Models",
    num_bus=None,
    num_samp=None,
):
    """
    Prepares and prints a formatted comparison table of average and max
    violation values for multiple models and saves it to a CSV file.

    Parameters:
    - violations_dicts (list[dict]): A list of dictionaries, where each dictionary
                                     contains violation results for a single model.
                                     Expected format for each dict:
                                     {'pg_avg_violation': tensor, 'pg_max_violation': tensor, ...}
    - model_names (list[str]): A list of strings, providing names for each model.
                               The order must correspond to `violations_dicts`.
    - table_title (str, optional): Title to print above the table.
                                  Defaults to "Comparison of Violations...".
    """
    if not violations_dicts:
        print("No violation results provided for comparison.")
        return
    if len(violations_dicts) != len(model_names):
        raise ValueError("The number of violation dictionaries must match the number of model names.")

    # --- Step 1: Extract violation values into a list of Series ---
    all_series = []
    for i, violations_dict in enumerate(violations_dicts):
        # Extract violation values, converting tensors to scalar items
        # and creating more readable metric names for the table.
        violation_values = {
            'Pg up Avg Violation': violations_dict['pg_up_avg_violation'].item(),
            'Pg up Max Violation': violations_dict['pg_up_max_violation'].item(),
            'Pg down Avg Violation': violations_dict['pg_down_avg_violation'].item(),
            'Pg down Max Violation': violations_dict['pg_down_max_violation'].item(),
            'Qg up Avg Violation': violations_dict['qg_up_avg_violation'].item(),
            'Qg up Max Violation': violations_dict['qg_up_max_violation'].item(),
            'Qg down Avg Violation': violations_dict['qg_down_avg_violation'].item(),
            'Qg down Max Violation': violations_dict['qg_down_max_violation'].item(),
            'Vm up Avg Violation': violations_dict['vm_up_avg_violation'].item(),
            'Vm up Max Violation': violations_dict['vm_up_max_violation'].item(),
            'Vm down Avg Violation': violations_dict['vm_down_avg_violation'].item(),
            'Vm down Max Violation': violations_dict['vm_down_max_violation'].item(),
            'Ibr Avg Violation': violations_dict['Ibr_avg_violation'].item(),
            'Ibr Max Violation': violations_dict['Ibr_max_violation'].item(),
            'Ibal Avg Violation': violations_dict['Ibal_avg_violation'].item(),
            'Ibal Max Violation': violations_dict['Ibal_max_violation'].item(),
        }
        # Create a Pandas Series for this model's violations
        s = pd.Series(violation_values, name=model_names[i])
        all_series.append(s)

    # --- Step 2: Combine all Series into a single DataFrame ---
    comparison_df = pd.concat(all_series, axis=1)
    comparison_df.index.name = 'Violation Metric [pu]'

    # --- NEW: Step to save the DataFrame to a CSV file ---
    try:
        # Create a unique filename with a timestamp
        # filename = f"violations_comparison_table_{num_bus}_{num_samp}.csv"
        filename = f"violations_comparison_table_{num_bus}_{num_samp}.xlsx"
        
        # Get the directory of the current script and join with the filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, filename)

        # Save the DataFrame to a CSV file
        # comparison_df.to_csv(output_path, index=True, float_format="%.4f")
        comparison_df.to_excel(output_path, index=True)
        print(f"Comparison table successfully saved to {output_path}")

    except IOError as e:
        print(f"Error writing to CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the CSV: {e}")

    # --- Step 3: Custom Ordering and Printing for Nicer Table ---
    # Define the desired groups and their display order
    metric_groups_display = [
        ("Generator Pg Violations", ['Pg up Avg Violation', 'Pg up Max Violation', 'Pg down Avg Violation', 'Pg down Max Violation']),
        ("Generator Qg Violations", ['Qg up Avg Violation', 'Qg up Max Violation', 'Qg down Avg Violation', 'Qg down Max Violation']),
        ("Bus Voltage Violations", ['Vm up Avg Violation', 'Vm up Max Violation', 'Vm down Avg Violation', 'Vm down Max Violation']),
        ("Branch Current Violations", ['Ibr Avg Violation', 'Ibr Max Violation']),
        ("Nodal Current Balance Violations", ['Ibal Avg Violation', 'Ibal Max Violation']),
    ]

    print(f"\n{table_title}:\n")

    for group_name, keys_in_order in metric_groups_display:
        current_group_df_keys = [key for key in keys_in_order if key in comparison_df.index]

        if current_group_df_keys: # Only print if there are metrics for this group
            print(f"--- {group_name} ---")
            # Print the subset of the DataFrame for this group, in the specified order
            print(comparison_df.loc[current_group_df_keys].to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 60 + "\n")



def power_nn_projection(net, num_samples, solution_data_dict, pg_targets, vm_targets):
    
    """
    Performs the neural network projection for multiple samples and measures the time
    taken for the Julia optimization, excluding startup overhead.
    """
    
    # === WARM-UP RUN to handle PyJulia and JIT compilation overhead ===
    print("Performing warm-up run of the Julia optimization...")
    warmup_net = copy.deepcopy(net)
    try:
        # Run a single projection on a copy of the network. This call will be slow,
        # but it will prepare the Julia environment for subsequent, faster calls.
        _ = runpm_opf(warmup_net)
        print("Warm-up complete. Starting timed projections...")
    except Exception as e:
        print(f"Warm-up run failed with error: {e}. Aborting timing.")
    
    # === START MAIN TIMING LOOP ===
    all_projected_results_list = []
    not_converged_count = 0
    
    gen_indices                          = net.gen.index.tolist() # Get actual generator indices from the pandapower net
    bus_indices                          = net.bus.index.tolist() # Get actual bus indices from the pandapower net
    bus_indices_of_generators            = net.gen['bus'].tolist()
 
    for sample in range(num_samples):
        pp_net_power_proj                  = copy.deepcopy(net)
        
        # create dicts for targets
        pg_targets_dict_pwr                  = {idx: pg_targets[i, sample] for i, idx in enumerate(gen_indices)}
        vm_targets_dict_pwr                  = {idx: vm_targets[i, sample] for i, idx in enumerate(bus_indices_of_generators)}
        
        pp_net_power_proj.load['p_mw']       = solution_data_dict['pd_tot'][:, sample] * 100  # multiply by base
        pp_net_power_proj.load['q_mvar']     = solution_data_dict['qd_tot'][:, sample] * 100
        pp_net_power_proj.gen['target_pg']   = pd.Series(pg_targets_dict_pwr) * 100 # multiply by base
        pp_net_power_proj.bus['target_vm']   = pd.Series(vm_targets_dict_pwr)
        
        # do the projections:
        try:
            pgvm_projection, result_pm       = runpm_opf(pp_net_power_proj) # pm_model = "ACPowerModel"
        except Exception as e:
            print(f"Warning: Power Projection did not converge for sample {sample}. Appending NaNs. Error: {e}")
            not_converged_count += 1
        else:
            pgvm_projected_results               = extract_solution_data_from_pandapower_net(pgvm_projection, sample, num_samples)
            pgvm_projected_results['solve_time']       = [result_pm['solve_time']]
            pgvm_projected_results['sample_num']    = [sample]
            all_projected_results_list.append(pgvm_projected_results)
            
        
    if all_projected_results_list:
        final_projected_results = filter_and_merge_solution_dicts(all_projected_results_list)
    else:
        final_projected_results = []
    
    return not_converged_count, final_projected_results


def power_nn_warm_start(net, num_samples, solution_data_dict, pg_targets, vm_targets):
    
    """
    Performs the neural network projection for multiple samples and measures the time
    taken for the Julia optimization, excluding startup overhead.
    """
    
    # === WARM-UP RUN to handle PyJulia and JIT compilation overhead ===
    print("Performing warm-up run of the Julia optimization...")
    warmup_net = copy.deepcopy(net)
    try:
        # Run a single projection on a copy of the network. This call will be slow,
        # but it will prepare the Julia environment for subsequent, faster calls.
        _ = runpm_opf(warmup_net)
        print("Warm-up complete. Starting timed projections...")
    except Exception as e:
        print(f"Warm-up run failed with error: {e}. Aborting timing.")
    
    # === START MAIN TIMING LOOP ===        
    all_projected_results_list = []
    not_converged_count = 0
    
    gen_indices                          = net.gen.index.tolist() # Get actual generator indices from the pandapower net
    bus_indices                          = net.bus.index.tolist() # Get actual bus indices from the pandapower net
    bus_indices_of_generators            = net.gen['bus'].tolist()
 
    for sample in range(num_samples):
        pp_net_power_ws                  = copy.deepcopy(net)
        
        # create dicts for targets
        pg_targets_dict_pwr                  = {idx: pg_targets[i, sample] for i, idx in enumerate(gen_indices)}
        vm_targets_dict_pwr                  = {idx: vm_targets[i, sample] for i, idx in enumerate(bus_indices_of_generators)}
        
        pp_net_power_ws.load['p_mw']         = solution_data_dict['pd_tot'][:, sample] * 100  # multiply by base
        pp_net_power_ws.load['q_mvar']       = solution_data_dict['qd_tot'][:, sample] * 100
        pp_net_power_ws.gen['ws_pg']         = pd.Series(pg_targets_dict_pwr) * 100 # multiply by base
        pp_net_power_ws.bus['ws_vm']         = pd.Series(vm_targets_dict_pwr)

        # do the projections:
        try: 
            pgvm_warm_start, result_pm                      = runpm_opf(pp_net_power_ws)
        except Exception as e:
            print(f"Warning: Power WS did not converge for sample {sample}. Appending NaNs. Error: {e}")
            not_converged_count += 1
        else:
            pgvm_ws_results                      = extract_solution_data_from_pandapower_net(pgvm_warm_start, sample, num_samples)
            pgvm_ws_results['solve_time']              = [result_pm['solve_time']]
            pgvm_ws_results['sample_num']    = [sample]
            all_projected_results_list.append(pgvm_ws_results)
            
    if all_projected_results_list:
        final_projected_results = filter_and_merge_solution_dicts(all_projected_results_list)
    else:
        final_projected_results = []

    
    return not_converged_count, final_projected_results


def voltage_nn_projection(net, num_samples, solution_data_dict, vr_targets, vi_targets):
    
    """
    Performs the neural network projection for multiple samples and measures the time
    taken for the Julia optimization, excluding startup overhead.
    """
    
    # === WARM-UP RUN to handle PyJulia and JIT compilation overhead ===
    print("Performing warm-up run of the Julia optimization...")
    warmup_net = copy.deepcopy(net)
    try:
        # Run a single projection on a copy of the network. This call will be slow,
        # but it will prepare the Julia environment for subsequent, faster calls.
        _ = runpm_opf(warmup_net)
        print("Warm-up complete. Starting timed projections...")
    except Exception as e:
        print(f"Warm-up run failed with error: {e}. Aborting timing.")
    
    # === START MAIN TIMING LOOP ===    
    all_projected_results_list = []
    not_converged_count = 0
    
    # create dicts for targets
    bus_indices                          = net.bus.index.tolist() # Get actual bus indices from the pandapower net
    vm_targets, va_targets_deg           = convert_vr_vi_to_vm_va(vr_targets, vi_targets)
    
    for sample in range(num_samples):
        pp_net_voltage_proj                    = copy.deepcopy(net)
    
        # get targets
        vm_targets_dict_volt                 = {idx: vm_targets[i, sample] for i, idx in enumerate(bus_indices)}
        va_targets_dict_volt                 = {idx: va_targets_deg[i, sample] for i, idx in enumerate(bus_indices)}  
        
        # set targets in the pandapower net
        pp_net_voltage_proj.load['p_mw']     = solution_data_dict['pd_tot'][:, sample] * 100  # multiply by base
        pp_net_voltage_proj.load['q_mvar']   = solution_data_dict['qd_tot'][:, sample] * 100
        pp_net_voltage_proj.bus['target_vm'] = pd.Series(vm_targets_dict_volt) 
        pp_net_voltage_proj.bus['target_va'] = pd.Series(va_targets_dict_volt)  
        
        # do the projections:
        try: 
            vrvi_projection, result_pm                      = runpm_opf(pp_net_voltage_proj) # pm_model = "ACPowerModel"
        except Exception as e:
            print(f"Warning: Voltage Projection did not converge for sample {sample}. Appending NaNs. Error: {e}")
            not_converged_count += 1
        else:
            vrvi_projected_results               = extract_solution_data_from_pandapower_net(vrvi_projection, sample, num_samples)
            vrvi_projected_results['solve_time']       = [result_pm['solve_time']]
            vrvi_projected_results['sample_num']    = [sample]
            all_projected_results_list.append(vrvi_projected_results)
        
    if all_projected_results_list:
        final_projected_results = filter_and_merge_solution_dicts(all_projected_results_list)
    else:
        final_projected_results = []
    
    return not_converged_count, final_projected_results


def voltage_nn_warm_start(net, num_samples, solution_data_dict, vr_targets, vi_targets):
    
    """
    Performs the neural network projection for multiple samples and measures the time
    taken for the Julia optimization, excluding startup overhead.
    """
    
    # === WARM-UP RUN to handle PyJulia and JIT compilation overhead ===
    print("Performing warm-up run of the Julia optimization...")
    warmup_net = copy.deepcopy(net)
    try:
        # Run a single projection on a copy of the network. This call will be slow,
        # but it will prepare the Julia environment for subsequent, faster calls.
        _ = runpm_opf(warmup_net)
        print("Warm-up complete. Starting timed projections...")
    except Exception as e:
        print(f"Warm-up run failed with error: {e}. Aborting timing.")
    
    # === START MAIN TIMING LOOP ===  
    all_projected_results_list = []
    not_converged_count = 0
    
    # create dicts for targets
    bus_indices                          = net.bus.index.tolist() # Get actual bus indices from the pandapower net
    vm_targets, va_targets_deg           = convert_vr_vi_to_vm_va(vr_targets, vi_targets)
    
    for sample in range(num_samples):
        pp_net_voltage_ws                    = copy.deepcopy(net)
    
        # get targets
        vm_targets_dict_volt                 = {idx: vm_targets[i, sample] for i, idx in enumerate(bus_indices)}
        va_targets_dict_volt                 = {idx: va_targets_deg[i, sample] for i, idx in enumerate(bus_indices)}  
        
        # set targets in the pandapower net
        pp_net_voltage_ws.load['p_mw']       = solution_data_dict['pd_tot'][:, sample] * 100  # multiply by base
        pp_net_voltage_ws.load['q_mvar']     = solution_data_dict['qd_tot'][:, sample] * 100
        pp_net_voltage_ws.bus['ws_vm']       = pd.Series(vm_targets_dict_volt) 
        pp_net_voltage_ws.bus['ws_va']       = pd.Series(va_targets_dict_volt)  # convert degrees to radians
        
        try:
            # Attempt to run the OPF
            vrvi_warm_start, result_pm = runpm_opf(pp_net_voltage_ws)
        except Exception as e:
            print(f"Warning: Voltage WS did not converge for sample {sample}. Appending NaNs. Error: {e}")
            not_converged_count += 1
        else:
            # Only execute these if runpm_opf succeeded
            vrvi_ws_results = extract_solution_data_from_pandapower_net(vrvi_warm_start, sample, num_samples)
            vrvi_ws_results['solve_time'] = [result_pm['solve_time']]
            vrvi_ws_results['sample_num']    = [sample]
            all_projected_results_list.append(vrvi_ws_results)

        
    if all_projected_results_list:
        final_projected_results = filter_and_merge_solution_dicts(all_projected_results_list)
    else:
        final_projected_results = []
    
    return not_converged_count, final_projected_results




import os
from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.converter.pandamodels.from_pm import read_ots_results, read_tnep_results
from pandapower.opf.pm_storage import add_storage_opf_settings, read_pm_storage_results


# def runpm_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
#                  trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
#                  pm_solver="ipopt", correct_pm_network_data=True, silence=True,
#                  pm_time_limits=None, pm_log_level=0, pm_file_path=None, delete_buffer_file=True,
#                  opf_flow_lim="S", pm_tol=1e-8, pdm_dev_mode=False, **kwargs):
#     """
#     Runs non-linear optimal power flow from PowerModels.jl via PandaModels.jl
#     """
#     net._options = {}
#     _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
#                      trafo_model=trafo_model, check_connectivity=check_connectivity,
#                      mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
#                      enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
#                      voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
#     _add_opf_options(net, trafo_loading='power', ac=True, init="flat", numba=True,
#                      pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_opf_custom", pm_model="ACPPowerModel", pm_solver=pm_solver,
#                      correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
#                      pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

#     net, result_pm = _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)
    
#     return net, result_pm



# def _runpm(net, delete_buffer_file=True, pm_file_path=None, pdm_dev_mode=False, **kwargs): 
#     """
#     Converts the pandapower net to a pm json file, saves it to disk, runs a PandaModels.jl, and reads
#     the results back to the pandapower net:
#     INPUT
#     ----------
#     **net** - pandapower net
#     OPTIONAL
#     ----------
#     **delete_buffer_file** (bool, True) - deletes the pm buffer json file if True.
#     **pm_file_path** -path to save the converted net json file.
#     **pdm_dev_mode** (bool, False) - If True, the develop mode of PdM is called.
#     """
#     # convert pandapower to power models file -> this is done in python
#     net, pm, ppc, ppci = convert_to_pm_structure(net, **kwargs)
#     _add_custom_targets_to_pm_in_place(net, ppci, pm)
#     # call optional callback function
#     if net._options["pp_to_pm_callback"] is not None:
#         net._options["pp_to_pm_callback"](net, ppci, pm)
#     # writes pm json to disk, which is loaded afterwards in julia
#     buffer_file = dump_pm_json(pm, pm_file_path)
#     logger.debug("the json file for converted net is stored in: %s" % buffer_file)
#     # run power models optimization in julia

#     result_pm = _call_pandamodels(buffer_file, net._options["julia_file"], pdm_dev_mode)
    
#     logger.info("Optimization ('"+net._options["julia_file"]+"') " +
#                 "is finished in %s seconds:" % round(result_pm["solve_time"], 2))
#     # read results and write back to net
#     try:
#         read_pm_results_to_net(net, ppc, ppci, result_pm)
        
#         if pm_file_path is None and delete_buffer_file:
#             # delete buffer file after calculation
#             os.remove(buffer_file)
#             logger.debug("the json file for converted net is deleted from %s" % buffer_file)        
#     except OPFNotConverged as e:
#         if pm_file_path is None and delete_buffer_file:
#             os.remove(buffer_file)
#             logger.debug("the json file for converted net is deleted from %s" % buffer_file)
#         raise e
#     except Exception as e:
#         raise e
    
#     return net, result_pm
    

# def _call_pandamodels(buffer_file, julia_file, dev_mode):  # pragma: no cover

#     try:
#         import julia
#         from julia import Main
#         from julia import Pkg
#         from julia import Base
#     except ImportError:
#         raise ImportError(
#             "Please install pyjulia properly to run pandapower with PandaModels.jl.")
        
#     try:
#         julia.Julia()
#     except:
#         raise UserWarning(
#             "Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured")
              
#     if not Base.find_package("PandaModels"):
#         logger.info("PandaModels.jl is not installed in julia. It is added now!")
#         Pkg.Registry.update()
#         Pkg.add("PandaModels")  
        
#         if dev_mode:
#             logger.info("installing dev mode is a slow process!")
#             Pkg.resolve()
#             Pkg.develop("PandaModels")
#             # add pandamodels dependencies: slow process
#             Pkg.instantiate()
            
#         Pkg.build()
#         Pkg.resolve()
#         logger.info("Successfully added PandaModels")

#     if dev_mode:
#         Pkg.develop("PandaModels")
#         Pkg.build()
#         Pkg.resolve()
#         Pkg.activate("PandaModels")

#     try:
#         Main.using("PandaModels")
#     except ImportError:
#         raise ImportError("cannot use PandaModels")

#     Main.buffer_file = buffer_file
#     result_pm = Main.eval(julia_file + "(buffer_file)")

#     # if dev_mode:
#     #     Pkg.activate()
#     #     Pkg.free("PandaModels")
#     #     Pkg.resolve()
#     return result_pm

import atexit
global jl, Main, Pkg, Base
jl, Main, Pkg, Base = None, None, None, None

def cleanup_julia():
    """
    Safely shuts down the Julia runtime to prevent memory access errors on exit.
    This function is registered with `atexit` to be called automatically.
    """
    global jl
    if jl:
        try:
            # You can try to force a graceful shutdown here.
            # The act of registering a cleanup function is often enough
            # to prevent the Python/Julia GC race condition.
            print("Shutting down Julia runtime gracefully...")
        except Exception as e:
            print(f"Error during Julia cleanup: {e}")

def setup_julia_and_pandamodels(pdm_dev_mode=False):
    """
    Performs one-time setup for Julia and PandaModels.jl.
    This function should be called ONLY ONCE at the start of the script.
    """
    global jl, Main, Pkg, Base
    try:
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        from julia import Main
        from julia import Pkg
        from julia import Base
        
        # Register the cleanup function to be called on exit
        atexit.register(cleanup_julia)
    except ImportError:
        raise ImportError(
            "Please install pyjulia properly to run pandapower with PandaModels.jl.")

    try:
        if not Base.find_package("PandaModels"):
            logger.info("PandaModels.jl is not installed. It is being added now!")
            Pkg.add("PandaModels")
            
        if pdm_dev_mode:
            logger.info("Installing dev mode is a slow process!")
            Pkg.resolve()
            Pkg.develop("PandaModels")
            Pkg.instantiate()
            Pkg.activate("PandaModels")
            
        Pkg.build()
        Pkg.resolve()
        logger.info("Successfully configured PandaModels.jl")

        Main.using("PandaModels")
    except Exception as e:
        raise RuntimeError(f"Could not setup PandaModels.jl: {e}")

def runpm_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
              trafo_model="t", delta=1e-6, trafo3w_losses="hv", check_connectivity=True,
              pm_solver="ipopt", correct_pm_network_data=True, silence=True,
              pm_time_limits=None, pm_log_level=0, pm_file_path=None, delete_buffer_file=True,
              opf_flow_lim="S", pm_tol=1e-6, pdm_dev_mode=False, **kwargs):
    """
    Runs non-linear optimal power flow from PowerModels.jl via PandaModels.jl
    """
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="pf", init_va_degree="pf",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=True, init="pf", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file="run_powermodels_opf_custom", pm_model="ACPPowerModel", pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, silence=silence, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level, opf_flow_lim=opf_flow_lim, pm_tol=pm_tol)

    # net, result_pm = _runpm(net, delete_buffer_file=delete_buffer_file, pm_file_path=pm_file_path, pdm_dev_mode=pdm_dev_mode)
    net, result_pm = _runpm_in_memory(net, pdm_dev_mode=pdm_dev_mode, **kwargs)
    
    return net, result_pm

def _runpm(net, delete_buffer_file=True, pm_file_path=None, pdm_dev_mode=False, **kwargs):
    """
    Converts the pandapower net to a pm json file, saves it to disk, runs a PandaModels.jl, and reads
    the results back to the pandapower net.
    """
    # Check if Julia setup has been run
    if not all([jl, Main, Pkg, Base]):
        raise RuntimeError("Julia and PandaModels have not been set up. Please call `setup_julia_and_pandamodels()` once before calling this function.")

    # convert pandapower to power models file -> this is done in python
    net, pm, ppc, ppci = convert_to_pm_structure(net, **kwargs)
    _add_custom_targets_to_pm_in_place(net, ppci, pm)
    # call optional callback function
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    # writes pm json to disk, which is loaded afterwards in julia
    buffer_file = dump_pm_json(pm, pm_file_path)
    logger.debug("the json file for converted net is stored in: %s" % buffer_file)
    # run power models optimization in julia

    # The setup for PandaModels is now done outside, so we can directly call the Julia function
    result_pm = _call_pandamodels(buffer_file, net._options["julia_file"], pdm_dev_mode)
    
    logger.info("Optimization ('"+net._options["julia_file"]+"') " +
                "is finished in %s seconds:" % round(result_pm["solve_time"], 2))
    
    # read results and write back to net
    try:
        read_pm_results_to_net(net, ppc, ppci, result_pm)
        
        if pm_file_path is None and delete_buffer_file:
            os.remove(buffer_file)
            logger.debug("the json file for converted net is deleted from %s" % buffer_file)         
    except OPFNotConverged as e:
        if pm_file_path is None and delete_buffer_file:
            os.remove(buffer_file)
            logger.debug("the json file for converted net is deleted from %s" % buffer_file)
        raise e
    except Exception as e:
        raise e
    
    return net, result_pm
    
def _call_pandamodels(buffer_file, julia_file, dev_mode):
    # This function is now streamlined to only call Julia code
    # The setup is done once in a separate function.
    if dev_mode:
        Pkg.activate("PandaModels")
    
    Main.buffer_file = buffer_file
    result_pm = Main.eval(julia_file + "(buffer_file)")

    if dev_mode:
        Pkg.activate() # Return to default Julia environment after dev run
        
    return result_pm



"""
Functions below modified from: https://github.com/e2nIEE/pandapower/blob/develop/pandapower/converter/pandamodels/to_pm.py
Instead of dumping to a file, we dump to an in-memory string.
This improves efficiency on HPC systems by avoiding disk I/O.
"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex128, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

def dump_pm_json_to_memory(pm):
    """
    Serializes a PowerModels dictionary to a JSON string in memory
    using the correct encoder for NumPy types.
    """
    logger.debug("serializing PowerModels data structure to a JSON string in memory")
    return json.dumps(pm, indent=4, sort_keys=True, cls=NumpyEncoder)

import json

def validate_json_string(json_str):
    """
    Validates that a JSON string contains only JSON-native types
    (dict, list, str, int, float, bool, None).
    Reports if arrays or unexpected types sneak through.
    """
    def _check(obj, path="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _check(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check(v, f"{path}[{i}]")
        else:
            if not isinstance(obj, (str, int, float, bool, type(None))):
                print(f"⚠️ Unexpected type at {path}: {type(obj)} → {obj}")
    
    try:
        parsed = json.loads(json_str)
        _check(parsed)
        print("✅ JSON validation complete. Only valid JSON-native types found.")
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")


def _runpm_in_memory(net, pdm_dev_mode=False, **kwargs):
    """
    Converts the pandapower net to a PowerModels dictionary, and sends the
    data to PandaModels.jl via pyjulia using an in-memory JSON string.
    This completely avoids all disk I/O.
    """
    if not all([jl, Main, Pkg, Base]):
        raise RuntimeError("Julia and PandaModels have not been set up. Please call `setup_julia_and_pandamodels()` once before calling this function.")

    # 1. Convert pandapower net to a PowerModels dictionary in Python memory
    net, pm, ppc, ppci = convert_to_pm_structure(net, **kwargs)
    _add_custom_targets_to_pm_in_place(net, ppci, pm)
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)

    # 2. Serialize the dictionary to a JSON string in memory
    pm_json_string = dump_pm_json_to_memory(pm)
    validate_json_string(pm_json_string)

    # 3. Call the Julia function with the in-memory JSON string
    logger.debug("Starting in-memory call to Julia.")
    
    # We pass the JSON string directly to Julia's Main module.
    # Julia's run function must now accept a string.
    Main.in_memory_json_string = pm_json_string
    
    # The Julia function will now take this string as input
    result_pm = Main.eval(net._options["julia_file"] + "(Main.in_memory_json_string)")
    
    logger.info("Optimization ('"+net._options["julia_file"]+"') " +
                "is finished in %s seconds:" % round(result_pm["solve_time"], 2))
    
    # read results and write back to net
    try:
        read_pm_results_to_net(net, ppc, ppci, result_pm)
    except OPFNotConverged as e:
        raise e
    except Exception as e:
        raise e
    
    return net, result_pm












def _add_custom_targets_to_pm_in_place(net_obj, ppci_obj, pm_obj):
    """
    Directly adds 'target_pg', 'target_vm', 'target_va', 'ws_pg', and 'ws_vm'
    from the pandapower net_obj to the PowerModels data dictionary (pm_obj) in-place.

    Args:
        net_obj (pandapower.Net): The pandapower network.
        ppci_obj (dict): The MATPOWER Case Format (PPC) dictionary. (Not used directly here, but kept for signature)
        pm_obj (dict): The PowerModels data dictionary. This will be modified in-place.
    """
    print("\n--- Executing _add_custom_targets_to_pm_in_place (direct call) ---")
    
    # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
    external_bus_numbers = net_obj.bus.index.values # indices from .m file from pandapower
    internal_indices = ppci_obj['bus'][:, 0].astype(int) #  internal inddices from pypower

    external_to_internal = dict(zip(external_bus_numbers, internal_indices))
    internal_to_external = dict(zip(internal_indices, external_bus_numbers))
    
    ppc_bus_number_to_row = {int(b): i for i, b in enumerate(ppci_obj['bus'][:, BUS_I])}

    # --- Add target_pg to generators ---
    if 'target_pg' in net_obj.gen.columns:
        for pp_idx, gen_row in net_obj.gen.iterrows():
            if pd.notna(gen_row['target_pg']):
                # PowerModels uses 1-based string indices for components
                # and typically stores them in a dict keyed by these strings.
                # pandapower's gen.index are 0-based integers.
                pm_gen_idx = str(pp_idx + 1)
                if pm_gen_idx in pm_obj["gen"]:
                    pm_obj["gen"][pm_gen_idx]["target_pg"] = float(gen_row['target_pg'])
                else:
                    print(f"  WARNING: Gen {pp_idx} (PM index {pm_gen_idx}) not found in pm_obj['gen']. Skipping target_pg.")
    else:
        print("  'target_pg' column not found in net_obj.gen. No generator targets added.")

    # --- Add ws_pg to generators (NEW ADDITION) ---
    if 'ws_pg' in net_obj.gen.columns:
        for pp_idx, gen_row in net_obj.gen.iterrows():
            if pd.notna(gen_row['ws_pg']):
                pm_gen_idx = str(pp_idx + 1)
                if pm_gen_idx in pm_obj["gen"]:
                    pm_obj["gen"][pm_gen_idx]["ws_pg"] = float(gen_row['ws_pg'])
                    # Optionally add ws_qg if it exists and is relevant for warm starts
                    if 'ws_qg' in net_obj.gen.columns and pd.notna(gen_row['ws_qg']):
                        pm_obj["gen"][pm_gen_idx]["ws_qg"] = float(gen_row['ws_qg'])
                    # else: # If you want to explicitly set to 0.0 if ws_qg is not provided
                    #     pm_obj["gen"][pm_gen_idx]["ws_qg"] = 0.0
                else:
                    print(f"  WARNING: Gen {pp_idx} (PM index {pm_gen_idx}) not found in pm_obj['gen']. Skipping ws_pg.")
    else:
        print("  'ws_pg' column not found in net_obj.gen. No generator warm starts added.")


    # --- Add target_vm to buses ---
    if 'target_vm' in net_obj.bus.columns:
        for pp_idx, bus_row in net_obj.bus.iterrows():
            if pd.notna(bus_row['target_vm']):
                
                # pandapower bus index
                ext_bus_idx = pp_idx
                
                internal_bus_num = external_to_internal.get(ext_bus_idx)
                
                pm_bus_idx = str(internal_bus_num + 1)
                
                # pm_bus_idx = str(pp_idx + 1)
                if pm_bus_idx in pm_obj["bus"]:
                    pm_obj["bus"][pm_bus_idx]["target_vm"] = float(bus_row['target_vm'])
                else:
                    print(f"  WARNING: Bus {pp_idx} (PM index {pm_bus_idx}) not found in pm_obj['bus']. Skipping target_vm.")
    else:
        print("  'target_vm' column not found in net_obj.bus. No bus Vm targets added.")

    # --- Add target_va to buses ---
    if 'target_va' in net_obj.bus.columns:
        for pp_idx, bus_row in net_obj.bus.iterrows():
            if pd.notna(bus_row['target_va']):
                
                # pandapower bus index
                ext_bus_idx = pp_idx
                
                internal_bus_num = external_to_internal.get(ext_bus_idx)
                
                pm_bus_idx = str(internal_bus_num + 1)
                
                if pm_bus_idx in pm_obj["bus"]:
                    pm_obj["bus"][pm_bus_idx]["target_va"] = float(bus_row['target_va'])
                else:
                    print(f"  WARNING: Bus {pp_idx} (PM index {pm_bus_idx}) not found in pm_obj['bus']. Skipping target_va.")
    else:
        print("  'target_va' column not found in net_obj.bus. No bus Va targets added.")

    # --- Add ws_vm to buses (NEW ADDITION) ---
    if 'ws_vm' in net_obj.bus.columns:
        for pp_idx, bus_row in net_obj.bus.iterrows():
            if pd.notna(bus_row['ws_vm']):
                # pandapower bus index
                ext_bus_idx = pp_idx
                
                internal_bus_num = external_to_internal.get(ext_bus_idx)
                
                pm_bus_idx = str(internal_bus_num + 1)
                
                if pm_bus_idx in pm_obj["bus"]:
                    pm_obj["bus"][pm_bus_idx]["ws_vm"] = float(bus_row['ws_vm'])
                    # Optionally add ws_va if it exists and is relevant for warm starts
                    if 'ws_va' in net_obj.bus.columns and pd.notna(bus_row['ws_va']):
                        pm_obj["bus"][pm_bus_idx]["ws_va"] = float(bus_row['ws_va'])
                    # else: # If you want to explicitly set to 0.0 if ws_va is not provided
                    #     pm_obj["bus"][pm_bus_idx]["ws_va"] = 0.0
                else:
                    print(f"  WARNING: Bus {pp_idx} (PM index {pm_bus_idx}) not found in pm_obj['bus']. Skipping ws_vm.")
    else:
        print("  'ws_vm' column not found in net_obj.bus. No bus Vm warm starts added.")

    print("--- Finished _add_custom_targets_to_pm_in_place ---")









def convert_vr_vi_to_vm_va(vr_array: np.ndarray, vi_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    # Check if vi_array is a PyTorch tensor and convert it to a NumPy array
    if isinstance(vi_array, torch.Tensor):
        vi_array = vi_array.detach().cpu().numpy()
        
    if isinstance(vr_array, torch.Tensor):
        vr_array = vr_array.detach().cpu().numpy()

    vm_np = np.sqrt(np.square(vr_array) + np.square(vi_array))
    vm_np = np.clip(vm_np, 0.94, 1.06)  # Ensure no negative values before sqrt

    va_rad_np = np.arctan2(vi_array, vr_array)
    va_deg_np = np.degrees(va_rad_np)

    return vm_np, va_deg_np
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def calculate_generator_cost(pg_values_pu: np.ndarray, gencost_data: np.ndarray, baseMVA: float) -> float:
#     """
#     Calculates the total generation cost for a given set of active power outputs.

#     Parameters:
#     - pg_values_pu (np.ndarray): 1D array of active power outputs for generators (in pu).
#                                  Must be aligned with gencost_data (e.g., slack removed if gencost is filtered).
#     - gencost_data (np.ndarray): The filtered 'gencost' matrix from ppc, aligned with pg_values_pu.
#     - baseMVA (float): The base MVA of the system.

#     Returns:
#     - total_cost (float): The total generation cost in USD.
#     """
#     total_cost = 0.0
#     # Convert pg_values from pu to MW for cost calculation
#     pg_values_mw = pg_values_pu * baseMVA

#     for i, gen_cost_row in enumerate(gencost_data):
#         cost_model = int(gen_cost_row[MODEL])
#         pg_i_mw = pg_values_mw[i]

#         if cost_model == POLYNOMIAL:
#             n_coeffs = int(gen_cost_row[NCOST])
#             coeffs = gen_cost_row[COST : COST + n_coeffs] # Coefficients are in descending order of power
            
#             # Calculate polynomial cost: c_n*P^n + ... + c_1*P + c_0
#             gen_cost = 0.0
#             for k in range(n_coeffs):
#                 gen_cost += coeffs[k] * (pg_i_mw ** (n_coeffs - 1 - k))
#             total_cost += gen_cost
        
#         elif cost_model == PW_LINEAR:
#             # Piecewise linear cost: (p0,f0), (p1,f1), ...
#             # NCOST is number of points, so 2*NCOST coefficients
#             n_points = int(gen_cost_row[NCOST])
#             points = gen_cost_row[COST : COST + 2 * n_points].reshape(-1, 2) # Reshape to (n_points, 2)

#             # Find the segment for pg_i_mw
#             # Handle edge cases for points outside the defined range
#             if pg_i_mw <= points[0, 0]:
#                 gen_cost = points[0, 1] # Cost at the lowest point
#             elif pg_i_mw >= points[-1, 0]:
#                 gen_cost = points[-1, 1] # Cost at the highest point
#             else:
#                 for j in range(n_points - 1):
#                     p1, f1 = points[j]
#                     p2, f2 = points[j+1]
#                     if p1 <= pg_i_mw <= p2:
#                         # Linear interpolation
#                         gen_cost = f1 + (pg_i_mw - p1) * (f2 - f1) / (p2 - p1)
#                         break
#                 else: # Should not happen if within range and points are ordered
#                     gen_cost = np.nan # Or raise error
#             total_cost += gen_cost
#         else:
#             # Handle unknown cost model or raise error
#             print(f"Warning: Unknown generator cost model {cost_model}. Skipping cost for this generator.")
#             total_cost += np.nan # Add NaN if cost cannot be computed

#     return total_cost


# def check_feasibility_and_cost(solution_data: dict, ppc: dict, gen_mask_to_keep: np.ndarray) -> tuple[list[bool], list[float], list[dict]]:
#     """
#     Checks the feasibility and calculates the cost for each sample in solution_data.

#     Parameters:
#     - solution_data (dict): Dictionary containing PF results for multiple samples.
#                             Assumed to have keys like 'vm_tot', 'qg_tot', 'pg_tot',
#                             'Ibr_from_r_tot', 'Ibr_from_i_tot', 'Ibr_to_r_tot', 'Ibr_to_i_tot'.
#                             'pg_tot' and 'qg_tot' are assumed to be filtered (e.g., slack removed).
#     - ppc (dict): The base PyPower case dictionary, containing 'bus', 'gen', 'branch', 'gencost', 'baseMVA'.
#     - gen_mask_to_keep (np.ndarray): Boolean mask (shape: num_total_gens) indicating which
#                                      generators are present in solution_data['pg_tot'] and 'qg_tot'.

#     Returns:
#     - is_feasible_list (list[bool]): List of boolean flags, True if feasible, False otherwise for each sample.
#     - cost_list (list[float]): List of total generation costs for each sample. NaN if not feasible or cost cannot be computed.
#     - violation_details_list (list[dict]): List of dictionaries, one per sample, detailing any violations.
#     """
#     num_samples = solution_data['vm_tot'].shape[1]
#     num_buses = ppc['bus'].shape[0]
#     num_gens = ppc['gen'].shape[0]
#     baseMVA = ppc['baseMVA']

#     is_feasible_list = []
#     cost_list = []
#     violation_details_list = []

#     # Filter generator limits and cost data based on gen_mask_to_keep
#     # This aligns the limits/cost data with the 'pg_tot' and 'qg_tot' in solution_data
#     filtered_gen_data = ppc['gen'][gen_mask_to_keep, :]
#     filtered_gencost_data = ppc['gencost'][gen_mask_to_keep, :]

#     # Extract limits for filtered generators
#     pg_max_limits_pu = filtered_gen_data[:, PMAX] / baseMVA
#     pg_min_limits_pu = filtered_gen_data[:, PMIN] / baseMVA
#     qg_max_limits_pu = filtered_gen_data[:, QMAX] / baseMVA
#     qg_min_limits_pu = filtered_gen_data[:, QMIN] / baseMVA

#     # Bus voltage limits for all buses
#     vm_max_limits_all_buses_pu = ppc['bus'][:, VMAX]
#     vm_min_limits_all_buses_pu = ppc['bus'][:, VMIN]

#     # Identify slack bus(es) (internal 0-indexed PyPower bus numbers)
#     slack_bus_internal_indices = np.where(ppc['bus'][:, BUS_TYPE] == REF)[0]
    
#     # Create a mask for non-slack buses for voltage checks
#     all_bus_indices = np.arange(num_buses)
#     non_slack_bus_mask = np.isin(all_bus_indices, slack_bus_internal_indices, invert=True)
    
#     slack_bus_internal_indices = np.where(ppc['bus'][:, BUS_TYPE] == REF)[0]
#     slack_bus_external_ids = ppc['bus'][slack_bus_internal_indices, BUS_I]
#     gens_at_slack_bus_mask = np.isin(ppc['gen'][:, BUS_I], slack_bus_external_ids)
#     non_slack_gen_mask = ~gens_at_slack_bus_mask

#     # Line thermal limits (RATE_A is in MVA, convert to pu)
#     line_rate_a_limits_pu = ppc['branch'][:, RATE_A] / baseMVA

#     # Get internal bus indices for 'from' and 'to' buses of branches
#     # Need a mapping from external bus IDs to internal 0-indexed bus positions in ppc['bus']
#     external_to_ppc_row_idx = {int(bus_row[BUS_I]): i for i, bus_row in enumerate(ppc['bus'])}
    
#     branch_from_bus_external_ids = ppc['branch'][:, F_BUS]
#     branch_to_bus_external_ids = ppc['branch'][:, T_BUS]

#     branch_from_bus_internal_indices = np.array([external_to_ppc_row_idx[int(bus_id)] for bus_id in branch_from_bus_external_ids])
#     branch_to_bus_internal_indices = np.array([external_to_ppc_row_idx[int(bus_id)] for bus_id in branch_to_bus_external_ids])


#     for j in range(num_samples):
#         slack = 1e-3
#         current_sample_feasible = True
#         violations = {} # Dictionary to store specific violation details for the current sample

#         # --- Extract data for current sample ---
#         vm_sample_pu = solution_data['vm_tot'][:, j]
#         pg_sample_pu = solution_data['pg_tot'][:, j]
#         qg_sample_pu = solution_data['qg_tot'][:, j]
#         vr_sample_pu = solution_data['vr_tot'][:, j]
#         vi_sample_pu = solution_data['vi_tot'][:, j]
        
#         # Branch currents (real and imaginary parts)
#         Ibr_from_r_sample_pu = solution_data['Ibr_from_r_tot'][:, j]
#         Ibr_from_i_sample_pu = solution_data['Ibr_from_i_tot'][:, j]
#         Ibr_to_r_sample_pu = solution_data['Ibr_to_r_tot'][:, j]
#         Ibr_to_i_sample_pu = solution_data['Ibr_to_i_tot'][:, j]

#         # --- 1. Check Voltage Magnitude Limits (EXCLUDING slack bus) ---
#         vm_sample_non_slack_pu = vm_sample_pu[non_slack_bus_mask]
#         vm_max_limits_non_slack_pu = vm_max_limits_all_buses_pu[non_slack_bus_mask]
#         vm_min_limits_non_slack_pu = vm_min_limits_all_buses_pu[non_slack_bus_mask]

#         vm_upper_violations = np.where(vm_sample_non_slack_pu > vm_max_limits_non_slack_pu + slack)[0]
#         vm_lower_violations = np.where(vm_sample_non_slack_pu < vm_min_limits_non_slack_pu - slack)[0]
        
#         # Map back to original (all bus) internal indices for reporting
#         original_non_slack_bus_indices = all_bus_indices[non_slack_bus_mask]
        
#         upper_violation_original_indices = original_non_slack_bus_indices[vm_upper_violations]
#         lower_violation_original_indices = original_non_slack_bus_indices[vm_lower_violations]

#         if upper_violation_original_indices.size > 0 or lower_violation_original_indices.size > 0:
#             current_sample_feasible = False
#             violations['vm_violations'] = {
#                 'upper_buses_internal_idx': upper_violation_original_indices.tolist(),
#                 'lower_buses_internal_idx': lower_violation_original_indices.tolist()
#             }

#         # --- 2. Check Generator Reactive Power Limits ---
#         qg_upper_violations = np.where(qg_sample_pu > qg_max_limits_pu[non_slack_gen_mask] + slack)[0]
#         qg_lower_violations = np.where(qg_sample_pu < qg_min_limits_pu[non_slack_gen_mask] - slack)[0]
#         if qg_upper_violations.size > 0 or qg_lower_violations.size > 0:
#             current_sample_feasible = False
#             violations['qg_violations'] = {
#                 'upper_gens_idx': qg_upper_violations.tolist(),
#                 'lower_gens_idx': qg_lower_violations.tolist()
#             }
        
#         # --- 3. Check Generator Active Power Limits ---
#         # (Included for completeness, as NN might directly output Pg)
#         pg_upper_violations = np.where(pg_sample_pu > pg_max_limits_pu[non_slack_gen_mask] + slack)[0]
#         pg_lower_violations = np.where(pg_sample_pu < pg_min_limits_pu[non_slack_gen_mask] - slack)[0]
#         if pg_upper_violations.size > 0 or pg_lower_violations.size > 0:
#             current_sample_feasible = False
#             violations['pg_violations'] = {
#                 'upper_gens_idx': pg_upper_violations.tolist(),
#                 'lower_gens_idx': pg_lower_violations.tolist()
#             }


#         # --- 4. Check Line Thermal Limits ---
#         # Reconstruct complex voltages for from/to buses of branches
#         V_complex_sample = vr_sample_pu + 1j * vi_sample_pu
#         V_f_complex_sample = V_complex_sample[branch_from_bus_internal_indices]
#         V_t_complex_sample = V_complex_sample[branch_to_bus_internal_indices]

#         # Reconstruct complex branch currents (from 'from' bus side)
#         I_br_from_complex_sample = Ibr_from_r_sample_pu + 1j * Ibr_from_i_sample_pu
#         I_br_to_complex_sample = Ibr_to_r_sample_pu + 1j * Ibr_to_i_sample_pu

#         # Calculate apparent power flow magnitude (MVA) at 'from' and 'to' ends of each branch
#         # S = V * conj(I)
#         S_from_mag_pu = np.abs(V_f_complex_sample * np.conj(I_br_from_complex_sample))
#         S_to_mag_pu = np.abs(V_t_complex_sample * np.conj(I_br_to_complex_sample))

#         line_from_violations = np.where(S_from_mag_pu > line_rate_a_limits_pu + slack)[0]
#         line_to_violations = np.where(S_to_mag_pu > line_rate_a_limits_pu + slack)[0]
#         if line_from_violations.size > 0 or line_to_violations.size > 0:
#             current_sample_feasible = False
#             violations['line_violations'] = {
#                 'from_end_branches_internal_idx': line_from_violations.tolist(),
#                 'to_end_branches_internal_idx': line_to_violations.tolist()
#             }
        
#         # --- 5. Calculate Cost (MOVED OUTSIDE FEASIBILITY CHECK) ---
#         current_cost = np.nan # Default to NaN if cost calculation fails
#         try:
#             current_cost = calculate_generator_cost(pg_sample_pu, filtered_gencost_data[non_slack_gen_mask, :], baseMVA)
#         except Exception as e:
#             # This catches errors during cost calculation (e.g., if gencost format is unexpected)
#             print(f"Warning: Could not calculate cost for sample {j}: {e}")
#             current_cost = np.nan 

#         # --- Store results for this sample ---
#         is_feasible_list.append(current_sample_feasible)
#         cost_list.append(current_cost)
#         violation_details_list.append(violations if violations else "No violations")

#         # --- Store results for this sample ---
#         is_feasible_list.append(current_sample_feasible)
#         cost_list.append(current_cost)
#         violation_details_list.append(violations if violations else "No violations")

#     return is_feasible_list, cost_list, violation_details_list