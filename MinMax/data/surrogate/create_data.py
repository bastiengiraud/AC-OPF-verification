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
from pypower.idx_brch import F_BUS, T_BUS, BR_B, BR_R, BR_X
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
    data_dir = os.path.join(base_dir, f'surrogate_data/{n_buses}')
    output_dir = os.path.join(data_dir, f'Dataset')

    
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
    data_dir = os.path.join(base_dir, f'surrogate_data/{n_buses}')
    output_dir = os.path.join(data_dir, f'Dataset')
    

    L_Val = pd.read_csv(os.path.join(output_dir, f'NN_input_{nn_config}.csv'), header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]
    Gen_out = pd.read_csv(os.path.join(output_dir, f'NN_output_{nn_config}.csv'), header=None).to_numpy()[s_point+n_data_points:s_point+n_total][:]
    
    #L_Val=pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_input_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    #Gen_out = pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_output_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
        
    x_test = np.concatenate([L_Val], axis=0)
    return x_test, Gen_out


import cvxpy as cp

def generate_power_system_data(simulation_parameters, save_csv=True):
    
    """ 
    Generates training data for AC OPF by simulating various load profiles,
    solving the AC OPF, and storing inputs (scaled loads) and outputs
    (generator active power and bus voltage magnitudes) to CSV files.
    
    """
    
    # =============== Extract Parameters from simulation_parameters ==============
    true_system_params = simulation_parameters['true_system']
    general_params = simulation_parameters['general']
    data_creation_params = simulation_parameters['data_creation']

    n_buses = general_params['n_buses']
    n_gbus = general_params['n_gbus'] # Number of generators
    n_data_points = data_creation_params['n_data_points']
       
    
    # ============= specify pglib-opf case based on n_buses ==================
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
    
    combination = simulation_parameters['nn_output']
    
    if n_buses == 118:
        case_name = 'pglib_opf_case118_ieee.m'
    elif n_buses == 300:
        case_name = 'pglib_opf_case300_ieee.m'
    elif n_buses == 793:
        case_name = 'pglib_opf_case793_goc.m'
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
    Ybus, Yf, Yt = makeYbus(base_ppc['baseMVA'], base_ppc['bus'], base_ppc['branch'])
    
    # get branch data
    branches = base_ppc['branch']
    n_branch = branches.shape[0]

    fbus = branches[:, 0].astype(int)  # From bus
    tbus = branches[:, 1].astype(int)  # To bus

    # Get admittances of the lines (in series)
    r = branches[:, 2]  # resistance
    x = branches[:, 3]  # reactance
    z = r + 1j * x      # complex impedance
    y = 1 / z           # series admittance

    # Net data
    Sbase = base_ppc['baseMVA'] # Sbase from PYPOWER case
    n_bus = base_ppc['bus'].shape[0]
    n_gens = base_ppc['gen'].shape[0]
    n_loads = len(net.load) # Number of distinct load elements in pandapower's view

    print(f"System details: {n_bus} buses, {n_gens} generators, {n_loads} loads.")

    # Obtain nominal loads (from the loaded case's ppc)
    pd_nom = np.array(net.load['p_mw']).reshape(-1, 1) # Active power for each load element
    qd_nom = np.array(net.load['q_mvar']).reshape(-1, 1) # Reactive power for each load element
    loads_nominal = np.vstack([pd_nom, qd_nom]) # Combined P and Q nominal loads for each load element

    # Define load perturbation bounds (relative to nominal loads)
    lb_factor = 0.6 * np.ones(loads_nominal.shape[0])
    ub_factor = 1.0 * np.ones(loads_nominal.shape[0])

    # Generate load scaling factors
    X_factors = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, n_data_points)

    # Calculate actual load values for each data point (MW/Mvar)
    X_unscaled_loads_mw = loads_nominal * X_factors

    # Separate active and reactive power components for adjustment
    pd_tot_mw = X_unscaled_loads_mw[:n_loads, :]
    qd_tot_mvar = X_unscaled_loads_mw[n_loads:, :]

    # --- Input Scaling (for NN input) ---
    # Convert to per-unit for NN input, as it's common practice
    X_loads_pu = X_unscaled_loads_mw / Sbase

    # Min-max scaling for loads (per unit)
    load_min = np.zeros_like(loads_nominal)  # zero load vector
    load_max = loads_nominal / Sbase          # nominal load in per-unit

    # To safely broadcast, reshape min and max:
    load_min = load_min.reshape((-1, 1))  
    load_max = load_max.reshape((-1, 1))

    # Min-max scale: (x - min) / (max - min), here max-min = max since min=0
    denominator = np.where(load_max == 0, 1, load_max)  # avoid division by zero
    X_scaled = (X_loads_pu - load_min) / denominator  # shape: (2*n_loads, n_data_points)

    # Transpose to (n_data_points, 2*n_loads) for NN input format
    X_nn_input = X_scaled.T

    # --- Output Collection (for NN labels) ---
    vr_f_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    vi_f_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    vr_t_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    vi_t_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    G_l_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    B_l_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    
    I_rect_tot = torch.zeros(2, int(n_data_points*2*n_branch), dtype=torch.float32)
    
    Sf_mag_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    St_mag_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    Pf_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    Qf_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    Pt_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    Qt_tot = torch.zeros(n_branch, int(n_data_points), dtype=torch.float32)
    
    I_mag_tot = torch.zeros(1, int(n_data_points*2*n_branch), dtype=torch.float32)
    
    # -------------------------------------------------------------
    
    # Get the internal PYPOWER case from the pandapower network
    base_ppc = net._ppc
    
    # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
    external_bus_numbers = net.bus.index.values # indices from .m file from pandapower
    internal_indices = base_ppc['bus'][:, 0].astype(int) #  internal inddices from pypower

    external_to_internal = dict(zip(external_bus_numbers, internal_indices))
    internal_to_external = dict(zip(internal_indices, external_bus_numbers))


    # ============ Data Generation Loop (using PYPOWER) ================
    print(f"Solving {n_data_points} AC-OPF problems with PYPOWER...")
    for entry in tqdm(range(int(n_data_points)), position=0, leave=True):
        current_ppc = copy.deepcopy(base_ppc)
        
        for load_idx_pp, bus_idx_pp_internal in net.load['bus'].items():
            # Look up PYPOWER bus ID (MATPOWER bus number) using your dictionary
            original_matpower_bus_id = external_to_internal.get(bus_idx_pp_internal, None)
            
            if original_matpower_bus_id is None:
                print(f"Warning: No PYPOWER bus mapping found for pandapower bus {bus_idx_pp_internal}. Skipping load adjustment.")
                continue
            
            # Find the row in ppc['bus'] corresponding to this bus ID
            ppc_bus_row_idx = np.where(current_ppc['bus'][:, BUS_I] == original_matpower_bus_id)[0]
            
            if len(ppc_bus_row_idx) == 0:
                print(f"Warning: Bus {original_matpower_bus_id} for load {load_idx_pp} not found in PYPOWER bus matrix. Skipping load adjustment.")
                continue
            ppc_bus_row_idx = ppc_bus_row_idx[0]

            # Adjust loads 
            current_ppc['bus'][ppc_bus_row_idx, PD] = pd_tot_mw[load_idx_pp, entry] # / Sbase
            current_ppc['bus'][ppc_bus_row_idx, QD] = qd_tot_mvar[load_idx_pp, entry] # / Sbase

        # Run the OPF with PYPOWER
        try:
            results = runopf(current_ppc, ppopt)
            success = (results['success'] == 1)
        except Exception:
            print(f"Warning: PYPOWER OPF failed for entry {entry}. Error. Skipping this sample.")
   
            continue
        
        # Store results if OPF converged
        if success: # PYPOWER's runopf returns True for success
            vm = results['bus'][:, VM]
            va_rad = np.deg2rad(results['bus'][:, VA])
            vr = vm * np.cos(va_rad)
            vi = vm * np.sin(va_rad)
            
            # get vector of f and t buses
            fbus = results['branch'][:, F_BUS].astype(int)
            tbus = results['branch'][:, T_BUS].astype(int)
            
            # Get series impedance components
            r = results['branch'][:, BR_R]
            x = results['branch'][:, BR_X]

            # Avoid division by zero
            z = r + 1j * x
            y = 1 / z
            
            # obtain G (conductance) and B (susceptance) of lines
            g = y.real
            b = y.imag
            
            # get vectorized rectangular voltages
            vr_f = vr[fbus]
            vr_t = vr[tbus]
            vi_f = vi[fbus]
            vi_t = vi[tbus]
            
            # get complex flows
            V = vm * np.exp(1j * np.deg2rad(va_rad))            
            
            # Compute line currents
            If = Yf @ V   # Current injected into the "from" end of each line
            Ifr = If.real
            Ifi = If.imag
            Ifmag = Ifr**2 + Ifi**2
            
            It = Yt @ V   # Current injected into the "to" end of each line
            Itr = It.real
            Iti = It.imag
            Itmag = Itr**2 + Iti**2
            
            # stack current from both ends
            If_rect = np.vstack((Ifr, Ifi))
            It_rect = np.vstack((Itr, Iti))
            Irect = np.hstack((If_rect, It_rect))
            Imag = np.hstack((Ifmag, Itmag))
                        
            # Complex power flows
            Sf = V[fbus] * np.conj(If)
            St = V[tbus] * np.conj(It)

            # Real and reactive parts
            Sf_mag = np.abs(Sf)
            St_mag = np.abs(St)
            Pf = Sf.real
            Qf = Sf.imag
            Pt = St.real
            Qt = St.imag
            
            # store NN inputs
            start_idx = entry * 2 * n_branch
            end_idx = (entry + 1) * 2 * n_branch

            vr_f_tot[:, entry] = torch.tensor(vr_f, dtype=torch.float32)
            vi_f_tot[:, entry] = torch.tensor(vi_f, dtype=torch.float32)
            vr_t_tot[:, entry] = torch.tensor(vr_t, dtype=torch.float32)
            vi_t_tot[:, entry] = torch.tensor(vi_t, dtype=torch.float32)
            G_l_tot[:, entry] = torch.tensor(g, dtype=torch.float32)
            B_l_tot[:, entry] = torch.tensor(b, dtype=torch.float32)
            
            I_rect_tot[:, start_idx:end_idx] = torch.tensor(Irect, dtype=torch.float32)
            
            # store NN outputs
            Sf_mag_tot[:, entry] = torch.tensor(Sf_mag, dtype=torch.float32)
            St_mag_tot[:, entry] = torch.tensor(St_mag, dtype=torch.float32)
            Pf_tot[:, entry] = torch.tensor(Pf, dtype=torch.float32)
            Qf_tot[:, entry] = torch.tensor(Qf, dtype=torch.float32)
            Pt_tot[:, entry] = torch.tensor(Pt, dtype=torch.float32)
            Qt_tot[:, entry] = torch.tensor(Qt, dtype=torch.float32)
            
            I_mag_tot[:, start_idx:end_idx] = torch.tensor(Imag, dtype=torch.float32)
            
            

        else:
            print(f"Warning: PYPOWER OPF did not converge for entry {entry}. Storing zeros.")


    if combination == 'surrogate_s':
        # Obtain labels (NN output)
        X_nn = torch.stack([
            vr_f_tot,  # shape (n_branch, n_data_points)
            vi_f_tot,
            vr_t_tot,
            vi_t_tot,
            G_l_tot,
            B_l_tot
        ], dim=2)  # shape becomes (n_branch, n_data_points, 6)

        X_nn = X_nn.permute(1, 0, 2).reshape(-1, 6)  # shape (n_branch * n_data_points, 6)
        
        Y_nn = torch.stack([
            Sf_mag_tot,
            St_mag_tot,
            Pf_tot,
            Qf_tot,
            Pt_tot,
            Qt_tot
        ], dim=2)  # shape (n_branch, n_data_points, 6)

        Y_nn = Y_nn.permute(1, 0, 2).reshape(-1, 6)  # shape (n_branch * n_data_points, 6)
        
        Y_nn_output = Y_nn  # transpose to (n_data_points, features)
        X_nn_input = X_nn
        
    elif combination == 'surrogate_i':
        # Obtain labels (NN output)
        X_nn = torch.stack([
            I_rect_tot,  
        ], dim=2)  

        X_nn = X_nn.permute(1, 0, 2).reshape(-1, 2)  
        
        Y_nn = torch.stack([
            I_mag_tot,
        ], dim=2)  

        Y_nn = Y_nn.permute(1, 0, 2).reshape(-1, 1)  
        
        Y_nn_output = Y_nn  # transpose to (n_data_points, features)
        X_nn_input = X_nn
        
    else:
        raise ValueError("Surrogate type not recognized!")
        
    

    # --- Save to CSV Files ---
    output_data_dir = os.path.join(
        base_dir,
        'data/surrogate_data', # Or your desired top-level folder for AC_OPF data
        str(n_buses),
        'Dataset'
    )

    os.makedirs(output_data_dir, exist_ok=True)
    print(f"Saving generated data to: {output_data_dir}")

    input_filename = os.path.join(output_data_dir, f"NN_input_{combination}.csv")
    output_filename = os.path.join(output_data_dir, f"NN_output_{combination}.csv")

    if save_csv:
        pd.DataFrame(X_nn_input).to_csv(input_filename, index=False, header=False)
        pd.DataFrame(Y_nn_output).to_csv(output_filename, index=False, header=False)

    print("Data generation and saving complete.")
    print(f"NN Input shape (for CSV): {X_nn_input.shape}")
    print(f"NN Output shape (for CSV): {Y_nn_output.shape}")

    return X_nn_input, Y_nn_output



# --- Example Usage (How to call this function) ---
if __name__ == "__main__":
    
    from data.surrogate.create_example_parameters import create_example_parameters

    test_n_buses = 118 # Example: for a 6-bus system
    
    # Check if data_dir for parameters exists before attempting to load
    current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_data_dir = os.path.join(current_script_dir, f'surrogate_data/{test_n_buses}')

    if not os.path.exists(param_data_dir):
        print(f"Error: Parameter data directory not found: {param_data_dir}")
        print("Please ensure you have the correct CSV files ")
        print(f"in a folder named surrogate_data/{test_n_buses} relative to this script.")
    else:
        # 1. Create/Load simulation parameters (from your provided function)
        simulation_params = create_example_parameters(test_n_buses)

        # 2. Generate training data using the new function
        X_train_data, Y_train_data = generate_power_system_data(simulation_params)

        print("\nGenerated Training Data Shapes:")
        print(f"Scaled Load Profiles (X_train_data): {X_train_data.shape}")
        print(f"Generator Dispatches (Y_train_data): {Y_train_data.shape}")

