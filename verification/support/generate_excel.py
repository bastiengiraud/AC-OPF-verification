import os
import numpy as np
import pandas as pd
from pypower.api import ppoption, runopf, makeYbus
import pandapower.converter as pc
import pandapower as pp
#from pandapower.pypower import pp_maker


"""
cd "/home/bagir/Documents/1) Projects/2) AC verification/verification"


"""

def create_bounds_file(n_buses: int, output_type: str = None):
    """
    Creates an Excel file containing the load bounds for a given power system case.

    The function loads a standard MATPOWER case, extracts the nominal active and reactive
    loads, and calculates the min/max bounds as a percentage deviation from the nominal values.

    Args:
        n_buses (int): The number of buses in the system to load. Supported cases
                       are 118, 300, 793, 1354, and 2869.
        deviation (float): The percentage deviation from the nominal load to use for
                           the min/max bounds. A value of 0.5 means +/- 50%.
    """
    # --- Part 1: Load the power system case and extract nominal loads ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(base_dir)

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
    else:
        raise ValueError(f"No case file found for n_buses = {n_buses}. Supported cases are 118, 300, 793, 1354, 2869.")
        
    case_path = os.path.join(base_dir, 'MinMax/pglib-opf', case_name)
    
    # Load the MATPOWER case from a local .m file
    net = pc.from_mpc(case_path, casename_mpc_file=True)
    base_ppc = pc.to_ppc(net, init = 'flat')
    
    # Run OPF (not strictly necessary for this task, but kept from original code)
    ppopt = ppoption(OUT_ALL=0)
    results = runopf(base_ppc, ppopt)
    Ybus, _, _ = makeYbus(base_ppc['baseMVA'], base_ppc['bus'], base_ppc['branch'])
    Sbase = base_ppc['baseMVA']
    
    n_loads = len(net.load) # Number of distinct load elements
    
    # Obtain nominal loads (from the loaded case's ppc)
    pd_nom = np.array(net.load['p_mw']).reshape(-1, 1) / Sbase # Active power for each load element
    qd_nom = np.array(net.load['q_mvar']).reshape(-1, 1) / Sbase # Reactive power for each load element
    
    print(f"System details: {n_buses} buses, {len(net.gen)} generators, {n_loads} loads.")
    
    # --- Part 2: Calculate the min/max bounds for the loads ---
    pd_min = 0.6 * pd_nom
    pd_max = 1.0 * pd_nom 
    
    qd_min = 0.6 * qd_nom 
    qd_max = 1.0 * qd_nom 
    
    # Stack the min and max values for the 'lower bound' and 'upper bound' columns
    lower_bounds = np.vstack([pd_min, qd_min])
    upper_bounds = np.vstack([pd_max, qd_max])
    
    # --- Part 3: Create the input features list ---
    # Create a list of strings like 'X_0', 'X_1', ...
    feature_list = [f"X_{i}" for i in range(2 * n_loads)]
    
    # --- Part 4: Assemble the DataFrame and export to Excel ---
    input_data_dict = {
        'Input features': feature_list,
        'Lower bound': lower_bounds.flatten(),
        'Upper bound': upper_bounds.flatten()
    }
    input_bounds_df = pd.DataFrame(input_data_dict)
    
    # Define the output directory and create it if it doesn't exist
    output_dir = os.path.join(base_dir, 'verification', 'nn_models', 'bounds')
    output_dir_crown = os.path.join(base_dir, 'verification', 'alpha-beta-CROWN/complete_verifier/acopf')
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"load_bounds_case{n_buses}.xlsx"
    output_path = os.path.join(output_dir, output_filename)
    output_path_crown = os.path.join(output_dir_crown, output_filename)

    with pd.ExcelWriter(output_path) as writer:
        input_bounds_df.to_excel(writer, sheet_name='inputs', index=False)
        
        # Add the outputs sheet if the output_type is 'vrvi'
        if output_type == 'vrvi':
            output_list = list(range(2 * n_buses))
            outputs_df = pd.DataFrame({'outputs': output_list})
            outputs_df.to_excel(writer, sheet_name='outputs', index=False)
            
    with pd.ExcelWriter(output_path_crown) as writer:
        input_bounds_df.to_excel(writer, sheet_name='inputs', index=False)
        
        # Add the outputs sheet if the output_type is 'vrvi'
        if output_type == 'vrvi':
            output_list = list(range(2 * n_buses))
            outputs_df = pd.DataFrame({'outputs': output_list})
            outputs_df.to_excel(writer, sheet_name='outputs', index=False)
    
    print(f"Successfully created '{output_filename}' at path '{output_path}'.")


if __name__ == '__main__':
    create_bounds_file(n_buses=118, output_type = 'vrvi')  