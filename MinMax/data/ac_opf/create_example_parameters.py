import pandas as pd
import numpy as np
import os
import pandapower as pp
import pandapower.converter as pc
from pypower.makeYbus import makeYbus
from pypower.idx_brch import F_BUS, T_BUS, BR_B, BR_R, BR_X
import copy


def create_example_parameters(n_buses: int):
    """
    Creates a basic set of parameters that are used in the following processes:
    * data creation if measurements are to be simulated
    * setting up the neural network model
    * training procedure

    Parameters
    ----------
    n_buses : int
        Integer number of buses in the system.
    case_path : str
        Absolute path to the MATPOWER .m case file (or other format pandapower can load).

    Returns
    -------
    simulation_parameters : dict
        Dictionary that holds all parameters.
    """
    # Initialize net to None before the try block
    net = None 
    
    # ============= specify pglib-opf case based on n_buses ==================
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

    # -----------------------------------------------------------------------------------------------
    # Load the pandapower network from the PGLib case file
    # -----------------------------------------------------------------------------------------------
    try:
        net = pc.from_mpc(case_path, casename_mpc_file=True)
        print(f"✅ Successfully loaded case file: {case_name}")
    except Exception as e:
        print(f"Error loading case file {case_path}: {e}")

    try:
        pp.runpp(net, verbose=False)
        if not net.converged:
            raise RuntimeError("OPF did not converge.")
    except Exception as e:
        print(f"OPF failed: {e}. Falling back to standard Power Flow (runpp)...")

    
    # Get the internal PYPOWER case from the pandapower network
    ppc = pc.to_ppc(net, init = 'flat') # net._ppc
    Sbase = ppc['baseMVA']
    
    # get vector of f and t buses
    fbus = ppc['branch'][:, F_BUS].astype(int)
    tbus = ppc['branch'][:, T_BUS].astype(int)
    
    # Get the mapping from external (Matpower-style) to internal (Pandapower) bus indices
    external_bus_numbers = net.bus.index.values # indices from .m file from pandapower
    internal_indices = ppc['bus'][:, 0].astype(int) #  internal inddices from pypower

    external_to_internal = dict(zip(external_bus_numbers, internal_indices))
    internal_to_external = dict(zip(internal_indices, external_bus_numbers))
    
    # -----------------------------------------------------------------------------------------------
    # Extract parameters from the pandapower network object
    # -----------------------------------------------------------------------------------------------

    # Ensure n_buses matches the actual number of buses in the loaded network
    if n_buses != len(net.bus):
        print(f"Warning: n_buses ({n_buses}) passed to function does not match actual buses in case ({len(net.bus)}). Using actual.")
        n_buses = len(net.bus)
        
    n_gbus = len(net.gen) # Number of generators
    g_bus = net.gen.index.to_numpy() # Generator indices (pandapower uses 0-indexed)

    n_lbus = len(net.load) # Number of loads
    l_bus = net.load.index.to_numpy() # Load indices (pandapower uses 0-indexed)

    # Generator limits
    Pg_max = net.gen['max_p_mw'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)
    Pg_min = net.gen['min_p_mw'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)
    Qg_max = net.gen['max_q_mvar'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)
    Qg_min = net.gen['min_q_mvar'].to_numpy().reshape((1, n_gbus)) if n_gbus > 0 else np.array([]).reshape(1,0)

    # Calculate Gen_delta and Gen_max
    epsilon = 1e-8
    if n_gbus > 0:
        Sg_delta = np.concatenate((Pg_max - Pg_min + epsilon, Qg_max - Qg_min + epsilon), axis=1).reshape((2 * n_gbus, 1))
        Sg_max = np.concatenate((Pg_max, Qg_max), axis=1).reshape((2 * n_gbus, 1))
    else:
        Sg_delta = np.array([]).reshape(0,1)
        Sg_max = np.array([]).reshape(0,1)

    # Bus voltage limits
    if 'max_vm_pu' in net.bus.columns and not net.bus.empty:
        Volt_max = net.bus['max_vm_pu'].fillna(1.10).values  # fill NaNs with 1.10
    else:
        Volt_max = np.ones(len(net.bus)) * 1.10  # default 1.10 for all buses

    if 'min_vm_pu' in net.bus.columns and not net.bus.empty:
        Volt_min = net.bus['min_vm_pu'].fillna(0.90).values  # fill NaNs with 0.90
    else:
        Volt_min = np.ones(len(net.bus)) * 0.90  # default 0.90 for all buses
        
    # line current limits
    base_kv_from_buses = ppc['bus'][fbus, 9]
    mva_limits = ppc['branch'][:, 5]
    current_limits_pu = np.zeros(ppc['branch'].shape[0])
    current_limits_amps = np.zeros(ppc['branch'].shape[0])
    
    for i in range(ppc['branch'].shape[0]):
        mva_limit = mva_limits[i]
        
        # If the MVA limit is non-zero, calculate the current limit
        if mva_limit > 0:
            current_limits_pu[i] = mva_limit / Sbase
            # Actual current limit in Amps:
            current_limits_amps[i] = (mva_limit * 1000) / (np.sqrt(3) * base_kv_from_buses[i])
            
    # Demand limits (using nominal load values as 'max' if no explicit max is defined)
    Pd_max_loads = net.load['p_mw'].to_numpy().reshape(-1, 1)
    Qd_max_loads = net.load['q_mvar'].to_numpy().reshape(-1, 1)
    Pd_max = Pd_max_loads # Keeping the original variable name for consistency with your dict structure
    
    # Assume Pd_min = 0 for all loads (zero minimum demand)
    Pd_min = np.zeros_like(Pd_max_loads)
    Pd_delta = Pd_max_loads - Pd_min
    Pd_min = Pd_min.reshape((n_lbus, 1))
    Pd_delta = Pd_delta.reshape((n_lbus, 1))
    Qd_min = np.zeros_like(Qd_max_loads)  
    Qd_delta = Qd_max_loads - Qd_min
    
    # Stack real and reactive parts if needed
    Sd_max = np.concatenate((Pd_max_loads, Qd_max_loads), axis=0).reshape((2 * n_lbus, 1)) + epsilon
    Sd_min = np.vstack([Pd_min, Qd_min])
    Sd_delta = np.vstack([Pd_delta, Qd_delta]) + epsilon

    # Mappings (Map_g, Map_L)
    Map_g = np.zeros((2 * n_gbus, 2 * n_buses)) # for both P and Q
    # net.gen['bus'].values should contain the correct 0-based internal bus indices
    for i, external_bus_id in enumerate(net.gen['bus'].values):
        internal_bus_idx = external_to_internal[external_bus_id]
        Map_g[i, internal_bus_idx] = 1
        Map_g[n_gbus + i, n_buses + internal_bus_idx] = 1


    Map_L = np.zeros((2 * n_lbus, 2 * n_buses))
    # net.load['bus'].values should contain the correct 0-based internal bus indices
    for i, external_bus_id in enumerate(net.load['bus'].values):
        internal_bus_idx = external_to_internal[external_bus_id]
        Map_L[i, internal_bus_idx] = 1
        Map_L[n_lbus + i, n_buses + internal_bus_idx] = 1
        
    # get bus shunts...
    bus_bs = np.array(ppc['bus'][:, 5]) / Sbase

    # ===============================================================================================
    # Admittance Matrices (Y, Yconj, Ybr, IM) and Line Limits (L_limit) using PYPOWER
    # ===============================================================================================
    try:
        # Use pypower's makeYbus to get the bus admittance matrix (Ybus),
        # and branch admittance matrices for 'from' and 'to' ends (Yf, Yt).
        # ppc['branch'] and ppc['bus'] are the PYPOWER branch and bus matrices.
        Ybus, Yf, Yt = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

        # 1. Y (Bus Admittance Matrix)
        Y = Ybus.todense()

        # 2. Yconj (Imaginary part of Ybus, often referred to as B matrix in power flow)
        Yconj = Y.imag 

        # 3. Ybr (Branch Admittance Matrix for Flows)
        # Yf and Yt are crucial for calculating branch power/current flows in AC.
        Ybr = np.vstack([Yf.real.todense(), Yf.imag.todense(), Yt.real.todense(), Yt.imag.todense()])
        
        # Determine the number of lines/branches from the PYPOWER branch data
        n_line = len(ppc['branch']) 

        # 4. IM (Incidence Matrix for Flows/Currents)
        num_branches = n_line
        num_buses_ppc = len(ppc['bus']) # Use num_buses_ppc if different from n_buses parameter
        
        # KVL incidence matrix for voltage drops
        IM = np.zeros((2 * num_branches, 2 * num_buses_ppc))

        # Get from_bus and to_bus indices from PYPOWER branch data (0-indexed)
        # These are the original bus numbers before internal reordering, if any.
        br_from = ppc['branch'][:, 0].astype(int) 
        br_to = ppc['branch'][:, 1].astype(int)   

        for i in range(num_branches):
            f_bus = br_from[i]
            t_bus = br_to[i]
            
            # Map for real part of voltages (or angles for DC equivalent)
            IM[i, f_bus] = 1
            IM[i, t_bus] = -1

            # Map for imaginary part of voltages (or magnitudes for DC equivalent)
            IM[num_branches + i, num_buses_ppc + f_bus] = 1
            IM[num_branches + i, num_buses_ppc + t_bus] = -1
            
        # KCL incidence matrix
        kcl_im = np.zeros((num_buses_ppc, num_branches), dtype=int)

        # Populate the matrix
        for j in range(num_branches):
            from_bus = br_from[j]
            to_bus = br_to[j]
            
            # Branch j leaves bus 'from_bus'
            kcl_im[from_bus, j] = 1
            
            # Branch j enters bus 'to_bus'
            kcl_im[to_bus, j] = -1
            
        # Convert complex admittance matrices to rectangular form
        Gf, Bf = Yf.real.todense(), Yf.imag.todense()
        Gt, Bt = Yt.real.todense(), Yt.imag.todense()

        Yf_rect = np.vstack([np.hstack([Gf, -Bf]), np.hstack([Bf, Gf])])
        Yt_rect = np.vstack([np.hstack([Gt, -Bt]), np.hstack([Bt, Gt])])
        Ybr_rect = np.vstack([Yf_rect, Yt_rect]) 
        
        # 5. L_limit (Line Limits)
        # For AC OPF, this usually refers to apparent power (MVA) or current limits.
        # In PYPOWER's branch matrix, column 5 (index 5) is RATE_A (MVA limit).
        L_limit = ppc['branch'][:, 5].reshape(1, n_line)
        # Handle lines with zero or NaN limits (often means no limit) by setting a large value
        L_limit[L_limit <= 0] = 99999.0 # Placeholder for effectively infinite capacity
        L_limit[np.isnan(L_limit)] = 99999.0 # Handle NaN limits too
        
        # 6. Identify Slack, PV, and PQ buses
        BUS_TYPE = ppc['bus'][:, 1].astype(int)
        slack_buses = np.where(BUS_TYPE == 3)[0]
        pv_buses = np.where(BUS_TYPE == 2)[0]
        pq_buses = np.where(BUS_TYPE == 1)[0]

        # Slack bus index (used earlier for Bp/Bpp construction)
        slack_bus_idx = slack_buses[0] if len(slack_buses) > 0 else None
        
        # Identify generator indices with Pg_max > 0 excluding the slack generator
        gen_bus_ids = net.gen['bus'].values  # external bus IDs for each generator
        gen_bus_internal = np.array([external_to_internal[bid] for bid in gen_bus_ids])
        pg_active_indices = [i for i, (pmax, bus_idx) in enumerate(zip(Pg_max.flatten(), gen_bus_internal)) if pmax > 1e-9 and bus_idx != slack_bus_idx]
        
        pg_active_bus_indices = [bus_idx for i, (pmax, bus_idx) in enumerate(zip(Pg_max.flatten(), gen_bus_internal))if pmax > 1e-9 and bus_idx != slack_bus_idx]

        
        # 7. Bp and Bpp matrices for Fast Decoupled Load Flow
        # ---------------------------------------------------
        # Standard method: use the imaginary part of Ybus
        B_full = Ybus.imag  # already a scipy sparse matrix

        # Remove slack bus row and column (index assumed to be 0, or fetch from ppc)
        slack_bus_idx = np.where(ppc['bus'][:, 1] == 3)[0][0]  # type 3 = slack
        buses = list(range(len(ppc['bus'])))
        pv_pq_buses = [i for i in buses if i != slack_bus_idx]

        # Extract submatrices for Bp and Bpp
        Bp = -B_full[pv_pq_buses, :][:, pv_pq_buses].tocsc()
        Bpp = -B_full[pq_buses, :][:, pq_buses].tocsc()

        # Optional: convert to dense arrays if needed for torch
        Bp_dense = np.array(Bp.todense())
        Bpp_dense = np.array(Bpp.todense())
        
        # additional data for surrogate
        
        
        # Get series impedance components
        r = ppc['branch'][:, BR_R]
        x = ppc['branch'][:, BR_X]

        # Avoid division by zero
        z = r + 1j * x
        y = 1 / z
        
        # obtain G (conductance) and B (susceptance) of lines
        g = y.real
        b = y.imag
        
        # dummy to try!
        S = np.zeros((2, 4 * n_line))
    
        S[0, 1] = 1          # Ifr[0]
        S[1, n_line+1] = 1     # Ifi[0]


    except Exception as e:
        print(f"Error extracting admittance matrices or line limits using pypower: {e}")
        # Set to None or empty arrays to indicate failure, or raise specific error
        Y, Yconj, Ybr, IM, L_limit, n_line = None, None, None, None, None, 0
        print("Continuing with default (None/empty) values for these matrices. Downstream code may fail.")


    # -----------------------------------------------------------------------------------------------
    # True system parameters dictionary
    # -----------------------------------------------------------------------------------------------
    true_system_parameters = {'Sg_delta': Sg_delta,
                              'Sg_max': Sg_max,
                              'qg_min': Qg_min,
                              'pg_min': Pg_min,
                              'Sd_max': Sd_max,
                              'Sd_min': Sd_min, # This refers to nominal P for loads
                              'Sd_delta': Sd_delta, # This refers to nominal P for loads
                              'Sd_max': Sd_max, # This refers to nominal P for loads
                              'Volt_max': Volt_max,
                              'Volt_min': Volt_min,
                              'I_max_pu': current_limits_pu, # Current limits in pu
                              'Ybus': Y,
                              'Yconj': Yconj, # Placeholder/requires specific AC derivation
                              'Ybr': Ybr,     # Placeholder/requires specific AC derivation
                              'Ybr_rect': Ybr_rect,     # Placeholder/requires specific AC derivation
                              'Map_g': Map_g,
                              'Map_L': Map_L,
                              'kcl_im': kcl_im, # KCL incidence matrix
                              'IM': IM,       # Placeholder/requires specific AC derivation
                              'g_bus': g_bus, # 0-indexed pandapower generator bus indices
                              'n_lbus': n_lbus,
                              'n_line': n_line, # Updated to actual branches
                              'L_limit': L_limit, # Placeholder/requires specific AC derivation
                              'Bp': Bp_dense,
                              'Bpp': Bpp_dense,
                              'pq_buses': pq_buses,
                              'pv_buses': pv_buses,
                              'pv_buses_nz': pg_active_bus_indices,
                              'pg_active': pg_active_indices,
                              'slack_bus': slack_bus_idx,
                              'fbus': fbus,
                              'tbus': tbus,
                              'g': g,
                              'b': b,
                              'S': S,
                              'bus_bs': bus_bs,

                              }

    # -----------------------------------------------------------------------------------------------
    # General parameters dictionary
    # -----------------------------------------------------------------------------------------------
    general_parameters = {'n_buses': n_buses,
                          'g_bus': g_bus,
                          'n_gbus': n_gbus
                          }

    # -----------------------------------------------------------------------------------------------
    # Parameters for data creation
    # -----------------------------------------------------------------------------------------------
    n_data_points = 8_000
    n_test_data_points = 2_000

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_test_data_points': n_test_data_points,
                                's_point': 0}

    # -----------------------------------------------------------------------------------------------
    # Combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters,
                             'net_object': ppc, 
                             'pp_net': net,
                             'nn_output': 'vr_vi'} # 'pg_vm', 'pg_qg', 'vr_vi', 'surrogate'

    return simulation_parameters
