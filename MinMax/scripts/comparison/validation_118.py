"""

Pypower (primal-dual interior point method) is faster than powermodels for AC-OPF, and it's maybe not that strange: 
https://energy-markets-school.dk/wp/wp-content/uploads/2024/05/PowerModels.jl-and-an-Introduction-to-Quantum-Computing-Carleton-Coffrin.pdf
https://arxiv.org/pdf/2203.11328

# -------- how to run this script: ------------
cd scripts/comparison
conda activate ab-crown-old
python-jl validation.py # PyJulia call to run julia code form python
# --------------------------------------------

# call julia on HPC: /zhome/f7/1/166694/apps/julia-1.6.7/bin/julia

"""
# import julia
# print(julia.Julia().eval('Sys.BINDIR'))

import numpy as np
import pandas as pd
import os
import sys
import torch
import pandapower as pp
import copy
import time
import pickle


# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils', 'config', 'scripts/comparison', 'scripts/validation', 'scripts/validation/nn_inference']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))
    

from ac_opf.create_example_parameters import create_example_parameters
import loadsampling as ls
from types import SimpleNamespace

from validation_support import solve_ac_opf_and_collect_data, load_and_prepare_voltage_nn_for_inference, load_and_prepare_power_nn_for_inference, \
    voltage_nn_inference, power_nn_inference, compare_accuracy_with_mse, print_comparison_table, runpm_opf, \
        extract_solution_data_from_pandapower_net, convert_vr_vi_to_vm_va, power_nn_projection, voltage_nn_projection, \
        power_nn_warm_start, voltage_nn_warm_start, calculate_violations, print_violations_comparison_table, setup_julia_and_pandamodels


def create_config():
    parameters_dict = {
        'test_system': 793,
        'hidden_layer_size': 50,
        'n_hidden_layers': 3,
        'epochs': 1000,
        'batch_size': 50,
        'learning_rate': 1e-3,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'pg_viol_weight': 1e1,
        'qg_viol_weight': 0,
        'vm_viol_weight': 1e1,
        'line_viol_weight': 1e1,
        'crit_weight': 1e1,
        'PF_weight': 1e1,
        'LPF_weight': 1e1,
        'N_enrich': 50,
        'Algo': True, # if True, add worst-case violation CROWN bounds during training
        'Enrich': False,
        'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
    }
    config = SimpleNamespace(**parameters_dict)
    return config

def main(only_violations):  
    
    
    # import julia
    # from julia import Main
    # import os
    
    # julia_project_path = "/home/bagir/Documents/1) Projects/2) AC verification/MinMax/scripts/comparison"
    # Main.eval(f'using Pkg; Pkg.activate("{julia_project_path}")')
    # print(f"Julia project activated: {julia_project_path}")

    # Now, try to load PandaModels and get its path within this Julia session
    # try:
    #     Main.eval('using PandaModels')
    #     loaded_pandamodels_path = Main.eval('Base.pathof(PandaModels)')
    #     print(f"PandaModels.jl is loaded from: {loaded_pandamodels_path}")

    #     # You can also confirm Pkg.status from within the Python-initiated Julia session
    #     pkg_status_output = Main.eval('using Pkg; sprint(io -> Pkg.status(io=io))')
    #     print("\n--- Pkg.status('PandaModels') from Python-initiated Julia session ---")
    #     print(pkg_status_output)
    #     print("--------------------------------------------------------------------")

    # except Exception as e:
    #     print(f"Error checking PandaModels.jl path in Julia: {e}")
    #     print("This might happen if PandaModels.jl is not accessible or installed in this Julia environment.")

    try:
        setup_julia_and_pandamodels()
    except (ImportError, RuntimeError) as e:
        print(f"Failed to set up Julia environment: {e}")
        exit()
        
    # --- The rest of your main script goes here ---
    # Your original main function would be called here
    # E.g., main(only_violations=True)
    
    print("Julia and PandaModels setup complete. You can now call runpm_opf multiple times without re-initialization.")
    
    config = create_config()
    
    # define test system
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)
    net = simulation_parameters['pp_net']
    
    # Configuration
    num_solves = 200
    seed = 42
    TARGET_NUM_SAMPLES = 200
    SOLUTION_DATA_FILE = f"solution_data_1000_samples_{n_buses}_bus.pkl"
    
    solution_data_dict = solve_ac_opf_and_collect_data(seed, n_buses, TARGET_NUM_SAMPLES)

    # # --- Load or create solution data dict ---
    # if os.path.exists(SOLUTION_DATA_FILE):
    #     # Load existing data
    #     with open(SOLUTION_DATA_FILE, "rb") as f:
    #         solution_data_dict = pickle.load(f)
    #     existing_samples = solution_data_dict['pg_tot'].shape[1]  # Assuming shape [n_gens, n_samples]
    #     print(f"Loaded solution data with {existing_samples} samples from {SOLUTION_DATA_FILE}.")

    #     # Check if more samples are needed
    #     if existing_samples < TARGET_NUM_SAMPLES:
    #         num_additional_samples = TARGET_NUM_SAMPLES - existing_samples
    #         print(f"Generating {num_additional_samples} additional samples...")

    #         # Generate additional samples (modify your function to accept a start index)
    #         additional_data = solve_ac_opf_and_collect_data(seed, n_buses, num_additional_samples)

    #         # Merge with existing dict
    #         for key in solution_data_dict.keys():
    #             solution_data_dict[key] = np.concatenate(
    #                 (solution_data_dict[key], additional_data[key]),
    #                 axis=1  # Concatenate along the sample dimension
    #             )

    #         # Save updated dict
    #         with open(SOLUTION_DATA_FILE, "wb") as f:
    #             pickle.dump(solution_data_dict, f)
    #         print(f"Updated solution data saved with {TARGET_NUM_SAMPLES} samples.")
    # else:
    #     # File does not exist → generate all data
    #     print(f"Generating full solution data with {TARGET_NUM_SAMPLES} samples...")
    #     solution_data_dict = solve_ac_opf_and_collect_data(seed, n_buses, TARGET_NUM_SAMPLES)

    #     # Save to file
    #     with open(SOLUTION_DATA_FILE, "wb") as f:
    #         pickle.dump(solution_data_dict, f)
    #     print(f"Solution data saved to {SOLUTION_DATA_FILE}. Exiting.")
    #     exit()  # Stop script after saving
    
    if n_buses == 14:
        nn_file_name_power_false                = 'checkpoint_14_15_False_pg_vm_final.pt' # 'checkpoint_118_50_False_pg_vm_final.pt'
        nn_file_name_power_true                 = 'checkpoint_14_15_True_pg_vm_final.pt' #'checkpoint_118_50_True_pg_vm_final.pt'
        nn_file_name_volt_false                 = 'checkpoint_14_15_False_vr_vi_final.pt' # 'checkpoint_118_50_False_vr_vi_final.pt'
        nn_file_name_volt_true                  = 'checkpoint_14_15_True_vr_vi_final.pt' # 'checkpoint_118_50_True_vr_vi_final.pt'
    elif n_buses == 57:
        nn_file_name_power_false                = 'checkpoint_57_25_False_pg_vm_final.pt' # 'checkpoint_118_50_False_pg_vm_final.pt'
        nn_file_name_power_true                 = 'checkpoint_57_25_True_pg_vm_final.pt' #'checkpoint_118_50_True_pg_vm_final.pt'
        nn_file_name_volt_false                 = 'checkpoint_57_25_False_vr_vi_final.pt' # 'checkpoint_118_50_False_vr_vi_final.pt'
        nn_file_name_volt_true                  = 'checkpoint_57_25_True_vr_vi_final.pt' # 'checkpoint_118_50_True_vr_vi_final.pt'
    elif n_buses == 118:
        nn_file_name_power_false                = 'checkpoint_118_50_False_pg_vm_final.pt'
        nn_file_name_power_true                 = 'checkpoint_118_50_True_pg_vm_final.pt'
        nn_file_name_volt_false                 = 'checkpoint_118_50_False_vr_vi_final.pt'
        nn_file_name_volt_true                  = 'checkpoint_118_50_True_vr_vi_final.pt'
    elif n_buses == 300:
        nn_file_name_power_false                = 'checkpoint_300_75_False_pg_vm_final.pt'
        nn_file_name_power_true                 = 'checkpoint_300_75_True_pg_vm_final.pt'
        nn_file_name_volt_false                 = 'checkpoint_300_75_False_vr_vi_final.pt'
        nn_file_name_volt_true                  = 'checkpoint_300_75_True_vr_vi_final.pt'
    elif n_buses == 793:
        nn_file_name_power_false                = 'checkpoint_793_100_False_pg_vm_final.pt'
        nn_file_name_power_true                 = 'checkpoint_793_100_True_pg_vm_final.pt'
        nn_file_name_volt_false                 = 'checkpoint_793_100_False_vr_vi_final.pt'
        nn_file_name_volt_true                  = 'checkpoint_793_100_True_vr_vi_final.pt'
        


    # ----------- load the Pg Vm model without worst-case penalties ------------    
    power_net_false                      = load_and_prepare_power_nn_for_inference(nn_file_name_power_false, n_buses, config, simulation_parameters, solution_data_dict)
    power_nn_results_false               = power_nn_inference(net, n_buses, power_net_false, solution_data_dict, simulation_parameters)
    mse_power_nn_false                   = compare_accuracy_with_mse(solution_data_dict, power_nn_results_false)
    power_violations_false               = calculate_violations(power_nn_results_false, simulation_parameters)

    pg_targets_false                     = power_nn_results_false['pg_tot']
    vm_targets_false                     = power_nn_results_false['vm_tot']
    
    # store results in a dict
    all_violation_results                = [mse_power_nn_false]
    all_violation_names                  = ["Pg Vm Model False"]
    
    # ----------- load the Pg Vm model with worst-case penalties ------------    
    power_net_true                      = load_and_prepare_power_nn_for_inference(nn_file_name_power_true, n_buses, config, simulation_parameters, solution_data_dict)
    power_nn_results_true               = power_nn_inference(net, n_buses, power_net_true, solution_data_dict, simulation_parameters)
    mse_power_nn_true                   = compare_accuracy_with_mse(solution_data_dict, power_nn_results_true)
    power_violations_true               = calculate_violations(power_nn_results_true, simulation_parameters)

    pg_targets_true                     = power_nn_results_true['pg_tot']
    vm_targets_true                     = power_nn_results_true['vm_tot']
    
    # store results in a dict
    all_violation_results.append(mse_power_nn_true)
    all_violation_names.append("Pg Vm Model True")
    
    # ----------- load the Vr Vi model without worst-case penalties -------------   
    voltage_net_false                          = load_and_prepare_voltage_nn_for_inference(nn_file_name_volt_false, n_buses, config, simulation_parameters, solution_data_dict)
    voltage_nn_results_false                   = voltage_nn_inference(net, n_buses, voltage_net_false, solution_data_dict, simulation_parameters)
    mse_voltage_nn_false                       = compare_accuracy_with_mse(solution_data_dict, voltage_nn_results_false)
    voltage_violations_false                   = calculate_violations(voltage_nn_results_false, simulation_parameters)
    
    vr_targets_false                           = voltage_nn_results_false['vr_tot']
    vi_targets_false                           = voltage_nn_results_false['vi_tot']
    
    # add results to the all_mse_results
    all_violation_results.append(mse_voltage_nn_false)
    all_violation_names.append("Vr Vi Model False")
    
    # ----------- load the Vr Vi model with worst-case penalties -------------   
    voltage_net_true                          = load_and_prepare_voltage_nn_for_inference(nn_file_name_volt_true, n_buses, config, simulation_parameters, solution_data_dict)
    voltage_nn_results_true                   = voltage_nn_inference(net, n_buses, voltage_net_true, solution_data_dict, simulation_parameters)
    mse_voltage_nn_true                       = compare_accuracy_with_mse(solution_data_dict, voltage_nn_results_true)
    voltage_violations_true                   = calculate_violations(voltage_nn_results_true, simulation_parameters)
    
    vr_targets_true                           = voltage_nn_results_true['vr_tot']
    vi_targets_true                           = voltage_nn_results_true['vi_tot']
    
    # add results to the all_mse_results
    all_violation_results.append(mse_voltage_nn_true)
    all_violation_names.append("Vr Vi Model True")
    
    # -------------- compare the results ----------------
    violations_dicts_to_compare = [power_violations_false, power_violations_true, voltage_violations_false, voltage_violations_true]
    model_names_list = ["Pg Vm Model False", "Pg Vm Model True", "Vr Vi Model False", "Vr Vi Model True"]
    print_violations_comparison_table(
        violations_dicts_to_compare,
        model_names_list,
        num_bus = n_buses,
        num_samp=num_solves
    )
    
    print_comparison_table(
        mse_results_list=all_violation_results,
        model_names=all_violation_names,
        table_title="Accuracy Comparison of Models",
        num_bus = n_buses,
        num_samp=num_solves,
        goal="inference"
    )
    
    
    if only_violations == False:
        
        """ 
        Time the projection and warm start! First the NNs trained without w/c penalties:
        """
    
        # --------------- do the projections ----------------   
        time_total_false = {}
        not_conv_power_proj_false, pgvm_projected_results_false               = power_nn_projection(net, num_solves, solution_data_dict, pg_targets_false, vm_targets_false)
        mse_pgvm_projection_false                  = compare_accuracy_with_mse(solution_data_dict, pgvm_projected_results_false)

        # add results to the all_mse_results
        all_mse_results_false                      = [mse_pgvm_projection_false]
        all_model_names_false                      = ["Pg Vm Projection False"]
        time_total_false['Pg Vm Projection False'] = pgvm_projected_results_false['solve_time']
        
        not_conv_volt_proj_false, vrvi_projected_results_false               = voltage_nn_projection(net, num_solves, solution_data_dict, vr_targets_false, vi_targets_false)
        mse_vrvi_projection_false                  = compare_accuracy_with_mse(solution_data_dict, vrvi_projected_results_false)
        
        # add results to the all_mse_results
        all_mse_results_false.append(mse_vrvi_projection_false)
        all_model_names_false.append("Vr Vi Projection False")
        time_total_false['Vr Vi Projection False'] = vrvi_projected_results_false['solve_time']
        
        # -------------- do the warm starts ----------------
        not_conv_power_ws_false, pgvm_ws_results_false                      = power_nn_warm_start(net, num_solves, solution_data_dict, pg_targets_false, vm_targets_false)
        mse_pgvm_ws_false                          = compare_accuracy_with_mse(solution_data_dict, pgvm_ws_results_false)
        
        # add results to the all_mse_results
        all_mse_results_false.append(mse_pgvm_ws_false)
        all_model_names_false.append("Pg Vm Warm Start False")
        time_total_false['Pg Vm Warm Start False'] = pgvm_ws_results_false['solve_time']
        
        not_conv_volt_proj_false, vrvi_ws_results_false                      = voltage_nn_warm_start(net, num_solves, solution_data_dict, vr_targets_false, vi_targets_false)
        mse_vrvi_ws_false                          = compare_accuracy_with_mse(solution_data_dict, vrvi_ws_results_false)
        
        # add results to the all_mse_results
        all_mse_results_false.append(mse_vrvi_ws_false)
        all_model_names_false.append("Vr Vi Warm Start False")
        time_total_false['Pg Vm Warm Start False'] = vrvi_ws_results_false['solve_time']
        
        
        """ 
        Then the NNs trained with w/c penalties:
        """
    
        # --------------- do the projections ----------------   
        time_total_true = {}
        not_conv_power_proj_true, pgvm_projected_results_true               = power_nn_projection(net, num_solves, solution_data_dict, pg_targets_true, vm_targets_true)
        mse_pgvm_projection_true                  = compare_accuracy_with_mse(solution_data_dict, pgvm_projected_results_true)

        # add results to the all_mse_results
        all_mse_results_true                      = [mse_pgvm_projection_true]
        all_model_names_true                      = ["Pg Vm Projection True"]
        time_total_true['Pg Vm Projection True'] = pgvm_projected_results_true['solve_time']
        
        not_conv_volt_proj_true, vrvi_projected_results_true               = voltage_nn_projection(net, num_solves, solution_data_dict, vr_targets_true, vi_targets_true)
        mse_vrvi_projection_true                  = compare_accuracy_with_mse(solution_data_dict, vrvi_projected_results_true)
        
        # add results to the all_mse_results
        all_mse_results_true.append(mse_vrvi_projection_true)
        all_model_names_true.append("Vr Vi Projection True")
        time_total_true['Vr Vi Projection True'] = vrvi_projected_results_true['solve_time']
        
        # -------------- do the warm starts ----------------
        not_conv_power_ws_true, pgvm_ws_results_true                      = power_nn_warm_start(net, num_solves, solution_data_dict, pg_targets_true, vm_targets_true)
        mse_pgvm_ws_true                          = compare_accuracy_with_mse(solution_data_dict, pgvm_ws_results_true)
        
        # add results to the all_mse_results
        all_mse_results_true.append(mse_pgvm_ws_true)
        all_model_names_true.append("Pg Vm Warm Start True")
        time_total_true['Pg Vm Warm Start True'] = pgvm_ws_results_true['solve_time']
        
        not_conv_volt_proj_true, vrvi_ws_results_true                      = voltage_nn_warm_start(net, num_solves, solution_data_dict, vr_targets_true, vi_targets_true)
        mse_vrvi_ws_true                          = compare_accuracy_with_mse(solution_data_dict, vrvi_ws_results_true)
        
        # add results to the all_mse_results
        all_mse_results_true.append(mse_vrvi_ws_true)
        all_model_names_true.append("Vr Vi Warm Start True")
        time_total_true['Vr Vi Warm Start True'] = vrvi_ws_results_true['solve_time']
        
        # -------------- compare the results ----------------
        print(
            f"Number of times the power flow did not converge out of {num_solves} solves:\n"
            f"Power NN true {num_solves - int((power_nn_results_true['pf_convergence']).sum().item())} out of {num_solves},\n"
            f"Power NN false {num_solves - int((power_nn_results_false['pf_convergence']).sum().item())} out of {num_solves},\n"
            f"Number of not converged projections/warm starts for models without w/c penalties:\n"
            f"Power Projection {not_conv_power_proj_false} out of {num_solves},\n"
            f"Voltage Projection {not_conv_volt_proj_false} out of {num_solves},\n"
            f"Power WS {not_conv_power_ws_false} out of {num_solves},\n"
            f"Voltage WS {not_conv_volt_proj_false} out of {num_solves}.\n"
            f"Number of not converged projections/warm starts for models with w/c penalties:\n"
            f"Power Projection {not_conv_power_proj_true} out of {num_solves},\n"
            f"Voltage Projection {not_conv_volt_proj_true} out of {num_solves},\n"
            f"Power WS {not_conv_power_ws_true} out of {num_solves},\n"
            f"Voltage WS {not_conv_volt_proj_true} out of {num_solves}."
        )

        #-------- collect all time results and print csv table --------
        df_false = pd.DataFrame(list(time_total_false.items()), columns=["Key", "Time_False"])
        df_true = pd.DataFrame(list(time_total_true.items()), columns=["Key", "Time_True"])

        # Merge them on the "Key" column
        df = pd.merge(df_false, df_true, on="Key", how="outer")

        # Save to Excel
        df.to_excel(f"time_totals_{n_buses}.xlsx", index=False)
        # -------------------------------------------------------------
        
        print_comparison_table(
            mse_results_list=all_mse_results_false,
            model_names=all_model_names_false,
            table_title="Accuracy Comparison of Proxies False",
            num_bus = n_buses,
            num_samp=num_solves,
            goal="False"
        )
        
        print_comparison_table(
            mse_results_list=all_mse_results_true,
            model_names=all_model_names_true,
            table_title="Accuracy Comparison of Proxies True",
            num_bus = n_buses,
            num_samp=num_solves,
            goal="True"
        )
    
    
    




if __name__ == '__main__':
        
    # only compare statistical violations
    only_violations = False # True = without projection & warmstart, False = with projection & warm start
    
    main(only_violations)
    

    

    
    
    
    
    
    # import copy
    # import time
    # import numpy as np
    # import pandapower as pp
    # import pandapower.converter as pc
    # from pypower.api import runopf, ppoption
    # from tqdm import tqdm
    # # Note: You need to have the pandamodels library installed to run this part
    # # from pandamodels.opf.run_opf import run_pm_opf

    # # ==================================
    # # SCRIPT CONFIGURATION
    # # ==================================
    # NUM_RUNS = 10  # Number of OPF solves to time


    # def sample_kumaraswamy_loads(net, seed):
    #     """
    #     Samples new load values for a pandapower network using a Kumaraswamy distribution.
        
    #     Args:
    #         net (pandapower.Net): The network to modify.
        
    #     Returns:
    #         pandapower.Net: A new network with modified loads.
    #     """
    #     np.random.seed(seed)
        
    #     # Use a deep copy to ensure the original network is not modified
    #     new_net = copy.deepcopy(net)

    #     # Obtain nominal loads
    #     pd_nom = new_net.load['p_mw'].values.reshape(-1, 1)  # [MW]
    #     qd_nom = new_net.load['q_mvar'].values.reshape(-1, 1)  # [MVar]
    #     loads_nominal = np.vstack([pd_nom, qd_nom])

    #     n_loads = len(new_net.load)
        
    #     # Define load perturbation bounds (relative to nominal loads)
    #     lb_factor = 0.6 * np.ones(loads_nominal.shape[0])
    #     ub_factor = 1.0 * np.ones(loads_nominal.shape[0])
        
    #     # Generate load scaling factors
    #     X_factors = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, 1)
        
    #     # Calculate actual load values for each data point (MW/Mvar)
    #     X_unscaled_loads_mw = loads_nominal * X_factors
        
    #     # Separate active and reactive power components for adjustment
    #     pd_tot_mw_data = X_unscaled_loads_mw[:n_loads, :]
    #     qd_tot_mvar_data = X_unscaled_loads_mw[n_loads:, :]
        
    #     # Update the pandapower network object
    #     new_net.load['p_mw'] = pd_tot_mw_data.flatten()
    #     new_net.load['q_mvar'] = qd_tot_mvar_data.flatten()
        
    #     return new_net
    
    # # Determine case name based on the number of buses
    # case_name_map = {
    #     118: 'pglib_opf_case118_ieee.m',
    #     300: 'pglib_opf_case300_ieee.m',
    #     793: 'pglib_opf_case793_goc.m',
    #     1354: 'pglib_opf_case1354_pegase.m',
    #     2869: 'pglib_opf_case2869_pegase.m'
    # }

    # case_name = case_name_map.get(118)
    # if case_name is None:
    #     raise ValueError(f"Unsupported case number: {118}. "
    #                      f"Supported cases are: {list(case_name_map.keys())}")

    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root_dir = os.path.dirname(os.path.dirname(current_script_dir)) # Go up two levels from current script dir
    # base_dir = os.path.join(project_root_dir, 'pglib-opf') # Assuming pglib-opf is here

    # case_path = os.path.join(base_dir, case_name)

    # if not os.path.exists(case_path):
    #     raise FileNotFoundError(f"Case file not found at: {case_path}. "
    #                             "Please ensure 'pglib-opf' repository is cloned and correctly located.")

    # # Load the MATPOWER case
    # net = pc.from_mpc(case_path, casename_mpc_file=True)
    # base_ppc = pc.to_ppc(net, init='flat') # Get initial PPC for OPF solves
    
    # # Extract constants from the base case
    # Sbase = base_ppc['baseMVA']
    # n_bus = base_ppc['bus'].shape[0]
    # n_gens = base_ppc['gen'].shape[0]
    # n_loads = len(net.load)
    # n_branches = base_ppc['branch'].shape[0]
    # print(f"System details: {n_bus} buses, {n_gens} generators, {n_loads} loads.")

    # # Obtain nominal loads (from the loaded case's ppc)
    # pd_nom = np.array(net.load['p_mw']).reshape(-1, 1)              # [MW]
    # qd_nom = np.array(net.load['q_mvar']).reshape(-1, 1)            # [MVar]
    # loads_nominal = np.vstack([pd_nom, qd_nom]) 

    # # Define load perturbation bounds (relative to nominal loads)
    # lb_factor = 0.6 * np.ones(loads_nominal.shape[0])
    # ub_factor = 1.0 * np.ones(loads_nominal.shape[0])

    # # Generate load scaling factors
    # np.random.seed(42)  # Set seed for reproducibility
    # X_factors = ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, lb_factor, ub_factor, 1)

    # # Calculate actual load values for each data point (MW/Mvar)
    # X_unscaled_loads_mw = loads_nominal * X_factors
    # X_loads_pu = X_unscaled_loads_mw / Sbase
    # x = X_loads_pu
    # X_nn_input = x.T

    # # Separate active and reactive power components for adjustment
    # pd_tot_mw_data = X_unscaled_loads_mw[:n_loads, :]           # [MW]
    # qd_tot_mvar_data = X_unscaled_loads_mw[n_loads:, :]         # [MVar]

    # # --- Initialize Output Tensors -

    # # --- PYPOWER OPF Setup ---
    # ppopt = ppoption(OUT_ALL=0) # Suppress verbose output from PYPOWER
    
    # from pypower.api import makeYbus
    
    # # Re-calculate Ybus here, as it depends on baseMVA, bus, branch from base_ppc
    # Ybus, _, _ = makeYbus(base_ppc['baseMVA'], base_ppc['bus'], base_ppc['branch'])
    # Ybus_dense = Ybus.toarray()

    # # Get the internal PYPOWER case from the pandapower network
    # initial_ppc = base_ppc.copy()  # Make a copy of the initial PPC for modification
    # initial_net = copy.deepcopy(net)
    # net = initial_net
    
    
    
    

    # # ==================================
    # # SETUP - GETTING A TEST NETWORK
    # # ==================================
    # # Create a standard IEEE 118-bus network for the test
    # #net = pp.networks.case118()
    # ppc = pc.to_ppc(net, init='flat')  # Converts pandapower format to pypower format

    # # ==================================
    # # WARM-UP FOR PANDAMODELS
    # # ==================================
    # print("--- Starting Pandamodels Warm-up ---")
    # try:
    #     net, pm = runpm_opf(net, pm_solver='ipopt')
    #     print("Skipping pandamodels warm-up as run_pm_opf is commented out.")
    # except Exception as e:
    #     print(f"Pandamodels warm-up failed: {e}. Timed results may be inconsistent.")

    # # ==================================
    # # TIMING THE PYPOWER SOLVER
    # # ==================================
    # pypower_times = []
    # print(f"\n--- Timing {NUM_RUNS} runs with PyPower ---")
    # ppopt = ppoption(VERBOSITY=0)

    # for i in tqdm(range(NUM_RUNS)):
    #     # Sample a new load profile for this run
    #     modified_net = sample_kumaraswamy_loads(net, i)
        
    #     # Use a deep copy to ensure each run is independent
    #     current_ppc = pc.to_ppc(modified_net, init='flat')
        
    #     start_time = time.perf_counter()
    #     runopf(current_ppc, ppopt)
    #     end_time = time.perf_counter()
        
    #     pypower_times.append(end_time - start_time)

    # # ==================================
    # # TIMING THE PANDAMODELS SOLVER
    # # ==================================
    # pandamodels_times = []
    # print(f"\n--- Timing {NUM_RUNS} runs with PandaModels ---")

    # for i in tqdm(range(NUM_RUNS)):
    #     # Sample a new load profile for this run
    #     modified_net = sample_kumaraswamy_loads(net, i)
        
    #     # Use a deep copy to ensure each run is independent
    #     current_net = copy.deepcopy(modified_net)
        
    #     start_time = time.perf_counter()
    #     net, pm = runpm_opf(current_net, pm_solver='ipopt')
    #     end_time = time.perf_counter()
        
    #     # We are specifically interested in the pure solver time,
    #     # but since run_pm_opf is not available, we use wall time.
    #     pandamodels_times.append(end_time - start_time)

    # print("\n================================")
    # print("           RESULTS             ")
    # print("================================")
    # print(f"Average PyPower solve time over {NUM_RUNS} runs: {np.mean(pypower_times):.4f} seconds")
    # print(f"Average PandaModels solve time over {NUM_RUNS} runs: {np.mean(pandamodels_times):.4f} seconds")
    # print("================================")