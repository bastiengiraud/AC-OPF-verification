"""

# -------- how to run this script: ------------
cd scripts/comparison
conda activate ab-crown-old
python-jl validation.py # PyJulia call to run julia code form python
# --------------------------------------------

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


# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils', 'config', 'scripts/comparison', 'scripts/validation', 'scripts/validation/nn_inference']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))
    

from ac_opf.create_example_parameters import create_example_parameters
from types import SimpleNamespace

from validation_support import solve_ac_opf_and_collect_data, load_and_prepare_voltage_nn_for_inference, load_and_prepare_power_nn_for_inference, \
    voltage_nn_inference, power_nn_inference, compare_accuracy_with_mse, print_comparison_table, runpm_opf, \
        extract_solution_data_from_pandapower_net, convert_vr_vi_to_vm_va, power_nn_projection, voltage_nn_projection, \
        power_nn_warm_start, voltage_nn_warm_start, calculate_violations, print_violations_comparison_table


def create_config():
    parameters_dict = {
        'test_system': 118,
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
    
    import julia
    from julia import Main
    import os
    
    julia_project_path = "/home/bagir/Documents/1) Projects/2) AC verification/MinMax/scripts/comparison"
    Main.eval(f'using Pkg; Pkg.activate("{julia_project_path}")')
    print(f"Julia project activated: {julia_project_path}")

    # Now, try to load PandaModels and get its path within this Julia session
    try:
        Main.eval('using PandaModels')
        loaded_pandamodels_path = Main.eval('Base.pathof(PandaModels)')
        print(f"PandaModels.jl is loaded from: {loaded_pandamodels_path}")

        # You can also confirm Pkg.status from within the Python-initiated Julia session
        pkg_status_output = Main.eval('using Pkg; sprint(io -> Pkg.status(io=io))')
        print("\n--- Pkg.status('PandaModels') from Python-initiated Julia session ---")
        print(pkg_status_output)
        print("--------------------------------------------------------------------")

    except Exception as e:
        print(f"Error checking PandaModels.jl path in Julia: {e}")
        print("This might happen if PandaModels.jl is not accessible or installed in this Julia environment.")

    
    config = create_config()
    
    # define test system
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)
    net = simulation_parameters['pp_net']

    # solve some AC-OPFs for reference
    num_solves = 100

    print(f"Starting data generation for case {n_buses} with {num_solves} OPF solves...")
    solution_data_dict = solve_ac_opf_and_collect_data(n_buses, num_solves)

    # ----------- load the Pg Vm model without worst-case penalties ------------    
    nn_file_name_power_false             = 'checkpoint_118_50_False_pg_vm_final.pt'
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
    nn_file_name_power_true             = 'checkpoint_118_50_True_pg_vm_final.pt'
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
    nn_file_name_volt_false                    = 'checkpoint_118_50_False_vr_vi_final.pt'
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
    nn_file_name_volt_true                    = 'checkpoint_118_50_True_vr_vi_final.pt'
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
        num_samp=num_solves
    )
    
    if only_violations == False:
    
        # --------------- do the projections ----------------   
        pgvm_projected_results               = power_nn_projection(net, num_solves, solution_data_dict, pg_targets_false, vm_targets_false)
        mse_pgvm_projection                  = compare_accuracy_with_mse(solution_data_dict, pgvm_projected_results)

        # add results to the all_mse_results
        all_mse_results                      = [mse_pgvm_projection]
        all_model_names                      = ["Pg Vm Projection"]
        
        vrvi_projected_results               = voltage_nn_projection(net, num_solves, solution_data_dict, vr_targets_false, vi_targets_false)
        mse_vrvi_projection                  = compare_accuracy_with_mse(solution_data_dict, vrvi_projected_results)
        
        # add results to the all_mse_results
        all_mse_results.append(mse_vrvi_projection)
        all_model_names.append("Vr Vi Projection")
        
        # -------------- do the warm starts ----------------
        pgvm_ws_results                      = power_nn_warm_start(net, num_solves, solution_data_dict, pg_targets_false, vm_targets_false)
        mse_pgvm_ws                          = compare_accuracy_with_mse(solution_data_dict, pgvm_ws_results)
        
        # add results to the all_mse_results
        all_mse_results.append(mse_pgvm_ws)
        all_model_names.append("Pg Vm Warm Start")
        
        vrvi_ws_results                      = voltage_nn_warm_start(net, num_solves, solution_data_dict, vr_targets_false, vi_targets_false)
        mse_vrvi_ws                          = compare_accuracy_with_mse(solution_data_dict, vrvi_ws_results)
        
        # add results to the all_mse_results
        all_mse_results.append(mse_vrvi_ws)
        all_model_names.append("Vr Vi Warm Start")
        
        # -------------- compare the results ----------------
        print_comparison_table(
            mse_results_list=all_mse_results,
            model_names=all_model_names,
            table_title="Accuracy Comparison of Proxies"
        )
    
    
    




if __name__ == '__main__':
    # only compare statistical violations
    only_violations = 'True'
    
    main(only_violations)
    
    
    
    
    
    
    
    
    
# Run the feasibility check
# ppc = simulation_parameters['net_object']
# gen_mask_to_keep = np.array([True] * ppc['gen'].shape[0])
# is_feasible, costs, violation_details = check_feasibility_and_cost(voltage_nn_results, ppc, gen_mask_to_keep)

# print("\n--- Feasibility and Cost Results ---")
# for i in range(num_solves):
#     print(f"Sample {i+1}:")
#     print(f"  Feasible: {is_feasible[i]}")
#     print(f"  Cost: {costs[i]:.2f} USD" if not np.isnan(costs[i]) else "  Cost: N/A")
#     if violation_details[i] != "No violations":
#         print(f"  Violations: {violation_details[i]}")
#     print("-" * 20)
    
# is_feasible, costs, violation_details = check_feasibility_and_cost(power_nn_results, ppc, gen_mask_to_keep)

# print("\n--- Feasibility and Cost Results ---")
# for i in range(num_solves):
#     print(f"Sample {i+1}:")
#     print(f"  Feasible: {is_feasible[i]}")
#     print(f"  Cost: {costs[i]:.2f} USD" if not np.isnan(costs[i]) else "  Cost: N/A")
#     if violation_details[i] != "No violations":
#         print(f"  Violations: {violation_details[i]}")
#     print("-" * 20)