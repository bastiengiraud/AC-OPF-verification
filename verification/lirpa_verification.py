
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)
print(ROOT_DIR)

for subdir in ['MinMax/data', 'MinMax/models', 'verification/alpha-beta-CROWN/complete_verifier']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))
    
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from load_model_ac import load_weights
from ac_opf.create_example_parameters import create_example_parameters
from ac_opf.create_data import create_test_data, create_data
from types import SimpleNamespace
from neural_network.lightning_nn_crown import OutputWrapper


plt.style.use(['/mnt/c/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/0) DTU Admin/5) Templates/thesis.mplstyle'])
plt.rcParams['text.usetex'] = False

from matplotlib import font_manager

font_manager.fontManager.addfont('/mnt/c/Users/bagir/OneDrive - Danmarks Tekniske Universitet/Dokumenter/0) DTU Admin/5) Templates/palr45w.ttf')
plt.rcParams['font.family'] = 'Palatino' # Set the font globally
#plt.rcParams['font.family'] = 'sans-serif'


def create_config(n_buses):
    parameters_dict = {
        'test_system': n_buses,
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
    
    if n_buses == 14:
        parameters_dict['hidden_layer_size'] = 15
        parameters_dict['learning_rate'] = 2e-4
        parameters_dict['batch_size'] = 15
    elif n_buses == 57:
        parameters_dict['hidden_layer_size'] = 25
        parameters_dict['learning_rate'] = 5e-4
        parameters_dict['batch_size'] = 25
    elif n_buses == 118:
        parameters_dict['hidden_layer_size'] = 50
        parameters_dict['learning_rate'] = 10e-4
        parameters_dict['batch_size'] = 50
    elif n_buses == 300:
        parameters_dict['hidden_layer_size'] = 75
        parameters_dict['learning_rate'] = 10e-4
        parameters_dict['batch_size'] = 75
    elif n_buses == 793:
        parameters_dict['hidden_layer_size'] = 100
        parameters_dict['learning_rate'] = 20e-4
        parameters_dict['batch_size'] = 100
    
    
    config = SimpleNamespace(**parameters_dict)
    return config




# load simulation parameters
config = create_config(57)
simulation_parameters = create_example_parameters(config.test_system)
n_buses = config.test_system
n_gens = simulation_parameters['general']['n_gbus']






# limits
sd_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float() / 100
sd_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float() / 100
vmag_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float()
vmag_min = torch.tensor(simulation_parameters['true_system']['Volt_min'][0]).float()
imag_max = torch.tensor(simulation_parameters['true_system']['I_max_pu']).float()
imag_max_tot = torch.cat((imag_max, imag_max), dim = 0)
map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32)
sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32) / 100
pg_max = (sg_max.T @ map_g)[:, :n_buses]
qg_max = (sg_max.T @ map_g)[:, n_buses:]
pg_min = torch.tensor(simulation_parameters['true_system']['pg_min'], dtype=torch.float32) @ map_g[n_gens:, n_buses:] / 100
qg_min = torch.tensor(simulation_parameters['true_system']['qg_min'], dtype=torch.float32) @ map_g[n_gens:, :n_buses]  / 100


pg_max_zero_mask = simulation_parameters['true_system']['Sg_max'][:n_gens] < 1e-9
gen_mask_to_keep = ~pg_max_zero_mask  # invert mask to keep desired generators

map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32)
sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32)
pg_max_gens = (sg_max[:n_gens, :][gen_mask_to_keep.squeeze()] / 100).squeeze() # (sg_max.T @ map_g)[:, :n_buses]

pg_max_for_mask = (sg_max.T @ map_g)[:, :n_buses]
# qg_max = (sg_max.T @ map_g)[:, n_buses:]

gen_bus_mask = (pg_max_for_mask > 0)

num_gen_nn = len(pg_max_gens)


def verify_and_save_to_excel(store_excel, n_buses, deltas, sd_min, sd_delta, pg_min, pg_max, qg_min, qg_max, vmag_min, vmag_max, imag_max_tot):
    """
    Performs verification on specified models for a range of delta values,
    collects all violation metrics, expresses them as a percentage of the delta=0.0
    baseline, and stores them in an Excel file and plots them.

    Parameters:
    - store_excel (bool): If True, save the final results table to an Excel file.
    - n_buses (int): Number of buses for the model.
    - deltas (list): A list of delta values to iterate through for the input space.
    - All other parameters are assumed to be defined globally as in the user's original script.
    """
    
    # Dictionaries to hold all results
    all_raw_results = defaultdict(dict)
    all_percentage_results = defaultdict(dict)
    
    # The models to verify
    if n_buses == 14:
        nn_to_verify = [
            f'checkpoint_{n_buses}_15_False_vr_vi_final.pt', 
            f'checkpoint_{n_buses}_15_True_vr_vi_final.pt',
            f'checkpoint_{n_buses}_15_False_pg_vm_final.pt', 
            f'checkpoint_{n_buses}_15_True_pg_vm_final.pt'
        ]
    elif n_buses == 57:
        nn_to_verify = [
            f'checkpoint_{n_buses}_25_False_vr_vi_final.pt', 
            f'checkpoint_{n_buses}_25_True_vr_vi_final.pt',
            f'checkpoint_{n_buses}_25_False_pg_vm_final.pt', 
            f'checkpoint_{n_buses}_25_True_pg_vm_final.pt'
        ]
    elif n_buses == 118:
        nn_to_verify = [
            f'checkpoint_{n_buses}_50_False_vr_vi_final.pt', 
            f'checkpoint_{n_buses}_50_True_vr_vi_final.pt',
            f'checkpoint_{n_buses}_50_False_pg_vm_final.pt', 
            f'checkpoint_{n_buses}_50_True_pg_vm_final.pt'
        ]
    elif n_buses == 300:
        nn_to_verify = [
            f'checkpoint_{n_buses}_75_False_vr_vi_final.pt', 
            f'checkpoint_{n_buses}_75_True_vr_vi_final.pt',
            f'checkpoint_{n_buses}_75_False_pg_vm_final.pt', 
            f'checkpoint_{n_buses}_75_True_pg_vm_final.pt'
        ]
    elif n_buses == 793:
        nn_to_verify = [
            f'checkpoint_{n_buses}_100_False_vr_vi_final.pt', 
            f'checkpoint_{n_buses}_100_True_vr_vi_final.pt',
            f'checkpoint_{n_buses}_100_False_pg_vm_final.pt', 
            f'checkpoint_{n_buses}_100_True_pg_vm_final.pt'
        ]

    # Shared verification setup
    optimize_bound_args = {
        'enable_alpha_crown': True,
        'enable_beta_crown': True
    }
    
    p_max = sd_min + sd_delta

    # --- Main Loop: Iterate through each delta value ---
    for delta in deltas:
        print("\n" + "#" * 80)
        print(f"### Verifying with delta_factor = {delta:.2f} ######################")
        print("#" * 80)

        # Calculate the new lower and upper bounds based on the formula
        lower_factor = 0.6 + delta
        upper_factor = 1.0 - delta

        # Safeguard to ensure the upper bound is always greater than the lower bound.
        if lower_factor >= upper_factor:
            print("Warning: Delta factor is too large, input space is invalid. Using original bounds.")
            x_min = sd_min.reshape(1, -1).clone().detach().float()
            x_max = (sd_min + sd_delta).reshape(1, -1).clone().detach().float()
        else:
            # Calculate the new bounds using the adjusted factors
            x_min = lower_factor * p_max
            x_max = upper_factor * p_max
            
        # Reshape and convert to float tensors for CROWN
        x_min = x_min.reshape(1, -1).clone().detach().float()
        x_max = x_max.reshape(1, -1).clone().detach().float()
        input_dim = len(x_min[0])
        
        if torch.any(x_max < x_min):
            print("Warning: x_max is smaller than x_min in some dimensions. Correcting...")
            temp_min = torch.min(x_min, x_max)
            temp_max = torch.max(x_min, x_max)
            x_min = temp_min
            x_max = temp_max

        # Calculate the center of the new interval
        x = (x_min + x_max) / 2

        # Set up input specification
        ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
        image = BoundedTensor(x, ptb)
        
        # Loop through all specified models for the current delta
        for name in nn_to_verify:
            print("--------------------------------------------------")
            print(f"Verifying '{name}' for delta={delta:.2f}")
            print("--------------------------------------------------")
            
            # This dictionary will hold the raw metrics for the current model and delta
            metrics = {}

            # Load the model weights based on the name convention
            if 'vr_vi' in name:
                model_type = 'vr_vi'
                output_dim = 2*n_buses
                nn_model = load_weights(config, model_type, name, input_dim = input_dim, num_classes = output_dim)
                # print(nn_model)
                
                if 'False' in name:
                    print("Model trained without McCormick Envelopes.")
                
                    # get bounds on voltages and currents for mccormick False.
                    vr_model = BoundedModule(OutputWrapper(nn_model, 0), torch.empty_like(x), optimize_bound_args)
                    lb_vr, ub_vr = vr_model.compute_bounds(x=(image,), method="alpha-CROWN")
                    
                    if torch.any(ub_vr < lb_vr):
                        print("Warning: ub_vr is smaller than lb_vr in some dimensions. Correcting...")
                        temp_min = torch.min(lb_vr, ub_vr)
                        temp_max = torch.max(lb_vr, ub_vr)
                        lb_vr = temp_min
                        ub_vr = temp_max
                    
                    vi_model = BoundedModule(OutputWrapper(nn_model, 1), torch.empty_like(x), optimize_bound_args)
                    lb_vi, ub_vi = vi_model.compute_bounds(x=(image,), method="alpha-CROWN")
                    
                    # Check if upper bound is less than lower bound and correct it
                    if torch.any(ub_vi < lb_vi):
                        print("Warning: ub_vi was less than lb_vi. Bounds were swapped.")
                        temp_min = torch.min(lb_vi, ub_vi)
                        temp_max = torch.max(lb_vi, ub_vi)
                        lb_vi = temp_min
                        ub_vi = temp_max
                    
                    ir_model = BoundedModule(OutputWrapper(nn_model, 2), torch.empty_like(x), optimize_bound_args)
                    lb_ir, ub_ir = ir_model.compute_bounds(x=(image,), method="alpha-CROWN")
                    
                    if torch.any(ub_ir < lb_ir):
                        print("Warning: ub_ir was less than lb_ir. Bounds were swapped.")
                        temp_min = torch.min(lb_ir, ub_ir)
                        temp_max = torch.max(lb_ir, ub_ir)
                        lb_ir = temp_min
                        ub_ir = temp_max
                    
                    ii_model = BoundedModule(OutputWrapper(nn_model, 3), torch.empty_like(x), optimize_bound_args)
                    lb_ii, ub_ii = ii_model.compute_bounds(x=(image,), method="alpha-CROWN")
                    
                    if torch.any(ub_ii < lb_ii):
                        print("Warning: ub_ii was less than lb_ii. Bounds were swapped.")
                        temp_min = torch.min(lb_ii, ub_ii)
                        temp_max = torch.max(lb_ii, ub_ii)
                        lb_ii = temp_min
                        ub_ii = temp_max
                    
                    # update worst-case mccormick bounds
                    nn_model.pinj_upper_nn.update_mccormick_bounds(lb_vr, ub_vr, lb_vi, ub_vi, lb_ir, ub_ir, lb_ii, ub_ii)
                    nn_model.qinj_upper_nn.update_mccormick_bounds(lb_vr, ub_vr, lb_vi, ub_vi, lb_ir, ub_ir, lb_ii, ub_ii)
                    nn_model.pinj_lower_nn.update_mccormick_bounds(lb_vr, ub_vr, lb_vi, ub_vi, lb_ir, ub_ir, lb_ii, ub_ii)
                    nn_model.qinj_lower_nn.update_mccormick_bounds(lb_vr, ub_vr, lb_vi, ub_vi, lb_ir, ub_ir, lb_ii, ub_ii)         
                    

                # Check upper generator real power violation
                pg_up_model = BoundedModule(OutputWrapper(nn_model, 7), torch.empty_like(x), optimize_bound_args)
                _, ub = pg_up_model.compute_bounds(x=(image,), method="alpha-CROWN")
                upper_pg_violation = torch.relu(ub - pg_max)
                metrics['Pg Up Max Violation'] = upper_pg_violation[gen_bus_mask].max().item()
                metrics['Pg Up Avg Violation'] = upper_pg_violation[gen_bus_mask].mean().item()

                # Check lower generator real power violation
                pg_down_model = BoundedModule(OutputWrapper(nn_model, 8), torch.empty_like(x), optimize_bound_args)
                lb, _ = pg_down_model.compute_bounds(x=(image,), method="alpha-CROWN")
                lower_pg_violation = torch.relu(pg_min - lb)
                metrics['Pg Down Max Violation'] = lower_pg_violation[gen_bus_mask].max().item()
                metrics['Pg Down Avg Violation'] = lower_pg_violation[gen_bus_mask].mean().item()
                
                metrics['Pg tot Max Violation'] = torch.max(torch.cat([upper_pg_violation[gen_bus_mask], lower_pg_violation[gen_bus_mask]])).item()
                metrics['Pg tot Avg Violation'] = torch.mean(torch.cat([upper_pg_violation[gen_bus_mask], lower_pg_violation[gen_bus_mask]])).item()
                
                
                ######################################################
                # try verifying individually instead of vectorized ###
                #######################################################
                
                # # --- VERIFICATION FOR PG_UP AND PG_DOWN VIOLATIONS ---
                # # Create a bounded module for the pg_up output vector
                # pg_up_model = BoundedModule(OutputWrapper(nn_model, 7), torch.empty_like(x), optimize_bound_args)
                # pg_down_model = BoundedModule(OutputWrapper(nn_model, 8), torch.empty_like(x), optimize_bound_args)
                
                # upper_violations = []
                # lower_violations = []
                
                # # Now, loop through each element of the output vectors
                # for g_idx in range(n_buses):
                #     # Create a one-hot C matrix to select a single output
                #     C = torch.zeros((1, n_buses), device=x.device, dtype=x.dtype)
                #     C[0, g_idx] = 1.0
                    
                #     # Compute bounds on only one generator at a time
                #     _, ub = pg_up_model.compute_bounds(x=(image,), C=C, method="backward")
                #     lb, _ = pg_down_model.compute_bounds(x=(image,), C=C, method="backward")
                    
                #     upper_violation = torch.relu(ub - pg_max[g_idx])
                #     upper_violations.append(upper_violation.item())

                #     lower_violation = torch.relu(pg_min[g_idx] - lb)
                #     lower_violations.append(lower_violation.item())

                # # Combine violations into a single tensor/vector
                # upper_violations = torch.tensor(upper_violations)
                # lower_violations = torch.tensor(lower_violations)

                # metrics['Pg Up Max Violation'] = upper_violations.max().item()
                # metrics['Pg Up Avg Violation'] = upper_violations.mean().item()
                # metrics['Pg Down Max Violation'] = lower_violations.max().item()
                # metrics['Pg Down Avg Violation'] = lower_violations.mean().item()
                # metrics['Pg tot Max Violation'] = torch.max(upper_violations.max(), lower_violations.max()).item()
                # metrics['Pg tot Avg Violation'] = torch.mean(torch.cat([upper_violations, lower_violations])).item()

                ###########################################################
                ##########################################################

                # Check upper generator reactive power violation
                qg_up_model = BoundedModule(OutputWrapper(nn_model, 9), torch.empty_like(x), optimize_bound_args)
                _, ub = qg_up_model.compute_bounds(x=(image,), method="alpha-CROWN")
                upper_qg_violation = torch.relu(ub - qg_max)
                metrics['Qg Up Max Violation'] = upper_qg_violation[gen_bus_mask].max().item()
                metrics['Qg Up Avg Violation'] = upper_qg_violation[gen_bus_mask].mean().item()

                # Check lower generator reactive power violation
                qg_down_model = BoundedModule(OutputWrapper(nn_model, 10), torch.empty_like(x), optimize_bound_args)
                lb, _ = qg_down_model.compute_bounds(x=(image,), method="alpha-CROWN")
                lower_qg_violation = torch.relu(qg_min - lb)
                metrics['Qg Down Max Violation'] = lower_qg_violation[gen_bus_mask].max().item()
                metrics['Qg Down Avg Violation'] = lower_qg_violation[gen_bus_mask].mean().item()
                
                metrics['Qg tot Max Violation'] = torch.max(torch.cat([upper_qg_violation[gen_bus_mask], lower_qg_violation[gen_bus_mask]])).item()
                metrics['Qg tot Avg Violation'] = torch.mean(torch.cat([upper_qg_violation[gen_bus_mask], lower_qg_violation[gen_bus_mask]])).item()
                
                
                # Check upper current magnitude violation
                imag_up_model = BoundedModule(OutputWrapper(nn_model, 4), torch.empty_like(x), optimize_bound_args)
                _, ub = imag_up_model.compute_bounds(x=(image,), method="backward")
                imag_up_violation = torch.relu(ub - imag_max_tot)
                metrics['Ibr tot Max Violation'] = imag_up_violation.max().item()
                metrics['Ibr tot Avg Violation'] = imag_up_violation.mean().item()
                
                # Check upper voltage magnitude violation
                vmag_up_model = BoundedModule(OutputWrapper(nn_model, 5), torch.empty_like(x), optimize_bound_args)
                _, ub = vmag_up_model.compute_bounds(x=(image,), method="backward")
                vmag_up_violation = torch.relu(ub - vmag_max)
                metrics['Vm Up Max Violation'] = vmag_up_violation.max().item()
                metrics['Vm Up Avg Violation'] = vmag_up_violation.mean().item()
                
                metrics['Vmg Up Max Violation'] = vmag_up_violation[gen_bus_mask].max().item()
                metrics['Vmg Up Avg Violation'] = vmag_up_violation[gen_bus_mask].mean().item()
                

                # Check lower voltage magnitude violation
                vmag_down_model = BoundedModule(OutputWrapper(nn_model, 6), torch.empty_like(x), optimize_bound_args)
                lb, _ = vmag_down_model.compute_bounds(x=(image,), method="backward")
                vmag_down_violation = torch.relu(vmag_min - lb)
                metrics['Vm Down Max Violation'] = vmag_down_violation.max().item()
                metrics['Vm Down Avg Violation'] = vmag_down_violation.mean().item()
                
                metrics['Vmg Down Max Violation'] = vmag_down_violation[gen_bus_mask].max().item()
                metrics['Vmg Down Avg Violation'] = vmag_down_violation[gen_bus_mask].mean().item()
                
                metrics['Vm tot Max Violation'] = torch.max(torch.cat([vmag_up_violation, vmag_down_violation])).item()
                metrics['Vm tot Avg Violation'] = torch.mean(torch.cat([vmag_up_violation, vmag_down_violation])).item()
                
                metrics['Vmg tot Max Violation'] = torch.max(torch.cat([vmag_up_violation[gen_bus_mask], vmag_down_violation[gen_bus_mask]])).item()
                metrics['Vmg tot Avg Violation'] = torch.mean(torch.cat([vmag_up_violation[gen_bus_mask], vmag_down_violation[gen_bus_mask]])).item()
                
                # Check current balance violation
                inj_real_model = BoundedModule(OutputWrapper(nn_model, 11), torch.empty_like(x), optimize_bound_args)
                inj_imag_model = BoundedModule(OutputWrapper(nn_model, 12), torch.empty_like(x), optimize_bound_args)
                
                lb_r, ub_r = inj_real_model.compute_bounds(x=(image,), method="alpha-CROWN")
                lb_i, ub_i = inj_imag_model.compute_bounds(x=(image,), method="alpha-CROWN")
                
                # the way this is computed gives an extremely loose bound
                worst_case_inj_real = torch.max(lb_r**2, ub_r**2)
                worst_case_inj_imag = torch.max(lb_i**2, ub_i**2)
                inj_violation = torch.sqrt(worst_case_inj_real + worst_case_inj_imag)
                
                metrics['Ibal tot Max Violation'] = inj_violation.max().item()
                metrics['Ibal tot Avg Violation'] = inj_violation.mean().item()

                
            elif 'pg_vm' in name:
                model_type = 'pg_vm'
                output_dim = n_gens + num_gen_nn
                nn_model = load_weights(config, model_type, name, input_dim = input_dim, num_classes=output_dim)
                model = BoundedModule(nn_model, torch.empty_like(x), optimize_bound_args)
                
                # This dictionary will hold the metrics for the current model
                metrics = {}

                lb, ub = model.compute_bounds(x=(image,), method="alpha-CROWN")
                
                # Split the output bounds based on the model's output features
                lb_pg = lb[:, :num_gen_nn]
                ub_pg = ub[:, :num_gen_nn]
                lb_vg = lb[:, num_gen_nn:]
                ub_vg = ub[:, num_gen_nn:]

                # Check lower generator real power violation
                lower_gen_violation = torch.relu(-lb_pg)
                metrics['Pg Down Max Violation'] = lower_gen_violation.max().item()
                metrics['Pg Down Avg Violation'] = lower_gen_violation.mean().item()
                
                # Check upper generator real power violation
                upper_gen_violation = torch.relu(ub_pg - pg_max_gens)
                metrics['Pg Up Max Violation'] = upper_gen_violation.max().item()
                metrics['Pg Up Avg Violation'] = upper_gen_violation.mean().item()
                
                metrics['Pg tot Max Violation'] = torch.max(torch.cat([upper_gen_violation, lower_gen_violation])).item()
                metrics['Pg tot Avg Violation'] = torch.mean(torch.cat([upper_gen_violation, lower_gen_violation])).item()
                
                # Check lower voltage violation
                vmag_down_violation = torch.relu(vmag_min - lb_vg)
                metrics['Vmg Down Max Violation'] = vmag_down_violation.max().item()
                metrics['Vmg Down Avg Violation'] = vmag_down_violation.mean().item()
                
                # Check upper voltage violation
                vmag_up_violation = torch.relu(ub_vg - vmag_max)
                metrics['Vmg Up Max Violation'] = vmag_up_violation.max().item()
                metrics['Vmg Up Avg Violation'] = vmag_up_violation.mean().item()
                
                
                metrics['Vmg tot Max Violation'] = torch.max(torch.cat([vmag_up_violation, vmag_down_violation])).item()
                metrics['Vmg tot Avg Violation'] = torch.mean(torch.cat([vmag_up_violation, vmag_down_violation])).item()
                    
            # Store the raw metrics for the current model in the main results dictionary
            all_raw_results[name][delta] = metrics

    # --- Step 1: Calculate percentage of baseline ---
    baseline_delta = deltas[0]
    baseline_results = {name: all_raw_results[name][baseline_delta] for name in nn_to_verify}

    for name in nn_to_verify:
        for delta in deltas:
            percentage_metrics = {}
            for metric_name, raw_value in all_raw_results[name][delta].items():
                baseline_value = baseline_results[name].get(metric_name)
                if baseline_value is not None and baseline_value > 0:
                    percentage_metrics[metric_name] = (raw_value / baseline_value) * 100
                else:
                    percentage_metrics[metric_name] = 0.0
            all_percentage_results[name][delta] = percentage_metrics
            
    # Get the list of violation metrics from the first entry
    metric_names = list(all_percentage_results[nn_to_verify[0]][deltas[0]].keys())
    
    if store_excel:
        # Create a list of dictionaries for the raw data
        raw_rows_list = []
        for metric_name in metric_names:
            row_dict = {'Metric': metric_name}
            for name in nn_to_verify:
                for delta in deltas:
                    col_name = f"{name}_{delta:.2f}"
                    # Use the raw results dictionary
                    row_dict[col_name] = all_raw_results[name][delta].get(metric_name)
            raw_rows_list.append(row_dict)
        
        raw_results_df = pd.DataFrame(raw_rows_list).set_index('Metric')

        try:
            filename = f"model_verification_results_{n_buses}_buses_raw.xlsx"
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, filename)
            raw_results_df.to_excel(output_path, index=True)
            print("\n" + "=" * 60)
            print(f"✅ Raw verification results successfully saved to:\n   {output_path}")
            print("=" * 60 + "\n")
        except ImportError:
            print("Error: The 'openpyxl' library is required to save to Excel.")
            print("Please install it with: pip install openpyxl")
        except IOError as e:
            print(f"Error writing to Excel file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the Excel file: {e}")

    # --- Step 2: Create a DataFrame from the percentage results ---
    # Create a list of dictionaries, where each dictionary is a row in the final DataFrame
    rows_list = []
    
    # Get the list of violation metrics from the first entry
    metric_names = list(all_percentage_results[nn_to_verify[0]][deltas[0]].keys())
    
    for metric_name in metric_names:
        row_dict = {'Metric': metric_name}
        for name in nn_to_verify:
            for delta in deltas:
                col_name = f"{name}_{delta:.2f}"
                row_dict[col_name] = all_percentage_results[name][delta].get(metric_name)
        rows_list.append(row_dict)

    results_df = pd.DataFrame(rows_list).set_index('Metric')

    # --- Step 3: Save the DataFrame to an Excel file ---
    if store_excel:
        try:
            filename = f"model_verification_results_{n_buses}_buses.xlsx"
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, filename)
            results_df.to_excel(output_path, index=True)
            print("\n" + "=" * 60)
            print(f"✅ All verification results successfully saved to:\n   {output_path}")
            print("=" * 60 + "\n")

        except ImportError:
            print("Error: The 'openpyxl' library is required to save to Excel.")
            print("Please install it with: pip install openpyxl")
        except IOError as e:
            print(f"Error writing to Excel file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the Excel file: {e}")

    # --- Step 4: Print the final DataFrame for a quick overview ---
    print("Final Verification Results Table (as percentage of baseline):")
    print(results_df.to_string(float_format="{:.4f}".format))
    print("\n")

    # --- Step 5: Generate and save plots ---
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # Filter for only Max violation metrics
    max_metric_names = [metric for metric in metric_names if 'tot Max' in metric]

    # Create a new DataFrame optimized for plotting
    for model_name in nn_to_verify:
        plt.figure(figsize=(12, 4))
        
        # Select data for the current model
        model_data = results_df[[col for col in results_df.columns if col.startswith(model_name)]]
        model_data.columns = deltas
        
        # Plot each metric against delta
        for metric_name in max_metric_names:
            if metric_name in model_data.index:
                plot_series = model_data.loc[metric_name, :]
                # Only plot if the series contains at least one non-NaN value
                if not plot_series.isnull().all():
                    plt.plot(deltas, plot_series, marker='o', label=metric_name)

        # plt.title(f'Violation Reduction for {model_name}', fontsize=16)
        plt.xlabel(r'$\delta$ Factor', fontsize=12)
        plt.ylabel(r'Guarantee $\nu$ (% of worst-case)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(deltas, [f'{d:.2f}' for d in deltas])
        plt.legend(loc='upper right') # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plot_dir, f'{model_name.replace(".pt", "")}_violations.png')
        plt.savefig(plot_path)
        plt.close() # Close the figure to free memory
        print(f"Plot saved for '{model_name}' to {plot_path}")



deltas = [0.0, 0.04, 0.08, 0.12, 0.16, 0.199] # , 0.0, 0.04, 0.08, 0.12, 0.16, 0.199
verify_and_save_to_excel(store_excel=True, n_buses=n_buses, deltas=deltas, 
                             sd_min=sd_min, sd_delta=sd_delta, 
                             pg_min=pg_min, pg_max=pg_max, 
                             qg_min=qg_min, qg_max=qg_max, 
                             vmag_min=vmag_min, vmag_max=vmag_max, 
                             imag_max_tot=imag_max_tot)












# def verify_and_save_to_excel(store_excel, n_buses, simulation_parameters, sd_min, sd_delta, pg_min, pg_max, qg_min, qg_max, vmag_min, vmag_max, imag_max_tot):
#     """
#     Performs verification on specified models, collects all violation metrics,
#     and stores them in a single Excel file for easy analysis.

#     Parameters:
#     - n_buses (int): Number of buses for the model.
#     - n_samp (int): Number of samples for the model.
#     - All other parameters are assumed to be defined globally as in the user's original script.
#     """
    
#     # A dictionary to hold all the results, with model name as key
#     all_verification_results = {}
    
#     # The models to verify
#     nn_to_verify = [
#         f'checkpoint_{n_buses}_50_False_vr_vi_final.pt', 
#         f'checkpoint_{n_buses}_50_True_vr_vi_final.pt',
#         f'checkpoint_{n_buses}_50_False_pg_vm_final.pt', 
#         f'checkpoint_{n_buses}_50_True_pg_vm_final.pt'
#     ]

#     # Shared verification setup
#     optimize_bound_args = {
#         'enable_alpha_crown': True,
#         'enable_beta_crown': True
#     }
    
#     # Define input region
#     delta_factor = 0.0  # Example: 10% reduction from both sides
#     p_max = sd_min + sd_delta

#     # Calculate the new lower and upper bounds based on the formula
#     lower_factor = 0.6 + delta_factor
#     upper_factor = 1.0 - delta_factor

#     # Safeguard to ensure the upper bound is always greater than the lower bound.
#     if lower_factor >= upper_factor:
#         print("Warning: Delta factor is too large, input space is invalid. Using original bounds.")
#         x_min = sd_min.reshape(1, -1).clone().detach().float()
#         x_max = (sd_min + sd_delta).reshape(1, -1).clone().detach().float()
#     else:
#         # Calculate the new bounds using the adjusted factors
#         x_min = lower_factor * p_max
#         x_max = upper_factor * p_max
        
#     # Reshape and convert to float tensors for CROWN
#     x_min = x_min.reshape(1, -1).clone().detach().float()
#     x_max = x_max.reshape(1, -1).clone().detach().float()

#     # Calculate the center of the new interval
#     x = (x_min + x_max) / 2

#     # Set up input specification
#     ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
#     image = BoundedTensor(x, ptb)
    
#     print(f"Input bounds: x_min.min(): {x_min.min():.4f}, x_max.max(): {x_max.max():.4f}")
#     print(f"Input bounds: x_min.avg(): {x_min.mean():.4f}, x_max.avg(): {x_max.mean():.4f}")
    
#     # Loop through all specified models
#     for name in nn_to_verify:
#         print("##############################################################")
#         print(f"### Verifying '{name}' ########### ")
#         print("##############################################################")

#         # Load the model weights based on the name convention
#         if 'vr_vi' in name:
#             model_type = 'vr_vi'
#             nn_model = load_weights(model_type, name)
#             # Create a BoundedModule for this specific model type
#             # model = BoundedModule(nn_model, torch.empty_like(sd_min.reshape(1, -1)), optimize_bound_args)
            
#             # This dictionary will hold the metrics for the current model
#             metrics = {}
            
#             # # Check upper generator real power violation
#             # vr = BoundedModule(OutputWrapper(nn_model, 0), torch.empty_like(x), optimize_bound_args)
#             # lb, ub = vr.compute_bounds(x=(image,), method="backward")
#             # print(lb, ub)
#             # vi = BoundedModule(OutputWrapper(nn_model, 0), torch.empty_like(x), optimize_bound_args)
#             # lb, ub = vi.compute_bounds(x=(image,), method="backward")
#             # print(lb, ub)
            


#             # Check upper generator real power violation
#             pg_up_model = BoundedModule(OutputWrapper(nn_model, 7), torch.empty_like(x), optimize_bound_args)
#             _, ub = pg_up_model.compute_bounds(x=(image,), method="backward")
#             upper_pg_violation = torch.relu(ub - pg_max)
#             metrics['Pg Up Max Violation'] = upper_pg_violation.max().item()
#             metrics['Pg Up Avg Violation'] = upper_pg_violation.mean().item()

#             # Check lower generator real power violation
#             pg_down_model = BoundedModule(OutputWrapper(nn_model, 8), torch.empty_like(x), optimize_bound_args)
#             lb, _ = pg_down_model.compute_bounds(x=(image,), method="backward")
#             lower_pg_violation = torch.relu(pg_min - lb)
#             metrics['Pg Down Max Violation'] = lower_pg_violation.max().item()
#             metrics['Pg Down Avg Violation'] = lower_pg_violation.mean().item()

#             # Check upper generator reactive power violation
#             qg_up_model = BoundedModule(OutputWrapper(nn_model, 9), torch.empty_like(x), optimize_bound_args)
#             _, ub = qg_up_model.compute_bounds(x=(image,), method="backward")
#             upper_qg_violation = torch.relu(ub - qg_max)
#             metrics['Qg Up Max Violation'] = upper_qg_violation.max().item()
#             metrics['Qg Up Avg Violation'] = upper_qg_violation.mean().item()

#             # Check lower generator reactive power violation
#             qg_down_model = BoundedModule(OutputWrapper(nn_model, 10), torch.empty_like(x), optimize_bound_args)
#             lb, _ = qg_down_model.compute_bounds(x=(image,), method="backward")
#             lower_qg_violation = torch.relu(qg_min - lb)
#             metrics['Qg Down Max Violation'] = lower_qg_violation.max().item()
#             metrics['Qg Down Avg Violation'] = lower_qg_violation.mean().item()
            
#             # Check upper current magnitude violation
#             imag_up_model = BoundedModule(OutputWrapper(nn_model, 4), torch.empty_like(x), optimize_bound_args)
#             _, ub = imag_up_model.compute_bounds(x=(image,), method="backward")
#             imag_up_violation = torch.relu(ub - imag_max_tot)
#             metrics['Ibr Up Max Violation'] = imag_up_violation.max().item()
#             metrics['Ibr Up Avg Violation'] = imag_up_violation.mean().item()
            
#             # Check upper voltage magnitude violation
#             vmag_up_model = BoundedModule(OutputWrapper(nn_model, 5), torch.empty_like(x), optimize_bound_args)
#             _, ub = vmag_up_model.compute_bounds(x=(image,), method="backward")
#             vmag_up_violation = torch.relu(ub - vmag_max)
#             metrics['Vm Up Max Violation'] = vmag_up_violation.max().item()
#             metrics['Vm Up Avg Violation'] = vmag_up_violation.mean().item()

#             # Check lower voltage magnitude violation
#             vmag_down_model = BoundedModule(OutputWrapper(nn_model, 6), torch.empty_like(x), optimize_bound_args)
#             lb, ub = vmag_down_model.compute_bounds(x=(image,), method="backward")
#             vmag_down_violation = torch.relu(vmag_min - lb)
#             metrics['Vm Down Max Violation'] = vmag_down_violation.max().item()
#             metrics['Vm Down Avg Violation'] = vmag_down_violation.mean().item()
            
#             # Check current balance violation
#             inj_real_model = BoundedModule(OutputWrapper(nn_model, 11), torch.empty_like(x), optimize_bound_args)
#             inj_imag_model = BoundedModule(OutputWrapper(nn_model, 12), torch.empty_like(x), optimize_bound_args)
            
#             lb_r, ub_r = inj_real_model.compute_bounds(x=(image,), method="backward")
#             lb_i, ub_i = inj_imag_model.compute_bounds(x=(image,), method="backward")
            
#             worst_case_inj_real = torch.max(lb_r**2, ub_r**2)
#             worst_case_inj_imag = torch.max(lb_i**2, ub_i**2)
#             inj_violation = torch.sqrt(worst_case_inj_real + worst_case_inj_imag)
            
#             metrics['Ibal Max Violation'] = inj_violation.max().item()
#             metrics['Ibal Avg Violation'] = inj_violation.mean().item()
            
#         elif 'pg_vm' in name:
#             model_type = 'pg_vm'
#             nn_model = load_weights(model_type, name)
#             model = BoundedModule(nn_model, torch.empty_like(x), optimize_bound_args)
            
#             # This dictionary will hold the metrics for the current model
#             metrics = {}

#             lb, ub = model.compute_bounds(x=(image,), method="backward")
            
#             # Split the output bounds based on the model's output features
#             lb_pg = lb[:, :18]
#             ub_pg = ub[:, :18]
#             lb_vg = lb[:, 18:]
#             ub_vg = ub[:, 18:]

#             # Check lower generator real power violation
#             lower_gen_violation = torch.relu(-lb_pg)
#             metrics['Pg Down Max Violation'] = lower_gen_violation.max().item()
#             metrics['Pg Down Avg Violation'] = lower_gen_violation.mean().item()
            
#             # Check upper generator real power violation
#             upper_gen_violation = torch.relu(ub_pg - pg_max_gens)
#             metrics['Pg Up Max Violation'] = upper_gen_violation.max().item()
#             metrics['Pg Up Avg Violation'] = upper_gen_violation.mean().item()
            
#             # Check lower voltage violation
#             vmag_down_violation = torch.relu(vmag_min - lb_vg)
#             metrics['Vm Down Max Violation'] = vmag_down_violation.max().item()
#             metrics['Vm Down Avg Violation'] = vmag_down_violation.mean().item()
            
#             # Check upper voltage violation
#             vmag_up_violation = torch.relu(ub_vg - vmag_max)
#             metrics['Vm Up Max Violation'] = vmag_up_violation.max().item()
#             metrics['Vm Up Avg Violation'] = vmag_up_violation.mean().item()

#         # Store the metrics for the current model in the main results dictionary
#         all_verification_results[name] = metrics

#     # --- Step 1: Create a DataFrame from the collected results ---
#     results_df = pd.DataFrame.from_dict(all_verification_results, orient='index').T

#     # --- Step 2: Save the DataFrame to an Excel file ---
#     if store_excel:
#         try:
#             # Create a descriptive filename
#             filename = f"model_verification_results_{n_buses}_buses.xlsx"

#             # Get the directory of the current script and join with the filename
#             script_dir = os.path.dirname(os.path.abspath(__file__))
#             output_path = os.path.join(script_dir, filename)

#             # Save the DataFrame to an Excel file
#             results_df.to_excel(output_path, index=True)
#             print("\n" + "=" * 60)
#             print(f"✅ All verification results successfully saved to:\n   {output_path}")
#             print("=" * 60 + "\n")

#         except ImportError:
#             print("Error: The 'openpyxl' library is required to save to Excel.")
#             print("Please install it with: pip install openpyxl")
#         except IOError as e:
#             print(f"Error writing to Excel file: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred while saving the Excel file: {e}")

#     # --- Step 3: Print the final DataFrame for a quick overview ---
#     print("Final Verification Results Table:")
#     print(results_df.to_string(float_format="{:.4f}".format))
#     print("\n")






# Example usage (assuming all necessary variables are defined above this function call)
#verify_and_save_to_excel(False, n_buses, simulation_parameters, sd_min, sd_delta, pg_min, pg_max, qg_min, qg_max, vmag_min, vmag_max, imag_max_tot)