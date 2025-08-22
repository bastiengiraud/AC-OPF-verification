
import os
import sys
import numpy as np
import torch
import pandas as pd

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




# load simulation parameters
config = create_config()
simulation_parameters = create_example_parameters(config.test_system)
n_buses = config.test_system

# limits
sd_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float() 
sd_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float() / 100
vmag_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float()
vmag_min = torch.zeros_like(vmag_max) + 0.94
imag_max = torch.tensor(simulation_parameters['true_system']['I_max_pu']).float()
imag_max_tot = torch.cat((imag_max, imag_max), dim = 0)
map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32)
sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32) / 100
pg_max = (sg_max.T @ map_g)[:, :n_buses]
qg_max = (sg_max.T @ map_g)[:, n_buses:]
pg_min = torch.zeros_like(pg_max)
qg_min = torch.zeros_like(qg_max)

n_gens = simulation_parameters['general']['n_gbus']

pg_max_zero_mask = simulation_parameters['true_system']['Sg_max'][:n_gens] < 1e-9
gen_mask_to_keep = ~pg_max_zero_mask  # invert mask to keep desired generators

map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32)
sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32)
pg_max_gens = sg_max[:n_gens, :][gen_mask_to_keep] / 100 # (sg_max.T @ map_g)[:, :n_buses]



def verify_and_save_to_excel(n_buses, simulation_parameters, sd_min, sd_delta, pg_min, pg_max, qg_min, qg_max, vmag_min, vmag_max, imag_max_tot):
    """
    Performs verification on specified models, collects all violation metrics,
    and stores them in a single Excel file for easy analysis.

    Parameters:
    - n_buses (int): Number of buses for the model.
    - n_samp (int): Number of samples for the model.
    - All other parameters are assumed to be defined globally as in the user's original script.
    """
    
    # A dictionary to hold all the results, with model name as key
    all_verification_results = {}
    
    # The models to verify
    nn_to_verify = [
        f'checkpoint_{n_buses}_50_False_vr_vi_final.pt', 
        f'checkpoint_{n_buses}_50_True_vr_vi_final.pt',
        f'checkpoint_{n_buses}_50_False_pg_vm_final.pt', 
        f'checkpoint_{n_buses}_50_True_pg_vm_final.pt'
    ]

    # Shared verification setup
    optimize_bound_args = {
        'enable_alpha_crown': True,
        'enable_beta_crown': True
    }
    
    # Define input region
    epsilon = 1e-3
    x = sd_min.reshape(1, -1) + sd_delta.reshape(1, -1) / 2
    x = x.clone().detach().float()
    x_min = sd_min.reshape(1, -1).clone().detach().float() #+ epsilon
    x_max = (sd_min + sd_delta).reshape(1, -1).clone().detach().float() #- epsilon

    # Set up input specification
    ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
    image = BoundedTensor(x, ptb)
    
    print(f"Input bounds: x_min.min(): {x_min.min():.4f}, x_max.max(): {x_max.max():.4f}")
    
    # Loop through all specified models
    for name in nn_to_verify:
        print("##############################################################")
        print(f"### Verifying '{name}' ########### ")
        print("##############################################################")

        # Load the model weights based on the name convention
        if 'vr_vi' in name:
            model_type = 'vr_vi'
            nn_model = load_weights(model_type, name)
            # Create a BoundedModule for this specific model type
            # model = BoundedModule(nn_model, torch.empty_like(sd_min.reshape(1, -1)), optimize_bound_args)
            
            # This dictionary will hold the metrics for the current model
            metrics = {}

            # Check upper generator real power violation
            pg_up_model = BoundedModule(OutputWrapper(nn_model, 7), torch.empty_like(x), optimize_bound_args)
            _, ub = pg_up_model.compute_bounds(x=(image,), method="backward")
            upper_pg_violation = torch.relu(ub - pg_max)
            metrics['Pg Up Max Violation'] = upper_pg_violation.max().item()
            metrics['Pg Up Avg Violation'] = upper_pg_violation.mean().item()

            # Check lower generator real power violation
            pg_down_model = BoundedModule(OutputWrapper(nn_model, 8), torch.empty_like(x), optimize_bound_args)
            lb, _ = pg_down_model.compute_bounds(x=(image,), method="backward")
            lower_pg_violation = torch.relu(pg_min - lb)
            metrics['Pg Down Max Violation'] = lower_pg_violation.max().item()
            metrics['Pg Down Avg Violation'] = lower_pg_violation.mean().item()

            # Check upper generator reactive power violation
            qg_up_model = BoundedModule(OutputWrapper(nn_model, 9), torch.empty_like(x), optimize_bound_args)
            _, ub = qg_up_model.compute_bounds(x=(image,), method="backward")
            upper_qg_violation = torch.relu(ub - qg_max)
            metrics['Qg Up Max Violation'] = upper_qg_violation.max().item()
            metrics['Qg Up Avg Violation'] = upper_qg_violation.mean().item()

            # Check lower generator reactive power violation
            qg_down_model = BoundedModule(OutputWrapper(nn_model, 10), torch.empty_like(x), optimize_bound_args)
            lb, _ = qg_down_model.compute_bounds(x=(image,), method="backward")
            lower_qg_violation = torch.relu(qg_min - lb)
            metrics['Qg Down Max Violation'] = lower_qg_violation.max().item()
            metrics['Qg Down Avg Violation'] = lower_qg_violation.mean().item()
            
            # Check upper current magnitude violation
            imag_up_model = BoundedModule(OutputWrapper(nn_model, 4), torch.empty_like(x), optimize_bound_args)
            _, ub = imag_up_model.compute_bounds(x=(image,), method="backward")
            imag_up_violation = torch.relu(ub - imag_max_tot)
            metrics['Ibr Up Max Violation'] = imag_up_violation.max().item()
            metrics['Ibr Up Avg Violation'] = imag_up_violation.mean().item()
            
            # Check upper voltage magnitude violation
            vmag_up_model = BoundedModule(OutputWrapper(nn_model, 5), torch.empty_like(x), optimize_bound_args)
            _, ub = vmag_up_model.compute_bounds(x=(image,), method="backward")
            vmag_up_violation = torch.relu(ub - vmag_max)
            metrics['Vm Up Max Violation'] = vmag_up_violation.max().item()
            metrics['Vm Up Avg Violation'] = vmag_up_violation.mean().item()

            # Check lower voltage magnitude violation
            vmag_down_model = BoundedModule(OutputWrapper(nn_model, 6), torch.empty_like(x), optimize_bound_args)
            lb, _ = vmag_down_model.compute_bounds(x=(image,), method="backward")
            vmag_down_violation = torch.relu(vmag_min - lb)
            metrics['Vm Down Max Violation'] = vmag_down_violation.max().item()
            metrics['Vm Down Avg Violation'] = vmag_down_violation.mean().item()
            
            # Check current balance violation
            inj_real_model = BoundedModule(OutputWrapper(nn_model, 11), torch.empty_like(x), optimize_bound_args)
            inj_imag_model = BoundedModule(OutputWrapper(nn_model, 12), torch.empty_like(x), optimize_bound_args)
            
            lb_r, ub_r = inj_real_model.compute_bounds(x=(image,), method="backward")
            lb_i, ub_i = inj_imag_model.compute_bounds(x=(image,), method="backward")
            
            worst_case_inj_real = torch.max(lb_r**2, ub_r**2)
            worst_case_inj_imag = torch.max(lb_i**2, ub_i**2)
            inj_violation = torch.sqrt(worst_case_inj_real + worst_case_inj_imag)
            
            metrics['Ibal Max Violation'] = inj_violation.max().item()
            metrics['Ibal Avg Violation'] = inj_violation.mean().item()
            
        elif 'pg_vm' in name:
            model_type = 'pg_vm'
            nn_model = load_weights(model_type, name)
            model = BoundedModule(nn_model, torch.empty_like(x), optimize_bound_args)
            
            # This dictionary will hold the metrics for the current model
            metrics = {}

            lb, ub = model.compute_bounds(x=(image,), method="backward")
            
            # Split the output bounds based on the model's output features
            lb_pg = lb[:, :18]
            ub_pg = ub[:, :18]
            lb_vg = lb[:, 18:]
            ub_vg = ub[:, 18:]

            # Check lower generator real power violation
            lower_gen_violation = torch.relu(-lb_pg)
            metrics['Pg Down Max Violation'] = lower_gen_violation.max().item()
            metrics['Pg Down Avg Violation'] = lower_gen_violation.mean().item()
            
            # Check upper generator real power violation
            upper_gen_violation = torch.relu(ub_pg - pg_max_gens)
            metrics['Pg Up Max Violation'] = upper_gen_violation.max().item()
            metrics['Pg Up Avg Violation'] = upper_gen_violation.mean().item()
            
            # Check lower voltage violation
            vmag_down_violation = torch.relu(vmag_min - lb_vg)
            metrics['Vm Down Max Violation'] = vmag_down_violation.max().item()
            metrics['Vm Down Avg Violation'] = vmag_down_violation.mean().item()
            
            # Check upper voltage violation
            vmag_up_violation = torch.relu(ub_vg - vmag_max)
            metrics['Vm Up Max Violation'] = vmag_up_violation.max().item()
            metrics['Vm Up Avg Violation'] = vmag_up_violation.mean().item()

        # Store the metrics for the current model in the main results dictionary
        all_verification_results[name] = metrics

    # --- Step 1: Create a DataFrame from the collected results ---
    results_df = pd.DataFrame.from_dict(all_verification_results, orient='index').T

    # --- Step 2: Save the DataFrame to an Excel file ---
    try:
        # Create a descriptive filename
        filename = f"model_verification_results_{n_buses}_buses.xlsx"

        # Get the directory of the current script and join with the filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, filename)

        # Save the DataFrame to an Excel file
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

    # --- Step 3: Print the final DataFrame for a quick overview ---
    print("Final Verification Results Table:")
    print(results_df.to_string(float_format="{:.4f}".format))
    print("\n")

# Example usage (assuming all necessary variables are defined above this function call)
verify_and_save_to_excel(n_buses, simulation_parameters, sd_min, sd_delta, pg_min, pg_max, qg_min, qg_max, vmag_min, vmag_max, imag_max_tot)
