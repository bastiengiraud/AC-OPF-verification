import os
import sys
import wandb
import pandas as pd

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from ac_opf.create_example_parameters import create_example_parameters
from types import SimpleNamespace
from typing import Dict, Any

def load_params_from_excel(file_path: str, n_buses: int, nn_type: str, algo: bool) -> Dict[str, Any]:
    """
    Loads configuration parameters from a specified Excel file.
    
    Args:
        file_path (str): The path to the Excel file.
        n_buses (int): The number of buses, used to select the correct worksheet.
        nn_type (str): The type of neural network ('pg_vm' or 'vr_vi').
        algo (bool): A boolean flag to determine the column ('_True' or '_False').

    Returns:
        Dict[str, Any]: A dictionary of parameters loaded from the Excel file.
    """
    # Read the correct worksheet directly into a DataFrame
    sheet_name = str(n_buses)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Construct the column name based on nn_type and algo
    column_name = f'{nn_type}_{algo}'
    
    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in sheet '{sheet_name}'.")
    
    # Get the parameter names from the first column and values from the correct column
    # Using .iloc for the first column is a robust way to get parameter names
    param_names = df.iloc[:, 0].tolist()
    
    # Directly get the values from the specified column
    param_values = df[column_name].tolist()
    
    # Combine them into a dictionary
    params_dict = dict(zip(param_names, param_values))
    return params_dict
            



def create_config(nn_type: str, algo: bool) -> SimpleNamespace:
    """
    Creates a configuration object, now with the option to load from Excel.
    """
    # Define a default set of parameters to be overwritten by Excel
    parameters_dict = {
        'sweep': False,
        'test_system': 118,
        'hidden_layer_size': 50,
        'n_hidden_layers': 3,
        'epochs': 1000,
        'batch_size': 50,
        'learning_rate': 1e-3, # 57: 5e-4, 118: 1e-3
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'pg_viol_weight': 0.0,
        'qg_viol_weight': 0.0,
        'vm_viol_weight': 0.0,
        'line_viol_weight': 1e0,
        'crit_weight': 0.0,
        'crit_volt_weight': 0.0,
        'crit_pg_weight': 0.0,
        'PF_weight': 1e0,
        'kcl_weight': 0.0,
        'LPF_weight': 0.0,
        'N_enrich': 50,
        'Algo': algo,
        'Enrich': False,
        'abc_method': 'backward',
    }

    # Use the Excel file path and test system from your setup
    dir_name = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    excel_file_path = os.path.join(dir_name, 'sweep_weights.xlsx')
    n_buses = parameters_dict['test_system'] # Or whatever your test_system is
    
    if n_buses == 57:
        parameters_dict['hidden_layer_size'] = 25
        parameters_dict['learning_rate'] = 5e-4
        parameters_dict['batch_size'] = 25
    elif n_buses == 118:
        parameters_dict['hidden_layer_size'] = 50
        parameters_dict['learning_rate'] = 10e-4
        parameters_dict['batch_size'] = 50
        
    print("hidden layer size: ", parameters_dict['hidden_layer_size'], "learning rate: ", parameters_dict['learning_rate'], "batch size: ", parameters_dict['batch_size'])
        
    
    # Load parameters from Excel and update the dictionary
    excel_params = load_params_from_excel(excel_file_path, n_buses, nn_type, algo)
    parameters_dict.update(excel_params)
    
    # Ensure the 'Algo' parameter from the function's argument is used
    parameters_dict['Algo'] = algo

    config = SimpleNamespace(**parameters_dict)
    return config





def main(nn_type, algo):

    config = create_config(nn_type, algo)
    config.Algo = algo
    
    # define test system
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)
    nn_config = nn_type # simulation_parameters['nn_output']
    
    with wandb.init(
        project="ac_verif_nn_training",
        group=f"sys_{config.test_system}_{nn_type}_Algo_{config.Algo}",
    ) as run:
        run.name=f"sys_{config.test_system}_{nn_type}_Algo_{config.Algo}_final"
    
        # define training paradigm
        if nn_config == 'pg_vm':
            from nn_training_ac_crown_pg_vm import train
            print("We're training the Power NN!")
        elif nn_config == 'pg_qg':
            from nn_training_ac_crown_pg_qg import train
        elif nn_config == 'vr_vi':
            from nn_training_ac_crown_vr_vi import train
            print("We're training the Voltage NN!")
        else:
            print("Training paradigm not recognized.")
        
        # start training
        train(config=config)



if __name__ == '__main__':
    
    nn_type_opts = ['vr_vi', 'pg_vm']
    algo_opts = [True, False]
    
    for nn_type in nn_type_opts:
        for algo in algo_opts:
            main(nn_type, algo)















# def create_config(nn_type, algo = False):
    
#     if nn_type == 'pg_vm':
#         if algo == False:
#             # -------- pg vm ----------
#             parameters_dict = {
#                 'sweep': False,
#                 'test_system': 57,
#                 'hidden_layer_size': 25,
#                 'n_hidden_layers': 3,
#                 'epochs': 500,
#                 'batch_size': 25,
#                 'learning_rate': 0.005929597604769215,
#                 'lr_decay': 0.97,
#                 'dataset_split_seed': 10,
#                 'pytorch_init_seed': 3,
#                 'pg_viol_weight': 81.59152776750348,
#                 'qg_viol_weight': 0,
#                 'vm_viol_weight': 7.834874053808168,
#                 'line_viol_weight': 1e0,
#                 'crit_volt_weight': 2013.1617129364492, # 1e5,
#                 'crit_pg_weight': 80.83001028207997, # 1e5,
#                 'PF_weight': 1e0,
#                 'LPF_weight': 92.59172173360572,
#                 'N_enrich': 50,
#                 'Algo': False, # if True, add worst-case violation CROWN bounds during training
#                 'Enrich': False,
#                 'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
#             }
#         elif algo == True:
#             # -------- pg vm ----------
#             parameters_dict = {
#                 'sweep': False,
#                 'test_system': 57,
#                 'hidden_layer_size': 25,
#                 'n_hidden_layers': 3,
#                 'epochs': 500,
#                 'batch_size': 50,
#                 'learning_rate': 0.004367648975106109,
#                 'lr_decay': 0.97,
#                 'dataset_split_seed': 10,
#                 'pytorch_init_seed': 3,
#                 'pg_viol_weight': 168.43402279441085,
#                 'qg_viol_weight': 0,
#                 'vm_viol_weight': 18.39561470189764,
#                 'line_viol_weight': 1e0,
#                 'crit_volt_weight': 15.903807471106374, # 1e5,
#                 'crit_pg_weight': 2890.5681419476955, # 1e5,
#                 'PF_weight': 1e0,
#                 'LPF_weight': 247.27493462127865,
#                 'N_enrich': 50,
#                 'Algo': True, # if True, add worst-case violation CROWN bounds during training
#                 'Enrich': False,
#                 'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
#             }
    
#     elif nn_type == 'vr_vi':
#         if algo == False:
#             # ------- vr vi ------------
#             parameters_dict = {
#                 'sweep': False,
#                 'test_system': 57,
#                 'hidden_layer_size': 25,
#                 'n_hidden_layers': 3,
#                 'epochs': 500,
#                 'batch_size': 25,
#                 'learning_rate': 0.0010874165829638284,
#                 'lr_decay': 0.97,
#                 'dataset_split_seed': 10,
#                 'pytorch_init_seed': 3,
#                 'pg_viol_weight': 6.972774487659531,
#                 'vm_viol_weight': 42.98384563313474,
#                 'line_viol_weight': 2.789589883535428,
#                 'crit_weight': 9069, # 1e5,
#                 'kcl_weight': 115, # weight for KCL violation
#                 'LPF_weight': 3.686471151222091,
#                 'N_enrich': 50,
#                 'Algo': False, # if True, add worst-case violation CROWN bounds during training
#                 'Enrich': False,
#                 'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
#             }
#         elif algo == True:
#             # ------- vr vi ------------
#             parameters_dict = {
#                 'sweep': False,
#                 'test_system': 57,
#                 'hidden_layer_size': 25,
#                 'n_hidden_layers': 3,
#                 'epochs': 500,
#                 'batch_size': 100,
#                 'learning_rate': 0.005055573805708793,
#                 'lr_decay': 0.97,
#                 'dataset_split_seed': 10,
#                 'pytorch_init_seed': 3,
#                 'pg_viol_weight': 1.1590475827878204,
#                 'vm_viol_weight': 3.875842235619783,
#                 'line_viol_weight': 72.87773745654778,
#                 'crit_weight': 5867, # 1e5,
#                 'kcl_weight': 630.394072244584, # weight for KCL violation
#                 'LPF_weight': 2.387856417356738,
#                 'N_enrich': 50,
#                 'Algo': True, # if True, add worst-case violation CROWN bounds during training
#                 'Enrich': False,
#                 'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
#             }
    
#     config = SimpleNamespace(**parameters_dict)
#     return config