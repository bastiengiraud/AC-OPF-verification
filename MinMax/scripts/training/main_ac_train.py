import os
import sys
import wandb

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from ac_opf.create_example_parameters import create_example_parameters
from types import SimpleNamespace


def create_config(nn_type, algo = False):
    
    if nn_type == 'pg_vm':
        if algo == False:
            # -------- pg vm ----------
            parameters_dict = {
                'sweep': False,
                'test_system': 118,
                'hidden_layer_size': 50,
                'n_hidden_layers': 3,
                'epochs': 1000,
                'batch_size': 25,
                'learning_rate': 0.005929597604769215,
                'lr_decay': 0.97,
                'dataset_split_seed': 10,
                'pytorch_init_seed': 3,
                'pg_viol_weight': 81.59152776750348,
                'qg_viol_weight': 0,
                'vm_viol_weight': 7.834874053808168,
                'line_viol_weight': 1e0,
                'crit_volt_weight': 2013.1617129364492, # 1e5,
                'crit_pg_weight': 80.83001028207997, # 1e5,
                'PF_weight': 1e0,
                'LPF_weight': 92.59172173360572,
                'N_enrich': 50,
                'Algo': False, # if True, add worst-case violation CROWN bounds during training
                'Enrich': False,
                'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
            }
        elif algo == True:
            # -------- pg vm ----------
            parameters_dict = {
                'sweep': False,
                'test_system': 118,
                'hidden_layer_size': 50,
                'n_hidden_layers': 3,
                'epochs': 1000,
                'batch_size': 50,
                'learning_rate': 0.004367648975106109,
                'lr_decay': 0.97,
                'dataset_split_seed': 10,
                'pytorch_init_seed': 3,
                'pg_viol_weight': 168.43402279441085,
                'qg_viol_weight': 0,
                'vm_viol_weight': 18.39561470189764,
                'line_viol_weight': 1e0,
                'crit_volt_weight': 15.903807471106374, # 1e5,
                'crit_pg_weight': 2890.5681419476955, # 1e5,
                'PF_weight': 1e0,
                'LPF_weight': 247.27493462127865,
                'N_enrich': 50,
                'Algo': True, # if True, add worst-case violation CROWN bounds during training
                'Enrich': False,
                'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
            }
    
    elif nn_type == 'vr_vi':
        if algo == False:
            # ------- vr vi ------------
            parameters_dict = {
                'sweep': False,
                'test_system': 118,
                'hidden_layer_size': 50,
                'n_hidden_layers': 3,
                'epochs': 1000,
                'batch_size': 25,
                'learning_rate': 0.0010874165829638284,
                'lr_decay': 0.97,
                'dataset_split_seed': 10,
                'pytorch_init_seed': 3,
                'pg_viol_weight': 6.972774487659531,
                'vm_viol_weight': 42.98384563313474,
                'line_viol_weight': 2.789589883535428,
                'crit_weight': 9069, # 1e5,
                'kcl_weight': 115, # weight for KCL violation
                'LPF_weight': 3.686471151222091,
                'N_enrich': 50,
                'Algo': False, # if True, add worst-case violation CROWN bounds during training
                'Enrich': False,
                'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
            }
        elif algo == True:
            # ------- vr vi ------------
            parameters_dict = {
                'sweep': False,
                'test_system': 118,
                'hidden_layer_size': 50,
                'n_hidden_layers': 3,
                'epochs': 1000,
                'batch_size': 100,
                'learning_rate': 0.005055573805708793,
                'lr_decay': 0.97,
                'dataset_split_seed': 10,
                'pytorch_init_seed': 3,
                'pg_viol_weight': 1.1590475827878204,
                'vm_viol_weight': 3.875842235619783,
                'line_viol_weight': 72.87773745654778,
                'crit_weight': 5867, # 1e5,
                'kcl_weight': 630.394072244584, # weight for KCL violation
                'LPF_weight': 2.387856417356738,
                'N_enrich': 50,
                'Algo': True, # if True, add worst-case violation CROWN bounds during training
                'Enrich': False,
                'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
            }
    
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
