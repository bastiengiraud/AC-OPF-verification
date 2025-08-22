import os
import sys

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from surrogate.create_example_parameters import create_example_parameters
from types import SimpleNamespace


def create_config():
    parameters_dict = {
        'test_system': 118,
        'hidden_layer_size': 4,
        'n_hidden_layers': 2,
        'epochs': 200,
        'batch_size': 64,
        'learning_rate': 1e-2,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'pg_viol_weight': 0,
        'qg_viol_weight': 0,
        'vm_viol_weight': 0,
        'sf_viol_weight': 1e3,
        'crit_weight': 1e3,
        'PF_weight': 1e-4,
        'LPF_weight': 1e-4,
        'N_enrich': 50,
        'Algo': True, # if True, add worst-case violation CROWN bounds during training
        'Enrich': False,
        'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN
    }
    config = SimpleNamespace(**parameters_dict)
    return config


def main():
    config = create_config()
    
    # define test system
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)
    nn_config = simulation_parameters['nn_output']
    
    # define training paradigm
    if nn_config == 'surrogate_s':
        from nn_training_surrogate_s import train
    elif nn_config == 'surrogate_i':
        from nn_training_surrogate_i import train
    else:
        print("Training paradigm not recognized.")
    
    # start training
    train(config=config)


if __name__ == '__main__':
    main()
