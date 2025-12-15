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


def create_config(test_system):
    
    # -------- pg vm ----------
    parameters_dict = {
        'sweep': True,
        'test_system': test_system,
        'hidden_layer_size': 25,
        'n_hidden_layers': 3,
        'epochs': 1000,
        'batch_size': 25,
        'learning_rate': 5e-4,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'pg_viol_weight': 1e4,
        'qg_viol_weight': 0,
        'vm_viol_weight': 1e4,
        'line_viol_weight': 1e0,
        'crit_volt_weight': 1e4, # 1e5,
        'crit_pg_weight': 1e4, # 1e5,
        'PF_weight': 1e0,
        'LPF_weight': 1e0,
        'N_enrich': 50,
        'Algo': False, # if True, add worst-case violation CROWN bounds during training
        'Enrich': False,
        'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
    }
    
    if test_system == 14:
        parameters_dict['hidden_layer_size'] = 15
        parameters_dict['learning_rate'] = 2e-4
        parameters_dict['batch_size'] = 15
    elif test_system == 57:
        parameters_dict['learning_rate'] = 5e-4
        parameters_dict['hidden_layer_size'] = 25
        parameters_dict['batch_size'] = 25
    elif test_system == 118:
        parameters_dict['learning_rate'] = 10e-4
        parameters_dict['hidden_layer_size'] = 50
        parameters_dict['batch_size'] = 50
    elif test_system == 300:
        parameters_dict['hidden_layer_size'] = 75
        parameters_dict['learning_rate'] = 10e-4
        parameters_dict['batch_size'] = 75
    elif test_system == 793:
        parameters_dict['hidden_layer_size'] = 100
        parameters_dict['learning_rate'] = 20e-4
        parameters_dict['batch_size'] = 100
        
    
    config = SimpleNamespace(**parameters_dict)
    return config

    

# --------- hyperparameter tuning code --------

sweep_config = {
    'method': 'bayes',  # Use Bayesian optimization for efficient searching
    'metric': {
      'name': 'val_loss', # The metric to optimize (we want to minimize it)
      'goal': 'minimize'   
    },
    # 'tags': ["NN-Voltage", "v1.0"],
    'parameters': {
        # 'batch_size': {
        #     'values': [25, 50, 75, 100] # A few discrete batch sizes to try
        # },
        # 'learning_rate': {
        #     'distribution': 'log_uniform_values',
        #     'min': 1e-4,
        #     'max': 1e-2
        # },
        # 'epochs': {
        #     'value': 700 # A fixed value that is not being tuned
        # },
        # Tunable loss weights from your original config
        'pg_viol_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e0,
            'max': 1e3
        },
        'vm_viol_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e0,
            'max': 1e3
        },
        'crit_volt_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e0,
            'max': 1e3
        },
        'crit_pg_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e0,
            'max': 1e3
        },
        'LPF_weight': {
            'distribution': 'log_uniform_values',
            'min': 1e0,
            'max': 1e3
        },
        # You can add other parameters if needed
        # 'hidden_layer_size': {'values': [30, 50, 70]},
    }
}


# --- Step 2: The Training Function for the W&B Agent ---
def train_for_sweep(test_system):
    """
    This function will be called by the W&B agent for each trial.
    It gets its configuration from the sweep automatically.
    """
    # Create the base config first
    nn_type = 'pg_vm'
    base_config = create_config(test_system)

    with wandb.init(
        project="ac_verif_nn_training",
        group=f"sys_{base_config.test_system}_{nn_type}_Algo_{base_config.Algo}",
    ) as run:
        run.name=f"sys_{base_config.test_system}_{nn_type}_Algo_{base_config.Algo}_sweep_lr_{base_config.learning_rate:.4f}"
        # Then, update the base config with the sweep's chosen values.
        # This merges the two sets of parameters.
        for key, value in run.config.as_dict().items():
            setattr(base_config, key, value)
        
        # Now, you can use the merged base_config object
        config = base_config

        # Define test system and training paradigm
        simulation_parameters = create_example_parameters(config.test_system)
        nn_config = simulation_parameters['nn_output']
        
        # You need to have the `train` functions defined in the same scope
        # or imported dynamically here. Your dynamic import logic is correct.
        if nn_type == 'pg_vm':
            from nn_training_ac_crown_pg_vm import train
        elif nn_type == 'vr_vi':
            from nn_training_ac_crown_vr_vi import train
        else:
            print("Training paradigm not recognized.")
            return

        # Start the training for this specific trial
        train(config=config)
        
# --- Step 3: Launch the Sweep ---
def main_hyperparameter_tuning(test_system):
    """The main entry point to start the hyperparameter tuning sweep."""
    
    # Create the sweep in the W&B server and get its ID
    sweep_id = wandb.sweep(sweep_config, project="ac_verif_nn_training")
    
    # Run the agent to execute trials
    wandb.agent(sweep_id, function=lambda: train_for_sweep(test_system), count=10)
    
# --------------------------------------------


if __name__ == '__main__':
    systems = [793] # [57, 118]
    # if hyperparameter tuning
    for system in systems:
        main_hyperparameter_tuning(system)

