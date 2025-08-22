from types import SimpleNamespace

import verification_dc

def create_config():
    parameters_dict = {
        'test_system': 300,
        'hidden_layer_size':50,
        'n_hidden_layers': 3,
        'epochs': 100,
        'batch_size': 100,
        'learning_rate': 0.001,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed':3,
        'PF_weight': 0.0001,
        'Ver_type':'GCP',
        'Adverse_attack' : 'Adverse_data_true', # warm start with gradient attack
        'corelated_input' : 'uncorrelated',
        'Max_gcp_iter' : 1, # add general cutting planes
        'BigM_bound': 'None', # MILP
        'Line_ID_max': 25,
        'TimeLimit': 3600,
        'callback' : False,
        'Algo':'True',
        'abc_method' : 'IBP',
        'ID': 'cz4ft2yd'

    }

    config = SimpleNamespace(**parameters_dict)
    return config


def train_single_run():
    run_config = create_config()
    verification_dc.train(config=run_config)


if __name__ == '__main__':
    train_single_run()
