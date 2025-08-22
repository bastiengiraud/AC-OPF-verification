from types import SimpleNamespace
from nn_training_dc_crown import train


def create_config():
    parameters_dict = {
        'test_system': 300,
        'hidden_layer_size': 50,
        'n_hidden_layers': 3,
        'epochs': 100,
        'batch_size': 100,
        'learning_rate': 0.001,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'GenV_weight': 1e-4,
        'PF_weight': 1e-4,
        'LPF_weight': 0.001,
        'N_enrich': 0,
        'Algo': True, # if True, add worst-case violation CROWN bounds during training
        'Enrich': True,
        'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN
    }
    config = SimpleNamespace(**parameters_dict)
    return config


def main():
    config = create_config()
    train(config=config)


if __name__ == '__main__':
    main()
