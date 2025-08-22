import time
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data', 'models', 'scripts/utils']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from dc_opf.create_example_parameters import create_example_parameters
from dc_opf.create_data import create_data, create_test_data
from functions.wcg_dc_opf import milp_wcg
from neural_network.lightning_nn_crown import NeuralNetwork
from LiRPANet import LiRPANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_np(x):
    return x.detach().numpy()

def train(config=None):
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)

    # Training Data
    Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)
    Dem_train = torch.tensor(Dem_train).float()
    Gen_train = torch.tensor(Gen_train).float()

    num_classes = Gen_train.shape[1]

    Gen_delta = simulation_parameters['true_system']['Pg_delta']
    Dem_min = simulation_parameters['true_system']['Pd_min']
    Dem_delta = simulation_parameters['true_system']['Pd_delta']

    Data_stat = {
        'Gen_delta': Gen_delta,
        'Dem_min': Dem_min,
        'Dem_delta': Dem_delta
    }

    # Test Data
    Dem_test, Gen_test = create_test_data(simulation_parameters=simulation_parameters)
    Dem_test = torch.tensor(Dem_test).float()
    Gen_test = torch.tensor(Gen_test).float()

    # Network
    network_gen = build_network(
        Dem_train.shape[1],
        num_classes,
        config.hidden_layer_size,
        config.n_hidden_layers,
        config.pytorch_init_seed
    )

    network_gen = normalise_network(network_gen, Dem_train, Data_stat)
    
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = f'checkpoint_{n_buses}_{config.hidden_layer_size}_{config.Algo}.pt'
    path_dir = os.path.join(model_save_directory, path)
    network_gen.load_state_dict(torch.load(path_dir, map_location=torch.device('cpu')).state_dict())

    wandbs = network_gen.state_dict()

    Bound_Function = LiRPANet(n_buses, Data_stat)
    Lower_bound, Upper_bound, Pg_min, Pg_max = Bound_Function(network_gen, config.abc_method)

    Z_min = np.zeros((config.hidden_layer_size, config.n_hidden_layers))
    Z_min[:, 0] = Lower_bound['/input'].cpu().detach().numpy()
    Z_min[:, 1] = Lower_bound['/input.3'].cpu().detach().numpy()
    Z_min[:, 2] = Lower_bound['/input.7'].cpu().detach().numpy()

    Z_max = np.zeros((config.hidden_layer_size, config.n_hidden_layers))
    Z_max[:, 0] = Upper_bound['/input'].cpu().detach().numpy()
    Z_max[:, 1] = Upper_bound['/input.3'].cpu().detach().numpy()
    Z_max[:, 2] = Upper_bound['/input.7'].cpu().detach().numpy()

    W_last = network_gen.L_4.weight.to(device).data.numpy()
    B_last = network_gen.L_4.bias.to(device).data.numpy()
    N_hid_l = config.n_hidden_layers

    Pg_hat_max = ((np.maximum(W_last, 0)) @ np.maximum(Z_max[:, N_hid_l - 1], 0) +
                  (np.minimum(W_last, 0)) @ np.maximum(Z_min[:, N_hid_l - 1], 0)).reshape((num_classes,)) + B_last.reshape((num_classes,))
    Pg_hat_min = ((np.minimum(W_last, 0)) @ np.maximum(Z_max[:, N_hid_l - 1], 0) +
                  (np.maximum(W_last, 0)) @ np.maximum(Z_min[:, N_hid_l - 1], 0)).reshape((num_classes,)) + B_last.reshape((num_classes,))

    print(network_gen)

    W = {
        0: network_gen.L_1.weight.to(device).data.numpy(),
        1: network_gen.L_2.weight.to(device).data.numpy(),
        2: network_gen.L_3.weight.to(device).data.numpy(),
        3: network_gen.L_4.weight.to(device).data.numpy()
    }

    B = {
        0: network_gen.L_1.bias.to(device).data.numpy(),
        1: network_gen.L_2.bias.to(device).data.numpy(),
        2: network_gen.L_3.bias.to(device).data.numpy(),
        3: network_gen.L_4.bias.to(device).data.numpy()
    }

    time_start = time.time()
    PF_violation = milp_wcg(
        config.pytorch_init_seed,
        n_buses,
        W,
        B,
        np.transpose(Gen_delta),
        np.transpose(Gen_delta),
        config,
        Z_min,
        Z_max,
        Pg_hat_min,
        Pg_hat_max
    )

    print("PF_violation:", PF_violation)

def build_network(n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    hidden_layer_size = [hidden_layer_size] * n_hidden_layers
    model = NeuralNetwork(n_input_neurons, hidden_layer_size=hidden_layer_size,
                          num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed)
    return model.to(device)


def normalise_network(model, Dem_train, Data_stat):
    pd_min = Data_stat['Dem_min']
    pd_delta = Data_stat['Dem_delta']
    pg_delta = Data_stat['Gen_delta']

    input_stats = (torch.from_numpy(pd_min.reshape(-1,).astype(np.float32)),
                   torch.from_numpy(pd_delta.reshape(-1,).astype(np.float32)))
    output_stats = torch.from_numpy(pg_delta.reshape(-1,).astype(np.float32))

    model.normalise_input(input_stats)
    model.normalise_output(output_stats)
    return model.to(device)
