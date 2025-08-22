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
from EarlyStopping import EarlyStopping
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from neural_network.lightning_nn_crown import NeuralNetwork
# from LiRPANet import LiRPANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_np(x):
    return x.detach().numpy()

def train(config):
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)

    # Training Data
    Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)
    Dem_train = torch.tensor(Dem_train).float().to(device)
    Gen_train = torch.tensor(Gen_train).float().to(device)
    Gen_train_typ = torch.ones(Gen_train.shape[0], 1).to(device)

    num_classes = Gen_train.shape[1]
    Gen_delta = simulation_parameters['true_system']['Pg_delta']
    Dem_min = simulation_parameters['true_system']['Pd_min']
    Dem_delta = simulation_parameters['true_system']['Pd_delta']

    Data_stat = {
        'Gen_delta': Gen_delta,
        'Dem_min': Dem_min,
        'Dem_delta': Dem_delta,
    }

    # Test Data
    Dem_test, Gen_test = create_test_data(simulation_parameters=simulation_parameters)
    Dem_test = torch.tensor(Dem_test).float().to(device)
    Gen_test = torch.tensor(Gen_test).float().to(device)

    network_gen = build_network(Dem_train.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed)
    network_gen = normalise_network(network_gen, Dem_train, Data_stat)

    # Convert NN to lirpa_model that can calculate bounds on output
    lirpa_model = BoundedModule(network_gen, torch.empty_like(Dem_train), device=device) 
    print('Running on', device)

    x = Dem_min.reshape(1, -1) + Dem_delta.reshape(1, -1) / 2
    x = torch.tensor(x).float().to(device)
    x_min = torch.tensor(Dem_min.reshape(1, -1)).float().to(device)
    x_max = torch.tensor(Dem_min.reshape(1, -1) + Dem_delta.reshape(1, -1)).float().to(device)
    
    # set up input specificiation. Define upper and lower bound. Boundedtensor wraps nominal input(x) and associates it with defined perturbation ptb.
    ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
    image = BoundedTensor(x, ptb).to(device)

    optimizer = torch.optim.Adam(network_gen.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** -config.lr_decay)

    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = f'checkpoint_{n_buses}_{config.hidden_layer_size}_{config.Algo}.pt'
    path_dir = os.path.join(model_save_directory, path)
    early_stopping = EarlyStopping(patience=100, verbose=False, NN_input=Dem_train, path=path_dir)

    train_losses = []
    test_losses = []

    for epoch in range(config.epochs):
        # after every 100 epochs, enrich dataset with worst-case data
        if epoch % 100 == 0 and epoch != 0 and config.Enrich:
            X, Y, typ = wc_enriching(network_gen, config, Dem_train, Data_stat)
            InputNN = torch.cat((Dem_train, X), 0).to(device)
            OutputNN = torch.cat((Gen_train, Y), 0).to(device)
            typNN = torch.cat((Gen_train_typ, typ), 0).to(device)
            idx = torch.randperm(InputNN.shape[0])
            InputNN, OutputNN, typNN = InputNN[idx], OutputNN[idx], typNN[idx]
        else:
            InputNN = Dem_train
            OutputNN = Gen_train
            typNN = Gen_train_typ

        start_time = time.time()
        training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN, optimizer, config, simulation_parameters)
        validation_loss = validate_epoch(network_gen, Dem_test, Gen_test)
        training_time = time.time() - start_time

        train_losses.append(training_loss.item())
        test_losses.append(validation_loss.item())

        early_stopping(validation_loss, network_gen)

        loss_weight = config.LPF_weight / (1 + epoch * 0.01)
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=config.abc_method) 
        PF_violation = torch.abs(ub) + torch.abs(lb)

        # after 100 epochs, start adding wc PF violation penalty
        if config.Algo and epoch >= 100:
            optimizer.zero_grad()
            pf_loss = loss_weight * PF_violation 
            pf_loss.backward()
            optimizer.step()

        scheduler.step()

        if early_stopping.early_stop:
            break
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("Early stopping")


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


def validate_epoch(network_gen, Dem_test, Gen_test):
    criterion = nn.MSELoss()
    output = network_gen.forward_train(Dem_test)
    return criterion(output, Gen_test)


def train_epoch(network_gen, Dem_train, Gen_train, typ, optimizer, config, simulation_parameters):
    network_gen.train()
    criterion = nn.MSELoss()
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    num_samples = Dem_train.shape[0]
    num_batches = num_samples // config.batch_size
    Gen_delta = simulation_parameters['true_system']['Pg_delta']
    RELU = nn.ReLU()
    loss_sum = 0

    for i in range(num_batches):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        Gen_output = network_gen.forward_train(Dem_train[slce])
        Gen_target = Gen_train[slce]

        loss_gen = criterion(Gen_output * typ[slce], Gen_target * typ[slce])                 # supervised learning, difference from label
        violation = config.GenV_weight * (                                                   # generator limit violation penatly   
            torch.mean((RELU(Gen_output - torch.from_numpy(Gen_delta).to(device))) ** 2) +
            torch.mean((RELU(0 - Gen_output)) ** 2)
        )
        PF_loss = config.PF_weight * power_flow_check(Dem_train[slce], Gen_output, simulation_parameters)  # power flow violation error. Not correct?
        total_loss = loss_gen + violation + PF_loss

        total_loss.backward()
        optimizer.step()
        loss_sum += loss_gen

    return loss_sum / num_batches


# def power_flow_check(P_Loads, P_Gens, simulation_parameters):
#     PTDF = torch.tensor(simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float32)).to(device)
#     Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'].astype(np.float32)).to(device)
#     Map_L = torch.tensor(simulation_parameters['true_system']['Map_L'].astype(np.float32)).to(device)
#     Pl_max = torch.tensor(simulation_parameters['true_system']['Pl_max'].astype(np.float32)).to(device)
#     RELU = nn.ReLU()
#     PF_error = torch.abs(torch.sum(P_Gens, 1) - torch.sum(P_Loads, 1))
#     return torch.mean(PF_error)


def power_flow_check(P_Loads, P_Gens, simulation_parameters):
    PTDF = torch.tensor(simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float32)).to(device)
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'].astype(np.float32)).to(device)
    Map_L = torch.tensor(simulation_parameters['true_system']['Map_L'].astype(np.float32)).to(device)
    Pl_max = torch.tensor(simulation_parameters['true_system']['Pl_max'].astype(np.float32)).to(device)
    RELU = nn.ReLU() # Used for violation calculation

    # 1. Map generator outputs and loads to bus injections
    P_gen_bus = P_Gens @ Map_g 
    P_load_bus = P_Loads @ Map_L 

    # 2. Calculate net injection at each bus
    P_net_bus = P_gen_bus - P_load_bus

    # 3. Calculate line flows
    Line_flows_raw_T = PTDF.T @ P_net_bus.T
    Line_flows_raw = Line_flows_raw_T.T

    # 4. Calculate violations for upper and lower limits
    violation_upper = RELU(Line_flows_raw - Pl_max) # (batch_size, n_lines)
    violation_lower = RELU(-Pl_max - Line_flows_raw) # (batch_size, n_lines)

    # 5. Compute a loss based on the violations (e.g., sum of squares or mean)
    total_violation_loss = torch.mean(violation_upper**2 + violation_lower**2)
    
    return total_violation_loss



def wc_enriching(network_gen, config, Dem_train, Data_stat):
    n_adver = config.N_enrich
    Gen_output = network_gen.forward_aft(Dem_train).cpu().detach().numpy() # forward pass with clamping
    PB = np.sum(Gen_output, axis=1) - np.sum(Dem_train.cpu().numpy(), axis=1) # power balance violation
    
    # over generation - positive gradient ascent samples
    ind_p = np.argpartition(PB, -4)[-n_adver // 2:] # identify 'n_adver' indices with worst-case violation
    adv_p = GradAscnt(network_gen, Dem_train[ind_p].cpu().numpy(), Data_stat, sign=1)
    
    # under generation - negative gradient descent samples
    ind_n = np.argpartition(-PB, -4)[-n_adver // 2:]
    adv_n = GradAscnt(network_gen, Dem_train[ind_n].cpu().numpy(), Data_stat)
    x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()
    y_g = torch.zeros(x_g.shape[0], Gen_output.shape[1])
    y_type = torch.zeros(x_g.shape[0], 1)
    return x_g, y_g, y_type


def GradAscnt(Network, x_starting, Data_stat, sign=-1, Num_iteration=100, lr=0.0001):
    '''
    x_starting: Starting points for the gradient ascent algorithm
    x_min,x_max :  Minimum and maximum value of x ( default is 0 and 1)
    Sign : direction for gradient ascent ( 1 --> Increase the violation , -1 --> reduce the violation (.i.e. make it more negative))
    Num_iteration: Number of gradient steps
    lr: larning rate
    '''
    x = torch.tensor(x_starting, requires_grad=True).float()
    optimizer = torch.optim.SGD([x], lr=lr)

    for _ in range(Num_iteration):
        optimizer.zero_grad()
        output = Network.forward_aft(x) # forward pass with clamping
        PB = torch.sum(output, dim=1) # power balance
        loss = sign * torch.mean(PB)
        loss.backward()
        optimizer.step()

    return x.detach().numpy()
