import time
import torch
import torch.nn as nn
import numpy as np
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
from ac_opf.create_data import create_data, create_test_data
from EarlyStopping import EarlyStopping
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from neural_network.lightning_nn_crown import NeuralNetwork
import torch.nn.utils.prune as prune
# from LiRPANet import LiRPANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize global pruning trackers
_total_prunable_params = 0
_current_total_sparsity = 0.0

def to_np(x):
    return x.detach().numpy()

def train(config):
    print("This is config: ", config)
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)
    simulation_parameters['nn_output'] = 'pg_vm'
    n_gens = simulation_parameters['general']['n_gbus']
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # system
    base_ppc = simulation_parameters['net_object']
    slack_bus_indices = np.where(base_ppc['bus'][:, 1] == 3)[0]  # BUS_TYPE == 3 (slack)
    gen_bus_indices = base_ppc['gen'][:, 0].astype(int)  # buses with generators
    mask = ~np.isin(gen_bus_indices, slack_bus_indices)
    gen_bus_indices_no_slack = gen_bus_indices[mask]

    # Training Data
    Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)
    Dem_train = torch.tensor(Dem_train).float().to(device)
    Gen_train = torch.tensor(Gen_train).float().to(device)
    Gen_train_typ = torch.ones(Gen_train.shape[0], 1).to(device)
    num_classes = Gen_train.shape[1]
    
    print("Shape training data: ", Dem_train.shape, Gen_train.shape)
    
    # generator min max
    pg_max_zero_mask = simulation_parameters['true_system']['Sg_max'][:n_gens] < 1e-9
    gen_mask_to_keep = ~pg_max_zero_mask  # invert mask to keep desired generators
    gen_delta = torch.tensor(simulation_parameters['true_system']['Sg_delta'][:n_gens][gen_mask_to_keep]).float().to(device).unsqueeze(1) / 100
    gen_min = torch.zeros_like(gen_delta)
    
    # output limits
    map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32, device=Gen_train.device)
    sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32, device=Gen_train.device)
    
    
    pg_max = sg_max[:n_gens, :][gen_mask_to_keep.squeeze()] / 100 # (sg_max.T @ map_g)[:, :n_buses]
    pg_min = torch.tensor(simulation_parameters['true_system']['pg_min'].T,dtype=torch.float32)[:, :][gen_mask_to_keep.squeeze()] / 100
    vmag_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float().to(device)
    vmag_min = torch.tensor(simulation_parameters['true_system']['Volt_min'][0]).float().to(device)
    
    # voltage min max   
    volt_min = torch.tensor(simulation_parameters['true_system']['Volt_min']).float().to(device).unsqueeze(1)[gen_bus_indices_no_slack]
    volt_max = torch.tensor(simulation_parameters['true_system']['Volt_max']).float().to(device).unsqueeze(1)[gen_bus_indices_no_slack]
    volt_delta = volt_max - volt_min
    
    output_min = torch.vstack((pg_min, volt_min))
    output_delta = torch.vstack((gen_delta, volt_delta))
    num_gen_nn = len(gen_delta)
    
    print(num_classes, len(gen_delta), len(pg_min), len(volt_min), len(volt_delta))
    
    dem_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float().to(device) / 100
    dem_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float().to(device) / 100
    
    # sampling bounds
    lower_bound_factor = 0.6
    upper_bound_factor = 1.0
    
    # Calculate the new, consistent bounds for verification
    p_max = dem_min + dem_delta
    
    # scaling factors
    sd_min_train_data = lower_bound_factor * p_max
    sd_delta_train_data = (upper_bound_factor - lower_bound_factor) * p_max
    sd_delta_train_data[sd_delta_train_data <= 1e-12] = 1.0
    
    # print(f"This is sd_train max and min: {Dem_train.max()}, {Dem_train.min()}")
    print(f"This is sd_train max and min: {sd_delta_train_data.max()}, {sd_min_train_data.min()}") # is
    print(f"This is vrvi_train max and min: {Gen_train.max()}, {Gen_train.min()}")

    data_stat = {
        'gen_min': output_min,
        'gen_delta': output_delta,
        'dem_min': sd_min_train_data,
        'dem_delta': sd_delta_train_data,
    }
    
    print(f"These are the scaling factors, gen_delta.max(): {data_stat['gen_delta'].max()}, dem_delta.max(): {data_stat['dem_delta'].max()}")

    # Test Data
    Dem_test, Gen_test = create_test_data(simulation_parameters=simulation_parameters)
    Dem_test = torch.tensor(Dem_test).float().to(device)
    Gen_test = torch.tensor(Gen_test).float().to(device) 
    
    print("shape test data: ", Dem_test.shape, Gen_test.shape)
    
    print(f"Training with {Dem_train.shape[0]} samples, validating with {Dem_test.shape[0]} samples")
    
    network_gen = build_network('pg_vm', Dem_train.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed, simulation_parameters)
    network_gen = normalise_network(network_gen, Dem_train, data_stat) 

    # Convert NN to lirpa_model that can calculate bounds on output
    lirpa_model = BoundedModule(network_gen, torch.empty_like(Dem_train), device=device) 
    print('Running on', device)

    # x = dem_min.reshape(1, -1) + dem_delta.reshape(1, -1) / 2
    # x = torch.tensor(x).float().to(device)
    # x_min = torch.tensor(dem_min.reshape(1, -1)).float().to(device)
    # x_max = torch.tensor(dem_min.reshape(1, -1) + dem_delta.reshape(1, -1)).float().to(device)
    
    # Apply the factors to the nominal load to get the correct min and max for the CROWN input space. 
    x_min = lower_bound_factor * p_max
    x_max = upper_bound_factor * p_max
        
    # Reshape and convert to float tensors for CROWN
    x_min = x_min.reshape(1, -1).clone().detach().float()
    x_max = x_max.reshape(1, -1).clone().detach().float()
    
    if torch.any(x_max < x_min):
        # Handle the error or log a warning
        print("Warning: x_max is smaller than x_min in some dimensions. Correcting...")
        # A robust way to ensure valid bounds is to use torch.min/max
        temp_min = torch.min(x_min, x_max)
        temp_max = torch.max(x_min, x_max)
        x_min = temp_min
        x_max = temp_max

    # Calculate the center of the new interval
    x = (x_min + x_max) / 2
    
    # set up input specificiation. Define upper and lower bound. Boundedtensor wraps nominal input(x) and associates it with defined perturbation ptb.
    ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
    image = BoundedTensor(x, ptb).to(device)

    optimizer = torch.optim.Adam(network_gen.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** -config.lr_decay)
    
    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = f'checkpoint_{n_buses}_{config.hidden_layer_size}_{config.Algo}_{simulation_parameters["nn_output"]}_final.pt'
    path_dir = os.path.join(model_save_directory, path)
    early_stopping = EarlyStopping(patience=500, verbose=False, NN_input=Dem_train, path=path_dir)
    
    train_losses = []
    test_losses = []

    # initialize datasets before enriching
    InputNN_total = Dem_train.clone()
    OutputNN_total = Gen_train.clone()
    typNN_total = Gen_train_typ.clone()

    for epoch in range(config.epochs):
        # after every 100 epochs, enrich dataset with worst-case data
        if epoch % 100 == 0 and epoch != 0 and config.Enrich:
            X_new, Y_new, typ_new = wc_enriching(network_gen, config, Dem_train, data_stat, simulation_parameters)
            
            # Concatenate new worst-case samples to the total dataset
            InputNN_total = torch.cat((InputNN_total, X_new.to(device)), dim=0)
            OutputNN_total = torch.cat((OutputNN_total, Y_new.to(device)), dim=0)
            typNN_total = torch.cat((typNN_total, typ_new.to(device)), dim=0)
            
        # Always use the full enriched dataset
        idx = torch.randperm(InputNN_total.shape[0])
        InputNN = InputNN_total[idx]
        OutputNN = OutputNN_total[idx]
        typNN = typNN_total[idx]

        start_time = time.time()
        mse_criterion, training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN, optimizer, config, simulation_parameters, epoch)
        validation_loss = validate_epoch(network_gen, Dem_test, Gen_test, simulation_parameters)
        training_time = time.time() - start_time
        
        wandb.log({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "mse_criterion": mse_criterion,
        })
        
        if epoch % 20 == 0 and epoch != 0:
            # Print MSE losses for both train and validation
            print(f"Epoch {epoch+1}/{config.epochs} — Train total loss: {training_loss:.6f}, Train MSE: {mse_criterion:.6f}, Validation MSE: {validation_loss:.6f}")

        train_losses.append(training_loss.item())
        test_losses.append(validation_loss.item())
        start_early_stop = 500
        if config.sweep == False and epoch > start_early_stop:
            if config.epochs == 1000 and start_early_stop != 500:
                raise ValueError("If training for 1000 epochs, start_early_stop should be 500")
            # only apply early stopping after weight pruning and if you're not sweeping hyperparameters
            early_stopping(validation_loss, network_gen)

        # after 100 epochs, start adding wc PF violation penalty
        if config.Algo and epoch >= 50:
            loss_weight = config.LPF_weight / (1 + epoch * 0.01)
            lb, ub = lirpa_model.compute_bounds(x=(image,), method=config.abc_method) 
            lb_pg = lb[:, :num_gen_nn]
            ub_pg = ub[:, :num_gen_nn]
            lb_vg = lb[:, num_gen_nn:]
            ub_vg = ub[:, num_gen_nn:]
            
            upper_gen_violation = torch.relu(ub_pg - pg_max)
            lower_gen_violation = torch.relu(pg_min - lb_pg)
            gen_violation_loss = (torch.mean(upper_gen_violation**2) + torch.mean(lower_gen_violation**2)) 
            
            vmag_up_violation = torch.relu(ub_vg - vmag_max)
            vmag_down_violation = torch.relu(vmag_min - lb_vg)
            vmag_violation = (torch.abs(vmag_up_violation ** 2).mean() + torch.abs(vmag_down_violation ** 2).mean())
            
            if epoch % 20 == 0:
                print(f"Average worst-case violation generator up: {upper_gen_violation.mean():.4f}")
                print(f"Average worst-case violation generator down: {lower_gen_violation.mean():.4f}")
                print(f"Average worst-case violation voltage up: {vmag_up_violation.mean():.4f}")
                print(f"Average worst-case violation voltage down: {vmag_down_violation.mean():.4f}")
        
            optimizer.zero_grad()
            pf_loss = loss_weight * (gen_violation_loss + vmag_violation) 
            pf_loss.backward()
            optimizer.step()
        
        # After some epoch, prune 50% neurons once
        if epoch == 500:
            apply_incremental_pruning(network_gen, 0.5) # Use the new function    

        scheduler.step()

        if early_stopping.early_stop and config.sweep == False:
            finalize_pruning(network_gen)
            break
        
    finalize_pruning(network_gen)
    
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(test_losses, label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Test Loss over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # print("Early stopping")


def build_network(nn_type, n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed, simulation_parameters, surrogate = None):
    hidden_layer_size = [hidden_layer_size] * n_hidden_layers
    model = NeuralNetwork(nn_type, n_input_neurons, hidden_layer_size=hidden_layer_size,
                          num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed, simulation_parameters = simulation_parameters, surrogate = surrogate)
    return model.to(device)


def normalise_network(model, Dem_train, data_stat):
    pd_min = data_stat['dem_min']
    pd_delta = data_stat['dem_delta']
    pg_delta = data_stat['gen_delta']
    pg_min = data_stat['gen_min']

    # input_stats = (torch.from_numpy(pd_min.reshape(-1,).astype(np.float64)),
    #                torch.from_numpy(pd_delta.reshape(-1,).astype(np.float64)))
    # output_stats = torch.from_numpy(pg_delta.reshape(-1,).astype(np.float64))
    
    input_stats = (pd_min.reshape(-1).float(), pd_delta.reshape(-1).float())
    output_stats = (pg_min.reshape(-1).float(), pg_delta.reshape(-1).float())

    model.normalise_input(input_stats)
    model.normalise_output(output_stats)
    return model.to(device)


def get_prunable_parameters(model):
    """Helper to collect parameters to prune."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            # If you also want to prune biases:
            # if module.bias is not None:
            #     parameters_to_prune.append((module, 'bias'))
    return parameters_to_prune

def calculate_current_sparsity(model):
    """Calculates the current total sparsity (fraction of zeros)."""

    nonzero = 0
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight
            nonzero += torch.count_nonzero(w).item()
            total += w.numel()
    if total == 0:
        return 0.0, 0 # No prunable weights
    
    current_sparsity_calculated = (total - nonzero) / total
    
    return current_sparsity_calculated, total # Returns (sparsity_fraction, total_params)




def apply_incremental_pruning(model, target_total_sparsity):
    """
    Applies pruning incrementally to reach a target total sparsity.
    target_total_sparsity: The desired *total* percentage of weights to be zeroed (e.g., 0.15, 0.30).
    """
    global _total_prunable_params, _current_total_sparsity

    # Calculate total prunable parameters if not already done
    if _total_prunable_params == 0:
        # Get total parameters from the model's structure
        parameters_to_prune = get_prunable_parameters(model)
        for module, param_name in parameters_to_prune:
            _total_prunable_params += getattr(module, param_name).numel()
        if _total_prunable_params == 0:
            print("Warning: No prunable parameters found in the model.")
            return

    # Calculate how much more to prune
    current_sparsity, _ = calculate_current_sparsity(model) # Get current actual sparsity
    
    # If already at or above target, no more pruning needed
    if current_sparsity >= target_total_sparsity:
        print(f"Current total sparsity ({current_sparsity*100:.2f}%) is already >= target ({target_total_sparsity*100:.2f}%). No additional pruning applied.")
        _current_total_sparsity = current_sparsity # Update global tracker
        return

    # Calculate the amount to prune from the *remaining* nonzero weights
    # This is (number of additional zeros needed) / (number of current nonzero weights)
    additional_zeros_needed = (target_total_sparsity - current_sparsity) * _total_prunable_params
    current_nonzero_params = (1 - current_sparsity) * _total_prunable_params

    if current_nonzero_params <= 0: # Avoid division by zero if all weights are already zero
        print("All weights are already zero. No further pruning possible.")
        _current_total_sparsity = 1.0 # Update global tracker
        return

    amount_to_prune_from_remaining = additional_zeros_needed / current_nonzero_params

    if amount_to_prune_from_remaining <= 0: # Should be caught by current_sparsity >= target_total_sparsity, but as a safeguard
        print("No additional pruning needed based on target.")
        _current_total_sparsity = target_total_sparsity # Update global tracker
        return

    # Apply global unstructured pruning
    parameters_to_prune = get_prunable_parameters(model) # Re-collect for prune.global_unstructured
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount_to_prune_from_remaining,
    )
    
    # Update the global tracker for total sparsity
    _current_total_sparsity, _ = calculate_current_sparsity(model)

    print(f"Applied incremental pruning to reach {target_total_sparsity*100:.2f}% total sparsity.")
    print(f"Current total sparsity: {_current_total_sparsity*100:.2f}%")

def finalize_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if prune.is_pruned(module): # Check if module is pruned before removing
                prune.remove(module, 'weight')
    print("Pruning re-parametrization removed (finalized).")



def validate_epoch(network_gen, Dem_test, Gen_test, simulation_parameters):
    criterion = nn.MSELoss()
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices)
    
    output = network_gen.forward_train(Dem_test)
    
    gen_pred = output[:, :n_act_gens] # * typ[slce]
    gen_true = Gen_test[:, :n_act_gens] # * typ[slce]
    mse_gen = criterion(gen_pred, gen_true)
    
    volt_pred = output[:, n_act_gens:] # * typ[slce]
    volt_true = Gen_test[:, n_act_gens:] # * typ[slce]
    mse_volt = criterion(volt_pred, volt_true)
    
    mse_criterion = mse_gen + mse_volt
        
    return mse_criterion


def train_epoch(network_gen, Dem_train, Gen_train, typ, optimizer, config, simulation_parameters, epoch):
    torch.autograd.set_detect_anomaly(True)

    network_gen.train()
    criterion = nn.MSELoss()
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    num_samples = Dem_train.shape[0]
    num_batches = num_samples // config.batch_size
    gen_delta = simulation_parameters['true_system']['Sg_delta']
    n_gens = simulation_parameters['general']['n_gbus']
    n_loads = Dem_train.shape[1]
    n_bus = simulation_parameters['general']['n_buses']
    RELU = nn.ReLU()
    loss_sum = 0
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    
    # get limits
    pg_max_values = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float64, device=device, requires_grad=False)[:n_gens][act_gen_indices]  / 100
    pg_min_values = torch.zeros_like(pg_max_values, requires_grad=False)
    vm_min_values = torch.tensor(simulation_parameters['true_system']['Volt_min'], dtype=torch.float64, device=device).unsqueeze(1)[pv_indices]
    vm_max_values = torch.tensor(simulation_parameters['true_system']['Volt_max'], dtype=torch.float64, device=device).unsqueeze(1)[pv_indices]
    
    # placeholder for Pg, and fill in with NN prediction
    Pg_place = torch.zeros((config.batch_size, n_gens), dtype=torch.float64, device=device, requires_grad=True)
    Vm_nn_place = torch.ones((config.batch_size, n_bus), dtype=torch.float64, device=device, requires_grad=True)
 
    preds = []
    targets = []

    for i in range(num_batches):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        Gen_output = network_gen.forward_train(Dem_train[slce])
        Gen_target = Gen_train[slce]
        
        # # compute actual pg and vm
        # Pg_active = Gen_output[:, :n_act_gens] 
        # Pg = Pg_place.clone()
        # Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64)
        
        # Vm_nn_g = Gen_output[:, n_act_gens:] 
        # Vm_nn = Vm_nn_place.clone()
        # Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
        
        gen_pred = Gen_output[:, :n_act_gens] # * typ[slce]
        gen_true = Gen_target[:, :n_act_gens] # * typ[slce]
        mse_gen = criterion(gen_pred, gen_true)
        
        volt_pred = Gen_output[:, n_act_gens:] # * typ[slce]
        volt_true = Gen_target[:, n_act_gens:] # * typ[slce]
        mse_volt = criterion(volt_pred, volt_true)
            
        # mse_criterion = criterion(Gen_output * typ[slce], Gen_target * typ[slce])
        mse_criterion = mse_gen + mse_volt
        loss_gen = config.crit_volt_weight * mse_volt + config.crit_pg_weight * mse_gen                # supervised learning, difference from label
        violation_pg = config.pg_viol_weight * (                                                   # generator limit violation penatly   
            torch.mean(RELU(gen_pred - pg_max_values.T) ** 2) +
            torch.mean((RELU(0 - gen_pred)) ** 2)
        )
        violation_vm = config.vm_viol_weight * (                                                   # voltage limit violation penatly   
            torch.mean(RELU(volt_pred - vm_max_values.T) ** 2) +
            torch.mean(RELU(vm_min_values.T - volt_pred) ** 2)
        )
        
        
        if epoch % 50 == 0 and epoch != 0 and i == 1:
            print(f"Pg loss: {violation_pg}, with weight: {config.pg_viol_weight}")
            print(f"Vm loss: {violation_vm}, with weight: {config.vm_viol_weight}")
            print(f"MSE loss: {loss_gen}, with volt weight: {config.crit_volt_weight} and gen weight: {config.crit_pg_weight}")


        

        total_loss = loss_gen + violation_pg + violation_vm 

        total_loss.backward()
        optimizer.step()
        loss_sum += loss_gen
    
    loss_epoch = loss_sum / num_batches

    return mse_criterion, loss_epoch


"""" 
convex relaxation of the OP for power flow.

"""

def linearized_pf_opt(nn_input, nn_output, simulation_parameters):
    """ 
    My NN only predicts the active power of generators which have pg_max > 0 and which are not the slack bus.
    The NN also predicts voltages at all generators. 
    
    """
    device = nn_output.device
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    
    # general system params
    n_gens = simulation_parameters['general']['n_gbus']
    n_loads = simulation_parameters['true_system']['n_lbus']
    n_bus = simulation_parameters['general']['n_buses']
    
    # Extract min/max Pg per generator:
    pg_max_values = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float64, device=device)[:n_gens] / 100 # divide by Sbase
    pg_min_values = torch.zeros_like(pg_max_values)
    vm_min_values = torch.tensor(simulation_parameters['true_system']['Volt_min'], dtype=torch.float64, device=device).unsqueeze(1)
    vm_max_values = torch.tensor(simulation_parameters['true_system']['Volt_max'], dtype=torch.float64, device=device).unsqueeze(1)
    pd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[:n_loads] / 100 # divide by Sbase
    qd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[n_loads:] / 100 # divide by Sbase
    
    # placeholder for Pg, and fill in with NN prediction
    Pg = torch.zeros((nn_output.shape[0], n_gens), dtype=torch.float64, device=device)
    Pg_active = nn_output[:, :n_act_gens] 
    Pg = Pg.clone()
    Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64)
    
    # placeholder for Vm, and fill in with NN prediction
    Vm_nn = torch.zeros((nn_output.shape[0], n_bus), dtype=torch.float64, device=device)
    Vm_nn_g = nn_output[:, n_act_gens:] 
    Vm_nn = Vm_nn.clone()
    Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
    
    # Fill in loads from NN input
    Pd = nn_input[:, :n_loads] 
    Qd = nn_input[:, n_loads:]
    
    # Add a small epsilon to denominators if you want (optional, but you had it for stability)
    eps = 1e-6
    pg_max_values += eps
    pg_min_values -= eps
    pg_denominator = pg_max_values - pg_min_values

    vm_max_values += eps
    vm_min_values -= eps
    vm_denominator = vm_max_values - vm_min_values
    
    # Inverse scaling of all states
    Pg = Pg * pg_denominator.T + pg_min_values.T  # shape (batch_size, n_gens)
    Vm_nn = Vm_nn * vm_denominator.T + vm_min_values.T  # shape (batch_size, n_bus)
    Pd = Pd * pd_nom.T
    Qd = Qd * qd_nom.T
    

    # mapping matrices gens and loads to buses
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float64, device=device)    # (n_bus-1, n_bus-1)
    Map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float64, device=device)
    
    # map injections to buses
    Pg = Pg @ Map_g[:n_gens, :n_bus] 
    Pd = Pd @ Map_l[:n_loads, :n_bus] 
    Qd = Qd @ Map_l[n_loads:, n_bus:] 
    
    assert Pg_active.shape[1] == len(act_gen_indices)
    assert Vm_nn_g.shape[1] == len(pv_indices)

    # Run FDLF to get Vm and delta
    Vm, delta, Qg = fdlf_solver(Pg, Pd, Qd, Vm_nn, simulation_parameters)

    # For now, let's just call the original function as an example
    violation_loss, S_line = ac_power_flow_check(Vm, delta, simulation_parameters)


    return violation_loss




def fdlf_solver(Pg, Pd, Qd, Vm_nn, simulation_parameters, max_iter=100, tol=1e-3):
    """
    Difficult to converge if power balance not satisfied.
    Inputs:
        Pg: Active power generation from NN (batch_size, n_bus)
        Pd, Qd: Loads at buses (batch_size, n_bus)
        Vm_nn: Voltage magnitude at generator buses from NN (batch_size, n_bus)
        simulation_parameters: contains Ybus, Bp, Bpp, slack_bus, bus types etc.
        
    Outputs:
        Vm: voltage magnitudes (fixed at PV buses = Vm_nn)
        delta: voltage angles (solved)
        Qg: reactive power generation (computed after convergence)
    """
    device = Pg.device
    batch_size, n_bus = Pg.shape[0], simulation_parameters['general']['n_buses']
    n_gens, n_loads = simulation_parameters['general']['n_gbus'], simulation_parameters['true_system']['n_lbus']

    # Extract matrices (assumed precomputed and stored in simulation_parameters)
    Bp = torch.tensor(simulation_parameters['true_system']['Bp'], dtype=torch.float64, device=device)    # (n_bus-1, n_bus-1)
    Bpp = torch.tensor(simulation_parameters['true_system']['Bpp'], dtype=torch.float64, device=device)  # (n_bus-1, n_bus-1)
    Ybus = torch.tensor(simulation_parameters['true_system']['Ybus'], dtype=torch.complex128, device=device)  # complex matrix
    
    # Add regularization
    # ε = 1e-6
    # Ip = torch.eye(Bp.shape[0], dtype=torch.float64, device=device)
    # Ipp = torch.eye(Bpp.shape[0], dtype=torch.float64, device=device)
    # Bp = Bp + ε * Ip
    # Bpp = Bpp + ε * Ipp
    
    Bp = torch.linalg.inv(Bp)
    Bpp = torch.linalg.inv(Bpp)

    slack_bus = simulation_parameters['true_system']['slack_bus']
    pq_indices = torch.tensor(simulation_parameters['true_system']['pq_buses'], dtype=torch.long, device=device)
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    pv_pq_indices = torch.cat([pv_indices, pq_indices])  # (n_bus - 1,)

    # Calculate net active power injection at each bus: P_inj = Pg - Pd
    P_inj = Pg  - Pd 

    # Initialize voltages: Vm = 1.0 p.u., delta = 0.0 radians
    Vm = torch.ones((batch_size, n_bus), dtype=torch.float64, device=device)
    delta = torch.zeros((batch_size, n_bus), dtype=torch.float64, device=device, requires_grad=True)

    # Set Vm at PV buses from NN output
    Vm = Vm.clone()
    Vm[:, pv_indices] = Vm_nn[:, pv_indices]

    max_step_d = 0.005
    max_step_v = 0.5
    

    for iteration in range(max_iter):
        # Calculate complex voltages
        V_complex = Vm * torch.exp(1j * delta)

        # Calculate complex power injections S = V * conj(Ybus * V)
        I_calc = torch.matmul(Ybus, V_complex.unsqueeze(-1)).squeeze(-1)
        S_calc = V_complex * torch.conj(I_calc)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # Calculate mismatches
        dP = P_inj[:, pv_pq_indices] - P_calc[:, pv_pq_indices]  # active power mismatch
        
        if iteration == 0:
            print("Initial ||ΔP||:", torch.norm(dP, dim = 1).mean().item())
            #print(torch.mean(dP, dim = 0))
        
        dP=torch.divide(dP,Vm[:, pv_pq_indices])
        dP = torch.clamp(dP, min=-0.5, max=0.5) # clip large dP updates for stabilization
        dDelta = torch.mm(Bp, dP.T).T   # (batch_size, n_bus-1)

        # Use dP and dQ norms as scale indicators
        mismatch_dP_norm = torch.norm(dP, dim=1, keepdim=True)  # shape (batch_size, 1)
        norm_dDelta = torch.norm(dDelta, dim=1, keepdim=True) + 1e-8

        # Gradient-descent style scale: move opposite mismatch direction with diminishing step
        scale_dDelta = torch.clamp(0.1 * mismatch_dP_norm / norm_dDelta, max=max_step_d)

        # Update delta and Vm separately
        delta_new = delta.clone()
        delta_new[:, pv_pq_indices] = delta[:, pv_pq_indices] + scale_dDelta * dDelta
        delta = delta_new
        
        # =========================
        
        # Calculate complex voltages
        V_complex = Vm * torch.exp(1j * delta)

        # Calculate complex power injections S = V * conj(Ybus * V)
        I_calc = torch.matmul(Ybus, V_complex.unsqueeze(-1)).squeeze(-1)
        S_calc = V_complex * torch.conj(I_calc)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # Calculate mismatches
        dQ = - Qd[:, pq_indices] - Q_calc[:, pq_indices]            # reactive power mismatch (load buses only)
        dQ=torch.divide(dQ,Vm[:, pq_indices])
        dVm = torch.mm(Bpp, dQ.T).T     # (batch_size, n_bus-1)
        
        # Use dP and dQ norms as scale indicators
        mismatch_dQ_norm = torch.norm(dQ, dim=1, keepdim=True)
        norm_dVm = torch.norm(dVm, dim=1, keepdim=True) + 1e-8

        # Gradient-descent style scale: move opposite mismatch direction with diminishing step
        scale_dVm = torch.clamp(0.1 * mismatch_dQ_norm / norm_dVm, max=max_step_v)
        
        Vm_new = Vm.clone()
        Vm_new[:, pq_indices] = Vm[:, pq_indices] + scale_dVm * dVm
        Vm = Vm_new
        

        
        
        print(torch.max(torch.abs(dDelta)), torch.max(torch.abs(dVm)))
        print(torch.norm(dP, dim = 1).mean().item(), torch.norm(dQ, dim = 1).mean().item())
        # print(torch.sum(converged_mask).item())

        # Check convergence
        if torch.max(torch.abs(dDelta)) < tol and torch.max(torch.abs(dVm)) < tol:
            print("OMGGG CONVERGED!")
            break
        elif iteration == max_iter - 1: 
            print(f"dDelta is {torch.max(torch.abs(dDelta))}, dVm is {torch.max(torch.abs(dVm))}. FDLF not converged...")
            print(torch.mean(dP, dim = 0))
            print(torch.mean(dDelta, dim = 0))
            raise RuntimeError(f"FDLF did not converge after {max_iter} iterations.")

    assert torch.all(delta[:, slack_bus] == 0.0), "delta at slack bus is not zero"
    assert torch.all(Vm[:, slack_bus] == 1.0), "voltage at slack bus is not zero"
    
    # # Fix slack bus values explicitly
    # delta = delta.clone()
    # Vm = Vm.clone()

    # delta[:, slack_bus] = 0.0
    # Vm[:, slack_bus] = 1.0

    # After convergence, compute Qg at generator buses:
    # Qg = Q_calc at PV buses + Qd at PV buses (since loads don’t exist at generator buses, usually Qd=0 there)
    Qg = Q_calc[:, pv_indices] + Qd[:, pv_indices]

    return Vm, delta, Qg





def ac_power_flow_check(Vm, delta, simulation_parameters):
    """
    Compute line flows using voltage magnitudes and angles from FDLF.
    Inputs:
        Vm: (batch_size, n_bus) voltage magnitudes (pu)
        delta: (batch_size, n_bus) voltage angles (rad)
        simulation_parameters: dict containing line data and limits
    Returns:
        total_violation_loss: scalar tensor measuring line flow violations
        S_line: (batch_size, n_lines) complex power flows
    """
    device = Vm.device
    RELU = nn.ReLU()

    # Extract system data
    R = torch.tensor(simulation_parameters['true_system']['R'], dtype=torch.float64, device=device)
    X = torch.tensor(simulation_parameters['true_system']['X'], dtype=torch.float64, device=device)
    Pl_max = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float64, device=device)

    from_buses = torch.tensor(simulation_parameters['true_system']['from_bus'], dtype=torch.long, device=device)
    to_buses = torch.tensor(simulation_parameters['true_system']['to_bus'], dtype=torch.long, device=device)

    # Convert to complex voltages
    V_complex = Vm.type(torch.complex128) * torch.exp(1j * delta.type(torch.complex128))  # (batch_size, n_bus)

    # Voltages at line ends
    V_from = V_complex[:, from_buses]  # (batch_size, n_lines)
    V_to = V_complex[:, to_buses]      # (batch_size, n_lines)

    # Line impedance
    Z = R + 1j * X  # (n_lines,)

    # Line currents
    I_line = (V_from - V_to) / Z  # (batch_size, n_lines)

    # Complex power flows
    S_line = V_from * torch.conj(I_line)  # (batch_size, n_lines)

    # Apparent power magnitudes
    S_magnitude = torch.abs(S_line) * 100 # (batch_size, n_lines)

    # Violations
    violation_upper = RELU(S_magnitude - Pl_max)

    # Total violation loss
    total_violation_loss = torch.mean(violation_upper**2)

    return total_violation_loss, S_line




def full_flow_check(nn_input, nn_output, simulation_parameters):
    """ 
    My NN only predicts the active power of generators which have pg_max > 0 and which are not the slack bus.
    The NN also predicts voltages at all generators. 
    
    """
    device = nn_output.device
    
    # get indices of active gens and pv buses
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    n_act_gens = len(act_gen_indices) # we're only predicting gens with pg_max > 0
    pv_indices = torch.tensor(simulation_parameters['true_system']['pv_buses'], dtype=torch.long, device=device)
    
    # general system params
    n_gens = simulation_parameters['general']['n_gbus']
    n_loads = simulation_parameters['true_system']['n_lbus']
    n_bus = simulation_parameters['general']['n_buses']
    
    # Extract min/max Pg per generator:
    pg_max_values = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float64, device=device)[:n_gens] / 100 # divide by Sbase
    pg_min_values = torch.zeros_like(pg_max_values)
    vm_min_values = torch.tensor(simulation_parameters['true_system']['Volt_min'], dtype=torch.float64, device=device).unsqueeze(1)
    vm_max_values = torch.tensor(simulation_parameters['true_system']['Volt_max'], dtype=torch.float64, device=device).unsqueeze(1)
    pd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[:n_loads] / 100 # divide by Sbase
    qd_nom = torch.tensor(simulation_parameters['true_system']['Sd_max'], dtype=torch.float64, device=device)[n_loads:] / 100 # divide by Sbase
    
    # placeholder for Pg, and fill in with NN prediction
    Pg = torch.zeros((nn_output.shape[0], n_gens), dtype=torch.float64, device=device)
    Pg_active = nn_output[:, :n_act_gens] 
    Pg = Pg.clone()
    Pg[:, act_gen_indices] = Pg_active.to(dtype=torch.float64)
    
    # placeholder for Vm, and fill in with NN prediction
    Vm_nn = torch.zeros((nn_output.shape[0], n_bus), dtype=torch.float64, device=device)
    Vm_nn_g = nn_output[:, n_act_gens:] 
    Vm_nn = Vm_nn.clone()
    Vm_nn[:, pv_indices] = Vm_nn_g.to(dtype=torch.float64)
    
    # Fill in loads from NN input
    Pd = nn_input[:, :n_loads] 
    Qd = nn_input[:, n_loads:]
        
    # print(torch.mean(Pg, dim= 0))
    # print(torch.mean(Pd, dim= 0))
    
    # print(torch.sum(torch.mean(Pg, dim= 0)))
    # print(torch.sum(torch.mean(Pd, dim= 0)))
    
    # mapping matrices gens and loads to buses
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float64, device=device)    # (n_bus-1, n_bus-1)
    Map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float64, device=device)
    
    # map injections to buses
    Pg = Pg @ Map_g[:n_gens, :n_bus] 
    Pd = Pd @ Map_l[:n_loads, :n_bus] 
    Qd = Qd @ Map_l[n_loads:, n_bus:] 
    
    assert Pg_active.shape[1] == len(act_gen_indices)
    assert Vm_nn_g.shape[1] == len(pv_indices)

    # Run FDLF to get Vm and delta
    Vm, delta, Qg = fdlf_solver(Pg, Pd, Qd, Vm_nn, simulation_parameters)

    # For now, let's just call the original function as an example
    violation_loss, S_line = ac_power_flow_check(Vm, delta, simulation_parameters)


    return violation_loss



def wc_enriching(network_gen, config, Dem_train, data_stat):
    n_adver = config.N_enrich

    # Forward pass to get generator outputs (Pg and Vm stacked)
    nn_output = network_gen.forward_aft(Dem_train).cpu().detach().numpy()
    
    n_gens = data_stat['gen_delta'].shape[0] // 2 
    n_loads = Dem_train.shape[1] // 2  # Pd and Qd stacked, so loads = half of input features

    # Split loads into Pd and Qd
    P_gen = nn_output[:, :n_gens]
    P_load = Dem_train[:, :n_loads].cpu().numpy()  # shape (n_loads, n_samples)

    # Only consider active power balance violation (Pg - Pd)
    # Sum over generators and loads per sample (axis=0)
    PB_P = np.sum(P_gen, axis=1) - np.sum(P_load, axis=1)  # shape (n_samples,)

    # Find indices with largest positive violations (overgeneration)
    ind_p = np.argpartition(PB_P, -n_adver // 2)[-n_adver // 2:]
    adv_p = GradAscnt(network_gen, Dem_train[ind_p, :].cpu().numpy(), data_stat, sign=1)

    # Find indices with largest negative violations (undergeneration)
    ind_n = np.argpartition(-PB_P, -n_adver // 2)[-n_adver // 2:]
    adv_n = GradAscnt(network_gen, Dem_train[ind_n, :].cpu().numpy(), data_stat, sign=-1)

    x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()  # concatenate on sample axis
    y_g = torch.zeros(x_g.shape[0], nn_output.shape[1])
    y_type = torch.zeros(x_g.shape[0], 1)

    return x_g, y_g, y_type





def GradAscnt(Network, x_starting, data_stat, sign=-1, Num_iteration=100, lr=0.0001):
    '''
    x_starting: Starting points for gradient ascent (shape: features x batch_size)
    sign: 1 to increase violation, -1 to decrease violation
    '''
    x = torch.tensor(x_starting, requires_grad=True).float()
    optimizer = torch.optim.SGD([x], lr=lr)

    n_gens = data_stat['gen_delta'].shape[0] // 2 
    n_loads = x.shape[1] // 2

    for _ in range(Num_iteration):
        optimizer.zero_grad()
        nn_output = Network.forward_aft(x)  # shape (n_gens, batch_size)

        # Active power from generator
        P_gen = nn_output[:, :n_gens]

        # Active power from load (input)
        P_load = x[:, :n_loads]  # Pd

        # Power balance violation (Pg - Pd)
        PB_P = torch.sum(P_gen, dim=1) - torch.sum(P_load, dim=1)

        loss = sign * torch.mean(PB_P)  # mean violation scaled by sign

        loss.backward()
        optimizer.step()

    return x.detach().numpy()


# def power_flow_check(P_Loads, P_Gens, simulation_parameters):
#     PTDF = torch.tensor(simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float64)).to(device)
#     Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'].astype(np.float64)).to(device)
#     Map_L = torch.tensor(simulation_parameters['true_system']['Map_L'].astype(np.float64)).to(device)
#     Pl_max = torch.tensor(simulation_parameters['true_system']['Pl_max'].astype(np.float64)).to(device)
#     RELU = nn.ReLU() # Used for violation calculation

#     # 1. Map generator outputs and loads to bus injections
#     P_gen_bus = P_Gens @ Map_g 
#     P_load_bus = P_Loads @ Map_L 

#     # 2. Calculate net injection at each bus
#     P_net_bus = P_gen_bus - P_load_bus

#     # 3. Calculate line flows
#     Line_flows_raw_T = PTDF.T @ P_net_bus.T
#     Line_flows_raw = Line_flows_raw_T.T

#     # 4. Calculate violations for upper and lower limits
#     violation_upper = RELU(Line_flows_raw - Pl_max) # (batch_size, n_lines)
#     violation_lower = RELU(-Pl_max - Line_flows_raw) # (batch_size, n_lines)

#     # 5. Compute a loss based on the violations (e.g., sum of squares or mean)
#     total_violation_loss = torch.mean(violation_upper**2 + violation_lower**2)
    
#     return total_violation_loss



# def wc_enriching(network_gen, config, Dem_train, data_stat):
#     n_adver = config.N_enrich
#     Gen_output = network_gen.forward_aft(Dem_train).cpu().detach().numpy() # forward pass with clamping
#     PB = np.sum(Gen_output, axis=1) - np.sum(Dem_train.cpu().numpy(), axis=1) # power balance violation
    
#     # over generation - positive gradient ascent samples
#     ind_p = np.argpartition(PB, -4)[-n_adver // 2:] # identify 'n_adver' indices with worst-case violation
#     adv_p = GradAscnt(network_gen, Dem_train[ind_p].cpu().numpy(), data_stat, sign=1)
    
#     # under generation - negative gradient descent samples
#     ind_n = np.argpartition(-PB, -4)[-n_adver // 2:]
#     adv_n = GradAscnt(network_gen, Dem_train[ind_n].cpu().numpy(), data_stat)
#     x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()
#     y_g = torch.zeros(x_g.shape[0], Gen_output.shape[1])
#     y_type = torch.zeros(x_g.shape[0], 1)
#     return x_g, y_g, y_type




# def GradAscnt(Network, x_starting, data_stat, sign=-1, Num_iteration=100, lr=0.0001):
#     '''
#     x_starting: Starting points for the gradient ascent algorithm
#     x_min,x_max :  Minimum and maximum value of x ( default is 0 and 1)
#     Sign : direction for gradient ascent ( 1 --> Increase the violation , -1 --> reduce the violation (.i.e. make it more negative))
#     Num_iteration: Number of gradient steps
#     lr: larning rate
#     '''
#     x = torch.tensor(x_starting, requires_grad=True).float()
#     optimizer = torch.optim.SGD([x], lr=lr)

#     for _ in range(Num_iteration):
#         optimizer.zero_grad()
#         output = Network.forward_aft(x) # forward pass with clamping
#         PB = torch.sum(output, dim=1) # power balance
#         loss = sign * torch.mean(PB)
#         loss.backward()
#         optimizer.step()

#     return x.detach().numpy()