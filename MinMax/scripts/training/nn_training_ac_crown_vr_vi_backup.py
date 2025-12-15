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
from neural_network.lightning_nn_crown import NeuralNetwork, OutputWrapper
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
    simulation_parameters['nn_output'] = 'vr_vi'
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # # get labels for generators
    simulation_parameters_gens = simulation_parameters.copy()
    simulation_parameters_gens['nn_output'] = 'pg_vm'
    act_gen_indices = simulation_parameters['true_system']['pg_active']
    num_act_gens = len(act_gen_indices)
    n_gens = simulation_parameters['general']['n_gbus']
    # _, pgvm_train = create_data(simulation_parameters=simulation_parameters_gens)
    # pgvm_train = torch.tensor(pgvm_train[:, :num_act_gens]).float().to(device)

    # Training Data
    sd_train, vrvi_train = create_data(simulation_parameters=simulation_parameters)    
    sd_train = torch.tensor(sd_train).float().to(device)
    n_loads = sd_train.shape[1] // 2
    vrvi_train = torch.tensor(vrvi_train).float().to(device)
    vrvi_train_typ = torch.ones(vrvi_train.shape[0], 1).to(device)
    
    map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32, device=vrvi_train.device)
    sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32, device=vrvi_train.device) / 100
    pg_max = (sg_max.T @ map_g)[:, :n_buses]
    qg_max = (sg_max.T @ map_g)[:, n_buses:]
    pg_min = torch.tensor(simulation_parameters['true_system']['pg_min'], dtype=torch.float32) @ map_g[n_gens:, n_buses:] / 100
    qg_min = torch.tensor(simulation_parameters['true_system']['qg_min'], dtype=torch.float32) @ map_g[n_gens:, n_buses:] / 100

    num_classes = vrvi_train.shape[1]
    sd_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float().to(device) / 100
    sd_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float().to(device) / 100
    vmag_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float().to(device)
    vmag_min = torch.tensor(simulation_parameters['true_system']['Volt_min'][0]).float().to(device)
    imag_max = torch.tensor(simulation_parameters['true_system']['I_max_pu']).float().to(device)
    imag_max_tot = torch.cat((imag_max, imag_max), dim = 0)
    
    # vrvi_max = torch.tensor(simulation_parameters['true_system']['Volt_max'][0]).float().to(device)
    # vrvi_min = -vrvi_max
    # vrvi_delta = vrvi_max - vrvi_min
    # vrvi_delta[vrvi_delta <= 1e-12] = 1.0
    
    # sampling bounds
    lower_bound_factor = 0.6
    upper_bound_factor = 1.0
    
    # Calculate the new, consistent bounds for verification
    p_max = sd_min + sd_delta
    
    # scaling factors
    sd_min_train_data = lower_bound_factor * p_max
    sd_delta_train_data = (upper_bound_factor - lower_bound_factor) * p_max
    sd_delta_train_data[sd_delta_train_data <= 1e-12] = 1.0
    
    # Compute stats for real and imaginary parts
    vr_train = vrvi_train[:, :n_buses]
    vi_train = vrvi_train[:, n_buses:]

    vr_max = vr_train.max(dim=0).values
    vr_min = vr_train.min(dim=0).values
    vr_delta = vr_max - vr_min
    vr_delta[vr_delta <= 1e-12] = 1.0  # Avoid division by zero

    vi_max = vi_train.max(dim=0).values
    vi_min = vi_train.min(dim=0).values
    vi_delta = vi_max - vi_min
    vi_delta[vi_delta <= 1e-12] = 1.0

    # Concatenate into unified vectors
    vrvi_max = torch.cat([vr_max, vi_max], dim=0)
    vrvi_min = torch.cat([vr_min, vi_min], dim=0)
    vrvi_delta = torch.cat([vr_delta, vi_delta], dim=0)
    
    print(f"This is sd_train max and min: {sd_delta_train_data.max()}, {sd_min_train_data.min()}") # is
    print(f"This is vrvi_train max and min: {vrvi_train.max()}, {vrvi_train.min()}")

    data_stat = {
        'sd_min': sd_min_train_data,
        'sd_delta': sd_delta_train_data,
        'vrvi_min': vrvi_min,
        'vrvi_delta': vrvi_delta,
    }
    
    print(f"These are the scaling factors, sd_delta.max(): {data_stat['sd_delta'].max()}, vrvi_delta.max(): {data_stat['vrvi_delta'].max()}")

    # Test Data
    sd_test, vrvi_test = create_test_data(simulation_parameters=simulation_parameters)
    # _, pgvm_test = create_test_data(simulation_parameters=simulation_parameters_gens)
    # pgvm_test = torch.tensor(pgvm_test[:, :num_act_gens]).float().to(device)
    sd_test = torch.tensor(sd_test).float().to(device)
    vrvi_test = torch.tensor(vrvi_test).float().to(device)
    
    network_gen = build_network('vr_vi', sd_train.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed, simulation_parameters)
    network_gen = normalise_network(network_gen, sd_train, data_stat) 
    
    print(f"Training with {sd_train.shape[0]} samples, validating with {sd_test.shape[0]} samples")

    
    # multi output network
    lirpa_vr, lirpa_vi, lirpa_ir, lirpa_ii, lirpa_iu, lirpa_vu, lirpa_vl, lirpa_pgu, lirpa_pgl, lirpa_qgu, lirpa_qgl, lirpa_injr, lirpa_inji = \
    initialize_lirpa_modules(network_gen, torch.empty_like(sd_train).float(), device)
    
    print('Running on', device)

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

    # x = sd_min.reshape(1, -1) + sd_delta.reshape(1, -1) / 2
    # x = x.clone().detach().float().to(device)
    # x_min = torch.tensor(sd_min.reshape(1, -1)).float().to(device)
    # x_max = torch.tensor(sd_min.reshape(1, -1) + sd_delta.reshape(1, -1)).float().to(device)
    
    # set up input specificiation. Define upper and lower bound. Boundedtensor wraps nominal input(x) and associates it with defined perturbation ptb.
    ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
    image = BoundedTensor(x, ptb).to(device)

    optimizer = torch.optim.Adam(network_gen.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** -config.lr_decay)

    model_save_directory = os.path.join(project_root_dir, 'models', 'best_model')
    path = f'checkpoint_{n_buses}_{config.hidden_layer_size}_{config.Algo}_{simulation_parameters["nn_output"]}_final.pt'
    path_dir = os.path.join(model_save_directory, path)
    early_stopping = EarlyStopping(patience=500, verbose=False, NN_input=sd_train, path=path_dir)
    
    train_losses = []
    test_losses = []
    
    # initialize datasets before enriching
    InputNN_total = sd_train.clone()
    OutputNN_total = vrvi_train.clone()
    typNN_total = vrvi_train_typ.clone()
    # pg_total = pgvm_train.clone()


    for epoch in range(config.epochs):

        # after every 100 epochs, enrich dataset with worst-case data
        if epoch % 100 == 0 and epoch != 0 and config.Enrich:
            X_new, Y_new, typ_new = wc_enriching(network_gen, config, sd_train, data_stat, simulation_parameters)
            
            # Concatenate new worst-case samples to the total dataset
            InputNN_total = torch.cat((InputNN_total, X_new.to(device)), dim=0)
            OutputNN_total = torch.cat((OutputNN_total, Y_new.to(device)), dim=0)
            typNN_total = torch.cat((typNN_total, typ_new.to(device)), dim=0)

            
        # Always use the full enriched dataset
        idx = torch.randperm(InputNN_total.shape[0])
        InputNN = InputNN_total[idx]
        OutputNN = OutputNN_total[idx]
        typNN = typNN_total[idx]
        # pg_target = pg_total[idx]

        start_time = time.time()
        mse_criterion, training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN, optimizer, config, simulation_parameters, epoch) 
        validation_loss = validate_epoch(network_gen, sd_test, vrvi_test)
        training_time = time.time() - start_time    
        
        # log the results
        wandb.log({
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "mse_criterion": mse_criterion,
        })
            
        if epoch % 20 == 0 and epoch != 0:
            # Print MSE losses for both train and validation
            current_sparsity_val, current_total_params = calculate_current_sparsity(network_gen)
            print(f"Epoch {epoch}/{config.epochs} — Train total loss: {training_loss:.6f}, Train MSE: {mse_criterion:.6f}, Validation MSE: {validation_loss:.6f}")

        train_losses.append(mse_criterion.item())
        test_losses.append(validation_loss.item())
        start_early_stop = 500
        if config.sweep == False and epoch > start_early_stop:
            if config.epochs == 1000 and start_early_stop != 500:
                raise ValueError("If training for 1000 epochs, start_early_stop should be 500")
            # only apply early stopping after weight pruning and if you're not sweeping hyperparameters
            early_stopping(validation_loss, network_gen)
            
        
        # after 100 epochs, start adding wc violation penalty
        if config.Algo and epoch >= 50:
            
            loss_weight = config.LPF_weight / (1 + epoch * 0.01)
            
            # crown bounds voltage magnitudes and currents
            _, ub_iu = lirpa_iu.compute_bounds(x=(image,), method=config.abc_method)
            _, ub_vu = lirpa_vu.compute_bounds(x=(image,), method=config.abc_method)
            lb_vl, _ = lirpa_vl.compute_bounds(x=(image,), method=config.abc_method)
            
            # check if the bounds are not NaN or Inf
            def sanitize_bounds(tensor, name):
                """ 
                This breaks the computational graph, but at least flags the problem.
                
                """
                nan_mask = torch.isnan(tensor)
                posinf_mask = tensor == float('inf')
                neginf_mask = tensor == float('-inf')

                nan_count = nan_mask.sum().item()
                posinf_count = posinf_mask.sum().item()
                neginf_count = neginf_mask.sum().item()

                if nan_count > 0 or posinf_count > 0 or neginf_count > 0:
                    print(
                        f"Issues in {name}: "
                        f"{nan_count} NaNs, {posinf_count} +Infs, {neginf_count} -Infs. "
                        "Replacing with finite values."
                    )
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

                return tensor

            ub_iu = sanitize_bounds(ub_iu, "ub_iu")
            ub_vu = sanitize_bounds(ub_vu, "ub_vu")
            lb_vl = sanitize_bounds(lb_vl, "lb_vl")
            
            vmag_up_violation = torch.relu(ub_vu - vmag_max)
            vmag_down_violation = torch.relu(vmag_min - lb_vl)
            vmag_violation = (torch.clamp(vmag_up_violation, max=1e3).square().mean() + torch.clamp(vmag_down_violation, max=1e3).square().mean())
            
            imag_up_violation = torch.relu(ub_iu - imag_max_tot) # get actual current ratings
            imag_violation = (torch.abs(imag_up_violation ** 2).mean())
            
            gen_violation_loss = 0
            inj_violation = 0
            
            if epoch % 20 == 0:
                print(f"Average worst-case current mag: {ub_iu.mean():.4f}")
                print(f"Average worst-case violation voltage mag up: {vmag_up_violation.mean():.4f}")
                print(f"Average worst-case violation voltage mag down: {vmag_down_violation.mean():.4f}")
                print(f"Worst-case violation voltage mag up: {vmag_up_violation.max():.4f}")
                print(f"Worst-case violation voltage mag down: {vmag_down_violation.max():.4f}")
                # print(f"Average worst-case violation voltage rect up: {vrect_up_violation.mean():.4f}")
                # print(f"Average worst-case violation voltage rect down: {vrect_down_violation.mean():.4f}")
                
            
            if epoch > 300:
                """
                Compute the worst-case violations on generator setpoints and injections.
                
                """
                # crown bounds generator active and reactive power
                _, ub_pg = lirpa_pgu.compute_bounds(x=(image,), method=config.abc_method)
                lb_pg, _ = lirpa_pgl.compute_bounds(x=(image,), method=config.abc_method)
                _, ub_qg = lirpa_qgu.compute_bounds(x=(image,), method=config.abc_method)
                lb_qg, _ = lirpa_qgl.compute_bounds(x=(image,), method=config.abc_method)
                
                upper_gen_violation = torch.relu(torch.stack([ub_pg - pg_max, ub_qg - qg_max], dim=0))
                lower_gen_violation = torch.relu(torch.stack([pg_min - lb_pg, qg_min - lb_qg], dim=0))
                gen_violation_loss = (torch.mean(upper_gen_violation) + torch.mean(lower_gen_violation)) 
                
                # crown bounds on KCL current injections
                lb_injr, ub_injr = lirpa_injr.compute_bounds(x=(image,), method=config.abc_method)
                lb_inji, ub_inji = lirpa_inji.compute_bounds(x=(image,), method=config.abc_method)
                
                worst_case_inj_real = torch.max(lb_injr.square(), ub_injr.square())
                worst_case_inj_imag = torch.max(lb_inji.square(), ub_inji.square())

                with torch.autograd.set_detect_anomaly(True):
                    inj_violation = torch.mean(torch.sqrt(torch.clamp(worst_case_inj_real + worst_case_inj_imag, min=1e-9)))  # sum of squares
                
                if epoch % 20 == 0:
                    print(f"Average worst-case violation generator up: {upper_gen_violation.mean():.4f}")
                    print(f"Average worst-case violation generator down: {lower_gen_violation.mean():.4f}")
                    print(f"Average worst-case violation inj real: {worst_case_inj_real.mean():.4f}")
                    print(f"Average worst-case violation inj imag: {worst_case_inj_imag.mean():.4f}")
                        
            # combine
            optimizer.zero_grad()
            wc_violation = vmag_violation + imag_violation + gen_violation_loss + inj_violation # + vrect_violation
            pf_loss = loss_weight * wc_violation
            pf_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0 and epoch != 0:
                print(f"Epoch {epoch}/{config.epochs} — WC violation loss: {wc_violation:.6f}")

            
            
        if config.Algo and epoch >= 50:
            if epoch > 300 - 1:
                """ 
                The tightness of mccormick depends on the voltage and current bounds. Add w/c penalties after some epochs.
                The same goed for the nodal balance.
                
                """
                if epoch == 300 - 1:
                    # crown bounds generator active and reactive power
                    _, ub_pg = lirpa_pgu.compute_bounds(x=(image,), method=config.abc_method)
                    lb_pg, _ = lirpa_pgl.compute_bounds(x=(image,), method=config.abc_method)
                    _, ub_qg = lirpa_qgu.compute_bounds(x=(image,), method=config.abc_method)
                    lb_qg, _ = lirpa_qgl.compute_bounds(x=(image,), method=config.abc_method)
                # every 50 epochs, tighten mccormick bounds
                if epoch == 300 - 1 or epoch % 50 == 0:
                    lb_vr, ub_vr = lirpa_vr.compute_bounds(x=(image,), method=config.abc_method)
                    lb_vi, ub_vi = lirpa_vi.compute_bounds(x=(image,), method=config.abc_method)
                    lb_ir, ub_ir = lirpa_ir.compute_bounds(x=(image,), method=config.abc_method)
                    lb_ii, ub_ii = lirpa_ii.compute_bounds(x=(image,), method=config.abc_method)
            
                    global_vr_l = lb_vr.min(dim=0).values.detach() 
                    global_vr_u = ub_vr.max(dim=0).values.detach() 
                    global_vi_l = lb_vi.min(dim=0).values.detach() 
                    global_vi_u = ub_vi.max(dim=0).values.detach() 
                    global_ir_l = lb_ir.min(dim=0).values.detach() 
                    global_ir_u = ub_ir.max(dim=0).values.detach() 
                    global_ii_l = lb_ii.min(dim=0).values.detach() 
                    global_ii_u = ub_ii.max(dim=0).values.detach() 
            
                    network_gen.pinj_upper_nn.update_mccormick_bounds(
                        global_vr_l, global_vr_u,
                        global_vi_l, global_vi_u,
                        global_ir_l, global_ir_u,
                        global_ii_l, global_ii_u
                    )
                    network_gen.pinj_lower_nn.update_mccormick_bounds(
                        global_vr_l, global_vr_u,
                        global_vi_l, global_vi_u,
                        global_ir_l, global_ir_u,
                        global_ii_l, global_ii_u
                    )
                    
                    network_gen.qinj_upper_nn.update_mccormick_bounds(
                        global_vr_l, global_vr_u,
                        global_vi_l, global_vi_u,
                        global_ir_l, global_ir_u,
                        global_ii_l, global_ii_u
                    )
                    network_gen.qinj_lower_nn.update_mccormick_bounds(
                        global_vr_l, global_vr_u,
                        global_vi_l, global_vi_u,
                        global_ir_l, global_ir_u,
                        global_ii_l, global_ii_u
                    )
            
                    lirpa_vr, lirpa_vi, lirpa_ir, lirpa_ii, lirpa_iu, lirpa_vu, lirpa_vl, lirpa_pgu, lirpa_pgl, lirpa_qgu, lirpa_qgl, lirpa_injr, lirpa_inji = \
                        initialize_lirpa_modules(network_gen, torch.empty_like(sd_train).float(), device)
            
        # After some epoch, prune 50% neurons once
        # if epoch == 500:
        #     apply_incremental_pruning(network_gen, 0.5)     
                

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
    
  
def initialize_lirpa_modules(model, dummy_input, device):
    lirpa_vr    = BoundedModule(OutputWrapper(model, 0), dummy_input, device=device)
    lirpa_vi    = BoundedModule(OutputWrapper(model, 1), dummy_input, device=device)
    lirpa_ir    = BoundedModule(OutputWrapper(model, 2), dummy_input, device=device)
    lirpa_ii    = BoundedModule(OutputWrapper(model, 3), dummy_input, device=device)
    lirpa_iu    = BoundedModule(OutputWrapper(model, 4), dummy_input, device=device)
    lirpa_vu    = BoundedModule(OutputWrapper(model, 5), dummy_input, device=device)
    lirpa_vl    = BoundedModule(OutputWrapper(model, 6), dummy_input, device=device)
    lirpa_pgu   = BoundedModule(OutputWrapper(model, 7), dummy_input, device=device)
    lirpa_pgl   = BoundedModule(OutputWrapper(model, 8), dummy_input, device=device)
    lirpa_qgu   = BoundedModule(OutputWrapper(model, 9), dummy_input, device=device)
    lirpa_qgl   = BoundedModule(OutputWrapper(model, 10), dummy_input, device=device)
    lirpa_injr   = BoundedModule(OutputWrapper(model, 11), dummy_input, device=device)
    lirpa_inji   = BoundedModule(OutputWrapper(model, 12), dummy_input, device=device)
    return lirpa_vr, lirpa_vi, lirpa_ir, lirpa_ii, lirpa_iu, lirpa_vu, lirpa_vl, lirpa_pgu, lirpa_pgl, lirpa_qgu, lirpa_qgl, lirpa_injr, lirpa_inji

    
def min_max_scale_tensor(data):
    data_min = data.min(dim=0, keepdim=True).values
    data_max = data.max(dim=0, keepdim=True).values
    scaled = (data - data_min) / (data_max - data_min + 1e-8)
    return scaled, data_min, data_max


def build_network(nn_type, n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed, simulation_parameters, surrogate = None):
    hidden_layer_size = [hidden_layer_size] * n_hidden_layers
    model = NeuralNetwork(nn_type, n_input_neurons, hidden_layer_size=hidden_layer_size,
                          num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed, simulation_parameters = simulation_parameters, surrogate = surrogate)
    return model.to(device)


def normalise_network(model, sd_train, data_stat):
    sd_min = data_stat['sd_min']
    sd_delta = data_stat['sd_delta']
    vrvi_min = data_stat['vrvi_min']
    vrvi_delta = data_stat['vrvi_delta']
    
    # print(sd_min.dtype, sd_delta.dtype, vrvi_min.dtype, vrvi_delta.dtype)

    input_stats = (sd_min.reshape(-1).float(), sd_delta.reshape(-1).float())
    output_stats = (vrvi_min.reshape(-1).float(), vrvi_delta.reshape(-1).float())


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



def validate_epoch(network_gen, sd_test, vrvi_test):
    criterion = nn.MSELoss()
    output = network_gen.forward_train(sd_test)
    return criterion(output, vrvi_test)


def train_epoch(network_gen, sd_train, vrvi_train, typ, optimizer, config, simulation_parameters, epoch):
    torch.autograd.set_detect_anomaly(True)

    network_gen.train()
    criterion = nn.MSELoss()
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    num_samples = sd_train.shape[0]
    num_batches = num_samples // config.batch_size

    n_loads = sd_train.shape[1] // 2
    n_bus = simulation_parameters['general']['n_buses']
    n_lines = simulation_parameters['true_system']['n_line']
    n_gens = simulation_parameters['general']['n_gbus']
    f_bus = torch.tensor(simulation_parameters['true_system']['fbus'], dtype=torch.float32, device=vrvi_train.device)
    t_bus = torch.tensor(simulation_parameters['true_system']['tbus'], dtype=torch.float32, device=vrvi_train.device)
    # act_gen_indices = simulation_parameters['true_system']['pg_active']
    pv_buses_nz = simulation_parameters['true_system']['pv_buses_nz']
    pv_buses = simulation_parameters['true_system']['pv_buses']
    size_output = vrvi_train.shape[1]
    RELU = nn.ReLU()
    loss_sum = 0
    mse_sum = 0
    
    # get voltage bounds
    volt_max = simulation_parameters['true_system']['Volt_max'][0]
    volt_min = simulation_parameters['true_system']['Volt_min'][0]
    
    vr_vi_min = torch.full((config.batch_size, n_bus), volt_min, dtype=torch.float32, device=vrvi_train.device)
    vr_vi_max = torch.full((config.batch_size, n_bus), volt_max, dtype=torch.float32, device=vrvi_train.device)
    s_max = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float32, device=vrvi_train.device) / 100
    
    # Y bus
    Ybr_rect = torch.tensor(simulation_parameters['true_system']['Ybr_rect'], dtype=torch.float32, device=vrvi_train.device)
    Ybus = torch.tensor(simulation_parameters['true_system']['Ybus'], dtype=torch.complex128, device=vrvi_train.device)
    Ybus_real = Ybus.real.float()
    Ybus_imag = Ybus.imag.float()

    map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float32, device=vrvi_train.device)
    map_g = torch.tensor(simulation_parameters['true_system']['Map_g'], dtype=torch.float32, device=vrvi_train.device)
    kcl_im = torch.tensor(simulation_parameters['true_system']['kcl_im'], dtype=torch.float32, device=vrvi_train.device)
    bs_values = torch.tensor(simulation_parameters['true_system']['bus_bs'], dtype=torch.float32, device=vrvi_train.device).unsqueeze(1)
    kcl_from_im = torch.relu(kcl_im) # +1 at from-bus, 0 elsewhere
    kcl_to_im = -torch.relu(-kcl_im) # +1 at to-bus, 0 elsewhere
    
    # generator capacity
    sg_max = torch.tensor(simulation_parameters['true_system']['Sg_max'], dtype=torch.float32, device=vrvi_train.device) / 100
    pg_max = (sg_max.T @ map_g)[:, :n_bus]
    qg_max = (sg_max.T @ map_g)[:, n_bus:]
    qg_min = torch.tensor(simulation_parameters['true_system']['qg_min'], dtype=torch.float32) @ map_g[n_gens:, n_bus:] / 100

    # Initialize lists to accumulate current magnitudes for epoch-level metrics
    epoch_I_mag_surrogate = []
    epoch_I_mag_amb = []
    epoch_I_mag_check = []
 
    preds = []
    targets = []

    for i in range(num_batches):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        Gen_output = network_gen.forward_train(sd_train[slce])
        Gen_target = vrvi_train[slce]
        # pg_batch = pg_target[slce]
        
        # Compute magnitude from surrogate 
        Ibr =  Gen_output @ Ybr_rect.T 
        
        # first half is 'from' currents
        I_f = Ibr[:, :2 * n_lines]
        Ir_f = I_f[:, :n_lines]
        Ii_f = I_f[:, n_lines:2*n_lines]
        I_mag_f = torch.sqrt(Ir_f**2 + Ii_f**2)

        # second half is 'to' currents
        I_t = Ibr[:, 2 * n_lines:]
        Ir_t = I_t[:, :n_lines]
        Ii_t = I_t[:, n_lines:]
        I_mag_t = torch.sqrt(Ir_t**2 + Ii_t**2)
        
        I_mag_check = torch.cat((I_mag_f, I_mag_t), dim=1)

        # Compute over approximated current with a-max b-min for comparison with ground truth
        I_mag_f_amb = network_gen.amb_magnitude_current(I_f.detach())
        I_mag_t_amb = network_gen.amb_magnitude_current(I_t.detach())
        I_mag_amb = torch.cat((I_mag_f_amb, I_mag_t_amb), dim=1)

        # Accumulate for epoch-level metrics
        epoch_I_mag_amb.append(I_mag_amb.detach())
        epoch_I_mag_check.append(I_mag_check.detach())
        
        # ========== add loss term for KCL at each node ===========
        vr = Gen_output[:, :n_bus]
        vi = Gen_output[:, n_bus:]
        
        Ir = torch.matmul(vr, Ybus_real.T) - torch.matmul(vi, Ybus_imag.T)
        Ii = torch.matmul(vr, Ybus_imag.T) + torch.matmul(vi, Ybus_real.T)
        
        I_shunt_r = -bs_values * vi.T 
        I_shunt_i = bs_values * vr.T
        
        
        if config.kcl_weight != 0:            
            current_inj_real = torch.matmul(kcl_from_im, Ir_f.T) - torch.matmul(kcl_to_im, Ir_t.T)
            current_inj_imag = torch.matmul(kcl_from_im, Ii_f.T) - torch.matmul(kcl_to_im, Ii_t.T)

            delta_inj_real = Ir - current_inj_real.T - I_shunt_r.T
            delta_inj_imag = Ii - current_inj_imag.T - I_shunt_i.T
            
            kcl_loss = torch.mean(delta_inj_real ** 2 + delta_inj_imag ** 2)
            kcl_loss = config.kcl_weight * kcl_loss
                    
        # ========= generator capacity violation ==========
        if config.pg_viol_weight != 0:
            
            pinj = vr * Ir + vi * Ii
            qinj = vi * Ir - vr * Ii
              
            pd = sd_train[slce][:, :n_loads]
            qd = sd_train[slce][:, n_loads:]
            
            pg = pinj + torch.einsum('bi,ij->bj', pd, map_l[:n_loads, :n_bus])
            qg = qinj + torch.einsum('bi,ij->bj', qd, map_l[n_loads:, n_bus:])
            
            upper_gen_violation = RELU(torch.stack([pg - pg_max, qg - qg_max], dim=0))
            lower_gen_violation = RELU(torch.stack([-pg, qg_min - qg], dim=0))
            
            pg_pred = pg[:, pv_buses_nz]
            
            # mse_gen = criterion(pg_pred, pg_batch)  
            
            gen_violation_loss = config.pg_viol_weight * (
                torch.mean(upper_gen_violation ** 2) + torch.mean(lower_gen_violation ** 2)
            ) # + mse_gen               
            
        else:
            gen_violation_loss = 0
        
        # supervised learning, difference from label
        if config.crit_weight != 0:
            mse_criterion = criterion(Gen_output * typ[slce], Gen_target * typ[slce])           
            
            loss_gen = config.crit_weight * mse_criterion      
        else:
            loss_gen = 0
                   
         # voltage violation
        if config.vm_viol_weight != 0:
            volt_magn = torch.square(vr**2 + vi**2)
            upper_volt_violation = RELU(volt_magn - vr_vi_max)
            lower_volt_violation = RELU(vr_vi_min - volt_magn)

            voltage_violation_loss = config.vm_viol_weight * (
                torch.mean(upper_volt_violation ** 2) + torch.mean(lower_volt_violation ** 2)
            )
        else:
            voltage_violation_loss = 0
        
        # Penalize line flow violations
        if config.line_viol_weight != 0:
            V_complex = vr + 1j * vi
            
            # Compute branch currents
            I_complex_f = Ir_f + 1j * Ii_f  # (batch_size, n_lines)
            I_complex_t = Ir_t + 1j * Ii_t  # (batch_size, n_lines)

            # Voltages at from and to buses
            V_from = V_complex[:, f_bus.long()]  # shape: (batch_size, n_bus)
            V_to = V_complex[:, t_bus.long()]    # shape: (batch_size, n_bus)

            # Complex power flows S = V * conj(I)
            S_from = V_from * torch.conj(I_complex_f)  # shape: (batch_size, n_lines)
            S_to = V_to * torch.conj(I_complex_t)      # shape: (batch_size, n_lines)
            
            S_mag_f = torch.abs(S_from)  # (batch_size, n_lines)
            S_mag_t = torch.abs(S_to)

            # Compute violations
            S_viol_f = torch.clamp(S_mag_f - s_max, min=0.0)
            S_viol_t = torch.clamp(S_mag_t - s_max, min=0.0)

            # Max violation per sample
            S_viol = torch.mean(torch.stack([S_viol_f.mean(dim=1), S_viol_t.mean(dim=1)], dim=1), dim=1)
            
            flow_violation_loss = config.line_viol_weight * torch.mean(S_viol)
        else:
            flow_violation_loss = 0

        total_loss = loss_gen + voltage_violation_loss + gen_violation_loss + flow_violation_loss + kcl_loss

        total_loss.backward()
        optimizer.step()
        loss_sum += total_loss
        mse_sum += mse_criterion
        
    # Concatenate all accumulated tensors for epoch-level metrics
    epoch_I_mag_amb = torch.cat(epoch_I_mag_amb)
    epoch_I_mag_check = torch.cat(epoch_I_mag_check)
    
    # Compute epoch-level metrics
    mae_target, rmse_amb_epoch, not_ub_pct_amb_epoch, amb_avg_over_pct, amb_max_over_pct = compute_metrics(epoch_I_mag_amb, epoch_I_mag_check)
    
    if epoch % 50 == 0 and epoch != 0 and i == 1:
        print(f"KCL loss: {kcl_loss}, with weight: {config.kcl_weight}")
        print(f"Gen loss: {gen_violation_loss}, with weight: {config.pg_viol_weight}")
        print(f"MSE loss: {mse_criterion}, with weight: {config.crit_weight}")
        print(f"Voltage loss: {voltage_violation_loss}, with weight: {config.vm_viol_weight}")
        print(f"Line loss: {flow_violation_loss}, with weight: {config.line_viol_weight}")
        
    #     print(f"  [Over approximation]  target: {mae_target:.4f},   RMSE: {rmse_amb_epoch:.4f},   Not Upper Bound: {not_ub_pct_amb_epoch:.2f}%,   Avg Overapprox: {amb_avg_over_pct:.2f}%,   Max Overapprox: {amb_max_over_pct:.2f}%")
    #     print("\n")
    
    loss_epoch = loss_sum / num_batches
    mse_epoch = mse_sum / num_batches

    return mse_epoch, loss_epoch

    

def compute_metrics(pred, true):
    target = torch.mean(true).item()
    rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
    not_upper_bound_pct = 100.0 * torch.mean((pred < true).float()).item()

    # Compute percentage over-approximation (only when pred >= true to avoid negatives)
    over_approx_pct = ((pred - true) / true) * 100.0
    avg_over_pct = torch.mean(over_approx_pct).item()
    max_over_pct = torch.max(over_approx_pct).item()

    return target, rmse, not_upper_bound_pct, avg_over_pct, max_over_pct



def wc_enriching(network_gen, config, sd_train, data_stat, simulation_parameters):
    n_adver = config.N_enrich

    # Forward pass to get generator outputs (Pg and Vm stacked)
    nn_output = network_gen.forward_aft(sd_train).cpu().detach()# .numpy()
    Ybr_rect = torch.tensor(simulation_parameters['true_system']['Ybr_rect'], dtype=torch.float32, device=nn_output.device)
    f_bus = torch.tensor(simulation_parameters['true_system']['fbus'], dtype=torch.float32, device=nn_output.device)
    t_bus = torch.tensor(simulation_parameters['true_system']['tbus'], dtype=torch.float32, device=nn_output.device)
    s_max = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float32, device=nn_output.device) / 100
    
    n_bus = nn_output.shape[1] // 2 
    n_loads = sd_train.shape[1] // 2  # Pd and Qd stacked, so loads = half of input features

    # Split output
    vr = nn_output[:, :n_bus]
    vi = nn_output[:, n_bus:]
    
    V_complex = vr + 1j * vi  # (batch_size, n_bus)

    # Compute branch currents
    Ibr = nn_output @ Ybr_rect.T  # shape: (batch_size, 4 * n_lines) if from-to current format

    n_lines = Ibr.shape[1] // 4

    # Extract from-end and to-end currents
    Ir_f = Ibr[:, :n_lines]
    Ii_f = Ibr[:, n_lines:2 * n_lines]
    Ir_t = Ibr[:, 2 * n_lines:3 * n_lines]
    Ii_t = Ibr[:, 3 * n_lines:]

    I_complex_f = Ir_f + 1j * Ii_f  # (batch_size, n_lines)
    I_complex_t = Ir_t + 1j * Ii_t  # (batch_size, n_lines)

    # Voltages at from and to buses
    V_from = V_complex[:, f_bus.long()]  # shape: (batch_size, n_bus)
    V_to = V_complex[:, t_bus.long()]    # shape: (batch_size, n_bus)

    # Complex power flows S = V * conj(I)
    S_from = V_from * torch.conj(I_complex_f)  # shape: (batch_size, n_lines)
    S_to = V_to * torch.conj(I_complex_t)      # shape: (batch_size, n_lines)
    
    S_mag_f = torch.abs(S_from)  # (batch_size, n_lines)
    S_mag_t = torch.abs(S_to)

    # Compute worst-case violations
    S_viol_f = torch.clamp(S_mag_f - s_max, min=0.0)
    S_viol_t = torch.clamp(S_mag_t - s_max, min=0.0)

    # Max violation per sample
    S_viol = torch.max(torch.stack([S_viol_f.max(dim=1).values, S_viol_t.max(dim=1).values], dim=1), dim=1).values  # shape: (batch_size,)

    # Select worst-case indices
    topk = torch.topk(S_viol, n_adver, largest=True)
    worst_idx = topk.indices.cpu().numpy()

    # Apply gradient ascent to selected inputs
    adv_samples = GradAscnt(network_gen, sd_train[worst_idx].cpu().numpy(), simulation_parameters, sign=1)

    x_g = torch.tensor(adv_samples).float()
    y_g = torch.zeros(x_g.shape[0], nn_output.shape[1])  # placeholder targets
    y_type = torch.zeros(x_g.shape[0], 1)

    return x_g, y_g, y_type





def GradAscnt(Network, x_starting, simulation_parameters, sign=1, Num_iteration=100, lr=1e-5):
    '''
    x_starting: Starting points for gradient ascent (shape: batch_size x features)
    sign: 1 to increase violation, -1 to decrease violation
    '''
    x = torch.tensor(x_starting, requires_grad=True).float()
    optimizer = torch.optim.SGD([x], lr=lr)

    # Load simulation parameters from data_stat or pass separately as needed
    # You might need to adapt this depending on where you keep Ybr_rect, fbus, tbus, s_max
    Ybr_rect = torch.tensor(simulation_parameters['true_system']['Ybr_rect'], dtype=torch.float32, device=x.device)
    f_bus = torch.tensor(simulation_parameters['true_system']['fbus'], dtype=torch.long, device=x.device)
    t_bus = torch.tensor(simulation_parameters['true_system']['tbus'], dtype=torch.long, device=x.device)
    s_max = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float32, device=x.device) / 100
    
    sd_min = torch.tensor(simulation_parameters['true_system']['Sd_min']).float().to(device)
    sd_delta = torch.tensor(simulation_parameters['true_system']['Sd_delta']).float().to(device)
    
    min_bound = sd_min.view(1, -1)        
    max_bound = (sd_min + sd_delta).view(1, -1)

    n_lines = Ybr_rect.shape[0] // 4

    for _ in range(Num_iteration):
        optimizer.zero_grad()
        nn_output = Network.forward_aft(x)  # shape: (batch_size, output_size)
        
        n_bus = nn_output.shape[1] // 2

        vr = nn_output[:, :n_bus]
        vi = nn_output[:, n_bus:]
        V_complex = vr + 1j * vi

        # Branch currents
        Ibr = nn_output @ Ybr_rect.T
        Ir_f = Ibr[:, :n_lines]
        Ii_f = Ibr[:, n_lines:2 * n_lines]
        Ir_t = Ibr[:, 2 * n_lines:3 * n_lines]
        Ii_t = Ibr[:, 3 * n_lines:]

        I_complex_f = Ir_f + 1j * Ii_f
        I_complex_t = Ir_t + 1j * Ii_t

        V_from = V_complex[:, f_bus]
        V_to = V_complex[:, t_bus]

        S_from = V_from * torch.conj(I_complex_f)
        S_to = V_to * torch.conj(I_complex_t)

        S_mag_f = torch.abs(S_from)
        S_mag_t = torch.abs(S_to)

        # Calculate violations beyond limits
        S_viol_f = torch.clamp(S_mag_f - s_max, min=0.0)
        S_viol_t = torch.clamp(S_mag_t - s_max, min=0.0)

        # Aggregate violation (mean or max across lines and batch)
        # Using mean violation here; you can use max if preferred
        violation = torch.mean(S_viol_f + S_viol_t)

        # Loss tries to increase violation when sign=1
        loss = -sign * violation

        loss.backward()
        optimizer.step()
        
        x.data = torch.clamp(x.data, min=min_bound, max=max_bound)

        
    # print("this is the loss: ", loss)

    return x.detach().cpu().numpy()



def compute_flow_violation(surrogate, nn_output, simulation_parameters):
    
    # general params
    batch_size = nn_output.shape[0]
    slack_idx = simulation_parameters['true_system']['slack_bus']
    n_bus = simulation_parameters['general']['n_buses'] 
    n_line = simulation_parameters['true_system']['n_line']  
    
    # get line parameters
    fbus = torch.tensor(simulation_parameters['true_system']['fbus'], dtype=torch.long)
    tbus = torch.tensor(simulation_parameters['true_system']['tbus'], dtype=torch.long)
    g_l = torch.tensor(simulation_parameters['true_system']['g'], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    b_l = torch.tensor(simulation_parameters['true_system']['b'], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    s_max = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float32) / 100
    s_lim = s_max
    
    # extract rect voltages (excluding slack)
    vr_nn = nn_output[:, :n_bus - 1]
    vi_nn = nn_output[:, n_bus - 1:]
    
    # insert slack voltages at correct position
    vr = torch.cat([vr_nn[:, :slack_idx], torch.ones(batch_size, 1), vr_nn[:, slack_idx:]], dim=1)
    vi = torch.cat([vi_nn[:, :slack_idx], torch.zeros(batch_size, 1), vi_nn[:, slack_idx:]], dim=1)
    
    # get vectorized rectangular voltages
    vr_f = vr[:, fbus]  # shape (batch, n_lines)
    vi_f = vi[:, fbus]
    vr_t = vr[:, tbus]
    vi_t = vi[:, tbus]
    
    # prepare surrogate input: shape (batch * n_line, 6)
    surrogate_input = torch.stack([vr_f, vi_f, vr_t, vi_t, g_l, b_l], dim=2)
    surrogate_input = surrogate_input.reshape(-1, 6)
    
    # Extract normalized input
    out = normalize_surrogate_data(simulation_parameters, vrect=surrogate_input)
    surrogate_input_norm = out['vrect_scaled']

    # run surrogate model
    flows_surrogate = surrogate.forward_train(surrogate_input_norm)  # shape: (batch * n_line, 6)
    
    # extract Sf and St magnitudes (first and fourth outputs)
    sf = flows_surrogate[:, 0].reshape(batch_size, n_line) * (2 * s_max) - s_max
    st = flows_surrogate[:, 3].reshape(batch_size, n_line) * (2 * s_max) - s_max
    s_mag = torch.maximum(torch.abs(sf), torch.abs(st)) 
        
    # calculate violations
    s_lim = s_lim.unsqueeze(0).to(s_mag.device)
    violations = torch.relu(s_mag - s_lim)  # shape (batch, n_line)
    line_loss = violations.mean()
    
    line_loading = s_mag / s_lim
    avg_line_loading = line_loading.mean().item() * 100
    max_line_loading = line_loading.max().item() * 100

    # print(f"\n✅ Average line loading: {avg_line_loading:.2f}%")
    # print(f"🔥 Maximum line loading: {max_line_loading:.2f}%")
    # stop

    return line_loss



def normalize_surrogate_data(simulation_parameters, vrect=None, sfl=None):
    """
    Normalize vrect and sfl per instance, using per-line parameters.

    Args:
        simulation_parameters (dict): Contains line-wise limits like b, g, L_limit.
        vrect (torch.Tensor, optional): Input tensor of shape (n_samples, 6).
        sfl (torch.Tensor, optional): Output tensor of shape (n_samples, 6).

    Returns:
        Dict containing:
            - vrect_scaled (optional)
            - sfl_scaled (optional)
            - vrect_min, vrect_delta (if vrect given)
            - sfl_min, sfl_delta (if sfl given)
    """
    results = {}

    n_lines = len(simulation_parameters['true_system']['b'])

    b_vals = torch.tensor(simulation_parameters['true_system']['b'], dtype=torch.float32)
    g_vals = torch.tensor(simulation_parameters['true_system']['g'], dtype=torch.float32)
    l_limits = torch.tensor(simulation_parameters['true_system']['L_limit'], dtype=torch.float32).T / 100

    if vrect is not None:
        n_samples = vrect.shape[0]

        # Global min/max for g, b
        b_min_val, b_max_val = torch.min(b_vals), torch.max(b_vals)
        g_min_val, g_max_val = torch.min(g_vals), torch.max(g_vals)

        b_min_tensor = torch.full((n_samples, 1), b_min_val, dtype=torch.float32, device=vrect.device)
        b_max_tensor = torch.full((n_samples, 1), b_max_val, dtype=torch.float32, device=vrect.device)
        g_min_tensor = torch.full((n_samples, 1), g_min_val, dtype=torch.float32, device=vrect.device)
        g_max_tensor = torch.full((n_samples, 1), g_max_val, dtype=torch.float32, device=vrect.device)

        repeats_per_line = n_samples // n_lines
        l_limits_exp = l_limits.repeat_interleave(repeats_per_line).unsqueeze(1)

        volt_max = simulation_parameters['true_system']['Volt_max'][0]
        volt_min = -volt_max

        vr_vi_min = torch.full((n_samples, 4), volt_min, dtype=torch.float32, device=vrect.device)
        vr_vi_max = torch.full((n_samples, 4), volt_max, dtype=torch.float32, device=vrect.device)

        vrect_min = torch.cat([vr_vi_min, g_min_tensor, b_min_tensor], dim=1)
        vrect_max = torch.cat([vr_vi_max, g_max_tensor, b_max_tensor], dim=1)
        vrect_delta = vrect_max - vrect_min
        vrect_delta[vrect_delta <= 1e-12] = 1.0

        vrect_scaled = (vrect - vrect_min) / vrect_delta

        results.update({
            'vrect_scaled': vrect_scaled,
            'vrect_min': vrect_min,
            'vrect_delta': vrect_delta,
        })

    if sfl is not None:
        n_samples = sfl.shape[0]
        repeats_per_line = n_samples // n_lines
        l_limits_exp = l_limits.repeat_interleave(repeats_per_line).unsqueeze(1)

        sfl_min = -l_limits_exp.repeat(1, 6)
        sfl_max = l_limits_exp.repeat(1, 6)
        sfl_delta = sfl_max - sfl_min
        sfl_delta[sfl_delta <= 1e-12] = 1.0

        sfl_scaled = (sfl - sfl_min) / sfl_delta

        results.update({
            'sfl_scaled': sfl_scaled,
            'sfl_min': sfl_min,
            'sfl_delta': sfl_delta,
        })

    return results


def denormalize_surrogate_data(sfl_scaled, sfl_min, sfl_delta):
    return sfl_scaled * sfl_delta + sfl_min



