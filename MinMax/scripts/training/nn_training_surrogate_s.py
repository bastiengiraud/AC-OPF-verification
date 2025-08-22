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

from surrogate.create_example_parameters import create_example_parameters
from surrogate.create_data import create_data, create_test_data
from EarlyStopping import EarlyStopping
from neural_network.lightning_nn_surrogate import SurrogateModel
# from LiRPANet import LiRPANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_np(x):
    return x.detach().numpy()


def train(config):
    n_buses = config.test_system
    simulation_parameters = create_example_parameters(n_buses)
    n_lines = simulation_parameters['true_system']['n_line'] 

    # === Raw Training Data ===
    vrect_train_init, sfl_train_init = create_data(simulation_parameters=simulation_parameters)
    vrect_train_init = torch.tensor(vrect_train_init).float().to(device)
    sfl_train_init = torch.tensor(sfl_train_init).float().to(device)
    Gen_train_typ = torch.ones(sfl_train_init.shape[0], 1).to(device)
    num_classes = sfl_train_init.shape[1]
    
    # ==== check data loaded aligned for appropriate scaling ===
    first_sample = vrect_train_init[0, -2:]  # shape (2,)
    b_first = torch.tensor(simulation_parameters['true_system']['b'][0], dtype=torch.float32)
    g_first = torch.tensor(simulation_parameters['true_system']['g'][0], dtype=torch.float32)
    true_first = torch.stack([g_first, b_first])  # shape (2,)
    
    last_sample = vrect_train_init[n_lines - 1, -2:]  # shape (2,)
    b_last = torch.tensor(simulation_parameters['true_system']['b'][n_lines - 1], dtype=torch.float32)
    g_last = torch.tensor(simulation_parameters['true_system']['g'][n_lines - 1], dtype=torch.float32)
    true_last = torch.stack([g_last, b_last])  # shape (2,)

    # Check if they are close (allow small numerical tolerance)
    errors = []
    if not torch.allclose(first_sample, true_first, atol=1e-4):
        errors.append(f"First sample g,b {first_sample.tolist()} does NOT match true values {true_first.tolist()}")
    if not torch.allclose(last_sample, true_last, atol=1e-4):
        errors.append(f"Last sample g,b {last_sample.tolist()} does NOT match true values {true_last.tolist()}")

    if errors:
        raise ValueError("Data alignment check failed:\n" + "\n".join(errors))
    else:
        print("✅ First and last sample g,b match true line parameters.")

    # === Split into Proper Multiple of n_lines ===
    n_train_total = vrect_train_init.shape[0]
    n_train_aligned = (n_train_total // n_lines) * n_lines
    n_train_leftover = n_train_total - n_train_aligned

    # Split aligned and leftover training samples
    vrect_train_aligned = vrect_train_init[:n_train_aligned]
    sfl_train_aligned = sfl_train_init[:n_train_aligned]

    vrect_leftover = vrect_train_init[n_train_aligned:]
    sfl_leftover = sfl_train_init[n_train_aligned:]
    
    # ========== track line indices ==========
    n_samples_train = vrect_train_aligned.shape[0] // n_lines
    line_indices_train = torch.arange(n_lines).repeat(n_samples_train)  # shape (n_samples * n_lines,)

    # === Normalize Training Data ===
    out = normalize_surrogate_data(simulation_parameters, vrect=vrect_train_aligned, sfl=sfl_train_aligned)
    
    vrect_train = out.get('vrect_scaled', None)
    sfl_train = out.get('sfl_scaled', None)
    vrect_min = out.get('vrect_min', None)
    vrect_delta = out.get('vrect_delta', None)
    sfl_min = out.get('sfl_min', None)
    sfl_delta = out.get('sfl_delta', None)

    # Store normalization info
    data_stat = {}
    if vrect_min is not None:
        data_stat['vrect_min'] = vrect_min
        data_stat['vrect_delta'] = vrect_delta
    if sfl_min is not None:
        data_stat['sfl_min'] = sfl_min
        data_stat['sfl_delta'] = sfl_delta

    # === Test Data ===
    vrect_test_init, sfl_test_init = create_test_data(simulation_parameters=simulation_parameters)
    vrect_test_init = torch.tensor(vrect_test_init).float().to(device)
    sfl_test_init = torch.tensor(sfl_test_init).float().to(device)

    # Append leftover training samples to test set
    if n_train_leftover > 0:
        vrect_test_init = torch.cat([vrect_leftover, vrect_test_init], dim=0)
        sfl_test_init = torch.cat([sfl_leftover, sfl_test_init], dim=0)
        
    # ==== check data loaded aligned for appropriate scaling ===
    first_sample = vrect_test_init[0, -2:]  # shape (2,)
    b_first = torch.tensor(simulation_parameters['true_system']['b'][0], dtype=torch.float32)
    g_first = torch.tensor(simulation_parameters['true_system']['g'][0], dtype=torch.float32)
    true_first = torch.stack([g_first, b_first])  # shape (2,)

    last_sample = vrect_test_init[n_lines - 1, -2:]  # shape (2,)
    b_last = torch.tensor(simulation_parameters['true_system']['b'][n_lines - 1], dtype=torch.float32)
    g_last = torch.tensor(simulation_parameters['true_system']['g'][n_lines - 1], dtype=torch.float32)
    true_last = torch.stack([g_last, b_last])  # shape (2,)

    # Check if they are close (allow small numerical tolerance)
    errors = []
    if not torch.allclose(first_sample, true_first, atol=1e-4):
        errors.append(f"First sample g,b {first_sample.tolist()} does NOT match true values {true_first.tolist()}")
    if not torch.allclose(last_sample, true_last, atol=1e-4):
        errors.append(f"Last sample g,b {last_sample.tolist()} does NOT match true values {true_last.tolist()}")

    if errors:
        raise ValueError("Data alignment check failed:\n" + "\n".join(errors))
    else:
        print("✅ First and last sample g,b match true line parameters.")

    # === Normalize Test Data with training stats ===
    len_test_data = vrect_test_init.shape[0]
    vrect_test = (vrect_test_init - vrect_min[:len_test_data, :]) / vrect_delta[:len_test_data, :]
    sfl_test = (sfl_test_init - sfl_min[:len_test_data, :]) / sfl_delta[:len_test_data, :]
    
    # ========== track line indices ==========
    n_samples_test = vrect_test.shape[0] // n_lines
    line_indices_test = torch.arange(n_lines).repeat(n_samples_test)  # shape (n_samples * n_lines,)

    # build network
    network_gen = build_network(vrect_train.shape[1], num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed)
    # network_gen = normalise_network(network_gen, vrect_train, data_stat) # data is already normalized

    optimizer = torch.optim.Adam(network_gen.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) ** -config.lr_decay)

    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_save_directory = os.path.join(project_root_dir, 'models', 'surrogate')
    
    # ✅ Create the directory if it doesn't exist
    os.makedirs(model_save_directory, exist_ok=True)

    path = f'checkpoint_surrogate_{n_buses}.pt'
    path_dir = os.path.join(model_save_directory, path)
    
    early_stopping = EarlyStopping(patience=100, verbose=False, NN_input=vrect_train, path=path_dir)
    

    train_losses = []
    test_losses = []

    for epoch in range(config.epochs):
        # after every 100 epochs, enrich dataset with worst-case data
        if epoch % 100 == 0 and epoch != 0 and config.Enrich:
            X, Y, typ = wc_enriching(network_gen, config, vrect_train, data_stat)
            InputNN = torch.cat((vrect_train, X), 0).to(device)
            OutputNN = torch.cat((sfl_train, Y), 0).to(device)
            typNN = torch.cat((Gen_train_typ, typ), 0).to(device)
            
            # shuffle data
            idx = torch.randperm(InputNN.shape[0])
            InputNN, OutputNN, typNN = InputNN[idx], OutputNN[idx], typNN[idx]
        else:
            InputNN = vrect_train
            OutputNN = sfl_train
            typNN = Gen_train_typ
            line_indices_all = line_indices_train
            
            # shuffle data
            idx = torch.randperm(InputNN.shape[0])
            InputNN = InputNN[idx]
            OutputNN = OutputNN[idx]
            typNN = typNN[idx]
            line_indices_all = line_indices_all[idx]
            
            # Shuffle test data once before validation, inside epoch loop if you want
            idx_test = torch.randperm(vrect_test.shape[0])
            # vrect_test_shuffled = vrect_test[idx_test]
            # sfl_test_shuffled = sfl_test[idx_test]
            # line_indices_test_shuffled = line_indices_test[idx_test]

        start_time = time.time()
        mse_criterion, training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN, optimizer, config, line_indices_all, simulation_parameters, epoch)
        validation_loss = validate_epoch(network_gen, vrect_test, sfl_test, simulation_parameters, line_indices_test)
        training_time = time.time() - start_time
        
        if epoch % 50 == 0 and epoch != 0:
            # Print MSE losses for both train and validation
            print(f"Epoch {epoch+1}/{config.epochs} — Train total loss: {training_loss:.6f}, Train MSE: {mse_criterion:.6f}, Validation MSE: {validation_loss:.6f}")

        train_losses.append(mse_criterion.item())
        test_losses.append(validation_loss.item())

        early_stopping(validation_loss, network_gen)

        # loss_weight = config.LPF_weight / (1 + epoch * 0.01)
        # lb, ub = lirpa_model.compute_bounds(x=(image,), method=config.abc_method) 
        # PF_violation = torch.abs(ub) + torch.abs(lb)

        # # after 100 epochs, start adding wc PF violation penalty
        # if config.Algo and epoch >= 100:
        #     optimizer.zero_grad()
        #     pf_loss = loss_weight * PF_violation 
        #     pf_loss.backward()
        #     optimizer.step()

        scheduler.step()

        if early_stopping.early_stop:
            break
    
    # store as .h5 model
    best_model = torch.load(path_dir)
    early_stopping.export_to_h5(best_model)
    
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
    
def min_max_scale_tensor(data):
    data_min = data.min(dim=0, keepdim=True).values
    data_max = data.max(dim=0, keepdim=True).values
    scaled = (data - data_min) / (data_max - data_min + 1e-8)
    return scaled, data_min, data_max


def build_network(n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    hidden_layer_size = [hidden_layer_size] * n_hidden_layers
    model = SurrogateModel(n_input_neurons, hidden_layer_size=hidden_layer_size,
                          num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed)
    return model.to(device)


def normalise_network(model, vrect_train, data_stat):
    vrect_min = data_stat['vrect_min']
    vrect_delta = data_stat['vrect_delta']
    sfl_min = data_stat['sfl_min']
    sfl_delta = data_stat['sfl_delta']

    input_stats = (vrect_min.reshape(-1).float(), vrect_delta.reshape(-1).float())
    output_stats = (sfl_min.reshape(-1).float(), sfl_delta.reshape(-1).float())



    model.normalise_input(input_stats)
    model.normalise_output(output_stats)
    return model.to(device)


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
        
        # Line loading in %
        s_mag = torch.maximum(sfl[:, 0], sfl[:, 3])  # assuming columns 0 and 3 are |Sf| and |St|
        s_max = l_limits_exp.squeeze(1)  # shape (n_samples,)
        line_loading_pct = 100 * s_mag / s_max
        average_line_loading = line_loading_pct.mean().item()

        print(f"Average line loading: {average_line_loading:.2f}%")

        results.update({
            'sfl_scaled': sfl_scaled,
            'sfl_min': sfl_min,
            'sfl_delta': sfl_delta,
        })

    return results



def denormalize_surrogate_data(sfl_scaled, sfl_min, sfl_delta):
    return sfl_scaled * sfl_delta + sfl_min



def validate_epoch(network_gen, vrect_test, Gen_test, simulation_parameters, line_indices_test):
    criterion = nn.MSELoss()
    network_gen.eval()
    
    # get line limits
    l_limits = torch.tensor(simulation_parameters['true_system']['L_limit'][0])
    
    with torch.no_grad():
        output = network_gen.forward_train(vrect_test)
        line_indices_batch = line_indices_test
        
        # Get per-line limits
        limits_batch = l_limits[line_indices_batch]
        sfl_delta = (2 * limits_batch).unsqueeze(1)

        # # Optional: convert to physical units
        # surrogate_flows = output * sfl_delta - limits_batch.unsqueeze(1)
        # target_flows = Gen_test * sfl_delta - limits_batch.unsqueeze(1)
        
    return criterion(output, Gen_test)


def train_epoch(network_gen, vrect_train, Gen_train, typ, optimizer, config, line_indices_train, simulation_parameters, epoch):
    torch.autograd.set_detect_anomaly(True)

    network_gen.train()
    criterion = nn.MSELoss()
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    num_samples = vrect_train.shape[0]
    num_batches = num_samples // config.batch_size
    
    # get line limits
    l_limits = torch.tensor(simulation_parameters['true_system']['L_limit'][0])

    RELU = nn.ReLU()
    loss_sum = 0

    preds = []
    targets = []


    for i in range(num_batches):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        line_indices_batch = line_indices_train[slce]
        Gen_output = network_gen.forward_train(vrect_train[slce])
        Gen_target = Gen_train[slce]
        
        # Get per-line limits
        limits_batch = l_limits[line_indices_batch]
        sfl_delta = (2 * limits_batch).unsqueeze(1)
        sfl_min = -limits_batch.unsqueeze(1)

        # Optional: convert to physical units
        surrogate_flows = Gen_output * sfl_delta + sfl_min
        target_flows = Gen_target * sfl_delta + sfl_min
                
        # surrogate violation
        upper_violation = RELU(surrogate_flows - limits_batch.unsqueeze(1)) 
        lower_violation = RELU(-limits_batch.unsqueeze(1) - surrogate_flows)
        
        upper_selected = upper_violation[:, [0, 3]] # only penalize apparent flows
        lower_selected = lower_violation[:, [0, 3]]

        surrogate_violation_loss = config.sf_viol_weight * (
            torch.mean(upper_selected ** 2) + torch.mean(lower_selected ** 2)
        )
        
        # mse of NN outputs
        mse_criterion = criterion(Gen_output * typ[slce], Gen_target * typ[slce])
        loss_gen = config.crit_weight * mse_criterion                 # supervised learning, difference from label
        
        total_loss = loss_gen + surrogate_violation_loss

        total_loss.backward()
        optimizer.step()
        loss_sum += loss_gen
    
    loss_epoch = loss_sum / num_batches

    return mse_criterion, loss_epoch


"""" 
convex relaxation of the OP for power flow.

"""





def wc_enriching(network_gen, config, vrect_train, Data_stat):
    n_adver = config.N_enrich

    # Forward pass to get generator outputs (Pg and Vm stacked)
    nn_output = network_gen.forward_aft(vrect_train).cpu().detach().numpy()
    
    n_gens = Data_stat['Gen_delta'].shape[0] // 2 
    n_loads = vrect_train.shape[1] // 2  # Pd and Qd stacked, so loads = half of input features

    # Split loads into Pd and Qd
    P_gen = nn_output[:, :n_gens]
    P_load = vrect_train[:, :n_loads].cpu().numpy()  # shape (n_loads, n_samples)

    # Only consider active power balance violation (Pg - Pd)
    # Sum over generators and loads per sample (axis=0)
    PB_P = np.sum(P_gen, axis=1) - np.sum(P_load, axis=1)  # shape (n_samples,)

    # Find indices with largest positive violations (overgeneration)
    ind_p = np.argpartition(PB_P, -n_adver // 2)[-n_adver // 2:]
    adv_p = GradAscnt(network_gen, vrect_train[ind_p, :].cpu().numpy(), Data_stat, sign=1)

    # Find indices with largest negative violations (undergeneration)
    ind_n = np.argpartition(-PB_P, -n_adver // 2)[-n_adver // 2:]
    adv_n = GradAscnt(network_gen, vrect_train[ind_n, :].cpu().numpy(), Data_stat, sign=-1)

    x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()  # concatenate on sample axis
    y_g = torch.zeros(x_g.shape[0], nn_output.shape[1])
    y_type = torch.zeros(x_g.shape[0], 1)

    return x_g, y_g, y_type





def GradAscnt(Network, x_starting, Data_stat, sign=-1, Num_iteration=100, lr=0.0001):
    '''
    x_starting: Starting points for gradient ascent (shape: features x batch_size)
    sign: 1 to increase violation, -1 to decrease violation
    '''
    x = torch.tensor(x_starting, requires_grad=True).float()
    optimizer = torch.optim.SGD([x], lr=lr)

    n_gens = Data_stat['Gen_delta'].shape[0] // 2 
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



# def wc_enriching(network_gen, config, vrect_train, Data_stat):
#     n_adver = config.N_enrich
#     Gen_output = network_gen.forward_aft(vrect_train).cpu().detach().numpy() # forward pass with clamping
#     PB = np.sum(Gen_output, axis=1) - np.sum(vrect_train.cpu().numpy(), axis=1) # power balance violation
    
#     # over generation - positive gradient ascent samples
#     ind_p = np.argpartition(PB, -4)[-n_adver // 2:] # identify 'n_adver' indices with worst-case violation
#     adv_p = GradAscnt(network_gen, vrect_train[ind_p].cpu().numpy(), Data_stat, sign=1)
    
#     # under generation - negative gradient descent samples
#     ind_n = np.argpartition(-PB, -4)[-n_adver // 2:]
#     adv_n = GradAscnt(network_gen, vrect_train[ind_n].cpu().numpy(), Data_stat)
#     x_g = torch.tensor(np.concatenate([adv_n, adv_p], axis=0)).float()
#     y_g = torch.zeros(x_g.shape[0], Gen_output.shape[1])
#     y_type = torch.zeros(x_g.shape[0], 1)
#     return x_g, y_g, y_type




# def GradAscnt(Network, x_starting, Data_stat, sign=-1, Num_iteration=100, lr=0.0001):
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