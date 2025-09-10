
import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl, torch, torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np


import os
import sys

# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, ROOT_DIR)

for subdir in ['MinMax/data', 'MinMax/models']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from neural_network.lightning_nn_crown import NeuralNetwork
from ac_opf.create_example_parameters import create_example_parameters
from types import SimpleNamespace

# ------------------- load config and simulation parameters -------------------

def create_config():
    parameters_dict = {
        'test_system': 118,
        'hidden_layer_size': 50,
        'n_hidden_layers': 3,
        'epochs': 1000,
        'batch_size': 50,
        'learning_rate': 1e-3,
        'lr_decay': 0.97,
        'dataset_split_seed': 10,
        'pytorch_init_seed': 3,
        'pg_viol_weight': 1e1,
        'qg_viol_weight': 0,
        'vm_viol_weight': 1e1,
        'line_viol_weight': 1e1,
        'crit_weight': 1e1,
        'PF_weight': 1e1,
        'LPF_weight': 1e1,
        'N_enrich': 50,
        'Algo': True, # if True, add worst-case violation CROWN bounds during training
        'Enrich': False,
        'abc_method': 'backward', # "CROWN", "Dynamic-Forward", CROWN-Optimized, IBP, alpha-CROWN, backward
    }
    config = SimpleNamespace(**parameters_dict)
    return config



# --------------------- load the weights ------------------

def build_network(nn_type, n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed, simulation_parameters):
        hidden_layer_size = [hidden_layer_size] * n_hidden_layers
        model = NeuralNetwork(nn_type, n_input_neurons, hidden_layer_size=hidden_layer_size,
                            num_output=n_output_neurons, pytorch_init_seed=pytorch_init_seed, simulation_parameters = simulation_parameters)
        return model#.to(device)


def load_weights(config, nn_type, nn_file_name, input_dim = 198, num_classes = 236):
    
    # config = create_config()
    simulation_parameters = create_example_parameters(config.test_system)   
    
    network_gen = build_network(nn_type, input_dim, num_classes, config.hidden_layer_size,
                                config.n_hidden_layers, config.pytorch_init_seed,
                                simulation_parameters) # Pass full simulation_parameters
    
    # base dir
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    model_save_directory = os.path.join(base_dir, 'MinMax', 'models', 'best_model')
    path = nn_file_name # f"checkpoint_118_50_True_vr_vi.pt"
    path_dir = os.path.join(model_save_directory, path)

    # Step 3: Load the saved weights
    network_gen = torch.load(path_dir, map_location=torch.device('cpu'))
    network_gen.eval()  # set to evaluation mode if you're doing inference
    
    return network_gen


# load_weights(input_dim=198, num_classes=236)  # Example dimensions for case 118 with vr and vi












# -------------------- end weight loading ------------------




















# class PinjMcCormickNN(nn.Module):
#     def __init__(self, mode='upper', num_features=None):
#         super().__init__()
#         self.mccormick1 = McCormickBilinearNN(mode, num_features=num_features)
#         self.mccormick2 = McCormickBilinearNN(mode, num_features=num_features)
        
#     def update_mccormick_bounds(self, vr_l, vr_u, vi_l, vi_u, ir_l, ir_u, ii_l, ii_u):
#         self.mccormick1.update_bounds(vr_l, vr_u, ir_l, ir_u)
#         self.mccormick2.update_bounds(vi_l, vi_u, ii_l, ii_u)

#     def forward(self, Vr, Vi, Ir, Ii):
#         # bounds: dict of tensors (B,)
#         p1 = self.mccormick1(Vr, Ir)
#         p2 = self.mccormick2(Vi, Ii)
#         return p1 + p2


# class QinjMcCormickNN(nn.Module):
#     def __init__(self, mode='upper', num_features=None):
#         super().__init__()
#         self.mccormick1 = McCormickBilinearNN(mode, num_features=num_features)  # For Vi * Ir
#         self.mccormick2 = McCormickBilinearNN(mode, num_features=num_features)  # For Vr * Ii
        
#     def update_mccormick_bounds(self, vr_l, vr_u, vi_l, vi_u, ir_l, ir_u, ii_l, ii_u):
#         self.mccormick1.update_bounds(vr_l, vr_u, ir_l, ir_u)
#         self.mccormick2.update_bounds(vi_l, vi_u, ii_l, ii_u)

#     def forward(self, Vr, Vi, Ir, Ii):
#         q1 = self.mccormick1(Vi, Ir)  # Vi * Ir
#         q2 = self.mccormick2(Vr, Ii)  # Vr * Ii
#         return q1 - q2



    
    
    
# class McCormickLayer(nn.Module):
#     def __init__(self, n_buses: int):
#         super().__init__()
#         self.n_buses = n_buses

#         # define parameters
#         self.vrl_cp = cp.Parameter(shape=(self.n_buses,))
#         self.vru_cp = cp.Parameter(shape=(self.n_buses,))
#         self.vil_cp = cp.Parameter(shape=(self.n_buses,))
#         self.viu_cp = cp.Parameter(shape=(self.n_buses,))
#         self.irl_cp = cp.Parameter(shape=(self.n_buses,))
#         self.iru_cp = cp.Parameter(shape=(self.n_buses,))
#         self.iil_cp = cp.Parameter(shape=(self.n_buses,))
#         self.iiu_cp = cp.Parameter(shape=(self.n_buses,))
        
#         # you can't multiply two parameters, define multiplication
#         self.vrl_irl_cp = cp.Parameter(shape=(self.n_buses,), name="vrl_irl_cp")
#         self.vru_iru_cp = cp.Parameter(shape=(self.n_buses,), name="vru_iru_cp")
#         self.vru_irl_cp = cp.Parameter(shape=(self.n_buses,), name="vru_irl_cp")
#         self.vrl_iru_cp = cp.Parameter(shape=(self.n_buses,), name="vrl_iru_cp")
        
#         self.vil_iil_cp = cp.Parameter(shape=(self.n_buses,), name="vil_iil_cp")
#         self.viu_iiu_cp = cp.Parameter(shape=(self.n_buses,), name="viu_iiu_cp")
#         self.viu_iil_cp = cp.Parameter(shape=(self.n_buses,), name="viu_iil_cp")
#         self.vil_iiu_cp = cp.Parameter(shape=(self.n_buses,), name="vil_iiu_cp")
        
#         self.vil_irl_cp = cp.Parameter(shape=(self.n_buses,), name="vil_irl_cp")
#         self.viu_iru_cp = cp.Parameter(shape=(self.n_buses,), name="viu_iru_cp")
#         self.viu_irl_cp = cp.Parameter(shape=(self.n_buses,), name="viu_irl_cp")
#         self.vil_iru_cp = cp.Parameter(shape=(self.n_buses,), name="vil_iru_cp")
        
#         self.vrl_iil_cp = cp.Parameter(shape=(self.n_buses,), name="vrl_iil_cp")
#         self.vru_iiu_cp = cp.Parameter(shape=(self.n_buses,), name="vru_iiu_cp")
#         self.vru_iil_cp = cp.Parameter(shape=(self.n_buses,), name="vru_iil_cp")
#         self.vrl_iiu_cp = cp.Parameter(shape=(self.n_buses,), name="vrl_iiu_cp")

#         # define variables
#         vr_var = cp.Variable(shape=(self.n_buses,))
#         vi_var = cp.Variable(shape=(self.n_buses,))
#         ir_var = cp.Variable(shape=(self.n_buses,))
#         ii_var = cp.Variable(shape=(self.n_buses,))
#         z1_var = cp.Variable(shape=(self.n_buses,)) # Represents Vr*Ir
#         z2_var = cp.Variable(shape=(self.n_buses,)) # Represents Vi*Ii
#         z3_var = cp.Variable(shape=(self.n_buses,)) # Represents Vi*Ir
#         z4_var = cp.Variable(shape=(self.n_buses,)) # Represents Vr*Ii

#         # Constraints - IMPORTANT: Use cp.multiply for element-wise products!
#         constraints = [
#             vr_var >= self.vrl_cp, vr_var <= self.vru_cp,
#             vi_var >= self.vil_cp, vi_var <= self.viu_cp,
#             ir_var >= self.irl_cp, ir_var <= self.iru_cp,
#             ii_var >= self.iil_cp, ii_var <= self.iiu_cp,

#             # McCormick Envelope Constraints for z1 = vr * ir
#             z1_var >= cp.multiply(self.vrl_cp, ir_var) + cp.multiply(vr_var, self.irl_cp) - self.vrl_irl_cp,
#             z1_var >= cp.multiply(self.vru_cp, ir_var) + cp.multiply(vr_var, self.iru_cp) - self.vru_iru_cp,
#             z1_var <= cp.multiply(self.vru_cp, ir_var) + cp.multiply(vr_var, self.irl_cp) - self.vru_irl_cp,
#             z1_var <= cp.multiply(self.vrl_cp, ir_var) + cp.multiply(vr_var, self.iru_cp) - self.vrl_iru_cp,

#             # McCormick Envelope Constraints for z2 = vi * ii
#             z2_var >= cp.multiply(self.vil_cp, ii_var) + cp.multiply(vi_var, self.iil_cp) - self.vil_iil_cp,
#             z2_var >= cp.multiply(self.viu_cp, ii_var) + cp.multiply(vi_var, self.iiu_cp) - self.viu_iiu_cp,
#             z2_var <= cp.multiply(self.viu_cp, ii_var) + cp.multiply(vi_var, self.iil_cp) - self.viu_iil_cp,
#             z2_var <= cp.multiply(self.vil_cp, ii_var) + cp.multiply(vi_var, self.iiu_cp) - self.vil_iiu_cp,

#             # McCormick Envelope Constraints for z3 = vi * ir
#             z3_var >= cp.multiply(self.vil_cp, ir_var) + cp.multiply(vi_var, self.irl_cp) - self.vil_irl_cp,
#             z3_var >= cp.multiply(self.viu_cp, ir_var) + cp.multiply(vi_var, self.iru_cp) - self.viu_iru_cp,
#             z3_var <= cp.multiply(self.viu_cp, ir_var) + cp.multiply(vi_var, self.irl_cp) - self.viu_irl_cp,
#             z3_var <= cp.multiply(self.vil_cp, ir_var) + cp.multiply(vi_var, self.iru_cp) - self.vil_iru_cp,

#             # McCormick Envelope Constraints for z4 = vr * ii
#             z4_var >= cp.multiply(self.vrl_cp, ii_var) + cp.multiply(vr_var, self.iil_cp) - self.vrl_iil_cp,
#             z4_var >= cp.multiply(self.vru_cp, ii_var) + cp.multiply(vr_var, self.iiu_cp) - self.vru_iiu_cp,
#             z4_var <= cp.multiply(self.vru_cp, ii_var) + cp.multiply(vr_var, self.iil_cp) - self.vru_iil_cp,
#             z4_var <= cp.multiply(self.vrl_cp, ii_var) + cp.multiply(vr_var, self.iiu_cp) - self.vrl_iiu_cp,
#         ]
        
#         # define four optimization problems
#         self.min_pinj_layers = nn.ModuleList()
#         self.max_pinj_layers = nn.ModuleList()
#         self.min_qinj_layers = nn.ModuleList()
#         self.max_qinj_layers = nn.ModuleList()

#         # add values to the parameters
#         self.all_params_ordered_cp = [
#             self.vrl_cp, self.vru_cp, self.vil_cp, self.viu_cp,
#             self.irl_cp, self.iru_cp, self.iil_cp, self.iiu_cp,
#             self.vrl_irl_cp, self.vru_iru_cp, self.vru_irl_cp, self.vrl_iru_cp,
#             self.vil_iil_cp, self.viu_iiu_cp, self.viu_iil_cp, self.vil_iiu_cp,
#             self.vil_irl_cp, self.viu_iru_cp, self.viu_irl_cp, self.vil_iru_cp,
#             self.vrl_iil_cp, self.vru_iiu_cp, self.vru_iil_cp, self.vrl_iiu_cp
#         ]
        
#         # define vars
#         self.all_vars_ordered_cp = [vr_var, vi_var, ir_var, ii_var, z1_var, z2_var, z3_var, z4_var]
        
#         # solver options
#         self.solver_options = {"solve_method":"ECOS",
#             'verbose': False, # False
#             'max_iters': 10000, # Maximum number of iterations
#             'reltol': 1e-3, # Convergence tolerance
#             # Add other ECOS solver arguments as needed
#         }
        
#         # # other solver option
#         # self.solver_options = SCS_solver_args={'use_indirect': False,
#         #     'gpu': False,
#         #     'verbose': False, # False
#         #     'normalize': True, #True heuristic data rescaling
#         #     'max_iters': 10000, #2500 giving the maximum number of iterations
#         #     'scale': 100, #1 if normalized, rescales by this factor
#         #     'eps':1e-3, #1e-3 convergence tolerance
#         #     'cg_rate': 2, #2 for indirect, tolerance goes down like 1/iter^cg_rate
#         #     'alpha': 1.5, #1.5 relaxation parameter
#         #     'rho_x':1e-3, #1e-3 x equality constraint scaling
#         #     'acceleration_lookback': 10, #10
#         #     'write_data_filename':None}

#         # define the optimization problem at each bus; min/max pinj and qinj
#         for i in range(n_buses):
#             expr_pinj_i = z1_var[i] + z2_var[i]
#             prob_min_pinj_i = cp.Problem(cp.Minimize(expr_pinj_i), constraints)
#             prob_max_pinj_i = cp.Problem(cp.Maximize(expr_pinj_i), constraints)
            
#             expr_qinj_i = z3_var[i] - z4_var[i]
#             prob_min_qinj_i = cp.Problem(cp.Minimize(expr_qinj_i), constraints)
#             prob_max_qinj_i = cp.Problem(cp.Maximize(expr_qinj_i), constraints)

#             self.min_pinj_layers.append(CvxpyLayer(prob_min_pinj_i, parameters=self.all_params_ordered_cp, variables=self.all_vars_ordered_cp))
#             self.max_pinj_layers.append(CvxpyLayer(prob_max_pinj_i, parameters=self.all_params_ordered_cp, variables=self.all_vars_ordered_cp))
#             self.min_qinj_layers.append(CvxpyLayer(prob_min_qinj_i, parameters=self.all_params_ordered_cp, variables=self.all_vars_ordered_cp))
#             self.max_qinj_layers.append(CvxpyLayer(prob_max_qinj_i, parameters=self.all_params_ordered_cp, variables=self.all_vars_ordered_cp))


#     def forward(self, lb_vr, ub_vr, lb_vi, ub_vi, lb_ir, ub_ir, lb_ii, ub_ii):
#         batch_size = lb_vr.shape[0]

#         # compute parameter multiplication values
#         vrl_irl_val = lb_vr * lb_ir
#         vru_iru_val = ub_vr * ub_ir
#         vru_irl_val = ub_vr * lb_ir
#         vrl_iru_val = lb_vr * ub_ir
        
#         vil_iil_val = lb_vi * lb_ii
#         viu_iiu_val = ub_vi * ub_ii
#         viu_iil_val = ub_vi * lb_ii
#         vil_iiu_val = lb_vi * ub_ii
        
#         vil_irl_val = lb_vi * lb_ir
#         viu_iru_val = ub_vi * ub_ir
#         viu_irl_val = ub_vi * lb_ir
#         vil_iru_val = lb_vi * ub_ir
        
#         vrl_iil_val = lb_vr * lb_ii
#         vru_iiu_val = ub_vr * ub_ii
#         vru_iil_val = ub_vr * lb_ii
#         vrl_iiu_val = lb_vr * ub_ii

#         # assign parameter values
#         params_numerical = [
#             lb_vr, ub_vr, lb_vi, ub_vi,
#             lb_ir, ub_ir, lb_ii, ub_ii,
#             vrl_irl_val, vru_iru_val, vru_irl_val, vrl_iru_val,
#             vil_iil_val, viu_iiu_val, viu_iil_val, vil_iiu_val,
#             vil_irl_val, viu_iru_val, viu_irl_val, vil_iru_val,
#             vrl_iil_val, vru_iiu_val, vru_iil_val, vrl_iiu_val
#         ]
        
#         params_numerical = [p.to(self.device).float() if isinstance(p, torch.Tensor) else p for p in params_numerical]

#         lb_pinj_list = []
#         ub_pinj_list = []
#         lb_qinj_list = []
#         ub_qinj_list = []

#         # solve the cvxpylayer optimization problems
#         for i in range(self.n_buses): #self.n_buses
#             _, _, _, _, z1_min_pinj, z2_min_pinj, _, _ = self.min_pinj_layers[i](*params_numerical, solver_args = self.solver_options)
#             lb_pinj_list.append(z1_min_pinj[:, i] + z2_min_pinj[:, i])

#             _, _, _, _, z1_max_pinj, z2_max_pinj, _, _ = self.max_pinj_layers[i](*params_numerical, solver_args = self.solver_options)
#             ub_pinj_list.append(z1_max_pinj[:, i] + z2_max_pinj[:, i])

#             _, _, _, _, _, _, z3_min_qinj, z4_min_qinj = self.min_qinj_layers[i](*params_numerical, solver_args = self.solver_options)
#             lb_qinj_list.append(z3_min_qinj[:, i] - z4_min_qinj[:, i])

#             _, _, _, _, _, _, z3_max_qinj, z4_max_qinj = self.max_qinj_layers[i](*params_numerical, solver_args = self.solver_options)
#             ub_qinj_list.append(z3_max_qinj[:, i] - z4_max_qinj[:, i])
            
#             if i % 20 == 0:
#                 print(f"{i} injections sovled")

#         # collect results
#         lb_pinj = torch.stack(lb_pinj_list, dim=1)
#         ub_pinj = torch.stack(ub_pinj_list, dim=1)
#         lb_qinj = torch.stack(lb_qinj_list, dim=1)
#         ub_qinj = torch.stack(ub_qinj_list, dim=1)

#         return lb_pinj, ub_pinj, lb_qinj, ub_qinj

#     @property
#     def device(self):
#         return next(self.parameters()).device if next(self.parameters(), None) is not None else torch.device("cpu")
    


# class Normalise(nn.Module):
#     def __init__(self, n_neurons):
#         super(Normalise, self).__init__()
#         self.register_buffer("minimum", torch.zeros(n_neurons))
#         self.register_buffer("delta", torch.ones(n_neurons))
#         self.eps = 1e-8

#     def forward(self, input):
#         return (input - self.minimum) / (self.delta + self.eps)

#     def set_normalisation(self, minimum, delta):
#         if minimum.ndim != 1 or delta.ndim != 1:
#             raise ValueError("Normalization stats must be 1D tensors.")
#         if torch.any(delta <= 1e-12):
#             raise ValueError("Delta contains zero or near-zero entries.")
#         self.minimum.copy_(minimum)
#         self.delta.copy_(delta)



# class Denormalise(nn.Module):
#     def __init__(self, n_neurons):
#         super(Denormalise, self).__init__()
#         self.register_buffer("delta", torch.ones(n_neurons))
#         self.register_buffer("minimum", torch.zeros(n_neurons))

#     def forward(self, input):
#         return input * self.delta + self.minimum

#     def set_normalisation(self, minimum, delta):
#         if delta.ndim != 1 or minimum.ndim != 1:
#             raise ValueError("Normalization statistics must be 1D tensors.")
#         if torch.any(delta <= 1e-12):
#             raise ValueError("Delta contains zero or near-zero entries.")
#         self.delta.copy_(delta)
#         self.minimum.copy_(minimum)


# class Clamp(nn.Module):
#     def __init__(self, n_neurons):
#         super(Clamp, self).__init__()
#         self.lower_bound = nn.Parameter(data=torch.zeros(1), requires_grad=False)
#         self.upper_bound = nn.Parameter(data=torch.ones(1), requires_grad=False)
        
#     def forward(self,input):
#         return input.clamp( self.lower_bound, self.upper_bound)

# class BoundClip(torch.nn.Module):
#     """ An activation function that clips the output to a given range."""
#     def __init__(self, lower: Tensor, upper: Tensor, which: str = "sigmoid"):
#         super().__init__()
#         assert lower.shape == upper.shape

#         self.register_buffer("lower_bound", lower)
#         self.register_buffer("upper_bound", upper)

#         which = which.lower()

#         assert which in ["hardtanh", "sigmoid", "clamp"]
        
#         if which == "hardtanh":
#             self._forward = self.hardtanh
#         elif which == "sigmoid":
#             self._forward = self.sigmoid
#         elif which == "clamp":
#             self._forward = self.clamp
#         else:
#             raise ValueError(f"Unknown bound clipping function: {which}")

#     def __repr__(self): return f"BoundClip(method={self._forward.__name__})"

#     def forward(self, x):
#         return self._forward(x)

#     def clamp(self, x):
#         return torch.clamp(x, self.lower_bound, self.upper_bound)
    
#     def hardtanh(self, x):
#         return F.hardtanh(x, self.lower_bound, self.upper_bound)
    
#     def sigmoid(self, x):
#         return torch.sigmoid(x) * (self.upper_bound - self.lower_bound) + self.lower_bound





