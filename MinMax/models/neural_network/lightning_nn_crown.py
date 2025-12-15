import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl, torch, torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np
import cvxpy as cp
from typing import List, Dict, Any
# from cvxpylayers.torch import CvxpyLayer
from auto_LiRPA import BoundedModule, BoundedTensor

class NeuralNetwork(pl.LightningModule):

    def __init__(self, nn_type, num_features, hidden_layer_size = [100,100,100], num_output=1, pytorch_init_seed=0, simulation_parameters = None, surrogate = None):
        torch.manual_seed(pytorch_init_seed)    
        super(NeuralNetwork, self).__init__()  
        
        self.simulation_parameters = simulation_parameters
        self.nn_type = nn_type
        self.n_bus = simulation_parameters['general']['n_buses']
        self.pinj_upper_nn = PinjMcCormickNN(mode='upper', num_features=self.n_bus)
        self.pinj_lower_nn = PinjMcCormickNN(mode='lower', num_features=self.n_bus)
        self.qinj_upper_nn = QinjMcCormickNN(mode='upper', num_features=self.n_bus)
        self.qinj_lower_nn = QinjMcCormickNN(mode='lower', num_features=self.n_bus)
                
        # input layer
        self.L_1 = nn.Linear(num_features, hidden_layer_size[0])
        self.L_2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.L_3 = nn.Linear(hidden_layer_size[1], hidden_layer_size[2])
        self.L_4 = nn.Linear(hidden_layer_size[2],num_output)

        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        self.Input_Normalise = Normalise(num_features)
        self.Output_De_Normalise = Denormalise(num_output)
        self.clamp = Clamp(1)        
        self.return_type = 'all' # Default return type is 'all'
        
        # Register constants as buffers
        Ybr_rect = torch.tensor(self.simulation_parameters['true_system']['Ybr_rect'], dtype=torch.float32)
        self.register_buffer('Ybr_rect', Ybr_rect)
        Ybus_complex = torch.tensor(self.simulation_parameters['true_system']['Ybus'], dtype=torch.complex128)
        self.register_buffer('Ybus_real', Ybus_complex.real.float())
        self.register_buffer('Ybus_imag', Ybus_complex.imag.float())
        map_l = torch.tensor(simulation_parameters['true_system']['Map_L'], dtype=torch.float32)
        self.register_buffer('map_l', map_l)
        kcl_im = torch.tensor(simulation_parameters['true_system']['kcl_im'], dtype=torch.float32)
        kcl_from_im = torch.relu(kcl_im) # +1 at from-bus, 0 elsewhere
        kcl_to_im = -torch.relu(-kcl_im) # +1 at to-bus, 0 elsewhere
        self.register_buffer('kcl_from_im', kcl_from_im)
        self.register_buffer('kcl_to_im', kcl_to_im)
        bs_values = torch.tensor(simulation_parameters['true_system']['bus_bs'], dtype=torch.float32).unsqueeze(1)
        self.register_buffer('bs_values', bs_values)
        
    def normalise_input(self, input_statistics):
        self.Input_Normalise.set_normalisation(minimum=input_statistics[0], delta=input_statistics[1])

    def normalise_output(self, output_statistics):
        self.Output_De_Normalise.set_normalisation(minimum=output_statistics[0], delta=output_statistics[1])


    def forward(self, x):
        """ 
        NOT USED FOR TRAINING:
        Forward includes the constraint at the final layere we're constructing a bound through. 
        
        """
        
        #Layer 1
        n_bus = self.map_l.shape[1] // 2
        sd = (x @ self.map_l)
        pd = sd[:, :n_bus]
        qd = sd[:, :n_bus]
        
        x=self.Input_Normalise(x)

        x = self.L_1(x)
        x = self.activation(x)
        x = self.L_2(x)
        x = self.activation(x)
        x = self.L_3(x)
        x = self.activation(x)
        x = self.L_4(x)

        x = self.Output_De_Normalise(x)
        
        if self.nn_type == 'vr_vi':

            # ------------------- vr vi ----------------------
            # === current bounds ===
            Ibr =  x @ self.Ybr_rect.T # x is rectangular voltages
            n_lines = Ibr.shape[1] // 4  # since it's [Ir_f, Ii_f, Ir_t, Ii_t]
        
            I_f = Ibr[:, :2*n_lines]             # [Ir_f, Ii_f]
            I_t = Ibr[:, 2*n_lines:4*n_lines] # [Ir_t, Ii_t]
            
            Ir_f = I_f[:, :n_lines]  # Real part of from-side currents
            Ii_f = I_f[:, n_lines:]  # Imaginary part of from-side currents
            Ir_t = I_t[:, :n_lines]
            Ii_t = I_t[:, n_lines:]

            # Apply magnitude approximation
            I_mag_f = self.amb_magnitude_current(I_f)
            I_mag_t = self.amb_magnitude_current(I_t)

            # Combine
            i_bound = torch.cat((I_mag_f, I_mag_t), dim=1)
            
            # === voltage bounds ===
            Vr = x[:, :n_bus]    
            Vi = x[:, n_bus:] 

            # Apply magnitude approximation
            u_bound = self.amb_magnitude_voltage_up(Vr, Vi)
            l_bound = self.amb_magnitude_voltage_down(Vr, Vi)
            

            # === current injections ===
            Ir = torch.matmul(Vr, self.Ybus_real.T) - torch.matmul(Vi, self.Ybus_imag.T)
            Ii = torch.matmul(Vr, self.Ybus_imag.T) + torch.matmul(Vi, self.Ybus_real.T)
            
            # === KCL balance ===
            current_inj_real = torch.matmul(Ir_f, self.kcl_from_im.T) - torch.matmul(Ir_t, self.kcl_to_im.T)
            current_inj_imag = torch.matmul(Ii_f, self.kcl_from_im.T) - torch.matmul(Ii_t, self.kcl_to_im.T)

            I_shunt_r = -torch.matmul(Vi, self.bs_values)
            I_shunt_i = torch.matmul(Vr, self.bs_values)

            delta_inj_real = Ir - current_inj_real - I_shunt_r
            delta_inj_imag = Ii - current_inj_imag - I_shunt_i
            
            """
            the transpose of the variables is a problem for Crown!!
            """      
            
            # === compute pinj ===
            
            pg_up =  self.pinj_upper_nn(Vr, Vi, Ir, Ii) + pd # Vr * Ir + Vi * Ii + pd
            pg_down = self.pinj_lower_nn(Vr, Vi, Ir, Ii) + pd
            
            qg_up = self.qinj_upper_nn(Vr, Vi, Ir, Ii) + qd # Vi * Ir - Vr * Ii + qd
            qg_down = self.qinj_lower_nn(Vr, Vi, Ir, Ii) + qd 
            
            # ------------------------------------------------------

            # # --- Flexible Return based on return_type ---
            # return_map = {
            #     'u_bound': u_bound,
            #     'l_bound': l_bound,
            #     'i_bound': i_bound,
            #     'pg_up': pg_up,
            #     'pg_down': pg_down,
            #     'qg_up': qg_up,
            #     'qg_down': qg_down,
            #     'delta_inj_real': delta_inj_real,
            #     'delta_inj_imag': delta_inj_imag,
            #     'Vr': Vr,
            #     'Vi': Vi,
            #     'Ir': Ir,
            #     'Ii': Ii,
            #     'all': (Vr, Vi, Ir, Ii, i_bound, u_bound, l_bound, pg_up, pg_down, qg_up, qg_down, delta_inj_real, delta_inj_imag)
            # }

            # # Use the .get() method with a default value to return 'all' if the
            # # specified return_type is not found.
            # return return_map.get(self.return_type, return_map['all'])
            
            return Vr, Vi, Ir, Ii, i_bound, u_bound, l_bound, pg_up, pg_down, qg_up, qg_down, delta_inj_real, delta_inj_imag
        
        elif self.nn_type == 'pg_vm':
            
            return x

    def forward_train(self, x):
        """ 
        Forward without clamping.
        
        """

        x=self.Input_Normalise(x)
        x = self.L_1(x)
        x = self.activation(x)
        x = self.L_2(x)
        x = self.activation(x)
        x = self.L_3(x)
        x = self.activation(x)
        x = self.L_4(x)

        x = self.Output_De_Normalise(x)

        
        return x

    def forward_aft(self, x):
        """ 
        Forward with clamping.
        
        """
        #Layer 1

        x=self.Input_Normalise(x)
        x = self.L_1(x)
        x = self.activation(x)
        x = self.L_2(x)
        x = self.activation(x)
        x = self.L_3(x)
        x = self.activation(x)
        x = self.L_4(x)
        
        x = self.clamp(x)
        
        x = self.Output_De_Normalise(x)

        
        return x
    
    def abs_relu(self, x):
        return torch.relu(x) + torch.relu(-x)

    def amb_magnitude_current(self, I_complex, alpha = 1,      beta = (2**0.5) - 1):
        """
        I_complex: Tensor of shape (batch_size, 2 * n), where
                first n columns = real parts,
                last n columns = imaginary parts.

        Returns: approx magnitude (batch_size, n)
        best/tightest approximation:       alpha = 0.9604, beta = 0.3978
        guaranteed over approximation:     alpha = 1,      beta = 0.4142 (sqrt(2) - 1)
        """
        n = I_complex.shape[1] // 2
        Ir = I_complex[:, :n]
        Ii = I_complex[:, n:]
        
        abs_Ir = self.abs_relu(Ir)
        abs_Ii = self.abs_relu(Ii)
        
        # max_val = torch.relu(abs_Ir - abs_Ii) + abs_Ii
        # min_val = abs_Ir + abs_Ii - max_val
        
        max_val = torch.maximum(abs_Ir, abs_Ii)
        min_val = torch.minimum(abs_Ir, abs_Ii)
        
        # true_value = torch.sqrt(Ir**2 + Ii**2)
        
        return alpha * max_val + beta * min_val
    
    def amb_magnitude_voltage_up(self, Vr, Vi, alpha = 1,      beta = (2**0.5) - 1):
        
        abs_Vr = self.abs_relu(Vr)
        abs_Vi = self.abs_relu(Vi)
        
        # max_val = torch.relu(abs_Ir - abs_Ii) + abs_Ii
        # min_val = abs_Ir + abs_Ii - max_val
        
        max_val = torch.maximum(abs_Vr, abs_Vi)
        min_val = torch.minimum(abs_Vr, abs_Vi)
        
        # true_value = torch.sqrt(Ir**2 + Ii**2)
        
        return alpha * max_val + beta * min_val
    
    def amb_magnitude_voltage_down(self, Vr, Vi, alpha = 1,      beta = (2**0.5) - 1):
        
        abs_Vr = self.abs_relu(Vr)
        abs_Vi = self.abs_relu(Vi)
        
        # scaled l1 norm ambm approximation
        alpha_l1 = 0.7071
        beta_l1 = 0.7071
        
        max_val_l1 = torch.maximum(abs_Vr, abs_Vi)
        min_val_l1 = torch.minimum(abs_Vr, abs_Vi)
        
        # linf norm ambm approximation
        alpha_linf = 1
        beta_linf = 0
        
        max_val_linf = torch.maximum(abs_Vr, abs_Vi)
        min_val_linf = torch.minimum(abs_Vr, abs_Vi)
        
        # max lower bound
        ambm_l1 = alpha_l1 * max_val_l1 + beta_l1 * min_val_l1
        ambm_linf = alpha_linf * max_val_linf + beta_linf * min_val_linf
        
        under_approx = torch.maximum(ambm_l1, ambm_linf)
        
        return under_approx
    
    # def set_return_type(self, return_type: str):
    #     """Sets the return type for the forward pass."""
    #     valid_types = [
    #         'u_bound', 'l_bound', 'i_bound', 'pg_up', 'pg_down', 'qg_up', 'qg_down',
    #         'delta_inj_real', 'delta_inj_imag', 'Vr', 'Vi', 'Ir', 'Ii', 'all'
    #     ]
    #     if return_type not in valid_types:
    #         raise ValueError(f"Invalid return type: '{return_type}'. Valid types are: {valid_types}")
    #     self.return_type = return_type
    
    
    
    
class OutputWrapper(nn.Module):
    def __init__(self, original_model, output_index):
        super().__init__()
        self.original_model = original_model
        self.output_index = output_index
    
    def forward(self, x):
        # Call the original model's forward method
        outputs = self.original_model(x) 
        # Return the specific output from the tuple/list
        return outputs[self.output_index]


class McCormickBilinearNN(nn.Module):
    def __init__(self, mode='upper', num_features=None):
        super().__init__()
        assert mode in ['upper', 'lower']
        self.mode = mode
        self.relu = nn.ReLU() # For the min/max operations
        
        if num_features is None:
            raise ValueError("num_features must be provided for per-feature bounds.")

        
        # Initialize with loose values, as they will be updated.
        self.register_buffer('x_lower_bound', torch.full((1, num_features), -1e3))
        self.register_buffer('x_upper_bound', torch.full((1, num_features), 1e3))
        self.register_buffer('y_lower_bound', torch.full((1, num_features), -1e3))
        self.register_buffer('y_upper_bound', torch.full((1, num_features), 1e3))
        
    def update_bounds(self, new_x_l, new_x_u, new_y_l, new_y_u):
        with torch.no_grad():
            self.x_lower_bound.copy_(new_x_l.reshape(self.x_lower_bound.shape))
            self.x_upper_bound.copy_(new_x_u.reshape(self.x_upper_bound.shape))
            self.y_lower_bound.copy_(new_y_l.reshape(self.y_lower_bound.shape))
            self.y_upper_bound.copy_(new_y_u.reshape(self.y_upper_bound.shape))

    def forward(self, x, y):
                    
        # # Or simpler if a global fixed bound is okay for tracing dummy:
        x_l = self.x_lower_bound # .expand_as(x)
        x_u = self.x_upper_bound # .expand_as(x)
        y_l = self.y_lower_bound # .expand_as(y)
        y_u = self.y_upper_bound # .expand_as(y)
            
        # Lower bound affine planes
        L1 = y_l * x + x_l * y - x_l * y_l
        L2 = y_u * x + x_u * y - x_u * y_u

        # Upper bound affine planes
        U1 = y_u * x + x_l * y - x_u * y_l
        U2 = y_l * x + x_u * y - x_l * y_u
        
        if self.mode == 'upper':
            return U1 - self.relu(U1 - U2)
        elif self.mode == 'lower':
            return L1 + self.relu(L2 - L1)
        else:
            print("Mode not defined.")


class PinjMcCormickNN(nn.Module):
    def __init__(self, mode='upper', num_features=None):
        super().__init__()
        self.mccormick1 = McCormickBilinearNN(mode, num_features=num_features)
        self.mccormick2 = McCormickBilinearNN(mode, num_features=num_features)
        
    def update_mccormick_bounds(self, vr_l, vr_u, vi_l, vi_u, ir_l, ir_u, ii_l, ii_u):
        self.mccormick1.update_bounds(vr_l, vr_u, ir_l, ir_u)
        self.mccormick2.update_bounds(vi_l, vi_u, ii_l, ii_u)

    def forward(self, Vr, Vi, Ir, Ii):
        # bounds: dict of tensors (B,)
        p1 = self.mccormick1(Vr, Ir)
        p2 = self.mccormick2(Vi, Ii)
        return p1 + p2


class QinjMcCormickNN(nn.Module):
    def __init__(self, mode='upper', num_features=None):
        super().__init__()
        self.mccormick1 = McCormickBilinearNN(mode, num_features=num_features)  # For Vi * Ir
        self.mccormick2 = McCormickBilinearNN(mode, num_features=num_features)  # For Vr * Ii
        
    def update_mccormick_bounds(self, vr_l, vr_u, vi_l, vi_u, ir_l, ir_u, ii_l, ii_u):
        self.mccormick1.update_bounds(vr_l, vr_u, ir_l, ir_u)
        self.mccormick2.update_bounds(vi_l, vi_u, ii_l, ii_u)

    def forward(self, Vr, Vi, Ir, Ii):
        q1 = self.mccormick1(Vi, Ir)  # Vi * Ir
        q2 = self.mccormick2(Vr, Ii)  # Vr * Ii
        return q1 - q2




class Normalise(nn.Module):
    """
    Min max normalization to map features to [0, 1].
    
    """
    def __init__(self, n_neurons):
        super(Normalise, self).__init__()
        self.register_buffer("minimum", torch.zeros(n_neurons))
        self.register_buffer("delta", torch.ones(n_neurons))
        self.eps = 1e-8

    def forward(self, input):
        return (input - self.minimum) / (self.delta + self.eps)

    def set_normalisation(self, minimum, delta):
        if minimum.ndim != 1 or delta.ndim != 1:
            raise ValueError("Normalization stats must be 1D tensors.")
        if torch.any(delta <= 1e-12):
            raise ValueError("Delta contains zero or near-zero entries.")
        self.minimum.copy_(minimum)
        self.delta.copy_(delta)



class Denormalise(nn.Module):
    def __init__(self, n_neurons):
        super(Denormalise, self).__init__()
        self.register_buffer("delta", torch.ones(n_neurons))
        self.register_buffer("minimum", torch.zeros(n_neurons))

    def forward(self, input):
        return input * self.delta + self.minimum

    def set_normalisation(self, minimum, delta):
        if delta.ndim != 1 or minimum.ndim != 1:
            raise ValueError("Normalization statistics must be 1D tensors.")
        if torch.any(delta <= 1e-12):
            raise ValueError("Delta contains zero or near-zero entries.")
        self.delta.copy_(delta)
        self.minimum.copy_(minimum)


class Clamp(nn.Module):
    def __init__(self, n_neurons):
        super(Clamp, self).__init__()
        self.lower_bound = nn.Parameter(data=torch.zeros(1), requires_grad=False)
        self.upper_bound = nn.Parameter(data=torch.ones(1), requires_grad=False)
        
    def forward(self,input):
        return input.clamp( self.lower_bound, self.upper_bound)

class BoundClip(torch.nn.Module):
    """ An activation function that clips the output to a given range."""
    def __init__(self, lower: Tensor, upper: Tensor, which: str = "sigmoid"):
        super().__init__()
        assert lower.shape == upper.shape

        self.register_buffer("lower_bound", lower)
        self.register_buffer("upper_bound", upper)

        which = which.lower()

        assert which in ["hardtanh", "sigmoid", "clamp"]
        
        if which == "hardtanh":
            self._forward = self.hardtanh
        elif which == "sigmoid":
            self._forward = self.sigmoid
        elif which == "clamp":
            self._forward = self.clamp
        else:
            raise ValueError(f"Unknown bound clipping function: {which}")

    def __repr__(self): return f"BoundClip(method={self._forward.__name__})"

    def forward(self, x):
        return self._forward(x)

    def clamp(self, x):
        return torch.clamp(x, self.lower_bound, self.upper_bound)
    
    def hardtanh(self, x):
        return F.hardtanh(x, self.lower_bound, self.upper_bound)
    
    def sigmoid(self, x):
        return torch.sigmoid(x) * (self.upper_bound - self.lower_bound) + self.lower_bound





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