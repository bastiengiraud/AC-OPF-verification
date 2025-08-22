import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl, torch, torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

class SurrogateModel(pl.LightningModule):

    def __init__(self, num_features, hidden_layer_size = [5,5], num_output=1,pytorch_init_seed=0):
        torch.manual_seed(pytorch_init_seed)    
        super(SurrogateModel, self).__init__()  
        # input layer
        self.L_1 = nn.Linear(num_features, hidden_layer_size[0])

        self.bn1 = nn.BatchNorm1d(hidden_layer_size[0])

        self.L_2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        
        self.bn2 = nn.BatchNorm1d(hidden_layer_size[1])

        self.L_3 = nn.Linear(hidden_layer_size[1],num_output)

        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        
        self.Input_Normalise = Normalise(num_features)

        self.Output_De_Normalise = Denormalise(num_output)
        
        self.Input_Standardize = Standardize(num_features)

        self.Output_Destandardize = Destandardize(num_output)
    
        self.clamp = Clamp(1)
    def normalise_input(self, input_statistics):
        self.Input_Normalise.set_normalisation(minimum=input_statistics[0], delta=input_statistics[1])

    def normalise_output(self, output_statistics):
        self.Output_De_Normalise.set_normalisation(output_statistics[0], output_statistics[1])
        
    def standardize_input(self, input_statistics):
        self.Input_Standardize.set_stats(mean=input_statistics[0], std=input_statistics[1])

    def destandardize_input(self, output_statistics):
        self.Output_Destandardize.set_stats(mean=output_statistics[0], std=output_statistics[1])



    def forward(self, x):
        """ 
        Forward without clamping.
        
        """
        #Layer 1

        x=self.Input_Standardize(x)

        # x = F.linear(x, self.W_1, self.b_1)
        x = self.L_1(x)
        # x = self.bn1(x)
        x = self.activation(x)
        
        #Layer 2
        # x = F.linear(x, self.W_4, self.b_4)
        
        x = self.L_2(x)
        x = self.activation(x)
        x = self.L_3(x)

        x = self.Output_Destandardize(x)
        

        
        return x

    def forward_aft(self, x):
        """ 
        Forward with clamping.
        
        """
        #Layer 1

        x=self.Input_Normalise(x)

        # x = F.linear(x, self.W_1, self.b_1)
        x = self.L_1(x)
        # x = self.bn1(x)
        x = self.activation(x)

        
        #Layer 3
        # x = F.linear(x, self.W_4, self.b_4)
        
        x = self.L_2(x)
        x = self.activation(x)
        x = self.L_3(x)
        
        
        x = self.clamp(x)
        
        x = self.Output_De_Normalise(x)

        
        return x
    
class Standardize(nn.Module):
    def __init__(self, n_neurons):
        super(Standardize, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.std = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = torch.tensor(1e-8)

    def forward(self, input):
        return (input - self.mean) / (self.std + self.eps)

    def set_stats(self, mean, std):
        if len(mean.shape) != 1 or len(std.shape) != 1:
            raise Exception('Stats must be 1-D tensors.')
        if torch.any(std <= 1e-12):
            raise Exception('Standard deviation contains zero or near-zero values.')

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.std = nn.Parameter(data=std, requires_grad=False)

    def get_stats(self):
        return self.mean, self.std
    
    
    
class Destandardize(nn.Module):
    def __init__(self, n_neurons):
        super(Destandardize, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.std = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)

    def forward(self, input):
        return input * self.std + self.mean

    def set_stats(self, mean, std):
        if len(mean.shape) != 1 or len(std.shape) != 1:
            raise Exception('Stats must be 1-D tensors.')
        if torch.any(std <= 1e-12):
            raise Exception('Standard deviation contains zero or near-zero values.')

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.std = nn.Parameter(data=std, requires_grad=False)

    def get_stats(self):
        return self.mean, self.std



class Normalise(nn.Module):
    def __init__(self, n_neurons):
        super(Normalise, self).__init__()
        self.minimum = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.delta = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = torch.tensor(1e-8)

    def forward(self, input):

        return (input - self.minimum) / (self.delta + self.eps)

    def set_normalisation(self, minimum, delta):
        # if not len(delta.shape) == 1 or not len(minimum.shape) == 1:
        #     raise Exception('Input statistics are not 1-D tensors.')

        # # if not torch.nonzero(self.delta).shape[0] == delta.shape[0]:
        # #     raise Exception('Standard deviation in normalisation contains elements equal to 0.')
        
        # if not torch.nonzero(delta).shape[0] == delta.shape[0]:
        #     raise Exception('Standard deviation in normalisation contains elements equal to 0.')
        
        if len(minimum.shape) != 1 or len(delta.shape) != 1:
            raise Exception('Input statistics are not 1-D tensors.')

        if torch.any(delta <= 1e-12):
            raise Exception('Standard deviation in normalisation contains elements equal to 0 or near zero.')



        self.minimum = nn.Parameter(data=minimum, requires_grad=False)
        self.delta = nn.Parameter(data=delta, requires_grad=False)
    
    def get_normalisation_stats(self): # Add this for inspection
        return self.minimum, self.delta


class Denormalise(nn.Module):
    def __init__(self, n_neurons):
        super(Denormalise, self).__init__()
        self.minimum = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.delta = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)

    def forward(self, input):
        return input * self.delta + self.minimum

    def set_normalisation(self, minimum, delta):
        if not (len(minimum.shape) == 1 and len(delta.shape) == 1):
            raise Exception('Input statistics are not 1-D tensors.')

        if torch.any(delta <= 1e-12):
            raise Exception('Standard deviation in denormalisation contains elements equal to 0 or near zero.')

        self.minimum = nn.Parameter(data=minimum, requires_grad=False)
        self.delta = nn.Parameter(data=delta, requires_grad=False)
    
    def get_normalisation_stats(self): # Add this for inspection
        return self.minimum, self.delta


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
