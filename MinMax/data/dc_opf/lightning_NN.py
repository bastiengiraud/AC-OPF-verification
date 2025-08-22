import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl, torch, torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

class NeuralNetwork(pl.LightningModule):

    def __init__(self, num_features, hidden_layer_size = [100,100,100], num_output=1,pytorch_init_seed=0):
        torch.manual_seed(pytorch_init_seed)    
        super(NeuralNetwork, self).__init__()  
        # input layer
        self.L_1 = nn.Linear(num_features, hidden_layer_size[0])

        self.L_2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])

        self.L_3 = nn.Linear(hidden_layer_size[1], hidden_layer_size[2])

        self.L_4 = nn.Linear(hidden_layer_size[2],num_output)

        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        
        self.Input_Normalise = Normalise(num_features)

        self.Output_De_Normalise = Denormalise(num_output)
    
    
    def normalise_input(self, input_statistics):
        self.Input_Normalise.set_normalisation(minimum=input_statistics[1], delta=input_statistics[0])

    def normalise_output(self, output_statistics):
        self.Output_De_Normalise.set_normalisation(delta=output_statistics)


    def forward(self, x):
        #Layer 1

        x=self.Input_Normalise(x)

        # x = F.linear(x, self.W_1, self.b_1)
        x = self.L_1(x)
        x = self.activation(x)

        #Layer 2
        # x = F.linear(x, self.W_2, self.b_2)
        x = self.L_2(x)
        x = self.activation(x)
        
        # x = F.linear(x, self.W_3, self.b_3)
        x = self.L_3(x)
        x = self.activation(x)
        
        #Layer 3
        # x = F.linear(x, self.W_4, self.b_4)
        
        x = self.L_4(x)

        x = self.Output_De_Normalise(x)

        
        return x


class Normalise(nn.Module):
    def __init__(self, n_neurons):
        super(Normalise, self).__init__()
        self.minimum = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.delta = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input):
        return (input - self.minimum) / (self.delta + self.eps)

    def set_normalisation(self, minimum, delta):
        if not len(delta.shape) == 1 or not len(minimum.shape) == 1:
            raise Exception('Input statistics are not 1-D tensors.')

        if not torch.nonzero(self.delta).shape[0] == delta.shape[0]:
            raise Exception('Standard deviation in normalisation contains elements equal to 0.')

        self.minimum = nn.Parameter(data=minimum, requires_grad=False)
        self.delta = nn.Parameter(data=delta, requires_grad=False)


class Denormalise(nn.Module):
    def __init__(self, n_neurons):
        super(Denormalise, self).__init__()

        self.delta = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)

    def forward(self, input):
        return  input * self.delta

    def set_normalisation(self, delta):
        if not len(delta.shape) == 1 :
            raise Exception('Input statistics are not 1-D tensors.')

        if not torch.nonzero(self.delta).shape[0] == delta.shape[0]:
            raise Exception('Standard deviation in normalisation contains elements equal to 0.')

        self.delta =  nn.Parameter(data=delta, requires_grad=False)

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
