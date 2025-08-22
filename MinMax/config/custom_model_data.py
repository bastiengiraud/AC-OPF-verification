#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
## Copyright (C) 2021-2022, Huan Zhang <huan@huan-zhang.com>           ##
##                     Kaidi Xu, Zhouxing Shi, Shiqi Wang              ##
##                     Linyi Li, Jinqi (Kathryn) Chen                  ##
##                     Zhuolin Yang, Yihan Wang                        ##
##                                                                     ##
##      See CONTRIBUTORS for author contacts and affiliations.         ##
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""
This file shows how to use customized models and customized dataloaders.

An example config file, `exp_configs/custom_model.py` has been provided.

python abcrown.py --config exp_configs/custom_model_data_example.yaml
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd


# Define root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

# Optional: subfolders if needed
for subdir in ['data']:
    sys.path.insert(0, os.path.join(ROOT_DIR, subdir))

from dc_opf.lightning_NN import NeuralNetwork
from dc_opf.create_example_parameters import create_example_parameters
from dc_opf.create_data import create_data

def simple_conv_model(in_channel, out_dim):
    """Simple Convolutional model."""
    model = nn.Sequential(
        nn.Conv2d(in_channel, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, out_dim)
    )
    return model

def dense_pf():
    class ReluLayer(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(ReluLayer, self).__init__()
            self.linear = nn.Linear(n_in, n_out)
            self.ReLU   = nn.ReLU()

        def forward(self, x):
            return self.ReLU(self.linear(x))

    class LinearLayer(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(LinearLayer, self).__init__()
            self.linear = nn.Linear(n_in, n_out)
    
        def forward(self, x):
            return self.linear(x) # Linear Layer -- No ReLU!!
    
    #model = torch.nn.Sequential(ReluLayer(38,10),
    #                            ReluLayer(10,10),
    #                            LinearLayer(10,38))
    model = torch.nn.Sequential(ReluLayer(38,250),
                                ReluLayer(250,250),
                                LinearLayer(250,38))
    
    print(os.getcwd())
    model.load_state_dict(torch.load('./verify-powerflow/ne39_500ReLU.pth'))
    model.eval()
    print(model)
    
    return model

def powerflow_nn_pgmax(n_in, n_out, n_relu, n_layers, model_name, upper_limit_file, upward_scale):
    # define NN
    class ReluLayer(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(ReluLayer, self).__init__()
            self.linear = nn.Linear(n_in, n_out)
            self.ReLU   = nn.ReLU()

        def forward(self, x):
            return self.ReLU(self.linear(x))

    class LinearLayer(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(LinearLayer, self).__init__()
            self.linear = nn.Linear(n_in, n_out)

        def forward(self, x):
            return self.linear(x) # Linear Layer -- No ReLU!!
        
    class SplitLayer(torch.nn.Module):
        def __init__(self, n_in, n_out, n_split):
            super(SplitLayer, self).__init__()
            self.linear  = nn.Linear(n_in, n_out)
            self.ReLU    = nn.ReLU()
            self.n_split = n_split
            self.n_out   = n_out

        def forward(self, x):
            lin_output = self.linear(x)
            return torch.cat([lin_output[:,0:self.n_split], self.ReLU(lin_output[:,self.n_split:self.n_out])], dim=1)

    class MaxLayer_even(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(MaxLayer_even, self).__init__()
            self.ReLU        = nn.ReLU()
            self.linear_diff = nn.Linear(n_in, n_out)
            self.linear_feed = nn.Linear(n_in, n_out)
            
            # define the weights
            Ix = torch.eye(n_out)
            Zx = torch.zeros(n_out,n_out)
            Id = torch.cat([ Ix, -Ix], dim = 1)
            If = torch.cat([ Ix, Zx], dim = 1)
            self.linear_diff.weight = torch.nn.Parameter(Id)
            self.linear_diff.bias   = torch.nn.Parameter(torch.zeros(n_out))
            self.linear_feed.weight = torch.nn.Parameter(If)
            self.linear_feed.bias   = torch.nn.Parameter(torch.zeros(n_out))

        def forward(self, x):
            return -self.ReLU(self.linear_diff(x)) + self.linear_feed(x)

    class MaxLayer_odd(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super(MaxLayer_odd, self).__init__()
            self.ReLU        = nn.ReLU()
            self.linear_diff = nn.Linear(n_in, n_out)
            self.linear_feed = nn.Linear(n_in, n_out)
            self.linear_0    = nn.Linear(n_in, n_out)
            self.n_out       = n_out
            self.n_in        = n_in
            
            # 0 term first -- feeds the first input through unchanged
            Z0 = torch.zeros(n_out,n_in)
            self.linear_0.weight = torch.nn.Parameter(Z0)
            self.linear_0.weight.data[0,0] = torch.tensor(1.0)
            self.linear_0.bias = torch.nn.Parameter(torch.zeros(n_out))
            
            # diff term -- takes the difference between two sets of inputs
            Ix = torch.eye(n_out-1)
            Id = torch.cat([Ix, -Ix], dim = 1)
            self.linear_diff.weight = torch.nn.Parameter(torch.zeros(n_out, n_in))
            self.linear_diff.weight.data[1:n_out,1:n_in] = Id
            self.linear_diff.bias = torch.nn.Parameter(torch.zeros(n_out))
            
            # feedthrough term -- calls the first set of inputs
            Zx = torch.zeros(n_out-1,n_out-1)
            If = torch.cat([Ix, Zx], dim = 1)
            self.linear_feed.weight = torch.nn.Parameter(torch.zeros(n_out, n_in))
            self.linear_feed.weight.data[1:n_out,1:n_in] = If
            self.linear_feed.bias = torch.nn.Parameter(torch.zeros(n_out))

        def forward(self, x):
            #return torch.cat([x[:,0:2], self.ReLU(self.linear_diff(x[:,1:self.n_in])) + self.linear_feed(x[:,1:self.n_in])], dim = 1)
            #return torch.cat([x[:,0].reshape((-1,1)), x[:,1:self.n_in]], dim = 1)
            return self.linear_0(x) - self.ReLU(self.linear_diff(x)) + self.linear_feed(x)

    #=====================================================================================
    # the nn takes the following form: nn(pd) -> pg -> pg_max - pg -> max(pg_max - pg) = t
    #
    # build the original model        
    layers = []
    for kk in range(n_layers):
        if kk == 0:
            # first layer
            layers.append(nn.Linear(n_in, n_relu))
            layers.append(nn.ReLU())
        elif kk == n_layers-1:
            # last layer
            layers.append(nn.Linear(n_relu, n_out))
        else:
            # middle layer
            layers.append(nn.Linear(n_relu, n_relu))
            layers.append(nn.ReLU())
    
    # now create the full model
    net = nn.Sequential(*layers)
    net.load_state_dict(torch.load("./verify-powerflow/models/"+model_name+".pth"))
    net.eval()
    
    # loop over layers
    v_layers = []
    for ii in range(len(net)):
        v_layers.append(net[ii])
    
    # violation_layer: pg_max - pg
    upper_limit_csv = pd.read_csv("./verify-powerflow/data/"+upper_limit_file)
    upper_limit     = upward_scale*torch.from_numpy(upper_limit_csv.values).float()[0]
    violation_layer = LinearLayer(n_out,n_out)
    violation_layer.linear.weight = torch.nn.Parameter(-torch.eye(n_out))
    violation_layer.linear.bias   = torch.nn.Parameter(upper_limit)
    v_layers.append(violation_layer)
    
    # maximization layer: max(x)
    n_in = n_out
    
    # loop until no longer needed (i.e., when the NN spits out the maximum violation)
    add_more_layers = True
    
    while add_more_layers == True:
        if (n_in % 2) == 0:
            n_out       = int(n_in/2)
            maxlayer_even = MaxLayer_even(n_in, n_out)
            v_layers.append(maxlayer_even) # append this layer!

            # reset for the next layer
            n_in = n_out
            
        else: # in this case, the number of outputs is off
            n_out = int((n_in-1)/2 + 1)
            maxlayer_odd = MaxLayer_odd(n_in, n_out)
            v_layers.append(maxlayer_odd) # append this layer!
        
            # reset for the next layer
            n_in = n_out
        
        if n_out == 1:
            add_more_layers = False
    
    # now create the full model
    v_net = nn.Sequential(*v_layers)
    v_net.eval()
    
    return v_net




def dc_opf_training(n_buses,hidden_layer_size,n_hidden_layers,pytorch_init_seed):
    
    simulation_parameters = create_example_parameters(n_buses)
    
    # Getting Training Data
    Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)
    
    # Defining the tensors
    Dem_train = torch.tensor(Dem_train).float()
    Gen_train = torch.tensor(Gen_train).float()

    #--------------------------------------------------------------------- 
    # Gen type defines if the data belongs to training dataset or if it 
    # belongs to the additional points collected from the training space
    # type = 1 means it is part of the training set and the generation measurements are present
    #---------------------------------------------------------------------

    Gen_delta=simulation_parameters['true_system']['Pg_delta'] 
    Dem_min=simulation_parameters['true_system']['Pd_min']
    Dem_delta=simulation_parameters['true_system']['Pd_delta']
    
    # NNs for predicting Generation (network_gen) and Volatage (network_Volt)
    network_gen = build_network(Dem_train.shape[1],
                            Gen_train.shape[1],
                            hidden_layer_size,
                            n_hidden_layers,
                            pytorch_init_seed)
    
    network_gen = normalise_network(network_gen,  Dem_min, Dem_delta, Gen_delta,Dem_train)
    
    path ='./models/checkpoint_'+str(n_buses)+'_'+str(pytorch_init_seed)+'_gpu_.pth'
    network_gen.load_state_dict(torch.load(path))
    
    network_gen.eval()
    
    v_layers = []
    # for ii in range(len(network_gen)):
    # v_layers.append(network_gen.Input_Normalise)
    v_layers.append(network_gen.L_1)
    v_layers.append(network_gen.activation)
    v_layers.append(network_gen.L_2)
    v_layers.append(network_gen.activation)
    v_layers.append(network_gen.L_3)
    v_layers.append(network_gen.activation)
    v_layers.append(network_gen.L_4)
    # v_layers.append(network_gen.Output_De_Normalise)
    v_net = nn.Sequential(*v_layers)
    v_net.eval()
    
    # net = network_gen.get_nn_sequential()
    # net.eval()    
    return v_net

def build_network(n_input_neurons, n_output_neurons,hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    model = NeuralNetwork(num_features=n_input_neurons,
                            hidden_layer_size=[hidden_layer_size,hidden_layer_size,hidden_layer_size],
                            num_output= n_output_neurons,
                            pytorch_init_seed=pytorch_init_seed )


    return model

def normalise_network(model, pd_min, pd_delta, pg_delta,Dem_train):
    input_statistics = (torch.from_numpy(pd_delta.reshape(-1,).astype(np.float32)),torch.from_numpy(pd_min.reshape(-1,).astype(np.float32)))
    output_statistics = torch.from_numpy(pg_delta.reshape(-1,).astype(np.float32))
    input_stat=torch.std_mean(Dem_train, dim=0, unbiased=False)

    model.normalise_input(input_statistics=input_statistics)
    model.normalise_output(output_statistics=output_statistics)

    return model
    
def dense_pf_box(eps=0.1):
    """a customized box data: x=[-1, 1], y=[-1, 1]"""
    X = torch.tensor([[ 0.976 ,  0.    ,  3.22  ,  5.    ,  0.    ,  0.    ,  2.338 ,
        5.22  ,  0.065 ,  0.    ,  0.    ,  0.0853,  0.    ,  0.    ,
        3.2   ,  3.29  ,  0.    ,  1.58  ,  0.    ,  6.8   ,  2.74  ,
        0.    ,  2.475 ,  3.086 ,  2.24  ,  1.39  ,  2.81  ,  2.06  ,
        2.835 ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        0.    ,  0.    , 11.04  ]]).float()
    labels   = torch.tensor([0]).long()
    eps_temp = torch.tensor(eps).reshape(1, -1)
    data_max = X + 0.01
    data_min = X - 0.01
    return X, labels, data_max, data_min, eps_temp

def two_relu_toy_model(in_dim=2, out_dim=2):
    """A very simple model, 2 inputs, 2 ReLUs, 2 outputs"""
    model = nn.Sequential(
        nn.Linear(in_dim, 2),
        nn.ReLU(),
        nn.Linear(2, out_dim)
    )
    """[relu(x+2y)-relu(2x+y)+2, 0*relu(2x-y)+0*relu(-x+y)]"""
    model[0].weight.data = torch.tensor([[1., 2.], [2., 1.]])
    model[0].bias.data = torch.tensor([0., 0.])
    model[2].weight.data = torch.tensor([[1., -1.], [0., 0.]])
    model[2].bias.data = torch.tensor([2., 0.])
    return model

def simple_box_data(eps=2.):
    """a customized box data: x=[-1, 1], y=[-1, 1]"""
    X = torch.tensor([[0., 0.]]).float()
    labels = torch.tensor([0]).long()
    eps_temp = torch.tensor(eps).reshape(1, -1)
    data_max = torch.tensor(10.).reshape(1, -1)
    data_min = torch.tensor(-10.).reshape(1, -1)
    return X, labels, data_max, data_min, eps_temp

def box_data(dim, low=0., high=1., segments=10, num_classes=10, eps=None):
    """Generate fake datapoints."""
    step = (high - low) / segments
    data_min = torch.linspace(low, high - step, segments).unsqueeze(1).expand(segments, dim)  # Per element lower bounds.
    data_max = torch.linspace(low + step, high, segments).unsqueeze(1).expand(segments, dim)  # Per element upper bounds.
    X = (data_min + data_max) / 2.  # Fake data.
    labels = torch.remainder(torch.arange(0, segments, dtype=torch.int64), num_classes)  # Fake label.
    eps = None  # Lp norm perturbation epsilon. Not used, since we will return per-element min and max.
    return X, labels, data_max, data_min, eps

def cifar10(eps, use_bounds=False):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    if use_bounds:
        # Option 1: for each example, we return its element-wise lower and upper bounds.
        # If you use this option, set --spec_type ("specifications"->"type" in config) to 'bound'.
        absolute_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        absolute_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        # Be careful with normalization.
        new_eps = torch.reshape(eps / std, (1, -1, 1, 1))
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        # In this case, the epsilon does not matter here.
        ret_eps = None
    else:
        # Option 2: return a single epsilon for all data examples, as well as clipping lower and upper bounds.
        # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        if eps is None:
            raise ValueError('You must specify an epsilon')
        # Rescale epsilon.
        ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps

def simple_cifar10(eps):
    """Example dataloader. For MNIST and CIFAR you can actually use existing ones in utils.py."""
    assert eps is not None
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    # You can access the mean and std stored in config file.
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)
    test_data = datasets.CIFAR10(database_path, train=False, download=True,\
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    # Load entire dataset.
    testloader = torch.utils.data.DataLoader(test_data,\
            batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Set data_max and data_min to be None if no clip. For CIFAR-10 we clip to [0,1].
    data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
    if eps is None:
        raise ValueError('You must specify an epsilon')
    # Rescale epsilon.
    ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, ret_eps
