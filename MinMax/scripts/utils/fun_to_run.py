# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:40:07 2023

@author: rnelli
"""
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


from custom_model_data import DC_OPF_Training
from DC_OPF.create_data import create_data
from DC_OPF.create_example_parameters import create_example_parameters

class IntremBound:

    def __init__(self, model_ori, c=None, device=None,
                 cplex_processes=None):
        self.c = c
        # self.model_ori = model_ori
        # net = copy.deepcopy(model_ori)
        self.net = model_ori
        self.net.eval()
        self.interm_transfer = True
        self.final_name = self.net.final_name

        self.new_input_split = False

    def get_interm_bounds(self, lb, ub=None, init=True, device=None):
        """Get the intermediate bounds.

        By default, we also add final layer bound after applying C
        (lb and lb+inf).
        """

        lower_bounds, upper_bounds = {}, {}
        if init:
            self._get_split_nodes()
            for layer in self.net.layers_requiring_bounds + self.net.split_nodes:
                lower_bounds[layer.name] = layer.lower.detach()
                upper_bounds[layer.name] = layer.upper.detach()
        elif self.interm_transfer:
            for layer in self.net.layers_requiring_bounds:
                lower_bounds[layer.name] = self._transfer(
                    layer.lower.detach(), device)
                upper_bounds[layer.name] = self._transfer(
                    layer.upper.detach(), device)

        lower_bounds[self.final_name] = lb.flatten(1).detach()
        if ub is None:
            ub = lb + torch.inf
        upper_bounds[self.final_name] = ub.flatten(1).detach()

        if self.new_input_split:
            self.root = self.net[self.net.root_names[0]]
            lower_bounds[self.root.name] = self.root.lower.detach()
            upper_bounds[self.root.name] = self.root.upper.detach()

        return lower_bounds, upper_bounds

    def _get_split_nodes(self, verbose=False):
        self.net.get_split_nodes(input_split=self.new_input_split)
        self.split_activations = self.net.split_activations
        if verbose:
            print('Split layers:')
            for layer in self.net.split_nodes:
                print(f'  {layer}: {self.split_activations[layer.name]}')
            print('Nonlinear functions:')
            for node in self.net.nodes():
                if node.perturbed and len(node.requires_input_bounds):
                    print('  ', node)

n_buses= 2869
hidden_layer_size = 80
n_hidden_layers = 3
pytorch_init_seed =0

simulation_parameters = create_example_parameters(n_buses)

Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)

# Defining the tensors
Dem_train = torch.tensor(Dem_train).float()
Gen_train = torch.tensor(Gen_train).float()



model = DC_OPF_Training(n_buses,hidden_layer_size,n_hidden_layers,pytorch_init_seed)

if torch.cuda.is_available():
    Dem_train = Dem_train.cuda()
    model = model.cuda()

Para= list(model.parameters())  

optimizer = torch.optim.Adam(Para,lr=0.001)

lirpa_model = BoundedModule(model, torch.empty_like(Dem_train), device=Dem_train.device)
print('Running on', Dem_train.device)

x=torch.load('x.pt')
x_min= torch.load('data_min.pt')
x_max= torch.load('data_max.pt')

ptb = PerturbationLpNorm( x_L=x_min, x_U=x_max/2)

# ptb = PerturbationLpNorm( eps = 0.2)

image = BoundedTensor(x, ptb)
# Get model prediction as usual
pred = lirpa_model(image)

needed_A_dict = dict([(node, []) for node in lirpa_model._modules])

A = torch.load('A.pt')

lb, ub, A= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method="IBP" ) #, reference_bounds = A)

torch.save(A, 'A.pt')
torch.save(lb, 'lb.pt')
torch.save(ub, 'ub.pt')

model_LiRPANet = IntremBound(lirpa_model)
interm_lb,interm_ub=model_LiRPANet.get_interm_bounds(lb=lb,ub=ub)
torch.save(interm_lb,'interm_lb.pt')
torch.save(interm_ub,'interm_ub.pt')