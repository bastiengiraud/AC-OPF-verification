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

Para= list(model.parameters())  

optimizer = torch.optim.Adam(Para,lr=0.001)

if torch.cuda.is_available():
    Dem_train = Dem_train.cuda()
    model = model.cuda()

lirpa_model = BoundedModule(model, torch.empty_like(Dem_train), device=Dem_train.device)
print('Running on', Dem_train.device)


ptb = PerturbationLpNorm( x_L=0, x_U=1)

image = BoundedTensor(Dem_train, ptb)
# Get model prediction as usual
pred = lirpa_model(image)

needed_A_dict = dict([(node, []) for node in lirpa_model._modules])

A = torch.load('A.pt')

lb, ub, A= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method="IBP"  , reference_bounds = A)


torch.save(A, 'A.pt')
torch.save(lb, 'lb.pt')
torch.save(ub, 'ub.pt')

print(lb)



