# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:40:07 2023

@author: rnelli
"""
import os
from collections import defaultdict
from typing import Any
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


from config.custom_model_data import dc_opf_training
from data.dc_opf.create_data import create_data
from data.dc_opf.create_example_parameters import create_example_parameters

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

class LiRPANet:

    def __init__(self,n_buses,Data_stat) -> None:

        self.n_buses= n_buses
        self.Dem_min = Data_stat['Dem_min'].reshape(1,-1)
        self.Dem_max = Data_stat['Dem_min'].reshape(1,-1) + Data_stat['Dem_delta'].reshape(1,-1)
        self.x = (self.Dem_min + self.Dem_max)/2


    def __call__(self,model,method):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        simulation_parameters = create_example_parameters(self.n_buses)

        Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)

        # Defining the tensors
        Dem_train = torch.tensor(Dem_train).float().to(device)
        Gen_train = torch.tensor(Gen_train).float().to(device)

        # model = copy.deepcopy(model)
        
        model = model.to(device)

        lirpa_model = BoundedModule(model, torch.empty_like(Dem_train), device=device)
        print('Running on', device)

        x=(torch.tensor(self.x).float()).to(device)
        x_min= (torch.tensor(self.Dem_min).float()).to(device)
        x_max= (torch.tensor(self.Dem_max).float()).to(device)

        ptb = PerturbationLpNorm( x_L=x_min, x_U=x_max)

        image = BoundedTensor(x, ptb).to(device)
        # Get model prediction as usual
        # pred = lirpa_model(image).to(device)

        needed_A_dict = dict([(node, []) for node in lirpa_model._modules])

        # A = torch.load('A.pt').to(device)


        if method == 'IBP':
            lb, ub= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method="IBP", IBP= True)
        else:
            lb, ub, A= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method=method) #, reference_bounds = A)

        # lb, ub, A= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method="IBP", IBP= True)
        # lb, ub= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method="IBP", IBP= True)
        
        model_LiRPANet = IntremBound(lirpa_model)
        interm_lb,interm_ub=model_LiRPANet.get_interm_bounds(lb=lb,ub=ub)

        lirpa_model = BoundedModule(model, torch.empty_like(Dem_train), device=device)
        
        ptb = PerturbationLpNorm( x_L=x_min, x_U=x_max)

        image = BoundedTensor(x, ptb).to(device)

        lb, ub= lirpa_model.compute_bounds(x=(image,), return_A=True, forward = True, needed_A_dict = needed_A_dict, method="IBP", IBP= True)

        return interm_lb,interm_ub,lb.cpu().detach().numpy(), ub.cpu().detach().numpy()