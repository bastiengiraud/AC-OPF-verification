
import time

import torch

import torch.nn as nn

import numpy as np
from DC_OPF.create_example_parameters import create_example_parameters
from DC_OPF.create_data import create_data, create_test_data
from EarlyStopping import EarlyStopping
# from PINNs.WCG import MILP_WCG
# from multiprocessing import Pool
from Neural_Network.lightning_NN import NeuralNetwork
from Neural_Network.cvxpy_layer_new import cvxpy_layer
from numpy.random import randn, rand
from LiRPANet import LiRPANet
import wandb

def to_np(x):
    return x.detach().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        training_loss =  100
        
        n_buses=config.test_system 
        simulation_parameters = create_example_parameters(n_buses)
        
        # Getting Training Data
        Dem_train, Gen_train = create_data(simulation_parameters=simulation_parameters)
        
        # Defining the tensors
        Dem_train = torch.tensor(Dem_train).float().to(device)
        Gen_train = torch.tensor(Gen_train).float().to(device)
        Gen_train_typ = torch.ones(Gen_train.shape[0],1).to(device)
        #--------------------------------------------------------------------- 
        # Gen typ defines if the data belongs to training dataset or if it 
        # belongs to the additional points collected from the training space
        # typ = 1 means it is part of the training set and the generation measurements are present
        #---------------------------------------------------------------------
        num_classes =  Gen_train.shape[1]
        
        Gen_delta=simulation_parameters['true_system']['Pg_delta'] 
        Dem_min=simulation_parameters['true_system']['Pd_min']
        Dem_delta=simulation_parameters['true_system']['Pd_delta']
        
        Data_stat = {}
        Data_stat['Gen_delta'] = Gen_delta
        Data_stat['Dem_min'] = Dem_min
        Data_stat['Dem_delta'] = Dem_delta

        # Test Data
        Dem_test, Gen_test = create_test_data(simulation_parameters=simulation_parameters)
        Dem_test = torch.tensor(Dem_test).float().to(device)
        Gen_test = torch.tensor(Gen_test).float().to(device)

        # NNs for predicting Generation (network_gen) and Volatage (network_Volt)
        network_gen = build_network(Dem_train.shape[1],
                                num_classes,
                                config.hidden_layer_size,
                                config.n_hidden_layers,
                                config.pytorch_init_seed)
        
        network_gen = normalise_network(network_gen, Dem_train,Data_stat)
        
        Para= list(network_gen.parameters())
        
        optimizer = torch.optim.Adam(Para,lr=config.learning_rate)
        lambda1 = lambda epoch: (epoch+1)**(-config.lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        
        # The NN will be trained for 200 iterations before enriching the NN database
        # config.epochs should be greater than 200 
        
        #path='Best_Model/checkpoint_'+str(n_buses)+'_'+str(config.hidden_layer_size)+'_'+str(config.Algo)+'_'+str(config.GenV_weight)+'_'+str(config.PF_weight)+'_.pt'

        path='Best_Model/checkpoint_'+str(n_buses)+'_'+str(config.hidden_layer_size)+'_'+str(config.Algo)+'_'+ wandb.run.id + '_.pt'

        early_stopping = EarlyStopping(patience=100, verbose=False, NN_input=Dem_train, path = path)
        
        training_needed= True
        gradient = 0
        RELU = nn.ReLU()
        PF_violation = 0
        if training_needed == True:
            for epoch in range(config.epochs):

                start_time=time.time()
                if epoch%100 == 0 and epoch != 0 and config.Algo == True:
                    X,Y,typ = wc_enriching(network_gen, config, Dem_train,Data_stat)
                    InputNN = torch.cat((Dem_train, X), 0).to(device)
                    OutputNN = torch.cat((Gen_train, Y), 0).to(device)
                    typNN  = torch.cat((Gen_train_typ, typ), 0).to(device)
                    
                    shuffled_ind=torch.randperm(InputNN.shape[0])
                    InputNN = InputNN[shuffled_ind]
                    OutputNN = OutputNN[shuffled_ind]
                    typNN = typNN[shuffled_ind]  
                else:
                    InputNN = Dem_train
                    OutputNN = Gen_train
                    typNN  = Gen_train_typ                   
            

                # Training NN and getting the training loss
                training_loss = train_epoch(network_gen, InputNN, OutputNN, typNN,optimizer,config,simulation_parameters)
                validation_loss = validate_epoch(network_gen, Dem_test,Gen_test)
                # 
                training_time= time.time() - start_time
                start_time= time.time()

                early_stopping(validation_loss, network_gen)

                if epoch%10 == 0:
                    Bound_Function=LiRPANet(n_buses,Data_stat)
                    Lower_bound, Upper_bound,Pg_min,Pg_max = Bound_Function(network_gen,config.abc_method)

                    LiRPANet_time= time.time() - start_time
                    start_time= time.time()
                    Pg_max=np.maximum(Pg_max,0)
                    Pg_min=np.minimum(Pg_min,0)
                    # np.save('Best_Model/Pg_min_'+str(config.Algo)+'_'+ wandb.run.id + '_.npy', Pg_min)
                    # np.save('Best_Model/Pg_max_'+str(config.Algo)+'_'+ wandb.run.id + '_.npy', Pg_max)

                Z_min = np.zeros((config.hidden_layer_size, config.n_hidden_layers))

                Z_min[:,0] = np.minimum(Lower_bound['/input'].cpu().detach().numpy(),0)
                Z_min[:,1] = np.minimum(Lower_bound['/input.3'].cpu().detach().numpy(),0)
                Z_min[:,2] = np.minimum(Lower_bound['/input.7'].cpu().detach().numpy(),0)

                Z_max = np.zeros((config.hidden_layer_size, config.n_hidden_layers))
                Z_max[:,0] = np.maximum(Upper_bound['/input'].cpu().detach().numpy(),0)
                Z_max[:,1] = np.maximum(Upper_bound['/input.3'].cpu().detach().numpy(),0)
                Z_max[:,2] = np.maximum(Upper_bound['/input.7'].cpu().detach().numpy(),0)

                W_last = network_gen.L_4.weight.to(device).data.numpy()
                B_last = network_gen.L_4.bias.to(device).data.numpy()

                N_hid_l = config.n_hidden_layers

                Pg_hat_max = ((np.maximum(W_last, 0))@np.maximum(Z_max[:,N_hid_l-1], 0) + 
                        (np.minimum(W_last, 0))@np.maximum(Z_min[:,N_hid_l-1], 0)).reshape((num_classes,)) + B_last.reshape((num_classes,))
    
                Pg_hat_min = ((np.minimum(W_last, 0))@np.maximum(Z_max[:,N_hid_l-1], 0) 
                                    + (np.maximum(W_last, 0))@np.maximum(Z_min[:,N_hid_l-1], 0)).reshape((num_classes,)) + B_last.reshape((num_classes,))
                
                wandbs=network_gen.state_dict()

                W={}; B={}
                W[0] = network_gen.L_1.weight.to(device)
                B[0] = network_gen.L_1.bias.to(device)

                W[1] = network_gen.L_2.weight.to(device)
                B[1] = network_gen.L_2.bias.to(device)

                W[2] = network_gen.L_3.weight.to(device)
                B[2] = network_gen.L_3.bias.to(device)

                W[3] = network_gen.L_4.weight.to(device)
                B[3] = network_gen.L_4.bias.to(device)
                
                if epoch%10 == 0 and epoch >= 100:
                    cvxpylayer_1, cvxpylayer_2 = cvxpy_layer(Dem_train.shape[1],config.hidden_layer_size,num_classes, config.n_hidden_layers, Data_stat, Z_min,Z_max,Pg_hat_min,Pg_hat_max)
                    #cvxpylayer_1, cvxpylayer_2 = cvxpy_layer(Dem_train.shape[1],config.hidden_layer_size,num_classes, config.n_hidden_layers, Data_stat, Z_min,Z_max)

                    if torch.cuda.is_available():
                        cvxpylayer_1.cuda()
                        cvxpylayer_2.cuda()
                    solution_1, = cvxpylayer_1(W[0],W[1],W[2],W[3],B[0],B[1],B[2],B[3])
                    solution_2, = cvxpylayer_2(W[0],W[1],W[2],W[3],B[0],B[1],B[2],B[3])

                    if torch.cuda.is_available():
                        solution_1 = solution_1.cuda()
                        solution_2 = solution_2.cuda()
                
                loss_weight = config.LPF_weight/(1+(epoch)*0.01)   

                if epoch >= 100:
                    PF_violation= ((solution_1**2).sum()) + ((solution_2**2).sum())

                if config.Algo == True and epoch >= 100: 
                    optimizer.zero_grad()
                    batch_loss = loss_weight*PF_violation
                    batch_loss.backward()
                    gradient = torch.mean(W[0].grad)
                    optimizer.step()
                    

                LP_time = time.time() - start_time
                start_time= time.time()
                scheduler.step()
                wandb.log({"training_loss": training_loss, "validation_loss": validation_loss, "Linear_PF_violation": PF_violation, "epoch": epoch, "LP_time": LP_time,"LiRPANet_time": LiRPANet_time,"training_time": training_time, "gradient": gradient }) 
                   
                if early_stopping.early_stop:
                    print("Early stopping")
                    break 
        else:
            network_gen.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  

        wandbs=network_gen.state_dict()
        W={}; b={}
        for k in range(config.n_hidden_layers):
            W[k] = wandbs['dense_layers.dense_' +str(k) +'.weight'].cpu().data.numpy()
            b[k] = wandbs['dense_layers.dense_' +str(k) +'.bias'].cpu().data.numpy()
            
        b[config.n_hidden_layers]=wandbs['dense_layers.output_layer.bias'].cpu().data.numpy()
        W[config.n_hidden_layers]=wandbs['dense_layers.output_layer.weight'].cpu().data.numpy()
        
        time_start= time.time()
        max_wc_g = 0
        
        # wc_g = MILP_WCG(i,n_buses,W,b,np.transpose(Gen_delta),np.transpose(Gen_delta))
        
        # wandb.log({"Max_WC_G": max_wc_g, "execution_time":time.time()-time_start})
      
def build_network(n_input_neurons, n_output_neurons,hidden_layer_size, n_hidden_layers, pytorch_init_seed):
    hidden_layer_size = [hidden_layer_size,hidden_layer_size,hidden_layer_size]
    model = NeuralNetwork(n_input_neurons, 
                                hidden_layer_size=hidden_layer_size,
                                num_output=n_output_neurons,
                                pytorch_init_seed=pytorch_init_seed)

    return model.to(device)

def normalise_network(model, Dem_train,Data_stat):
    pd_min=Data_stat['Dem_min']
    pd_delta=Data_stat['Dem_delta']
    pg_delta = Data_stat['Gen_delta']

    input_statistics = (torch.from_numpy(pd_min.reshape(-1,).astype(np.float32)),torch.from_numpy(pd_delta.reshape(-1,).astype(np.float32)))
    output_statistics = torch.from_numpy(pg_delta.reshape(-1,).astype(np.float32))

    model.normalise_input(input_statistics=input_statistics)
    model.normalise_output(output_statistics=output_statistics)

    return model.to(device)

def validate_epoch(network_gen, Dem_test,Gen_test):
    criterion = nn.MSELoss()
    output = network_gen.forward(Dem_test)
    validate_loss = criterion(output, Gen_test)
    
    return validate_loss
    

def train_epoch(network_gen, Dem_train, Gen_train, typ, optimizer, config,simulation_parameters):
    n_bus = simulation_parameters['general']['n_buses']
    Gen_delta=simulation_parameters['true_system']['Pg_delta'] 
    
    # NNs for predicting Generation (network_gen)
    network_gen.train()
    # initializing parameters
    cur_gen_loss = 0
    
    # NN loss function
    criterion = nn.MSELoss()
    
    # Defining Data set slicing algorithm
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    
    num_samples_train = Dem_train.shape[0]
    num_batches_train = int(num_samples_train // config.batch_size)
    
    RELU = nn.ReLU()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, config.batch_size)
        
        # NN_Gen
        Gen_output = network_gen.forward(Dem_train[slce])
        Gen_target = Gen_train[slce]
        # computeing NN prediction error
        batch_loss_gen = criterion(Gen_output*typ[slce], Gen_target*typ[slce])
        
        # Computing Gen Violations
        batch_loss_gen_violation= config.GenV_weight*torch.mean(torch.square(RELU(Gen_output- torch.from_numpy(Gen_delta).to(device))))+\
            config.GenV_weight*torch.mean(torch.square(RELU(0-Gen_output)))
        
        batch_loss= batch_loss_gen +  batch_loss_gen_violation     
        
        # # Calculating Power Flow error
        batch_PF_loss = power_flow_check(Dem_train[slce], Gen_output, simulation_parameters)
        batch_loss += config.PF_weight*batch_PF_loss
        
        # Computing batch loss gradient
        batch_loss.backward()
        optimizer.step()
        
        # Storing for looging to WandB
        cur_gen_loss += batch_loss_gen 
    
    # wandb.log({"batch_PF_loss": batch_PF_loss})
    
    Gen_train_loss = (cur_gen_loss / num_batches_train)

    return Gen_train_loss


def power_flow_check(P_Loads, P_Gens, simulation_parameters):
    # n_bus = simulation_parameters['general']['n_buses']
    # Y = torch.tensor(simulation_parameters['true_system']['Y'].astype(np.float32))
    # Ybr = torch.tensor(simulation_parameters['true_system']['Ybr'].astype(np.float32))
    # # Incidence matrix
    PTDF = torch.tensor(simulation_parameters['true_system']['PTDF'].to_numpy().astype(np.float32)).to(device)
    
    g_bus = simulation_parameters['true_system']['g_bus']
    # Maping generators to bus
    Map_g = torch.tensor(simulation_parameters['true_system']['Map_g'].astype(np.float32)).to(device)
    # maping loads to buses
    Map_L = torch.tensor(simulation_parameters['true_system']['Map_L'].astype(np.float32)).to(device)    

    # Line limits
    Pl_max = torch.tensor(simulation_parameters['true_system']['Pl_max'].astype(np.float32)).to(device)
    # n_line = simulation_parameters['true_system']['n_line']
    
    # PowerFlow Equation
     
    RELU = nn.ReLU()
    PF_error = torch.abs(torch.sum(P_Gens,1) - torch.sum(P_Loads,1))
    #PF_error = PF_error + torch.sum(RELU(torch.matmul((torch.matmul(P_Gens,Map_g)-torch.matmul(P_Loads,Map_L)),PTDF) - Pl_max), axis=1)
    #PF_error = PF_error + torch.sum(RELU(Pl_max-torch.matmul((torch.matmul(P_Gens,Map_g)-torch.matmul(P_Loads,Map_L)),PTDF)), axis=1)
 
    return torch.mean(PF_error)


def wc_enriching(network_gen, config, Dem_train,Data_stat):
    n_adver =config.N_enrich
    Gen_output = network_gen.forward_aft(Dem_train).cpu().detach().numpy()
    Gen_delta=(Data_stat['Gen_delta'])
    Dem_min=(Data_stat['Dem_min'])
    Dem_delta=(Data_stat['Dem_delta'])
    PB = np.sum(Gen_output,1) - np.sum(Dem_train.cpu().detach().numpy(),1)
    # PB = np.sum(Gen_output,1)

    ind=np.argpartition(PB, -4)[-n_adver//2:]

    Adverse_example_p = Dem_train[ind][:].cpu().detach().numpy()

    Adverse_example_p = GradAscnt(network_gen,Adverse_example_p,Data_stat,sign=1)

    ind=np.argpartition(-PB, -4)[-n_adver//2:]

    Adverse_example_n = Dem_train[ind][:].cpu().detach().numpy()

    Adverse_example_n = GradAscnt(network_gen,Adverse_example_n,Data_stat)
    Adverse_example = np.append(Adverse_example_n,Adverse_example_p, axis=0)
    x_g= torch.tensor(Adverse_example).float()
    y_g = torch.zeros(x_g.shape[0], Gen_output.shape[1])
    y_type = torch.zeros(x_g.shape[0], 1)
    return x_g,y_g,y_type

def GradAscnt(Network,x_starting, Data_stat, sign = -1, 
              Num_iteration=100, lr=0.0001):
    '''
    x_starting: Starting points for the gradient ascent algorithm
    x_min,x_max :  Minimum and maximum value of x ( default is 0 and 1)
    Sign : direction for gradient ascent ( 1 --> Increase the violation , -1 --> reduce the violation (.i.e. make it more negative))
    Num_iteration: Number of gradient steps
    lr: larning rate
    '''
    Gen_delta=torch.tensor(Data_stat['Gen_delta'])
    Dem_min=torch.tensor(Data_stat['Dem_min'])
    Dem_delta=torch.tensor(Data_stat['Dem_delta'])
    x_min = Dem_min 
    x_max = Dem_min+Dem_delta 
    x = torch.tensor(x_starting.astype(np.float32), requires_grad=True).to(device)
    optimizer = torch.optim.Adam([x],lr=lr)
    for i in range(Num_iteration):
        optimizer.zero_grad() 
        x=torch.tensor(x.detach(), requires_grad=True).to(device)
        P_g = Network.forward_aft(x)
        #loss = torch.sum(torch.sum(P_g*Gen_delta,1) -  torch.sum(Dem_min + x*Dem_delta,1))
        loss = torch.sum(torch.sum(P_g,1) -  torch.sum(x,1))
        loss.backward()
        optimizer.step()
        # x = x.clone().detach() + sign*torch.tensor(0.0001).to(device)*x.grad
        x = torch.nan_to_num(x.clone().detach())
        x = torch.maximum(x.clone().detach(),x_min).float()
        x = torch.minimum(x.clone().detach(),x_max).float()
    return x.cpu().detach().numpy()