import torch
import numpy as np
#lb=torch.load('interm_lb.pt')
#ub=torch.load('interm_ub.pt')

# ub=torch.load('solution.pt')

run_id = 'brb8tdpb'
Algo = 'False'
Pg_min = np.load('Best_Model/Pg_min_'+ Algo + '_' +  run_id + '_.npy')
Pg_max = np.load('Best_Model/Pg_max_'+ Algo + '_' +  run_id + '_.npy')

# NN=torch.load('Best_Model/checkpoint_300_50_True_0.9835655679547288_0.0001970887948959955_.pt',map_location=torch.device('cpu'))