import numpy as np
import pandas as pd
import os
# import loadsampling as ls
from tqdm import tqdm

def create_data(simulation_parameters):
    
    n_buses=simulation_parameters['general']['n_buses'] 
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = simulation_parameters['data_creation']['s_point']
    
    # Get the absolute path to the folder containing this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the data directory path
    data_dir = os.path.join(base_dir, f'dc_opf_data/{n_buses}')
    output_dir = os.path.join(data_dir, f'Dataset')
    
    
    L_Val = pd.read_csv(os.path.join(output_dir, 'NN_input.csv')).to_numpy()[s_point:s_point+n_data_points][:] 
    Gen_out = pd.read_csv(os.path.join(output_dir, 'NN_output.csv')).to_numpy()[s_point:s_point+n_data_points][:]
    
    #L_Val=pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_input_actual.csv').to_numpy()[s_point:s_point+n_data_points][:] 
    #Gen_out = pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_output_actual.csv').to_numpy()[s_point:s_point+n_data_points][:]

    x_training = L_Val
    return x_training, Gen_out


def create_test_data(simulation_parameters):

    n_buses=simulation_parameters['general']['n_buses'] 
    n_test_data_points = simulation_parameters['data_creation']['n_test_data_points']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    s_point = simulation_parameters['data_creation']['s_point']
    n_total = n_data_points + n_test_data_points
    
    # Get the absolute path to the folder containing this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct the data directory path
    data_dir = os.path.join(base_dir, f'dc_opf_data/{n_buses}')
    output_dir = os.path.join(data_dir, f'Dataset')
    

    L_Val = pd.read_csv(os.path.join(output_dir, 'NN_input.csv')).to_numpy()[s_point+n_data_points:s_point+n_total][:]
    Gen_out = pd.read_csv(os.path.join(output_dir, 'NN_output.csv')).to_numpy()[s_point+n_data_points:s_point+n_total][:]
    
    #L_Val=pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_input_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
    #Gen_out = pd.read_csv('verify-powerflow/dc_opf_data/'+str(n_buses)+'/NN_output_actual.csv').to_numpy()[s_point+n_data_points:s_point+n_total][:]
        
    x_test = np.concatenate([L_Val], axis=0)
    return x_test, Gen_out


import cvxpy as cp

def generate_power_system_data(simulation_parameters):

    # --- Extract Parameters from simulation_parameters ---
    true_system_params = simulation_parameters['true_system']
    general_params = simulation_parameters['general']
    data_creation_params = simulation_parameters['data_creation']

    n_buses = general_params['n_buses']
    n_gbus = general_params['n_gbus'] # Number of generators
    n_data_points = data_creation_params['n_data_points']

    # Generator parameters
    Pg_delta = true_system_params['Pg_delta'].flatten() 
    Pg_delta_safe = np.where(Pg_delta == 0, 1e-9, Pg_delta)
    Map_g = true_system_params['Map_g'] # Map from generator index to bus index (n_gbus, n_buses)

    # Load parameters
    Pd_min = true_system_params['Pd_min']
    Pd_delta = true_system_params['Pd_delta']
    Map_L = true_system_params['Map_L']
    
    # Check for zero Pd_delta to avoid division by zero for buses with fixed load
    Pd_delta_safe = np.where(Pd_delta == 0, 1e-9, Pd_delta) # Replace 0 with a small number to avoid div by zero

    # Line parameters
    PTDF_matrix = true_system_params['PTDF'].to_numpy()
    Pl_max = true_system_params['Pl_max'].flatten() 
    n_lines = PTDF_matrix.shape[0] 

    # --- Pre-calculate constant parts of constraint matrices ---
    M_flow_g = PTDF_matrix.T @ Map_g.T 
    M_flow_l = PTDF_matrix.T @ Map_L.T 

    # --- Initialize lists for storing training data ---
    x_training_list = [] # List to store scaled load profiles
    y_training_list = [] # List to store generator dispatches

    print(f"Starting data generation for {n_data_points} samples using CVXPY...")

    # --- Define CVXPY variables and parameters outside the loop if possible ---
    # This is an important optimization for CVXPY to avoid rebuilding the problem graph
    # for each iteration if only parameters change.
    G = cp.Variable(n_gbus, name="generator_output")
    current_load_profile_param = cp.Parameter(len(Pd_min), name="load_profile")

    # Objective: Minimize sum of generator outputs
    objective = cp.Minimize(cp.sum(G))

    # Constraints
    constraints = []

    # 1. Power Balance
    constraints.append(cp.sum(G) == cp.sum(current_load_profile_param))

    # 2. Generator Limits
    constraints.append(G >= 0)
    constraints.append(G <= Pg_delta)

    # 3. Transmission Line Thermal Limits   
    line_flows = M_flow_g @ G - (M_flow_l @ current_load_profile_param)

    constraints.append(line_flows <= Pl_max)
    constraints.append(line_flows >= -Pl_max) # Or cp.abs(line_flows) <= Pl_max

    # Create the CVXPY Problem object
    problem = cp.Problem(objective, constraints)

    # --- Data Generation Loop ---
    for i in range(n_data_points):
        # a. Generate a random load profile for ALL buses (unscaled)
        current_load_profile_unscaled = Pd_min + np.random.rand(len(Pd_min)) * Pd_delta

        # b. Scale the load profile to be between 0 and 1 (for NN input)
        current_load_profile_scaled = (current_load_profile_unscaled - Pd_min) / Pd_delta_safe
        current_load_profile_scaled = np.where(Pd_delta == 0, 0.0, current_load_profile_scaled)

        # c. Assign the unscaled load profile to the CVXPY parameter
        current_load_profile_param.value = current_load_profile_unscaled

        # d. Solve the DCOPF problem using CVXPY
        try:
            # You can specify a solver if you have one installed (e.g., solver=cp.GUROBI, cp.MOSEK)
            # Default solver will be chosen by CVXPY
            problem.solve(solver=cp.SCS, verbose=False) # ECOS is a good default for LPs
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                generator_dispatch = G.value # Get the numerical solution for G
                
                # Check for NaNs or None in solution (can happen with numerical issues or specific solver outcomes)
                if generator_dispatch is not None and not np.isnan(generator_dispatch).any():
                    generator_dispatch_scaled = generator_dispatch / Pg_delta_safe
                    
                    # For generators with Pg_delta originally 0, ensure their scaled value is exactly 0.0
                    generator_dispatch_scaled = np.where(Pg_delta == 0, 0.0, generator_dispatch_scaled)

                    # Append scaled load profile and solved generator dispatch
                    x_training_list.append(current_load_profile_scaled)
                    y_training_list.append(generator_dispatch_scaled.flatten()) # Flatten if it's a column vector
                else:
                    # print(f"Warning: CVXPY solver returned NaN/None for sample {i+1}. Status: {problem.status}")
                    pass # Skip if solution is invalid
            else:
                # print(f"Warning: CVXPY solver status for sample {i+1}: {problem.status}. Message: {problem.solution.status}")
                pass # Skip if not optimal
        except Exception as e:
            # print(f"Error solving DCOPF for sample {i+1} with CVXPY: {e}")
            pass # Skip if an error occurs during solving

    print(f"Finished generating {len(x_training_list)} feasible data samples.")

    # Convert lists to numpy arrays
    x_training = np.array(x_training_list)
    y_training = np.array(y_training_list)

    return x_training, y_training



# --- Example Usage (How to call this function) ---
if __name__ == "__main__":
    
    from create_example_parameters import create_example_parameters

    test_n_buses = 300 # Example: for a 6-bus system
    
    # Check if data_dir for parameters exists before attempting to load
    current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_data_dir = os.path.join(current_script_dir, f'dc_opf_data/{test_n_buses}')

    if not os.path.exists(param_data_dir):
        print(f"Error: Parameter data directory not found: {param_data_dir}")
        print("Please ensure you have the correct CSV files (Gen.csv, Bus.csv, PTDF.csv, branches.csv)")
        print(f"in a folder named dc_opf_data/{test_n_buses} relative to this script.")
    else:
        # 1. Create/Load simulation parameters (from your provided function)
        simulation_params = create_example_parameters(test_n_buses)

        # 2. Generate training data using the new function
        X_train_data, Y_train_data = generate_power_system_data(simulation_params)

        print("\nGenerated Training Data Shapes:")
        print(f"Scaled Load Profiles (X_train_data): {X_train_data.shape}")
        print(f"Generator Dispatches (Y_train_data): {Y_train_data.shape}")

        # Optional: Save generated data to CSV
        output_dir = os.path.join(param_data_dir, f'Dataset')
        
        # Saving with header=False and index=False as per your original load function
        pd.DataFrame(X_train_data).to_csv(os.path.join(output_dir, "NN_input.csv"), index=False, header=False)
        pd.DataFrame(Y_train_data).to_csv(os.path.join(output_dir, "NN_output.csv"), index=False, header=False)
        print(f"\nGenerated data saved to {output_dir}/NN_input.csv and NN_output.csv")
        print("\nExample Scaled Load Profile:")
        print(X_train_data[0])
        print("\nExample Generator Dispatch:")
        print(Y_train_data[0])